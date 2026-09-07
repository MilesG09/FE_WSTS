import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import warnings
from .utils import get_means_stds_missing_values, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime


class FireSpreadDataset(Dataset):

    # Canonical order and gating of the engineered centroid channels. THIS IS THE SINGLE SOURCE
    # OF TRUTH: _compute_centroid_channels emits channels in exactly this order, and src/train.py
    # derives the model's in_channels from it via n_centroid_channels() -- neither side hardcodes
    # a count. Adding a future channel means adding a row here and computing it; nothing
    # downstream changes. Order must stay stable: it is the channel index a trained checkpoint's
    # first conv layer was fit against.
    CENTROID_CHANNEL_SPEC = (
        # (channel name, name of the boolean flag that gates it)
        # NB: "x" is the ROW coordinate (axis of length H) and "y" the COLUMN (axis of length W),
        # inherited from the original implementation. Kept as-is to avoid churn.
        ("centroid_x",     "use_centroid_position"),
        ("centroid_y",     "use_centroid_position"),
        ("centroid_vx",    "use_centroid_velocity"),
        ("centroid_vy",    "use_centroid_velocity"),
        ("position_valid", "use_centroid_position_validity"),
        ("velocity_valid", "use_centroid_velocity_validity"),
    )

    @staticmethod
    def centroid_channel_names(use_centroid_position: bool = False,
                               use_centroid_velocity: bool = False,
                               use_centroid_position_validity: bool = False,
                               use_centroid_velocity_validity: bool = False,
                               n_timesteps: int = 1) -> List[str]:
        """Names of the centroid channels emitted under these flags, in emission order.

        One channel per enabled family PER model-visible timestep, so the count is
        `n_timesteps * (enabled families)`. Layout is **family-major** and, within a family,
        ordered oldest -> newest -- matching both the tensor the producer stacks and the
        oldest-first ordering the imagery channels already have after
        flatten_and_remove_duplicate_features_.

        The suffix is the **lag relative to the most recent input frame**, so `_lag0` always
        names the last day the model sees whatever `n_timesteps` is. That keeps a suffix
        meaning the same thing across arms with different `n_leading_observations`, which
        matters when comparing per-channel first-layer weights between arms. (A positional
        `_t0` suffix would instead mean "5 days ago" at C and "today" at A.)

        Returns names rather than a bare count so the same call also answers "what is channel
        i?" -- needed for debugging, for the verification script, and for logging the channel
        layout as an MLflow tag. A count is derivable from names; names are not derivable from
        a count.
        """
        enabled = {
            "use_centroid_position": use_centroid_position,
            "use_centroid_velocity": use_centroid_velocity,
            "use_centroid_position_validity": use_centroid_position_validity,
            "use_centroid_velocity_validity": use_centroid_velocity_validity,
        }
        names = []
        for family, flag in FireSpreadDataset.CENTROID_CHANNEL_SPEC:
            if enabled[flag]:
                names.extend(f"{family}_lag{n_timesteps - 1 - t}" for t in range(n_timesteps))
        return names

    @staticmethod
    def n_centroid_channels(use_centroid_position: bool = False,
                            use_centroid_velocity: bool = False,
                            use_centroid_position_validity: bool = False,
                            use_centroid_velocity_validity: bool = False,
                            n_timesteps: int = 1) -> int:
        """Number of centroid channels appended to x under these flags. Called from src/train.py
        before any dataset object exists, so it must depend only on the config values."""
        return len(FireSpreadDataset.centroid_channel_names(
            use_centroid_position, use_centroid_velocity,
            use_centroid_position_validity, use_centroid_velocity_validity,
            n_timesteps=n_timesteps))

    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                 crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                 stats_years: List[int], n_leading_observations_test_adjustment: Optional[int] = None, 
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False, is_pad: Optional[bool] = False,
                 hdf5_cache_size: int = 64, use_centroid_position: bool = False, use_centroid_velocity: bool = False, use_centroid_position_validity: bool = False, use_centroid_velocity_validity: bool = False):
        """_summary_

        Args:
            data_dir (str): _description_ Root directory of the dataset, should contain several folders, each corresponding to a different fire.
            included_fire_years (List[int]): _description_ Years in dataset_root that should be used in this instance of the dataset.
            n_leading_observations (int): _description_ Number of days to use as input observation. 
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF. 
            is_train (bool): _description_ Whether this dataset is used for training or not. If True, apply geometric data augmentations. If False, only apply center crop to get the required dimensions.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            stats_years (List[int]): _description_ Which years to use for computing the mean and standard deviation of each feature. This is important for the test set, which should be standardized using the same statistics as the training set.
            n_leading_observations_test_adjustment (Optional[int], optional): _description_. Adjust the test set to look like it would with n_leading_observations set to this value. 
        In practice, this means that if n_leading_observations is smaller than this value, some samples are skipped. Defaults to None. If None, nothing is skipped. This is especially used for the train and val set. 
            features_to_keep (Optional[List[int]], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            is_pad (book, optional): _description_. Whether to zero-pad image to 224x224 for SwinUnet/TransUnet
            hdf5_cache_size (int, optional): _description_. Maximum number of open h5py.File handles this dataset
        keeps cached per worker process. Defaults to 64.
        Raises:
            ValueError: _description_ Raised if input values are not in the expected ranges.
        """
        super().__init__()

        self.stats_years = stats_years
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir
        self.is_pad = is_pad

        # Centroid Features
        self.use_centroid_position = use_centroid_position
        self.use_centroid_velocity = use_centroid_velocity
        self.use_centroid_position_validity = use_centroid_position_validity
        self.use_centroid_velocity_validity = use_centroid_velocity_validity
        # Peek-back frames: how many days BEFORE the model's own window load_imgs must fetch so
        # that the oldest model-visible frame still has a previous day to difference against.
        # Velocity is defined per model-visible frame (n_timesteps channels, not n_timesteps-1),
        # which is what makes a velocity channel possible at all at n_leading_observations=1 --
        # and what keeps the feature's definition identical across every arm on the dose-response
        # curve. See administrative/EXPERIMENTS.md P2 (design revision, 2026-08-25).
        self._n_peek_frames = 1 if (use_centroid_velocity or use_centroid_velocity_validity) else 0

        # Resolved once here rather than per-sample: __getitem__ runs millions of times and only
        # needs "any centroid channel at all?" plus the fixed emission order.
        self._centroid_channel_names = FireSpreadDataset.centroid_channel_names(
            use_centroid_position, use_centroid_velocity,
            use_centroid_position_validity, use_centroid_velocity_validity,
            n_timesteps=n_leading_observations)
        self._use_any_centroid = len(self._centroid_channel_names) > 0

        # ------------------------------------------------------------------ #
        # Training-efficiency machinery. Every optimisation below keeps its    #
        # original code path behind a WSTS_DISABLE_* environment flag, so      #
        # scripts/verify_pipeline_equivalence.py can build the same dataset    #
        # twice and prove the outputs are bit-identical. Nothing here may      #
        # change a single value the model sees -- these runs are compared      #
        # against the WSTS+ paper, and a silent 1e-7 drift invalidates that.   #
        # ------------------------------------------------------------------ #

        # Cap on distinct open h5py.File handles this worker keeps cached. Uncapped caching
        # grows each persistent worker's private memory roughly 3x over a few minutes of
        # training, tracking open-file-descriptor count, and eventually trips the OOM killer.
        # An LRU bounds that while keeping repeated reads from the same fire cheap.
        self._hdf5_cache_max = hdf5_cache_size
        self._hdf5_file_cache = OrderedDict()
        # Which process the cached handles belong to; see _get_hdf5_file for why.
        self._hdf5_cache_pid = os.getpid()

        # Partial-read (hyperslab) optimisation. load_imgs used to pull every channel at full
        # resolution and let Python throw most of it away -- the 128x128 crop in
        # preprocess_and_augment, then features_to_keep. Measured read amplification was ~18x:
        # ~7.9 MB off disk per sample to feed ~0.44 MB to the model.
        #
        # Only the raw channels feeding features_to_keep can affect the output. That is
        # verified empirically rather than inferred: scripts/verify_channel_dependency.py
        # zeroes each raw channel in turn and confirms the pipeline output is unchanged for
        # the channels we skip. Channels outside the set are zero-filled after the read, so
        # every downstream fixed index -- land cover at 16, indices_of_degree_features, the
        # x[:, -1] active-fire channel -- keeps working with no other code change.
        #
        # Escape hatch: WSTS_DISABLE_HDF5_READ_OPT=1 restores the full-read path.
        self._raw_channels_needed = None
        if self.load_from_hdf5 and not os.environ.get("WSTS_DISABLE_HDF5_READ_OPT"):
            self._raw_channels_needed = self.raw_channels_for_features(
                self.features_to_keep)

        # Crop-search optimisation (see augment). The random-crop loop scored ten candidates by
        # materialising the full (T, C, H, W) crop each time, but only ever read one channel of
        # it. Setting WSTS_DISABLE_CROP_OPT=1 restores the original path for A/B verification.
        self._crop_search_materialises = bool(
            os.environ.get("WSTS_DISABLE_CROP_OPT"))

        # Land-cover one-hot skip. preprocess_and_augment expands channel 16 into 17 one-hot
        # channels (24 -> 40), but nothing downstream reads them unless features_to_keep asks
        # for an index in [16, 32]. When it does not, the expansion is built and discarded on
        # every single sample.
        #
        # Skipping it leaves the tensor 24-wide, so every index that refers to the 40-channel
        # space has to be remapped. features_to_keep on the CONFIG stays in 40-space on purpose
        # -- train.py feeds it to get_n_features() and the feature COUNT must not change -- so
        # the remap lives here, on the instance, and never leaks outward.
        #
        # Escape hatch: WSTS_DISABLE_ONEHOT_SKIP=1.
        self._skip_one_hot = False
        self._features_to_keep_effective = self.features_to_keep
        _, dynamic_ids = self.get_static_and_dynamic_features_to_keep(
            self.features_to_keep)
        self._dynamic_ids_effective = dynamic_ids

        if not os.environ.get("WSTS_DISABLE_ONEHOT_SKIP"):
            remapped_keep = self.remap_indices_without_one_hot(self.features_to_keep)
            remapped_dynamic = self.remap_indices_without_one_hot(dynamic_ids)
            # Only safe when NOTHING requested lives inside the one-hot block; otherwise the
            # expansion is load-bearing and we silently keep the original path.
            if remapped_keep is not None and remapped_dynamic is not None:
                self._skip_one_hot = True
                self._features_to_keep_effective = remapped_keep
                self._dynamic_ids_effective = remapped_dynamic

        self.validate_inputs()

        # Compute how many samples to skip in the test set, to make it look like it would with n_leading_observations set to this value.
        if self.n_leading_observations_test_adjustment is None:
            self.skip_initial_samples = 0
        else:
            self.skip_initial_samples = self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.skip_initial_samples < 0:
                raise ValueError(f"n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations, but got {self.n_leading_observations_test_adjustment=} and {self.n_leading_observations=}")

        # Create an inventory of all images in the dataset, and how many data points each fire contains. Since we have multiple data points per fire,
        # we need to know how many data points each fire contains, to be able to map a dataset index to a specific fire.
        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum([sum(self.datapoints_per_fire[fire_year].values())
                          for fire_year in self.datapoints_per_fire])

        # Used in preprocessing and normalization. Better to define it once than build/call for every data point
        # The one-hot matrix is used for one-hot encoding of land cover classes
        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()

    def find_image_index_from_dataset_index(self, target_id) -> (int, str, int):
        """_summary_ Given the index of a data point in the dataset, find the corresponding fire that contains it, 
        and its index within that fire.

        Args:
            target_id (_type_): _description_ Dataset index of the data point.

        Raises:
            RuntimeError: _description_ Raised if the dataset index is out of range.

        Returns:
            (int, str, int): _description_ Year, name of fire, index of data point within fire.
        """

        # Handle negative indexing, e.g. -1 should be the last item in the dataset
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(
                f"Tried to access item {target_id}, but maximum index is {self.length - 1}.")

        # The index is relative to the length of the full dataset. However, we need to make sure that we know which
        # specific fire the queried index belongs to. We know how many data points each fire contains from
        # self.datapoints_per_fire.
        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:
            # `break` only exits the inner loop, so without this guard the outer
            # loop keeps going and re-matches the first fire of every later year
            # (target_id - first_id_in_current_fire goes negative, which is < any
            # count) -> every index resolves to a fire in the LAST year. Guard so
            # we stop updating once the fire is found. Matches upstream WSTS.
            if found_fire_year is None:
                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                    if target_id - first_id_in_current_fire < datapoints_in_fire:
                        found_fire_year = fire_year
                        found_fire_name = fire_name
                        break
                    else:
                        first_id_in_current_fire += datapoints_in_fire

        in_fire_index = target_id - first_id_in_current_fire

        return found_fire_year, found_fire_name, in_fire_index

    # Number of channels physically present in the HDF5 files, before any preprocessing.
    # preprocess_and_augment turns these 23 into 40: it appends a binary active-fire mask
    # (23 -> 24), then expands the land-cover integer at 16 into 17 one-hot channels,
    # giving 16 + 17 + 7 = 40.
    N_RAW_HDF5_CHANNELS = 23

    # Density above which the partial read is NOT worth doing, as a fraction of
    # N_RAW_HDF5_CHANNELS. The partial read trades I/O for CPU: h5py builds a channel
    # selection and the result is scattered into a full-width zero array. These HDF5
    # files are contiguous and unchunked, so a scattered selection covering most of the
    # channel range drags the same extent past the block layer anyway -- the saving
    # vanishes while the scatter cost stays.
    #
    # Measured, not guessed (scripts/benchmark_hyperslab_density.py, 2026-09-04, evenly
    # spread channels, cold cache, 3 files x 3 trials). Read time vs the plain full read:
    #
    #   fraction of channels :  0.09  0.17  0.26  0.35  0.43  0.52  0.70
    #   T=1 (2 frames read)  : 3.28x 1.46x 1.15x 0.94x 0.76x 0.74x 0.65x
    #   T=5 (6 frames read)  : 4.16x 2.65x 1.68x 1.29x 1.05x 0.90x 0.75x
    #
    # The crossover MOVES with the number of frames read -- more frames means more bytes
    # per read and more to save -- so the guard is set by the tighter case (T=1, which
    # turns negative just past 0.26) rather than by an average that would silently
    # regress the monotemporal configs. End-to-end this matches: the vegetation config
    # (6 of 23 = 0.26) measured 1.20x faster __getitem__ at T=1 and 1.35x at T=5, while
    # the multi-feature config (16 of 23 = 0.70) measured 0.82x -- a 22% REGRESSION, with
    # zero bytes saved, which is what this guard exists to prevent.
    #
    # Override at construction time with WSTS_HDF5_READ_OPT_MAX_FRACTION, which is read
    # per call rather than at import, so it behaves like the WSTS_DISABLE_* flags.
    HDF5_READ_OPT_MAX_FRACTION = 0.3

    @staticmethod
    def raw_channels_for_features(features_to_keep: Optional[List[int]]) -> Optional[List[int]]:
        """Map post-preprocessing feature indices (0..39) back to raw HDF5 channels.

        The mapping follows directly from how preprocess_and_augment builds its 40
        channels (see N_RAW_HDF5_CHANNELS):

            post 0..15  <- raw 0..15   passed through unchanged
            post 16..32 <- raw 16      land cover, one-hot expanded into 17 channels
            post 33..38 <- raw 17..22  shifted right by the one-hot expansion
            post 39     <- raw 22      binary active-fire mask, derived from x[:, -1]

        Returns a sorted list of raw channel indices, or None meaning "read everything"
        (no feature subset requested, or an unrecognised layout -- fail safe, not fast).
        """
        if not features_to_keep:
            return None

        n_raw = FireSpreadDataset.N_RAW_HDF5_CHANNELS
        needed = set()
        for post_idx in features_to_keep:
            if post_idx < 16:
                needed.add(post_idx)
            elif post_idx <= 32:
                needed.add(16)
            elif post_idx <= 38:
                needed.add(post_idx - 16)
            else:
                needed.add(n_raw - 1)

        # The label y is always the last channel of the last timestep, regardless of
        # which features are kept, so it must always be read.
        needed.add(n_raw - 1)

        if any(c < 0 or c >= n_raw for c in needed):
            return None

        # Too dense to be worth it -- fall back to the plain full read. Returning None
        # (rather than a channel list nobody benefits from) keeps the decision in one
        # place: everywhere downstream, `_raw_channels_needed is None` already means
        # "read everything".
        max_fraction = float(os.environ.get(
            "WSTS_HDF5_READ_OPT_MAX_FRACTION",
            FireSpreadDataset.HDF5_READ_OPT_MAX_FRACTION))
        if len(needed) > max_fraction * n_raw:
            return None

        return sorted(needed)

    @staticmethod
    def remap_indices_without_one_hot(
            indices: Optional[List[int]]) -> Optional[List[int]]:
        """Translate 40-channel-space indices into the 24-channel pre-one-hot space.

        preprocess_and_augment builds its 40 channels as
        `[x[:, :16], one_hot(landcover, 17), x[:, 17:]]` over a 24-channel tensor
        (23 raw + appended active-fire mask). Undoing that expansion:

            post 0..15  -> 0..15    (before the insertion point, unshifted)
            post 16..32 -> the one-hot block itself; no pre-expansion equivalent
            post 33..39 -> 17..23   (shifted back by the 16 channels the block added)

        Returns None if ANY index falls inside the one-hot block, meaning the expansion is
        actually needed and must not be skipped. Returning None rather than raising keeps
        this a fast-path check: callers fall back to the original behaviour.
        """
        if indices is None:
            return None
        out = []
        for idx in indices:
            if idx < 16:
                out.append(idx)
            elif idx <= 32:
                return None
            else:
                out.append(idx - 16)
        return out

    def close_hdf5_cache(self):
        """Close and forget every cached handle. Call before forking DataLoader workers.

        FireSpreadDataModule.setup() can iterate the whole dataset in the PARENT process
        (the ignition filters), which populates this cache there. Forking after that
        would hand every worker a duplicate of the parent's file descriptors, and HDF5
        is explicitly not fork-safe across shared handles -- concurrent reads through
        inherited descriptors can return corrupt data with no error.
        """
        while self._hdf5_file_cache:
            _, f = self._hdf5_file_cache.popitem()
            try:
                f.close()
            except Exception:  # noqa: BLE001 - a handle we are discarding anyway
                pass

    def _get_hdf5_file(self, path):
        """Return a cached h5py.File handle for `path`, opening it on first use.

        Each DataLoader worker is normally a forked copy of this dataset with an EMPTY
        cache, so each worker populates its own handles and none is ever shared. With
        persistent_workers=True a worker lives for the entire run, so an unbounded cache
        would keep every touched HDF5 file open forever; this is an LRU capped at
        self._hdf5_cache_max instead.

        The pid check is the safety net for the case where that assumption breaks: if
        anything reads samples in the parent before workers fork (the ignition filters in
        FireSpreadDataModule.setup() do exactly that), the child inherits live HDF5
        descriptors. Those are NOT safe to use -- so a child that finds handles from
        another pid discards the dict and reopens its own. They are dropped without
        close(), deliberately: closing an inherited descriptor reaches into HDF5 state
        the parent still owns, which is the failure this check exists to avoid.
        """
        if self._hdf5_cache_pid != os.getpid():
            self._hdf5_file_cache = OrderedDict()
            self._hdf5_cache_pid = os.getpid()

        if path in self._hdf5_file_cache:
            self._hdf5_file_cache.move_to_end(path)
            return self._hdf5_file_cache[path]

        if len(self._hdf5_file_cache) >= self._hdf5_cache_max:
            _, oldest_file = self._hdf5_file_cache.popitem(last=False)
            oldest_file.close()

        f = h5py.File(path, 'r')
        self._hdf5_file_cache[path] = f
        return f

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        """_summary_ Load the images corresponding to the specified data point from disk.

        Args:
            found_fire_year (_type_): _description_ Year of the fire that contains the data point.
            found_fire_name (_type_): _description_ Name of the fire that contains the data point.
            in_fire_index (_type_): _description_ Index of the data point within the fire.

        Returns:
            _type_: _description_ (x,y) or (x,y,doy) tuple, depending on whether return_doy is True or False. 
            x is a tensor of shape (n_leading_observations, n_features, height, width), containing the input data. 
            y is a tensor of shape (height, width) containing the binary next day's active fire mask.
            doy is a tensor of shape (n_leading_observations) containing the day of the year for each observation.
        """

        in_fire_index += self.skip_initial_samples

        # Peek-back window. We read self._n_peek_frames rows FURTHER BACK than the model's own
        # window so the oldest model-visible frame has a previous day to difference against.
        # Frames are contiguous rows of one HDF5 dataset, so this widens the existing range read
        # rather than adding a second one.
        #
        # When the window starts at the fire's first observed day the peek row does not exist. We
        # zero-pad it instead of dropping the sample (his design, EXPERIMENTS.md P2 revision,
        # 2026-08-25): dropping it would cost one sample per fire and make peeking arms train on
        # a smaller set than non-peeking ones, contaminating the very contrast the feature is
        # meant to isolate. A zeroed frame has no fire pixels, so vel_valid falls out as 0 and the
        # velocity is gated through the ordinary code path -- "the day does not exist" and "the
        # day had no fire" become the same case, with no extra flag to thread anywhere

        n_pad = max(0, self._n_peek_frames - in_fire_index)
        start_index = in_fire_index - self._n_peek_frames + n_pad
        end_index = (in_fire_index + self.n_leading_observations + 1)

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            f = self._get_hdf5_file(hdf5_path)
            dset = f["data"]
            if self._raw_channels_needed is None:
                imgs = dset[start_index:end_index]
            else:
                # Hyperslab read: pull only the load-bearing channels off disk, then
                # scatter them back into a full-width zero array. The zero-fill is what
                # lets every downstream fixed channel index stay valid without touching
                # preprocess_and_augment. Slicing first (rather than allocating from
                # end_index) keeps the natural clamping when end_index runs past the end
                # of the fire's time series.
                sub = dset[start_index:end_index, self._raw_channels_needed]
                imgs = np.zeros(
                    (sub.shape[0], dset.shape[1]) + sub.shape[2:], dtype=sub.dtype)
                imgs[:, self._raw_channels_needed] = sub
            if self.return_doy:
                doys = dset.attrs["img_dates"][in_fire_index:(
                    end_index-1)]
                doys = self.img_dates_to_doys(doys)
                doys = torch.Tensor(doys)
            if n_pad:
                imgs = np.concatenate(
                    [np.zeros((n_pad,) + imgs.shape[1:], dtype=imgs.dtype), imgs], axis=0)
            x, y = np.split(imgs, [-1], axis=0)
            # Last image's active fire mask is used as label, rest is input data
            y = y[0, -1, ...]
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][start_index:end_index]
            imgs = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    imgs.append(ds.read())
            x = np.stack(imgs[:-1], axis=0)
            y = imgs[-1][-1, ...]

        if self.return_doy:
            return x, y, doys
        return x, y

    def __getitem__(self, index):

        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(
            index)
        loaded_imgs = self.load_imgs(
            found_fire_year, found_fire_name, in_fire_index)

        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        x, y = self.preprocess_and_augment(x, y)

        # Compute centroid kinematics from full (T, 40, H, W) tensor before any filtering.
        # Channel 39 is binary active fire; augmentation has already been applied at this point.
        if self._use_any_centroid:
            centroid_channels = self._compute_centroid_channels(x)

        # Drop the peek-back frame(s) now that the kinematics are computed. They exist only to
        # give the oldest model-visible frame a previous day; the model must not see them, and
        # everything downstream (get_n_features, flatten_and_remove_duplicate_features_) assumes
        # x holds exactly n_leading_observations frames.
        if self._n_peek_frames:
            x = x[self._n_peek_frames:]

        # Remove duplicate static features, which can greatly reduce the number of features, since we use 
        # one-hot encoded landcover types. The result would have different amounts of feature channels per 
        # time step, therefore, we flatten the temporal dimension.
        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)
        # Discard features that we don't want to use
        elif self.features_to_keep is not None:
            if len(x.shape) != 4:
                raise NotImplementedError(f"Removing features is only implemented for 4D tensors, but got {x.shape=}.")
            # Effective ids, not the raw config ones: identical to the config unless the
            # land-cover one-hot was skipped, in which case they are remapped to 24-space.
            x = x[:, self._features_to_keep_effective, ...]

        # Append the enabled centroid channels after all other processing. The count is no
        # longer fixed at 6 -- it is len(self._centroid_channel_names), 0 to 6 depending on which
        # of the four flags are on.
        #
        # The concat axis depends on x's RANK, and the two cases are not interchangeable (P3).
        # flatten_and_remove_duplicate_features_ only runs when n_lead > 1, so:
        #   n_lead > 1  -> x is 3-D (C_total, H, W); the channel axis is 0.
        #   n_lead == 1 -> x is still 4-D (T=1, C, H, W); the channel axis is 1, and dim=0 is
        #                  TIME. Concatenating on dim=0 there stacks the centroid maps as extra
        #                  timesteps: shape-valid, silently meaningless, and it was doing exactly
        #                  that for every A arm before this fix.
        # Keeping the n_lead=1 case 4-D (rather than squeezing x to 3-D) is deliberate: A0 and A1
        # then return the same rank and differ only in channel count, so nothing downstream has
        # to branch on which arm is running. BaseModel.forward flattens (B,T,C,H,W) ->
        # (B,T*C,H,W) at src/models/BaseModel.py:78, which absorbs the T=1 axis either way.
        if self._use_any_centroid:
            if x.dim() == 4:
                assert x.shape[0] == 1, (
                    f"4-D x with T={x.shape[0]} > 1 cannot take centroid channels; "
                    "remove_duplicate_features=True is required and should have flattened it.")
                x = torch.cat([x, centroid_channels.unsqueeze(0)], dim=1)
            else:
                x = torch.cat([x, centroid_channels], dim=0)

        if self.return_doy:
            return x, y, doys
        return x, y

    def __len__(self):
        return self.length

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError(
                "Returning day of year is only implemented for hdf5 files.")
        if self.n_leading_observations_test_adjustment is not None:
            if self.n_leading_observations_test_adjustment < self.n_leading_observations:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations.")
            if self.n_leading_observations_test_adjustment < 1:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to 1. Value 1 is used for having a single observation as input.")

    def read_list_of_images(self):
        """_summary_ Create an inventory of all images in the dataset.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries. 
            These dictionaries map names of fires that happened within the respective year to either
            a) the corresponding list of image files (in case hdf5 files are not used) or
            b) the individual hdf5 file for each fire.
        """
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()
                    
                    imgs_per_fire[fire_year][fire_name] = fire_img_paths

                    if len(fire_img_paths) == 0:
                        warnings.warn(f"In dataset preparation: Fire {fire_year}: {fire_name} contains no images.",
                                      RuntimeWarning)
            else:
                fires_in_year = glob.glob(
                    f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        """_summary_ Compute how many data points each fire contains. This is important for mapping a dataset index to a specific fire.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries. 
            The dictionaries map the fire name to the number of data points in that fire.
        """
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs) - self.skip_initial_samples
                else:
                    # Catch error case that there's no file
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"]) - self.skip_initial_samples
                # If we have two days of observations, and a lead of one day,
                # we can only predict the second day's fire mask, based on the first day's observation
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"In dataset preparation: Fire {fire_year}: {fire_name} does not contribute data points. It contains "
                        f"{len(fire_imgs)} images, which is too few for a lead of {self.n_leading_observations} observations.",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def standardize_features(self, x):
        """_summary_ Standardizes the input data, using the mean and standard deviation of each feature. 
        Some features are excluded from this, which are the degree features (e.g. wind direction), and the land cover class.
        The binary active fire mask is also excluded, since it's added after standardization.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)

        Returns:
            _type_: _description_ Standardized input data, of shape (time_steps, features, height, width)
        """

        x = (x - self.means) / self.stds

        return x

    def preprocess_and_augment(self, x, y):
        """_summary_ Preprocesses and augments the input data. 
        This includes: 
        1. Slight preprocessing of active fire features, if loading from TIF files.
        2. Geometric data augmentation.
        3. Applying sin to degree features, to ensure that the extreme degree values are close in feature space.
        4. Standardization of features. 
        5. Addition of the binary active fire mask, as an addition to the fire mask that indicates the time of detection. 
        6. One-hot encoding of land cover classes.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        x, y = torch.Tensor(x), torch.Tensor(y)

        # Preprocessing that has been done in HDF files already
        if not self.load_from_hdf5:

            # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
            # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
            # them with 0 instead.
            x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)

            # Turn active fire detection time from hhmm to hh.
            x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        y = (y > 0).long()

        # Augmentation has to come before normalization, because we have to correct the angle features when we change
        # the orientation of the image.
        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)
        
        # If using a model that expects images of larger size, use zero-padding 
        if self.is_pad:
            x, y = self.zero_pad_to_size(x, y)
        
        # Some features take values in [0,360] degrees. By applying sin, we make sure that values near 0 and 360 are
        # close in feature space, since they are also close in reality.
        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[:, self.indices_of_degree_features, ...]))

        # Compute binary mask of active fire pixels before normalization changes what 0 means. 
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        x = self.standardize_features(x)

        # Adds the binary fire mask as an additional channel to the input data.
        x = torch.cat([x, binary_af_mask], axis=1)

        # Replace NaN values with 0, thereby essentially setting them to the mean of the respective feature.
        x = torch.nan_to_num(x, nan=0.0)

        # Create land cover class one-hot encoding, put it where the land cover integer was.
        # Skipped entirely when no kept feature lives in the resulting [16, 32] block -- the
        # channels would be built and then immediately dropped. Downstream indices are
        # remapped in __init__ (see remap_indices_without_one_hot), so x stays 24-wide here.
        if not self._skip_one_hot:
            new_shape = (x.shape[0], x.shape[2], x.shape[3],
                         self.one_hot_matrix.shape[0])
            # -1 because land cover classes start at 1
            landcover_classes_flattened = x[:, 16, ...].long().flatten() - 1
            landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
                new_shape).permute(0, 3, 1, 2)
            x = torch.concatenate(
                [x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x, y

    def augment(self, x, y):
        """_summary_ Applies geometric transformations: 
          1. random square cropping, preferring images with a) fire pixels in the output and b) (with much less weight) fire pixels in the input
          2. rotate by multiples of 90°
          3. flip horizontally and vertically
        Adjustment of angles is done as in https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/image_utils.py

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """
    
        # Need square crop to prevent rotation from creating/destroying data at the borders, due to uneven side lengths.
        # Try several crops, prefer the ones with most fire pixels in output, followed by most fire_pixels in input
        best_n_fire_pixels = -1
        L = self.crop_side_length

        if self._crop_search_materialises:
            # Original path, kept for A/B verification (WSTS_DISABLE_CROP_OPT=1).
            best_crop = (None, None)

            for i in range(10):
                top = np.random.randint(0, x.shape[-2] - L)
                left = np.random.randint(0, x.shape[-1] - L)
                x_crop = TF.crop(x, top, left, L, L)
                y_crop = TF.crop(y, top, left, L, L)

                # We really care about having fire pixels in the target. But if we don't find any there,
                # we care about fire pixels in the input, to learn to predict that no new observations will be made,
                # even though previous days had active fires.
                n_fire_pixels = x_crop[:, -1, ...].mean() + \
                    1000 * y_crop.float().mean()
                if n_fire_pixels > best_n_fire_pixels:
                    best_n_fire_pixels = n_fire_pixels
                    best_crop = (x_crop, y_crop)

            x, y = best_crop
        else:
            # Same search, without building the nine crops it throws away.
            #
            # The score reads ONLY x's active-fire channel and y, but the original cropped
            # the full (T, C, H, W) tensor ten times to get them. That waste scales with
            # n_leading_observations, which is why augment costs proportionally more on the
            # multitemporal configs.
            #
            # Bit-identical, by construction:
            #  - identical np.random.randint calls, same count and order, so the RNG stream
            #    and therefore the candidate crops are unchanged;
            #  - TF.crop is `img[..., top:top+h, left:left+w]`, so slicing here yields a view
            #    with the same shape, strides and storage offset that the original scored --
            #    literally the same tensor, hence the same mean, hence the same argmax;
            #  - `>` is preserved, so ties still keep the FIRST candidate, not the last.
            best_top = best_left = None

            for i in range(10):
                top = np.random.randint(0, x.shape[-2] - L)
                left = np.random.randint(0, x.shape[-1] - L)

                x_af = x[:, -1, top:top + L, left:left + L]
                y_crop = y[..., top:top + L, left:left + L]

                n_fire_pixels = x_af.mean() + 1000 * y_crop.float().mean()
                if n_fire_pixels > best_n_fire_pixels:
                    best_n_fire_pixels = n_fire_pixels
                    best_top, best_left = top, left

            # Materialise the winner through the same TF.crop call as before. The later angle
            # corrections write in place (`x[:, degree_features] = 360 - ...`), so keeping the
            # identical view/copy semantics here matters -- this is not the place to be clever.
            x = TF.crop(x, best_top, best_left, L, L)
            y = TF.crop(y, best_top, best_left, L, L)

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = 360 - \
                x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)

            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (x[:, self.indices_of_degree_features,
                                                          ...] - 90 * rotate) % 360
        return x, y

    def center_crop_x32(self, x, y):
        """_summary_ Crops the center of the image to side lengths that are a multiple of 32, 
        which the ResNet U-net architecture requires. Only used for computing the test performance.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        T, C, H, W = x.shape
        # Test eval crop = crop_side_length (128) centre crop. This is a committed
        # decision in the WSTS+ lineage (it replaced the original WSTS `H//32*32`
        # near-full-res crop); empirically it reproduces the paper's Table 2
        # (mean 0.447 +/- 0.095 vs 0.455 +/- 0.090) better than H//32*32
        # (0.438 +/- 0.113). Set WSTS_TEST_CROP=fullres for the original behaviour.
        if os.environ.get("WSTS_TEST_CROP") == "fullres":
            H_new = H // 32 * 32
            W_new = W // 32 * 32
        else:
            H_new = W_new = self.crop_side_length

        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        
        return x, y

    def zero_pad_to_size(self, x, y, desired_size=224):
        """Zero-pads the input images to ensure they fit the desired size."""
        T, C, H, W = x.shape
        if H < desired_size or W < desired_size:
            pad_height = max(0, desired_size - H)
            pad_width = max(0, desired_size - W)

            padding = (pad_width // 2, pad_width - pad_width // 2,  
                    pad_height // 2, pad_height - pad_height // 2)  
            
            x = torch.nn.functional.pad(x, padding)
            y = torch.nn.functional.pad(y, padding)

        return x, y
    
    def flatten_and_remove_duplicate_features_(self, x):
        """_summary_ For a simple U-Net, static and forecast features can be removed everywhere but in the last time step
        to reduce the number of features. Since that would result in different numbers of channels for different
        time steps, we flatten the temporal dimension. 
        Also discards features that we don't want to use. 

        Args:
            x (_type_): _description_ Input tensor data of shape (n_leading_observations, n_features, height, width)

        Returns:
            _type_: _description_
        """
        # Effective ids, not the raw config ones: they are already remapped to the 24-channel
        # space when the land-cover one-hot is skipped, and identical to the config otherwise.
        dynamic_feature_ids = torch.tensor(self._dynamic_ids_effective).int()

        x_dynamic_only = x[:-1, dynamic_feature_ids, :, :].flatten(start_dim=0, end_dim=1)
        x_last_day = x[-1, self._features_to_keep_effective, ...].squeeze(0)

        return torch.cat([x_dynamic_only, x_last_day], axis=0)

    @staticmethod
    def get_static_and_dynamic_feature_ids():
        """_summary_ Returns the indices of static and dynamic features.
        Static features include topographical features and one-hot encoded land cover classes.

        Returns:
            _type_: _description_ Tuple of lists of integers, first list contains static feature indices, second list contains dynamic feature indices.
        """
        static_feature_ids = [12,13,14] + list(range(16,33))
        dynamic_feature_ids = list(range(12)) + [15] + list(range(33,40))
        return static_feature_ids, dynamic_feature_ids

    @staticmethod
    def get_static_and_dynamic_features_to_keep(features_to_keep:Optional[List[int]]):
        """_summary_ Returns the indices of static and dynamic features that should be kept, based on the input list of feature indices to keep.

        Args:
            features_to_keep (Optional[List[int]]): _description_

        Returns:
            _type_: _description_
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_feature_ids()
        
        if type(features_to_keep) == list:
            dynamic_features_to_keep = list(set(dynamic_features_to_keep) & set(features_to_keep))
            dynamic_features_to_keep.sort()

        if type(features_to_keep) == list:
            static_features_to_keep = list(set(static_features_to_keep) & set(features_to_keep))
            static_features_to_keep.sort()

        return static_features_to_keep, dynamic_features_to_keep

    @staticmethod
    def get_n_features(n_observations:int, features_to_keep:Optional[List[int]], deduplicate_static_features:bool):
        """_summary_ Computes the number of features that the dataset will have after preprocessing, 
        considering the number of input observations, which features to keep or discard, and whether to deduplicate static features.

        Args:
            n_observations (int): _description_
            features_to_keep (Optional[List[int]]): _description_
            deduplicate_static_features (bool): _description_

        Returns:
            _type_: _description_ If deduplicate_static_features is True, returns the total number of features, flattened across all time steps. 
            Otherwise, returns the number of features per time step.
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_features_to_keep(features_to_keep)

        n_static_features = len(static_features_to_keep)
        n_dynamic_features = len(dynamic_features_to_keep)
        n_all_features = n_static_features + n_dynamic_features

        # If we deduplicate static features, we remove them from all time steps but the last one.
        # The last day then gets dynamic and static features. All other days only get dynamic features. 
        n_features = (int(deduplicate_static_features)*n_dynamic_features)*(n_observations-1) + n_all_features

        return n_features


    @staticmethod
    def img_dates_to_doys(img_dates):
        """_summary_ Converts a list of date strings to day of year values.

        Args:
            img_dates (_type_): _description_ List of date strings

        Returns:
            _type_: _description_ List of day of year values
        """
        date_format = "%Y-%m-%d"
        # In old preprocessing, the dates still had a TIF file extension, which is also removed here.
        return [datetime.strptime(img_date.replace(".tif", ""), date_format).timetuple().tm_yday for img_date in img_dates]

    @staticmethod
    def map_channel_index_to_features():
        """_summary_ Maps the channel index to the feature name.

        Returns:
            _type_: _description_
        """
        return {0: 'VIIRS band M11',
                1: 'VIIRS band I2',
                2: 'VIIRS band I1',
                3: 'NDVI',
                4: 'EVI2',
                5: 'total precipitation',
                6: 'wind speed',
                7: 'wind direction',
                8: 'minimum temperature',
                9: 'maximum temperature',
                10: 'energy release component',
                11: 'specific humidity',
                12: 'slope',
                13: 'aspect',
                14: 'elevation',
                15: 'pdsi',
                16: 'Landcover_Type1',
                17: 'forecast total_precipitation',
                18: 'forecast wind speed',
                19: 'forecast wind direction',
                20: 'forecast temperature',
                21: 'forecast specific humidity',
                22: 'active fire'}

    def get_generator_for_hdf5(self):
        """_summary_ Creates a generator that is used to turn the dataset into HDF5 files. It applies a few 
        preprocessing steps to the active fire features that need to be applied anyway, to save some computation.

        Yields:
            _type_: _description_ Generator that yields tuples of (year, fire_name, img_dates, lnglat, img_array) 
            where img_array contains all images available for the respective fire, preprocessed such 
            that active fire detection times are converted to hours. lnglat contains longitude and latitude
            of the center of the image.
        """

        for year, fires_in_year in self.imgs_per_fire.items():
            for fire_name, img_files in fires_in_year.items():
                imgs = []
                lnglat = None
                for img_path in img_files:
                    with rasterio.open(img_path, 'r') as ds:
                        imgs.append(ds.read())
                        if lnglat is None:
                            lnglat = ds.lnglat()
                x = np.stack(imgs, axis=0)

                # Get dates from filenames
                img_dates = [img_path.split("/")[-1].split("_")[0].replace(".tif", "")
                             for img_path in img_files]

                # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
                # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
                # them with 0 instead.
                x[:, -1, ...] = np.nan_to_num(x[:, -1, ...], nan=0)

                # Turn active fire detection time from hhmm to hh.
                x[:, -1, ...] = np.floor_divide(x[:, -1, ...], 100)
                yield year, fire_name, img_dates, lnglat, x


    def _compute_centroid_channels(self, x):
        """Compute the enabled fire-centroid channels as constant (broadcast) spatial maps.

        Operates on the preprocessed tensor `(n_peek + T, C, H, W)` before feature filtering,
        where the leading `self._n_peek_frames` rows are the peek-back day(s) that the model
        will never see. The binary active-fire mask is always the LAST channel --
        preprocess_and_augment appends it after standardization -- so it is indexed with -1
        rather than a fixed 39: C is 24 rather than 40 whenever the land-cover one-hot is
        skipped (self._skip_one_hot).

        Every frame in x has been through the SAME crop, rotation and flip, because
        preprocess_and_augment runs on the whole stack. That is what makes an inter-frame
        centroid displacement meaningful: a separately-loaded peek frame would sit in a
        different spatial frame and the "velocity" would be measuring the crop offset -- which,
        since the crop search is fire-biased, would be a systematic bias pointing at the fire.

        Returns:
            Tensor of shape `(n, H, W)` with `n == len(self._centroid_channel_names)
            == T * (enabled families)`, family-major and oldest-first within a family, matching
            CENTROID_CHANNEL_SPEC. Position is patch-relative in [-1, 1]; velocity is the
            centroid's displacement from the previous day, normalized by H/4 (resp. W/4).
            Position falls back to the patch centre on a fire-free frame (which normalizes to
            exactly 0.0 -- deliberately indistinguishable from a genuinely centred fire), and
            velocity is gated to zero unless both the frame and its predecessor hold fire. Both
            fallbacks apply whether or not the matching validity channel is emitted: "validity
            off" means gate *silently* (EXPERIMENTS.md, "Decided conventions", 2026-08-05).
        """
        n_peek = self._n_peek_frames
        T_total, C, H, W = x.shape
        T = T_total - n_peek          # frames the model will actually see

        fire_masks = x[:, -1, :, :]   # (T_total, H, W), binary active fire

        totals = fire_masks.sum(dim=(-2, -1))   # (T_total,) fire-pixel count per frame
        totals_safe = totals.clamp(min=1.0)     # avoids 0/0 on fire-free frames

        h_idx = torch.arange(H, dtype=torch.float32)
        w_idx = torch.arange(W, dtype=torch.float32)
        cx_all = (fire_masks * h_idx.view(H, 1)).sum(dim=(-2, -1)) / totals_safe  # (T_total,)
        cy_all = (fire_masks * w_idx.view(1, W)).sum(dim=(-2, -1)) / totals_safe  # (T_total,)

        # Validity is read off the RAW frame totals, BEFORE the patch-centre fallback below
        # overwrites cx_all/cy_all -- otherwise the flag would describe the substituted value
        # instead of the data. `a[n_peek:] & a[n_peek-1:-1]` is the "this frame AND its
        # predecessor" idiom: two shifted views of one array, so the pairing is visible in the
        # slice rather than hidden in a loop index. Note `&` on tensors, not Python `and`, and
        # note that `&` binds TIGHTER than `>` -- every comparison needs its own parentheses.
        frame_has_fire = totals > 0                        # (T_total,) bool
        pos_valid = frame_has_fire[n_peek:].float()        # (T,)
        if n_peek:
            vel_valid = (frame_has_fire[n_peek:] & frame_has_fire[n_peek - 1:-1]).float()  # (T,)
        else:
            vel_valid = torch.zeros(T, dtype=torch.float32)

        # Fall back to the patch centre on fire-free frames.
        cx_all = torch.where(totals >= 1.0, cx_all, torch.full_like(cx_all, H / 2.0))
        cy_all = torch.where(totals >= 1.0, cy_all, torch.full_like(cy_all, W / 2.0))

        # Each value is a (T,) tensor -- one entry per model-visible frame. These `if`s decide
        # what is cheap and legal to COMPUTE.
        channels = {}
        if self.use_centroid_position:
            # A fire-free frame already normalizes to 0.0 (H/2 / (H/2) - 1), so no extra gating
            # by pos_valid is needed -- and that 0.0 is intentionally confusable with a genuinely
            # centred fire, which is what makes the validity channel a test of *disclosure*.
            channels["centroid_x"] = cx_all[n_peek:] / (H / 2.0) - 1.0
            channels["centroid_y"] = cy_all[n_peek:] / (W / 2.0) - 1.0
        if self.use_centroid_velocity:
            assert n_peek, "velocity channels require a peek-back frame"
            # /4.0 is a heuristic: a fire centroid is unlikely to travel more than a quarter of
            # the patch in a day, so this puts velocity on roughly the same scale as position.
            channels["centroid_vx"] = \
                (cx_all[n_peek:] - cx_all[n_peek - 1:-1]) / (H / 4.0) * vel_valid
            channels["centroid_vy"] = \
                (cy_all[n_peek:] - cy_all[n_peek - 1:-1]) / (W / 4.0) * vel_valid
        if self.use_centroid_position_validity:
            channels["position_valid"] = pos_valid
        if self.use_centroid_velocity_validity:
            channels["velocity_valid"] = vel_valid

        # CENTROID_CHANNEL_SPEC decides what is EMITTED and in what order; the `if`s above only
        # decide what is computed. Stacking in SPEC order and then flattening gives exactly the
        # family-major, oldest-first layout centroid_channel_names() declares, and the assert is
        # the seam that stops the two halves drifting apart silently.
        families = [f for f, _ in FireSpreadDataset.CENTROID_CHANNEL_SPEC if f in channels]
        stacked = torch.stack([channels[f] for f in families])          # (n_families, T)
        assert stacked.numel() == len(self._centroid_channel_names), (
            f"computed {stacked.numel()} channels but spec declares "
            f"{len(self._centroid_channel_names)}: {self._centroid_channel_names}")

        # reshape -> (n, 1, 1), then broadcast to (n, H, W). .expand() returns a zero-stride
        # view, so .contiguous() before it crosses the DataLoader's collate/pinning boundary.
        return stacked.reshape(-1, 1, 1).expand(-1, H, W).contiguous()

    @staticmethod
    def get_n_features(n_observations:int, features_to_keep:Optional[List[int]], deduplicate_static_features:bool):
        """_summary_ Computes the number of features that the dataset will have after preprocessing, 
        considering the number of input observations, which features to keep or discard, and whether to deduplicate static features.

        Args:
            n_observations (int): _description_
            features_to_keep (Optional[List[int]]): _description_
            deduplicate_static_features (bool): _description_

        Returns:
            _type_: _description_ If deduplicate_static_features is True, returns the total number of features, flattened across all time steps. 
            Otherwise, returns the number of features per time step.
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_features_to_keep(features_to_keep)

        n_static_features = len(static_features_to_keep)
        n_dynamic_features = len(dynamic_features_to_keep)
        n_all_features = n_static_features + n_dynamic_features

        # If we deduplicate static features, we remove them from all time steps but the last one.
        # The last day then gets dynamic and static features. All other days only get dynamic features. 
        
        #BUG IN LOGIC! When not deduplicating static features, n_features does not account for all n_observations days of n_all_features
        #n_features = (int(deduplicate_static_features)*n_dynamic_features)*(n_observations-1) + n_all_features
        #REFLACEMENT WITH CLEAR LOGIC:
        if deduplicate_static_features:
            n_features= n_all_features + (n_observations-1)*(n_dynamic_features)
        else:
            n_features = n_all_features * n_observations

        return n_features
