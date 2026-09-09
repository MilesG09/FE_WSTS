#!/usr/bin/env python
"""Shared dataset-construction helper for the efficiency verification/benchmark scripts.

Every training speed-up in FireSpreadDataset is gated on a `WSTS_DISABLE_*` env flag so
the original code path stays reachable. Proving a speed-up changed nothing then means
building the SAME dataset twice -- flag set, flag clear -- and comparing outputs. All the
scripts that do that need identical construction logic, so it lives here once.

A "profile" is a named (data config, override) pair chosen to exercise a distinct branch
of the optimised code. Which branches matter:

    features_to_keep touches [16, 32]?   -> land-cover one-hot expansion is load-bearing,
                                            the skip must disable itself
    features_to_keep is None?            -> hyperslab must disable itself (read everything)
    remove_duplicate_features and T > 1? -> flatten_and_remove_duplicate_features_ runs,
                                            which indexes with the remapped dynamic ids
    return_doy?                          -> the doy read shares the cached h5py handle

A speed-up that is only checked on the happy path is not checked.
"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[1]


def _data_dir_from_env_local() -> Optional[str]:
    """DATA_DIR out of the untracked env.local.sh, without executing it.

    The shell scripts source that file; these python gates cannot, so they read the one
    assignment they need. Deliberately a plain text scan and not a shell call: importing a
    helper must never run arbitrary code from a file on the machine.
    """
    path = REPO / "env.local.sh"
    if not path.is_file():
        return None
    found = None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?DATA_DIR=(.*)$", line)
        if m:  # last assignment wins, matching how the shell would evaluate the file
            value = m.group(1).strip().strip('"').strip("'")
            if value:
                found = value
    return found


# Dataset root, since every checked-in cfg points at a stale cluster path. Machine-local, so
# it is never hardcoded here: WSTS_DATA_DIR (one-off override) beats env.local.sh (the
# machine's standing setting). None when neither is set -- --data-dir then becomes required,
# which fails loudly instead of globbing an empty directory on someone else's box.
DEFAULT_DATA_DIR = os.environ.get("WSTS_DATA_DIR") or os.environ.get(
    "DATA_DIR") or _data_dir_from_env_local()

# name -> (config file, overrides applied on top of it)
PROFILES = {
    # The T=1 vegetation reproduction: the configuration every WSTS+ Table 2 comparison
    # in this repo is run under. If any optimisation breaks, it must break here.
    "veg_t1": ("cfgs/data_monotemporal_veg_features.yaml", {}),
    # Same features at T=5 (cfgs/unet/saad/table_5_multi_veg.yaml). remove_duplicate_features
    # with T > 1 is the ONLY path that reaches flatten_and_remove_duplicate_features_, and
    # therefore the only one that exercises the remapped dynamic feature ids.
    "veg_t5": ("cfgs/data_monotemporal_veg_features.yaml",
               {"n_leading_observations": 5}),
    # features_to_keep spans 16..32, so the one-hot expansion is load-bearing: this profile
    # checks the skip correctly refuses to engage while the hyperslab still does.
    "multi_t5": ("cfgs/data_multitemporal_multi_features.yaml", {}),
    # features_to_keep=None -> both optimisations self-disable. Guards against a speed-up
    # that "helps" a config it was never meant to touch.
    "full_t1": ("cfgs/data_monotemporal_full_features.yaml", {}),
    # return_doy=True reads dataset attrs alongside the image rows, through the same cached
    # h5py handle.
    "full_t5_doy": ("cfgs/data_multitemporal_full_features.yaml", {}),
}

DISABLE_FLAGS = (
    "WSTS_DISABLE_HDF5_READ_OPT",
    "WSTS_DISABLE_CROP_OPT",
    "WSTS_DISABLE_ONEHOT_SKIP",
)


def add_dataset_args(parser) -> None:
    """Register the dataset-selection flags every script here shares."""
    parser.add_argument("--profile", default="veg_t1", choices=sorted(PROFILES),
                        help="named dataset configuration to test (see PROFILES)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        required=DEFAULT_DATA_DIR is None,
                        help="dataset root; checked-in cfgs point at stale cluster paths")
    parser.add_argument("--years", default="2020",
                        help="comma-separated fire years to include")
    parser.add_argument("--stats-years", default="2018,2019",
                        help="comma-separated years for standardisation statistics")


def _years(spec: str) -> list:
    return [int(y) for y in spec.split(",") if y.strip()]


def profile_config(profile: str) -> dict:
    """Resolve a profile to the concrete kwargs the dataset takes."""
    rel, overrides = PROFILES[profile]
    cfg = yaml.safe_load((REPO / rel).read_text())
    cfg.update(overrides)
    return cfg


def build_dataset(args, is_train: bool, flags: Optional[list] = None,
                  disable: bool = False):
    """Build a FireSpreadDataset for `args.profile`, with `flags` set or cleared.

    The flags are read in FireSpreadDataset.__init__, so they must be set BEFORE
    construction; the import is deliberately local so this stays true even if a caller
    has already imported the module.
    """
    for f in (flags or []):
        if disable:
            os.environ[f] = "1"
        else:
            os.environ.pop(f, None)

    from src.dataloader.FireSpreadDataset import FireSpreadDataset

    cfg = profile_config(args.profile)
    return FireSpreadDataset(
        data_dir=args.data_dir,
        included_fire_years=_years(args.years),
        n_leading_observations=cfg["n_leading_observations"],
        crop_side_length=cfg["crop_side_length"],
        load_from_hdf5=cfg["load_from_hdf5"],
        is_train=is_train,
        remove_duplicate_features=cfg["remove_duplicate_features"],
        stats_years=tuple(_years(args.stats_years)),
        # Held at None on purpose: it only skips leading samples, and skipping them
        # identically in both arms of an A/B tests strictly less of the index space.
        n_leading_observations_test_adjustment=None,
        features_to_keep=cfg.get("features_to_keep"),
        return_doy=cfg.get("return_doy", False),
        is_pad=cfg.get("is_pad", False),
    )


def describe(ds) -> str:
    """One-line summary of which optimisations actually engaged, for the run header."""
    return (f"n_lead={ds.n_leading_observations} "
            f"remove_dup={ds.remove_duplicate_features} "
            f"is_pad={bool(ds.is_pad)} return_doy={ds.return_doy}\n"
            f"    features_to_keep : {ds.features_to_keep}\n"
            f"    hyperslab reads  : {ds._raw_channels_needed}\n"
            f"    one-hot skipped  : {ds._skip_one_hot} "
            f"(effective keep: {ds._features_to_keep_effective})\n"
            f"    crop search      : "
            f"{'original (materialising)' if ds._crop_search_materialises else 'optimised (sliced)'}")
