#!/usr/bin/env python
"""Empirically determine which RAW hdf5 channels the pipeline output depends on.

Why empirical: the 40-channel space `features_to_keep` indexes is built inside
preprocess_and_augment (append active-fire mask 23->24, then one-hot landcover 16 -> 17
channels, giving 16+17+7 = 40). Mapping post-indices back to raw indices by reading code
is exactly the kind of reasoning that silently goes wrong, and a wrong mapping would
corrupt a 12-fold run without raising anything.

Method (code-agnostic): take a real sample, zero one raw channel at a time, push it
through the UNMODIFIED pipeline, and see whether (x, y) changes. Channels whose zeroing
changes the output are load-bearing; the rest are provably discarded.

That "provably discarded" is what licenses the hyperslab optimisation: if zeroing a
channel cannot change the output, then never reading it cannot change the output either.
The script finishes by checking the measured set against what
FireSpreadDataset.raw_channels_for_features() claims, and fails if the code reads less
than the data actually depends on.

Usage:
    python scripts/verify_channel_dependency.py --profile veg_t1
    python scripts/verify_channel_dependency.py --profile multi_t5 --train
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import add_dataset_args, build_dataset  # noqa: E402


def deterministic_getitem(ds, idx: int, seed: int = 0):
    """__getitem__ does random cropping/augmentation; pin every RNG it can touch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ds[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6, help="distinct samples to test")
    ap.add_argument("--train", action="store_true",
                    help="use is_train=True (augmentation on)")
    add_dataset_args(ap)
    args = ap.parse_args()

    # The dependency question is about the PIPELINE, so it must be answered with the
    # hyperslab off -- otherwise the read itself has already dropped channels and every
    # skipped channel would trivially look irrelevant. The other flags stay at their
    # defaults so the answer describes the code that actually ships.
    ds = build_dataset(args, is_train=args.train,
                       flags=["WSTS_DISABLE_HDF5_READ_OPT"], disable=True)
    print(f"profile={args.profile}  n_lead={ds.n_leading_observations}  "
          f"is_train={args.train}  len={len(ds)}")
    print(f"features_to_keep={ds.features_to_keep}")

    original_load = ds.load_imgs
    zero_channel = {"c": None}

    def patched_load(year, name, idx):
        out = original_load(year, name, idx)
        x, y = out[0], out[1]
        if zero_channel["c"] is not None:
            x = x.copy()
            x[:, zero_channel["c"], ...] = 0.0
        return (x, y) if len(out) == 2 else (x, y, out[2])

    # load_imgs returns the RAW (T, 23, H, W) array before any preprocessing, so
    # patching here is the cleanest injection point for the zeroing experiment.
    ds.load_imgs = patched_load

    n_raw = None
    needed = set()
    indices = [int(i) for i in np.linspace(0, len(ds) - 1, args.samples)]

    for si, idx in enumerate(indices):
        zero_channel["c"] = None
        ref_x, ref_y = deterministic_getitem(ds, idx)[:2]
        ref_x = torch.as_tensor(ref_x).clone()
        ref_y = torch.as_tensor(ref_y).clone()

        if n_raw is None:
            raw = original_load(*ds.find_image_index_from_dataset_index(idx))[0]
            n_raw = raw.shape[1]
            print(f"raw channels on disk: {n_raw}   output x shape: {tuple(ref_x.shape)}\n")

        for c in range(n_raw):
            zero_channel["c"] = c
            got_x, got_y = deterministic_getitem(ds, idx)[:2]
            same = torch.allclose(
                torch.as_tensor(got_x), ref_x, equal_nan=True
            ) and torch.allclose(torch.as_tensor(got_y), ref_y, equal_nan=True)
            if not same:
                needed.add(c)
        print(f"  sample {si + 1}/{len(indices)} (idx {idx}) -> cumulative needed: "
              f"{sorted(needed)}")

    zero_channel["c"] = None
    ds.load_imgs = original_load

    from src.dataloader.FireSpreadDataset import FireSpreadDataset
    claimed = FireSpreadDataset.raw_channels_for_features(ds.features_to_keep)

    print(f"\n=== RESULT ({args.profile}) ===")
    print(f"raw channels that affect the output : {sorted(needed)}  "
          f"({len(needed)} of {n_raw})")
    print(f"raw channels provably discarded     : {sorted(set(range(n_raw)) - needed)}")
    print(f"raw channels the code reads         : "
          f"{claimed if claimed is not None else f'ALL {n_raw} (optimisation off)'}")
    print(f"\npotential I/O reduction: {n_raw}/{max(1, len(needed))} = "
          f"{n_raw / max(1, len(needed)):.1f}x")

    if claimed is None:
        print("\nOK - the code reads every channel for this profile, so it cannot be "
              "missing one. (No I/O saving here either.)")
        return 0

    # The only unsafe direction: a channel the output depends on that the code does not
    # read. Reading MORE than measured is merely conservative -- a channel can look
    # irrelevant on the sampled fires (e.g. active fire on a sample with no fire pixels)
    # and matter elsewhere, which is exactly why several samples are tested.
    missing = sorted(needed - set(claimed))
    extra = sorted(set(claimed) - needed)
    if missing:
        print(f"\nFAIL - code skips load-bearing channel(s) {missing}. Do not train.")
        return 1
    if extra:
        print(f"\nPASS - code reads {extra} which these samples did not exercise "
              "(conservative, safe).")
    else:
        print("\nPASS - the channels the code reads are exactly the channels that matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
