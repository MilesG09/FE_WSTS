#!/usr/bin/env python
"""Second, independent proof that the dataloader optimisations are exact.

scripts/verify_pipeline_equivalence.py A/Bs via `WSTS_DISABLE_*` env flags read in
__init__. That is the right check for "does the shipped code behave like the old code",
but it shares a failure mode with the thing it tests: if a flag were ever misspelled or
stopped being read, BOTH datasets would be built the same way and the comparison would
pass while testing nothing.

This closes that hole from the other side. It builds two datasets normally -- no env
flags at all -- and then forces one back onto the legacy path by resetting the four
instance attributes the optimisations hang off. The optimised object is the one training
actually uses, byte for byte.

WHY IT MATTERS: if any index mapping is off, the model trains on the WRONG CHANNELS
silently -- no shape error, because the widths still line up. That failure mode looks
like a uniform performance offset across every run, which is indistinguishable from a
genuine scientific result until someone checks.

Usage:  python scripts/verify_dataload_optimization.py --profiles veg_t1,veg_t5,multi_t5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import PROFILES, add_dataset_args, build_dataset  # noqa: E402

from src.dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402


def legacy(ds) -> None:
    """Force the pre-optimisation code path on an existing dataset object.

    FOUR attributes, not two. flatten_and_remove_duplicate_features_ (only reached when
    remove_duplicate_features and n_lead > 1) reads `_dynamic_ids_effective` as well as
    `_features_to_keep_effective`. Resetting only the latter leaves a remapped 24-channel
    index list being applied to a restored 40-channel tensor -- a mismatch that looks
    exactly like a bug in the optimisation and is not. Getting this wrong is how a
    verification script manufactures its own false positive.
    """
    ds._raw_channels_needed = None          # full-width read        (load_imgs)
    ds._skip_one_hot = False                # build the one-hot     (preprocess_and_augment)
    ds._crop_search_materialises = True     # original crop search  (augment)
    ds._features_to_keep_effective = ds.features_to_keep
    _, dynamic_ids = FireSpreadDataset.get_static_and_dynamic_features_to_keep(
        ds.features_to_keep)
    ds._dynamic_ids_effective = dynamic_ids


def get(ds, idx: int, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = ds[idx]
    return torch.as_tensor(out[0]), torch.as_tensor(out[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="veg_t1,veg_t5,multi_t5")
    ap.add_argument("--samples", type=int, default=12)
    add_dataset_args(ap)
    args = ap.parse_args()

    profiles = sorted(PROFILES) if args.profiles.strip() == "all" else [
        p.strip() for p in args.profiles.split(",") if p.strip()]

    all_ok = True
    total = 0
    for profile in profiles:
        args.profile = profile
        for is_train in (False, True):
            ds_new = build_dataset(args, is_train)
            ds_old = build_dataset(args, is_train)
            legacy(ds_old)

            if is_train is False:
                print(f"\n=== {profile} ({len(ds_new)} samples) ===")
                print(f"  optimised: skip_one_hot={ds_new._skip_one_hot}  "
                      f"raw_channels_needed={ds_new._raw_channels_needed}")
                print(f"             features_effective={ds_new._features_to_keep_effective}")

            idxs = [int(i) for i in np.linspace(0, len(ds_new) - 1, args.samples)]
            bad = 0
            worst = 0.0
            for n, i in enumerate(idxs):
                # Both objects draw from the same pinned stream, so the augmented path
                # picks the same crop and flips and the comparison stays meaningful.
                xn, yn = get(ds_new, i, 7000 + n)
                xo, yo = get(ds_old, i, 7000 + n)
                if xn.shape != xo.shape:
                    print(f"  is_train={is_train} idx {i:>5}: SHAPE {tuple(xn.shape)} "
                          f"vs {tuple(xo.shape)}")
                    bad += 1
                    continue
                if not (torch.equal(xn, xo) and torch.equal(yn, yo)):
                    worst = max(worst, (xn.float() - xo.float()).abs().max().item())
                    bad += 1
            total += len(idxs)
            all_ok &= bad == 0
            status = "ok   " if bad == 0 else f"FAIL {bad}/{len(idxs)}"
            extra = "" if bad == 0 else f"  max|dx|={worst:.3e}"
            print(f"  is_train={str(is_train):<5} {status} "
                  f"{len(idxs)} samples  x{tuple(xn.shape)}{extra}")

    print("\n" + "=" * 74)
    print(f"VERDICT: optimisations are EXACT over {total} samples -- the data path is "
          "unchanged." if all_ok else
          "VERDICT: OPTIMISATIONS CHANGE THE DATA. Every run since is suspect.")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
