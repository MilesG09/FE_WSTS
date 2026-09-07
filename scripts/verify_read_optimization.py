#!/usr/bin/env python
"""Prove the hyperslab partial-read optimisation changes nothing about the data.

The optimisation in FireSpreadDataset.load_imgs reads only the raw HDF5 channels that
feed features_to_keep and zero-fills the rest. That is a claim about equivalence, and an
unverified equivalence claim is how a 12-fold run silently produces corrupt numbers. So:
build the SAME dataset twice -- once with the optimisation disabled via
WSTS_DISABLE_HDF5_READ_OPT, once with it on -- and compare (x, y) element-for-element.

This is the focused, single-optimisation check; scripts/verify_pipeline_equivalence.py
generalises it to any WSTS_DISABLE_* flag or combination. It also reports the read
reduction the mapping buys, so a mapping that silently degrades to "read everything" is
visible rather than passing quietly.

Exit code 0 only if every compared sample is bit-identical. Anything else is a hard
failure and the caller must not proceed to training.

Usage:
    python scripts/verify_read_optimization.py --profile veg_t1 --samples 40
    python scripts/verify_read_optimization.py --profile multi_t5 --samples 40 --train
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

FLAG = "WSTS_DISABLE_HDF5_READ_OPT"


def get(ds, idx: int, seed: int):
    """Pin RNG so random crop/augmentation choices match between the two datasets."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = ds[idx]
    return torch.as_tensor(out[0]), torch.as_tensor(out[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--train", action="store_true")
    add_dataset_args(ap)
    args = ap.parse_args()

    ref = build_dataset(args, args.train, [FLAG], disable=True)
    opt = build_dataset(args, args.train, [FLAG], disable=False)

    n_raw = ref.N_RAW_HDF5_CHANNELS
    print(f"profile={args.profile}  is_train={args.train}  len={len(ref)}")
    print(f"  reference path : read all {n_raw} channels "
          f"(_raw_channels_needed={ref._raw_channels_needed})")
    print(f"  optimised path : _raw_channels_needed={opt._raw_channels_needed}")
    if opt._raw_channels_needed is None:
        print("\nFAIL: optimisation inactive - nothing was actually tested. "
              "Pick a profile whose features_to_keep is a subset (e.g. veg_t1).")
        return 2
    n_read = len(opt._raw_channels_needed)
    print(f"  read reduction : {n_raw}/{n_read} = {n_raw / n_read:.1f}x fewer channels")

    idxs = [int(i) for i in np.linspace(0, len(ref) - 1, args.samples)]
    mismatches = []
    max_abs = 0.0

    for n, idx in enumerate(idxs):
        seed = 1000 + n
        rx, ry = get(ref, idx, seed)
        ox, oy = get(opt, idx, seed)

        if rx.shape != ox.shape or ry.shape != oy.shape:
            mismatches.append((idx, f"shape {tuple(rx.shape)} vs {tuple(ox.shape)}"))
            continue

        x_eq = torch.equal(rx, ox)
        y_eq = torch.equal(ry, oy)
        if not (x_eq and y_eq):
            d = (rx.float() - ox.float()).abs().max().item()
            max_abs = max(max_abs, d)
            mismatches.append((idx, f"x_equal={x_eq} y_equal={y_eq} max|dx|={d:.3e}"))

    print(f"\ncompared {len(idxs)} samples")
    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}")
        for idx, why in mismatches[:10]:
            print(f"  idx {idx}: {why}")
        print(f"\nFAIL - do not train on this. max|dx| = {max_abs:.3e}")
        return 1

    print("PASS - every sample bit-identical between full-read and partial-read paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
