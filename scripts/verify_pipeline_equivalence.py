#!/usr/bin/env python
"""A/B any dataloader optimisation against the original path and prove they match.

Each training speed-up in FireSpreadDataset keeps its original code path behind a
`WSTS_DISABLE_*` env flag. This builds the dataset twice -- once with those flags set
(reference) and once without (optimised) -- pins the RNG, and compares (x, y)
element-for-element.

Why bit-identical and not "close enough": these runs are compared against the WSTS+
paper and against each other across folds. An optimisation that shifts values by 1e-7
does not throw, it just quietly makes the reproduction non-comparable and invalidates
the science. Either the tensors match exactly or the change does not ship.

Exit code 0 only if every compared sample matches. Anything else means do not train.

Usage:
    python scripts/verify_pipeline_equivalence.py --flags WSTS_DISABLE_CROP_OPT
    python scripts/verify_pipeline_equivalence.py --flags all --profiles all --samples 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import (  # noqa: E402
    DISABLE_FLAGS, PROFILES, add_dataset_args, build_dataset, describe)


def get(ds, idx: int, seed: int):
    """Pin every RNG the crop/augment path can draw from, so candidates match."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = ds[idx]
    return torch.as_tensor(out[0]), torch.as_tensor(out[1])


def compare(args, profile: str, is_train: bool, flags: list) -> tuple:
    args.profile = profile
    ref = build_dataset(args, is_train, flags, disable=True)
    opt = build_dataset(args, is_train, flags, disable=False)

    idxs = [int(i) for i in np.linspace(0, len(ref) - 1, args.samples)]
    bad = []
    for n, idx in enumerate(idxs):
        seed = 4000 + n
        rx, ry = get(ref, idx, seed)
        ox, oy = get(opt, idx, seed)
        if rx.shape != ox.shape or ry.shape != oy.shape:
            bad.append((idx, f"shape {tuple(rx.shape)} vs {tuple(ox.shape)}"))
        elif not (torch.equal(rx, ox) and torch.equal(ry, oy)):
            d = (rx.float() - ox.float()).abs().max().item()
            bad.append((idx, f"max|dx|={d:.3e}"))

    tag = f"{profile:<12} is_train={str(is_train):<5}"
    if bad:
        print(f"  {tag}  FAIL  {len(bad)}/{len(idxs)} mismatched")
        for idx, why in bad[:5]:
            print(f"      idx {idx}: {why}")
    else:
        print(f"  {tag}  ok    {len(idxs)}/{len(idxs)} bit-identical  "
              f"x{tuple(get(opt, idxs[0], 0)[0].shape)}")
    return len(bad), len(idxs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True,
                    help="comma-separated WSTS_DISABLE_* env flags, or 'all'")
    ap.add_argument("--profiles", default="veg_t1,veg_t5,multi_t5",
                    help="comma-separated profile names, or 'all'")
    ap.add_argument("--samples", type=int, default=30)
    add_dataset_args(ap)
    args = ap.parse_args()

    flags = list(DISABLE_FLAGS) if args.flags.strip() == "all" else [
        f.strip() for f in args.flags.split(",") if f.strip()]
    unknown = [f for f in flags if f not in DISABLE_FLAGS]
    if unknown:
        print(f"unknown flag(s): {unknown}\nknown: {list(DISABLE_FLAGS)}")
        return 2

    profiles = sorted(PROFILES) if args.profiles.strip() == "all" else [
        p.strip() for p in args.profiles.split(",") if p.strip()]

    print(f"A/B flags : {', '.join(flags)}")
    print(f"profiles  : {', '.join(profiles)}   samples/config: {args.samples}")
    print(f"data_dir  : {args.data_dir}   years: {args.years}\n")

    # Show what each profile actually engages, so a run that silently tested nothing
    # is visible rather than passing vacuously.
    inert = []
    for p in profiles:
        args.profile = p
        ds = build_dataset(args, is_train=True, flags=flags, disable=False)
        print(f"  [{p}] {describe(ds)}")
        engaged = (ds._raw_channels_needed is not None) or ds._skip_one_hot \
            or not ds._crop_search_materialises
        if not engaged:
            inert.append(p)
    print()

    total_bad = total_n = 0
    for p in profiles:
        for is_train in (False, True):
            b, n = compare(args, p, is_train, flags)
            total_bad += b
            total_n += n

    print()
    if total_bad:
        print(f"FAIL - {total_bad}/{total_n} samples differ. Do not train on this.")
        return 1
    print(f"PASS - all {total_n} samples bit-identical across every configuration.")
    if inert:
        print(f"NOTE - no optimisation engaged for: {', '.join(inert)} "
              "(expected for full-feature profiles; they still check the fallback path).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
