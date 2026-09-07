#!/usr/bin/env python
"""Alternating in-process A/B of __getitem__ cost, for effects near the noise floor.

Why this exists: the crop and one-hot optimisations each move per-sample time by
roughly 10-20%, and run-to-run variance between separate `python` invocations is
about the same size. Two runs cannot distinguish those. This removes the two biggest
confounds:

  * separate-process variance -- both variants are timed in ONE process;
  * page-cache order effects -- the variants alternate (A B A B ...) and every
    variant touches the same sample indices, after a warmup pass, so neither gets to
    run against a colder cache than the other.

Reports the MEDIAN of each variant's trials, which is robust to the occasional slow
trial from background activity, plus min/max so you can see the spread rather than
trusting a single ratio.

Uses time.perf_counter, not cProfile: cProfile's per-call overhead is large relative
to a ~15 ms/sample workload and would distort exactly the call-overhead differences
being measured.

Usage:
    python scripts/benchmark_getitem_ab.py --profile veg_t1 --flags WSTS_DISABLE_CROP_OPT
    python scripts/benchmark_getitem_ab.py --profile veg_t5 \
        --flags WSTS_DISABLE_CROP_OPT,WSTS_DISABLE_ONEHOT_SKIP --trials 7
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import (  # noqa: E402
    DISABLE_FLAGS, add_dataset_args, build_dataset, describe)


def timed_pass(ds, idxs) -> float:
    """Wall-clock ms/sample for one sweep over idxs."""
    np.random.seed(0)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    for i in idxs:
        ds[i]
    return (time.perf_counter() - t0) * 1000.0 / len(idxs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True,
                    help="comma-separated WSTS_DISABLE_* flags, or 'all'")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--trials", type=int, default=7)
    add_dataset_args(ap)
    args = ap.parse_args()

    flags = list(DISABLE_FLAGS) if args.flags.strip() == "all" else [
        f.strip() for f in args.flags.split(",") if f.strip()]
    ref = build_dataset(args, True, flags, disable=True)    # original path
    opt = build_dataset(args, True, flags, disable=False)   # optimised path
    print(f"[{args.profile}] {describe(opt)}\n")

    idxs = [int(i) for i in np.linspace(0, len(ref) - 1, args.samples)]

    # Warm both the page cache and each dataset's h5py handle cache before timing.
    timed_pass(ref, idxs)
    timed_pass(opt, idxs)

    ref_t, opt_t = [], []
    for _ in range(args.trials):
        ref_t.append(timed_pass(ref, idxs))   # alternate, so neither wins on cache state
        opt_t.append(timed_pass(opt, idxs))

    r, o = statistics.median(ref_t), statistics.median(opt_t)
    print(f"profile={args.profile}  flags={','.join(flags)}  "
          f"{args.samples} samples x {args.trials} trials\n")
    print(f"  original  : median {r:6.2f} ms/sample   "
          f"(min {min(ref_t):.2f}, max {max(ref_t):.2f})")
    print(f"  optimised : median {o:6.2f} ms/sample   "
          f"(min {min(opt_t):.2f}, max {max(opt_t):.2f})")
    print(f"\n  speedup   : {r / o:.3f}x   ({100 * (1 - o / r):+.1f}% time)")

    # If the distributions overlap, the ratio is not trustworthy -- say so rather than
    # letting a median difference imply more confidence than the data supports. The test
    # is the SYMMETRIC interval overlap: `min(ref) < max(opt)` alone fires whenever the
    # optimised variant is slower, which is precisely when the result is most worth
    # believing, and a false "it's just noise" there is how a regression gets shipped.
    if min(ref_t) <= max(opt_t) and min(opt_t) <= max(ref_t):
        print("\n  WARNING: trial ranges overlap -- effect is within noise at this trial "
              "count. Increase --trials/--samples before quoting this ratio.")
    elif o > r:
        print("\n  REGRESSION: the optimised path is reliably SLOWER here (ranges do not "
              "overlap). This configuration should not use it.")


if __name__ == "__main__":
    main()
