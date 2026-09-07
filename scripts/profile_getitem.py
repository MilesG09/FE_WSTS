#!/usr/bin/env python
"""cProfile FireSpreadDataset.__getitem__ to see where per-sample CPU time goes.

System counters (GPU util, worker CPU%, disk MB/s) can say "nothing is saturated"
without saying what the workers are actually spending their time on. This opens
that box: it profiles the real __getitem__ path in-process and ranks the callees.

Run it while the GPU sweep is idle for clean numbers, or accept contention noise --
the RANKING is stable either way, which is what matters for choosing a lever.

Usage:
    python scripts/profile_getitem.py --profile veg_t1 --samples 120
"""

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import add_dataset_args, build_dataset, describe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--top", type=int, default=18)
    add_dataset_args(ap)
    args = ap.parse_args()

    ds = build_dataset(args, is_train=True)
    print(f"[{args.profile}] {describe(ds)}\n")
    idxs = [int(i) for i in np.linspace(0, len(ds) - 1, args.samples)]

    # Warm the file-handle cache so we profile steady state, not first-open cost.
    for i in idxs[:5]:
        ds[i]

    np.random.seed(0)
    torch.manual_seed(0)

    pr = cProfile.Profile()
    pr.enable()
    for i in idxs:
        ds[i]
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(args.top)
    out = s.getvalue()

    # Trim cProfile's preamble to the table itself.
    lines = out.splitlines()
    start = next((n for n, l in enumerate(lines) if "ncalls" in l), 0)
    print(f"=== {args.profile}: cProfile of {len(idxs)} __getitem__ calls "
          f"(n_lead={ds.n_leading_observations}) ===")
    print("\n".join(lines[start:start + args.top + 2]))

    total = pstats.Stats(pr).total_tt
    print(f"\ntotal {total:.2f}s for {len(idxs)} samples "
          f"= {1000 * total / len(idxs):.1f} ms/sample")


if __name__ == "__main__":
    main()
