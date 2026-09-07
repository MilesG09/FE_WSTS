#!/usr/bin/env python
"""Prove the cached HDF5 handles survive the parent-scan-then-fork path intact.

FireSpreadDataset caches open h5py.File handles so repeated reads from one fire stay
cheap. That is safe as long as every handle is opened AFTER the DataLoader workers fork,
which is normally guaranteed: workers start with an empty cache.

FireSpreadDataModule breaks that guarantee. Its ignition filters (`ignition_only_val_test`
is set in cfgs/data_monotemporal_veg_features.yaml) iterate the entire dataset in the
PARENT process, so the parent's cache is full of live descriptors before any worker
forks. HDF5 is not fork-safe across shared descriptors, and the failure mode is silent:
concurrent reads through inherited descriptors can return WRONG DATA with no exception --
a corruption that would look like a modelling result.

The test therefore reproduces the dangerous ordering exactly -- scan in the parent, then
read the same samples through a multi-worker DataLoader -- and compares against samples
read with the cache never populated in the parent. It also asserts the guards actually
fired, so a future refactor that removes them fails here instead of in a sweep.

Usage:  python scripts/verify_hdf5_cache_fork_safety.py --profile veg_t1 --workers 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import add_dataset_args, build_dataset  # noqa: E402


def digest(ds, picks, workers):
    """Read `picks` through a real multi-worker DataLoader, in order."""
    dl = DataLoader(Subset(ds, picks), batch_size=4, shuffle=False,
                    num_workers=workers,
                    persistent_workers=workers > 0,
                    **({"prefetch_factor": 2} if workers > 0 else {}))
    out = []
    for batch in dl:
        x, y = batch[0], batch[1]
        out.append((float(x.double().sum()), float(x.double().square().sum()),
                    float(y.double().sum())))
    del dl
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=48)
    add_dataset_args(ap)
    args = ap.parse_args()

    # is_train=False everywhere: augmentation would make two DataLoader passes differ
    # for RNG reasons that have nothing to do with file handles.
    clean = build_dataset(args, is_train=False)
    picks = [int(i) for i in np.linspace(0, len(clean) - 1, args.samples)]
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"profile={args.profile}  workers={args.workers}  samples={len(picks)}\n")

    ref = digest(clean, picks, args.workers)

    # Now the dangerous ordering: fill the parent's handle cache first.
    dirty = build_dataset(args, is_train=False)
    for i in picks:
        dirty[i]
    n_open = len(dirty._hdf5_file_cache)
    print(f"  parent scan opened {n_open} cached handle(s) before forking")
    if n_open == 0:
        print("  FAIL - the parent scan cached nothing, so the hazard was not reproduced.")
        return 2

    got = digest(dirty, picks, args.workers)
    ok = got == ref
    print(f"  {'ok  ' if ok else 'FAIL'} fork-after-parent-scan matches clean read")

    # And the tidy path the data module uses.
    tidy = build_dataset(args, is_train=False)
    for i in picks:
        tidy[i]
    tidy.close_hdf5_cache()
    closed_ok = len(tidy._hdf5_file_cache) == 0
    got2 = digest(tidy, picks, args.workers)
    ok2 = got2 == ref
    print(f"  {'ok  ' if closed_ok else 'FAIL'} close_hdf5_cache() empties the cache")
    print(f"  {'ok  ' if ok2 else 'FAIL'} read after close_hdf5_cache() matches clean read")

    print()
    if ok and ok2 and closed_ok:
        print("PASS - the handle cache is safe across the parent-scan-then-fork path.")
        return 0
    print("FAIL - inherited HDF5 handles changed the data. Do not train on this.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
