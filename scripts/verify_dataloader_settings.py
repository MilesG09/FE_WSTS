#!/usr/bin/env python
"""Check which FireSpreadDataModule speed settings are output-neutral -- and which is not.

The dataset-level optimisations are proved bit-identical by
scripts/verify_pipeline_equivalence.py and scripts/verify_dataload_optimization.py. This
covers the other half: the DataLoader knobs, which act on the loader rather than on any
one sample, so they need a different test -- iterate whole epochs and compare the batches
that come out.

    prefetch_factor    expected NEUTRAL. Buffers batches ahead; changes neither the number
                       of worker RNG streams nor sample order.
    hdf5_cache_size    expected NEUTRAL. Only how many h5py handles stay open.
    persistent_workers expected NEUTRAL IN EPOCH 0, DIFFERENT AFTER. PyTorch seeds each
                       worker's numpy RNG from a base_seed drawn when the ITERATOR is
                       created, so respawned workers get a fresh seed every epoch while
                       persistent workers carry one stream forward. Neither degenerates;
                       the streams are simply not the same sequence. That is why it is a
                       config flag rather than a hardcoded True.

Epochs are consumed IN FULL. Comparing a truncated prefix instead makes prefetch_factor
look like it changes the data: stopping early leaves workers having speculatively built a
different number of extra samples, so their RNGs sit at different offsets when the next
epoch starts. That is an artefact of the measurement, not of the setting, and --limit
keeps a full epoch cheap enough to avoid needing the shortcut.

Usage:
    python scripts/verify_dataloader_settings.py --profile veg_t1 --workers 2
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


def epochs(ds, *, workers, prefetch, persistent, batch_size, n_epochs, seed):
    """Return a digest of every batch, per epoch, under a pinned seed.

    Shuffling is driven by an explicit torch.Generator so sample ORDER is identical
    across configurations and any difference that shows up is augmentation, not ordering.
    Every epoch is consumed to exhaustion -- see the module docstring for why a prefix
    would give a false positive on prefetch_factor.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = dict(num_workers=workers, shuffle=True, generator=g,
                  batch_size=batch_size, drop_last=True)
    if workers > 0:
        kwargs.update(prefetch_factor=prefetch, persistent_workers=persistent)
    dl = DataLoader(ds, **kwargs)

    out = []
    for _ in range(n_epochs):
        digests = []
        for batch in dl:
            x, y = batch[0], batch[1]
            digests.append((float(x.double().sum()), float(y.double().sum()),
                            float(x.double().square().sum())))
        out.append(digests)
    del dl
    return out


def report(name, a, b, expect_same: bool) -> bool:
    same = a == b
    verdict = "identical" if same else "DIFFERENT"
    ok = same == expect_same
    flag = "ok  " if ok else "FAIL"
    print(f"  {flag} {name:<46} {verdict}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=96,
                    help="samples per epoch; the epoch is run to exhaustion, so this "
                         "bounds runtime without truncating it")
    ap.add_argument("--seed", type=int, default=0)
    add_dataset_args(ap)
    args = ap.parse_args()

    if args.workers < 1:
        print("--workers must be >= 1: with 0 workers there is no worker RNG stream and "
              "none of these settings exist.")
        return 2

    full = build_dataset(args, is_train=True)
    # Evenly spaced rather than the first N: consecutive dataset indices are consecutive
    # days of ONE fire, so a head slice would test a single fire's imagery.
    picks = [int(i) for i in np.linspace(0, len(full) - 1, min(args.limit, len(full)))]
    ds = Subset(full, picks)
    print(f"profile={args.profile}  workers={args.workers}  "
          f"batch_size={args.batch_size}  samples/epoch={len(ds)} (full epochs)\n")

    common = dict(workers=args.workers, batch_size=args.batch_size, seed=args.seed)

    # --- prefetch_factor: must not change anything, in any epoch ------------------
    pf2 = epochs(ds, prefetch=2, persistent=True, n_epochs=2, **common)
    pf4 = epochs(ds, prefetch=4, persistent=True, n_epochs=2, **common)
    ok = report("prefetch_factor 2 vs 4 (epoch 0)", pf2[0], pf4[0], expect_same=True)
    ok &= report("prefetch_factor 2 vs 4 (epoch 1)", pf2[1], pf4[1], expect_same=True)
    ok &= report("prefetch_factor 2 vs 4 (both epochs)", pf2, pf4, expect_same=True)

    # --- hdf5_cache_size: must not change anything -------------------------------
    small_base = build_dataset(args, is_train=True)
    small_base._hdf5_cache_max = 1
    ds_small = Subset(small_base, picks)
    small = epochs(ds_small, prefetch=2, persistent=True, n_epochs=1, **common)
    ok &= report("hdf5_cache_size 64 vs 1 (epoch 0)", pf2[0], small[0], expect_same=True)

    # --- persistent_workers: neutral in epoch 0, expected to diverge after --------
    nonp = epochs(ds, prefetch=2, persistent=False, n_epochs=2, **common)
    ok &= report("persistent_workers True vs False (epoch 0)", pf2[0], nonp[0],
                 expect_same=True)

    # The documented behaviour difference. Not asserted as pass/fail -- it is a
    # property of PyTorch's worker seeding, not of this repo's code -- but it IS
    # printed, because a run that silently loses cross-epoch augmentation diversity
    # is exactly the kind of thing that gets discovered far too late.
    print()
    print(f"  note persistent=False epoch0 vs epoch1 replays the same augmentation "
          f"stream: {nonp[0] == nonp[1]}")
    print(f"  note persistent=True  epoch0 vs epoch1 replays the same augmentation "
          f"stream: {pf2[0] == pf2[1]}")

    print()
    if ok:
        print("PASS - prefetch_factor and hdf5_cache_size are output-neutral, and "
              "persistent_workers is neutral for epoch 0.")
        return 0
    print("FAIL - a setting expected to be output-neutral changed the batches. "
          "Do not train on this.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
