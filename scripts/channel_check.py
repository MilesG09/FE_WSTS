#!/usr/bin/env python
"""Audit the channel dimension of every HDF5 file, per year.

Why: cfgs/data_base.yaml sets features_to_keep=[0,1,2,3,4,38,39], which assumes a
40-channel layout, but a sampled 2020 file turned out to have only 23 channels.
If that is systemic, `x[:, features_to_keep]` in FireSpreadDataset would be
indexing past the end -- so this checks whether the shape is consistent across the
whole dataset, and which files (if any) can actually satisfy index 39.
"""

import collections
from pathlib import Path

import h5py

DATA = Path("/home/miles/FE_WSTS/hdf5_data")
MAX_FEATURE_INDEX = 39  # highest index requested by features_to_keep


def main() -> None:
    total = collections.Counter()
    examples: dict[int, str] = {}
    errors: list[str] = []

    for year in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(year.glob("*.hdf5"))
        per_year: collections.Counter = collections.Counter()
        for p in files:
            try:
                with h5py.File(p, "r") as f:
                    nch = f["data"].shape[1]
                per_year[nch] += 1
                total[nch] += 1
                examples.setdefault(nch, str(p))
            except Exception as e:  # noqa: BLE001
                per_year["ERR"] += 1
                errors.append(f"{p}: {e}")
        print(f"{year.name}: {len(files):>4} files  channels={dict(per_year)}")

    print(f"\nGLOBAL channel histogram: {dict(total)}")
    for nch, path in sorted(examples.items()):
        print(f"  n_channels={nch}  e.g. {path}")

    ok = sum(v for k, v in total.items() if isinstance(k, int) and k > MAX_FEATURE_INDEX)
    bad = sum(v for k, v in total.items() if isinstance(k, int) and k <= MAX_FEATURE_INDEX)
    print(f"\nfiles that can satisfy feature index {MAX_FEATURE_INDEX}: {ok}")
    print(f"files that CANNOT (would IndexError):              {bad}")

    if errors:
        print(f"\n{len(errors)} unreadable file(s); first few:")
        for e in errors[:5]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
