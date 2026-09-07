#!/usr/bin/env python
"""Find where the hyperslab partial read stops paying for itself.

The partial read in load_imgs trades I/O for CPU: h5py builds a channel selection, and
the result is then scattered into a full-width zero array so downstream fixed channel
indices keep working. When the requested channels are a small fraction of the file that
is a large win. When they are most of the file it is pure overhead -- the HDF5 files here
are CONTIGUOUS and UNCHUNKED, so a scattered selection covering most of the channel range
still drags the same extent past the block layer, and the zero-fill is added on top.

Measured directly rather than assumed, on the real files, at the real timestep count:
bytes actually read from the block layer (/proc/self/io, page-cache hits excluded) and
wall time, for channel subsets of increasing density. The crossover this prints is the
evidence behind FireSpreadDataset.HDF5_READ_OPT_MAX_FRACTION.

Usage:
    python scripts/benchmark_hyperslab_density.py --profile veg_t1 --files 3
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import add_dataset_args, profile_config  # noqa: E402

from src.dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402


def proc_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def evict(path: Path) -> None:
    """posix_fadvise(DONTNEED) -- unprivileged, and evicts only this file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def spread(k: int, n: int) -> list:
    """k channels spread evenly over 0..n-1, always including the last (the AF mask).

    Evenly spread, not a prefix: a prefix is the best case for contiguous storage and
    would flatter the optimisation. Real feature sets are scattered.
    """
    if k >= n:
        return list(range(n))
    picks = {n - 1}
    picks.update(int(round(i * (n - 2) / max(1, k - 2))) for i in range(max(1, k - 1)))
    out = sorted(picks)
    return out[:k] if len(out) > k else out


def measure(path: Path, n_days: int, reads: int, channels, n_raw: int) -> tuple:
    """Bytes/sample off disk and ms/sample for one strategy, from a cold cache."""
    evict(path)
    b0 = proc_read_bytes()
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        d = f["data"]
        n = d.shape[0]
        for k in range(reads):
            i = (k * 3) % max(1, n - n_days - 1)
            j = i + n_days + 1
            if channels is None:
                d[i:j]
            else:
                sub = d[i:j, channels]
                # The scatter is part of the cost of this strategy, so it is timed.
                imgs = np.zeros((sub.shape[0], n_raw) + sub.shape[2:], dtype=sub.dtype)
                imgs[:, channels] = sub
    dt = (time.perf_counter() - t0) * 1000 / reads
    mb = (proc_read_bytes() - b0) / 1048576 / reads
    return mb, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=int, default=10)
    ap.add_argument("--files", type=int, default=3)
    ap.add_argument("--trials", type=int, default=3)
    add_dataset_args(ap)
    args = ap.parse_args()

    cfg = profile_config(args.profile)
    n_days = int(cfg["n_leading_observations"])
    n_raw = FireSpreadDataset.N_RAW_HDF5_CHANNELS
    files = sorted(Path(args.data_dir).glob("*/*.hdf5"))[: args.files]
    if not files:
        raise SystemExit(f"No HDF5 files under {args.data_dir}")

    print(f"profile={args.profile}  timesteps read per sample={n_days + 1}  "
          f"files={len(files)}  reads/measurement={args.reads}\n")
    print(f"  {'channels':>9}  {'frac':>5}  {'MB/sample':>10}  {'ms/sample':>10}  "
          f"{'vs full read':>13}")
    print("  " + "-" * 56)

    def run(channels):
        mbs, mss = [], []
        for _ in range(args.trials):
            for path in files:
                mb, ms = measure(path, n_days, args.reads, channels, n_raw)
                mbs.append(mb)
                mss.append(ms)
        return statistics.median(mbs), statistics.median(mss)

    base_mb, base_ms = run(None)
    print(f"  {n_raw:>9}  {1.0:>5.2f}  {base_mb:>10.2f}  {base_ms:>10.2f}  "
          f"{'(baseline)':>13}")

    crossover = None
    for k in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20):
        chans = spread(k, n_raw)
        mb, ms = run(chans)
        faster = ms < base_ms
        if crossover is None and not faster:
            crossover = k / n_raw
        print(f"  {len(chans):>9}  {len(chans) / n_raw:>5.2f}  {mb:>10.2f}  {ms:>10.2f}  "
              f"{base_ms / ms:>12.2f}x{'' if faster else '  <- SLOWER'}")

    print()
    if crossover is None:
        print("  The partial read was faster at every density tested.")
    else:
        print(f"  Crossover: the partial read stops paying off at about "
              f"{crossover:.2f} of the channels.")
    print(f"  Current guard: HDF5_READ_OPT_MAX_FRACTION = "
          f"{FireSpreadDataset.HDF5_READ_OPT_MAX_FRACTION}")


if __name__ == "__main__":
    main()
