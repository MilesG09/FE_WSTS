#!/usr/bin/env python
"""Measure real DataLoader throughput vs num_workers, under cold and warm page cache.

Settles two questions that per-sample micro-benchmarks cannot:

1. Does page-cache residency actually explain the gap between ~14 ms/sample measured
   warm and ~91 ms/sample observed during a live sweep? Run the same config cold and
   warm; the ratio is the answer.

2. Does adding workers still help? Worker count and page cache COMPETE for the same
   RAM (each persistent worker holds ~0.5 GB RSS), so more workers can buy parallel
   I/O waits while simultaneously evicting the cache that would have avoided them.
   That interaction cannot be reasoned out; it has to be measured together.

Cold state is produced with posix_fadvise(DONTNEED) over the HDF5 files, which works
unprivileged (unlike /proc/sys/vm/drop_caches) and evicts only this dataset rather
than the whole machine's cache.

Reports batches/s, which is what training actually consumes.

Usage:
    python scripts/benchmark_dataloader_throughput.py --profile veg_t1 --workers 4,8,12,16
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import (  # noqa: E402
    add_dataset_args, build_dataset, describe, profile_config)


def build(args):
    ds = build_dataset(args, is_train=True)
    return ds, int(profile_config(args.profile)["batch_size"]), Path(args.data_dir)


def evict(data_dir: Path, years: list[int]) -> float:
    """Drop this dataset's files from the page cache. Returns GB evicted (file bytes)."""
    total = 0
    for y in years:
        for p in (data_dir / str(y)).glob("*.hdf5"):
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                total += p.stat().st_size
            finally:
                os.close(fd)
    return total / 1024**3


def cache_state() -> str:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("Cached", "MemAvailable", "MemFree"):
            out[k] = int(v.split()[0]) / 1024**2  # GB
    return "  ".join(f"{k}={v:.1f}G" for k, v in out.items())


def run_grid(ds, batch_size: int, workers: int, prefetch: int, batches: int) -> float:
    """Batches/s for one (num_workers, prefetch_factor) cell.

    num_workers and prefetch_factor are NOT independent -- their product sets how many
    batches are in flight -- so they must be searched jointly, not as two 1-D sweeps.
    """
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        prefetch_factor=prefetch if workers > 0 else None,
        persistent_workers=False, drop_last=True,
    )
    it = iter(dl)
    next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        try:
            next(it)
        except StopIteration:
            break
        n += 1
    dt = time.perf_counter() - t0
    del it, dl
    return n / dt if dt > 0 else 0.0


def run(ds, batch_size: int, workers: int, batches: int, shuffle: bool = False) -> float:
    """Batches per second over `batches` batches, excluding worker startup.

    shuffle defaults to False so the cold and warm passes read the SAME samples.
    With shuffle=True the warm pass draws a different random subset, so it is not
    actually warm for what it reads and the cold/warm ratio measures nothing.
    """
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=False, drop_last=True,
    )
    it = iter(dl)
    next(it)  # pay worker spin-up and first fill outside the timed region
    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        try:
            next(it)
        except StopIteration:
            break
        n += 1
    dt = time.perf_counter() - t0
    del it, dl
    return n / dt if dt > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="4,8,12,16")
    ap.add_argument("--batches", type=int, default=25)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prefetch", default="",
                    help="comma-separated prefetch_factors; enables JOINT grid search "
                         "over num_workers x prefetch_factor (cold/warm split is skipped, "
                         "since residency was measured to be worth ~0)")
    add_dataset_args(ap)
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")]
    worker_counts = [int(w) for w in args.workers.split(",")]

    ds, batch_size, data_dir = build(args)
    print(f"[{args.profile}] {describe(ds)}")
    print(f"years={years}  batch_size={batch_size}  "
          f"len={len(ds)}  batches/measurement={args.batches}")
    print(f"mem before: {cache_state()}\n")

    if args.prefetch:
        prefetches = [int(p) for p in args.prefetch.split(",")]
        # One warm-up so no cell pays first-touch cost the others avoid.
        run_grid(ds, batch_size, worker_counts[0], prefetches[0], 5)
        header = "  nw\\pf ".rjust(8) + "".join(f"{p:>10}" for p in prefetches)
        print(header)
        print("-" * len(header))
        # Repeat every cell and take the median. A single 30-batch measurement swings by
        # ~25% here (nw=8 measured 2.63 then 3.35 b/s on consecutive runs), which is larger
        # than any difference between cells -- so one sample per cell measures noise, not
        # configuration. Interleaving trials also stops slow drift being attributed to
        # whichever cell happened to run during it.
        results: dict[tuple[int, int], float] = {}
        raw: dict[tuple[int, int], list[float]] = {(w, p): [] for w in worker_counts
                                                   for p in prefetches}
        for _ in range(args.trials):
            for w in worker_counts:
                for p in prefetches:
                    raw[(w, p)].append(run_grid(ds, batch_size, w, p, args.batches))
        for w in worker_counts:
            row = f"{w:>8}"
            for p in prefetches:
                results[(w, p)] = statistics.median(raw[(w, p)])
                row += f"{results[(w, p)]:>10.2f}"
            print(row)

        spread = max(max(v) - min(v) for v in raw.values())
        print(f"\n  worst within-cell spread across trials: {spread:.2f} b/s")
        best = max(results, key=results.get)
        base = results.get((8, 2))
        print(f"\n  best: num_workers={best[0]}, prefetch_factor={best[1]}  "
              f"-> {results[best]:.2f} b/s")
        if base:
            print(f"  vs current (nw=8, pf=2) {base:.2f} b/s  =  {results[best] / base:.2f}x")
        print("\n  NOTE: this is dataloader capacity with NO consumer -- no GPU work, no")
        print("  backpressure. It is a SCREEN. Confirm finalists with real training runs")
        print("  (scripts/training_parameter_search.sh) before trusting the wall-clock gain.")
        return

    print(f"{'workers':>8} {'COLD b/s':>10} {'WARM b/s':>10} {'warm/cold':>10}   mem after warm")
    print("-" * 74)

    for w in worker_counts:
        gb = evict(data_dir, years)
        cold = run(ds, batch_size, w, args.batches)
        # Second pass with the cache now populated by the cold pass.
        warm = run(ds, batch_size, w, args.batches)
        ratio = warm / cold if cold > 0 else float("nan")
        print(f"{w:>8} {cold:>10.2f} {warm:>10.2f} {ratio:>10.2f}x   {cache_state()}")
        print(f"{'':>8} (evicted {gb:.0f} GB before cold pass)")

    print("\nRead it as:")
    print("  warm/cold ratio  -> how much page-cache residency is worth")
    print("  b/s vs workers   -> whether added concurrency still buys throughput")
    print("  if warm b/s flattens with more workers, CPU/GPU is the wall, not I/O")


if __name__ == "__main__":
    main()
