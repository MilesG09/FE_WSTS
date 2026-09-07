#!/usr/bin/env python
"""Measure read amplification in the FireSpreadDataset HDF5 access pattern.

Context: profiling A0_fold8_seed0_bs64 during its train phase showed the run is
I/O-bound (3.77 GB/s sustained reads, GPU at 18-29%, workers ~48%). The suspected
cause is FireSpreadDataset.load_imgs, which does

    imgs = f["data"][in_fire_index:end_index]

reading EVERY channel at FULL spatial resolution, then discarding most of it in
Python (crop to crop_side_length, then x[:, features_to_keep]).

This script quantifies that by measuring BYTES ACTUALLY READ FROM DISK per sample
under three strategies. Bytes -- not wall time -- is the headline metric here,
because it is deterministic and unaffected by a training run competing for I/O.
That matters: this is safe to run while a sweep is in flight.

Strategies
  A current    f["data"][i:j]                  -- read all channels, filter after
  B hyperslab  f["data"][i:j, channels]        -- let HDF5 select channels
  C chunked    same as B, on a rechunked+lzf copy of the file

Usage:
    python scripts/benchmark_hdf5_read_strategies.py --profile veg_t1
    python scripts/benchmark_hdf5_read_strategies.py --profile multi_t5 --samples 16 --files 3
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wsts_profiles import add_dataset_args, profile_config  # noqa: E402

# The post-40-channel -> raw-23-channel mapping is NOT duplicated here. It lives in
# FireSpreadDataset.raw_channels_for_features(), which is the same function training
# uses and which scripts/verify_channel_dependency.py checks empirically. A second copy
# in a benchmark is a copy that goes stale and then quietly measures the wrong thing.
from src.dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402


def load_cfg(args) -> tuple:
    """Pull the real training settings so the benchmark matches the actual workload."""
    cfg = profile_config(args.profile)
    feats = cfg.get("features_to_keep")
    channels = FireSpreadDataset.raw_channels_for_features(feats)
    return (
        Path(args.data_dir),
        feats,
        channels,
        int(cfg["crop_side_length"]),
        int(cfg["n_leading_observations"]),
    )


def proc_read_bytes() -> int:
    """Bytes this process actually pulled from the block layer (page-cache hits excluded)."""
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def evict(path: Path) -> None:
    """Drop this file from the page cache so the next read genuinely hits disk.

    posix_fadvise(DONTNEED) works unprivileged, unlike /proc/sys/vm/drop_caches.
    Without this the benchmark would measure memcpy speed, not I/O.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def bench(label: str, path: Path, n_days: int, samples: int, channels: list[int] | None) -> dict:
    evict(path)
    b0, t0 = proc_read_bytes(), time.time()
    total_elems = 0
    with h5py.File(path, "r") as f:
        d = f["data"]
        n = d.shape[0]
        for k in range(samples):
            i = (k * 3) % max(1, n - n_days - 1)
            j = i + n_days + 1
            arr = d[i:j] if channels is None else d[i:j, channels]
            total_elems += int(np.prod(arr.shape))
    dt = time.time() - t0
    rb = proc_read_bytes() - b0
    return {
        "label": label,
        "read_mb": rb / 1048576,
        "per_sample_mb": rb / 1048576 / samples,
        "secs": dt,
        "elems_mb": total_elems * 4 / 1048576 / samples,
    }


def make_chunked_copy(src: Path, dst: Path, crop: int) -> None:
    """Rewrite one file chunked along (time, all-channels, crop, crop) with lzf.

    Chunk shape is the lever: a chunk is the smallest unit HDF5 can read, so aligning
    it to how the data is consumed (one timestep, a crop-sized spatial tile) is what
    makes a partial read actually partial.
    """
    with h5py.File(src, "r") as fs, h5py.File(dst, "w") as fd:
        d = fs["data"]
        t, c, h, w = d.shape
        fd.create_dataset(
            "data",
            data=d[...],
            chunks=(1, c, min(crop, h), min(crop, w)),
            compression="lzf",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12, help="reads per strategy per file")
    ap.add_argument("--files", type=int, default=2, help="how many HDF5 files to test")
    ap.add_argument("--skip-chunked", action="store_true", help="skip strategy C (it copies a file)")
    add_dataset_args(ap)
    args = ap.parse_args()

    data_dir, feats, channels, crop, n_lead = load_cfg(args)
    n_days = n_lead  # load_imgs reads n_leading_observations + 1 timesteps
    print(f"profile={args.profile}  features_to_keep={feats}")
    print(f"        crop={crop}  n_leading_observations={n_lead}"
          f"  -> each __getitem__ reads {n_days + 1} timesteps")
    if channels is None:
        raise SystemExit(
            "This profile keeps every feature, so there is no channel subset to read. "
            "Pick a profile with a features_to_keep list (e.g. veg_t1).")
    print(f"        raw channels needed: {channels} ({len(channels)} of "
          f"{FireSpreadDataset.N_RAW_HDF5_CHANNELS})\n")

    files = sorted(data_dir.glob("*/*.hdf5"))[: args.files]
    if not files:
        raise SystemExit(f"No HDF5 files under {data_dir}")

    agg: dict[str, list[float]] = {}
    for path in files:
        with h5py.File(path, "r") as f:
            shape, dtype = f["data"].shape, f["data"].dtype
        print(f"=== {path.name}  shape={shape} dtype={dtype} ===")

        results = [
            bench(f"A current   (all {FireSpreadDataset.N_RAW_HDF5_CHANNELS} channels)",
                  path, n_days, args.samples, None),
            bench(f"B hyperslab ({len(channels)} raw channels)",
                  path, n_days, args.samples, channels),
        ]

        if not args.skip_chunked:
            tmp = Path(tempfile.gettempdir()) / f"chunked_{path.name}"
            try:
                make_chunked_copy(path, tmp, crop)
                results.append(
                    bench(f"C chunked+lzf ({len(channels)} raw channels)", tmp,
                          n_days, args.samples, channels)
                )
                print(f"    (chunked copy: {tmp.stat().st_size / 1048576:.0f} MB vs "
                      f"{path.stat().st_size / 1048576:.0f} MB original)")
            except Exception as e:  # noqa: BLE001 - benchmark should survive a bad file
                print(f"    strategy C skipped: {e}")
            finally:
                tmp.unlink(missing_ok=True)

        for r in results:
            print(f"  {r['label']:<32} disk={r['per_sample_mb']:7.2f} MB/sample   "
                  f"({r['secs']:.2f}s for {args.samples})")
            agg.setdefault(r["label"], []).append(r["per_sample_mb"])
        print()

    print("=== mean bytes read per __getitem__ ===")
    base = None
    for label, vals in agg.items():
        m = sum(vals) / len(vals)
        if base is None:
            base = m
        print(f"  {label:<32} {m:7.2f} MB/sample   {base / m if m else 0:5.1f}x less I/O than A"
              if base and m else f"  {label:<32} {m:7.2f} MB/sample")

    # What the model actually consumes, for the amplification denominator.
    used = (1 * len(feats) * crop * crop * 4) / 1048576
    print(f"\n  model actually uses ~{used:.2f} MB/sample "
          f"({len(feats)} channels x {crop}x{crop} float32)")
    if base:
        print(f"  => current amplification: {base / used:.0f}x")


if __name__ == "__main__":
    main()
