#!/usr/bin/env python
"""Sample where a live training run's wall-clock is actually going.

Answers one question: is the run GPU-bound, CPU-bound, disk-bound, or
memory-pressure-bound? Those have opposite fixes, so guessing wastes days.

Read the output like this:
  GPU util high (>80%)                -> compute-bound. Only a faster GPU helps.
  GPU low + workers pegged near 100%  -> CPU-bound in the dataloader. More workers
                                         (up to core count) or cheaper transforms.
  GPU low + workers idle + disk high  -> I/O-bound. Storage layout / read volume.
  GPU low + swap si/so nonzero        -> memory-pressure-bound. Raise the RAM cap
                                         or lower num_workers * prefetch_factor.

Usage:
    python scripts/profile_training_bottleneck.py            # 20s sample
    python scripts/profile_training_bottleneck.py --secs 60
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

PROC_PATTERN = "train.py"

# Sustained read rate this machine's NVMe actually demonstrated under the pre-optimisation
# dataloader (2026-08-18, A0 fold 8 train phase). Used as the denominator for "is the disk
# actually the wall?" -- an absolute MB/s threshold is meaningless without it. 437 MB/s looks
# large in isolation but is only ~12% of what this device has been seen to deliver.
DEVICE_READ_CEILING_MB = 3770.0


def read_vmstat() -> dict:
    return {
        k: int(v)
        for k, v in (
            line.split() for line in Path("/proc/vmstat").read_text().splitlines() if " " in line
        )
    }


def read_diskstats() -> tuple[int, int]:
    """Sum sectors read/written across real block devices (skip loop/ram)."""
    r = w = 0
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) < 10:
            continue
        name = f[2]
        if name.startswith(("loop", "ram")) or re.search(r"\d$", name):
            continue  # skip partitions, count whole devices only
        r += int(f[5])
        w += int(f[9])
    return r, w


def train_pids() -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    return [
        int(line.split()[0])
        for line in out.splitlines()
        if PROC_PATTERN in line and "grep" not in line and line.split()[0].isdigit()
    ]


def proc_cpu_jiffies(pid: int) -> int | None:
    """utime+stime from /proc/<pid>/stat, fields 14 and 15 (1-indexed)."""
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(parts[11]) + int(parts[12])
    except (OSError, IndexError, ValueError):
        return None


def proc_rss_gb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def gpu_sample() -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=20, help="sampling window in seconds")
    args = ap.parse_args()

    pids = train_pids()
    if not pids:
        raise SystemExit("No train.py processes found - is a run active?")
    print(f"Sampling {len(pids)} train.py process(es) for {args.secs}s ...\n")

    hz = 100  # kernel USER_HZ; jiffies -> seconds
    v0, (dr0, dw0) = read_vmstat(), read_diskstats()
    c0 = {p: proc_cpu_jiffies(p) for p in pids}
    g0 = gpu_sample()
    t0 = time.time()

    time.sleep(args.secs)

    elapsed = time.time() - t0
    v1, (dr1, dw1) = read_vmstat(), read_diskstats()
    g1 = gpu_sample()

    print("=== GPU ===")
    print("  (util%, memutil%, W, MiB)")
    print(f"  start: {g0}")
    print(f"  end  : {g1}")

    print("\n=== dataloader worker CPU (100% = one core fully busy) ===")
    total_cpu = 0.0
    for p in sorted(pids):
        j1 = proc_cpu_jiffies(p)
        if c0.get(p) is None or j1 is None:
            continue
        pct = (j1 - c0[p]) / hz / elapsed * 100
        total_cpu += pct
        role = "MAIN " if p == min(pids) else "worker"
        print(f"  {role} pid={p:<8} cpu={pct:6.1f}%   rss={proc_rss_gb(p):.2f} GB")
    print(f"  ---- aggregate cpu across train procs: {total_cpu:.0f}%  (of {len(pids) * 100}% possible)")

    print("\n=== disk (whole devices) ===")
    rd_mb = (dr1 - dr0) * 512 / 1048576 / elapsed
    wr_mb = (dw1 - dw0) * 512 / 1048576 / elapsed
    print(f"  read : {rd_mb:8.1f} MB/s")
    print(f"  write: {wr_mb:8.1f} MB/s")

    print("\n=== memory pressure ===")
    swin = (v1["pswpin"] - v0["pswpin"]) * 4096 / 1048576 / elapsed
    swout = (v1["pswpout"] - v0["pswpout"]) * 4096 / 1048576 / elapsed
    majf = (v1["pgmajfault"] - v0["pgmajfault"]) / elapsed
    print(f"  swap in : {swin:8.2f} MB/s")
    print(f"  swap out: {swout:8.2f} MB/s")
    print(f"  major faults: {majf:.0f}/s   (each one blocks a worker on disk)")

    print("\n=== verdict hint ===")
    gpu_util = int(g1.split(",")[0]) if g1 != "n/a" and g1.split(",")[0].strip().isdigit() else -1
    swap_mb = swin + swout
    # Order matters: compare magnitudes, don't trip on the first threshold that fires.
    # A few MB/s of swap next to GB/s of disk reads is noise, not the bottleneck.
    if gpu_util >= 80:
        print("  GPU-bound: the GPU is the limit. Dataloader tuning will not help.")
    elif rd_mb > DEVICE_READ_CEILING_MB * 0.6 and rd_mb > swap_mb * 10:
        print(f"  I/O-BOUND: {rd_mb:.0f} MB/s sustained reads, "
              f"{100 * rd_mb / DEVICE_READ_CEILING_MB:.0f}% of this device's measured ceiling.")
        print("  The lever is READ VOLUME PER SAMPLE, not RAM and not worker count.")
        print("  Check for read amplification: are you reading whole arrays off disk and")
        print("  then discarding most of it in Python (channel selection, cropping)?")
    elif swap_mb > 5:
        print("  MEMORY-PRESSURE-bound: swap traffic is stealing time. Raise the WSL RAM")
        print("  cap (.wslconfig) or lower num_workers * prefetch_factor.")
    elif total_cpu > 80 * max(1, len(pids) - 1):
        print("  CPU-bound in the dataloader: workers are saturated. More workers (up to")
        print("  core count) or cheaper per-sample transforms.")
    else:
        print("  NOTHING SATURATED -- this is a latency/serialisation regime, not a")
        print("  throughput wall. Read it as: no single resource is the wall, so no single")
        print("  knob (workers, prefetch, RAM) buys much. Look instead at per-sample work")
        print("  that is done and then discarded, and at per-__getitem__ latency.")
        print(f"    GPU {gpu_util}% | workers ~{total_cpu / max(1, len(pids) - 1):.0f}% each"
              f" | disk {rd_mb:.0f} MB/s ({100 * rd_mb / DEVICE_READ_CEILING_MB:.0f}% of ceiling)"
              f" | swap {swap_mb:.1f} MB/s")
        print("  If this was the TEST phase, re-run during TRAIN: they differ a lot.")


if __name__ == "__main__":
    main()
