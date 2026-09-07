"""Compare an arm against its '<arm>v' re-run, fold by fold.

Purpose: measure the pipeline's actual RUN-TO-RUN variance at fixed seed, using runs
that already exist. This is the noise floor every arm delta has to clear, and until now
it has never been measured (EXPERIMENTS.md specced 4 seed runs for it; they were never done).

Prints per fold: both test_APs, their difference, each run's start time and duration, and
whether the two runs' `command` tags are byte-identical -- so a difference can be attributed
to genuine nondeterminism rather than to a silently different invocation.
"""
import argparse
import os
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS = os.path.join(REPO, "mlruns")


def read_tag(run_dir, tag):
    p = os.path.join(run_dir, "tags", tag)
    return open(p).read().strip() if os.path.exists(p) else None


def last_metric(run_dir, key):
    p = os.path.join(run_dir, "metrics", key)
    if not os.path.exists(p):
        return None
    lines = [l.split() for l in open(p) if l.strip()]
    return float(lines[-1][1]) if lines else None


def n_metric_points(run_dir, key):
    p = os.path.join(run_dir, "metrics", key)
    return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0


def meta(run_dir, field):
    p = os.path.join(run_dir, "meta.yaml")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith(field + ":"):
            return line.split(":", 1)[1].strip()
    return None


def index_runs():
    out = {}
    for exp in os.listdir(MLRUNS):
        exp_dir = os.path.join(MLRUNS, exp)
        if not os.path.isdir(exp_dir) or exp == "models":
            continue
        for rid in os.listdir(exp_dir):
            d = os.path.join(exp_dir, rid)
            name = read_tag(d, "mlflow.runName")
            if name:
                out.setdefault(name, d)
    return out


def fmt_time(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%m-%d %H:%M")
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["A0", "B0", "C0"])
    ap.add_argument("--suffix", default="_seed0_bs64")
    args = ap.parse_args()
    runs = index_runs()

    for arm in args.pairs:
        print(f"\n{'=' * 96}\n{arm}  vs  {arm}v   (suffix '{args.suffix}')\n{'=' * 96}")
        print(f"{'fold':>4} {arm:>9} {arm + 'v':>9} {'diff':>9} {'same cmd':>9} "
              f"{'start A':>12} {'start B':>12} {'trainpts A/B':>14}")
        diffs, identical = [], 0
        for fold in range(12):
            a = runs.get(f"{arm}_fold{fold}{args.suffix}")
            b = runs.get(f"{arm}v_fold{fold}{args.suffix}")
            if not a or not b:
                continue
            ap_a, ap_b = last_metric(a, "test_AP"), last_metric(b, "test_AP")
            if ap_a is None or ap_b is None:
                continue
            d = ap_b - ap_a
            diffs.append(d)
            if d == 0.0:
                identical += 1
            same_cmd = read_tag(a, "command") == read_tag(b, "command")
            print(f"{fold:>4} {ap_a:>9.6f} {ap_b:>9.6f} {d:>+9.6f} {str(same_cmd):>9} "
                  f"{fmt_time(meta(a, 'start_time')):>12} {fmt_time(meta(b, 'start_time')):>12} "
                  f"{n_metric_points(a, 'train_loss_step')}/{n_metric_points(b, 'train_loss_step'):>6}")
        if diffs:
            nz = [abs(x) for x in diffs if x != 0.0]
            print(f"\n  bit-identical folds: {identical}/{len(diffs)}")
            if nz:
                print(f"  differing folds:     {len(nz)}   |diff| min {min(nz):.6f}  "
                      f"max {max(nz):.6f}  mean {sum(nz) / len(nz):.6f}")
            else:
                print("  differing folds:     0  -- fully reproducible on this arm")


if __name__ == "__main__":
    main()
