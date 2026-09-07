"""Paired A0-vs-C0 diagnostic over the bs64 12-fold runs.

Two questions this answers that `find_runs.py`'s arm means cannot:

  1. Is the A0 > C0 ordering a real effect, or one fold? Arm means are UNPAIRED --
     they carry the huge between-year variance (~0.09 AP) into a comparison whose
     true effect size is ~0.02. The correct statistic for a same-fold, same-seed,
     one-variable-changed design is the per-fold difference d_f = C0_f - A0_f;
     between-year variance cancels exactly. (EXPERIMENTS.md, "Reporting standard".)

  2. Was any run undertrained or unstable? Reads each run's val_avg_precision
     history and reports peak / step-at-peak / final. If peaks sit near the last
     step, max_steps=10000 is the binding constraint, not the arm.

Usage:  python scripts/diagnose_arm_gap.py [--arms A0 C0] [--suffix _seed0_bs64]
"""
import argparse
import math
import os
import re
from statistics import mean, stdev

MLRUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")


def read_tag(run_dir, tag):
    p = os.path.join(run_dir, "tags", tag)
    return open(p).read().strip() if os.path.exists(p) else None


def read_metric(run_dir, key):
    """MLflow file store: one '<timestamp> <value> <step>' line per logged point."""
    p = os.path.join(run_dir, "metrics", key)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        parts = line.split()
        if len(parts) >= 3:
            out.append((int(parts[2]), float(parts[1])))
    return out


def collect(suffix):
    """-> {run_name: run_dir} for every run whose name ends in `suffix`."""
    runs = {}
    for exp in os.listdir(MLRUNS):
        exp_dir = os.path.join(MLRUNS, exp)
        if not os.path.isdir(exp_dir) or exp == "models":
            continue
        for rid in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, rid)
            name = read_tag(run_dir, "mlflow.runName")
            if not (name and name.endswith(suffix)):
                continue
            # A run NAME is not unique in the file store: a crashed run and its re-run share one.
            # Always prefer the copy that actually has a logged test_AP, or a fold that was
            # successfully re-run gets silently dropped because its crashed sibling happened to be
            # enumerated first. (Cost of not doing this, 2026-08-28: A1 fold 0 vanished from
            # comparison 1 and the paired test quietly ran on 11 folds instead of 12.)
            if name in runs and not read_metric(run_dir, "test_AP"):
                continue
            runs[name] = run_dir
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs=2, default=["A0", "C0"])
    ap.add_argument("--suffix", default="_seed0_bs64")
    args = ap.parse_args()
    a_arm, b_arm = args.arms

    runs = collect(args.suffix)
    fold_re = re.compile(r"^(\w+?)_fold(\d+)")

    ap_by = {a_arm: {}, b_arm: {}}
    dirs_by = {a_arm: {}, b_arm: {}}
    for name, d in runs.items():
        m = fold_re.match(name)
        if not m or m.group(1) not in ap_by:
            continue
        arm, fold = m.group(1), int(m.group(2))
        test = read_metric(d, "test_AP")
        if test:
            ap_by[arm][fold] = test[-1][1]
            dirs_by[arm][fold] = d

    folds = sorted(set(ap_by[a_arm]) & set(ap_by[b_arm]))
    if not folds:
        raise SystemExit(f"no folds shared by {a_arm} and {b_arm} with suffix {args.suffix}")

    print(f"\n=== Paired per-fold test_AP  ({b_arm} - {a_arm}), suffix '{args.suffix}' ===")
    print(f"{'fold':>4} {a_arm:>8} {b_arm:>8} {'delta':>9}")
    deltas = []
    for f in folds:
        d = ap_by[b_arm][f] - ap_by[a_arm][f]
        deltas.append(d)
        flag = "   <-- outlier" if abs(d) > 0.10 else ""
        print(f"{f:>4} {ap_by[a_arm][f]:>8.4f} {ap_by[b_arm][f]:>8.4f} {d:>+9.4f}{flag}")

    n = len(deltas)
    md, sd = mean(deltas), stdev(deltas)
    se = sd / math.sqrt(n)
    print(f"\n{a_arm} mean {mean(ap_by[a_arm][f] for f in folds):.4f}   "
          f"{b_arm} mean {mean(ap_by[b_arm][f] for f in folds):.4f}")
    print(f"paired mean delta = {md:+.4f}   sd(d) = {sd:.4f}   SE = {se:.4f}   "
          f"t({n - 1}) = {md / se:+.2f}")

    print("\nleave-one-fold-out mean delta (how load-bearing is any single fold):")
    loo = [(f, mean([d for g, d in zip(folds, deltas) if g != f])) for f in folds]
    for f, m in sorted(loo, key=lambda x: abs(x[1] - md), reverse=True)[:3]:
        print(f"  drop fold {f:>2}: {m:+.4f}   (shift {m - md:+.4f})")

    print("\n=== val_avg_precision trajectory (checkpoint-selection metric) ===")
    print(f"{'arm':>4} {'fold':>4} {'peak':>7} {'@step':>7} {'final':>7} {'peak/last':>10}")
    for arm in (a_arm, b_arm):
        for f in folds:
            hist = read_metric(dirs_by[arm][f], "val_avg_precision")
            if not hist:
                continue
            best_step, best = max(hist, key=lambda sv: sv[1])
            last_step, last = hist[-1]
            print(f"{arm:>4} {f:>4} {best:>7.4f} {best_step:>7} {last:>7.4f} "
                  f"{best_step / last_step:>10.2f}")


if __name__ == "__main__":
    main()
