#!/usr/bin/env python
"""Write a post-hoc `test_AP` (and friends) onto the wandb run where each fold was TRAINED.

Why this exists
---------------
The 2026-08/09 veg-row2 sweeps split training and testing into two processes: a
`--do_train` phase, then a separate `--do_test` sweep from the checkpoints. The test
phase logged to its own throwaway wandb run (or to none at all), so the fold's real
training run -- the one with the loss/val curves -- had no `test_AP` in its summary.
`src/train.py` now runs test in the same process (determinism is dropped just before
`trainer.test`, which is what used to force the split), so NEW runs get `test_AP`
natively and do not need this script. It stays for the already-fragmented groups.

What it does
------------
For each fold in a results TSV, finds the matching *training* run in a wandb group by
name (`...fold<N>`), and sets `summary["test_AP"] = <value>`. Dry-run unless --apply.
A run that already carries the same value (within 1e-6) is left alone, so re-running is
safe.

Results TSV format: any lines containing both `fold<N>` and `test_AP=<float>`, e.g.
    2026-09-03T12:20:00-07:00\tfold0\ttest_AP=0.5499971508979797
(this is exactly what run_remaining.sh / run_sweep.sh write).

Usage
-----
    export WANDB_API_KEY=...            # see memory `wandb-new-key-old-client`
    python scripts/attach_test_ap.py \
        --group veg_row2_t1_checkval1 \
        --results logs/veg_row2_wandb/detrun/results_final.tsv          # dry run
    python scripts/attach_test_ap.py --group ... --results ... --apply  # write
"""
import argparse
import re
import sys

import wandb

DEFAULT_PROJECT = "milesgoodman09-viewpoint-school/FE_WSTS"
FOLD_RE = re.compile(r"fold(\d+)\b")
AP_RE = re.compile(r"test_AP=([0-9.]+)")
# summary keys that only a real training run carries -- used to tell a training run
# apart from a bare `--do_test` run that happens to share the fold in its name.
TRAIN_MARKERS = ("val_avg_precision.max", "val_loss.min", "trainer/global_step", "epoch")


def load_results(path):
    """-> {fold_int: test_AP_float}. Last value wins if a fold appears twice."""
    out = {}
    for line in open(path):
        mf, ma = FOLD_RE.search(line), AP_RE.search(line)
        if mf and ma:
            out[int(mf.group(1))] = float(ma.group(1))
    if not out:
        sys.exit(f"no `fold<N> ... test_AP=<float>` lines found in {path}")
    return out


def is_training_run(run):
    # A real, completed training run: finished, carries training-only summary keys,
    # and got most of the way through max_steps (a crashed 3.9k-step run is not it).
    if run.state != "finished":
        return False
    if not any(k in run.summary for k in TRAIN_MARKERS):
        return False
    return run.summary.get("trainer/global_step", 0) >= 9000


def fold_of(name):
    m = FOLD_RE.search(name or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--key", default="test_AP", help="summary key to write (default: test_AP)")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    results = load_results(args.results)
    api = wandb.Api()
    runs = list(api.runs(args.project, filters={"group": args.group}))
    if not runs:
        sys.exit(f"no runs in group {args.group!r} under {args.project}")

    by_fold = {}
    for r in runs:
        by_fold.setdefault(fold_of(r.name), []).append(r)

    print(f"group {args.group!r}: {len(runs)} runs, {len(results)} folds in results\n")
    print(f"{'fold':<6}{'target':>10}  action")
    print("-" * 60)

    wrote = skipped = missing = 0
    for fold in sorted(results):
        target = results[fold]
        cands = [r for r in by_fold.get(fold, []) if is_training_run(r)]
        cands.sort(key=lambda r: (r.state == "finished", r.summary.get("trainer/global_step", 0)), reverse=True)

        if not cands:
            orphan = [r for r in runs if abs(float(r.summary.get(args.key, "nan") or "nan") - target) < 1e-6]
            note = f"orphan test-only run(s) {[o.id for o in orphan]}" if orphan else "no run"
            print(f"{fold:<6}{target:>10.4f}  -- NO TRAINING RUN ({note})")
            missing += 1
            continue

        run = cands[0]
        extra = f"  (+{len(cands)-1} other match)" if len(cands) > 1 else ""
        cur = run.summary.get(args.key)
        if cur is not None and abs(float(cur) - target) < 1e-6:
            print(f"{fold:<6}{target:>10.4f}  ok already ({run.name} {run.id}){extra}")
            skipped += 1
            continue

        verb = "SET" if cur is None else f"OVERWRITE {float(cur):.4f}->"
        print(f"{fold:<6}{target:>10.4f}  {verb} on {run.name} {run.id}{extra}")
        if args.apply:
            run.summary[args.key] = target
            run.summary["test_AP_source"] = f"post-hoc via attach_test_ap.py from {args.results}"
            run.summary.update()
            wrote += 1

    print("-" * 60)
    if args.apply:
        print(f"wrote {wrote}, already-ok {skipped}, no-training-run {missing}")
    else:
        print(f"DRY RUN -- {len(results)} folds inspected; re-run with --apply to write")
        print(f"(already-ok {skipped}, no-training-run {missing})")


if __name__ == "__main__":
    main()
