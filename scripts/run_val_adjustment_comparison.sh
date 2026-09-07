#!/bin/bash
# Runs A0v, B0v, C0v back-to-back (sequentially, not concurrently -- single GPU, same reason
# prior arms were never launched in parallel) via run_fold_subset_cv.sh. Each arm matches its
# un-adjusted counterpart's num_workers/prefetch_factor exactly, differing only in
# n_leading_observations_val_adjustment (cfgs/data_val_adjustment.yaml) -- see
# administrative/EXPERIMENTS.md, "Val-adjustment comparison arms", decided 2026-08-22.
#
# Fold subset [1,5,6,11] matches C0's prior 4-fold replication (his call, 2026-08-22), so all
# three arms are paired fold-for-fold and C0v doesn't repay full 12-fold cost at its ~3x
# slower-per-step rate.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

FOLDS="1 5 6 11"
EXTRA=cfgs/data_val_adjustment.yaml
LR=1e-3
SUFFIX=bs64

echo "=== [$(date)] Starting val-adjustment comparison: A0v -> B0v -> C0v, folds=[$FOLDS] ==="

bash scripts/run_fold_subset_cv.sh A0 A0v "$EXTRA" "$FOLDS" 8 3 "$LR" "$SUFFIX"
bash scripts/run_fold_subset_cv.sh B0 B0v "$EXTRA" "$FOLDS" 8 4 "$LR" "$SUFFIX"
bash scripts/run_fold_subset_cv.sh C0 C0v "$EXTRA" "$FOLDS" 8 3 "$LR" "$SUFFIX"

echo "=== [$(date)] Val-adjustment comparison complete: A0v, B0v, C0v ==="
