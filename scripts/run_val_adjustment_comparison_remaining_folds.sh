#!/bin/bash
# Extends the val-adjustment comparison (scripts/run_val_adjustment_comparison.sh, which covers
# folds [1,5,6,11]) to full 12-fold depth for all three arms, by running the complementary 8
# folds. His decision, 2026-08-22: run all three to full depth (including C0v), reversing the
# cost-based 4-fold cap that applied to the original C0 -- see administrative/EXPERIMENTS.md,
# "Val-adjustment comparison arms".
#
# Sequential, not concurrent -- single GPU, same reason as run_val_adjustment_comparison.sh.
# Meant to be chained after that script via scripts/launch_chained_after.sh so it starts
# automatically, unattended, once the [1,5,6,11] batch finishes.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

FOLDS="0 2 3 4 7 8 9 10"
EXTRA=cfgs/data_val_adjustment.yaml
LR=1e-3
SUFFIX=bs64

echo "=== [$(date)] Starting val-adjustment comparison remaining folds: A0v -> B0v -> C0v, folds=[$FOLDS] ==="

bash scripts/run_fold_subset_cv.sh A0 A0v "$EXTRA" "$FOLDS" 8 3 "$LR" "$SUFFIX"
bash scripts/run_fold_subset_cv.sh B0 B0v "$EXTRA" "$FOLDS" 8 4 "$LR" "$SUFFIX"
bash scripts/run_fold_subset_cv.sh C0 C0v "$EXTRA" "$FOLDS" 8 3 "$LR" "$SUFFIX"

echo "=== [$(date)] Val-adjustment comparison remaining folds complete: A0v, B0v, C0v now at full 12-fold ==="
