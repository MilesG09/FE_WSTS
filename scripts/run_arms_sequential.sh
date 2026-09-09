#!/bin/bash
# Run several arms back-to-back on the same fold subset, one after another, by delegating each to
# the already-validated scripts/run_fold_subset_cv.sh. Nothing new about the training path -- this
# only removes the need to sit up and launch the second arm by hand once the first finishes.
#
# Every scientific parameter still comes from the committed configs (cfgs/data_base.yaml +
# cfgs/arms/<ARM>.yaml); the CLI carries only run identity, the fold list, and machine-local
# dataloader tuning, same convention as run_12fold_cv.sh and run_fold_subset_cv.sh.
#
# A failing arm does NOT stop the ones after it: an overnight job that dies on arm 1 at 2 a.m.
# should still deliver arm 2 by morning. Per-arm exit status is echoed and summarised at the end.
#
# Usage: run_arms_sequential.sh <arm_list> <fold_list> <num_workers> <prefetch_factor> <lr> <run_suffix> [seed] [run_name_tag]
# Example (A1 then A2, pre-committed 4-fold subset, matching A0's settings):
#   run_arms_sequential.sh "A1 A2" "1 5 6 11" 8 3 1e-3 bs64
#
# run_name_tag (optional 8th arg) is appended to every run NAME in this invocation (not the
# wandb group/tags). Since it hits every arm in <arm_list>, launch a re-run you want tagged --
# e.g. A0 on a second machine -- as its own invocation, separate from the untagged arms.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

ARMS=${1:?arm list required, e.g. "A1 A2"}
FOLDS=${2:?fold list required, e.g. "1 5 6 11"}
NW=${3:?num_workers required}
PF=${4:?prefetch_factor required}
LR=${5:?lr required}
SUFFIX=${6:?run_suffix required, e.g. bs64}
SEED=${7:-0}
RUN_TAG=${8:-}

# Fail fast on a typo'd arm name rather than 4.5 hours into the night.
for arm in $ARMS; do
    if [ ! -f "cfgs/arms/${arm}.yaml" ]; then
        echo "ERROR: cfgs/arms/${arm}.yaml does not exist" >&2
        exit 1
    fi
done

echo "=== [$(date)] Sequential arm run: [$ARMS] folds=[$FOLDS] nw=$NW pf=$PF lr=$LR seed=$SEED run_tag=${RUN_TAG:-<none>} ==="

declare -a RESULTS=()
for arm in $ARMS; do
    echo "=== [$(date)] === ARM $arm starting ==="
    bash scripts/run_fold_subset_cv.sh "$arm" "$arm" none "$FOLDS" "$NW" "$PF" "$LR" "$SUFFIX" "$SEED" "$RUN_TAG"
    code=$?
    echo "=== [$(date)] === ARM $arm finished (exit $code) ==="
    RESULTS+=("$arm:$code")
done

echo "=== [$(date)] Sequential arm run complete: ${RESULTS[*]} ==="
