#!/bin/bash
# Detached launcher for scripts/run_fold_subset_cv.sh -- setsid, not just nohup: when launched
# through `wsl.exe -- bash -lc ...` the WSL session is torn down as soon as wsl.exe returns, and a
# plain background job dies with it (see launch_12fold_cv.sh for the original 2026-08-19 finding).
#
# Usage: launch_fold_subset_cv.sh <arm_yaml> <run_prefix> <extra_data_yaml|none> <fold_list> \
#                                  <num_workers> <prefetch_factor> <lr> <run_suffix>
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

ARM_YAML=${1:?arm yaml name required, e.g. A0}
RUN_PREFIX=${2:?run name prefix required, e.g. A0v}
EXTRA_DATA=${3:?extra data yaml path required, or the literal string 'none'}
FOLDS=${4:?fold list required, e.g. "1 5 6 11"}
NW=${5:?num_workers required}
PF=${6:?prefetch_factor required}
LR=${7:?lr required}
SUFFIX=${8:?run_suffix required, e.g. bs64}

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/cv_${RUN_PREFIX}_${STAMP}.log"

setsid nohup bash scripts/run_fold_subset_cv.sh "$ARM_YAML" "$RUN_PREFIX" "$EXTRA_DATA" "$FOLDS" "$NW" "$PF" "$LR" "$SUFFIX" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 5
if kill -0 "$PID" 2>/dev/null; then
    echo "cv runner pid : $PID"
else
    echo "WARNING: runner died within 5s -- check $LOG" >&2
fi
echo "log           : $LOG"
echo "$PID" > "logs/cv_${RUN_PREFIX}.pid"
echo "$LOG" > "logs/cv_${RUN_PREFIX}.logpath"
