#!/bin/bash
# Detached launcher for scripts/run_arms_sequential.sh -- setsid, not just nohup: when launched
# through `wsl.exe -- bash -lc ...` the WSL session is torn down as soon as wsl.exe returns and a
# plain background job dies with it (original finding 2026-08-19, see launch_12fold_cv.sh).
#
# Usage: launch_arms_sequential.sh <arm_list> <fold_list> <num_workers> <prefetch_factor> <lr> <run_suffix> [seed]
# Example:
#   bash scripts/launch_arms_sequential.sh "A1 A2" "1 5 6 11" 8 3 1e-3 bs64
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

ARMS=${1:?arm list required, e.g. "A1 A2"}
FOLDS=${2:?fold list required, e.g. "1 5 6 11"}
NW=${3:?num_workers required}
PF=${4:?prefetch_factor required}
LR=${5:?lr required}
SUFFIX=${6:?run_suffix required, e.g. bs64}
SEED=${7:-0}

TAG=$(echo "$ARMS" | tr ' ' '_')
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/arms_${TAG}_${STAMP}.log"

setsid nohup bash scripts/run_arms_sequential.sh "$ARMS" "$FOLDS" "$NW" "$PF" "$LR" "$SUFFIX" "$SEED" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 5
if kill -0 "$PID" 2>/dev/null; then
    echo "runner pid : $PID"
else
    echo "WARNING: runner died within 5s -- check $LOG" >&2
fi
echo "log        : $LOG"
echo "$PID" > "logs/arms_${TAG}.pid"
echo "$LOG" > "logs/arms_${TAG}.logpath"
