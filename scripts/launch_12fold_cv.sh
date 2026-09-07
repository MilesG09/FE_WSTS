#!/bin/bash
# Detached launcher for scripts/run_12fold_cv.sh -- generic over arm, so any future 12-fold CV
# (B0, A1, B1, C1, ...) launches the same way instead of a fresh one-off command each time.
#
# setsid, not just nohup: when launched through `wsl.exe -- bash -lc ...` the WSL session is torn
# down as soon as wsl.exe returns, and a plain background job dies with it (observed 2026-08-19:
# orchestrator gone within 90s, 0-byte log). setsid gives the job its own session with no
# controlling terminal so it survives the launching shell -- which is the whole point of an
# overnight run.
#
# Usage: launch_12fold_cv.sh <arm name> <num_workers> <prefetch_factor> <lr> <run_suffix> [seed]
#   e.g. launch_12fold_cv.sh A1 8 2 1e-3 bs64
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

ARM=${1:?arm name required, e.g. A1}
NW=${2:?num_workers required}
PF=${3:?prefetch_factor required}
LR=${4:?lr required}
SUFFIX=${5:?run_suffix required, e.g. bs64}

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/cv_${ARM}_${STAMP}.log"

setsid nohup bash scripts/run_12fold_cv.sh "$ARM" "$NW" "$PF" "$LR" "$SUFFIX" "${6:-0}" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 5
if kill -0 "$PID" 2>/dev/null; then
    echo "cv runner pid : $PID"
else
    echo "WARNING: runner died within 5s -- check $LOG" >&2
fi
echo "log           : $LOG"
echo "$PID" > "logs/cv_${ARM}.pid"
echo "$LOG" > "logs/cv_${ARM}.logpath"
