#!/bin/bash
# Detached launcher for scripts/run_val_adjustment_comparison.sh -- setsid, not just nohup, for
# the same reason as launch_12fold_cv.sh (WSL session teardown kills a plain background job).
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/cv_val_adjustment_comparison_${STAMP}.log"

setsid nohup bash scripts/run_val_adjustment_comparison.sh \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 5
if kill -0 "$PID" 2>/dev/null; then
    echo "cv runner pid : $PID"
else
    echo "WARNING: runner died within 5s -- check $LOG" >&2
fi
echo "log           : $LOG"
echo "$PID" > "logs/cv_val_adjustment_comparison.pid"
echo "$LOG" > "logs/cv_val_adjustment_comparison.logpath"
