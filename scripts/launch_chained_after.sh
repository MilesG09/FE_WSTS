#!/bin/bash
# Detached launcher for wait_and_chain.sh -- setsid, not just nohup, so it survives the launching
# shell exiting (same reason as launch_12fold_cv.sh / launch_val_adjustment_comparison.sh).
#
# Usage: launch_chained_after.sh <pid> <command...>
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/chain_after_${1}_${STAMP}.log"

setsid nohup bash scripts/wait_and_chain.sh "$@" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "chain waiter pid : $PID"
else
    echo "WARNING: chain waiter died within 2s -- check $LOG" >&2
fi
echo "log              : $LOG"
