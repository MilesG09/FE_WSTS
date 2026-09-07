#!/bin/bash
# Detached launcher for overnight_grid_then_folds.sh.
#
# setsid, not just nohup: when launched through `wsl.exe -- bash -lc ...` the WSL session is
# torn down as soon as wsl.exe returns, and a plain background job dies with it (observed
# 2026-08-19: orchestrator gone within 90s, 0-byte log). setsid gives the job its own session
# with no controlling terminal so it survives.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

mkdir -p logs
LOG="logs/overnight_$(date +%Y%m%d_%H%M%S).log"

setsid nohup bash scripts/overnight_grid_then_folds.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
sleep 5
if kill -0 "$PID" 2>/dev/null; then
    echo "orchestrator pid : $PID"
else
    echo "WARNING: orchestrator died within 5s -- check $LOG" >&2
fi
echo "log              : $LOG"
echo "$PID" > logs/overnight.pid
echo "$LOG" > logs/overnight.logpath
