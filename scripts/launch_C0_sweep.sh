#!/bin/bash
# Launch the C0 12-fold CV sweep detached, with a timestamped log.
#
# Exists because invoking this through `wsl.exe -- bash -lc '...'` from the Windows
# side mangles shell variable expansion ($LOG, $!) before bash ever sees it, which
# silently produced a no-op launch. Putting the whole thing in a file removes the
# Windows quoting layer entirely.
#
# Settings match the A0 bs=64 sweep (num_workers=8, prefetch_factor=2, lr=1e-3) so
# the two arms stay comparable: every scientific parameter comes from the committed
# configs, only run identity and machine-local tuning are passed here.
#
# Usage: bash scripts/launch_C0_sweep.sh
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

mkdir -p logs
LOG="logs/C0_12fold_cv_$(date +%Y%m%d_%H%M%S).log"

# setsid, not just nohup. When this is invoked via `wsl.exe -- bash -lc ...` the WSL
# session is torn down as soon as wsl.exe returns, and a plain `nohup ... &` job dies
# with it (observed: orchestrator gone within 90s, 0-byte log). setsid puts the job in
# its own session with no controlling terminal, so it survives. A sweep launched from
# an interactive VSCode terminal never hit this, which is why the A0 run was fine.
setsid nohup bash scripts/run_12fold_cv.sh C 8 2 1e-3 bs64 > "$LOG" 2>&1 < /dev/null &
PID=$!
# Give the child a moment to actually start before reporting success.
sleep 5
if ! kill -0 "$PID" 2>/dev/null; then
    echo "WARNING: orchestrator $PID not alive 5s after launch - check $LOG" >&2
fi

echo "orchestrator pid : $PID"
echo "log              : $LOG"
echo "$PID" > logs/C0_sweep.pid
echo "$LOG" > logs/C0_sweep.logpath
