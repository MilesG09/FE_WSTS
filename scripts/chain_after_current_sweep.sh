#!/bin/bash
# Waits for the currently-running sequential arm sweep to finish, then launches the next
# sequence. Detached with setsid by its own launcher so it outlives the session that armed it
# (same reasoning as launch_arms_sequential.sh: a plain background job dies when wsl.exe returns).
#
# Trigger is the runner's own completion banner, which run_arms_sequential.sh prints regardless
# of per-arm exit codes -- so a failing arm still hands off, matching that script's "a failing arm
# does NOT stop the ones after it" contract.
#
# To CANCEL before it fires: create scripts/ABORT_CHAIN (any contents), or kill this pid.
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

WATCH_LOG=${1:?log of the sweep to wait on}
NEXT_ARMS=${2:?arm list to launch next}
FOLDS=${3:?fold list}
NW=${4:?num_workers}
PF=${5:?prefetch_factor}
LR=${6:?lr}
SUFFIX=${7:?run_suffix}
SEED=${8:-0}

echo "=== [$(date)] chain armed: waiting on $WATCH_LOG, will then launch [$NEXT_ARMS] ==="

while true; do
    if [ -f scripts/ABORT_CHAIN ]; then
        echo "=== [$(date)] ABORT_CHAIN present -- chain cancelled, launching nothing ==="
        exit 0
    fi
    if grep -aq "Sequential arm run complete" "$WATCH_LOG"; then
        echo "=== [$(date)] detected completion banner ==="
        grep -a "Sequential arm run complete" "$WATCH_LOG" | tail -1
        break
    fi
    sleep 60
done

# Re-check the abort sentinel after the wait: the whole point is that it can be created at any
# time up to the moment of launch.
if [ -f scripts/ABORT_CHAIN ]; then
    echo "=== [$(date)] ABORT_CHAIN present at launch time -- cancelled ==="
    exit 0
fi

# Do not stack two sweeps on one GPU. The completion banner means the runner exited, but give any
# lingering train.py (wandb teardown can run minutes past the last fold) time to actually go.
for _ in $(seq 1 60); do
    pgrep -f 'src/train.py' >/dev/null || break
    echo "=== [$(date)] train.py still alive, waiting ==="
    sleep 60
done
if pgrep -f 'src/train.py' >/dev/null; then
    echo "=== [$(date)] ERROR: train.py STILL running after 60 min -- refusing to launch and stack on the GPU ==="
    exit 1
fi

echo "=== [$(date)] launching next sequence: [$NEXT_ARMS] folds=[$FOLDS] nw=$NW pf=$PF lr=$LR seed=$SEED ==="
bash scripts/launch_arms_sequential.sh "$NEXT_ARMS" "$FOLDS" "$NW" "$PF" "$LR" "$SUFFIX" "$SEED"
echo "=== [$(date)] chain done ==="
