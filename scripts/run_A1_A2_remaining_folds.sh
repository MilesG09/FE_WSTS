#!/bin/bash
# EXP-03, 2026-08-27: complete A1 and A2 to all 12 folds.
#
# WHY NOW: A0 passes its validation gate (EXPERIMENTS.md, 2026-08-26), and comparisons 1 and 2
# are A-arm-only -- they do not touch C0, whose gate failure stays documented as an open
# limitation. The reproduction-gap chase has eliminated AdamW eps, focal alpha/gamma, the
# learning rate, the torchmetrics AP implementation, and (2026-08-27) the fork's own diff against
# upstream d60631f, whose data-loading optimizations were verified byte-identical for A0/B0/C0 by
# dataset_probing/verify_dataload_optimization.py. The two remaining suspects need no GPU, so the
# GPU goes to the actual research question.
#
# WHAT THIS BUYS:
#   comparison 1 = A1 - A0  "is centroid POSITION useful, and not already learned, where the model
#                            sees the same single frame the centroid is derived from?"  <- headline
#   comparison 2 = A2 - A1  "does DISCLOSING a fabricated (patch-centre) centroid help?"
# Both pair against A0 baselines that already exist on all 12 folds, so no new control runs are
# needed and the between-year variance cancels (EXPERIMENTS.md, "Reporting standard").
#
# FOLDS: A1 and A2 already exist on [1, 5, 6, 11]. This fills in the other 8 of the 12.
#
# prefetch_factor=3 ON PURPOSE, matching the existing A1/A2 folds rather than the A0 baselines'
# pf=2. Within-arm consistency outranks cross-arm cosmetic matching: all 12 folds of an arm must
# be trained identically. (Safe either way -- pf changes DataLoader queue depth, not sample order.
# Demonstrated 2026-08-26: 6 of 12 A0/A0v fold pairs came back BIT-IDENTICAL across a pf 2->3
# change, which could not happen if pf perturbed training.)
#
# ARM ORDER: all of A1, then all of A2. Comparison 1 is the headline question, so if the night is
# cut short a complete A1 is worth more than two half-arms.
#
# THREE ROBUSTNESS GUARDS, added after the first launch attempt (2026-08-28 01:24) lost fold 0 to
# a transient `CUDA error: unknown error` that coincided with a WSL service disconnect. The fold
# had already trained two steps successfully, so it was an environment hiccup, not a config fault
# -- exactly the class of failure that should not cost an unattended night:
#   1. WAIT for any train.py already on the GPU before starting (so relaunching cannot put two
#      trainings on one card and OOM them both).
#   2. SKIP any fold that already has a logged test_AP (makes the script resumable, and means a
#      relaunch never redoes finished work).
#   3. RETRY each fold once on nonzero exit (recovers from exactly the transient above).
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
FOLDS=(0 2 3 4 7 8 9 10)
mkdir -p logs

# --- guard 1: do not start while another training holds the GPU ---
#
# The pattern MUST include the interpreter path. A bare "src/train.py" matches any process whose
# full command line merely CONTAINS that string -- `pgrep -f` matches the whole command line, and
# the `[s]` bracket trick only stops pgrep from matching its own invocation, nothing else.
# Cost of getting this wrong, 2026-08-28: an idle `git diff d60631f -- src/dataloader src/models
# src/train.py` pager sat in the process table overnight, this loop matched it, and the GPU stayed
# idle for twelve hours waiting on a paused pager.
GPU_PATTERN="${PY} src/train.py"
#
# And the wait is BOUNDED. An unbounded wait converts any false positive into a silently lost
# night; a bounded one degrades to "start anyway and let guard 3 retry" instead. A wait guard
# without a timeout is a single point of failure for an unattended job.
WAIT_LIMIT=$((30 * 60))
waited=0
while pgrep -f "$GPU_PATTERN" > /dev/null; do
    if [ "$waited" -ge "$WAIT_LIMIT" ]; then
        echo "=== [$(date '+%m-%d %H:%M:%S')] waited ${WAIT_LIMIT}s for the GPU; proceeding anyway"
        break
    fi
    echo "=== [$(date '+%m-%d %H:%M:%S')] waiting for the GPU (${waited}s):"
    pgrep -af "$GPU_PATTERN" | sed 's/^/      /'
    sleep 120
    waited=$((waited + 120))
done

# --- guard 2: is this run already finished? ---
already_done () {   # $1 = run name
    local n="$1"
    for f in mlruns/*/*/tags/mlflow.runName; do
        if [ "$(cat "$f" 2>/dev/null)" = "$n" ]; then
            [ -s "${f%/tags/mlflow.runName}/metrics/test_AP" ] && return 0
        fi
    done
    return 1
}

launch_once () {   # $1 = arm, $2 = fold, $3 = run name
    $PY src/train.py \
        --config=cfgs/models/unet_res18.yaml \
        --trainer=cfgs/trainer_single_gpu_mlflow.yaml \
        --do_train=True --do_test=True \
        --optimizer.init_args.lr=1e-3 \
        --data=cfgs/data_base.yaml \
        --data="cfgs/arms/${1}.yaml" \
        --data.data_fold_id="$2" \
        --data.num_workers=8 --data.prefetch_factor=3 \
        --trainer.max_steps=10000 \
        --trainer.logger.init_args.run_name="$3" \
        >> "logs/${3}.log" 2>&1
}

launch () {   # $1 = arm, $2 = fold
    local arm="$1" fold="$2"
    local name="${arm}_fold${fold}_seed0_bs64"
    if already_done "$name"; then
        echo "=== [$(date '+%m-%d %H:%M:%S')] SKIP  $name (already has a logged test_AP)"
        return 0
    fi
    echo "=== [$(date '+%m-%d %H:%M:%S')] START $name"
    local t0=$(date +%s)
    launch_once "$arm" "$fold" "$name"
    local rc=$?
    # --- guard 3: one retry, for transient CUDA/driver failures ---
    if [ $rc -ne 0 ]; then
        echo "=== [$(date '+%m-%d %H:%M:%S')] FAILED $name exit=$rc -- retrying once in 60s"
        sleep 60
        launch_once "$arm" "$fold" "$name"
        rc=$?
    fi
    echo "=== [$(date '+%m-%d %H:%M:%S')] DONE  $name  exit=$rc  ($(( ($(date +%s) - t0) / 60 )) min)"
}

for arm in A1 A2; do
    echo "########## ARM ${arm}: folds ${FOLDS[*]} ##########"
    for f in "${FOLDS[@]}"; do
        launch "$arm" "$f"
    done
done

echo "########## [$(date)] ALL DONE ##########"
echo "Read with:  python scripts/diagnose_arm_gap.py --arms A0 A1     # comparison 1"
echo "            python scripts/diagnose_arm_gap.py --arms A1 A2     # comparison 2"
