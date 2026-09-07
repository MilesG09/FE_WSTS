#!/bin/bash
# 12-fold LOYO cross-validation runner, 2026-08-18: reruns the 2026-08-16 A0 12-fold
# replication (EXPERIMENTS.md, "A0 12-fold replication -- RESULT") under settings corrected /
# confirmed the same night as this rerun -- batch_size=64 (fixed 2026-08-17, was 32 before), the
# freshly re-measured fastest dataloader config (num_workers=8, prefetch_factor=2 for A0,
# confirmed a real floor by a follow-up nw=10/12 probe rather than an artifact of an incomplete
# grid -- both were *slower* than nw=8), and lr=1e-3 (confirmed best of 4 tested by the LR
# sweep, not carried over unverified). Requested and run autonomously per explicit instruction --
# arm, protocol, and settings are all his call, this only executes it.
#
# Individual fold failures (incl. OOM) are logged and skipped, not fatal to the run -- matches
# the tolerance already built into training_parameter_search.sh / lr_sweep.sh. Every scientific
# parameter (loss, pretraining, feature set, crop size, etc.) comes from the committed configs,
# untouched -- only run identity and machine-local tuning are on the CLI, same convention as the
# 2026-08-16 run.
#
# Usage: run_12fold_cv.sh <arm name, e.g. A1> <num_workers> <prefetch_factor> <lr> <run_suffix> [seed]
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
# FE_WSTS tracks with wandb, not MLflow. WSTS_original is the only env whose torch
# (2.7.0+cu128) compiles for sm_120 -- the RTX 5070's compute capability. The FE_WSTS
# env tops out at sm_86 and cannot run on this GPU at all.
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$PWD:$PWD/src"
PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
TRAIN=src/train.py

# Full arm NAME now (A0, A1, C0, ...), not just the letter: the old form hardcoded a
# trailing "0" and so could only ever launch the centroid-free control arms.
ARM=${1:?arm name required, e.g. A1}
NW=${2:?num_workers required}
PF=${3:?prefetch_factor required}
LR=${4:?lr required}
# NOTE: SUFFIX is no longer used in the run name or the wandb group -- it duplicated a
# config value (batch_size) and could go stale against it. The positional SLOT is kept
# deliberately: nine scripts share this signature, and deleting a middle positional would
# silently shift SEED into its place. Pass anything; it is ignored.
SUFFIX=${5:?deprecated, ignored in naming}
SEED=${6:-0}
ARM_ID="$ARM"

if [ ! -f "cfgs/arms/${ARM}.yaml" ]; then
    echo "ERROR: cfgs/arms/${ARM}.yaml does not exist" >&2
    exit 1
fi

# LR is a SCIENTIFIC parameter and lives in cfgs/unet/res18_monotemporal.yaml, never on the
# command line (cfgs/data_base.yaml header rule -- the fix for the batch-size confound that hit
# EXP-01/02). The $LR argument is therefore an ASSERTION, not an override: it states what the
# caller believes the config says, and the run is refused if they disagree. Nine scripts share
# this signature, so the slot stays; only its meaning changed.
CFG_LR=$(grep -A3 "^optimizer:" cfgs/unet/res18_monotemporal.yaml | grep -oP "lr:\s*\K[0-9.eE+-]+")
if [ "$(printf '%s' "$CFG_LR" | awk '{printf "%g", $1}')" != "$(printf '%s' "$LR" | awk '{printf "%g", $1}')" ]; then
    echo "ERROR: lr mismatch -- caller asserted '$LR' but cfgs/unet/res18_monotemporal.yaml says '$CFG_LR'." >&2
    echo "       Change the YAML, not the command line." >&2
    exit 1
fi
echo "=== lr asserted: config says $CFG_LR, caller expects $LR -- match ==="

echo "=== [$(date)] Starting ${ARM} 12-fold CV: num_workers=$NW prefetch_factor=$PF lr=$LR suffix=$SUFFIX ==="

for fold in $(seq 0 11); do
    run_name="${ARM}_seed${SEED}_fold${fold}"
    # arm_id / fold / seed as wandb tags, written at the source. EXPERIMENTS.md records
    # centroid runs arriving with no arm/fold/config tags at all -- unattributable after
    # the fact. The run name alone is a string; tags are queryable.
    export ARM_ID="$ARM_ID"
    # Native wandb grouping: all 12 folds of one arm collapse to a single row
    # in the UI with mean/stddev across folds -- which is exactly the unit of
    # analysis for a LOYO cross-validation.
    export WANDB_RUN_GROUP="${ARM_ID}_seed${SEED}"
    export WANDB_TAGS="arm_${ARM_ID},fold_${fold},seed_${SEED}"
    echo "=== [$(date)] Starting fold $fold ($run_name) ==="

    "$PY" "$TRAIN" \
        --config=cfgs/unet/res18_monotemporal.yaml \
        --trainer=cfgs/trainer_single_gpu.yaml \
        --do_train=True \
        --do_test=True \
        --data=cfgs/data_base.yaml \
        --data=cfgs/arms/${ARM}.yaml \
        --data.data_fold_id="$fold" \
        --data.num_workers="$NW" \
        --data.prefetch_factor="$PF" \
        --seed_everything="$SEED" \
        --trainer.max_steps=10000 \
        --trainer.logger.init_args.name="$run_name"
    exit_code=$?

    echo "=== [$(date)] Fold $fold exited $exit_code ==="
    if [ $exit_code -ne 0 ]; then
        echo "$run_name FAILED (exit $exit_code) -- continuing to next fold"
    fi
done

echo "=== [$(date)] 12-fold CV complete for ${ARM} ==="
