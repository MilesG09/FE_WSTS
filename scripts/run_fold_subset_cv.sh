#!/bin/bash
# Generalizes run_12fold_cv.sh to (a) an arbitrary fold subset instead of always 0-11, and (b) an
# optional extra --data override file, so a run like the val-adjustment comparison arms (A0v/
# B0v/C0v -- administrative/EXPERIMENTS.md, "Val-adjustment comparison arms") doesn't need a new
# one-off script. run_12fold_cv.sh is left untouched since it's the already-validated path for
# the original A0/B0/C0 gate arms.
#
# Every scientific parameter still comes from the committed configs (cfgs/data_base.yaml +
# cfgs/arms/<ARM>.yaml + optional extra override) -- only run identity, fold list, and
# machine-local dataloader tuning are on the CLI, same convention as run_12fold_cv.sh.
#
# Usage: run_fold_subset_cv.sh <arm_yaml e.g. A0> <run_prefix e.g. A0v> <extra_data_yaml|none> \
#                               <fold_list e.g. "1 5 6 11"> <num_workers> <prefetch_factor> <lr> <run_suffix> \
#                               [seed] [run_name_tag]
# Example (A0v, matching A0's original nw=8/pf=3):
#   run_fold_subset_cv.sh A0 A0v cfgs/data_val_adjustment.yaml "1 5 6 11" 8 3 1e-3 bs64
# Example (A0 re-run on another box, tagged so it does not shadow the canonical A0 runs):
#   run_fold_subset_cv.sh A0 A0 none "1 5 6 11" 12 4 1e-3 ignored 0 test
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
# FE_WSTS tracks with wandb, not MLflow. WSTS_original is the only env whose torch
# (2.7.0+cu128) compiles for sm_120 -- the RTX 5070's compute capability. The FE_WSTS
# env tops out at sm_86 and cannot run on this GPU at all.
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$PWD:$PWD/src"
PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
DATA_DIR=""
# Machine-local overrides (interpreter path, dataset path) -- untracked, one per machine.
# Sourced AFTER the defaults so the local file wins; absent on the home box, so no-op there.
[ -f env.local.sh ] && . ./env.local.sh
[ -x "$PY" ] || { echo "ERROR: PY=$PY is not executable -- set it in env.local.sh" >&2; exit 1; }
[ -z "$DATA_DIR" ] || [ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR=$DATA_DIR does not exist" >&2; exit 1; }
DATA_DIR_ARG=(); [ -n "$DATA_DIR" ] && DATA_DIR_ARG=(--data.data_dir="$DATA_DIR")
# wandb auth gate. Without a key wandb does not fail -- it silently switches to OFFLINE and
# the run never reaches the server (cause of the stray offline-run-* dirs, 2026-09). A 12-fold
# sweep that logs nowhere is worse than one that refuses to start, so refuse. Set WANDB_MODE
# explicitly (offline/disabled) to opt out on purpose.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    echo "ERROR: no wandb credentials (WANDB_API_KEY unset, no ~/.netrc)." >&2
    echo "       Runs would log OFFLINE and never reach the server." >&2
    echo "       Set WANDB_API_KEY in env.local.sh, or export WANDB_MODE=offline to accept it." >&2
    exit 1
fi
TRAIN=src/train.py

ARM_YAML=${1:?arm yaml name required, e.g. A0}
RUN_PREFIX=${2:?run name prefix required, e.g. A0v}
EXTRA_DATA=${3:?extra data yaml path required, or the literal string 'none'}
FOLDS=${4:?fold list required, e.g. "1 5 6 11"}
NW=${5:?num_workers required}
PF=${6:?prefetch_factor required}
LR=${7:?lr required}
# NOTE: SUFFIX is no longer used in the run name or the wandb group -- it duplicated a
# config value (batch_size) and could go stale against it. The positional SLOT is kept
# deliberately: nine scripts share this signature, and deleting a middle positional would
# silently shift SEED into its place. Pass anything; it is ignored.
SUFFIX=${8:?deprecated, ignored in naming}
# Optional 9th arg so existing callers keep working. seed_everything is pinned to 0 in
# cfgs/unet/res18_monotemporal.yaml, so a run merely NAMED seed1 would still be seed 0 --
# the seed has to be passed to the CLI, not just into the run name.
SEED=${9:-0}
# Optional 10th arg: a free-form label appended to the run NAME only -- not to WANDB_RUN_GROUP
# or WANDB_TAGS -- so a re-run of an already-validated arm on a different machine is
# distinguishable in the run list while still landing in its arm's group for direct
# comparison (the machine is already a wandb tag, see src/train.py). Unlike the dead SUFFIX
# slot this carries no config value and cannot go stale against one. Empty -> names unchanged.
RUN_TAG=${10:-}

ARM_ID="$ARM_YAML"

EXTRA_DATA_ARG=()
if [ "$EXTRA_DATA" != "none" ]; then
    EXTRA_DATA_ARG=(--data="$EXTRA_DATA")
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

echo "=== [$(date)] Starting ${RUN_PREFIX} fold-subset CV: folds=[$FOLDS] num_workers=$NW prefetch_factor=$PF lr=$LR extra_data=$EXTRA_DATA run_tag=${RUN_TAG:-<none>} ==="

for fold in $FOLDS; do
    run_name="${RUN_PREFIX}_seed${SEED}_fold${fold}${RUN_TAG:+_${RUN_TAG}}"
    # arm_id / fold / seed as wandb tags, written at the source. EXPERIMENTS.md records
    # centroid runs arriving with no arm/fold/config tags at all -- unattributable after
    # the fact. The run name alone is a string; tags are queryable.
    export ARM_ID="$ARM_ID"
    # Native wandb grouping: all 12 folds of one arm collapse to a single row
    # in the UI with mean/stddev across folds -- which is exactly the unit of
    # analysis for a LOYO cross-validation.
    export WANDB_RUN_GROUP="${ARM_ID}_seed${SEED}"
    # RUN_TAG (if any) also rides along as a plain wandb tag so the side runs are filterable
    # in the UI, not just distinguishable by name. Group stays "${ARM_ID}_seed${SEED}" so a
    # consistency-check re-run still lands beside the canonical runs for direct comparison.
    export WANDB_TAGS="arm_${ARM_ID},fold_${fold},seed_${SEED}${RUN_TAG:+,${RUN_TAG}}"
    echo "=== [$(date)] Starting fold $fold ($run_name) ==="

    "$PY" "$TRAIN" \
        --config=cfgs/unet/res18_monotemporal.yaml \
        --trainer=cfgs/trainer_single_gpu.yaml \
        --do_train=True \
        --do_test=True \
        --data=cfgs/data_base.yaml \
        --data=cfgs/arms/${ARM_YAML}.yaml \
        "${EXTRA_DATA_ARG[@]}" \
        "${DATA_DIR_ARG[@]}" \
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

echo "=== [$(date)] Fold-subset CV complete for ${RUN_PREFIX} ==="
