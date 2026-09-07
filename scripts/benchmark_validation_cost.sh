#!/bin/bash
# How much wall time does validation cost, and can it be cut without changing results?
#
# Measured on the fold 0 run of 2026-09-04: 2431.6s / 128 epochs = 19.0s per epoch, of
# which the train bar is ~14s and validation ~5s -- validation is ~26% of wall time. It is
# expensive because `val_avg_precision` is EXACT AP (thresholds=None), so it accumulates and
# sorts every prediction in the validation set once per epoch. Across a 12-arm x 12-fold
# EXP-03 that is roughly 26 hours of pure validation.
#
# Fold 0 is a known-exact fixture: baseline selects epoch 126 and returns
# test_AP = 0.5499971508979797. Two variants are timed against it.
#
#   A  WSTS_AP_THRESHOLDS=1000     binned AP. NOTE this changes the REPORTED metric too,
#                                  because val_avg_precision is a clone of
#                                  test_avg_precision -- so its test_AP is EXPECTED to move
#                                  by the documented ~0.003 and that is not a failure. The
#                                  signal that matters is whether it still selects epoch
#                                  126: if it does, binning is safe for CHECKPOINT
#                                  SELECTION, and a val-only implementation (constructing
#                                  val_avg_precision separately instead of cloning) would
#                                  then give a bit-identical test_AP for free.
#
#   B  check_val_every_n_epoch=2   validate half as often. 126 is even, so the baseline's
#                                  winning epoch is still reachable and the comparison is
#                                  meaningful rather than rigged.
#
# Everything else is pinned to the fixture: persistent_workers=false, prefetch_factor=2,
# num_workers=8, seed 0, deterministic, duplicate kept, max_steps=10000 (so both variants
# do the SAME number of training steps and only validation differs).
#
# Progress bar is left ON deliberately: the 40:31.6 baseline was measured with it, and
# turning it off here would make the wall times not comparable.
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
export PYTHONPATH="$PWD:$PWD/src"
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_MODE=offline
export WANDB_SILENT=true

run_variant() {   # $1 label, $2 extra trainer arg (may be empty), $3 AP thresholds env
    local label=$1 extra=$2 apt=$3
    local out="logs/veg_row2_wandb/valcost/$label"
    mkdir -p "$out"
    echo "=== [$(date +%H:%M:%S)] $label  (extra='$extra'  WSTS_AP_THRESHOLDS=$apt) ==="
    WSTS_AP_THRESHOLDS="$apt" /usr/bin/time -f "WALLCLOCK $label wall=%E user=%U sys=%S" \
    "$PY" src/train.py \
      --config=cfgs/unet/res18_monotemporal.yaml \
      --trainer=cfgs/trainer_single_gpu.yaml \
      --data=cfgs/data_monotemporal_veg_features.yaml \
      --data.data_dir=/home/miles/research/old_repo_FE_WSTS/hdf5_data \
      --data.data_fold_id=0 --data.num_workers=8 \
      --data.persistent_workers=false --data.prefetch_factor=2 \
      --data.is_pad=false --data.ignition_only_val_test=false --data.ignition_only_train=false \
      --seed_everything=0 --trainer.deterministic=true --trainer.max_steps=10000 \
      $extra \
      --trainer.default_root_dir="$PWD/$out" \
      --trainer.logger.init_args.name="fold0_valcost_$label" \
      --do_train=true --do_test=true --do_predict=false > "$out/run.log" 2>&1
    local ck; ck=$(find "$out" -name "*.ckpt" 2>/dev/null | head -1)
    echo "  best ckpt : $([ -n "$ck" ] && basename "$ck" || echo NONE)"
    echo "  test_AP   : $(grep -aE 'test_AP[[:space:]]+[0-9]' "$out/run.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)"
    grep -a "WALLCLOCK" "$out/run.log" || echo "  (no wallclock line)"
    echo
}

echo "############ validation-cost experiment, fold 0 ############"
echo "baseline (already measured): wall=40:31.60  best-epoch=126  test_AP=0.5499971508979797"
echo
run_variant "A_binned_ap"   ""                                    "1000"
run_variant "B_val_every_2" "--trainer.check_val_every_n_epoch=2"  "none"
echo "############ done ############"
