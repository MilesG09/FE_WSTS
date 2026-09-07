#!/bin/bash
# Measure exactly how much hdf5_data/2019/WildfireSpreadTS.hdf5 moves the reported test_AP.
#
# That file is a BYTE-IDENTICAL duplicate of 2019/fire_23572745.hdf5 (verified: all 19
# frames equal, same lnglat, same size) with only its `fire_name` and `img_dates` attrs
# corrupted by a Windows path-separator bug. `read_list_of_images` globs `*.hdf5`, so it
# counts as a 75th 2019 fire and that one fire's samples enter every 2019 split twice.
#
# Folds 3, 5 and 10 are the ones where 2019 is the TEST year, so those are the only folds
# where it can move a REPORTED number. Test-time transforms are deterministic
# (center_crop_x32, no augmentation), so the duplicate produces exactly the same (x, y)
# pairs as the fire it clones -- the effect is precisely "one fire's pixels counted twice
# in the pooled PR curve", nothing subtler.
#
# The dataset on disk is NOT touched. The clean condition runs against a directory of
# SYMLINKS to the real files that simply omits the stray, so there is no rename to undo
# and no way for an interrupted run to leave the dataset in a modified state.
#
# Usage: scripts/measure_stray_2019_impact.sh [fold ...]        (default: 3 5 10)
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

FOLDS=${@:-3 5 10}
PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
DIRTY=${WSTS_DATA_DIR:-/home/miles/research/old_repo_FE_WSTS/hdf5_data}
CLEAN=${WSTS_CLEAN_DATA_DIR:?set WSTS_CLEAN_DATA_DIR to the symlink mirror that omits the stray}
OUTDIR=$(mktemp -d)

export PYTHONPATH="$PWD:$PWD/src"
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_MODE=offline
export WANDB_SILENT=true

ckpt_for() {  # newest checkpoint for a fold, preferring the detrun/ckpt2 rerun
    find logs/veg_row2_wandb -path "*fold$1/*" -name "*.ckpt" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

run_test() {  # $1 fold, $2 label, $3 data_dir, $4 ckpt
    "$PY" src/train.py \
        -c cfgs/unet/res18_monotemporal.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --data cfgs/data_monotemporal_veg_features.yaml \
        --data.data_dir="$3" \
        --data.data_fold_id="$1" \
        --data.num_workers=8 \
        --data.is_pad=false \
        --data.ignition_only_val_test=false \
        --data.ignition_only_train=false \
        --trainer.default_root_dir="$OUTDIR" \
        --trainer.enable_progress_bar=false \
        --do_train=false --do_test=true --do_predict=false \
        --ckpt_path="$4" > "$OUTDIR/f$1_$2.log" 2>&1
    # Lightning prints the test metrics as a boxed table. The metric is logged as
    # `test_AP` (BaseModel), NOT `test_avg_precision` -- that name only exists for the
    # validation metric. Matching the wrong one yields an empty string, which reads as
    # "the run failed" when the run was fine.
    grep -E "test_AP[[:space:]]+[0-9]" "$OUTDIR/f$1_$2.log" \
        | grep -oE "[0-9]+\.[0-9]+" | tail -1
}

echo "=== stray-file impact on test_AP: folds $FOLDS ==="
echo "=== logs in $OUTDIR ==="
printf "\n%-6s %-12s %-12s %-12s  %s\n" fold with_stray without delta ckpt
for f in $FOLDS; do
    ck=$(ckpt_for "$f")
    if [ -z "$ck" ]; then printf "%-6s no checkpoint found\n" "$f"; continue; fi
    a=$(run_test "$f" dirty "$DIRTY" "$ck")
    b=$(run_test "$f" clean "$CLEAN" "$ck")
    if [ -z "$a" ] || [ -z "$b" ]; then
        printf "%-6s FAILED (see $OUTDIR/f${f}_*.log)\n" "$f"; continue
    fi
    d=$($PY -c "print(f'{float('$b')-float('$a'):+.6f}')")
    printf "%-6s %-12s %-12s %-12s  %s\n" "$f" "$a" "$b" "$d" "$(basename "$ck")"
done
echo
echo "=== done. logs in $OUTDIR ==="
