#!/bin/bash
# End-to-end A/B of the dataloader speed-ups on a REAL training run.
#
# Everything else measures __getitem__ in isolation, which answers "is the dataloader
# cheaper" but not "is the run shorter". Those diverge: once reads stop dominating, the
# remaining cost is GPU compute, validation and Python overhead, so a 1.7x on per-sample
# CPU does NOT mean a 1.7x run. This measures the thing that actually matters.
#
# Runs src/train.py twice with byte-identical arguments -- once with every WSTS_DISABLE_*
# flag set (original code paths) and once with none set (optimised) -- and reports wall,
# user and sys time for each. sys time is worth watching separately: the hyperslab's whole
# point is cutting kernel time spent in reads.
#
# Order is REVERSED on the second repeat (opt-first) so the page cache, which the first
# run warms for the second, cannot systematically favour either arm.
#
# Usage: scripts/benchmark_training_wallclock.sh [max_steps] [num_workers] [repeats]
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

STEPS=${1:-300}
NW=${2:-8}
REPEATS=${3:-1}

PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
DATA_DIR=${WSTS_DATA_DIR:-/home/miles/research/old_repo_FE_WSTS/hdf5_data}
OUTDIR=$(mktemp -d)

export PYTHONPATH="$PWD:$PWD/src"
export HDF5_USE_FILE_LOCKING=FALSE
# Offline so a network stall cannot land inside the timed region.
export WANDB_MODE=offline
export WANDB_SILENT=true

run_one() {  # $1 = label, $2 = "disabled"|"enabled"
    local label=$1 mode=$2
    if [ "$mode" = "disabled" ]; then
        export WSTS_DISABLE_HDF5_READ_OPT=1 WSTS_DISABLE_CROP_OPT=1 WSTS_DISABLE_ONEHOT_SKIP=1
    else
        unset WSTS_DISABLE_HDF5_READ_OPT WSTS_DISABLE_CROP_OPT WSTS_DISABLE_ONEHOT_SKIP
    fi
    echo "--- $label ($mode) ---"
    /usr/bin/time -f "$label  wall=%E  user=%U  sys=%S  maxrss=%MKB" \
        "$PY" src/train.py \
        -c cfgs/unet/res18_monotemporal.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --data cfgs/data_monotemporal_veg_features.yaml \
        --data.data_dir="$DATA_DIR" \
        --data.data_fold_id=0 \
        --data.num_workers="$NW" \
        --trainer.max_steps="$STEPS" \
        --trainer.default_root_dir="$OUTDIR" \
        --trainer.enable_progress_bar=false \
        --do_train=true --do_test=false --do_predict=false \
        > "$OUTDIR/$label.log" 2>&1
    # A crash at startup still produces a `wall=` line, and a 10s "run" quietly
    # reported as a result is worse than no result. Check the run actually finished.
    if ! grep -q "max_steps=$STEPS. reached" "$OUTDIR/$label.log"; then
        echo "  $label DID NOT COMPLETE -- tail of log:"
        tail -12 "$OUTDIR/$label.log"
        return 1
    fi
    grep -E "wall=" "$OUTDIR/$label.log"
}

echo "=== training wall-clock A/B: max_steps=$STEPS num_workers=$NW repeats=$REPEATS ==="
echo "=== logs in $OUTDIR ==="
for r in $(seq 1 "$REPEATS"); do
    echo
    echo "### repeat $r"
    if [ $((r % 2)) -eq 1 ]; then
        run_one "r${r}_original"  disabled
        run_one "r${r}_optimised" enabled
    else
        run_one "r${r}_optimised" enabled
        run_one "r${r}_original"  disabled
    fi
done
echo
echo "=== done. Full Lightning logs kept in $OUTDIR ==="
