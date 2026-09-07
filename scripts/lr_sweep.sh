#!/bin/bash
# Learning-rate sweep for A0, fold 2, 50 epochs, no-test runs, logged into the mlflow
# experiment "Testing computer settings for training" (same throwaway-sweep experiment used
# by training_parameter_search.sh, kept separate from "WSTS FE Experiments" so this stays out
# of the real experimental record).
#
# num_workers/prefetch_factor are NOT hardcoded here -- pass in whatever
# training_parameter_search.sh found to be fastest on this machine, e.g.:
#   scripts/lr_sweep.sh --num_workers 4 --prefetch_factor 2
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
TRAIN=src/train.py

NUM_WORKERS=""
PREFETCH_FACTOR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --prefetch_factor) PREFETCH_FACTOR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$NUM_WORKERS" ] || [ -z "$PREFETCH_FACTOR" ]; then
    echo "Usage: $0 --num_workers <N> --prefetch_factor <N>"
    echo "(use whatever training_parameter_search.sh found fastest for arm A)"
    exit 1
fi

LR_LIST=(1e-2 1e-3 1e-4 1e-5)
EXPERIMENT_NAME="Testing computer settings for training"
RUN_PREFIX="lr_sweep_A0_fold2"

echo "=== LR sweep: A0, fold 2, 50 epochs, num_workers=$NUM_WORKERS, prefetch_factor=$PREFETCH_FACTOR ==="
for lr in "${LR_LIST[@]}"; do
    run_name="${RUN_PREFIX}_lr${lr}"
    echo "=== Training lr=$lr (run_name=$run_name) ==="
    "$PY" "$TRAIN" \
        --config=cfgs/models/unet_res18.yaml \
        --trainer=cfgs/trainer_single_gpu_mlflow.yaml \
        --do_train=True \
        --do_test=False \
        --optimizer.init_args.lr="$lr" \
        --data=cfgs/data_base.yaml \
        --data.data_dir=/home/miles/FE_WSTS/hdf5_data \
        --data=cfgs/arms/A0.yaml \
        --data.data_fold_id=2 \
        --data.num_workers="$NUM_WORKERS" \
        --data.prefetch_factor="$PREFETCH_FACTOR" \
        --trainer.max_epochs=50 \
        --trainer.logger.init_args.experiment_name="$EXPERIMENT_NAME" \
        --trainer.logger.init_args.run_name="$run_name"
    if [ $? -ne 0 ]; then
        echo "$run_name FAILED -- continuing to next lr"
    fi
done

echo "=== Sweep runs complete. Finding best lr by peak val_avg_precision ==="
"$PY" scripts/lr_sweep_analyze.py --run_prefix "$RUN_PREFIX"
