#!/bin/bash
# Quick follow-up probe, 2026-08-18: confirm whether num_workers=10/12 buy anything over the
# 2026-08-17/18 sweep's nw=8 winner for A0, or whether that sweep already hit the compute-bound
# floor. The full sweep's marginal-gain trend (65s / 26.7s / 7.8s-per-worker as nw went
# 2->3->4->8, and all four prefetch_factor values landing within 8s of each other at nw=8)
# strongly suggested diminishing returns rather than still being dataloader-bound -- this
# settles it empirically with 2 runs instead of committing to a full new grid. prefetch_factor
# fixed at 2 (that sweep's overall arm-A winner), so num_workers is the only variable.
#
# Arm C intentionally skipped: already known to OOM-kill at num_workers>=12 even before the
# batch_size 32->64 fix made worker memory pressure (num_workers x prefetch_factor x batch_size)
# worse, not better (TODO.md, "Throughput sweep may be invalid").
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
TRAIN=src/train.py
RESULTS_CSV=training_parameter_search_results.csv
PF=2

get_train_args() {
    local nw=$1 run_name=$2
    TRAIN_ARGS=(
        "--config=cfgs/models/unet_res18.yaml"
        "--trainer=cfgs/trainer_single_gpu_mlflow.yaml"
        "--do_train=True"
        "--do_test=False"
        "--data=cfgs/data_base.yaml"
        "--data.data_dir=/home/miles/FE_WSTS/hdf5_data"
        "--data.data_fold_id=0"
        "--data.num_workers=$nw"
        "--data.prefetch_factor=$PF"
        "--data=cfgs/arms/A0.yaml"
        "--trainer.max_steps=500"
        "--trainer.logger.init_args.experiment_name=Testing computer settings for training"
        "--trainer.logger.init_args.run_name=$run_name"
    )
}

wait_for_memory_settled() {
    for i in $(seq 1 60); do
        read -r _ total used free_mem shared cache avail < <(free -m | awk '/^Mem:/{print}')
        if [ "$avail" -gt 8000 ]; then
            echo "memory settled: available=${avail}MB"
            return 0
        fi
        sleep 5
    done
    echo "WARNING: memory did not settle after 5 minutes, proceeding anyway"
}

echo "=== Warm-up (untimed) for arm A, priming OS disk cache ==="
get_train_args 4 "warmup_A0_probe"
"$PY" "$TRAIN" "${TRAIN_ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "Warm-up FAILED -- continuing anyway"
fi

for nw in 10 12; do
    run_name="throughput_A0_nw${nw}_pf${PF}"
    get_train_args "$nw" "$run_name"

    echo "=== waiting for settled memory before $run_name ==="
    wait_for_memory_settled

    echo "Running $run_name..."
    start=$(date +%s.%N)
    "$PY" "$TRAIN" "${TRAIN_ARGS[@]}"
    exit_code=$?
    end=$(date +%s.%N)

    if [ $exit_code -ne 0 ]; then
        echo "$run_name FAILED (exit $exit_code)"
        continue
    fi

    seconds=$(echo "$end - $start" | bc)
    echo "A,$nw,$PF,$seconds" >> "$RESULTS_CSV"
done

echo "=== Probe results (nw=10,12 vs. existing nw=8 baseline, all at pf=2) ==="
{ head -1 "$RESULTS_CSV"; tail -n +2 "$RESULTS_CSV" | awk -F, '$1=="A" && $3==2' | sort -t, -k4 -n; } | column -s, -t
echo "=== Probe complete ==="
