#!/bin/bash
# Overnight job, 2026-08-20:
#   Phase 1 - 6-config REAL grid (num_workers x prefetch_factor), 500 real training steps each.
#   Phase 2 - C0 folds 6-11, completing the 12-fold CV started 2026-08-19.
#
# Why real 500-step runs and not scripts/benchmark_dataloader_throughput.py: that benchmark
# measures DataLoader capacity with no consumer -- no GPU work competing for CPU, no
# backpressure -- and its 10s measurement window is so noisy it returned three mutually
# contradictory winners for the same question. A 500-step run measures wall-clock under real
# contention AND averages over ~3 minutes, which is what makes it trustworthy.
#
# PHASE 2 USES nw=8 pf=2 ON PURPOSE, regardless of what Phase 1 wins.
# Folds 0-5 ran at nw=8/pf=2. num_workers determines how many dataloader RNG streams exist,
# so changing it changes which augmentations land on which samples. Mixing configs inside one
# 12-fold CV would mean the 12 folds were not trained identically. The grid result is for
# FUTURE sweeps; internal consistency of this CV outranks a possible ~10% speedup.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="logs/grid_results_${STAMP}.csv"

mkdir -p logs
echo "num_workers,prefetch_factor,seconds,exit_code" > "$RESULTS"

common_args() {
    echo "--config=cfgs/models/unet_res18.yaml \
--trainer=cfgs/trainer_single_gpu_mlflow.yaml \
--optimizer.init_args.lr=1e-3 \
--data=cfgs/data_base.yaml \
--data=cfgs/arms/C0.yaml"
}

echo "=== [$(date)] PHASE 1: real 500-step grid (C0, fold 0) ==="

# Warm-up: the first run of the night pays cold-cache and CUDA-context costs that would
# otherwise be charged to whichever grid cell happened to run first.
echo "=== [$(date)] warm-up run (untimed) ==="
$PY src/train.py $(common_args) \
    --do_train=True --do_test=False \
    --data.data_fold_id=0 --data.num_workers=8 --data.prefetch_factor=2 \
    --trainer.max_steps=200 \
    --trainer.logger.init_args.experiment_name="Testing computer settings for training" \
    --trainer.logger.init_args.run_name="warmup_C0_${STAMP}" > /dev/null 2>&1
echo "=== [$(date)] warm-up done ==="

for nw in 8 12 16; do
    for pf in 2 4; do
        name="grid_C0_nw${nw}_pf${pf}_${STAMP}"
        echo "=== [$(date)] grid cell nw=$nw pf=$pf ==="
        start=$(date +%s)
        $PY src/train.py $(common_args) \
            --do_train=True --do_test=False \
            --data.data_fold_id=0 --data.num_workers="$nw" --data.prefetch_factor="$pf" \
            --trainer.max_steps=500 \
            --trainer.logger.init_args.experiment_name="Testing computer settings for training" \
            --trainer.logger.init_args.run_name="$name" > "logs/${name}.log" 2>&1
        rc=$?
        elapsed=$(( $(date +%s) - start ))
        echo "$nw,$pf,$elapsed,$rc" >> "$RESULTS"
        echo "=== [$(date)] nw=$nw pf=$pf -> ${elapsed}s (exit $rc) ==="
    done
done

echo ""
echo "=== [$(date)] PHASE 1 COMPLETE - results ==="
cat "$RESULTS"
# Fastest successful cell, for the record. Not applied to Phase 2 -- see header.
best=$(awk -F, 'NR>1 && $4==0 {print $3","$1","$2}' "$RESULTS" | sort -n | head -1)
echo "fastest cell (seconds,nw,pf): ${best:-none succeeded}"
echo "NOTE: Phase 2 deliberately uses nw=8 pf=2 for CV consistency, not this winner."
echo ""

echo "=== [$(date)] PHASE 2: C0 folds 6-11 at nw=8 pf=2 ==="
for fold in 6 7 8 9 10 11; do
    run_name="C0_fold${fold}_seed0_bs64"
    echo "=== [$(date)] Starting fold $fold ($run_name) ==="
    $PY src/train.py $(common_args) \
        --do_train=True --do_test=True \
        --data.data_fold_id="$fold" --data.num_workers=8 --data.prefetch_factor=2 \
        --trainer.max_steps=10000 \
        --trainer.logger.init_args.run_name="$run_name"
    rc=$?
    echo "=== [$(date)] Fold $fold exited $rc ==="
    if [ $rc -ne 0 ]; then
        echo "$run_name FAILED (exit $rc) -- continuing to next fold"
    fi
done

echo "=== [$(date)] ALL DONE: grid + C0 folds 6-11 ==="
