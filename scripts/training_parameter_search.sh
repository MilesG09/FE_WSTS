#!/bin/bash
# WSL2 translation of powershell_scripts/training_parameter_search.ps1, re-run because the
# WSL2 migration (persistent_workers=True + per-worker HDF5 handle caching, 2026-08-15) makes
# the old Windows-tuned (num_workers, prefetch_factor) grid non-transferable: fork-based worker
# pools don't pay the respawn cost that spawn-based Windows pools did, so the old optimum has no
# reason to still be optimal. Sweeps only machine-local tuning params -- batch_size and every
# other scientific parameter stay fixed in cfgs/data_base.yaml / cfgs/arms/, untouched here.
#
# Updated 2026-08-16 after the OOM investigation: hdf5_cache_size now defaults to 64 (LRU-capped
# HDF5 file handle cache -- was unbounded, causing gradual per-worker memory growth), val
# workers are no longer persistent (halves aggregate worker-process count), and C0's num_workers
# grid is capped at 8 -- confirmed 12 persistent train workers alone still OOM-kill on this
# machine for C0's larger per-sample footprint, even with both fixes above applied. A0 never hit
# this problem at any tested worker count, so its grid is untouched. Also adds a memory-settled
# wait between every timed run: residual pressure from one run was contaminating the next run's
# timing in the original version of this sweep, undermining the exact thing being measured.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
TRAIN=src/train.py

# Updated 2026-08-17 after batch_size 32->64 (see CHANGELOG.md, paper-fidelity fix): per-worker
# prefetch memory scales as num_workers x prefetch_factor x batch_size, so doubling batch_size
# doubles that product for every combo that was in the old (batch_size=32) grid. Re-centered
# rather than just shrunk: dropped the old top end (A's 12/16, C's 8) since those were already
# OOM-adjacent before the doubling, and added a new bottom end (nw=2,3; pf=1) since the old
# floor (nw=4, pf=2) is no longer guaranteed to be the new optimum either -- it may now sit
# below where the old grid started. If tonight's results don't show a clear peak strictly below
# the cut point (i.e. throughput is still improving at the new ceiling), that's a signal the old
# top end needs a separate follow-up run rather than being ruled out permanently.
declare -A ARM_NUM_WORKERS=( [A]="2 3 4 8" [C]="2 3 4" )
PREFETCH_FACTOR_LIST=(1 2 3 4)
EXP_ARMS=(A C)

wait_for_memory_settled() {
    # Gate on *available* memory only, not swap_used -- swap doesn't self-clear (Linux only
    # swaps pages back in on demand), so requiring low swap_used waits on a stale number that
    # reflects past pressure, not current headroom.
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

get_train_args() {
    local letter=$1 nw=$2 pf=$3 run_name=$4
    TRAIN_ARGS=(
        "--config=cfgs/models/unet_res18.yaml"
        "--trainer=cfgs/trainer_single_gpu_mlflow.yaml"
        "--do_train=True"
        "--do_test=False"
        "--data=cfgs/data_base.yaml"
        "--data.data_dir=/home/miles/FE_WSTS/hdf5_data"
        "--data.data_fold_id=0"
        "--data.num_workers=$nw"
        "--data.prefetch_factor=$pf"
        "--data=cfgs/arms/${letter}0.yaml"
        "--trainer.max_steps=500"
        "--trainer.logger.init_args.experiment_name=Testing computer settings for training"
        "--trainer.logger.init_args.run_name=$run_name"
    )
}

# --- Fix 2: untimed warm-up, one per arm, before any timed run starts. ---
# All (nw, pf) configs within an arm train on the same fold with the same fixed seed
# (seed_everything: 0 in cfgs/models/unet_res18.yaml), so they read a heavily-overlapping
# sequence of bytes from the same HDF5 files. Without this, whichever config happens to run
# LAST benefits most from OS page-cache warming built up by every earlier run -- a bias in run
# *position*, not in (nw, pf), that could flip the ranking outright. Warming the cache for both
# arms up front, before any measurement, means every timed run starts from the same
# (mostly-warm) cache state.
echo "=== Warm-up runs (untimed, priming OS disk cache for each arm) ==="
for letter in "${EXP_ARMS[@]}"; do
    get_train_args "$letter" 4 "${PREFETCH_FACTOR_LIST[0]}" "warmup_${letter}0"
    echo "Warm-up for arm $letter..."
    "$PY" "$TRAIN" "${TRAIN_ARGS[@]}"
    if [ $? -ne 0 ]; then
        echo "Warm-up for arm $letter FAILED -- continuing anyway"
    fi
done

# --- Fix 1: build the full grid as a flat list, then shuffle it. ---
# Nested loops always iterate in the same fixed order every run. Flattening first and shuffling
# means any residual cache-warming effect (favoring later positions) becomes noise uncorrelated
# with (nw, pf), instead of a systematic bias favoring whichever combo happens to be iterated
# last.
combos=()
for letter in "${EXP_ARMS[@]}"; do
    for nw in ${ARM_NUM_WORKERS[$letter]}; do
        for pf in "${PREFETCH_FACTOR_LIST[@]}"; do
            combos+=("${letter}:${nw}:${pf}")
        done
    done
done
combos=($(printf '%s\n' "${combos[@]}" | shuf))

echo "=== Timed sweep: ${#combos[@]} configs, shuffled order ==="
RESULTS_CSV=training_parameter_search_results.csv
echo "n_days_observation,num_workers,prefetch_factor,seconds" > "$RESULTS_CSV"

for combo in "${combos[@]}"; do
    IFS=':' read -r letter nw pf <<< "$combo"
    run_name="throughput_${letter}0_nw${nw}_pf${pf}"
    get_train_args "$letter" "$nw" "$pf" "$run_name"

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
    echo "$letter,$nw,$pf,$seconds" >> "$RESULTS_CSV"
done

echo "=== Results (sorted by seconds) ==="
{ head -1 "$RESULTS_CSV"; tail -n +2 "$RESULTS_CSV" | sort -t, -k4 -n; } | column -s, -t
echo "Full results saved to $RESULTS_CSV"
