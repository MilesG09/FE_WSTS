#!/bin/bash
# Overnight orchestration, 2026-08-17: run the (num_workers, prefetch_factor) throughput search
# for arms A/C, pick arm A's fastest combo, then feed it straight into the A0 LR sweep. Written
# as one script (not a chain of separately-launched commands) so the whole two-stage pipeline
# keeps running as a single detached background process even if the Claude session that
# launched it gets interrupted -- see administrative/CHANGELOG.md.
#
# Both sub-scripts already tolerate individual run failures/OOM internally (log the failure,
# continue to the next combo/lr) -- this wrapper only needs to react if a whole *stage* dies
# outright, which it does by refusing to proceed to the next stage and leaving a clear status
# marker rather than guessing.
set -uo pipefail

cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

STAMP=20260817_221811
LOG_DIR=logs
mkdir -p "$LOG_DIR"
SEARCH_LOG="$LOG_DIR/training_parameter_search_${STAMP}.log"
LR_LOG="$LOG_DIR/lr_sweep_${STAMP}.log"
STATUS_FILE="$LOG_DIR/overnight_sweep_status_${STAMP}.txt"

write_status() { echo "$1" > "$STATUS_FILE"; }

echo "search_log=$SEARCH_LOG"
echo "lr_log=$LR_LOG"
echo "status_file=$STATUS_FILE"

write_status "stage=search_running started=$(date -Iseconds)"
echo "=== [$(date)] Starting training_parameter_search.sh ==="
bash scripts/training_parameter_search.sh >"$SEARCH_LOG" 2>&1
search_exit=$?
echo "=== [$(date)] training_parameter_search.sh exited $search_exit ==="

if [ $search_exit -ne 0 ]; then
    write_status "stage=FAILED_search exit_code=$search_exit time=$(date -Iseconds)"
    exit 1
fi

if [ ! -f training_parameter_search_results.csv ]; then
    write_status "stage=FAILED_search_no_csv time=$(date -Iseconds)"
    exit 1
fi

# Winning A0 config = min `seconds` among arm-A rows (column 1 holds the arm letter despite the
# CSV header's stale "n_days_observation" label -- see training_parameter_search.sh, that
# label was never updated when the script switched from sweeping n_leading_observations to
# sweeping arm letters; harmless since this reads by position, but worth fixing someday).
read -r nw pf secs < <(awk -F, '$1=="A" {print $2, $3, $4}' training_parameter_search_results.csv | sort -t' ' -k3,3 -n | head -1)

if [ -z "${nw:-}" ] || [ -z "${pf:-}" ]; then
    write_status "stage=FAILED_no_successful_A_rows time=$(date -Iseconds)"
    exit 1
fi

echo "=== Winning A0 config: num_workers=$nw prefetch_factor=$pf (${secs}s/500 steps) ==="
write_status "stage=search_done winning_num_workers=$nw winning_prefetch_factor=$pf winning_seconds=$secs time=$(date -Iseconds)"

write_status "stage=lr_running num_workers=$nw prefetch_factor=$pf started=$(date -Iseconds)"
echo "=== [$(date)] Starting lr_sweep.sh --num_workers $nw --prefetch_factor $pf ==="
bash scripts/lr_sweep.sh --num_workers "$nw" --prefetch_factor "$pf" >"$LR_LOG" 2>&1
lr_exit=$?
echo "=== [$(date)] lr_sweep.sh exited $lr_exit ==="

if [ $lr_exit -ne 0 ]; then
    write_status "stage=FAILED_lr exit_code=$lr_exit num_workers=$nw prefetch_factor=$pf time=$(date -Iseconds)"
    exit 1
fi

write_status "stage=all_done num_workers=$nw prefetch_factor=$pf time=$(date -Iseconds)"
echo "=== [$(date)] Overnight sweep complete: num_workers=$nw prefetch_factor=$pf ==="
