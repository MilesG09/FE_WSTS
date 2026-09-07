#!/bin/bash
# Tail one or more sweep log files and surface only high-signal lines (stage banners, errors,
# OOM/kill signatures, key results) -- built so a Monitor can babysit an overnight training run
# without drowning in per-step training output. Reusable across future sweeps, not tied to one
# run's log paths.
#
# Usage: watch_sweep_logs.sh <logfile> [logfile ...]
set -uo pipefail

### NOTE: Lightning's tqdm progress bar writes one line per *step* with every metric (incl.
### val_avg_precision) embedded as postfix text -- do NOT add per-step metric names here, that
### floods the filter (tens of thousands of matching lines per LR run) and gets a Monitor
### auto-stopped for excess output. Stage banners ("===") plus failure signatures only.
###
### Also drops "waiting for settled memory before <run_name>" -- fires once per grid combo
### (dozens of times a night) with no signal in it. The distinct "WARNING: memory did not
### settle" line (wait_for_memory_settled's failure path) is kept.
tail -F "$@" 2>&1 | grep -E --line-buffered \
    "^===|Traceback|CUDA out of memory|Killed|CalledProcessError|Exception|FAILED|[Ee]rror|memory did not settle|Best lr:" \
    | grep -v --line-buffered "waiting for settled memory before"
