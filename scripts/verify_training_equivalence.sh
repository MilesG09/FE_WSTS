#!/bin/bash
# End-to-end proof that the dataloader speed-ups do not change TRAINING.
#
# The other gates compare (x, y) at the dataloader's output. This compares the far end:
# two real training runs, identical arguments and seed, one with every WSTS_DISABLE_*
# flag set and one with none, then diffs the per-step losses and a hash of the final
# model weights. Any difference in any input at any step propagates into the weights,
# so matching weights after N optimiser steps is the strongest statement available --
# nothing that reaches the model can hide from it.
#
# num_workers=0 by default and deliberately: with workers, augmentation RNG lives in
# forked processes whose seeding depends on worker count and iterator lifetime, so two
# runs would legitimately differ for reasons that have nothing to do with the
# optimisations. Single-process makes the comparison exact.
#
# Usage: scripts/verify_training_equivalence.sh [max_steps] [num_workers]
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

STEPS=${1:-30}
NW=${2:-0}
PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
DATA_DIR=${WSTS_DATA_DIR:-/home/miles/research/old_repo_FE_WSTS/hdf5_data}
OUTDIR=$(mktemp -d)

export PYTHONPATH="$PWD:$PWD/src"
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_MODE=offline
export WANDB_SILENT=true
# cuDNN autotuning picks algorithms by timing, so it can select different (equally
# valid) kernels between runs and produce last-bit differences that have nothing to do
# with the data. Pin it, or this test measures nondeterminism instead of equivalence.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_one() {  # $1 = label, $2 = disabled|enabled
    local label=$1 mode=$2
    if [ "$mode" = "disabled" ]; then
        export WSTS_DISABLE_HDF5_READ_OPT=1 WSTS_DISABLE_CROP_OPT=1 WSTS_DISABLE_ONEHOT_SKIP=1
    else
        unset WSTS_DISABLE_HDF5_READ_OPT WSTS_DISABLE_CROP_OPT WSTS_DISABLE_ONEHOT_SKIP
    fi
    WSTS_PROBE_OUT="$OUTDIR/$label.json" "$PY" src/train.py \
        -c cfgs/unet/res18_monotemporal.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --data cfgs/data_monotemporal_veg_features.yaml \
        --data.data_dir="$DATA_DIR" \
        --data.data_fold_id=0 \
        --data.num_workers="$NW" \
        --seed_everything=0 \
        --trainer.max_steps="$STEPS" \
        --trainer.limit_val_batches=0 \
        --trainer.num_sanity_val_steps=0 \
        --trainer.deterministic=true \
        --trainer.benchmark=false \
        --trainer.default_root_dir="$OUTDIR" \
        --trainer.enable_progress_bar=false \
        --trainer.callbacks+=scripts.train_probe.TrainProbe \
        --do_train=true --do_test=false --do_predict=false \
        > "$OUTDIR/$label.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  $label FAILED (exit $rc). Tail of log:"
        tail -20 "$OUTDIR/$label.log"
        return $rc
    fi
    echo "  $label ok"
}

echo "=== training equivalence: max_steps=$STEPS num_workers=$NW ==="
echo "=== logs in $OUTDIR ==="
run_one original  disabled || exit 1
run_one optimised enabled  || exit 1

"$PY" - "$OUTDIR/original.json" "$OUTDIR/optimised.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
print()
print(f"  steps recorded          : {a['n_steps']} vs {b['n_steps']}")
same_losses = a["losses"] == b["losses"]
same_weights = a["weight_sha256"] == b["weight_sha256"]
print(f"  per-step losses identical: {same_losses}")
if not same_losses:
    for i, (x, y) in enumerate(zip(a["losses"], b["losses"])):
        if x != y:
            print(f"    first divergence at step {i}: {x!r} vs {y!r}")
            break
print(f"  final weight sha256      : {a['weight_sha256'][:32]}")
print(f"                             {b['weight_sha256'][:32]}")
print(f"  weights identical        : {same_weights}")
print()
if a["n_steps"] == 0:
    print("FAIL - no steps recorded; the probe callback did not run.")
    sys.exit(2)
if same_losses and same_weights:
    print("PASS - the optimisations change nothing about training.")
    sys.exit(0)
print("FAIL - training diverges. Do not train on this.")
sys.exit(1)
PYEOF
rc=$?
echo "=== exit $rc ==="
exit $rc
