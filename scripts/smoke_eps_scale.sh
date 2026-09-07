#!/bin/bash
# Pre-flight smoke test for the 2026-08-26 AdamW-eps / focal-scale investigation.
#
# Verifies, in ~3 minutes and BEFORE committing ~10 GPU-hours, that:
#   1. jsonargparse actually accepts --optimizer.init_args.eps (it is not in any config file,
#      only in torch.optim.AdamW's signature).
#   2. --model.init_args.focal_loss_scale=unit reaches BaseModel through SMPModel's **kwargs.
#   3. "unit" really does raise train_loss by ~(1 + pos_class_weight) ~ 626x. If the logged
#      losses came back identical the flag silently did nothing, and the whole comparison
#      would have been a null result for the wrong reason.
#
# Args are held in a bash ARRAY, not a string: the experiment name contains spaces and a
# plain string gets word-split into unrecognized arguments (cost: one failed smoke run).
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

COMMON=(
    --config=cfgs/models/unet_res18.yaml
    --trainer=cfgs/trainer_single_gpu_mlflow.yaml
    --optimizer.init_args.lr=1e-3
    --data=cfgs/data_base.yaml --data=cfgs/arms/A0.yaml
    --data.data_fold_id=1 --data.num_workers=8 --data.prefetch_factor=2
    --do_train=True --do_test=False --trainer.max_steps=60
    "--trainer.logger.init_args.experiment_name=Testing computer settings for training"
)

run () {  # $1 = run name, $2.. = extra args
    local name="$1"; shift
    echo "=== [$(date +%H:%M:%S)] $name  extra: $* ==="
    $PY src/train.py "${COMMON[@]}" --trainer.logger.init_args.run_name="$name" "$@" \
        > "logs/${name}.log" 2>&1
    echo "  exit=$?"
}

run "smoke_baseline_${STAMP}"
run "smoke_eps12_${STAMP}"  --optimizer.init_args.eps=1e-12
run "smoke_unit_${STAMP}"   --model.init_args.focal_loss_scale=unit

echo
echo "=== logged first train_loss_step per run ==="
$PY - "$STAMP" <<'PY'
import os, sys
stamp = sys.argv[1]
for exp in os.listdir("mlruns"):
    d = os.path.join("mlruns", exp)
    if not os.path.isdir(d) or exp == "models":
        continue
    for r in os.listdir(d):
        t = os.path.join(d, r, "tags", "mlflow.runName")
        if not os.path.exists(t):
            continue
        n = open(t).read().strip()
        if not (n.startswith("smoke_") and n.endswith(stamp)):
            continue
        m = os.path.join(d, r, "metrics", "train_loss_step")
        sp = os.path.join(r"", d, r, "params", "model.init_args.focal_loss_scale")
        v = open(m).readline().split()[1] if os.path.exists(m) else "NO METRIC"
        sc = open(sp).read().strip() if os.path.exists(sp) else "(absent)"
        print(f"  {n:36s} focal_loss_scale={sc:11s} first train_loss={v}")
PY
