#!/bin/bash
# EXP-06, 2026-08-27: does removing the AdamW-eps throttle recover the reproduction gap?
#
# Background (administrative/EXPERIMENTS.md, "FOUND 2026-08-26"): BaseModel.py normalises
# pos_class_weight to w/(1+w) so torchvision's sigmoid_focal_loss gets an alpha in [0,1]. That
# fixes the positive:negative ratio but shrinks the WHOLE loss by ~1/(1+w) ~ 1/626, which pushes
# sqrt(v_hat) below AdamW's default eps=1e-8 for 90.5% of parameters and throttles their updates
# to a mean 22.6% of the scale-invariant ideal.
#
# TWO ARMS, deliberately run SEPARATELY rather than combined:
#   (a) A0eps12  --optimizer.init_args.eps=1e-12   -> shrink the denominator of sqrt(v_hat)/eps
#   (b) A0unit   --model.init_args.focal_loss_scale=unit -> grow the numerator by (1+w)
#
# They are two levers on the SAME ratio, not two independent fixes: applying both would be
# redundant and unattributable. Run apart, they are a consistency check on the diagnosis --
# if (a) and (b) land at the same test_AP, the mechanism really is sqrt(v_hat)/eps. If they
# diverge materially, the diagnosis is incomplete and that is the informative outcome.
#
# Arm (a) is the cleaner PROBE (no code path changes, loss/ratio/paper-fidelity untouched).
# Arm (b) is the candidate FIX (default eps, loss magnitude comparable to Dice/BCE, and the
# literal "alpha = inverse positive frequency" reading of WSTS+). Do not conflate them.
#
# Folds [1,5,6,11] = the pre-committed subset covering all four test years exactly once
# (2020, 2019, 2021, 2018). A0 baselines already exist on all four at bs=64, so every run here
# is a PAIRED comparison against an existing number -- no new control is needed.
#
# Arm order: all of (a), then all of (b). If the night is cut short, a complete (a) arm is
# worth more than two half-arms.
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")" || exit 1

export MLFLOW_ALLOW_FILE_STORE=true
PY=/home/miles/miniconda3/envs/WSTS_env/bin/python
FOLDS=(1 5 6 11)
mkdir -p logs

COMMON=(
    --config=cfgs/models/unet_res18.yaml
    --trainer=cfgs/trainer_single_gpu_mlflow.yaml
    --optimizer.init_args.lr=1e-3
    --data=cfgs/data_base.yaml --data=cfgs/arms/A0.yaml
    --data.num_workers=8 --data.prefetch_factor=2
    --do_train=True --do_test=True
    --trainer.max_steps=10000
)

launch () {   # $1 = arm label, $2 = fold, $3.. = arm-specific args
    local arm="$1" fold="$2"; shift 2
    local name="${arm}_fold${fold}_seed0_bs64"
    echo "=== [$(date '+%m-%d %H:%M:%S')] START $name  extra: $*"
    local t0=$(date +%s)
    $PY src/train.py "${COMMON[@]}" \
        --data.data_fold_id="$fold" \
        --trainer.logger.init_args.run_name="$name" \
        "$@" > "logs/${name}.log" 2>&1
    local rc=$?
    echo "=== [$(date '+%m-%d %H:%M:%S')] DONE  $name  exit=$rc  ($(( ($(date +%s) - t0) / 60 )) min)"
    return $rc
}

echo "########## ARM (a): eps=1e-12, focal scale unchanged ##########"
for f in "${FOLDS[@]}"; do
    launch A0eps12 "$f" --optimizer.init_args.eps=1e-12
done

echo "########## ARM (b): unit focal scale, eps at its 1e-8 default ##########"
for f in "${FOLDS[@]}"; do
    launch A0unit "$f" --model.init_args.focal_loss_scale=unit
done

echo "########## [$(date)] ALL DONE ##########"
echo "Compare with:  python scripts/diagnose_arm_gap.py --arms A0 A0eps12"
echo "               python scripts/diagnose_arm_gap.py --arms A0 A0unit"
echo "               python scripts/diagnose_arm_gap.py --arms A0eps12 A0unit"
