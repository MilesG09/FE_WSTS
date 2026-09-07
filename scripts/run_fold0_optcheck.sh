#!/bin/bash
# Full-fold confirmation that the 2026-09-04 dataloader speed-ups change nothing.
#
# Reproduces the fold 0 training run of 2026-09-01 (checkpoint
# detrun/ckpt/fold0/.../best-epoch=126-val_avg_precision=0.46.ckpt, tested to
# test_AP = 0.549997) with EVERY RNG-affecting difference neutralised, so the only
# thing that differs is the three dataset optimisations:
#
#   --data.persistent_workers=false   the new default is True, and it is the one setting
#                                     that is NOT output-neutral: PyTorch seeds each
#                                     worker's numpy RNG from a base_seed taken when the
#                                     iterator is created, so persistent workers carry one
#                                     stream across epochs while respawned workers get a
#                                     fresh seed each epoch. Epoch 0 matches either way;
#                                     everything after diverges. False == the old code.
#   --data.prefetch_factor=2          explicit, equals the old DataLoader default (verified
#                                     output-neutral over whole epochs, but pinned anyway
#                                     so nothing is left to a default that could change).
#   --data.num_workers=8              worker count sets how many RNG streams exist, so it
#                                     must match the original run exactly.
#   data_dir with the duplicate       2019/WildfireSpreadTS.hdf5 is deliberately KEPT: the
#                                     baseline number was produced with it, and changing
#                                     the dataset at the same time would confound the test.
#
# PASS = test_AP is EXACTLY 0.549997 and the best checkpoint is epoch 126. Anything else
# means the optimisations are not behaviour-preserving at full scale, and no sweep should
# run until it is understood.
set -uo pipefail
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

OUT=logs/veg_row2_wandb/optcheck/fold0
PY=/home/miles/miniconda3/envs/WSTS_original/bin/python
export PYTHONPATH="$PWD:$PWD/src"
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_MODE=offline
export WANDB_SILENT=true

echo "=== [$(date)] fold 0 optimisation check starting ==="
echo "=== baseline to match: test_AP=0.549997, best epoch=126 ==="

/usr/bin/time -f "WALLCLOCK wall=%E user=%U sys=%S maxrss=%MKB" \
"$PY" src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_veg_features.yaml \
  --data.data_dir=/home/miles/research/old_repo_FE_WSTS/hdf5_data \
  --data.data_fold_id=0 \
  --data.num_workers=8 \
  --data.persistent_workers=false \
  --data.prefetch_factor=2 \
  --data.is_pad=false \
  --data.ignition_only_val_test=false \
  --data.ignition_only_train=false \
  --seed_everything=0 \
  --trainer.deterministic=true \
  --trainer.check_val_every_n_epoch=1 \
  --trainer.max_steps=10000 \
  --trainer.default_root_dir="$PWD/$OUT" \
  --trainer.logger.init_args.name=res18unet_veg_t1_fold0_optcheck \
  --do_train=true --do_test=true --do_predict=false

echo "=== [$(date)] fold 0 optimisation check finished (exit $?) ==="
