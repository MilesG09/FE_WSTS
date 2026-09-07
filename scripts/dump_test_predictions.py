"""Run a saved checkpoint over a fold's test set once and dump (logits, targets) to .npz.

WHY: several questions are pure functions of the model's predictions and need no retraining --
"what is my real tie density?", "how extreme do my logits get?", "does a different AP
implementation give a different number on MY data?", "what does the PR curve actually look
like?". Producing the predictions once and interrogating them offline is far cheaper than
re-running training for each question.

This file exists mainly to hand you the plumbing you have not written before:
  (1) restoring a LightningModule from a checkpoint, and
  (2) building the same test dataloader the training run used.
The judgement calls are left as TODOs.

Usage:
    python scripts/dump_test_predictions.py \
        --ckpt "mlruns/362456860332584908/<run_id>/artifacts/epoch=.../epoch=....ckpt" \
        --fold 1 --n-lead 1 --out preds_A0_fold1.npz

Find a checkpoint path with:
    find mlruns/362456860332584908 -name "*.ckpt" | grep -i a0_fold1   # or read the run dir
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dataloader.FireSpreadDataModule import FireSpreadDataModule  # noqa: E402
from models.SMPModel import SMPModel  # noqa: E402

# These mirror cfgs/data_base.yaml. They are duplicated here rather than parsed from the YAML
# so the script has no jsonargparse dependency -- but that means they can DRIFT from the config.
# If cfgs/data_base.yaml changes, this must change too.
DATA_DIR = "/home/miles/FE_WSTS/hdf5_data"
FEATURES_TO_KEEP = [0, 1, 2, 3, 4, 38, 39]
CROP_SIDE_LENGTH = 128
TEST_ADJUSTMENT = 5


def build_test_loader(fold, n_lead, num_workers):
    """Reproduce the exact test dataloader a training run used for this fold and arm.

    Two things make the test split different from train/val, and both matter here:
      - `is_train=False` inside FireSpreadDataset: no random cropping, so images come through at
        full size. That is why test runs at batch_size=1 (see FireSpreadDataModule docstring):
        different fires have different image dimensions and cannot be batched together.
      - `n_leading_observations_test_adjustment=5` is applied to the TEST set only, so the test
        set is identical across arms with different n_leading_observations. (Validation is NOT
        adjusted by default -- that asymmetry is the confound written up in EXPERIMENTS.md,
        "Confound found 2026-08-20".)
    """
    dm = FireSpreadDataModule(
        data_dir=DATA_DIR,
        batch_size=1,
        n_leading_observations=n_lead,
        n_leading_observations_test_adjustment=TEST_ADJUSTMENT,
        crop_side_length=CROP_SIDE_LENGTH,
        load_from_hdf5=True,
        num_workers=num_workers,
        remove_duplicate_features=True,
        features_to_keep=FEATURES_TO_KEEP,
        data_fold_id=fold,
    )
    dm.setup("test")
    return dm.test_dataloader()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-lead", type=int, required=True,
                   help="1 for an A arm, 2 for B, 5 for C -- must match the run's arm config")
    p.add_argument("--out", required=True)
    p.add_argument("--max-pixels", type=float, default=5e7,
                   help="stop once this many pixels are collected; see TODO 2")
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    # (1) Restore the model. load_from_checkpoint reads the hyper_parameters dict that
    # save_hyperparameters() stored, so encoder_name / n_channels / loss_function / etc. all come
    # back from the checkpoint itself -- you do NOT re-specify them, and you must not, or you can
    # silently build a different architecture than the weights were trained for.
    model = SMPModel.load_from_checkpoint(args.ckpt, map_location="cuda")
    model.eval()
    model.freeze()
    print(f"restored {type(model).__name__}: n_channels={model.hparams.n_channels}, "
          f"loss={model.hparams.loss_function}, step={torch.load(args.ckpt, map_location='cpu', weights_only=False).get('global_step')}")

    loader = build_test_loader(args.fold, args.n_lead, args.num_workers)
    print(f"test set: {len(loader.dataset)} samples (fold {args.fold}, n_lead {args.n_lead})")

    logits_chunks, target_chunks, total = [], [], 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, y = batch[0].cuda(), batch[1]
            # BaseModel.test_step logs `self.test_avg_precision(y_hat, y)` with RAW LOGITS --
            # no sigmoid. Keep it that way so this dump reproduces what the pipeline measures.
            y_hat = model(x).squeeze(1).float().cpu()
            logits_chunks.append(y_hat.flatten().numpy().astype(np.float32))
            target_chunks.append(y.flatten().numpy().astype(np.int8))
            total += logits_chunks[-1].size
            if i % 50 == 0:
                print(f"  batch {i}/{len(loader)}  pixels so far {total:,}")
            if total >= args.max_pixels:
                print(f"  stopping at {total:,} pixels (--max-pixels)")
                break

    scores = np.concatenate(logits_chunks)
    labels = np.concatenate(target_chunks)
    np.savez_compressed(args.out, scores=scores, labels=labels)

    uniq = len(np.unique(scores))
    print(f"\nsaved {args.out}")
    print(f"  pixels      : {scores.size:,}")
    print(f"  positives   : {int(labels.sum()):,} ({labels.mean():.5%})")
    print(f"  logit range : [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  distinct    : {uniq:,}")
    print(f"  TIE DENSITY : {1 - uniq / scores.size:.6f}")


# ---------------------------------------------------------------------------------------
# TODO (yours):
#
# TODO 1 -- WHICH CHECKPOINT, AND WHY THAT ONE? Any A0 fold works for measuring tie density and
#   logit range. But if you want to re-derive a reported test_AP and check it against MLflow,
#   the checkpoint has to be the one ModelCheckpoint actually selected. Convince yourself the
#   path you pass is that one before trusting any number you compute from it.
#
# TODO 2 -- --max-pixels TRUNCATES THE TEST SET. That is fine for measuring tie density and
#   logit range (properties of the score distribution), and NOT fine for reproducing test_AP
#   (a property of the whole set). Which of your questions survive truncation and which do not?
#   Decide before you use a number from a truncated dump.
#   Related: truncation takes the FIRST k batches, not a random sample. Why might that be worse
#   than random for this dataset specifically? (Look at how the test set is ordered.)
#
# TODO 3 -- THE REAL PAYOFF. Once you have `logit range` and `TIE DENSITY`, go back to
#   scripts/ap_implementation_sweep.py and check whether the synthetic sweep actually covered
#   your operating point. If your logits reach |x| > 17, float32 sigmoid saturates completely
#   and the sweep's shift=4 never tested that regime -- rerun it with --shift set to match.
#
# TODO 4 -- CONFIG DRIFT. DATA_DIR / FEATURES_TO_KEEP / CROP_SIDE_LENGTH / TEST_ADJUSTMENT are
#   hardcoded above and duplicate cfgs/data_base.yaml. That is a second source of truth, which
#   is exactly the failure mode that produced the batch-size confound in EXP-01/02. Either add
#   an assertion that these match the checkpoint's logged `command` tag, or read them from the
#   YAML. Which is the better fix, and why?
#
# TODO 5 -- The test dataloader here is built from scratch rather than from the run's logged
#   config. What could differ between this loader and the one the original run used, and which
#   of those differences would change the predictions vs. merely change their order?
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
