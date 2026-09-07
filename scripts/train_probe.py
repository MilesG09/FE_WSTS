#!/usr/bin/env python
"""A Lightning callback that records enough of a training run to compare two of them exactly.

The dataset-level gates compare `(x, y)` tensors. This closes the loop at the other end:
it records the loss at every optimiser step and a hash of the model weights at the end,
so two runs can be compared through the ENTIRE path -- dataloader, collate, model,
optimiser -- not just at the dataloader's output.

Weight equality after N steps is the strongest available statement. Any difference in any
input at any step, however small, propagates into the weights and shows up here; nothing
that reaches the model can hide from it.

Wire it in from the CLI without touching train.py:

    WSTS_PROBE_OUT=/tmp/run.json python src/train.py ... \
        --trainer.callbacks+=scripts.train_probe.TrainProbe

Writes JSON to $WSTS_PROBE_OUT (default ./train_probe.json).
"""

import hashlib
import json
import os

import torch
from pytorch_lightning.callbacks import Callback


class TrainProbe(Callback):
    def __init__(self, out_path: str = None):
        super().__init__()
        self.out_path = out_path or os.environ.get(
            "WSTS_PROBE_OUT", "train_probe.json")
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None:
            # float() not .item() rounding: the full float32 value, repr'd exactly by
            # json, so a 1-ulp difference between two runs is visible rather than hidden
            # by display precision.
            self.losses.append(float(loss.detach().cpu()))

    def on_train_end(self, trainer, pl_module):
        h = hashlib.sha256()
        # Sorted by name so the digest does not depend on dict iteration order.
        for name, p in sorted(pl_module.state_dict().items()):
            h.update(name.encode())
            h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
        payload = {
            "n_steps": len(self.losses),
            "losses": self.losses,
            "weight_sha256": h.hexdigest(),
        }
        with open(self.out_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[train_probe] wrote {self.out_path}: {len(self.losses)} steps, "
              f"weights sha256={h.hexdigest()[:16]}")
