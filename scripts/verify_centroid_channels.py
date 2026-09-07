#!/usr/bin/env python
"""Verify _compute_centroid_channels and the peek-back loading path against known answers.

Why this exists: the centroid feature now has four independent flags AND emits one channel per
family per model-visible timestep, so the number of valid input layouts is 16 x n_lead. The count
src/train.py derives from config alone has to match the tensor the dataloader actually produces
for every one of them. That is far past the point where eyeballing a training run's first forward
pass is a real check.

Why most of it does NOT touch HDF5: the centroid math is a pure function of the fire-mask channel.
A synthetic (T, C, H, W) tensor with fire at chosen pixels makes every expected value computable
by hand, which a real sample never does. Sections 1-6 run in under a second with no data
directory, so there is no excuse not to run them.

The dataset object is built with object.__new__ to skip __init__ entirely -- __init__ globs a data
directory and reads normalization stats, none of which the centroid math depends on.

Usage:
    python scripts/verify_centroid_channels.py                # synthetic + end-to-end
    python scripts/verify_centroid_channels.py --skip-data    # synthetic only, no data dir needed
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402

H = W = 8
C = 24  # arbitrary; only the LAST channel (the binary fire mask) is ever read
FLAGS = ("use_centroid_position", "use_centroid_velocity",
         "use_centroid_position_validity", "use_centroid_velocity_validity")

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        failures.append(label)


def make_ds(n_timesteps, **flags):
    """A FireSpreadDataset carrying only what _compute_centroid_channels reads."""
    ds = object.__new__(FireSpreadDataset)
    for f in FLAGS:
        setattr(ds, f, flags.get(f, False))
    ds._n_peek_frames = 1 if (flags.get("use_centroid_velocity")
                              or flags.get("use_centroid_velocity_validity")) else 0
    ds._centroid_channel_names = FireSpreadDataset.centroid_channel_names(
        *[flags.get(f, False) for f in FLAGS], n_timesteps=n_timesteps)
    ds._use_any_centroid = len(ds._centroid_channel_names) > 0
    return ds


def make_x(fire_pixels):
    """(T_total, C, H, W); fire_pixels[t] is (row, col), or None for a fire-free frame.

    A None frame doubles as the zero-padded peek row that load_imgs prepends when the window
    starts at the fire's first observed day -- the two cases are identical by design.
    """
    x = torch.zeros(len(fire_pixels), C, H, W)
    for t, px in enumerate(fire_pixels):
        if px is not None:
            x[t, -1, px[0], px[1]] = 1.0
    return x


def emitted(ds, x):
    """channel name -> its constant scalar value, plus the raw tensor."""
    out = ds._compute_centroid_channels(x)
    values = {name: float(out[i, 0, 0]) for i, name in enumerate(ds._centroid_channel_names)}
    return out, values


def expect(values, wanted):
    for name, want in wanted.items():
        check(f"{name} == {want}", abs(values[name] - want) < 1e-6,
              f"got {values.get(name)}")


ALL_ON = {f: True for f in FLAGS}

# ------------------------------------------------------------------ 1. known answers, T=1
# One peek frame + one model frame. Fire is a single pixel, (row 2, col 1) -> (row 6, col 5).
#   centroid_x_lag0  = 6 / (H/2) - 1 = 6/4 - 1 = 0.50
#   centroid_y_lag0  = 5 / (W/2) - 1 = 5/4 - 1 = 0.25
#   centroid_vx_lag0 = (6 - 2) / (H/4) = 4/2   = 2.00
#   centroid_vy_lag0 = (5 - 1) / (W/4) = 4/2   = 2.00
print("\n[1] T=1 with peek: single fire pixel (2,1) -> (6,5)")
_, v = emitted(make_ds(1, **ALL_ON), make_x([(2, 1), (6, 5)]))
expect(v, {"centroid_x_lag0": 0.50, "centroid_y_lag0": 0.25,
           "centroid_vx_lag0": 2.00, "centroid_vy_lag0": 2.00,
           "position_valid_lag0": 1.0, "velocity_valid_lag0": 1.0})

# ------------------------------------------------------------------ 2. known answers, T=2
# Peek (1,1), then model frames (2,1) and (6,5). lag1 is the older model frame, lag0 the newer.
#   centroid_x_lag1  = 2/4 - 1 = -0.50      centroid_x_lag0  = 6/4 - 1 =  0.50
#   centroid_y_lag1  = 1/4 - 1 = -0.75      centroid_y_lag0  = 5/4 - 1 =  0.25
#   centroid_vx_lag1 = (2-1)/2 =  0.50      centroid_vx_lag0 = (6-2)/2 =  2.00
#   centroid_vy_lag1 = (1-1)/2 =  0.00      centroid_vy_lag0 = (5-1)/2 =  2.00
print("\n[2] T=2 with peek: (1,1) -> (2,1) -> (6,5); per-timestep values")
_, v = emitted(make_ds(2, **ALL_ON), make_x([(1, 1), (2, 1), (6, 5)]))
expect(v, {"centroid_x_lag1": -0.50, "centroid_x_lag0": 0.50,
           "centroid_y_lag1": -0.75, "centroid_y_lag0": 0.25,
           "centroid_vx_lag1": 0.50, "centroid_vx_lag0": 2.00,
           "centroid_vy_lag1": 0.00, "centroid_vy_lag0": 2.00,
           "position_valid_lag1": 1.0, "velocity_valid_lag1": 1.0})

# ------------------------------------------------------------------ 3. first day of the fire
print("\n[3] peek row absent (zero-padded): position real, velocity gated, flags disagree")
_, v = emitted(make_ds(1, **ALL_ON), make_x([None, (6, 5)]))
expect(v, {"centroid_x_lag0": 0.50, "centroid_vx_lag0": 0.00, "centroid_vy_lag0": 0.00,
           "position_valid_lag0": 1.0, "velocity_valid_lag0": 0.0})

# ------------------------------------------------------------------ 4. fire-free model frame
print("\n[4] model frame fire-free: position falls back to patch centre (0.0), both flags 0")
_, v = emitted(make_ds(1, **ALL_ON), make_x([(2, 1), None]))
expect(v, {"centroid_x_lag0": 0.00, "centroid_y_lag0": 0.00, "centroid_vx_lag0": 0.00,
           "position_valid_lag0": 0.0, "velocity_valid_lag0": 0.0})

# ------------------------------------------------------------------ 5. every layout x every T
print("\n[5] 16 flag combinations x n_timesteps in {1,2,5}: count, shape, constancy, contiguity")
for n_t in (1, 2, 5):
    for combo in itertools.product([False, True], repeat=4):
        flags = dict(zip(FLAGS, combo))
        bits = "".join("1" if c else "0" for c in combo)
        names = FireSpreadDataset.centroid_channel_names(*combo, n_timesteps=n_t)
        n = FireSpreadDataset.n_centroid_channels(*combo, n_timesteps=n_t)
        if n == 0:
            check(f"T={n_t} {bits} emits nothing (producer not called)", names == [])
            continue
        n_peek = 1 if (flags["use_centroid_velocity"]
                       or flags["use_centroid_velocity_validity"]) else 0
        x = make_x([(t % H, (2 * t) % W) for t in range(n_t + n_peek)])
        out, _ = emitted(make_ds(n_t, **flags), x)
        ok = (tuple(out.shape) == (n, H, W)
              and out.is_contiguous()
              # each channel is a broadcast scalar, so it must be spatially constant
              and bool(((out - out[:, :1, :1]).abs() < 1e-6).all()))
        check(f"T={n_t} {bits} -> {n} channels", ok,
              f"shape={tuple(out.shape)} contiguous={out.is_contiguous()}")

# ------------------------------------------------------------------ 6. the naming contract
print("\n[6] naming: family-major, oldest-first, lag0 == most recent frame")
check("T=1 all-on is one channel per family in SPEC order",
      FireSpreadDataset.centroid_channel_names(True, True, True, True, n_timesteps=1)
      == ["centroid_x_lag0", "centroid_y_lag0", "centroid_vx_lag0", "centroid_vy_lag0",
          "position_valid_lag0", "velocity_valid_lag0"])
check("T=3 position-only is family-major with descending lag",
      FireSpreadDataset.centroid_channel_names(True, False, False, False, n_timesteps=3)
      == ["centroid_x_lag2", "centroid_x_lag1", "centroid_x_lag0",
          "centroid_y_lag2", "centroid_y_lag1", "centroid_y_lag0"])
check("count scales linearly with n_timesteps",
      all(FireSpreadDataset.n_centroid_channels(True, True, True, True, n_timesteps=t) == 6 * t
          for t in (1, 2, 5)))

# ------------------------------------------------ 7. end-to-end channel count (P1 + P3 + P4)
# The only check that tests what P4 claims: that src/train.py can predict the dataloader's
# channel count from config alone, before any dataset object exists. Sections 1-6 only prove
# the producer is self-consistent with itself.
#
# Both n_lead values are exercised because they take DIFFERENT code paths (P3): at n_lead > 1
# the dataset returns 3-D (C, H, W), flatten_and_remove_duplicate_features_ having run; at
# n_lead = 1 it returns 4-D (1, C, H, W) and the centroid channels must join on axis 1, not
# axis 0. Velocity is now legal at n_lead=1 because load_imgs peeks one frame further back.
CFG_PATH = Path(__file__).resolve().parents[1] / "cfgs" / "data_base.yaml"


def build(n_lead, flags):
    import yaml
    cfg = yaml.safe_load(CFG_PATH.read_text())
    return cfg, FireSpreadDataset(
        data_dir=cfg["data_dir"], included_fire_years=[2020], n_leading_observations=n_lead,
        crop_side_length=cfg["crop_side_length"], load_from_hdf5=cfg["load_from_hdf5"],
        is_train=True, remove_duplicate_features=cfg["remove_duplicate_features"],
        stats_years=(2018, 2019), n_leading_observations_test_adjustment=None,
        features_to_keep=cfg["features_to_keep"], return_doy=False, **flags)


def deterministic_getitem(ds, idx, seed=0):
    """__getitem__ crops and augments randomly; pin every RNG it can reach."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ds[idx]


import yaml  # noqa: E402

if "--skip-data" in sys.argv:
    print("\n[7] SKIPPED (--skip-data)")
    print("[8] SKIPPED (--skip-data)")
elif not Path(yaml.safe_load(CFG_PATH.read_text())["data_dir"]).exists():
    print("\n[7] SKIPPED -- data_dir not found")
    print("[8] SKIPPED -- data_dir not found")
else:
    for n_lead in (5, 2, 1):
        print(f"\n[7] end-to-end, n_lead={n_lead}: real channel count == train.py's prediction")
        cfg = yaml.safe_load(CFG_PATH.read_text())
        base = FireSpreadDataset.get_n_features(
            n_lead, cfg["features_to_keep"], cfg["remove_duplicate_features"])
        for combo in itertools.product([False, True], repeat=4):
            flags = dict(zip(FLAGS, combo))
            bits = "".join("1" if c else "0" for c in combo)
            k = FireSpreadDataset.n_centroid_channels(*combo, n_timesteps=n_lead)
            predicted = base + k          # exactly the arithmetic src/train.py performs
            _, ds = build(n_lead, flags)
            x, _ = ds[0]                  # index 0 == first sample of a fire == peek row absent
            if x.dim() == 4:
                check(f"n_lead={n_lead} {bits} rank stays 4-D with T=1",
                      x.shape[0] == 1, f"got {tuple(x.shape)}")
                n_ch, tail = x.shape[1], x[0, x.shape[1] - k:]
            else:
                n_ch, tail = x.shape[0], x[x.shape[0] - k:]
            # Centroid channels are the LAST k and must be spatially constant; a real imagery
            # channel essentially never is, so this also proves they landed at the tail.
            tail_ok = k == 0 or bool(((tail - tail[:, :1, :1]).abs() < 1e-6).all())
            check(f"n_lead={n_lead} {bits} channels == {base} + {k} == {predicted}",
                  n_ch == predicted and tail_ok,
                  f"got shape={tuple(x.shape)} tail_constant={tail_ok}")

    # -------------------------------------------------------- 8. peek-back is crop-neutral
    # The crop search scores ten candidate windows by fire content and keeps the best. If the
    # peek-back frame were allowed into that score, its x-term would be re-weighted against the
    # y-term and the winning crop could change -- so a velocity arm and a position-only arm would
    # train on different crops of the same sample, a difference having nothing to do with the
    # feature under test. augment() slices the peek frames out of the score to prevent this.
    # Here: same seed, same index, imagery channels must come back bit-identical.
    print("\n[8] peek-back does not perturb the fire-biased crop search")
    for n_lead in (5, 2, 1):
        cfg = yaml.safe_load(CFG_PATH.read_text())
        base = FireSpreadDataset.get_n_features(
            n_lead, cfg["features_to_keep"], cfg["remove_duplicate_features"])
        _, ds_no_peek = build(n_lead, {"use_centroid_position": True})
        _, ds_peek = build(n_lead, {"use_centroid_position": True,
                                    "use_centroid_velocity": True})
        for idx in (0, 37):
            xa, ya = deterministic_getitem(ds_no_peek, idx)
            xb, yb = deterministic_getitem(ds_peek, idx)
            img_a = xa[0, :base] if xa.dim() == 4 else xa[:base]
            img_b = xb[0, :base] if xb.dim() == 4 else xb[:base]
            check(f"n_lead={n_lead} idx={idx} imagery identical with/without peek",
                  torch.equal(img_a, img_b) and torch.equal(ya, yb),
                  f"max abs diff {float((img_a - img_b).abs().max())}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
