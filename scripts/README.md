# scripts/

What each script here is for, grouped by purpose. All Python scripts are run from the
repo root (the shell scripts `cd` there themselves) in the `WSTS_original` conda env
(torch 2.7/cu128 — the RTX 5070 needs `sm_120`), with `HDF5_USE_FILE_LOCKING=FALSE`.
Most of these are one-off / rerunnable tools written to answer a specific question, not
a maintained CLI suite — read the header comment in the script itself for the full
reasoning behind any nonobvious choice.

The shell runners hardcode `PY=/home/miles/miniconda3/envs/WSTS_original/bin/python`
rather than relying on an activated env. An absolute path to an env's `bin/python`
selects that env's site-packages on its own, so whatever `conda activate` you happen to
be in does not affect a scripted run — only bare `python` typed by hand.

This directory was pruned on 2026-09-07 to the launch path actually in use plus the
correctness gates. Roughly forty superseded one-off scripts (dataloader/LR tuning
sweeps, MLflow-era result readers, the `run_12fold_cv.sh` generation of runners, and
the read-path benchmarks) were removed; they remain in git history at commit `a9cdfc4`
and earlier if a measurement needs to be repeated or a rationale re-read.

## Dataset profiles (`wsts_profiles.py`)

Every verification script takes `--profile <name>` instead of hardcoding a config, and
`--data-dir` (default `~/research/old_repo_FE_WSTS/hdf5_data`, since the checked-in
cfgs point at the original authors' cluster paths). A profile is a committed data config
plus overrides, chosen so that between them they exercise every branch of the optimised
code:

| profile | config | what it exercises |
|---|---|---|
| `veg_t1` | `data_monotemporal_veg_features.yaml` | the WSTS+ Table 2 reproduction config; hyperslab + one-hot skip both active |
| `veg_t5` | same, `n_leading_observations=5` | `flatten_and_remove_duplicate_features_`, the only path reading the remapped dynamic ids |
| `multi_t5` | `data_multitemporal_multi_features.yaml` | `features_to_keep` spans the one-hot block, so the skip must refuse to engage |
| `full_t1` | `data_monotemporal_full_features.yaml` | `features_to_keep=None`, so every optimisation must self-disable |
| `full_t5_doy` | `data_multitemporal_full_features.yaml` | `return_doy=True`, reading attrs through the cached handle |

Add a profile there rather than editing each script.

## Launching arm runs

This is the current launch path, and the only one — `launch_arms_sequential.sh` →
`run_arms_sequential.sh` → `run_fold_subset_cv.sh` → `src/train.py`. The env pin lives
in the innermost script, so every arm launched this way trains under `WSTS_original`.

- **`launch_arms_sequential.sh <arms> <folds> <nw> <pf> <lr> <suffix> [seed]`** —
  Detached launcher for the runner below: `setsid`, not just `nohup`, because when
  launched through `wsl.exe -- bash -lc ...` the WSL session is torn down as soon as
  `wsl.exe` returns and a plain background job dies with it. Writes a timestamped log
  under `logs/` and echoes the runner pid. Example:
  `bash scripts/launch_arms_sequential.sh "A1 A2" "1 5 6 11" 8 3 1e-3 bs64`
- **`run_arms_sequential.sh <arms> <folds> <nw> <pf> <lr> <suffix> [seed]`** — Runs
  several arms back-to-back on the same fold subset by delegating each to
  `run_fold_subset_cv.sh`. Fails fast on a typo'd arm name (checks `cfgs/arms/<ARM>.yaml`
  exists) rather than 4.5 hours into the night, and a failing arm does **not** stop the
  ones after it — an overnight job that dies on arm 1 at 2 a.m. should still deliver
  arm 2 by morning.
- **`run_fold_subset_cv.sh <arm> <run_prefix> <extra_data_yaml|none> <folds> <nw> <pf> <lr> <suffix>`**
  — Trains and tests one arm across an arbitrary fold subset, with an optional extra
  `--data` override file (used by the val-adjustment comparison arms A0v/B0v/C0v; see
  `administrative/EXPERIMENTS.md`). Every scientific parameter comes from the committed
  configs (`cfgs/data_base.yaml` + `cfgs/arms/<ARM>.yaml` + optional override) — only run
  identity, the fold list, and machine-local dataloader tuning are CLI args.

## Reading results

- **`find_runs.py`** — Pulls a wandb summary metric (`--metric`, default `test_AP`)
  for every run whose name matches `--run_name_like` (SQL-LIKE style, `%` the only
  wildcard) and prints one row per run plus the mean across them. Drops
  `superseded`/`invalidated`-tagged and non-`finished` runs by default, and
  collapses same-named reruns to the most recently created one
  (`--include_dropped` / `--include_unfinished` / `--keep_duplicates` to keep
  them). Auth via `~/.netrc` / `WANDB_API_KEY`; entity/project default to
  `milesgoodman09-viewpoint-school/FE_WSTS`.
- **`Testbook.ipynb`** — Scratch notebook for ad-hoc inspection of runs and results.

## Feature correctness

- **`verify_centroid_channels.py`** — Verifies `_compute_centroid_channels` and the
  peek-back loading path against known answers. The centroid feature has four independent
  flags and emits one channel per family per model-visible timestep, so there are
  16 × `n_lead` valid input layouts, and the count `src/train.py` derives from config
  alone has to match the tensor the dataloader actually produces for every one of them.
  Most sections use a synthetic `(T, C, H, W)` tensor with fire at chosen pixels, making
  every expected value computable by hand — they run in under a second with no data
  directory (`--skip-data`), so there is no excuse not to run them.

## Efficiency correctness gates (run these before any sweep)

Each speed-up in `FireSpreadDataset` keeps its original code path behind a
`WSTS_DISABLE_*` env flag (`WSTS_DISABLE_HDF5_READ_OPT`, `WSTS_DISABLE_CROP_OPT`,
`WSTS_DISABLE_ONEHOT_SKIP`). Exit code 0 only on a full pass — treat any nonzero exit as
"do not train on this."

- **`verify_pipeline_equivalence.py --flags all --profiles all`** — The general A/B:
  builds each profile twice, flags set vs clear, pins the RNG, and compares `(x, y)`
  element-for-element for train and eval. Use `--flags` to isolate one optimisation or
  `all` to prove they compose. Prints which optimisations actually engaged per profile,
  so a run that passed vacuously is visible.
- **`verify_dataload_optimization.py --profiles all`** — The same claim from the other
  side, and the reason both exist: it uses *no* env flags, so it still tests something
  if a flag were ever misspelled or stopped being read. Builds two datasets normally and
  forces one back to the legacy path by resetting the four instance attributes the
  optimisations hang off.
- **`verify_read_optimization.py --profile <p>`** — Focused correctness gate for the
  hyperslab read in `FireSpreadDataset.load_imgs`: builds the same dataset with the
  optimisation forced off (`WSTS_DISABLE_HDF5_READ_OPT=1`) and on, and asserts every
  compared sample is bit-identical.
- **`verify_channel_dependency.py --profile <p>`** — Empirically determines which raw
  HDF5 channels the pipeline's output actually depends on, by zeroing one raw channel at
  a time and checking whether `(x, y)` changes — a code-agnostic check that doesn't trust
  a by-hand index mapping. Then asserts `raw_channels_for_features()` reads a superset of
  what it measured; reading *less* is a hard fail.
- **`verify_dataloader_settings.py --profile <p> --workers 8`** — Covers the
  `FireSpreadDataModule` knobs, which act on the loader rather than on one sample, so
  they need whole epochs rather than sample comparison. Confirms `prefetch_factor` and
  `hdf5_cache_size` are output-neutral, and that `persistent_workers` is neutral in epoch
  0 (it draws a different — not degenerate — augmentation stream afterwards; see the
  `persistent_workers` docstring in `FireSpreadDataModule`).
- **`verify_hdf5_cache_fork_safety.py --profile <p> --workers 4`** — The cached
  `h5py.File` handles are only safe if opened *after* workers fork, and
  `FireSpreadDataModule`'s ignition filters iterate the whole dataset in the parent
  first. This reproduces that exact ordering and checks the data still matches, since
  inherited HDF5 descriptors corrupt silently rather than raising.
- **`verify_training_equivalence.sh [max_steps] [num_workers]`** — End-to-end proof that
  the speed-ups do not change *training*. The gates above compare `(x, y)` at the
  dataloader's output; this compares the far end — two real training runs, identical
  arguments and seed, one with every `WSTS_DISABLE_*` flag set and one with none, then
  diffs the per-step losses and a hash of the final model weights. `num_workers=0` by
  default and deliberately: with workers, augmentation RNG lives in forked processes
  whose seeding depends on worker count and iterator lifetime, so two runs would
  legitimately differ for reasons unrelated to the optimisations.
