# scripts/

What each script here is for, grouped by purpose. All Python scripts are run from the
repo root (the shell scripts `cd` there themselves) in the `WSTS_original` conda env
(torch 2.7/cu128 — the RTX 5070 needs `sm_120`), with `HDF5_USE_FILE_LOCKING=FALSE`.
Most of these are one-off / rerunnable tools written to answer a specific question, not
a maintained CLI suite — read the header comment in the script itself for the full
reasoning behind any nonobvious choice.

## Dataset profiles (`wsts_profiles.py`)

Every efficiency verification and benchmark script takes `--profile <name>` instead of
hardcoding a config, and `--data-dir` (default `~/research/old_repo_FE_WSTS/hdf5_data`,
since the checked-in cfgs point at the original authors' cluster paths). A profile is a
committed data config plus overrides, chosen so that between them they exercise every
branch of the optimised code:

| profile | config | what it exercises |
|---|---|---|
| `veg_t1` | `data_monotemporal_veg_features.yaml` | the WSTS+ Table 2 reproduction config; hyperslab + one-hot skip both active |
| `veg_t5` | same, `n_leading_observations=5` | `flatten_and_remove_duplicate_features_`, the only path reading the remapped dynamic ids |
| `multi_t5` | `data_multitemporal_multi_features.yaml` | `features_to_keep` spans the one-hot block, so the skip must refuse to engage |
| `full_t1` | `data_monotemporal_full_features.yaml` | `features_to_keep=None`, so every optimisation must self-disable |
| `full_t5_doy` | `data_multitemporal_full_features.yaml` | `return_doy=True`, reading attrs through the cached handle |

Add a profile there rather than editing each script.

## Dataloader tuning (num_workers / prefetch_factor / lr)

- **`training_parameter_search.sh`** — Grid-searches `(num_workers, prefetch_factor)`
  throughput for arms A and C on fold 0 (500 steps each), shuffled run order plus an
  untimed warm-up per arm to cancel OS page-cache bias, with a memory-settle wait
  between runs. Writes `training_parameter_search_results.csv`. Machine-local tuning
  only — no scientific parameters are touched.
- **`probe_higher_workers_A0.sh`** — Narrow follow-up to the above: checks whether
  `num_workers=10/12` beat the `nw=8` winner for A0 (prefetch_factor fixed at 2).
  Appends its results into the same `training_parameter_search_results.csv`.
- **`lr_sweep.sh`** — Learning-rate sweep for A0 fold 2 (50 epochs, 4 LR values),
  using whatever `(num_workers, prefetch_factor)` the throughput search found best
  (pass as `--num_workers`/`--prefetch_factor`). Calls `lr_sweep_analyze.py` at the
  end to report the winner.
- **`lr_sweep_analyze.py`** — Given a run-name prefix, pulls each MLflow run's *peak*
  `val_avg_precision` (not the last-logged value, since 50 epochs of no-early-stop
  training can overfit past its best point) and prints LRs ranked by that peak.
- **`run_overnight_sweep_and_lr.sh`** — Orchestrates the two scripts above as one
  detached overnight job: runs `training_parameter_search.sh`, extracts arm A's
  fastest `(nw, pf)` from the CSV, then feeds it into `lr_sweep.sh`. Writes a status
  file (`logs/overnight_sweep_status_*.txt`) so progress/failure survives even if the
  launching session disconnects.
- **`watch_sweep_logs.sh <logfile> [logfile ...]`** — Tails one or more sweep logs and
  filters to high-signal lines only (stage banners, errors, OOM/kill signatures,
  final results) — built so a log can be babysat without drowning in Lightning's
  per-step progress-bar spam.

## 12-fold cross-validation runs

- **`run_12fold_cv.sh <arm letter> <num_workers> <prefetch_factor> <lr> <run_suffix>`**
  — Runs all 12 LOYO folds for one arm back-to-back (train+test each), using
  whatever dataloader/LR settings were already confirmed best; per-fold failures are
  logged and skipped rather than aborting the whole run. Every scientific parameter
  comes from the committed configs — only run identity and machine-local tuning are
  CLI args.
- **`launch_C0_sweep.sh`** — Launches `run_12fold_cv.sh` for arm C, fully detached
  (`setsid nohup`) with a timestamped log under `logs/`. Exists specifically because
  launching a long-lived background job from Windows via `wsl.exe -- bash -lc '...'`
  otherwise dies when `wsl.exe` returns; `setsid` keeps it alive.
- **`shutdown_after_sweep.ps1`** (PowerShell, run from Windows, not WSL) — Polls
  until no `run_12fold_cv.sh`/`train.py` process is left running, waits a grace
  period for MLflow to flush, logs a final run inventory, then does a clean
  `wsl --shutdown` followed by a Windows shutdown. Create `scripts/ABORT_SHUTDOWN`
  (any contents) to cancel it before it fires; it also self-aborts past `-MaxHours`
  (default 12h) as a safety net against a stuck sweep triggering a surprise shutdown.

## Reading MLflow results

- **`inspect_mlflow_runs.py`** — Reads the MLflow *file store* under `mlruns/`
  directly (bypassing the UI/API), because a run that dies without MLflow's exit
  hook firing leaves `status: RUNNING` forever while the UI still shows a stale green
  check. Flags runs stuck in `RUNNING` and runs missing `test_*` metrics.
  `--filter <substr>` narrows by run name, `--show-metrics` prints every metric's
  last value, `--compare a,b,c` tabulates param values + epoch count + wall time
  across matching runs.
- **`find_runs.py`** — Pulls a wandb summary metric (`--metric`, default `test_AP`)
  for every run whose name matches `--run_name_like` (SQL-LIKE style, `%` the only
  wildcard) and prints one row per run plus the mean across them. Drops
  `superseded`/`invalidated`-tagged and non-`finished` runs by default, and
  collapses same-named reruns to the most recently created one
  (`--include_dropped` / `--include_unfinished` / `--keep_duplicates` to keep
  them). Auth via
  `~/.netrc` / `WANDB_API_KEY`; entity/project default to
  `milesgoodman09-viewpoint-school/FE_WSTS`.
- **`collect_12fold_results.py --run_name_like '<pattern>'`** — General version of
  the above: pulls a given `--metric` (default `test_avg_precision`) for every run
  matching a SQL `LIKE` pattern, sorts by fold number, and prints mean/population-std
  across folds (population std matches the paper's per-fold convention).

## HDF5 read-path optimization & verification

- **`channel_check.py`** — Audits every HDF5 file's channel-dimension shape,
  per year, to check whether `features_to_keep`'s assumption of a 40-channel layout
  actually holds across the whole dataset (a sampled 2020 file was found with only
  23 channels).
- **`benchmark_hdf5_read_strategies.py`** — Measures actual disk bytes read per
  `__getitem__` (via `/proc/self/io`, with page-cache eviction before each read) for
  three strategies: (A) current — read all channels then filter in Python, (B)
  hyperslab — let HDF5 select only the needed raw channels, (C) same as B but on a
  rechunked+lzf-compressed copy. Reports amplification vs. what the model actually
  consumes. `--samples`, `--files`, `--skip-chunked`.
- **`benchmark_hyperslab_density.py --profile <p>`** — Measures where the hyperslab
  stops paying for itself, by timing reads (and counting block-layer bytes) for channel
  subsets of increasing density on the real files. This is the evidence behind
  `FireSpreadDataset.HDF5_READ_OPT_MAX_FRACTION`: the crossover moves with the number of
  frames read, and past it the partial read is *slower* than the plain full read.
- **`verify_channel_dependency.py --profile <p>`** — Empirically determines which raw
  HDF5 channels the pipeline's output actually depends on, by zeroing one raw channel at
  a time and checking whether `(x, y)` changes — a code-agnostic check that doesn't trust
  a by-hand index mapping. Then asserts `raw_channels_for_features()` reads a superset of
  what it measured; reading *less* is a hard fail.
- **`verify_read_optimization.py --profile <p>`** — Focused correctness gate for the
  hyperslab read in `FireSpreadDataset.load_imgs`: builds the same dataset with the
  optimisation forced off (`WSTS_DISABLE_HDF5_READ_OPT=1`) and on, and asserts every
  compared sample is bit-identical.

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

## Profiling

- **`profile_training_bottleneck.py`** — Samples a *live* `train.py` process for
  N seconds (GPU util via `nvidia-smi`, worker CPU% via `/proc/<pid>/stat`, disk
  MB/s via `/proc/diskstats`, swap/major-faults via `/proc/vmstat`) and prints a
  verdict: GPU-bound, I/O-bound, memory-pressure-bound, CPU-bound, or none-saturated
  (a latency/serialization regime, not a throughput wall). `--secs` (default 20).
- **`profile_getitem.py --profile <p>`** — cProfile's `FireSpreadDataset.__getitem__`
  directly (not the full training loop) to rank which callees consume per-sample CPU
  time, after a short warm-up so file-handle-open cost doesn't pollute the profile.
  `--samples`, `--top`.
- **`benchmark_getitem_ab.py --profile <p> --flags <flags>`** — Alternating in-process
  A/B of per-sample cost for effects near the noise floor: both variants time in ONE
  process, alternating after a warmup so neither wins on cache state. Reports medians
  with min/max, and refuses to let an overlapping spread be quoted as a ratio.
- **`benchmark_dataloader_throughput.py --profile <p> --workers 4,8,12`** — Real
  `DataLoader` batches/s across `num_workers` (and, with `--prefetch`, a joint
  `num_workers x prefetch_factor` grid), cold and warm. A **screen only**: it measures
  loader capacity with no consumer — no GPU work competing for CPU, no backpressure — so
  confirm finalists with real training runs before believing a wall-clock gain.
