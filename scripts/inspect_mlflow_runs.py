#!/usr/bin/env python
"""Inspect MLflow *file-store* runs directly, bypassing the MLflow UI/API.

Why this exists: cfgs/trainer_single_gpu_mlflow.yaml sets `tracking_uri: mlruns`,
a plain-directory file store (NOT the legacy empty mlflow.db at the repo root).
When a run dies without MLflow's exit hook firing, its meta.yaml keeps
`status: RUNNING` forever and the UI still renders a green check from the last
logged metric -- so the UI is not a trustworthy record of completion. This reads
the on-disk truth instead.

Usage:
    python scripts/inspect_mlflow_runs.py                 # all runs
    python scripts/inspect_mlflow_runs.py --filter bs64   # runs whose name matches
    python scripts/inspect_mlflow_runs.py --filter A0 --show-metrics
"""

import argparse
import datetime as dt
from pathlib import Path

# MLflow encodes run status as an int in meta.yaml on some versions, str on others.
STATUS_CODES = {1: "SCHEDULED", 2: "RUNNING", 3: "FINISHED", 4: "FAILED", 5: "KILLED"}

REPO_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = REPO_ROOT / "mlruns"


def read_first_line(path: Path) -> str | None:
    """Tags/params in the file store hold their value as the file's entire body."""
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def parse_meta(meta_path: Path) -> dict:
    """meta.yaml here is flat `key: value`, so avoid a yaml dependency."""
    meta = {}
    try:
        for line in meta_path.read_text().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    except OSError:
        pass
    return meta


def fmt_time(ms: str | None) -> str:
    if not ms:
        return "-"
    try:
        return dt.datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "-"


def duration(start: str | None, end: str | None) -> str:
    try:
        return f"{(int(end) - int(start)) / 60000:.1f} min"
    except (ValueError, TypeError):
        return "-"


def last_metric_value(metric_file: Path) -> str:
    """Metric files are `<timestamp> <value> <step>` per line; we want the last."""
    try:
        lines = [ln for ln in metric_file.read_text().splitlines() if ln.strip()]
        if not lines:
            return "-"
        parts = lines[-1].split()
        return f"{float(parts[1]):.5f} @ step {parts[2] if len(parts) > 2 else '?'}"
    except (OSError, ValueError, IndexError):
        return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="substring match on run name")
    ap.add_argument("--show-metrics", action="store_true", help="print last value of every metric")
    ap.add_argument(
        "--compare",
        default="",
        help="comma-separated param names to tabulate (e.g. batch_size,max_steps,lr)",
    )
    args = ap.parse_args()

    if not MLRUNS.is_dir():
        raise SystemExit(f"No mlruns directory at {MLRUNS}")

    rows = []
    for exp_dir in MLRUNS.iterdir():
        # `models/` is the registry, not an experiment; experiment dirs are numeric ids.
        if not exp_dir.is_dir() or not exp_dir.name.isdigit():
            continue
        exp_name = parse_meta(exp_dir / "meta.yaml").get("name", exp_dir.name)

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = read_first_line(run_dir / "tags" / "mlflow.runName") or "(unnamed)"
            if args.filter and args.filter not in name:
                continue

            meta = parse_meta(run_dir / "meta.yaml")
            status = meta.get("status", "?")
            status = STATUS_CODES.get(int(status), status) if status.isdigit() else status

            metric_dir = run_dir / "metrics"
            metrics = sorted(p.name for p in metric_dir.iterdir()) if metric_dir.is_dir() else []
            test_metrics = [m for m in metrics if m.lower().startswith("test")]

            rows.append(
                {
                    "name": name,
                    "exp": exp_name,
                    "dir": run_dir,
                    "status": status,
                    "start": fmt_time(meta.get("start_time")),
                    "end": fmt_time(meta.get("end_time")),
                    "dur": duration(meta.get("start_time"), meta.get("end_time")),
                    "metrics": metrics,
                    "test_metrics": test_metrics,
                }
            )

    rows.sort(key=lambda r: r["start"])

    if args.compare:
        keys = [k.strip() for k in args.compare.split(",") if k.strip()]
        # Wall-clock per run is only comparable once you know how many optimizer
        # steps and how many samples-per-step each run actually did.
        header = f"{'run':<28} {'status':<9} {'epochs':>7} {'wall':>9}  " + "  ".join(
            f"{k:>12}" for k in keys
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            vals = []
            for k in keys:
                v = read_first_line(r["dir"] / "params" / k)
                vals.append(f"{(v if v is not None else '-'):>12}")
            ep = last_metric_value(r["dir"] / "metrics" / "epoch").split(" @ ")[0]
            ep = ep if ep != "-" else "-"
            print(
                f"{r['name']:<28} {r['status']:<9} {ep:>7} {r['dur']:>9}  " + "  ".join(vals)
            )
        print()
        # Show which param names exist, so you can pick valid --compare keys.
        if rows:
            pdir = rows[0]["dir"] / "params"
            if pdir.is_dir():
                print("available params:", ", ".join(sorted(p.name for p in pdir.iterdir())))
        return

    for r in rows:
        # The pairing that matters: a RUNNING status with no test_* metrics means
        # the process vanished before trainer.test() ever logged anything.
        flag = ""
        if r["status"] == "RUNNING":
            flag = "  <-- never closed out (no FINISHED/FAILED written)"
        print(f"=== {r['name']} ===")
        print(f"  experiment : {r['exp']}")
        print(f"  run dir    : {r['dir'].relative_to(REPO_ROOT)}")
        print(f"  status     : {r['status']}{flag}")
        print(f"  start/end  : {r['start']}  ->  {r['end']}   ({r['dur']})")
        print(f"  n metrics  : {len(r['metrics'])}")
        print(f"  test_*     : {', '.join(r['test_metrics']) if r['test_metrics'] else 'NONE - test phase never ran'}")
        if args.show_metrics:
            for m in r["metrics"]:
                print(f"      {m:<40} {last_metric_value(r['dir'] / 'metrics' / m)}")
        print()

    print(f"{len(rows)} run(s) matched.")
    stuck = [r for r in rows if r["status"] == "RUNNING"]
    untested = [r for r in rows if not r["test_metrics"]]
    print(f"  stuck in RUNNING : {len(stuck)}")
    print(f"  missing test_*   : {len(untested)}")


if __name__ == "__main__":
    main()
