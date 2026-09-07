"""Find the best learning rate from an lr_sweep.sh run.

Uses each run's *peak* val_avg_precision across its full metric history, not the
last-logged value -- that matches what the run's ModelCheckpoint (monitor=val_avg_precision,
mode=max) actually saved as the best epoch, since 50 epochs of no-early-stopping training can
overfit past its best point.
"""
import argparse
import re

import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--run_prefix", default="lr_sweep_A0_fold2")
args = parser.parse_args()

mlflow.set_tracking_uri("mlruns")
client = MlflowClient()

df = mlflow.search_runs(
    search_all_experiments=True,
    filter_string=f"tags.mlflow.runName LIKE '{args.run_prefix}%'",
    order_by=["tags.mlflow.runName"],
)

if df.empty:
    print(f"No runs found with run_name prefix '{args.run_prefix}'")
    raise SystemExit(1)

results = []
for _, row in df.iterrows():
    run_name = row["tags.mlflow.runName"]
    m = re.search(r"_lr([0-9eE.+-]+)$", run_name)
    lr = m.group(1) if m else "?"
    history = client.get_metric_history(row["run_id"], "val_avg_precision")
    if not history:
        print(f"{run_name}: no val_avg_precision logged, skipping")
        continue
    best_epoch_metric = max(history, key=lambda m: m.value)
    results.append((lr, best_epoch_metric.value, best_epoch_metric.step, run_name))

if not results:
    print("No runs had val_avg_precision logged.")
    raise SystemExit(1)

print(f"{'lr':<10}{'best_val_AP':<15}{'at_step':<10}run_name")
for lr, best_val, step, run_name in sorted(results, key=lambda r: -r[1]):
    print(f"{lr:<10}{best_val:<15.5f}{step:<10}{run_name}")

best = max(results, key=lambda r: r[1])
print(f"\nBest lr: {best[0]} (peak val_avg_precision={best[1]:.5f})")
