"""Pull a metric for every wandb run whose name matches a pattern, print the mean.

FE_WSTS tracks with Weights & Biases, not MLflow -- this is the wandb port of the
old mlflow-backed script. Same interface: give it a name pattern and it prints one
row per matching run plus the average of --metric across them (the LOYO unit of
analysis is the 12-fold mean).

Reruns keep the same wandb display name (`A0_seed0_fold0`), so a fold can have
several runs. Stale ones carry a `superseded` (or `invalidated`) tag and are
dropped by default. As a backstop for reruns nobody tagged, when two surviving
runs still share a name only the most recently created one is kept and the older
one is listed as skipped.

Auth comes from ~/.netrc / WANDB_API_KEY the same way training does; this never
calls `wandb login` (see the wandb-new-key-old-client note).

Examples:
    python scripts/find_runs.py --run_name_like 'A0_seed0_fold%'
    python scripts/find_runs.py --run_name_like 'C0_%' --metric test_f1
"""
import argparse
import re
import sys
import warnings

# wandb 0.15's public API does `from pkg_resources import parse_version` at import
# time, which setuptools>=81 warns about. Nothing here can act on it -- it's inside
# the pinned old client -- so silence just that message.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", module="wandb")

import wandb

DEFAULT_ENTITY = "milesgoodman09-viewpoint-school"
DEFAULT_PROJECT = "FE_WSTS"
DROP_TAGS = ("superseded", "invalidated")

parser = argparse.ArgumentParser()
parser.add_argument("--run_name_like", required=True,
                    help="SQL-LIKE-style pattern on the wandb run name, e.g. 'A0_seed0_fold%%'. "
                         "'%%' is the only wildcard (matches any run-name substring); "
                         "everything else is literal.")
parser.add_argument("--metric", default="test_AP",
                    help="wandb summary key to average (test_AP, test_f1, test_precision, ...).")
parser.add_argument("--entity", default=DEFAULT_ENTITY)
parser.add_argument("--project", default=DEFAULT_PROJECT)
parser.add_argument("--include_dropped", "--include_invalidated", action="store_true",
                    dest="include_dropped",
                    help=f"keep runs tagged {'/'.join(DROP_TAGS)} (dropped by default)")
parser.add_argument("--include_unfinished", action="store_true",
                    help="by default, only runs in wandb state 'finished' are counted")
parser.add_argument("--keep_duplicates", action="store_true",
                    help="do not collapse same-named runs to the newest one")
args = parser.parse_args()


def like_to_regex(pattern: str) -> str:
    """SQL LIKE -> anchored regex. Only '%' is a wildcard; '_' stays literal
    (run names are full of underscores: 'A0_seed0_fold0')."""
    return "^" + "".join(".*" if ch == "%" else re.escape(ch) for ch in pattern) + "$"


def natural_key(name: str):
    # 'fold2' before 'fold10': compare digit runs as ints
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


name_re = like_to_regex(args.run_name_like)
api = wandb.Api()
path = f"{args.entity}/{args.project}"

runs = list(api.runs(path, filters={"display_name": {"$regex": name_re}}))

if not runs:
    print(f"No runs matched LIKE '{args.run_name_like}' in {path}.")
    all_names = sorted({r.name for r in api.runs(path)})
    print(f"{len(all_names)} run names exist:")
    for n in all_names:
        print(f"  {n}")
    sys.exit(1)

kept = []          # (name, value, run) that pass every filter
skipped = []       # (name, reason)
for r in runs:
    dropped_tag = next((t for t in DROP_TAGS if t in r.tags), None)
    if dropped_tag and not args.include_dropped:
        skipped.append((r.name, dropped_tag))
        continue
    if r.state != "finished" and not args.include_unfinished:
        skipped.append((r.name, f"state={r.state}"))
        continue
    value = r.summary.get(args.metric)
    if value is None:
        skipped.append((r.name, f"no {args.metric}"))
        continue
    kept.append((r.name, float(value), r))

# Backstop de-dup: same name surviving twice -> keep the newest, skip the rest.
if not args.keep_duplicates:
    by_name = {}
    for name, value, r in kept:
        by_name.setdefault(name, []).append((name, value, r))
    deduped = []
    for name, entries in by_name.items():
        entries.sort(key=lambda e: e[2].created_at, reverse=True)
        deduped.append(entries[0])
        for _, _, r in entries[1:]:
            skipped.append((name, f"older duplicate ({r.created_at[:10]}, id={r.id})"))
    kept = deduped

rows = sorted(((name, value) for name, value, _ in kept), key=lambda x: natural_key(x[0]))
width = max((len(n) for n, _ in rows), default=0)
for name, value in rows:
    print(f"{name:<{width}}  {value:.6f}")

if skipped:
    # Tag drops are routine and bulky; summarise them. Anything else (unfinished,
    # missing metric, untagged duplicate) is worth seeing per-run.
    tag_drops = [s for s in skipped if s[1] in DROP_TAGS]
    other = [s for s in skipped if s[1] not in DROP_TAGS]
    print(f"\n{len(skipped)} run(s) skipped:")
    for tag in DROP_TAGS:
        n = sum(1 for _, why in tag_drops if why == tag)
        if n:
            print(f"  {n} tagged '{tag}'")
    for name, why in sorted(other, key=lambda s: natural_key(s[0])):
        print(f"  {name}  ({why})")

if rows:
    mean = sum(v for _, v in rows) / len(rows)
    print(f"\nAverage {args.metric} over {len(rows)} run(s): {mean:.6f}")
else:
    print(f"\nNo runs with a usable {args.metric}.")
    sys.exit(1)
