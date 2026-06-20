#!/usr/bin/env python3
"""
Select a subset of CUA-Gym tasks based on a config file.

Usage:
    python scripts/select_tasks.py --config configs/select_config.json
    python scripts/select_tasks.py --config configs/select_config.json --seed 42
    python scripts/select_tasks.py --config configs/select_config.json --tasks-dir /path/to/tasks

Config format (JSON):
    {
        "tasks_dir": "/optional/override/path/to/cua_gym_tasks",
        "seed": 42,
        "apps": {
            "libreoffice_calc": {
                "n": 50
            },
            "vscode": {
                "n": 30,
                "difficulty": {
                    "easy": 5,
                    "medium": 10,
                    "hard": 15
                }
            },
            "multi_apps": {
                "n": 20
            }
        }
    }

If "difficulty" is omitted for an app, tasks are sampled uniformly at random
from all available tasks for that app (across all difficulty levels).

If "difficulty" is specified, each key is a difficulty label ("easy", "medium",
"hard", "unknown") and its value is the number of tasks to sample from that
bucket. The total in "difficulty" should equal "n" (if both present); if they
differ, "difficulty" counts take precedence and "n" is ignored.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TASKS_DIR = Path(
    "/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all/cua_gym_tasks"
)
OUTPUT_BASE = Path("/lcars/home/q/qianranm/research/GUI/CUA-Gym/data")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_tasks(tasks_dir: Path) -> list[dict]:
    tasks = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_file = task_dir / "task.json"
        if not task_file.exists():
            continue
        try:
            with open(task_file) as f:
                data = json.load(f)
            data.setdefault("_dir", task_dir.name)
            tasks.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return tasks


# ---------------------------------------------------------------------------
# App normalization (mirrors analyze_tasks.py)
# ---------------------------------------------------------------------------

def canonical_app(app_type: str | None) -> str:
    if not app_type:
        return "unknown"
    if "," in app_type:
        return "multi_apps"
    return app_type.strip()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_tasks(
    tasks: list[dict],
    app_config: dict[str, dict],
    rng: random.Random,
) -> list[dict]:
    """Return selected task dicts according to app_config."""

    # Group tasks by canonical app, then by difficulty
    by_app: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        app = canonical_app(t.get("app_type"))
        diff = t.get("difficulty") or "unknown"
        by_app[app][diff].append(t)

    selected: list[dict] = []

    for app, spec in app_config.items():
        pool_by_diff = by_app.get(app, {})

        diff_spec: dict[str, int] | None = spec.get("difficulty")

        if diff_spec:
            # Per-difficulty quota
            for diff, quota in diff_spec.items():
                bucket = pool_by_diff.get(diff, [])
                if len(bucket) < quota:
                    print(
                        f"  Warning: {app}/{diff} — requested {quota} but only "
                        f"{len(bucket)} available; taking all."
                    )
                chosen = rng.sample(bucket, min(quota, len(bucket)))
                selected.extend(chosen)
        else:
            # Flat quota across all difficulties
            n = spec.get("n", 0)
            flat_pool: list[dict] = [t for bucket in pool_by_diff.values() for t in bucket]
            if len(flat_pool) < n:
                print(
                    f"  Warning: {app} — requested {n} but only "
                    f"{len(flat_pool)} available; taking all."
                )
            chosen = rng.sample(flat_pool, min(n, len(flat_pool)))
            selected.extend(chosen)

    return selected


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_summary(selected: list[dict], app_config: dict, config: dict, seed: int) -> dict:
    app_counts: Counter = Counter()
    diff_counts: Counter = Counter()
    app_diff: dict[str, Counter] = defaultdict(Counter)

    for t in selected:
        app = canonical_app(t.get("app_type"))
        diff = t.get("difficulty") or "unknown"
        app_counts[app] += 1
        diff_counts[diff] += 1
        app_diff[app][diff] += 1

    return {
        "total_selected": len(selected),
        "seed": seed,
        "config_apps_requested": list(app_config.keys()),
        "difficulty_distribution": dict(diff_counts.most_common()),
        "app_distribution": dict(app_counts.most_common()),
        "app_difficulty_breakdown": {
            app: dict(diff_counts_per_app.most_common())
            for app, diff_counts_per_app in sorted(
                app_diff.items(), key=lambda kv: app_counts[kv[0]], reverse=True
            )
        },
    }


def save_outputs(selected: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Flat list of task IDs
    task_ids = [t["id"] for t in selected]
    ids_path = output_dir / "task_ids.json"
    with open(ids_path, "w") as f:
        json.dump(task_ids, f, indent=2)
    print(f"  Saved task IDs  : {ids_path}  ({len(task_ids)} tasks)")

    # 2) Summary
    summary_path = output_dir / "selection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary   : {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select CUA-Gym tasks from a config file.")
    p.add_argument(
        "--config", required=True, metavar="CONFIG",
        help="Path to selection config JSON file.",
    )
    p.add_argument(
        "--tasks-dir", metavar="DIR",
        help="Override the tasks directory (default: value in config or built-in default).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config seed).",
    )
    p.add_argument(
        "--output-base", metavar="DIR", default=str(OUTPUT_BASE),
        help=f"Base directory for run output (default: {OUTPUT_BASE}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config: dict = json.load(f)

    # Resolve tasks dir: CLI > config > default
    if args.tasks_dir:
        tasks_dir = Path(args.tasks_dir)
    elif "tasks_dir" in config:
        tasks_dir = Path(config["tasks_dir"])
    else:
        tasks_dir = DEFAULT_TASKS_DIR

    if not tasks_dir.exists():
        sys.exit(f"Tasks directory not found: {tasks_dir}")

    # Resolve seed
    seed = args.seed if args.seed is not None else config.get("seed", 0)

    app_config: dict[str, dict] = config.get("apps", {})
    if not app_config:
        sys.exit("Config must have a non-empty 'apps' section.")

    # Build output dir:  data/run-<timestamp>/task_selection/
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / f"run-{timestamp}" / "task_selection"

    print(f"Tasks dir  : {tasks_dir}")
    print(f"Config     : {config_path}")
    print(f"Seed       : {seed}")
    print(f"Output dir : {output_dir}")
    print()

    print("Loading tasks...")
    tasks = load_all_tasks(tasks_dir)
    print(f"  Loaded {len(tasks):,} tasks")
    print()

    print("Selecting tasks...")
    rng = random.Random(seed)
    selected = select_tasks(tasks, app_config, rng)
    print(f"  Selected {len(selected):,} tasks total")
    print()

    summary = build_summary(selected, app_config, config, seed)

    print("Saving outputs...")
    save_outputs(selected, summary, output_dir)

    # Print a concise report
    print()
    print("=== Selection Summary ===")
    print(f"Total selected : {summary['total_selected']}")
    print()
    print("By difficulty:")
    for diff, cnt in summary["difficulty_distribution"].items():
        print(f"  {diff:<10} {cnt:>5}")
    print()
    print("By app:")
    for app, cnt in summary["app_distribution"].items():
        diff_parts = ", ".join(
            f"{d}={c}"
            for d, c in summary["app_difficulty_breakdown"][app].items()
        )
        print(f"  {app:<40} {cnt:>5}   [{diff_parts}]")


if __name__ == "__main__":
    main()
