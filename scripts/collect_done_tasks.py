"""
Scan all run-* directories under CUA-Gym/data, find task IDs that have a
completed trajectory (trajectory.jsonl present), and save them to
CUA-Gym/data/task_done.json.

A task is considered done if:
  data/run-*/trajectory/<app>/<task_id>/trajectory.jsonl exists
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_FILE = DATA_DIR / "task_done.json"


def collect_done_tasks() -> dict:
    """
    Returns a dict:
      {
        "task_ids": [sorted list of unique done task IDs],
        "count_by_app": {"pdf": 3, "chrome": 5, ...}
      }
    """
    all_done: set[str] = set()
    count_by_app: dict[str, int] = {}

    run_dirs = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and p.name.startswith("run-")
    )

    for run_dir in run_dirs:
        run_name = run_dir.name
        done_in_run = 0

        traj_root = run_dir / "trajectory"
        if traj_root.is_dir():
            for app_dir in sorted(traj_root.iterdir()):
                if not app_dir.is_dir():
                    continue
                app = app_dir.name
                for task_dir in sorted(app_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    if (task_dir / "trajectory.jsonl").exists():
                        if task_dir.name not in all_done:
                            count_by_app[app] = count_by_app.get(app, 0) + 1
                        all_done.add(task_dir.name)
                        done_in_run += 1

        print(f"  {run_name}: {done_in_run} done")

    return {
        "task_ids": sorted(all_done),
        "count_by_app": dict(sorted(count_by_app.items())),
    }


def main():
    print(f"Scanning {DATA_DIR} ...")
    result = collect_done_tasks()
    total = len(result["task_ids"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. {total} unique completed task(s) saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
