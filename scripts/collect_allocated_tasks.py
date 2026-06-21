"""
Scan all run-* directories under CUA-Gym/data, find task IDs that have been
allocated (i.e. task_selection/task_ids.json present), and save them to
CUA-Gym/data/task_allocated.json.

A task is considered allocated if it appears in:
  data/run-*/task_selection/task_ids.json
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TASKS_DIR = DATA_DIR / "cua_gym_all" / "cua_gym_tasks"
OUTPUT_FILE = DATA_DIR / "task_allocated.json"


def build_task_app_map() -> dict[str, str]:
    """Build a mapping of task_id -> canonical app name from the tasks directory."""
    mapping: dict[str, str] = {}
    if not TASKS_DIR.exists():
        return mapping
    for task_dir in TASKS_DIR.iterdir():
        if not task_dir.is_dir():
            continue
        task_file = task_dir / "task.json"
        if not task_file.exists():
            continue
        try:
            with open(task_file) as f:
                data = json.load(f)
            task_id = data.get("id", task_dir.name)
            app_type = data.get("app_type") or "unknown"
            mapping[task_id] = "multi_apps" if "," in app_type else app_type.strip()
        except (json.JSONDecodeError, OSError):
            continue
    return mapping


def collect_allocated_tasks() -> dict:
    """
    Returns a dict:
      {
        "task_ids": [sorted list of unique allocated task IDs],
        "count_by_run": {"run-20260620_192737": 10, ...},
        "count_by_app": {"pdf": 3, "chrome": 5, ...}
      }
    """
    all_allocated: set[str] = set()
    count_by_run: dict[str, int] = {}

    task_app_map = build_task_app_map()

    run_dirs = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and p.name.startswith("run-")
    )

    for run_dir in run_dirs:
        run_name = run_dir.name
        ids_file = run_dir / "task_selection" / "task_ids.json"

        if not ids_file.exists():
            print(f"  {run_name}: no task_ids.json, skipping")
            continue

        try:
            with open(ids_file) as f:
                task_ids: list[str] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  {run_name}: failed to read task_ids.json ({e}), skipping")
            continue

        new_in_run = [t for t in task_ids if t not in all_allocated]
        all_allocated.update(task_ids)
        count_by_run[run_name] = len(task_ids)
        print(f"  {run_name}: {len(task_ids)} allocated ({len(new_in_run)} new)")

    count_by_app: dict[str, int] = {}
    for task_id in all_allocated:
        app = task_app_map.get(task_id, "unknown")
        count_by_app[app] = count_by_app.get(app, 0) + 1

    return {
        "task_ids": sorted(all_allocated),
        "count_by_run": count_by_run,
        "count_by_app": dict(sorted(count_by_app.items())),
    }


def main():
    print(f"Scanning {DATA_DIR} ...")
    result = collect_allocated_tasks()
    total = len(result["task_ids"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. {total} unique allocated task(s) saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
