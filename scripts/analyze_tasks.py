#!/usr/bin/env python3
"""Analyze the CUA-Gym task dataset and save results to data/."""

import json
from collections import Counter, defaultdict
from pathlib import Path

TASKS_DIR = Path("/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all/cua_gym_tasks")
OUTPUT_DIR = Path("/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_DIFFICULTIES = ["easy", "medium", "hard", "unknown"]


def load_all_tasks(tasks_dir: Path) -> list[dict]:
    tasks = []
    missing_json = 0
    parse_errors = 0
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_file = task_dir / "task.json"
        if not task_file.exists():
            missing_json += 1
            continue
        try:
            with open(task_file) as f:
                data = json.load(f)
            data.setdefault("_dir", task_dir.name)
            tasks.append(data)
        except json.JSONDecodeError:
            parse_errors += 1
    if missing_json:
        print(f"  Warning: {missing_json} dirs missing task.json")
    if parse_errors:
        print(f"  Warning: {parse_errors} task.json files failed to parse")
    return tasks


def canonical_app(app_type: str | None) -> str:
    """Normalize app_type; multi-app tasks (comma-separated) → 'multi_apps'."""
    if not app_type:
        return "unknown"
    if "," in app_type:
        return "multi_apps"
    return app_type.strip()


def component_apps(app_type: str | None) -> list[str]:
    """Return individual app names, splitting comma-separated multi-app values."""
    if not app_type:
        return ["unknown"]
    return [a.strip() for a in app_type.split(",")]


def analyze(tasks: list[dict]) -> dict:
    total = len(tasks)

    # --- difficulty distribution ---
    difficulty_counts: Counter = Counter()
    for t in tasks:
        diff = t.get("difficulty") or "unknown"
        difficulty_counts[diff] += 1

    # --- canonical app distribution (multi-app → 'multi_apps') ---
    app_counts: Counter = Counter()
    for t in tasks:
        app = canonical_app(t.get("app_type"))
        app_counts[app] += 1

    # --- per-canonical-app difficulty breakdown ---
    app_difficulty: dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        app = canonical_app(t.get("app_type"))
        diff = t.get("difficulty") or "unknown"
        app_difficulty[app][diff] += 1

    # --- individual component-app mention counts (counting through multi_apps) ---
    component_counts: Counter = Counter()
    for t in tasks:
        for comp in component_apps(t.get("app_type")):
            component_counts[comp] += 1

    # sort apps by total descending
    app_difficulty_sorted = {
        app: dict(app_difficulty[app])
        for app in sorted(app_counts, key=lambda a: app_counts[a], reverse=True)
    }

    # --- multi-app task breakdown: how many apps involved ---
    multi_app_sizes: Counter = Counter()
    for t in tasks:
        app_type = t.get("app_type") or ""
        if "," in app_type:
            multi_app_sizes[len(app_type.split(","))] += 1

    return {
        "total_tasks": total,
        "total_unique_canonical_apps": len(app_counts),
        "total_unique_component_apps": len(component_counts),
        "difficulty_distribution": dict(difficulty_counts.most_common()),
        "app_distribution": dict(app_counts.most_common()),
        "app_difficulty_breakdown": app_difficulty_sorted,
        "component_app_mention_counts": dict(component_counts.most_common()),
        "multi_app_task_size_distribution": {
            str(k): v for k, v in sorted(multi_app_sizes.items())
        },
    }


def fmt_bar(count: int, total: int, width: int = 30) -> str:
    filled = round(count / total * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_report(stats: dict) -> None:
    total = stats["total_tasks"]
    sep = "=" * 70

    print(sep)
    print("CUA-Gym Task Dataset Analysis")
    print(sep)

    print(f"\nTotal tasks                  : {total:,}")
    print(f"Unique apps (canonical)      : {stats['total_unique_canonical_apps']}")
    print(f"Unique component apps        : {stats['total_unique_component_apps']}")

    print("\n--- Difficulty Distribution ---")
    for diff in KNOWN_DIFFICULTIES:
        count = stats["difficulty_distribution"].get(diff, 0)
        pct = count / total * 100
        bar = fmt_bar(count, total)
        print(f"  {diff:<10} {count:>6,}  ({pct:5.1f}%)  {bar}")
    # any unexpected difficulty labels
    for diff, count in stats["difficulty_distribution"].items():
        if diff not in KNOWN_DIFFICULTIES:
            print(f"  {diff:<10} {count:>6,}  ({count/total*100:5.1f}%)")

    print("\n--- App Distribution (canonical, tasks per app, top 30) ---")
    for app, count in list(stats["app_distribution"].items())[:30]:
        pct = count / total * 100
        print(f"  {app:<40} {count:>6,}  ({pct:5.1f}%)")
    remaining = len(stats["app_distribution"]) - 30
    if remaining > 0:
        tail_total = sum(list(stats["app_distribution"].values())[30:])
        print(f"  ... ({remaining} more apps, {tail_total:,} tasks total)")

    print("\n--- Multi-App Task Size Distribution ---")
    for size, count in stats["multi_app_task_size_distribution"].items():
        print(f"  {size} apps combined: {count:,} tasks")

    print("\n--- Per-App Difficulty Breakdown (top 20 apps) ---")
    diffs_to_show = KNOWN_DIFFICULTIES
    header = f"  {'app':<40} {'total':>6}  " + "  ".join(f"{d:>7}" for d in diffs_to_show)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for app, diff_map in list(stats["app_difficulty_breakdown"].items())[:20]:
        total_app = sum(diff_map.values())
        cols = "  ".join(f"{diff_map.get(d, 0):>7}" for d in diffs_to_show)
        print(f"  {app:<40} {total_app:>6}  {cols}")

    print("\n--- Individual Component-App Mention Counts (top 20) ---")
    for app, count in list(stats["component_app_mention_counts"].items())[:20]:
        pct = count / total * 100
        print(f"  {app:<40} {count:>6,}  ({pct:5.1f}%)")

    print(sep)


def save_results(stats: dict, output_dir: Path) -> None:
    total = stats["total_tasks"]

    # --- Full JSON ---
    json_path = output_dir / "task_analysis.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON summary : {json_path}")

    # --- Human-readable TXT ---
    txt_path = output_dir / "task_analysis_summary.txt"
    with open(txt_path, "w") as f:
        def w(line: str = "") -> None:
            f.write(line + "\n")

        sep = "=" * 70
        w(sep)
        w("CUA-Gym Task Dataset Analysis")
        w(sep)
        w()
        w(f"Total tasks                  : {total:,}")
        w(f"Unique apps (canonical)      : {stats['total_unique_canonical_apps']}")
        w(f"Unique component apps        : {stats['total_unique_component_apps']}")

        w()
        w("--- Difficulty Distribution ---")
        for diff in KNOWN_DIFFICULTIES:
            count = stats["difficulty_distribution"].get(diff, 0)
            pct = count / total * 100
            w(f"  {diff:<10} {count:>6,}  ({pct:5.1f}%)")

        w()
        w("--- App Distribution (canonical, tasks per app) ---")
        for app, count in stats["app_distribution"].items():
            pct = count / total * 100
            w(f"  {app:<40} {count:>6,}  ({pct:5.1f}%)")

        w()
        w("--- Multi-App Task Size Distribution ---")
        for size, count in stats["multi_app_task_size_distribution"].items():
            w(f"  {size} apps combined: {count:,} tasks")

        w()
        w("--- Per-App Difficulty Breakdown ---")
        diffs_to_show = KNOWN_DIFFICULTIES
        header = f"  {'app':<40} {'total':>6}  " + "  ".join(f"{d:>7}" for d in diffs_to_show)
        w(header)
        w("  " + "-" * (len(header) - 2))
        for app, diff_map in stats["app_difficulty_breakdown"].items():
            total_app = sum(diff_map.values())
            cols = "  ".join(f"{diff_map.get(d, 0):>7}" for d in diffs_to_show)
            w(f"  {app:<40} {total_app:>6}  {cols}")

        w()
        w("--- Individual Component-App Mention Counts ---")
        for app, count in stats["component_app_mention_counts"].items():
            pct = count / total * 100
            w(f"  {app:<40} {count:>6,}  ({pct:5.1f}%)")

    print(f"Saved TXT summary  : {txt_path}")

    # --- CSV: per canonical app with difficulty columns ---
    all_diffs = KNOWN_DIFFICULTIES + [
        d for d in {
            d for dm in stats["app_difficulty_breakdown"].values() for d in dm
        }
        if d not in KNOWN_DIFFICULTIES
    ]
    csv_path = output_dir / "task_analysis_per_app.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(["app", "total"] + all_diffs) + "\n")
        for app, diff_map in stats["app_difficulty_breakdown"].items():
            row = [app, str(sum(diff_map.values()))] + [
                str(diff_map.get(d, 0)) for d in all_diffs
            ]
            f.write(",".join(row) + "\n")
    print(f"Saved CSV per-app  : {csv_path}")

    # --- CSV: per component app mention counts ---
    comp_csv_path = output_dir / "task_analysis_component_apps.csv"
    with open(comp_csv_path, "w") as f:
        f.write("app,mention_count\n")
        for app, count in stats["component_app_mention_counts"].items():
            f.write(f"{app},{count}\n")
    print(f"Saved CSV comp-app : {comp_csv_path}")


def main() -> None:
    print(f"Loading tasks from: {TASKS_DIR}")
    tasks = load_all_tasks(TASKS_DIR)
    print(f"Loaded {len(tasks):,} tasks\n")

    stats = analyze(tasks)
    print_report(stats)
    save_results(stats, OUTPUT_DIR)


if __name__ == "__main__":
    main()
