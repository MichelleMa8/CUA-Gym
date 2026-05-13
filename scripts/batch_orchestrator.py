#!/usr/bin/env python3
"""
Batch Orchestrator Runner for CUA-Gym

Processes task JSON files through the orchestrator agent in parallel,
with concurrency control, progress tracking, and auto-resume.

Usage:
    # Run all calc tasks (default concurrency=3)
    python3 scripts/batch_orchestrator.py output/task_generation/calc_*.json

    # Higher concurrency (limited by VM budget)
    python3 scripts/batch_orchestrator.py -c 5 output/task_generation/calc_formatting.json

    # Filter specific tasks
    python3 scripts/batch_orchestrator.py --filter "fmt_bold" output/task_generation/calc_*.json

    # Dry run (see what would be processed)
    python3 scripts/batch_orchestrator.py --dry-run output/task_generation/calc_*.json

    # Retry only failed tasks
    python3 scripts/batch_orchestrator.py --retry-failed output/task_generation/calc_*.json

    # Specific task by ID
    python3 scripts/batch_orchestrator.py --task-id calc_fmt_bold_001 output/task_generation/calc_formatting.json
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = PROJECT_ROOT / "output" / "batch_status.json"
LOG_DIR = PROJECT_ROOT / "output" / "logs"
FINAL_DIR = PROJECT_ROOT / "output" / "final"
ENV_FILE = PROJECT_ROOT / ".env"

# Prepend osworld venv to PATH so all agent `python3` calls use it
_VENV_BIN = os.path.expanduser("~/.venvs/osworld-py312/bin")
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _VENV_BIN + ":" + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env():
    """Load .env file into os.environ (simple key=value parser)."""
    if not ENV_FILE.exists():
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip 'export ' prefix if present
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ[key] = value


def load_tasks(file_paths: list[str]) -> list[dict]:
    """Load all tasks from multiple JSON files."""
    def infer_domain(task: dict, source_path: str) -> str:
        # 1) Honor explicit domain from task payload when present.
        explicit = (task.get("domain") or "").strip()
        if explicit:
            return explicit

        # 2) Infer from task_id / source filename to avoid calc fallback leaks.
        task_id = str(task.get("task_id", "")).lower()
        source_name = Path(source_path).name.lower()
        hint = f"{task_id} {source_name}"

        if "writer" in hint:
            return "libreoffice_writer"
        if "impress" in hint or "ppt" in hint:
            return "libreoffice_impress"
        if "calc" in hint:
            return "libreoffice_calc"

        # 3) Conservative default.
        return "libreoffice_calc"

    tasks = []
    for fp in file_paths:
        fp = str(fp)
        try:
            with open(fp) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping {fp}: JSON parse error: {e}")
            continue
        if isinstance(data, dict) and "tasks" in data:
            data = data["tasks"]
        if not isinstance(data, list):
            data = [data]
        for i, task in enumerate(data):
            tasks.append({
                "source_file": os.path.relpath(fp, PROJECT_ROOT),
                "index": i,
                "task_id": task.get("task_id", f"{os.path.splitext(os.path.basename(fp))[0]}_{i:03d}"),
                "domain": infer_domain(task, fp),
                "task_payload": task,
            })
    return tasks


def load_status() -> dict:
    """Load batch status from disk."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def save_status(status: dict):
    """Save batch status to disk."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.rename(STATUS_FILE)


def task_is_complete(task_id: str, status: dict) -> bool:
    """Check if a task is already completed (has final outputs)."""
    # Check status file
    if status.get(task_id, {}).get("status") == "completed":
        return True
    # Also check if final outputs exist on disk
    final = FINAL_DIR / task_id
    if final.exists() and (final / "config.json").exists() and (final / "reward.py").exists():
        return True
    return False

# ---------------------------------------------------------------------------
# Core: run one task
# ---------------------------------------------------------------------------

async def run_task(
    task: dict,
    semaphore: asyncio.Semaphore,
    status: dict,
    args: argparse.Namespace,
):
    """Run orchestrator for a single task."""
    task_id = task["task_id"]
    domain = task["domain"]
    source = task["source_file"]
    index = task["index"]
    task_payload = task["task_payload"]

    # Skip completed
    if task_is_complete(task_id, status) and not args.force:
        return "skipped"

    # Skip non-failed if retry_failed mode
    if args.retry_failed:
        prev = status.get(task_id, {}).get("status", "")
        if prev not in ("failed", "error", "timeout"):
            return "skipped"

    async with semaphore:
        ts_start = datetime.now()
        print(f"  [{ts_start.strftime('%H:%M:%S')}] START  {task_id}  "
              f"(from {source}[{index}])")

        # Update status
        status[task_id] = {
            "status": "running",
            "source_file": source,
            "index": index,
            "domain": domain,
            "started_at": ts_start.isoformat(),
            "attempt": status.get(task_id, {}).get("attempt", 0) + 1,
        }
        save_status(status)

        if args.dry_run:
            print(f"  [DRY]   {task_id} — would run orchestrator")
            status[task_id]["status"] = "dry_run"
            save_status(status)
            return "dry_run"

        # Construct prompt:
        # Pass the selected task payload directly to avoid expensive reads of
        # large task_generation JSON files inside the agent session.
        task_payload_json = json.dumps(task_payload, ensure_ascii=False, indent=2)
        prompt = (
            f"Process tasks for domain: {domain}\n"
            f"Input file: {source}\n"
            f"Task index: {index}\n\n"
            "Selected task payload (authoritative):\n"
            f"{task_payload_json}\n\n"
            "IMPORTANT:\n"
            "- Use the task payload above as the selected task.\n"
            "- Do NOT read output/task_generation/*.json files.\n"
            "- Do NOT ask clarifying questions.\n"
            "- Execute the full orchestrator pipeline immediately."
        )

        # Build command
        cmd = [
            "claude",
            "--agent", "orchestrator",
            "-p", prompt,
            "--max-turns", str(args.max_turns),
            "--output-format", "stream-json",
            "--verbose",
        ]
        if args.model:
            cmd += ["--model", args.model]
        if args.dangerously_skip_permissions:
            cmd += ["--permission-mode", "dontAsk"]
        else:
            # Use "Task" (server-side name) not "Agent" (Mac name) for sub-agent spawning
            cmd += ["--allowedTools",
                    "Agent,Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch"]

        # Log file (stream-json for full trace, readable summary separate)
        log_file = LOG_DIR / f"{task_id}.jsonl"
        err_file = LOG_DIR / f"{task_id}.stderr.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        result_status = "failed"
        max_retries = args.api_retries
        for attempt_num in range(max_retries + 1):
            if attempt_num > 0:
                wait_secs = min(30 * attempt_num, 120)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] RETRY  {task_id}  "
                      f"(attempt {attempt_num + 1}/{max_retries + 1}, "
                      f"waiting {wait_secs}s)")
                await asyncio.sleep(wait_secs)

            # Use attempt-suffixed log files so each attempt is preserved
            if attempt_num == 0:
                attempt_log = log_file
                attempt_err = err_file
            else:
                attempt_log = LOG_DIR / f"{task_id}.attempt{attempt_num}.jsonl"
                attempt_err = LOG_DIR / f"{task_id}.attempt{attempt_num}.stderr.log"

            try:
                with open(attempt_log, "w") as lf, open(attempt_err, "w") as ef:
                    ef.write(f"=== {task_id} (attempt {attempt_num + 1}/{max_retries + 1}) ===\n")
                    ef.write(f"Command: {cmd[0]} {cmd[1]} {cmd[2]} -p '...'\n")
                    ef.write(f"Prompt: {prompt}\n")
                    ef.write(f"Started: {ts_start.isoformat()}\n")
                    ef.write("=" * 60 + "\n\n")
                    ef.flush()

                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=lf,
                        stderr=ef,
                        cwd=str(PROJECT_ROOT),
                    )

                    try:
                        await asyncio.wait_for(
                            proc.wait(),
                            timeout=args.timeout * 60,  # minutes → seconds
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        result_status = "timeout"
                        ef.write(f"\n\n=== TIMEOUT after {args.timeout} minutes ===\n")

                # Check result
                if result_status == "timeout":
                    break  # Don't retry on timeout

                final = FINAL_DIR / task_id
                if (final.exists()
                        and (final / "config.json").exists()
                        and (final / "reward.py").exists()
                        and (final / "initial_setup.py").exists()
                        and (final / "golden_patch.py").exists()):
                    result_status = "completed"
                    break  # Success — stop retrying
                else:
                    result_status = "failed"
                    status[task_id]["exit_code"] = proc.returncode
                    # Read stderr for error details
                    try:
                        stderr_content = Path(attempt_err).read_text()
                        # Extract last meaningful lines
                        lines = [l for l in stderr_content.strip().split('\n')
                                 if l.strip() and not l.startswith('===')]
                        if lines:
                            status[task_id]["last_error"] = '\n'.join(lines[-5:])
                    except Exception:
                        pass
                    # Symlink latest attempt logs for easy access
                    if attempt_num > 0:
                        for src, dst in [(attempt_log, log_file), (attempt_err, err_file)]:
                            try:
                                dst.unlink(missing_ok=True)
                                dst.symlink_to(src.name)
                            except Exception:
                                pass

            except Exception as e:
                result_status = "error"
                status[task_id]["error"] = str(e)
                break  # Don't retry on unexpected exceptions

        ts_end = datetime.now()
        duration = (ts_end - ts_start).total_seconds()
        status[task_id]["status"] = result_status
        status[task_id]["finished_at"] = ts_end.isoformat()
        status[task_id]["duration_seconds"] = round(duration)
        save_status(status)

        # Print error hint on failure
        if result_status in ("failed", "error", "timeout"):
            hint = status[task_id].get("last_error", status[task_id].get("error", ""))
            if hint:
                # Show last line of error
                last_line = hint.strip().split('\n')[-1][:120]
                print(f"          └─ {last_line}")
            print(f"          └─ Log: {err_file}")

        icon = {"completed": "✓", "failed": "✗", "timeout": "⏰", "error": "!"}.get(
            result_status, "?"
        )
        print(f"  [{ts_end.strftime('%H:%M:%S')}] {icon} {result_status.upper():9s} "
              f"{task_id}  ({duration:.0f}s)")

        return result_status

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Batch orchestrator runner for CUA-Gym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*",
        help="Task JSON files (glob-expanded by shell)",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=3,
        help="Max parallel tasks (default: 3, limited by VM budget)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=200,
        help="Max Claude turns per task (default: 200)",
    )
    parser.add_argument(
        "--timeout", type=int, default=240,
        help="Timeout per task in minutes (default: 45)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to use (default: inherit from project)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run tasks whose task_id contains this string",
    )
    parser.add_argument(
        "--task-id", type=str, default=None,
        help="Run only this specific task_id",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Only retry tasks with failed/error/timeout status",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even completed tasks",
    )
    parser.add_argument(
        "--api-retries", type=int, default=3,
        help="Max retries per task on API failure (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--dangerously-skip-permissions", action="store_true",
        help="Skip all Claude permission prompts (use with caution)",
    )
    args = parser.parse_args()

    # Load environment
    load_env()

    # Resolve file paths
    if not args.files:
        # Default: all calc task files
        args.files = sorted(
            str(p) for p in (PROJECT_ROOT / "output" / "task_generation").glob("calc_*.json")
        )
        if not args.files:
            print("Error: No task files found. Specify files or check output/task_generation/")
            sys.exit(1)

    # Load tasks
    tasks = load_tasks(args.files)

    # Apply filters
    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
    elif args.filter:
        tasks = [t for t in tasks if args.filter in t["task_id"]]

    if not tasks:
        print("No tasks match the filter criteria.")
        sys.exit(0)

    # Load status
    status = load_status()

    # Count
    total = len(tasks)
    already_done = sum(1 for t in tasks if task_is_complete(t["task_id"], status))
    pending = total - already_done

    # Summary
    sources = sorted(set(t["source_file"] for t in tasks))
    print("=" * 60)
    print("CUA-Gym Batch Orchestrator")
    print("=" * 60)
    print(f"  Task files:    {len(sources)}")
    print(f"  Total tasks:   {total}")
    print(f"  Completed:     {already_done}")
    print(f"  Pending:       {pending}")
    print(f"  Concurrency:   {args.concurrency}")
    print(f"  Timeout:       {args.timeout} min/task")
    print(f"  API retries:   {args.api_retries}")
    print(f"  Max turns:     {args.max_turns}")
    print(f"  Model:         {args.model or '(default)'}")
    print(f"  Status file:   {STATUS_FILE}")
    print(f"  Log dir:       {LOG_DIR}")
    if args.dry_run:
        print(f"  Mode:          DRY RUN")
    if args.retry_failed:
        print(f"  Mode:          RETRY FAILED ONLY")
    print("=" * 60)

    if pending == 0 and not args.force and not args.retry_failed:
        print("\nAll tasks already completed! Use --force to re-run.")
        sys.exit(0)

    # Run
    print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    results = await asyncio.gather(*[
        run_task(t, semaphore, status, args) for t in tasks
    ])

    elapsed = time.time() - t0

    # Final summary
    from collections import Counter
    counts = Counter(results)
    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Completed:  {counts.get('completed', 0)}")
    print(f"  Failed:     {counts.get('failed', 0)}")
    print(f"  Timeout:    {counts.get('timeout', 0)}")
    print(f"  Error:      {counts.get('error', 0)}")
    print(f"  Skipped:    {counts.get('skipped', 0)}")
    print(f"  Dry run:    {counts.get('dry_run', 0)}")
    print(f"  Elapsed:    {elapsed/60:.1f} minutes")
    print(f"  Status:     {STATUS_FILE}")

    # List failures
    failed_ids = [
        t["task_id"] for t, r in zip(tasks, results)
        if r in ("failed", "error", "timeout")
    ]
    if failed_ids:
        print(f"\nFailed tasks ({len(failed_ids)}):")
        for tid in failed_ids[:20]:
            reason = status.get(tid, {}).get("status", "unknown")
            print(f"  - {tid} ({reason})")
        if len(failed_ids) > 20:
            print(f"  ... and {len(failed_ids) - 20} more")
        print(f"\nRetry with: python3 scripts/batch_orchestrator.py --retry-failed {' '.join(args.files)}")


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def _sigint(sig, frame):
        print("\n\nInterrupted! Progress saved to batch_status.json")
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)

    asyncio.run(main())
