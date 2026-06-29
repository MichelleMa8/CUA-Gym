#!/usr/bin/env python3
"""Run Codex trajectory analyses in parallel for CUA-Gym success trajectories.

Reads trajectories from data/success/trajectory/<app>/<task_id>/,
builds one prompt per trajectory, and calls `codex exec` concurrently.
Each result is written to data/success/analyzed/<app>/<task_id>/.
Trajectories whose result.md already exists in the output directory are skipped.

Usage:
    # Analyze up to 10 trajectories (4 in parallel)
    python scripts/analyze_trajectories.py --limit 10 --parallel 4

    # Analyze all, skipping already-analyzed ones (default behavior)
    python scripts/analyze_trajectories.py --limit 0 --parallel 8

    # Dry run: write prompts/metadata without calling Codex
    python scripts/analyze_trajectories.py --limit 10 --dry-run

    # Analyze specific task IDs
    python scripts/analyze_trajectories.py \\
        --task-id audacity_export_flac_mono \\
        --task-id gimp_rotate_layer

    # Use a custom prompt file
    python scripts/analyze_trajectories.py --prompt-file my_prompt.txt

Outputs are written to data/success/analyzed/<app>/<task_id>/:
    result.md         — Codex analysis markdown
    analysis.json     — parsed machine-readable analysis
    run.log           — stdout/stderr from the Codex process
    meta.json         — metadata and command for this trajectory
    prompt.txt        — exact prompt sent to Codex
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # CUA-Gym/
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "data" / "success" / "trajectory"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "success" / "analyzed"
DEFAULT_TASKS_DIR = PROJECT_ROOT / "data" / "cua_gym_all" / "cua_gym_tasks"

JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
USAGE_RE = re.compile(
    r"(\d[\d,]*)\s*(?:prompt\s+)?(?:input\s+)?tokens?[^.]*?(\d[\d,]*)\s*(?:completion\s+)?(?:output\s+)?tokens?",
    re.IGNORECASE | re.DOTALL,
)
COST_RE = re.compile(r"(?:cost|total)[:\s]+\$\s*([\d.]+)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Model pricing table (USD per 1M tokens)
# ---------------------------------------------------------------------------
# fmt: off
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_1M, output_per_1M)

    # --- GPT-5 series ---
    "gpt-5.5":              (5.00,   30.00),
    "gpt-5.5-pro":          (30.00,  180.00),
    "gpt-5.4":              (2.50,   15.00),
    "gpt-5.4-mini":         (0.75,   4.50),
    "gpt-5.4-nano":         (0.20,   1.25),
    "gpt-5.4-pro":          (30.00,  180.00),
    "gpt-5.3-codex":        (1.75,   14.00),

    # --- GPT-4.1 series ---
    "gpt-4.1":              (2.00,   8.00),
    "gpt-4.1-mini":         (0.40,   1.60),
    "gpt-4.1-nano":         (0.10,   0.40),

    # --- GPT-4o series ---
    "gpt-4o":               (2.50,   10.00),
    "gpt-4o-mini":          (0.15,   0.60),

    # --- o-series reasoning models ---
    "o4-mini":              (1.10,   4.40),
    "o3":                   (2.00,   8.00),
    "o3-mini":              (1.10,   4.40),
    "o1":                   (15.00,  60.00),
    "o1-mini":              (1.10,   4.40),

    # --- Specialized ---
    "computer-use-preview": (1.50,   6.00),
}
# fmt: on

CODEX_DEFAULT_MODEL = "gpt-5.3-codex"


def get_model_pricing(model: str | None) -> tuple[float, float]:
    if not model:
        model = CODEX_DEFAULT_MODEL
    model = model.strip()
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    candidates = [(k, v) for k, v in MODEL_PRICING.items() if model.startswith(k)]
    if candidates:
        return max(candidates, key=lambda x: len(x[0]))[1]
    sys.stderr.write(
        f"Warning: unknown model '{model}', falling back to {CODEX_DEFAULT_MODEL} pricing.\n"
    )
    return MODEL_PRICING[CODEX_DEFAULT_MODEL]


DEFAULT_REVIEW_PROMPT = """You are auditing one desktop automation trajectory from CUA-Gym.

Goal:
- Decide whether the successful trajectory contains any obvious detour,
  redundant action, recovery loop, or avoidable exploration.
- Judge only the trajectory efficiency. Do not re-evaluate task correctness
  except as context.
- Use the task instruction and action transcript. If needed, inspect the
  referenced actions.json and images directory.
- Do not modify files.

Important labeling rule:
- If any specific step contains a detour or redundant action, label it
  explicitly by step number from the compact action transcript.
- If a detour spans several steps, use a step range such as "12-18".
- Label only the step(s) that create the mistake, detour, redundancy, or failed
  path. Do not label later step(s) whose purpose is to recover from or correct
  that earlier mistake.
- Keep labeled ranges limited to the mistake-causing actions. If step 12 takes a
  wrong path and steps 13-15 return to the correct state, label step 12 only.
- For repeated failed attempts, label the repeated mistaken attempts themselves,
  but exclude the eventual successful correction.
- Only label obvious detours. Do not label necessary waiting, normal task
  progress, or verification/checking as detours.

The final answer is saved as this trajectory's analysis result. Return concise
Markdown with these exact sections:

Task:
Verdict: Clean / Minor detour / Major detour / Unclear
Obvious detour?: yes/no
Estimated redundant tool calls:
Evidence:
- step(s): explanation
Detour step labels:
| step_or_range | label | severity | redundant_tool_calls | reason |
|---|---|---|---:|---|
Frequency tags:
- wait_loop: N
- repeated_ui_action: N
- failed_path_or_retry: N
One-sentence summary:

Machine-readable summary:
```json
{
  "task_id": "<task id>",
  "domain": "<app name>",
  "verdict": "Clean | Minor detour | Major detour | Unclear",
  "obvious_detour": true,
  "estimated_redundant_tool_calls": 0,
  "detour_steps": [
    {
      "step_or_range": "12-18",
      "label": "wait_loop | repeated_ui_action | failed_path_or_retry | other",
      "severity": "minor | major",
      "redundant_tool_calls": 3,
      "reason": "short explanation"
    }
  ],
  "frequency_tags": {
    "wait_loop": 0,
    "repeated_ui_action": 0,
    "failed_path_or_retry": 0
  },
  "summary": "one sentence"
}
```
"""


@dataclass(frozen=True)
class Trajectory:
    app: str        # subdirectory name under trajectory/ (e.g. "audacity")
    task_id: str    # subdirectory name under app/ (e.g. "audacity_export_flac_mono")
    directory: Path
    reward: Any
    instruction: str
    action_count: int


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value: Any, limit: int = 220) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def discover_trajectories(
    traj_root: Path,
    tasks_dir: Path | None,
) -> list[Trajectory]:
    """Walk traj_root/<app>/<task_id>/ and collect Trajectory objects."""
    trajectories: list[Trajectory] = []

    for app_dir in sorted(traj_root.iterdir()):
        if not app_dir.is_dir():
            continue
        app = app_dir.name

        for task_dir in sorted(app_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name

            summary_file = task_dir / "task_summary.json"
            if not summary_file.exists():
                continue
            try:
                summary = read_json(summary_file)
            except (OSError, json.JSONDecodeError):
                continue

            reward = summary.get("reward")

            # Look up instruction from the CUA-Gym task bundle
            instruction = ""
            if tasks_dir:
                task_json_path = tasks_dir / task_id / "task.json"
                if task_json_path.exists():
                    try:
                        task_data = read_json(task_json_path)
                        instruction = task_data.get(
                            "instruction", task_data.get("task_instruction", "")
                        )
                    except (OSError, json.JSONDecodeError):
                        pass

            # Count actions
            actions_file = task_dir / "actions.json"
            action_count = 0
            if actions_file.exists():
                try:
                    actions = read_json(actions_file)
                    action_count = len(actions) if isinstance(actions, list) else 0
                except (OSError, json.JSONDecodeError):
                    pass

            trajectories.append(
                Trajectory(
                    app=app,
                    task_id=task_id,
                    directory=task_dir,
                    reward=reward,
                    instruction=instruction,
                    action_count=action_count,
                )
            )

    return trajectories


def _format_tool_call(tool_call: dict) -> str:
    """Format a CUA-Gym tool_call dict into a compact readable string."""
    tool_name = tool_call.get("tool_name") or "unknown_action"
    params: list[str] = []
    for key in [
        "x", "y", "start_x", "start_y", "end_x", "end_y",
        "content", "text", "keys", "key", "hold_key", "tap_key",
        "direction", "amount", "seconds", "button",
    ]:
        val = tool_call.get(key)
        if val is None:
            continue
        val_str = str(val)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        params.append(f"{key}={val_str}")
    if params:
        return f"{tool_name}({', '.join(params)})"
    return tool_name


def build_transcript(traj_dir: Path, max_steps: int) -> str:
    """Build a compact, human-readable action transcript from actions.json."""
    actions_file = traj_dir / "actions.json"
    try:
        actions = read_json(actions_file)
    except (OSError, json.JSONDecodeError) as exc:
        return f"[could not read actions.json: {exc}]"

    if not isinstance(actions, list) or not actions:
        return "[no steps found in trajectory]"

    lines: list[str] = []
    total = len(actions)
    for record in actions:
        if len(lines) >= max_steps:
            remaining = total - len(lines)
            lines.append(f"... [{remaining} more steps omitted from prompt]")
            break

        # steps are 0-indexed in CUA-Gym; display as 1-indexed
        raw_step = record.get("step", len(lines))
        step_num = raw_step + 1

        action_text = record.get("action", "")
        try:
            step = json.loads(action_text) if action_text else {}
        except json.JSONDecodeError:
            step = {}

        tool_call = step.get("tool_call") or {}
        thought = step.get("thought") or step.get("note") or ""

        action_str = _format_tool_call(tool_call)[:90] if tool_call else "[no action]"
        reasoning = compact(thought, limit=260)

        line = f"{step_num:03d}. {action_str}"
        if reasoning:
            line += f"\n     note: {reasoning}"
        lines.append(line)

    return "\n".join(lines)


def build_prompt(
    base_prompt: str,
    trajectory: Trajectory,
    max_steps: int,
) -> str:
    actions_json_path = trajectory.directory / "actions.json"
    images_path = trajectory.directory / "images"
    transcript = build_transcript(trajectory.directory, max_steps=max_steps)
    instruction = trajectory.instruction or "[instruction not found]"
    return f"""{base_prompt.rstrip()}

Trajectory metadata:
- app: {trajectory.app}
- task_id: {trajectory.task_id}
- reward: {trajectory.reward}
- action_count: {trajectory.action_count}
- trajectory_dir: {trajectory.directory}
- actions_json: {actions_json_path}
- images_dir: {images_path}

Task instruction:
{instruction}

Compact action transcript:
{transcript}
"""


def shell_quote_for_log(args: list[str]) -> str:
    return " ".join(
        json.dumps(arg) if any(ch.isspace() for ch in arg) else arg
        for arg in args
    )


def extract_json_block(markdown: str) -> tuple[dict[str, Any] | None, str | None]:
    errors: list[str] = []
    for block in reversed(JSON_BLOCK_RE.findall(markdown)):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed, None
        errors.append(f"JSON block was {type(parsed).__name__}, expected object")
    if not errors:
        return None, "No fenced ```json block found in Codex output."
    return None, "; ".join(errors)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_usage_from_output(stdout: bytes, stderr: bytes) -> dict[str, Any]:
    combined = (stdout + stderr).decode("utf-8", errors="replace")
    result: dict[str, Any] = {}
    cost_match = COST_RE.search(combined)
    if cost_match:
        result["actual_cost_usd"] = float(cost_match.group(1))
    usage_match = USAGE_RE.search(combined)
    if usage_match:
        result["actual_input_tokens"] = int(usage_match.group(1).replace(",", ""))
        result["actual_output_tokens"] = int(usage_match.group(2).replace(",", ""))
    return result


def build_cost_info(
    prompt: str,
    response_text: str,
    stdout: bytes,
    stderr: bytes,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> dict[str, Any]:
    est_in = _estimate_tokens(prompt)
    est_out = _estimate_tokens(response_text)
    est_cost = (est_in * input_price_per_1m + est_out * output_price_per_1m) / 1_000_000

    info: dict[str, Any] = {
        "estimated_input_tokens": est_in,
        "estimated_output_tokens": est_out,
        "estimated_cost_usd": round(est_cost, 6),
        "pricing_model": {
            "input_per_1m_usd": input_price_per_1m,
            "output_per_1m_usd": output_price_per_1m,
        },
    }
    actual = _parse_usage_from_output(stdout, stderr)
    if actual:
        info.update(actual)
        if "actual_cost_usd" not in actual and "actual_input_tokens" in actual:
            cost = (
                actual["actual_input_tokens"] * input_price_per_1m
                + actual["actual_output_tokens"] * output_price_per_1m
            ) / 1_000_000
            info["actual_cost_usd"] = round(cost, 6)
    return info


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_analysis_json(
    markdown_path: Path,
    analysis_json_path: Path,
    trajectory: Trajectory,
    cost_info: dict[str, Any] | None = None,
) -> None:
    base: dict[str, Any] = {
        "task_id": trajectory.task_id,
        "domain": trajectory.app,
        "reward": trajectory.reward,
        "action_count": trajectory.action_count,
        "trajectory_dir": str(trajectory.directory),
        "_source_markdown": str(markdown_path),
    }
    if cost_info:
        base["cost"] = cost_info

    if not markdown_path.exists():
        base.update(
            {
                "_parse_status": "missing_markdown",
                "_parse_error": "Codex did not produce the Markdown output file.",
            }
        )
        analysis_json_path.write_text(
            json.dumps(base, indent=2) + "\n", encoding="utf-8"
        )
        return

    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    parsed, error = extract_json_block(markdown)
    if parsed is None:
        base.update(
            {
                "_parse_status": "failed",
                "_parse_error": error,
                "analysis_markdown": markdown,
            }
        )
        analysis_json_path.write_text(
            json.dumps(base, indent=2) + "\n", encoding="utf-8"
        )
        return

    result = {**base, **parsed}
    result["_parse_status"] = "ok"
    analysis_json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def run_one(
    trajectory: Trajectory,
    args: argparse.Namespace,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    # Output under analyzed/<app>/<task_id>/
    traj_out = args.output_dir / trajectory.app / trajectory.task_id
    traj_out.mkdir(parents=True, exist_ok=True)
    final_path = traj_out / "result.md"
    analysis_json_path = traj_out / "analysis.json"
    log_path = traj_out / "run.log"
    meta_path = traj_out / "meta.json"
    prompt_path = traj_out / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # Skip if result already exists (resume mode)
    if args.resume and final_path.exists():
        write_analysis_json(final_path, analysis_json_path, trajectory)
        return {
            "task_id": trajectory.task_id,
            "app": trajectory.app,
            "status": "skipped",
            "output": str(final_path),
            "analysis_json": str(analysis_json_path),
            "prompt": str(prompt_path),
        }

    cmd = [
        args.codex_bin,
        "exec",
        "--cd",
        str(PROJECT_ROOT),
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(final_path),
    ]
    if args.ephemeral:
        cmd.append("--ephemeral")
    if args.model:
        cmd.extend(["--model", args.model])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    for config in args.config:
        cmd.extend(["--config", config])
    cmd.extend(args.codex_arg)
    cmd.append("-")

    meta_path.write_text(
        json.dumps(
            {
                "app": trajectory.app,
                "task_id": trajectory.task_id,
                "reward": trajectory.reward,
                "action_count": trajectory.action_count,
                "trajectory_dir": str(trajectory.directory),
                "prompt": str(prompt_path),
                "analysis_json": str(analysis_json_path),
                "command": shell_quote_for_log(cmd),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        return {
            "task_id": trajectory.task_id,
            "app": trajectory.app,
            "status": "dry-run",
            "output": str(final_path),
            "analysis_json": str(analysis_json_path),
            "prompt": str(prompt_path),
            "log": str(log_path),
        }

    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(prompt.encode("utf-8"))
        log_path.write_bytes(
            b"STDOUT\n======\n" + stdout + b"\n\nSTDERR\n======\n" + stderr
        )

    response_text = (
        final_path.read_text(encoding="utf-8", errors="replace")
        if final_path.exists()
        else ""
    )
    input_price, output_price = get_model_pricing(args.model)
    cost_info = build_cost_info(
        prompt, response_text, stdout, stderr, input_price, output_price
    )
    write_analysis_json(final_path, analysis_json_path, trajectory, cost_info)

    return {
        "task_id": trajectory.task_id,
        "app": trajectory.app,
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "output": str(final_path),
        "analysis_json": str(analysis_json_path),
        "prompt": str(prompt_path),
        "log": str(log_path),
        "cost": cost_info,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Codex in parallel to audit CUA-Gym success trajectories.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help=f"Root of the trajectory dataset (default: {DEFAULT_TRAJECTORY_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root of the analyzed output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help=f"CUA-Gym task bundles directory for looking up instructions "
             f"(default: {DEFAULT_TASKS_DIR})",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=4,
        help="Number of concurrent Codex processes (default: 4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of trajectories to analyze. Use 0 for all (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260615,
        help="Random seed used when sampling trajectories (default: 20260615).",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Use deterministic sorted order instead of random sampling.",
    )
    parser.add_argument(
        "--app",
        action="append",
        default=[],
        help="Restrict to a specific app name (e.g. audacity). Can be repeated.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Analyze a specific task id. Can be repeated. Overrides random sampling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text to give each Codex model before trajectory metadata.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="File containing the prompt to give each Codex model.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=160,
        help="Maximum transcript steps embedded in each prompt (default: 160).",
    )
    # Codex invocation
    parser.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Extra codex --config override. Can be repeated.",
    )
    parser.add_argument(
        "--codex-arg",
        action="append",
        default=[],
        help="Extra raw argument passed to codex exec. Can be repeated.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-analyze trajectories even if result.md already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select trajectories and write prompts/metadata without calling Codex.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        default=True,
        help="Run Codex sessions without persisting session files (default: on).",
    )
    parser.add_argument(
        "--no-ephemeral",
        dest="ephemeral",
        action="store_false",
        help="Allow Codex to persist session files.",
    )
    return parser.parse_args()


def select_trajectories(
    trajectories: list[Trajectory],
    task_ids: list[str],
    app_filter: list[str],
    limit: int,
    seed: int,
    ordered: bool,
) -> list[Trajectory]:
    import random

    if task_ids:
        wanted = set(task_ids)
        selected = [t for t in trajectories if t.task_id in wanted]
        missing = sorted(wanted - {t.task_id for t in selected})
        if missing:
            raise SystemExit(
                f"Task id(s) not found in trajectory directory: {', '.join(missing)}"
            )
        return selected

    selected = list(trajectories)
    if app_filter:
        wanted_apps = set(app_filter)
        selected = [t for t in selected if t.app in wanted_apps]

    if not ordered:
        rng = random.Random(seed)
        rng.shuffle(selected)

    if limit and limit > 0:
        selected = selected[:limit]

    return selected


def print_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    width = 32
    filled = round(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = round(100 * done / total)
    sys.stderr.write(f"\rProcessing: [{bar}] {done}/{total} {percent:3d}%")
    sys.stderr.flush()


async def async_main() -> int:
    args = parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")

    args.trajectory_dir = args.trajectory_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks_dir = args.tasks_dir.expanduser().resolve() if args.tasks_dir else None
    if tasks_dir and not tasks_dir.exists():
        sys.stderr.write(
            f"Warning: tasks-dir not found ({tasks_dir}); instructions will be empty.\n"
        )
        tasks_dir = None

    # resume=True unless --no-resume is set
    args.resume = not args.no_resume

    if args.prompt_file:
        base_prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt:
        base_prompt = args.prompt
    else:
        base_prompt = DEFAULT_REVIEW_PROMPT

    # Collect all trajectories
    trajectories = discover_trajectories(args.trajectory_dir, tasks_dir)

    # Find already-analyzed task ids (by app/task_id pair) to skip by default
    already_analyzed: set[tuple[str, str]] = set()
    if args.resume and args.output_dir.exists():
        for app_dir in args.output_dir.iterdir():
            if not app_dir.is_dir():
                continue
            for task_dir in app_dir.iterdir():
                if task_dir.is_dir() and (task_dir / "result.md").exists():
                    already_analyzed.add((app_dir.name, task_dir.name))

    # Apply user selection (task IDs / app filter / limit / seed)
    selected = select_trajectories(
        trajectories,
        task_ids=args.task_id,
        app_filter=args.app,
        limit=args.limit,
        seed=args.seed,
        ordered=args.ordered,
    )

    # Filter out already-analyzed trajectories (unless --no-resume)
    if already_analyzed and args.resume:
        before = len(selected)
        selected = [t for t in selected if (t.app, t.task_id) not in already_analyzed]
        skipped_count = before - len(selected)
        if skipped_count:
            print(
                f"Skipping {skipped_count} already-analyzed trajectory(ies). "
                f"Use --no-resume to re-analyze them.",
                flush=True,
            )

    if not selected:
        print("No trajectories to analyze (all done or none matched).", flush=True)
        return 0

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trajectory_dir": str(args.trajectory_dir),
                "parallel": args.parallel,
                "limit": args.limit,
                "seed": args.seed,
                "ordered": args.ordered,
                "resume": args.resume,
                "count": len(selected),
                "pricing": {
                    "model": args.model or CODEX_DEFAULT_MODEL,
                    "input_per_1m_usd": get_model_pricing(args.model)[0],
                    "output_per_1m_usd": get_model_pricing(args.model)[1],
                },
                "tasks": [
                    {
                        "app": t.app,
                        "task_id": t.task_id,
                        "reward": t.reward,
                        "action_count": t.action_count,
                    }
                    for t in selected
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Selected {len(selected)} trajectories from {len(trajectories)} candidates.",
        flush=True,
    )
    print(f"Writing outputs to {args.output_dir}", flush=True)
    print(f"Parallel Codex processes: {args.parallel}", flush=True)
    _in_price, _out_price = get_model_pricing(args.model)
    print(
        f"Cost model: {args.model or CODEX_DEFAULT_MODEL} — "
        f"${_in_price}/1M input, ${_out_price}/1M output (estimated)",
        flush=True,
    )

    semaphore = asyncio.Semaphore(args.parallel)

    async def run_indexed(
        index: int, trajectory: Trajectory
    ) -> tuple[int, dict[str, Any]]:
        result = await run_one(
            trajectory,
            args,
            build_prompt(base_prompt, trajectory, args.max_steps),
            semaphore,
        )
        return index, result

    tasks = [
        asyncio.create_task(run_indexed(i, t)) for i, t in enumerate(selected)
    ]
    results_by_index: list[dict[str, Any] | None] = [None] * len(tasks)
    done_count = 0
    print_progress(0, len(tasks))
    for task in asyncio.as_completed(tasks):
        index, result = await task
        results_by_index[index] = result
        done_count += 1
        print_progress(done_count, len(tasks))
    sys.stderr.write("\n")

    results = [r for r in results_by_index if r is not None]
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    dry_run_results = [r for r in results if r["status"] == "dry-run"]
    ok_count = len(results) - len(failed) - len(skipped) - len(dry_run_results)

    # Cost summary
    costs = [r["cost"]["estimated_cost_usd"] for r in results if "cost" in r]
    actual_costs = [
        r["cost"]["actual_cost_usd"]
        for r in results
        if "cost" in r and "actual_cost_usd" in r["cost"]
    ]
    if costs:
        print(
            f"\nEstimated cost: ${sum(costs):.4f} total "
            f"(avg ${sum(costs)/len(costs):.4f}/trajectory)"
        )
    if actual_costs:
        print(
            f"Actual cost:    ${sum(actual_costs):.4f} total "
            f"(avg ${sum(actual_costs)/len(actual_costs):.4f}/trajectory)"
        )

    # Per-app cost breakdown
    if costs:
        from collections import defaultdict

        app_costs: dict[str, list[float]] = defaultdict(list)
        for r in results:
            if "cost" in r:
                app_costs[r["app"]].append(r["cost"]["estimated_cost_usd"])
        print("\nEstimated cost by app:")
        for app, dcosts in sorted(app_costs.items()):
            print(f"  {app:<30} ${sum(dcosts):.4f}  ({len(dcosts)} tasks)")

    print(
        f"\nDone: {ok_count} ok, {len(skipped)} skipped, "
        f"{len(dry_run_results)} dry-run, {len(failed)} failed."
    )
    print(f"Summary: {summary_path}")
    return 1 if failed else 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
