#!/usr/bin/env python3
"""
Collect trajectories on CUA-Gym task bundles using a pluggable model backend
(Holo-3.1-35B-A3B by default, or MiniMax M3).

Usage:
    # Run tasks selected by select_tasks.py (defaults to Holo-3.1)
    python scripts/run_trajectories.py --run-dir data/run-20260620_185032

    # Override vLLM endpoint (env var or CLI flag)
    python scripts/run_trajectories.py --run-dir ... --model-url http://nlpgpu06:8000
    HOLO_BASE_URL=http://nlpgpu06:8000/v1 python scripts/run_trajectories.py --run-dir ...

    # Use MiniMax M3 instead of Holo
    python scripts/run_trajectories.py --run-dir ... --model-provider minimax_m3
    MINIMAX_API_KEY=... python scripts/run_trajectories.py --run-dir ... --model-provider minimax_m3

    # Use Qwen3.7-plus (via Aliyun DashScope) instead of Holo
    python scripts/run_trajectories.py --run-dir ... --model-provider qwen
    DASHSCOPE_API_KEY=... python scripts/run_trajectories.py --run-dir ... --model-provider qwen

    # Use e2b cloud sandboxes instead of Aliyun / Docker
    python scripts/run_trajectories.py --run-dir ... --env-backend e2b
    python scripts/run_trajectories.py --run-dir ... --env-backend e2b --e2b-api-key <key>

    # Use local Docker containers
    python scripts/run_trajectories.py --run-dir ... --env-backend docker

    # Action-history summarization is ON by default (needed for small-context
    # models like Holo-3.1). Disable it for large-context models like Qwen3.6:
    python scripts/run_trajectories.py --run-dir ... --no-summarize-history

    # Override tasks source directory
    python scripts/run_trajectories.py --run-dir ... --tasks-dir /path/to/cua_gym_tasks

Output layout (inside --run-dir):
    trajectory/
      {app}/
        {task_id}/
          images/
            0000.png
            0001.png
            ...
          actions.json
          task_summary.json   # {"trajectory_id": ..., "reward": ...}
      summary.json            # run-level rollup: reward buckets + failure counts/reasons
"""

import argparse
import base64
import json
import os
import re
import shlex
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

from utils.env import Env
from holo_model import HoloAgent
from minimax_model import MinimaxAgent
from qwen_model import QwenAgent

DEFAULT_TASKS_DIR = Path(
    "/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all/cua_gym_tasks"
)

MAX_STEPS = 15

# ---------------------------------------------------------------------------
# Active environment registry — used for cleanup on KeyboardInterrupt
# ---------------------------------------------------------------------------

_active_envs: set = set()
_active_envs_lock = threading.Lock()


def _register_env(env) -> None:
    with _active_envs_lock:
        _active_envs.add(env)


def _deregister_env(env) -> None:
    with _active_envs_lock:
        _active_envs.discard(env)


def _cleanup_envs_on_interrupt() -> None:
    """Kill all active sandbox/VM environments on interrupt.

    Mirrors run_eval.py's _run_interrupt_cleanup: temporarily ignores
    further SIGINT/SIGTERM so a second Ctrl-C doesn't abort the cleanup.
    """
    print("\nInterrupt received. Cleaning up active environments. Please wait...")
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with _active_envs_lock:
            envs = list(_active_envs)
        cleaned, failed = [], []
        for env in envs:
            try:
                if hasattr(env, "kill"):
                    env.kill()
                    cleaned.append(getattr(getattr(env, "config", None), "instance_id", repr(env)))
            except Exception as exc:
                failed.append(f"{repr(env)}: {exc}")
        if cleaned:
            print(f"  Cleaned up {len(cleaned)} environment(s): {', '.join(str(c) for c in cleaned)}")
        else:
            print("  No active environments needed cleanup.")
        for msg in failed:
            print(f"  [WARN] cleanup failed — {msg}")
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

# ---------------------------------------------------------------------------
# App normalization
# ---------------------------------------------------------------------------

def canonical_app(app_type: str | None) -> str:
    if not app_type:
        return "unknown"
    if "," in app_type:
        return "multi_apps"
    return app_type.strip()


def _tool_call_to_pyautogui(tool: dict, screen_w: int, screen_h: int) -> list[str]:
    """Convert a Holo tool_call dict into pyautogui code lines.

    Coordinates in Holo are normalized to [0, 1000]; we scale them to actual pixels.
    Returns a list of Python code strings, or ["DONE"], ["FAIL"], ["WAIT"].
    """
    tool_name = (tool.get("tool_name") or "").lower().strip()

    def scale(x, y):
        return int(x / 1000 * screen_w), int(y / 1000 * screen_h)

    def get_xy():
        x, y = tool.get("x"), tool.get("y")
        if x is None or y is None:
            return None
        return scale(x, y)

    code: list[str] = []

    if tool_name in ("click_desktop", "click"):
        xy = get_xy()
        if xy:
            button = (tool.get("button") or "left").lower()
            if button == "right":
                code.append(f"pyautogui.rightClick({xy[0]}, {xy[1]})")
            else:
                code.append(f"pyautogui.click({xy[0]}, {xy[1]})")

    elif tool_name == "right_click":
        xy = get_xy()
        if xy:
            code.append(f"pyautogui.rightClick({xy[0]}, {xy[1]})")

    elif tool_name in ("double_click_desktop", "double_click"):
        xy = get_xy()
        if xy:
            code.append(f"pyautogui.doubleClick({xy[0]}, {xy[1]})")

    elif tool_name == "move_to_desktop":
        xy = get_xy()
        if xy:
            code.append(f"pyautogui.moveTo({xy[0]}, {xy[1]})")

    elif tool_name in ("write_desktop", "write", "type"):
        content = tool.get("content") or tool.get("text") or ""
        press_enter = tool.get("press_enter", False)
        code.append(
            f"pyperclip.copy({json.dumps(content)}); "
            f"pyautogui.hotkey('ctrl', 'v'); time.sleep(0.1)"
        )
        if press_enter:
            code.append("pyautogui.press('enter')")

    elif tool_name == "write_at_desktop":
        xy = get_xy()
        content = tool.get("content") or tool.get("text") or ""
        if xy:
            code.append(f"pyautogui.click({xy[0]}, {xy[1]}); time.sleep(0.2)")
        code.append(
            f"pyperclip.copy({json.dumps(content)}); "
            f"pyautogui.hotkey('ctrl', 'v'); time.sleep(0.1)"
        )

    elif tool_name in ("hotkey_desktop", "key"):
        keys = tool.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        if len(keys) > 1:
            keys_str = ", ".join(json.dumps(k) for k in keys)
            code.append(f"pyautogui.hotkey({keys_str})")
        elif keys:
            code.append(f"pyautogui.press({json.dumps(keys[0])})")

    elif tool_name == "hold_and_tap_key_desktop":
        hold = tool.get("hold_key") or ""
        tap = tool.get("tap_key") or ""
        if hold and tap:
            code.append(f"pyautogui.keyDown({json.dumps(hold)})")
            code.append(f"pyautogui.press({json.dumps(tap)})")
            code.append(f"pyautogui.keyUp({json.dumps(hold)})")

    elif tool_name == "key_down_desktop":
        key = tool.get("key") or ""
        if key:
            code.append(f"pyautogui.keyDown({json.dumps(key)})")

    elif tool_name == "key_up_desktop":
        key = tool.get("key") or ""
        if key:
            code.append(f"pyautogui.keyUp({json.dumps(key)})")

    elif tool_name in ("drag_and_drop", "drag", "left_click_drag"):
        sx = tool.get("start_x") or tool.get("startX")
        sy = tool.get("start_y") or tool.get("startY")
        ex = tool.get("end_x") or tool.get("endX") or tool.get("target_x")
        ey = tool.get("end_y") or tool.get("endY") or tool.get("target_y")
        if sx is not None and sy is not None and ex is not None and ey is not None:
            sx, sy = scale(sx, sy)
            ex, ey = scale(ex, ey)
            code.append(f"pyautogui.moveTo({sx}, {sy})")
            code.append(f"pyautogui.dragTo({ex}, {ey}, duration=0.5)")

    elif tool_name == "mouse_down_desktop":
        xy = get_xy()
        button = (tool.get("button") or "left").lower()
        if xy:
            code.append(f"pyautogui.mouseDown({xy[0]}, {xy[1]}, button={json.dumps(button)})")

    elif tool_name == "mouse_up_desktop":
        xy = get_xy()
        button = (tool.get("button") or "left").lower()
        if xy:
            code.append(f"pyautogui.mouseUp({xy[0]}, {xy[1]}, button={json.dumps(button)})")

    elif tool_name in ("scroll_desktop", "scroll"):
        xy = get_xy()
        direction = (tool.get("direction") or "down").lower()
        amount = int(tool.get("amount") or 3)
        pixels = amount if direction in ("up", "right") else -amount
        if xy:
            code.append(f"pyautogui.moveTo({xy[0]}, {xy[1]})")
        code.append(f"pyautogui.scroll({pixels})")

    elif tool_name in ("wait_desktop", "wait"):
        secs = tool.get("seconds") or tool.get("time") or 2
        code.append(f"time.sleep({secs})")

    elif tool_name == "update_plan":
        return ["WAIT"]

    elif tool_name in ("answer", "done", "finish", "complete", "terminate", "success"):
        success = tool.get("success", True)
        return ["DONE" if success else "FAIL"]

    elif tool_name in ("fail", "failure"):
        return ["FAIL"]

    return code if code else ["WAIT"]


def _dismiss_libreoffice_format_dialog(env) -> None:
    """Press Enter to dismiss the LibreOffice 'Keep Current Format?' dialog.

    Called after any Ctrl+S action so that LibreOffice keeps the original
    .xlsx/.pptx/.docx format instead of switching to ODF.  Safe to call even
    when no dialog is present — the keystroke is ignored by a focused document.
    """
    time.sleep(0.6)
    env.run_python("import pyautogui; pyautogui.press('return')")
    time.sleep(0.3)


def _force_save_before_score(env) -> None:
    """Send Ctrl+S to the active window just before scoring.

    Catches the common failure mode where the agent completes the task but
    forgets to save.  The keystroke is a no-op when no file is modified.
    After the save we dismiss any LibreOffice 'Keep Current Format?' dialog
    so the file is written in the original Microsoft format (.xlsx/.pptx/.docx).
    """
    try:
        env.run_python("import pyautogui; pyautogui.hotkey('ctrl', 's')")
        # Wait for the LibreOffice format dialog to appear (if any).
        time.sleep(1.0)
        # Dismiss with Enter — accepts 'Keep Current Format!' (the default button).
        env.run_python("import pyautogui; pyautogui.press('return')")
        # Give LibreOffice time to finish writing the file.
        time.sleep(0.8)
    except Exception as exc:
        print(f"  [WARN] _force_save_before_score failed: {exc}")


def _action_is_ctrl_s(tool: dict) -> bool:
    """Return True when the tool_call is a Ctrl+S (or Ctrl+Shift+S) save."""
    name = (tool.get("tool_name") or "").lower().strip()
    if name not in ("hotkey_desktop", "key"):
        return False
    keys = [k.lower() for k in (tool.get("keys") or [])]
    return "ctrl" in keys and "s" in keys


def execute_action(env: Env, action_text: str, screen_w: int = 1920, screen_h: int = 1080) -> bool:
    """Parse Holo JSON output and execute on the VM via pyautogui.

    Returns True if the agent signalled task completion (DONE/FAIL).
    """
    try:
        step = json.loads(action_text or "{}")
    except json.JSONDecodeError:
        return False

    tool = step.get("tool_call") or {}
    code_lines = _tool_call_to_pyautogui(tool, screen_w, screen_h)

    if not code_lines or code_lines == ["WAIT"]:
        return False
    if code_lines[0] in ("DONE", "FAIL"):
        return True

    script = (
        "import pyautogui, pyperclip, time\n"
        "pyautogui.FAILSAFE = False\n"
        "pyautogui.PAUSE = 0.1\n"
        + "\n".join(code_lines)
    )
    env.run_python(script)

    # After Ctrl+S, dismiss any "Keep Current Format?" dialog so LibreOffice
    # saves in the original Microsoft format (.xlsx/.pptx/.docx).
    if _action_is_ctrl_s(tool):
        _dismiss_libreoffice_format_dialog(env)

    return False


# ---------------------------------------------------------------------------
# Task setup / scoring
# ---------------------------------------------------------------------------

def setup_env(env: Env, task_dir: Path):
    """Run the task's initial_setup script on the VM, then honor any sleep steps
    in task.json's config section (e.g. wait for the app to finish loading)."""
    setup_py = task_dir / "initial_setup.py"
    if setup_py.exists():
        env.run_python(setup_py.read_text())
    else:
        found = False
        for ext in ["sh", "xlsx", "docx", "pptx"]:
            setup_file = task_dir / f"initial_setup.{ext}"
            if setup_file.exists():
                env.upload(str(setup_file), f"/home/user/initial_setup.{ext}")
                if ext == "sh":
                    env.execute(f"bash /home/user/initial_setup.{ext}")
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No initial_setup file found in {task_dir}")

    # Process config steps from task.json: open/launch GUI apps and sleep.
    # "open" and "launch" were previously ignored, leaving apps like LibreOffice
    # unopened even after initial_setup.py created the required files.
    try:
        task_json = json.loads((task_dir / "task.json").read_text())
        for step in task_json.get("config", []):
            step_type = step.get("type")
            params = step.get("parameters", {})
            if step_type == "sleep":
                time.sleep(params.get("seconds", 0))
            elif step_type == "open":
                path = params.get("path", "")
                if path:
                    env.launch(f"xdg-open {shlex.quote(path)}")
            elif step_type == "launch":
                cmd = params.get("command", [])
                if isinstance(cmd, list):
                    cmd = shlex.join(str(c) for c in cmd)
                if cmd:
                    env.launch(cmd)
    except Exception:
        pass


def _run_postconfig(env: Env, postconfig: list) -> None:
    """Execute evaluator.postconfig steps (e.g. ctrl+s to save before scoring)."""
    sent_ctrl_s = False
    for step in postconfig:
        step_type = step.get("type", "")
        params = step.get("parameters", {})
        if step_type == "execute":
            cmd = params.get("command", [])
            if isinstance(cmd, list):
                cmd = shlex.join(str(c) for c in cmd)
            env.execute(cmd)
            if "ctrl" in cmd.lower() and "s" in cmd.lower():
                sent_ctrl_s = True
        elif step_type == "sleep":
            time.sleep(params.get("seconds", 1))
    # LibreOffice shows a "Keep Current Format?" dialog when saving as .xlsx.
    # Dismiss it with Enter so the file is actually written before reward.py reads it.
    if sent_ctrl_s:
        time.sleep(0.5)
        env.execute("python3 -c \"import pyautogui; pyautogui.press('return')\"")
        time.sleep(0.5)


def score_env(env: Env, task_dir: Path) -> float:
    """Upload reward_judge.py, run evaluator postconfig, then run reward.py."""
    repo_root = Path(__file__).parent.parent
    env.upload(str(repo_root / "utils" / "reward_judge.py"), "/tmp/reward_judge.py")

    # Ensure packages required by GIMP reward scripts are available.
    # xcftools is a system package (apt); gimpformats is a Python package.
    # These are no-ops if already installed (e.g. in Docker image).
    reward_code_preview = (task_dir / "reward.py").read_text()
    if "xcftools" in reward_code_preview or "xcf2png" in reward_code_preview:
        env.execute("apt-get install -y -q xcftools 2>/dev/null || true")
    if "gimpformats" in reward_code_preview:
        env.execute("pip3 install -q gimpformats 2>/dev/null || true")

    # Run postconfig steps defined in task.json evaluator (e.g. ctrl+s to save)
    task_json = json.loads((task_dir / "task.json").read_text())
    postconfig = task_json.get("evaluator", {}).get("postconfig", [])
    if postconfig:
        _run_postconfig(env, postconfig)

    reward_code = (task_dir / "reward.py").read_text()
    result = env.run_python(reward_code)
    output = result.get("output", "") or ""
    # Most reward.py files print "REWARD: X.X" or "reward:X.X" — parse that first.
    # Use re.search (not re.match) so "REWARD:" is found even when prefixed by
    # status symbols like "✗ File missing → REWARD: 0.0".
    for line in reversed(output.splitlines()):
        m = re.search(r"reward\s*:\s*([\d.]+)", line.strip(), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    # Fallback: try parsing the last non-empty line as a plain float.
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line:
            try:
                return float(line)
            except ValueError:
                continue
    err = (result.get("error") or "").strip()
    rc = result.get("returncode", 0)
    print(f"  [WARN] reward.py produced no parseable score (rc={rc})"
          + (f": {err[:200]}" if err else f"; output={output[:200]!r}"))
    return 0.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_one_task(
    task_dir: Path,
    traj_root: Path,
    agent,
    max_steps: int = MAX_STEPS,
    force: bool = False,
) -> tuple[float, bool]:
    """Run a single task and persist trajectory to disk.

    Returns (reward, skipped). skipped=True means task_summary.json already
    existed and force=False, so the task was not re-run.
    """
    task_json = json.loads((task_dir / "task.json").read_text())
    instruction = task_json.get("instruction", task_json.get("task_instruction", ""))
    task_id = task_dir.name
    app = canonical_app(task_json.get("app_type"))

    task_out = traj_root / app / task_id
    images_dir = task_out / "images"

    # Skip if already done (unless --force)
    summary_file = task_out / "task_summary.json"
    if summary_file.exists():
        if force:
            summary_file.unlink()
        else:
            existing = json.loads(summary_file.read_text())
            print(f"  [SKIP] {task_id}  (reward={existing.get('reward')})")
            return existing.get("reward", 0.0), True

    # Clear any partial run leftover before starting fresh
    if images_dir.exists():
        for _f in images_dir.iterdir():
            _f.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    traj_jsonl = task_out / "trajectory.jsonl"
    traj_jsonl.write_text("")

    env_config_path = f"/tmp/{task_id}_env.json"
    actions: list[dict] = []
    reward = 0.0

    env = Env.create(task_id=task_id)
    _register_env(env)
    try:
        env.save_config(env_config_path)
        setup_env(env, task_dir)

        # Live desktop preview (e2b only)
        if hasattr(env, "get_stream_url"):
            try:
                stream_url = env.get_stream_url()
                print(f"  Desktop: {stream_url}")
            except Exception as _exc:
                print(f"  Desktop URL unavailable: {_exc}")

        size = env.get_screen_size()
        screen_w = size.get("width", 1920)
        screen_h = size.get("height", 1080)

        agent_state = agent.initial_state()

        # E2B sandbox keepalive: extend timeout every N steps so long tasks
        # don't expire mid-run.  set_timeout resets the TTL from *now*.
        _E2B_KEEPALIVE_INTERVAL = 10
        _e2b_sandbox = getattr(env, "_sandbox", None)

        for step in range(max_steps):
            # Renew E2B sandbox lifetime periodically.
            if (
                _e2b_sandbox is not None
                and step > 0
                and step % _E2B_KEEPALIVE_INTERVAL == 0
            ):
                try:
                    _e2b_sandbox.set_timeout(3600)
                    print(f"  [INFO] E2B sandbox timeout renewed at step {step}")
                except Exception as _exc:
                    print(f"  [WARN] E2B timeout renewal failed at step {step}: {_exc}")

            screenshot_bytes = env.screenshot()

            # Save screenshot immediately — no waiting until the end
            if screenshot_bytes:
                (images_dir / f"{step:04d}.png").write_bytes(screenshot_bytes)

            screenshot_b64 = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else ""

            action_text, agent_state = agent.call(instruction, screenshot_b64, agent_state)

            step_record = {"step": step, "action": action_text}
            actions.append(step_record)

            # Append to trajectory.jsonl immediately so partial runs are recoverable
            with open(traj_jsonl, "a") as _f:
                _f.write(json.dumps(step_record, ensure_ascii=False) + "\n")

            done = execute_action(env, action_text, screen_w, screen_h)
            if done:
                break

            time.sleep(0.5)

        # Force-save before scoring in case the agent forgot to press Ctrl+S.
        _force_save_before_score(env)
        reward = score_env(env, task_dir)

    finally:
        _deregister_env(env)
        # Prefer killing the in-memory object directly so cleanup works even if
        # save_config() never wrote the config file (e.g. disk-full on /tmp).
        if hasattr(env, "kill"):
            try:
                env.kill()
            except Exception as _exc:
                print(f"  [WARN] env.kill() failed: {_exc}")
        else:
            Env.delete_instance(env_config_path)

    # Write final summary files (images already saved per-step above)
    (task_out / "actions.json").write_text(json.dumps(actions, ensure_ascii=False, indent=2))
    (task_out / "task_summary.json").write_text(
        json.dumps({"trajectory_id": task_id, "reward": reward}, indent=2)
    )
    return reward, False


# ---------------------------------------------------------------------------
# Run-level summary.json
# ---------------------------------------------------------------------------

# Ordered (first match wins) list of (category, substrings) used to bucket a
# failure's exception message. Patterns are lowercase; matched against the
# lowercased exception text.
_FAILURE_PATTERNS: list[tuple[str, list[str]]] = [
    ("task_dir_not_found", ["task dir not found"]),
    ("e2b_sandbox_lost", ["sandbox was not found", "sandbox timeout", "screenshot failed after"]),
    ("malformed_model_response", ["'list' object has no attribute 'get'", "object has no attribute 'get'"]),
    ("invalid_image_url", ["url does not appear to be valid", "invalidparameter"]),
    ("context_exceeded", ["context length", "maximum context", "input length"]),
    ("api_rate_limited", ["rate limit", "429"]),
    ("api_error", ["400 bad request", "401 ", "403 ", "500 ", "502 ", "503 "]),
    ("connection_error", ["connection", "timed out", "timeout"]),
]


def classify_failure_reason(exc_msg: str) -> str:
    """Bucket a failure's exception message into a coarse category for summary.json."""
    low = (exc_msg or "").lower()
    for category, needles in _FAILURE_PATTERNS:
        if any(needle in low for needle in needles):
            return category
    return "other"


def _find_task_summary(traj_root: Path, task_id: str) -> Path | None:
    for app_dir in traj_root.iterdir():
        if not app_dir.is_dir():
            continue
        candidate = app_dir / task_id / "task_summary.json"
        if candidate.exists():
            return candidate
    return None


def write_run_summary(
    traj_root: Path,
    task_ids: list[str],
    failure_reasons: dict[str, str],
) -> Path:
    """Roll the whole run up into trajectory/summary.json.

    Buckets rewards into ==1 / ==0 / (0,1), and buckets every task that has no
    task_summary.json (i.e. crashed or was never run) by failure category.
    """
    reward_eq_1, reward_eq_0, reward_mid, invalid_reward = [], [], [], []
    failure_categories: dict[str, list[dict]] = {}

    for task_id in task_ids:
        summary_path = _find_task_summary(traj_root, task_id)
        if summary_path is not None:
            try:
                reward = json.loads(summary_path.read_text()).get("reward")
            except Exception:
                reward = None
            if reward == 1 or reward == 1.0:
                reward_eq_1.append(task_id)
            elif reward == 0 or reward == 0.0:
                reward_eq_0.append(task_id)
            elif isinstance(reward, (int, float)) and 0 < reward < 1:
                reward_mid.append(task_id)
            else:
                invalid_reward.append(task_id)
        else:
            reason = failure_reasons.get(task_id, "unknown (no exception captured)")
            category = classify_failure_reason(reason)
            failure_categories.setdefault(category, []).append(
                {"task_id": task_id, "reason": reason[:500]}
            )

    succeeded = len(reward_eq_1) + len(reward_eq_0) + len(reward_mid) + len(invalid_reward)
    failed = sum(len(items) for items in failure_categories.values())

    summary = {
        "total_tasks": len(task_ids),
        "succeeded": succeeded,
        "failed": failed,
        "reward": {
            "eq_1": {"count": len(reward_eq_1), "task_ids": reward_eq_1},
            "eq_0": {"count": len(reward_eq_0), "task_ids": reward_eq_0},
            "between_0_and_1": {"count": len(reward_mid), "task_ids": reward_mid},
            **({"invalid_or_missing": {"count": len(invalid_reward), "task_ids": invalid_reward}}
               if invalid_reward else {}),
        },
        "failure_categories": {
            category: {"count": len(items), "items": items}
            for category, items in sorted(failure_categories.items(), key=lambda kv: -len(kv[1]))
        },
    }

    out_path = traj_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect trajectories with a pluggable model backend")
    parser.add_argument(
        "--run-dir", required=True, metavar="DIR",
        help="Run directory produced by select_tasks.py (e.g. data/run-20260620_185032). "
             "Reads task_selection/task_ids.json; writes trajectory/ here.",
    )
    parser.add_argument(
        "--tasks-dir", default=None, metavar="DIR",
        help=f"Source of CUA-Gym task bundles (default: {DEFAULT_TASKS_DIR})",
    )
    parser.add_argument(
        "--model-provider",
        choices=["holo", "minimax_m3", "qwen"],
        default=os.getenv("MODEL_PROVIDER", "holo"),
        help="Which model backend to drive the agent with: 'holo' (Holo-3.1-35B-A3B, "
             "default), 'minimax_m3' (MiniMax M3), or 'qwen' (Qwen3.7-plus via Aliyun "
             "DashScope). Overrides MODEL_PROVIDER env var.",
    )
    parser.add_argument(
        "--model-url", default=None, metavar="URL",
        help="Base URL of the model's chat/completions endpoint, e.g. http://nlpgpu06:8000 "
             "(Holo), https://api.minimaxi.com (MiniMax), or "
             "https://dashscope.aliyuncs.com/compatible-mode (Qwen). '/v1' is appended "
             "automatically if absent. Overrides HOLO_BASE_URL / MINIMAX_BASE_URL / "
             "QWEN_BASE_URL depending on --model-provider.",
    )
    parser.add_argument(
        "--model-name", default=None, metavar="NAME",
        help="Model name/id to request. Overrides HOLO_MODEL / MINIMAX_MODEL / QWEN_MODEL "
             "depending on --model-provider.",
    )
    parser.add_argument(
        "--model-api-key", default=None, metavar="KEY",
        help="Bearer token for the model API. Overrides HOLO_API_KEY / MINIMAX_API_KEY / "
             "DASHSCOPE_API_KEY depending on --model-provider.",
    )
    parser.add_argument(
        "--env-backend",
        choices=["aliyun", "docker", "e2b"],
        default=os.getenv("ENV_BACKEND", "aliyun"),
        help="VM/sandbox backend to use: 'aliyun' (default), 'docker', or 'e2b'. "
             "Overrides ENV_BACKEND env var.",
    )
    parser.add_argument(
        "--e2b-api-key",
        default=None,
        metavar="KEY",
        help="E2B API key (overrides E2B_API_KEY env var). Only used with --env-backend e2b.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help=f"Max agent steps per task (default: {MAX_STEPS}).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run N tasks in parallel using threads (default: 1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks that already have a task_summary.json (overwrites existing results).",
    )
    parser.add_argument(
        "--no-summarize-history",
        dest="summarize_history",
        action="store_false",
        help="Disable compressing old steps into a text summary block once SUMMARY_INTERVAL "
             "steps accumulate (default: on). Needed for small-context models like Holo-3.1; "
             "safe to disable for large-context models (e.g. Qwen3.6) that can hold full history.",
    )
    parser.set_defaults(summarize_history=True)
    args = parser.parse_args()

    if args.e2b_api_key:
        os.environ["E2B_API_KEY"] = args.e2b_api_key
    os.environ["ENV_BACKEND"] = args.env_backend
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    task_ids_path = run_dir / "task_selection" / "task_ids.json"
    if not task_ids_path.exists():
        sys.exit(f"task_ids.json not found: {task_ids_path}")

    model_url = None
    if args.model_url:
        base = args.model_url.rstrip("/")
        model_url = base if base.endswith("/v1") else f"{base}/v1"

    if args.model_provider == "holo":
        agent = HoloAgent(
            base_url=model_url,
            model=args.model_name,
            api_key=args.model_api_key,
            enable_history_summary=args.summarize_history,
        )
    elif args.model_provider == "qwen":
        agent = QwenAgent(
            base_url=model_url,
            model=args.model_name,
            api_key=args.model_api_key,
            enable_history_summary=args.summarize_history,
        )
    else:
        agent = MinimaxAgent(
            base_url=model_url,
            model=args.model_name,
            api_key=args.model_api_key,
            enable_history_summary=args.summarize_history,
        )

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else DEFAULT_TASKS_DIR
    if not tasks_dir.exists():
        sys.exit(f"Tasks directory not found: {tasks_dir}")

    task_ids: list[str] = json.loads(task_ids_path.read_text())
    traj_root = run_dir / "trajectory"
    traj_root.mkdir(parents=True, exist_ok=True)

    parallel = max(1, args.parallel)

    print(f"Run dir    : {run_dir}")
    print(f"Tasks dir  : {tasks_dir}")
    print(f"Model provider: {args.model_provider}")
    print(f"Model URL  : {agent.base_url}")
    print(f"Model name : {agent.model}")
    print(f"Tasks to run: {len(task_ids)}")
    print(f"Parallel   : {parallel}")
    print(f"Summarize history: {agent.enable_history_summary}")
    print()

    total = len(task_ids)
    counters = {"done": 0, "skipped": 0, "failed": 0, "index": 0}
    counter_lock = threading.Lock()
    failure_reasons: dict[str, str] = {}
    force = args.force

    def run_one(task_id: str) -> None:
        with counter_lock:
            counters["index"] += 1
            idx = counters["index"]

        task_dir = tasks_dir / task_id
        if not task_dir.exists():
            print(f"  [{idx}/{total}] [MISS] {task_id}  (task dir not found)")
            with counter_lock:
                counters["failed"] += 1
                failure_reasons[task_id] = "task dir not found"
            return

        print(f"  [{idx}/{total}] [RUN]  {task_id}")
        try:
            reward, skipped = run_one_task(task_dir, traj_root, agent, max_steps=max_steps, force=force)
            if skipped:
                with counter_lock:
                    counters["skipped"] += 1
            else:
                print(f"  [DONE] {task_id}  reward={reward:.4f}")
                with counter_lock:
                    counters["done"] += 1
        except Exception as exc:
            print(f"  [FAIL] {task_id}  {exc}")
            with counter_lock:
                counters["failed"] += 1
                failure_reasons[task_id] = str(exc)

    try:
        if parallel == 1:
            for task_id in task_ids:
                run_one(task_id)
        else:
            pool = ThreadPoolExecutor(max_workers=parallel)
            futures = {pool.submit(run_one, task_id): task_id for task_id in task_ids}
            try:
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"  >> EXCEPTION {task_id}: {exc}")
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
    except KeyboardInterrupt:
        _cleanup_envs_on_interrupt()
        sys.exit(130)

    summary_path = write_run_summary(traj_root, task_ids, failure_reasons)

    print()
    print(f"Finished — done={counters['done']}  skipped={counters['skipped']}  failed={counters['failed']}")
    print(f"Trajectories saved to: {traj_root}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
