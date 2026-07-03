#!/usr/bin/env python3
"""
Collect trajectories on CUA-Gym task bundles using a pluggable model backend
(Holo-3.1-35B-A3B by default, or MiniMax M3, or Qwen3.7-plus).

MiniMax M3 and Qwen run through faithful ports of OSWorld's own agent
implementations (see scripts/minimax_model.py and scripts/qwen_model.py —
ported from /lcars/home/q/qianranm/research/GUI/OSWorld/mm_agents/{m3,qwen}):
system prompt, screenshot handling, coordinate normalization, history
truncation/folding and response parsing are all identical to how OSWorld
itself drives these two models. There is no "our own" JSON tool_call mode
for these two providers — Holo is the only backend still using CUA-Gym's own
action-space convention (agent_common.py's AGENT_SYSTEM_PROMPT), optionally
switchable to OSWorld's classic free-pyautogui-code baseline prompt via
--system-prompt osworld.

Usage:
    # Run tasks selected by select_tasks.py (defaults to Holo-3.1)
    python scripts/run_trajectories.py --run-dir data/run-20260620_185032

    # Override vLLM endpoint (env var or CLI flag)
    python scripts/run_trajectories.py --run-dir ... --model-url http://nlpgpu06:8000
    HOLO_BASE_URL=http://nlpgpu06:8000/v1 python scripts/run_trajectories.py --run-dir ...

    # Use MiniMax M3 instead of Holo (OSWorld-replica agent — see minimax_model.py)
    python scripts/run_trajectories.py --run-dir ... --model-provider minimax_m3
    MINIMAX_API_KEY=... python scripts/run_trajectories.py --run-dir ... --model-provider minimax_m3

    # Use Qwen3.7-plus via Aliyun DashScope instead of Holo (OSWorld-replica agent — see qwen_model.py)
    python scripts/run_trajectories.py --run-dir ... --model-provider qwen
    DASHSCOPE_API_KEY=... python scripts/run_trajectories.py --run-dir ... --model-provider qwen

    # Use e2b cloud sandboxes instead of Aliyun / Docker
    python scripts/run_trajectories.py --run-dir ... --env-backend e2b
    python scripts/run_trajectories.py --run-dir ... --env-backend e2b --e2b-api-key <key>

    # Use local Docker containers
    python scripts/run_trajectories.py --run-dir ... --env-backend docker

    # Action-history summarization (Holo --system-prompt osworld only; the
    # default cua_gym style follows H Company's official harness, which never
    # summarizes — Qwen/MiniMax use OSWorld's own truncation/folding policy):
    python scripts/run_trajectories.py --run-dir ... --no-summarize-history

    # Holo only: use OSWorld's original screenshot+pyautogui-code baseline
    # prompt instead of our structured JSON tool_call action space.
    python scripts/run_trajectories.py --run-dir ... --system-prompt osworld

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

from agent_common import PROMPT_STYLES, parse_pyautogui_response
from holo_model import HoloAgent
from minimax_model import MinimaxAgent
from qwen_model import QwenAgent

from utils.env import Env

DEFAULT_TASKS_DIR = Path(
    "/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all/cua_gym_tasks"
)

MAX_STEPS = 15
TASK_TIMEOUT = 90 * 60  # 90 minutes — per-task wall-clock watchdog (see run_one() in main())

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
        # The model occasionally emits a coordinate as a list/string instead
        # of a number (e.g. "x": [100, 200]) — treat that as no coordinate
        # rather than letting `x / 1000` raise a TypeError and kill the task.
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
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
            start = scale(sx, sy)
            end = scale(ex, ey)
            if start and end:
                code.append(f"pyautogui.moveTo({start[0]}, {start[1]})")
                code.append(f"pyautogui.dragTo({end[0]}, {end[1]}, duration=0.5)")

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


def _png_dimensions(png_bytes: bytes) -> tuple[int, int] | None:
    """Read (width, height) from a PNG's IHDR chunk, or None if not a PNG.

    The official docs require scaling [0, 1000] coordinates against the same
    image bytes the model saw ("any resize, crop, or DPI mismatch will
    misclick"), so we measure the screenshot itself rather than trusting the
    env-reported screen size.
    """
    if not png_bytes or len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(png_bytes[16:20], "big")
    height = int.from_bytes(png_bytes[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


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


def execute_action(env: Env, action_text: str, screen_w: int = 1920, screen_h: int = 1080) -> tuple[bool, str]:
    """Parse Holo JSON output and execute on the VM via pyautogui.

    Returns (done, result): done is True if the agent signalled task
    completion (DONE/FAIL); result is the execution output (stdout/stderr of
    the pyautogui script, usually empty), fed back to the agent as the
    official <tool_output> message.
    """
    try:
        step = json.loads(action_text or "{}")
    except json.JSONDecodeError:
        return False, ""

    if not isinstance(step, dict):
        # The model occasionally emits a top-level JSON array/scalar instead
        # of an object — still valid JSON, so it doesn't hit JSONDecodeError
        # above, but has no "tool_call" key to read. Treat as a no-op.
        return False, ""

    tool = step.get("tool_call") or {}
    code_lines = _tool_call_to_pyautogui(tool, screen_w, screen_h)

    if not code_lines or code_lines == ["WAIT"]:
        return False, ""
    if code_lines[0] in ("DONE", "FAIL"):
        return True, ""

    script = (
        "import pyautogui, pyperclip, time\n"
        "pyautogui.FAILSAFE = False\n"
        "pyautogui.PAUSE = 0.1\n"
        + "\n".join(code_lines)
    )
    res = env.run_python(script)

    # After Ctrl+S, dismiss any "Keep Current Format?" dialog so LibreOffice
    # saves in the original Microsoft format (.xlsx/.pptx/.docx).
    if _action_is_ctrl_s(tool):
        _dismiss_libreoffice_format_dialog(env)

    result = ""
    if isinstance(res, dict):
        output = (res.get("output") or "").strip()
        error = (res.get("error") or "").strip()
        result = output
        if error:
            result = f"{output}\n[error] {error}".strip()
    return False, result


_CTRL_KEY_RE = re.compile(r"""['"]ctrl['"]""", re.IGNORECASE)
_S_KEY_RE = re.compile(r"""['"]s['"]""")


def _code_contains_ctrl_s(code: str) -> bool:
    """Best-effort detection of a Ctrl+S (or Ctrl+Shift+S) save inside raw pyautogui
    code. Mirrors _action_is_ctrl_s()'s "ctrl and s both present" check for the JSON
    tool_call path, but looser: different agents emit this differently — Qwen uses
    pyautogui.hotkey('ctrl', 's'), MiniMax/M3 uses discrete
    keyDown('ctrl')/keyDown('s')/keyUp(...)/keyUp(...) pairs with no hotkey() call at
    all — so this just checks both quoted key names appear anywhere in the snippet."""
    return bool(_CTRL_KEY_RE.search(code) and _S_KEY_RE.search(code))


def execute_action_code(env: Env, action_text: str) -> bool:
    """Execute an OSWorld-style raw pyautogui-code response (prompt_style=='osworld').

    Mirrors execute_action() above but for free-text/code output instead of
    tool_call JSON: pulls fenced pyautogui code (or WAIT/DONE/FAIL) out of the
    model's raw response via parse_pyautogui_response() and runs it directly,
    since it's already real pyautogui source rather than a dict to translate.

    Returns True if the agent signalled task completion (DONE/FAIL).
    """
    for code in parse_pyautogui_response(action_text):
        if code in ("DONE", "FAIL"):
            return True
        if code == "WAIT":
            time.sleep(2)
            continue

        script = (
            "import pyautogui, pyperclip, time\n"
            "pyautogui.FAILSAFE = False\n"
            "pyautogui.PAUSE = 0.1\n"
            + code
        )
        env.run_python(script)

        if _code_contains_ctrl_s(code):
            _dismiss_libreoffice_format_dialog(env)

    return False


def execute_pyautogui_code_list(env: Env, pyautogui_code: list) -> bool:
    """Execute the pre-parsed action list returned by MinimaxAgent/QwenAgent.predict()
    (see scripts/minimax_model.py / scripts/qwen_model.py — faithful ports of
    OSWorld's mm_agents/m3 and mm_agents/qwen agents). Each item is either real
    pyautogui source (no translation needed — the agent's own parser already
    produced it) or one of the special tokens DONE/FAIL/WAIT/CALL_USER that
    OSWorld's desktop_env.env.step() recognises.

    CALL_USER has no meaning in our unattended pipeline (there's no human to
    call), so it's treated as a failure signal like FAIL.

    Returns True if the agent signalled task completion (DONE/FAIL/CALL_USER).
    """
    for code in pyautogui_code:
        if code in ("DONE", "FAIL", "CALL_USER"):
            return True
        if code == "WAIT":
            time.sleep(2)
            continue

        script = (
            "import pyautogui, pyperclip, time\n"
            "pyautogui.FAILSAFE = False\n"
            "pyautogui.PAUSE = 0.1\n"
            + code
        )
        env.run_python(script)

        if _code_contains_ctrl_s(code):
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
    agent_factory,
    max_steps: int = MAX_STEPS,
    force: bool = False,
    env_holder: dict | None = None,
) -> tuple[float, bool]:
    """Run a single task and persist trajectory to disk.

    Returns (reward, skipped). skipped=True means task_summary.json already
    existed and force=False, so the task was not re-run.

    agent_factory: zero-arg callable returning a fresh agent instance for
    this task. Required because MinimaxAgent/QwenAgent are stateful objects
    (self.screenshots/self.responses mutated by predict()/reset() — see
    minimax_model.py/qwen_model.py) — sharing one instance across --parallel
    task threads causes concurrent reset()/predict() calls to race on that
    state (observed as spurious "list index out of range" crashes). A fresh
    instance per task sidesteps that regardless of provider; HoloAgent is
    cheap enough to construct that there's no reason to special-case it.

    env_holder: optional dict the caller can pass in to get a handle on the
    live `env` object as soon as it's created (env_holder["env"] = env) —
    used by main()'s per-task watchdog to kill a stuck task's sandbox from
    outside this function's thread, since a blocked Python thread can't be
    force-stopped any other way.
    """
    agent = agent_factory()
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
    if env_holder is not None:
        env_holder["env"] = env
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

        # MinimaxAgent/QwenAgent are OSWorld-replica agents: stateful objects
        # with their own predict()/reset() lifecycle (see minimax_model.py /
        # qwen_model.py), unlike HoloAgent's external call(instruction, b64,
        # state) -> (text, new_state) convention.
        osworld_replica = isinstance(agent, (MinimaxAgent, QwenAgent))
        if osworld_replica:
            agent.reset()
        else:
            agent_state = agent.initial_state()

        # E2B sandbox keepalive: periodically push the timeout back up to its
        # ceiling (E2BEnv.E2B_TIMEOUT, 3600s on Hobby). This can't prevent a
        # sandbox from hitting the plan's continuous-runtime cap in the first
        # place, but E2BEnv now creates sandboxes with on_timeout="pause" +
        # auto_resume=True, so hitting that cap pauses (state preserved) and
        # transparently resumes on the next SDK call instead of killing the
        # sandbox. A resume grants a fresh window at E2B's 300s default, not
        # our 3600s ceiling, so this loop's job is to keep re-extending that
        # window on a long task so it pauses/resumes as few times as possible.
        _E2B_KEEPALIVE_INTERVAL = 10
        _e2b_sandbox = getattr(env, "_sandbox", None)

        _MAX_CONSECUTIVE_EMPTY_ACTIONS = 3
        consecutive_empty_actions = 0

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
            if screenshot_bytes is None:
                # env.screenshot() already retried internally and logged the
                # underlying error. The sandbox is effectively gone at this
                # point — don't send an empty/invalid image_url to the model
                # API (that just trades a clear error for a confusing 400
                # from the far end). Fail the task now with a clear reason
                # so it gets classified correctly and retried on the next run.
                raise RuntimeError(
                    f"screenshot capture failed at step {step} (sandbox lost) — "
                    f"see 'E2B screenshot failed after N attempts' in logs for the underlying error"
                )

            (images_dir / f"{step:04d}.png").write_bytes(screenshot_bytes)

            if osworld_replica:
                action_text, pyautogui_code = agent.predict(instruction, {"screenshot": screenshot_bytes})
            else:
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                action_text, agent_state = agent.call(instruction, screenshot_b64, agent_state)

            if action_text:
                consecutive_empty_actions = 0
            else:
                consecutive_empty_actions += 1
                if consecutive_empty_actions >= _MAX_CONSECUTIVE_EMPTY_ACTIONS:
                    raise RuntimeError(
                        f"agent returned {consecutive_empty_actions} consecutive empty actions "
                        f"at step {step} (likely a persistent model API failure) — aborting task "
                        f"instead of idling until max_steps"
                    )

            step_record = {"step": step, "action": action_text}
            actions.append(step_record)

            # Append to trajectory.jsonl immediately so partial runs are recoverable
            with open(traj_jsonl, "a") as _f:
                _f.write(json.dumps(step_record, ensure_ascii=False) + "\n")

            if osworld_replica:
                done = execute_pyautogui_code_list(env, pyautogui_code)
            elif agent.action_format == "code":
                done = execute_action_code(env, action_text)
            else:
                # Scale [0, 1000] coordinates against the actual screenshot
                # dimensions (official requirement), not the env-reported
                # screen size, in case the two ever differ.
                dims = _png_dimensions(screenshot_bytes)
                img_w, img_h = dims if dims else (screen_w, screen_h)
                done, tool_result = execute_action(env, action_text, img_w, img_h)
                if not done and hasattr(agent, "add_tool_output"):
                    # Official loop layout: assistant -> <tool_output> -> next
                    # <observation>.
                    agent.add_tool_output(agent_state, tool_result)
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
    ("task_timeout", ["wall-clock timeout"]),
    ("e2b_sandbox_lost", [
        "sandbox was not found", "sandbox timeout", "screenshot failed after",
        "screenshot capture failed", "sandbox lost",
    ]),
    ("malformed_model_response", [
        "'list' object has no attribute 'get'", "object has no attribute 'get'",
        "unsupported operand type(s) for /",
    ]),
    ("persistent_api_failure", ["consecutive empty actions"]),
    ("json_generation_aborted", [
        "model output became abnormal", "generation was aborted", "please retry the request",
    ]),
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
        "--model-max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Qwen/MiniMax only. Overrides the per-call max_tokens (OSWorld's own "
             "defaults: 32768 for Qwen, 8192 for MiniMax M3 — sized for those "
             "providers' cloud endpoints). Required when max_tokens would exceed "
             "your server's --max-model-len, e.g. a self-hosted vLLM deployment "
             "with a 16384 context window needs something like 4096.",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=TASK_TIMEOUT,
        metavar="SECONDS",
        help=f"Wall-clock watchdog per task (default: {TASK_TIMEOUT}s / {TASK_TIMEOUT // 60}min). "
             "If a task's whole run_one_task() call (sandbox + agent loop + scoring) hasn't "
             "returned within this many seconds, the task is marked failed and its sandbox is "
             "killed to unstick anything blocked on it. Needed because E2B sandboxes now pause "
             "and auto-resume on their own timeout instead of dying, so a genuinely stuck task "
             "no longer fails on its own.",
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
        help="Holo with --system-prompt osworld only. Disable compressing old steps into "
             "a text summary block once SUMMARY_INTERVAL steps accumulate (default: on). "
             "The default cua_gym style follows H Company's official harness (no "
             "summarization ever — this flag is ignored). MiniMax M3 and Qwen are "
             "OSWorld-replica agents with their own truncation/folding policy (see "
             "minimax_model.py / qwen_model.py) — no effect on them either.",
    )
    parser.set_defaults(summarize_history=True)
    parser.add_argument(
        "--system-prompt",
        choices=list(PROMPT_STYLES),
        default=os.getenv("SYSTEM_PROMPT_STYLE", "cua_gym"),
        help="Holo only. Which system prompt / action-output format Holo uses: 'cua_gym' "
             "(default) is our structured JSON tool_call action space; 'osworld' swaps "
             "in OSWorld's classic screenshot+pyautogui-code baseline prompt (the model "
             "returns raw pyautogui code in a fenced block, or WAIT/DONE/FAIL). MiniMax M3 "
             "and Qwen always run their OSWorld-replica agent (see minimax_model.py / "
             "qwen_model.py) — this flag has no effect on them. Overrides SYSTEM_PROMPT_STYLE env var.",
    )
    args = parser.parse_args()

    if args.e2b_api_key:
        os.environ["E2B_API_KEY"] = args.e2b_api_key
    os.environ["ENV_BACKEND"] = args.env_backend
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS
    task_timeout = args.task_timeout

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

    def make_agent():
        """Construct a fresh agent instance.

        Called once per task (see run_one_task) rather than shared across
        --parallel task threads: MinimaxAgent/QwenAgent hold mutable per-task
        state (self.screenshots/self.responses, mutated by predict()/reset()),
        so sharing one instance across concurrent threads races on that state.
        HoloAgent has no such state, but building fresh here too keeps this
        function simple and provider-agnostic.
        """
        if args.model_provider == "holo":
            return HoloAgent(
                base_url=model_url,
                model=args.model_name,
                api_key=args.model_api_key,
                enable_history_summary=args.summarize_history,
                prompt_style=args.system_prompt,
            )
        elif args.model_provider == "qwen":
            # OSWorld-replica agent (scripts/qwen_model.py) — no prompt_style /
            # enable_history_summary knobs; it always runs OSWorld's own prompt,
            # image handling, and truncation/folding policy.
            qwen_kwargs = {}
            if args.model_max_tokens is not None:
                qwen_kwargs["max_tokens"] = args.model_max_tokens
            return QwenAgent(
                base_url=model_url,
                model=args.model_name,
                api_key=args.model_api_key,
                **qwen_kwargs,
            )
        else:
            # OSWorld-replica agent (scripts/minimax_model.py) — same note as above.
            minimax_kwargs = {}
            if args.model_max_tokens is not None:
                minimax_kwargs["max_tokens"] = args.model_max_tokens
            return MinimaxAgent(
                base_url=model_url,
                model=args.model_name,
                api_key=args.model_api_key,
                **minimax_kwargs,
            )

    # One throwaway instance just to print config info below — never used to
    # run a task (see make_agent's docstring for why each task gets its own).
    agent = make_agent()

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
    print(f"Task timeout: {task_timeout}s ({task_timeout // 60}min)")
    if hasattr(agent, "enable_history_summary"):
        print(f"Summarize history: {agent.enable_history_summary}")
    if hasattr(agent, "prompt_style"):
        print(f"Prompt style: {agent.prompt_style} (action format: {agent.action_format})")
    else:
        print("Prompt style: osworld (native — OSWorld-replica agent, see minimax_model.py/qwen_model.py)")
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

        # Run the task in its own (inner) thread and join with a wall-clock
        # timeout. E2B sandboxes now pause + auto-resume on their own timeout
        # instead of dying, so a genuinely stuck task (sandbox wedged, or a
        # hang in our own code) no longer fails on its own — this watchdog is
        # the replacement backstop. A Python thread can't be force-stopped,
        # so on timeout we kill the task's sandbox (via env_holder, populated
        # by run_one_task as soon as it creates the env) to unstick anything
        # blocked on an E2B call, and abandon the inner thread — it exits on
        # its own once the blocked call errors out.
        env_holder: dict = {}
        outcome: dict = {}

        def _worker(_env_holder=env_holder, _outcome=outcome):
            try:
                reward, skipped = run_one_task(
                    task_dir, traj_root, make_agent, max_steps=max_steps, force=force,
                    env_holder=_env_holder,
                )
                _outcome["reward"] = reward
                _outcome["skipped"] = skipped
            except Exception as exc:
                _outcome["error"] = exc

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()
        worker_thread.join(timeout=task_timeout)

        if worker_thread.is_alive():
            print(f"  [TIMEOUT] {task_id}  exceeded {task_timeout}s — killing sandbox and moving on")
            env = env_holder.get("env")
            if env is not None and hasattr(env, "kill"):
                try:
                    env.kill()
                except Exception as kill_exc:
                    print(f"  [WARN] failed to kill stuck sandbox for {task_id}: {kill_exc}")
            with counter_lock:
                counters["failed"] += 1
                failure_reasons[task_id] = (
                    f"task exceeded {task_timeout}s wall-clock timeout — sandbox killed to unblock"
                )
            return

        if "error" in outcome:
            exc = outcome["error"]
            print(f"  [FAIL] {task_id}  {exc}")
            with counter_lock:
                counters["failed"] += 1
                failure_reasons[task_id] = str(exc)
        elif outcome.get("skipped"):
            with counter_lock:
                counters["skipped"] += 1
        else:
            print(f"  [DONE] {task_id}  reward={outcome['reward']:.4f}")
            with counter_lock:
                counters["done"] += 1

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
