#!/usr/bin/env python3
"""
Collect trajectories using Holo-3.1-35B-A3B on CUA-Gym task bundles.

Usage:
    # Run tasks selected by select_tasks.py
    python scripts/run_trajectories.py --run-dir data/run-20260620_185032

    # Override vLLM endpoint (env var or CLI flag)
    python scripts/run_trajectories.py --run-dir ... --model-url http://nlpgpu06:8000
    HOLO_BASE_URL=http://nlpgpu06:8000/v1 python scripts/run_trajectories.py --run-dir ...

    # Use e2b cloud sandboxes instead of Aliyun / Docker
    python scripts/run_trajectories.py --run-dir ... --env-backend e2b
    python scripts/run_trajectories.py --run-dir ... --env-backend e2b --e2b-api-key <key>

    # Use local Docker containers
    python scripts/run_trajectories.py --run-dir ... --env-backend docker

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
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

from utils.env import Env

DEFAULT_TASKS_DIR = Path(
    "/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all/cua_gym_tasks"
)

HOLO_BASE_URL = os.getenv("HOLO_BASE_URL", "http://localhost:8000/v1")
HOLO_MODEL = os.getenv("HOLO_MODEL", "holo-3.1")
MAX_STEPS = 15
HISTORY_N = 2         # steps that keep their screenshot in conversation turns
SUMMARY_INTERVAL = 10 # compress this many completed steps into a summary block
_MAX_RETRY = 3        # API call retries

_HOLO_SYSTEM_PROMPT = (
    "You are a computer use agent that controls a desktop GUI.\n"
    "At each step you receive a screenshot and output a structured action.\n"
    "All coordinates are integers in [0, 1000] (normalized to screen size).\n\n"
    "Available tools and their parameters:\n"
    "  click_desktop            — {tool_name, x, y, button}  button: \"left\"(default) or \"right\"\n"
    "  double_click_desktop     — {tool_name, x, y}\n"
    "  move_to_desktop          — {tool_name, x, y}\n"
    "  drag_and_drop            — {tool_name, start_x, start_y, end_x, end_y}\n"
    "  mouse_down_desktop       — {tool_name, x, y, button}\n"
    "  mouse_up_desktop         — {tool_name, x, y, button}\n"
    "  write_desktop            — {tool_name, content}\n"
    "  write_at_desktop         — {tool_name, x, y, content}\n"
    "  hotkey_desktop           — {tool_name, keys}  (e.g. keys: [\"ctrl\",\"s\"])\n"
    "  hold_and_tap_key_desktop — {tool_name, hold_key, tap_key}\n"
    "  key_down_desktop         — {tool_name, key}\n"
    "  key_up_desktop           — {tool_name, key}\n"
    "  scroll_desktop           — {tool_name, x, y, direction, amount}  direction: up/down/left/right\n"
    "  wait_desktop             — {tool_name, seconds}\n"
    "  answer                   — {tool_name, content}\n"
    "  update_plan              — {tool_name, content}\n\n"
    "Think step by step, then output the best action."
)

_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string"},
        "thought": {"type": "string"},
        "tool_call": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "element": {"type": "string"},
                "button": {"type": "string"},
                "start_x": {"type": "integer"},
                "start_y": {"type": "integer"},
                "end_x": {"type": "integer"},
                "end_y": {"type": "integer"},
                "content": {"type": "string"},
                "text": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "hold_key": {"type": "string"},
                "tap_key": {"type": "string"},
                "key": {"type": "string"},
                "direction": {"type": "string"},
                "amount": {"type": "integer"},
                "seconds": {"type": "number"},
                "success": {"type": "boolean"},
            },
            "required": ["tool_name"],
        },
    },
    "required": ["thought", "tool_call"],
}

_DONE_TOOL_NAMES = {"answer", "done", "finish", "complete", "terminate", "success", "fail", "failure"}


# ---------------------------------------------------------------------------
# App normalization
# ---------------------------------------------------------------------------

def canonical_app(app_type: str | None) -> str:
    if not app_type:
        return "unknown"
    if "," in app_type:
        return "multi_apps"
    return app_type.strip()


# ---------------------------------------------------------------------------
# Holo interaction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# History management helpers  (mirrors holo3_agent.py logic)
# ---------------------------------------------------------------------------

def _describe_tool_fallback(tool: dict) -> str:
    """Rule-based one-sentence description of a tool_call."""
    name = (tool.get("tool_name") or "").lower().strip()
    x, y = tool.get("x"), tool.get("y")
    coord = f" at ({x}, {y})" if x is not None and y is not None else ""
    elem = tool.get("element", "")
    elem_str = f" on '{elem}'" if elem else ""
    if name in ("click_desktop", "click"):
        return f"Clicked{elem_str}{coord}."
    if name in ("double_click_desktop", "double_click"):
        return f"Double-clicked{elem_str}{coord}."
    if name in ("write_desktop", "write_at_desktop", "write", "type"):
        content = tool.get("content") or tool.get("text") or ""
        short = (content[:60] + "…") if len(content) > 60 else content
        return f"Typed '{short}'."
    if name in ("hotkey_desktop", "key"):
        keys = tool.get("keys") or []
        return f"Pressed {'+'.join(str(k) for k in keys)}."
    if name in ("scroll_desktop", "scroll"):
        direction = tool.get("direction") or "down"
        amount = tool.get("amount") or 3
        return f"Scrolled {direction} by {amount}{coord}."
    if name in ("drag_and_drop", "drag", "left_click_drag"):
        sx, sy = tool.get("start_x"), tool.get("start_y")
        ex, ey = tool.get("end_x"), tool.get("end_y")
        return f"Dragged from ({sx}, {sy}) to ({ex}, {ey})."
    if name in ("wait_desktop", "wait"):
        secs = tool.get("seconds") or 2
        return f"Waited {secs} seconds."
    if name in ("answer", "done", "finish", "complete", "terminate", "success"):
        content = tool.get("content", "")
        return f"Completed task: {(content[:80] + '…') if len(content) > 80 else content}." if content else "Marked task complete."
    if name in ("fail", "failure"):
        return "Marked task failed."
    return f"Performed action '{name}'."


def _make_summary_block_rules(start: int, end: int, responses: list) -> str:
    lines = [f"Steps {start + 1}–{end}:"]
    for i in range(start, end):
        try:
            step = json.loads(responses[i] or "{}")
            desc = _describe_tool_fallback(step.get("tool_call") or {})
        except Exception:
            desc = "(action performed)"
        lines.append(f"  Step {i + 1}: {desc}")
    return "\n".join(lines)


def _make_summary_block_llm(start: int, end: int, responses: list) -> str:
    step_lines = []
    for i in range(start, end):
        try:
            step = json.loads(responses[i] or "{}")
            thought = (step.get("thought") or "").strip()
            tool_call = json.dumps(step.get("tool_call") or {})
            step_lines.append(
                f"Step {i + 1}:\n"
                f"  Observation & reasoning: {thought}\n"
                f"  Action taken: {tool_call}"
            )
        except Exception:
            step_lines.append(f"Step {i + 1}: (no data)")

    prompt = (
        "You are summarizing a GUI agent's action history.\n"
        "For each step below, write 1–2 sentences that capture:\n"
        "  1. What was observed on screen and why the action was chosen.\n"
        "  2. What action was taken.\n"
        "Be concise (≤ 30 words per step). Do not include any preamble.\n"
        "Format your entire response as:\n"
        "Step N: <1–2 sentences>\n\n"
        + "\n\n".join(step_lines)
    )
    resp = requests.post(
        f"{HOLO_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('HOLO_API_KEY', 'token')}"},
        json={
            "model": HOLO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        timeout=120,
    )
    if not resp.ok:
        raise RuntimeError(f"summary LLM call failed: {resp.status_code} {resp.text[:200]}")
    raw = resp.json()["choices"][0]["message"]["content"] or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    summaries: dict[int, str] = {}
    for line in raw.splitlines():
        m = re.match(r"Step\s+(\d+)\s*:\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            summaries[int(m.group(1))] = m.group(2).strip()

    lines = [f"Steps {start + 1}–{end}:"]
    for i in range(start, end):
        step_num = i + 1
        if step_num in summaries:
            desc = summaries[step_num]
        else:
            try:
                tool = json.loads(responses[i] or "{}").get("tool_call") or {}
            except Exception:
                tool = {}
            desc = _describe_tool_fallback(tool)
        lines.append(f"  Step {step_num}: {desc}")
    return "\n".join(lines)


def _make_summary_block(start: int, end: int, responses: list) -> str:
    try:
        return _make_summary_block_llm(start, end, responses)
    except Exception as exc:
        print(f"  [WARN] LLM summarization failed ({exc}); using rule-based fallback")
        return _make_summary_block_rules(start, end, responses)


def _build_holo_messages(
    instruction: str,
    current_b64: str,
    state: dict,
    effective_n: int,
) -> list:
    """Build messages with summary blocks in system prompt and image pruning."""
    summary_blocks = state["summary_blocks"]
    summary_coverage = state["summary_coverage"]
    screenshot_b64s = state["screenshot_b64s"]
    responses = state["responses"]

    sys_parts = [f"{_HOLO_SYSTEM_PROMPT}\n\nTask: {instruction}"]
    if summary_blocks:
        history_lines = ["[Completed action history]"]
        for _blk_start, _blk_end, blk_text in summary_blocks:
            history_lines.append(blk_text)
        sys_parts.append("\n".join(history_lines))
    messages: list = [{"role": "system", "content": "\n\n".join(sys_parts)}]

    prev_b64s = screenshot_b64s[summary_coverage:]
    prev_responses = responses[summary_coverage:]
    all_b64s = prev_b64s + [current_b64]
    n = len(all_b64s)
    image_start = max(0, n - effective_n)

    for i in range(n):
        is_current = (i == n - 1)
        if i >= image_start:
            user_content = [
                {"type": "text", "text": "<observation>\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{all_b64s[i]}"}},
                {"type": "text", "text": "\n</observation>"},
            ]
        else:
            user_content = [{"type": "text", "text": "<observation>\n[screenshot omitted]\n</observation>"}]
        messages.append({"role": "user", "content": user_content})
        if not is_current and i < len(prev_responses):
            messages.append({"role": "assistant", "content": prev_responses[i]})

    return messages


def _is_context_exceeded(text: str) -> bool:
    low = text.lower()
    return "context length" in low or "input length" in low or "maximum context" in low


def _call_holo_api(messages: list) -> tuple:
    """HTTP call to Holo with retry. Returns (response_text, context_exceeded)."""
    for attempt in range(_MAX_RETRY):
        try:
            resp = requests.post(
                f"{HOLO_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('HOLO_API_KEY', 'token')}"},
                json={
                    "model": HOLO_MODEL,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.8,
                    "structured_outputs": {"json": _STEP_SCHEMA},
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                timeout=120,
            )
            if not resp.ok:
                err_text = resp.text[:500]
                if _is_context_exceeded(err_text):
                    return "", True
                raise RuntimeError(f"{resp.status_code} {resp.reason}: {err_text}")
            text = resp.json()["choices"][0]["message"]["content"] or ""
            return text, False
        except RuntimeError:
            raise
        except Exception as exc:
            if _is_context_exceeded(str(exc)):
                return "", True
            if attempt < _MAX_RETRY - 1:
                time.sleep(5)
    return "", False


def call_holo(instruction: str, current_b64: str, state: dict) -> tuple:
    """Call Holo-3.1 with history summarization and image pruning.

    state keys: screenshot_b64s, responses, summary_blocks, summary_coverage.
    Returns (action_text, updated_state).
    """
    response_text = ""
    for effective_n in range(HISTORY_N, 0, -1):
        messages = _build_holo_messages(instruction, current_b64, state, effective_n)
        response_text, context_exceeded = _call_holo_api(messages)
        if not context_exceeded:
            break
        print(f"  [WARN] context exceeded with {effective_n} image(s); retrying with {effective_n - 1}")

    new_screenshot_b64s = state["screenshot_b64s"] + [current_b64]
    new_responses = state["responses"] + [response_text]
    new_summary_blocks = list(state["summary_blocks"])
    new_summary_coverage = state["summary_coverage"]

    unsummarized = len(new_responses) - new_summary_coverage
    if unsummarized >= SUMMARY_INTERVAL:
        start = new_summary_coverage
        end = start + SUMMARY_INTERVAL
        block = _make_summary_block(start, end, new_responses)
        new_summary_blocks.append((start, end, block))
        new_summary_coverage = end
        print(f"  [INFO] summarized steps {start + 1}–{end}")

    new_state = {
        "screenshot_b64s": new_screenshot_b64s,
        "responses": new_responses,
        "summary_blocks": new_summary_blocks,
        "summary_coverage": new_summary_coverage,
    }
    return response_text, new_state


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
    return False


# ---------------------------------------------------------------------------
# Task setup / scoring
# ---------------------------------------------------------------------------

def setup_env(env: Env, task_dir: Path):
    """Run the task's initial_setup script on the VM."""
    setup_py = task_dir / "initial_setup.py"
    if setup_py.exists():
        env.run_python(setup_py.read_text())
        return

    for ext in ["sh", "xlsx", "docx", "pptx"]:
        setup_file = task_dir / f"initial_setup.{ext}"
        if setup_file.exists():
            env.upload(str(setup_file), f"/home/user/initial_setup.{ext}")
            if ext == "sh":
                env.execute(f"bash /home/user/initial_setup.{ext}")
            return

    raise FileNotFoundError(f"No initial_setup file found in {task_dir}")


def score_env(env: Env, task_dir: Path) -> float:
    """Upload reward_judge.py and run reward.py on the VM, return the score."""
    repo_root = Path(__file__).parent.parent
    env.upload(str(repo_root / "utils" / "reward_judge.py"), "/tmp/reward_judge.py")
    reward_code = (task_dir / "reward.py").read_text()
    result = env.run_python(reward_code)
    try:
        return float(result.get("output", "0").strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_trajectory(
    task_id: str,
    app: str,
    actions: list[dict],
    screenshots_b64: list[str],
    reward: float,
    traj_root: Path,
) -> None:
    """Write images/, actions.json, task_summary.json under traj_root/app/task_id/."""
    task_out = traj_root / app / task_id

    # images/
    images_dir = task_out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(screenshots_b64):
        img_bytes = base64.b64decode(b64)
        (images_dir / f"{i:04d}.png").write_bytes(img_bytes)

    # actions.json
    (task_out / "actions.json").write_text(
        json.dumps(actions, ensure_ascii=False, indent=2)
    )

    # task_summary.json
    (task_out / "task_summary.json").write_text(
        json.dumps({"trajectory_id": task_id, "reward": reward}, indent=2)
    )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_one_task(task_dir: Path, traj_root: Path, max_steps: int = MAX_STEPS) -> float:
    """Run a single task and persist trajectory to disk. Returns the reward score."""
    task_json = json.loads((task_dir / "task.json").read_text())
    instruction = task_json.get("instruction", task_json.get("task_instruction", ""))
    task_id = task_dir.name
    app = canonical_app(task_json.get("app_type"))

    task_out = traj_root / app / task_id
    images_dir = task_out / "images"

    # Skip if already done
    if (task_out / "task_summary.json").exists():
        existing = json.loads((task_out / "task_summary.json").read_text())
        print(f"  [SKIP] {task_id}  (reward={existing.get('reward')})")
        return existing.get("reward", 0.0)

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

        holo_state = {
            "screenshot_b64s": [],
            "responses": [],
            "summary_blocks": [],
            "summary_coverage": 0,
        }
        for step in range(max_steps):
            screenshot_bytes = env.screenshot()

            # Save screenshot immediately — no waiting until the end
            if screenshot_bytes:
                (images_dir / f"{step:04d}.png").write_bytes(screenshot_bytes)

            screenshot_b64 = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else ""

            action_text, holo_state = call_holo(instruction, screenshot_b64, holo_state)

            step_record = {"step": step, "action": action_text}
            actions.append(step_record)

            # Append to trajectory.jsonl immediately so partial runs are recoverable
            with open(traj_jsonl, "a") as _f:
                _f.write(json.dumps(step_record, ensure_ascii=False) + "\n")

            done = execute_action(env, action_text, screen_w, screen_h)
            if done:
                break

            time.sleep(0.5)

        reward = score_env(env, task_dir)

    finally:
        Env.delete_instance(env_config_path)

    # Write final summary files (images already saved per-step above)
    (task_out / "actions.json").write_text(json.dumps(actions, ensure_ascii=False, indent=2))
    (task_out / "task_summary.json").write_text(
        json.dumps({"trajectory_id": task_id, "reward": reward}, indent=2)
    )
    return reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect trajectories with Holo-3.1-35B-A3B")
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
        "--model-url", default=None, metavar="URL",
        help="Base URL of the vLLM endpoint, e.g. http://nlpgpu06:8000. "
             "'/v1' is appended automatically if absent. "
             "Overrides the HOLO_BASE_URL environment variable.",
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

    if args.model_url:
        global HOLO_BASE_URL
        base = args.model_url.rstrip("/")
        HOLO_BASE_URL = base if base.endswith("/v1") else f"{base}/v1"

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else DEFAULT_TASKS_DIR
    if not tasks_dir.exists():
        sys.exit(f"Tasks directory not found: {tasks_dir}")

    task_ids: list[str] = json.loads(task_ids_path.read_text())
    traj_root = run_dir / "trajectory"
    traj_root.mkdir(parents=True, exist_ok=True)

    print(f"Run dir    : {run_dir}")
    print(f"Tasks dir  : {tasks_dir}")
    print(f"Model URL  : {HOLO_BASE_URL}")
    print(f"Tasks to run: {len(task_ids)}")
    print()

    total = len(task_ids)
    done = skipped = failed = 0
    for idx, task_id in enumerate(task_ids, 1):
        task_dir = tasks_dir / task_id
        if not task_dir.exists():
            print(f"  [{idx}/{total}] [MISS] {task_id}  (task dir not found)")
            failed += 1
            continue

        print(f"  [{idx}/{total}] [RUN]  {task_id}")
        try:
            reward = run_one_task(task_dir, traj_root, max_steps=max_steps)
            print(f"  [DONE] {task_id}  reward={reward:.4f}")
            done += 1
        except Exception as exc:
            print(f"  [FAIL] {task_id}  {exc}")
            failed += 1

    print()
    print(f"Finished — done={done}  skipped={skipped}  failed={failed}")
    print(f"Trajectories saved to: {traj_root}")


if __name__ == "__main__":
    main()
