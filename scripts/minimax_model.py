"""MiniMax M3 computer-use agent for run_trajectories.py.

This is a port of OSWorld's own M3 agent package (``mm_agents/m3/`` in
https://github.com/xlang-ai/OSWorld — see the local checkout at
/lcars/home/q/qianranm/research/GUI/OSWorld/mm_agents/m3/), which is the
actual OSWorld-official implementation for driving MiniMax M3. Everything
about *how the model is prompted and driven* is copied as-is:

  * System prompt : M3_SYSTEM_PROMPT_TEMPLATE verbatim (SYSTEM_CAPABILITY +
                     TOOLS blocks, [INFEASIBLE] handling, {date_str} /
                     {client_password} substitution).
  * Screenshots   : sent at native resolution (no resize), JPEG-encoded by
                     default (~6x smaller than PNG; M3_IMAGE_FORMAT=PNG to
                     turn that off).
  * Coordinates   : [0, 1000] normalized integers (coordinate_type="relative").
  * History       : "keep_min" recent screenshots (only_n_most_recent_images,
                     default 10), older ones replaced with a
                     "Tool result: Success" placeholder in chunks of
                     image_truncation_threshold (default 20) — see
                     m3/README.md's "Image truncation" section for the exact
                     sawtooth algorithm. The initial screenshot is always kept.
  * Parsing       : <tool_call>{"name": "computer", "arguments": {...}}</tool_call>
                     16-action computer-use tool enum, [INFEASIBLE] token,
                     BPE-artifact recovery (e.g. "_lick" -> "_click").
  * Retry         : predict() resamples up to M3_MAX_LLM_RETRIES (default 2)
                     extra times when the LLM call raises or the response
                     parses to zero actions despite non-empty text.

ONE deliberate deviation from the OSWorld original: transport. OSWorld's
m3/agent.py talks to M3 over the **Anthropic Messages API** (`anthropic`
SDK, `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY`). Our MiniMax integration is
confirmed (see this repo's git history) against MiniMax's own
**OpenAI-compatible** `/v1/chat/completions` endpoint
(https://api.minimax.io/v1) — there is no confirmed Anthropic-Messages-
compatible endpoint for our MiniMax deployment, and blindly switching wire
formats would silently break every existing run. So this module keeps the
OpenAI-compatible transport (borrowing OSWorld's own qwen/client.py retry
design, since that's OSWorld's reference implementation for this transport
family) while replicating every other M3 behavioural detail exactly. If you
do have an Anthropic-Messages-compatible M3 endpoint, swap _call_llm() for
the `anthropic` SDK call in m3/agent.py — everything else in this file
(prompt, image handling, message/history construction, parsing, retry
policy) carries over unchanged.

Env vars:
    MINIMAX_API_KEY          bearer token (required)
    MINIMAX_BASE_URL         API base (default: https://api.minimax.io/v1)
    MINIMAX_MODEL            model name (default: MiniMax-M3)
    MINIMAX_CLIENT_PASSWORD  sudo password templated into the system prompt
                             (default: "osworld-public-evaluation" — OSWorld's
                             own default; this is OSWorld's official VM image
                             credential, NOT verified for our e2b/Aliyun/Docker
                             sandboxes, but kept for prompt-fidelity since we
                             don't know the real one either)
    M3_MAX_LLM_RETRIES       extra resamples on parse-empty/exception (default: 2)
    M3_IMAGE_FORMAT          "JPEG" (default) or "PNG" on-wire screenshot encoding
    M3_IMAGE_QUALITY         JPEG quality 1-100 (default: 90)
    OSWORLD_MAX_RETRY_TIMES        transport retry attempts (default: 5)
    OSWORLD_OPENAI_TIMEOUT         request timeout override, seconds
    OSWORLD_HTTP_CONNECT_TIMEOUT   connect-timeout component of the default (default: 10)
    OSWORLD_HTTP_READ_TIMEOUT      read-timeout component of the default (default: 120)
"""

import base64
import json
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import openai
from PIL import Image
from requests.exceptions import SSLError

MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M3")

_IMAGE_FORMAT = os.environ.get("M3_IMAGE_FORMAT", "JPEG").upper()
_IMAGE_QUALITY = int(os.environ.get("M3_IMAGE_QUALITY", "90"))
_IMAGE_MIME = "image/jpeg" if _IMAGE_FORMAT == "JPEG" else f"image/{_IMAGE_FORMAT.lower()}"


def _encode_screenshot(image_bytes: bytes) -> Tuple[str, str]:
    """Return (base64_str, mime_type) for the on-wire image content.

    Default JPEG encoding shrinks wire bytes ~6x vs PNG with negligible pixel
    difference — matters once history accumulates many images. Set
    M3_IMAGE_FORMAT=PNG to skip the transcoding.
    """
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    save_kwargs: Dict[str, Any] = {"format": _IMAGE_FORMAT}
    if _IMAGE_FORMAT == "JPEG":
        save_kwargs["quality"] = _IMAGE_QUALITY
        save_kwargs["optimize"] = True
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), _IMAGE_MIME


# ---------------------------------------------------------------------------
# m3/prompts.py
# ---------------------------------------------------------------------------

M3_SYSTEM_PROMPT_TEMPLATE = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* TASK FEASIBILITY — this is a frequent failure point, read carefully. Some tasks are intentionally impossible. You may declare a task infeasible at any point: immediately after the first screenshot, or later after attempting actions and hitting a hard barrier. A task is infeasible when it cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory, fictional, or impossible requirements
  - The task requires a specific application, but you have verified that application does not provide the required feature or capability
  - Any other fundamental barrier that makes completion impossible
  When you conclude a task is infeasible, you MUST output the literal token "[INFEASIBLE]" (with the square brackets) in that same turn. This exact token is the ONLY signal the system recognizes for an impossible task.
  CRITICAL — when a task is infeasible, do NOT do any of the following instead:
  - Do NOT emit the `done` action. `done` means "I have successfully completed the task" and will be graded as WRONG for an impossible task.
  - Do NOT only explain in prose that it is "not possible" / "does not exist" / "cannot be done". Prose refusals are NOT detected — only the literal "[INFEASIBLE]" token counts.
  - Do NOT propose workarounds or alternatives, and do NOT ask the user what they prefer. Decide yourself, output "[INFEASIBLE]", and stop.
  Only declare a task infeasible when you are genuinely confident it is impossible. Do NOT give up on a task that is merely difficult, slow, or unfamiliar — try reasonable approaches first.
* The current date is {date_str}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is '{client_password}'.
* All `coordinate` values in `computer` tool calls are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen, regardless of the underlying screen resolution.
</SYSTEM_CAPABILITY>

<TOOLS>
You have access to the `computer` tool for interacting with the desktop GUI. Output one tool call per turn.

For each tool call, return a json object with name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "computer", "arguments": {{"action": "<action>", ...}}}}
</tool_call>

Supported `action` values include:
* `key`        — args: {{"text": "<key combo, e.g. ctrl+s>"}}
* `type`       — args: {{"text": "<text to type>"}}
* `mouse_move` — args: {{"coordinate": [x, y]}}
* `left_click` / `right_click` / `middle_click` / `double_click` / `triple_click`
                 — args: {{"coordinate": [x, y]}}, optional {{"text": "<modifier keys>"}}
* `left_click_drag` — args: {{"coordinate": [x, y]}}, optional {{"start_coordinate": [x, y]}}
* `scroll`     — args: {{"coordinate": [x, y], "scroll_direction": "up|down|left|right", "scroll_amount": <int>}}
* `wait`       — args: {{"duration": <seconds>}}
* `screenshot` — args: {{}}
* `hold_key`   — args: {{"text": "<keys>", "duration": <seconds>}}
* `left_mouse_down` / `left_mouse_up` — args: {{"coordinate": [x, y]}}
* `done`       — args: {{}}: declare the task has been completed successfully. Output this as the final tool call after you are confident the task is fully done.

Rules:
- Output exactly one short imperative `Action: <text>` line followed by exactly one <tool_call>...</tool_call> block.
- Coordinates are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen.
- When the task has been completed and there is nothing more to do, your FINAL turn must end with a <tool_call> using action `done` to signal completion.
- For impossible/infeasible tasks, your response MUST contain the literal `[INFEASIBLE]` token — not the `done` action, and not a prose-only explanation.
</TOOLS>"""


MACOS_KEYBOARD_MAPPING = """
IMPORTANT: You are operating on macOS. Key differences from Linux:
- Use 'command' key instead of 'ctrl' for shortcuts (e.g., command+c to copy, command+v to paste)
- Use 'option' key instead of 'alt'
- Close window: command+w; Quit app: command+q
- Spotlight search: command+space
- The menu bar is always at the top of the screen
- The Dock is at the bottom of the screen
- Use pyautogui.hotkey('command', 'c') not pyautogui.hotkey('ctrl', 'c')
"""


# ---------------------------------------------------------------------------
# m3/parser.py
# ---------------------------------------------------------------------------

_VALID_ACTIONS = frozenset({
    "click", "left click", "right click",
    "left_click", "right_click", "double_click", "middle_click",
    "left_press", "triple_click",
    "left_mouse_down", "left_mouse_up",
    "mouse_move", "left_click_drag",
    "hold_key", "key", "type",
    "scroll",
    "wait", "screenshot", "fail", "done", "call_user",
    "terminate", "cursor_position",
    "zoom", "zoom_in",
})


def tool_action_to_pyautogui(action: str, args: Dict, scaled_xy) -> List[str]:
    """Translate one computer-use tool action to pyautogui code.

    scaled_xy(x, y) maps the model's emitted (x, y) to actual screen pixels.
    Returns a list of code strings; special tokens ("WAIT", "DONE", "FAIL",
    "CALL_USER") are returned as their own one-element list.
    """
    code: List[str] = []

    action_conversion = {"left click": "click", "right click": "right_click"}
    action = action_conversion.get(action, action)
    if action == "click":
        action = "left_click"

    text = args.get("text")
    coordinate = args.get("coordinate")
    start_coordinate = args.get("start_coordinate")
    scroll_direction = args.get("scroll_direction")
    scroll_amount = args.get("scroll_amount")
    duration = args.get("duration")

    if coordinate is not None:
        coordinate = list(scaled_xy(coordinate[0], coordinate[1]))
    if start_coordinate is not None:
        start_coordinate = list(scaled_xy(start_coordinate[0], start_coordinate[1]))

    if action in ("left_mouse_down", "left_mouse_up"):
        if text:
            for k in text.split('+'):
                code.append(f"pyautogui.keyDown('{k.strip().lower()}')")
        if coordinate is not None:
            x, y = coordinate
            code.append(
                f"pyautogui.mouseDown({x}, {y})" if action == "left_mouse_down"
                else f"pyautogui.mouseUp({x}, {y})"
            )
        else:
            code.append(
                "pyautogui.mouseDown()" if action == "left_mouse_down"
                else "pyautogui.mouseUp()"
            )
        if text:
            for k in reversed(text.split('+')):
                code.append(f"pyautogui.keyUp('{k.strip().lower()}')")
    elif action == "hold_key":
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        for key in text.split('+'):
            code.append(f"pyautogui.keyDown('{key.strip().lower()}')")
    elif action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            raise ValueError(f"coordinate is required for {action}")
        if text is not None:
            raise ValueError(f"text is not accepted for {action}")
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f"{coordinate} must be a tuple of length 2")
        if not all(isinstance(i, int) for i in coordinate):
            raise ValueError(f"{coordinate} must be a tuple of ints")
        x, y = coordinate[0], coordinate[1]
        if action == "mouse_move":
            code.append(f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})")
        else:
            if start_coordinate:
                if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                    raise ValueError(f"{start_coordinate} must be a tuple of length 2")
                if not all(isinstance(i, int) for i in start_coordinate):
                    raise ValueError(f"{start_coordinate} must be a tuple of ints")
                sx, sy = start_coordinate
                code.append(f"pyautogui.moveTo({sx}, {sy}, duration={duration or 0.5})")
            code.append(f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})")
    elif action in ("key", "type"):
        if text is None:
            raise ValueError(f"text is required for {action}")
        if coordinate is not None:
            raise ValueError(f"coordinate is not accepted for {action}")
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        if action == "key":
            key_conversion = {
                "page_down": "pagedown",
                "page_up": "pageup",
                "super_l": "win",
                "super": "command",
                "escape": "esc",
            }
            keys = [key_conversion.get(k.strip().lower(), k.strip().lower()) for k in text.split('+')]
            for k in keys:
                code.append(f"pyautogui.keyDown('{k}')")
            for k in reversed(keys):
                code.append(f"pyautogui.keyUp('{k}')")
        else:
            for char in text:
                if char == '\n':
                    code.append("pyautogui.press('enter')")
                elif char == "'":
                    code.append('pyautogui.press("\'")')
                elif char == '\\':
                    code.append("pyautogui.press('\\\\')")
                elif char == '"':
                    code.append("pyautogui.press('\"')")
                else:
                    code.append(f"pyautogui.press('{char}')")
    elif action == "scroll":
        if text is not None:
            for k in text.split('+'):
                code.append(f"pyautogui.keyDown('{k.strip().lower()}')")
        if coordinate is None:
            if scroll_direction in ("up", "down"):
                code.append(f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount})")
            elif scroll_direction in ("left", "right"):
                code.append(f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount})")
        else:
            x, y = coordinate[0], coordinate[1]
            if scroll_direction in ("up", "down"):
                code.append(f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount}, {x}, {y})")
            elif scroll_direction in ("left", "right"):
                code.append(f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount}, {x}, {y})")
        if text is not None:
            for k in reversed(text.split('+')):
                code.append(f"pyautogui.keyUp('{k.strip().lower()}')")
    elif action in ("left_click", "right_click", "double_click", "middle_click", "left_press", "triple_click"):
        if text:
            for k in text.split('+'):
                code.append(f"pyautogui.keyDown('{k.strip().lower()}')")
        if coordinate is not None:
            x, y = coordinate
            mapping = {
                "left_click":   f"pyautogui.click({x}, {y})",
                "right_click":  f"pyautogui.rightClick({x}, {y})",
                "double_click": f"pyautogui.doubleClick({x}, {y})",
                "middle_click": f"pyautogui.middleClick({x}, {y})",
                "triple_click": f"pyautogui.tripleClick({x}, {y})",
            }
            if action == "left_press":
                code.append(f"pyautogui.mouseDown({x}, {y})")
                code.append("import time as _t; _t.sleep(1)")
                code.append(f"pyautogui.mouseUp({x}, {y})")
            else:
                code.append(mapping[action])
        else:
            mapping_no_xy = {
                "left_click":   "pyautogui.click()",
                "right_click":  "pyautogui.rightClick()",
                "double_click": "pyautogui.doubleClick()",
                "middle_click": "pyautogui.middleClick()",
                "triple_click": "pyautogui.tripleClick()",
            }
            if action == "left_press":
                code.append("pyautogui.mouseDown()")
                code.append("import time as _t; _t.sleep(1)")
                code.append("pyautogui.mouseUp()")
            else:
                code.append(mapping_no_xy[action])
        if text:
            for k in reversed(text.split('+')):
                code.append(f"pyautogui.keyUp('{k.strip().lower()}')")
    elif action == "wait":
        code.append("pyautogui.sleep(0.5)")
    elif action == "fail":
        code.append("FAIL")
    elif action == "done":
        code.append("DONE")
    elif action == "call_user":
        code.append("CALL_USER")
    elif action == "screenshot":
        code.append("pyautogui.sleep(0.1)")
    elif action == "terminate":
        status = str(args.get("status", "success")).lower()
        code.append("FAIL" if status in ("failure", "fail") else "DONE")
    elif action == "cursor_position":
        code.append("pyautogui.sleep(0.1)")
    elif action in ("zoom", "zoom_in"):
        print(f"  [INFO] action {action!r} -> no-op sleep (env doesn't expose native zoom)")
        code.append("pyautogui.sleep(0.1)")
    else:
        raise ValueError(f"Invalid action: {action}")

    return code


def parse_m3_response(
    response: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
    coordinate_type: str = "relative",
) -> Tuple[str, List[str]]:
    """Parse a single M3 response into (low_level_instruction, pyautogui_code).

    1. "[INFEASIBLE]" token -> ("[INFEASIBLE]", ["FAIL"]).
    2. Walk lines, picking up "Action: ..." and <tool_call>...</tool_call> blocks
       (bare JSON tool-call lines are also accepted).
    3. Each tool call is translated via tool_action_to_pyautogui with a coord
       scaler chosen by coordinate_type.
    4. Stop-sequence recovery: if the upstream ate </tool_call> (set as a stop
       string) the buffered body is still processed at end-of-input.
    """
    if not response or not response.strip():
        return "", []
    if "[INFEASIBLE]" in response:
        return "[INFEASIBLE]", ["FAIL"]

    def scaled_xy(x, y):
        if not (original_width and original_height):
            return int(x), int(y)
        if coordinate_type == "absolute":
            if processed_width and processed_height:
                sx = original_width / processed_width
                sy = original_height / processed_height
                return int(x * sx), int(y * sy)
            return int(x), int(y)
        return int(x * original_width / 1000), int(y * original_height / 1000)

    low_level_instruction = ""
    pyautogui_code: List[str] = []

    def emit(json_str: str) -> None:
        try:
            tool_call = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  [WARN] Failed to parse tool_call JSON: {e} - body: {json_str[:200]}")
            return
        if tool_call.get("name") != "computer":
            print(f"  [WARN] Expected name='computer', got {tool_call.get('name')!r} - skipping")
            return
        args = tool_call.get("arguments") or tool_call.get("input") or {}
        action = args.get("action")
        if not action:
            print(f"  [WARN] tool_call missing 'action': {tool_call}")
            return

        # BPE tokenizer artifact recovery (M3 quirks observed in production).
        if isinstance(action, str) and "_ " in action:
            fixed = action.replace("_ ", "_")
            print(f"  [INFO] [parser] BPE-fix action: {action!r} -> {fixed!r}")
            action = fixed
            args = dict(args, action=fixed)
        if isinstance(args, dict):
            renamed = {}
            changed = False
            for k, v in args.items():
                if isinstance(k, str) and "_ " in k:
                    nk = k.replace("_ ", "_")
                    print(f"  [INFO] [parser] BPE-fix arg key: {k!r} -> {nk!r}")
                    renamed[nk] = v
                    changed = True
                else:
                    renamed[k] = v
            if changed:
                args = renamed
        if isinstance(action, str) and "_lick" in action and action not in _VALID_ACTIONS:
            candidate = action.replace("_lick", "_click")
            if candidate in _VALID_ACTIONS:
                print(f"  [INFO] [parser] BPE click-recover: {action!r} -> {candidate!r}")
                action = candidate
                args = dict(args, action=candidate)
        if (isinstance(action, str)
                and action not in _VALID_ACTIONS
                and (action.endswith("_") or len(action) < 3)):
            print(f"  [WARN] [parser] truncated/short action {action!r} -> no-op")
            pyautogui_code.append("pyautogui.sleep(0.1)")
            return

        try:
            code = tool_action_to_pyautogui(action, args, scaled_xy)
        except ValueError as e:
            print(f"  [WARN] [parser] invalid action {action!r}: {e} -> no-op")
            pyautogui_code.append("pyautogui.sleep(0.1)")
            return
        if not code:
            return
        if len(code) == 1 and code[0] in ("FAIL", "DONE", "CALL_USER", "WAIT"):
            pyautogui_code.append(code[0])
        else:
            pyautogui_code.append("\n".join(code))

    inside = False
    buffer: List[str] = []
    for line in response.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("action:"):
            if not low_level_instruction:
                low_level_instruction = stripped.split(":", 1)[-1].strip()
            continue
        if stripped.startswith("<tool_call>"):
            inside = True
            continue
        if stripped.startswith("</tool_call>"):
            if buffer:
                emit("\n".join(buffer))
                buffer = []
            inside = False
            continue
        if inside:
            buffer.append(stripped)
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    emit(stripped)
            except json.JSONDecodeError:
                pass

    if inside and buffer:
        emit("\n".join(buffer))

    return low_level_instruction, pyautogui_code


# ---------------------------------------------------------------------------
# qwen/client.py-style OpenAI-compatible transport, reused here since M3's
# own transport (Anthropic Messages API) isn't available to us — see module
# docstring.
# ---------------------------------------------------------------------------

MAX_RETRY_TIMES = int(os.getenv("OSWORLD_MAX_RETRY_TIMES", "5"))


def _extract_message_field(message, field: str):
    value = getattr(message, field, None)
    if value is not None:
        return value
    if hasattr(message, "model_dump"):
        return message.model_dump().get(field)
    if isinstance(message, dict):
        return message.get(field)
    return None


def _extract_content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part.get("text", ""))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _merge_reasoning_content(content, reasoning_content) -> str:
    content_text = _extract_content_text(content)
    reasoning_text = _extract_content_text(reasoning_content).strip()
    if not reasoning_text:
        return content_text
    return f"<think>\n{reasoning_text}\n</think>\n\n{content_text.lstrip()}"


def _call_openai_compatible(messages: List[Dict], *, model: str, base_url: str, api_key: str,
                             max_tokens: int, temperature: Optional[float], top_p: Optional[float],
                             stop: Optional[List[str]]) -> str:
    default_timeout = str(
        float(os.environ.get("OSWORLD_HTTP_CONNECT_TIMEOUT", "10"))
        + float(os.environ.get("OSWORLD_HTTP_READ_TIMEOUT", "120"))
    )
    timeout_s = float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", default_timeout))
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    retryable_types = tuple(
        exc
        for exc in [
            SSLError,
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "BadRequestError", None),
            getattr(openai, "InternalServerError", None),
        ]
        if isinstance(exc, type)
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRY_TIMES + 1):
        try:
            create_kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if top_p is not None:
                create_kwargs["top_p"] = top_p
            if stop:
                create_kwargs["stop"] = list(stop)
            response = client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            content = _extract_message_field(message, "content")
            reasoning_content = _extract_message_field(message, "reasoning_content")
            return _merge_reasoning_content(content, reasoning_content)
        except retryable_types as exc:
            last_err = exc
            print(f"  [WARN] MinimaxAgent call_llm failed attempt {attempt}/{MAX_RETRY_TIMES}: {exc}")
            time.sleep(min(5.0 * attempt, 30.0))

    if last_err is not None:
        raise last_err
    return ""


# ---------------------------------------------------------------------------
# m3/agent.py — MinimaxAgent
# ---------------------------------------------------------------------------

class MinimaxAgent:
    """OSWorld's M3 agent, ported as-is except for transport (see module
    docstring). Stateful like the OSWorld original: call reset() once per
    task, then predict(instruction, obs) once per step, where
    obs == {"screenshot": <png bytes>}. Returns (response_text, pyautogui_code)
    — pyautogui_code is a list whose items are either real pyautogui source
    (already executable) or one of the special tokens
    "DONE" / "FAIL" / "CALL_USER".
    """

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = None,
        max_tokens: int = 8192,
        top_p: Optional[float] = None,
        temperature: Optional[float] = 0.6,
        coordinate_type: str = "relative",
        only_n_most_recent_images: int = 10,
        image_truncation_threshold: int = 20,
        system_date: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        client_password: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.platform = platform
        self.model = model or MINIMAX_MODEL
        self.base_url = (base_url or MINIMAX_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set (required for --model-provider minimax_m3)")

        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.coordinate_type = coordinate_type
        self.only_n_most_recent_images = only_n_most_recent_images
        self.image_truncation_threshold = image_truncation_threshold
        self.system_date = system_date or datetime.today().strftime("%A, %B %d, %Y")
        self.client_password = client_password or os.getenv(
            "MINIMAX_CLIENT_PASSWORD", "osworld-public-evaluation"
        )
        self.max_llm_retries = int(os.environ.get("M3_MAX_LLM_RETRIES", "2"))
        self.stop_sequences = stop_sequences if stop_sequences is not None else [
            "</tool_call>",
            "Perform the next action. Perform",
        ]

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []

    def reset(self, *args, **kwargs) -> None:
        self.thoughts.clear()
        self.actions.clear()
        self.observations.clear()
        self.responses.clear()
        self.screenshots.clear()

    def _build_system_prompt(self) -> str:
        prompt = M3_SYSTEM_PROMPT_TEMPLATE.format(
            date_str=self.system_date,
            client_password=self.client_password,
        )
        if self.platform == "macos":
            prompt += MACOS_KEYBOARD_MAPPING
        return prompt

    def _user_message_with_image(self, b64_img: str, text: Optional[str] = None) -> Dict:
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:{_IMAGE_MIME};base64,{b64_img}"}}
        ]
        if text:
            content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    @staticmethod
    def _user_message_tool_result_placeholder() -> Dict:
        return {"role": "user", "content": [{"type": "text", "text": "Tool result: Success"}]}

    def _build_messages(self, instruction: str) -> List[Dict]:
        """Assemble the full messages list for one predict() call.

        Image truncation: keep the most recent K=only_n_most_recent_images
        screenshots, drop older ones in chunks of T=image_truncation_threshold
        (the initial screenshot at index 0 is always kept). See m3/README.md's
        "Image truncation" section for the worked sawtooth example.
        """
        K = self.only_n_most_recent_images
        T = self.image_truncation_threshold
        k = len(self.responses)
        remove = max(0, k - K)
        remove -= remove % T

        messages: List[Dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            self._user_message_with_image(self.screenshots[0], instruction),
        ]
        for i in range(k):
            messages.append({"role": "assistant", "content": self.responses[i]})
            tool_result_idx = i + 1
            if tool_result_idx <= remove:
                messages.append(self._user_message_tool_result_placeholder())
            else:
                messages.append(self._user_message_with_image(self.screenshots[tool_result_idx]))
        return messages

    def _call_llm(self, messages: List[Dict]) -> str:
        return _call_openai_compatible(
            messages,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequences,
        )

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]
        image = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = image.size

        processed_b64, _ = _encode_screenshot(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_b64)))
        processed_width, processed_height = processed_img.size

        self.screenshots.append(processed_b64)
        messages = self._build_messages(instruction)

        response_text = ""
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        for attempt in range(self.max_llm_retries + 1):
            try:
                response_text = self._call_llm(messages)
            except Exception as exc:
                print(f"  [WARN] MinimaxAgent LLM call failed (attempt {attempt + 1}): {exc}")
                response_text = ""
                if attempt < self.max_llm_retries:
                    continue
                break

            low_level_instruction, pyautogui_code = parse_m3_response(
                response_text,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
                coordinate_type=self.coordinate_type,
            )
            if not pyautogui_code and response_text.strip():
                if attempt < self.max_llm_retries:
                    print(
                        f"  [WARN] MinimaxAgent parsed 0 actions from a "
                        f"{len(response_text)}-char response (attempt {attempt + 1}); resampling"
                    )
                    continue
            break

        self.responses.append(response_text)
        self.actions.append(low_level_instruction or "(no-op)")
        if low_level_instruction:
            self.thoughts.append(low_level_instruction)

        return response_text, pyautogui_code
