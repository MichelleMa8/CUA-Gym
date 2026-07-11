"""Qwen (Aliyun DashScope) computer-use agent for run_trajectories.py.

This is a faithful port of OSWorld's own Qwen agent package
(``mm_agents/qwen/`` in https://github.com/xlang-ai/OSWorld — see the local
checkout at /lcars/home/q/qianranm/research/GUI/OSWorld/mm_agents/qwen/),
which is the actual OSWorld-official implementation for driving
Qwen3.7-plus. Every behavioural detail is copied as-is:

  * Tool schema  : the "internal" ``computer_use`` function (qwen/prompts.py
                   ``build_internal_tools_def`` / ``build_internal_system_prompt``),
                   XML-wrapped JSON tool calls
                   (``<tool_call><function=computer_use>...</function></tool_call>``).
  * Screenshots  : resized with Qwen-VL's own ``smart_resize`` (patch-size-32
                   aligned, from ``mm_agents/utils/qwen_vl_utils.py``) before
                   base64-PNG encoding — NOT sent at native resolution.
  * Coordinates  : normalized by dividing by 999 (qwen/images.py
                   ``adjust_coordinates``), not 1000.
  * History      : text action log kept for the last ``history_n`` (default
                   100 — effectively unbounded for our <=15-step tasks)
                   steps; images "folded" (permanently collapsed to a fixed
                   placeholder string) once the running count exceeds
                   ``image_max`` (20), in chunks of ``fold_size`` (10).
  * Transport    : OpenAI-compatible ``chat.completions`` (qwen/client.py),
                   which already matches what our DashScope endpoint speaks —
                   no adaptation needed here.

This module has no dependency on agent_common.py / our own JSON tool_call
convention — it is a standalone port, same as OSWorld's own mm_agents/qwen
package is standalone from mm_agents/agent.py.

Env vars:
    DASHSCOPE_API_KEY      bearer token (required) — https://bailian.console.aliyun.com/
    QWEN_BASE_URL          API base (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    QWEN_MODEL             model name (default: qwen3.7-plus)
    OSWORLD_MAX_RETRY_TIMES        transport retry attempts (default: 5)
    OSWORLD_OPENAI_TIMEOUT         request timeout override, seconds
    OSWORLD_HTTP_CONNECT_TIMEOUT   connect-timeout component of the default (default: 10)
    OSWORLD_HTTP_READ_TIMEOUT      read-timeout component of the default (default: 120)
"""

import ast
import json
import math
import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

import openai
from PIL import Image
from requests.exceptions import SSLError

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.7-plus")


# ---------------------------------------------------------------------------
# mm_agents/utils/qwen_vl_utils.py — smart_resize (Qwen-VL's own image
# preprocessing convention: patch-size-aligned dimensions, bounded pixel
# budget). Ported verbatim.
# ---------------------------------------------------------------------------

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    max_long_side: int = 8192,
) -> Tuple[int, int]:
    """Resize so that: (1) both sides are divisible by `factor`, (2) total
    pixels stay within [min_pixels, max_pixels], (3) the longest side stays
    within max_long_side, (4) aspect ratio is preserved as closely as possible."""
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


# ---------------------------------------------------------------------------
# qwen/images.py
# ---------------------------------------------------------------------------

def process_image(image_bytes: bytes) -> str:
    """Resize + re-encode screenshot and return base64 PNG."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    image = image.resize((resized_width, resized_height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    import base64
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_size_from_bytes(image_bytes: bytes) -> Tuple[int, int]:
    image = Image.open(BytesIO(image_bytes))
    return image.size


def image_size_from_base64(image_b64: str) -> Tuple[int, int]:
    import base64
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    return image.size


def adjust_coordinates(
    x: float,
    y: float,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[int, int]:
    if not (original_width and original_height):
        return int(x), int(y)
    if coordinate_type == "absolute":
        if processed_width and processed_height:
            x_scale = original_width / processed_width
            y_scale = original_height / processed_height
            return int(x * x_scale), int(y * y_scale)
        return int(x), int(y)

    x_scale = original_width / 999
    y_scale = original_height / 999
    return int(x * x_scale), int(y * y_scale)


# ---------------------------------------------------------------------------
# qwen/prompts.py (internal tool variant only — the one mm_agents.qwen.QwenAgent
# actually uses)
# ---------------------------------------------------------------------------

INTERNAL_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
""".strip()


def build_description_prompt(processed_width: int, processed_height: int, coordinate_type: str) -> str:
    resolution = (
        f"* The screen's resolution is {processed_width}x{processed_height}."
        if coordinate_type == "absolute"
        else "* The screen's resolution is 1000x1000."
    )
    return "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            resolution,
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    )


def build_internal_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": INTERNAL_ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
                            "key_down",
                            "key_up",
                            "left_mouse_down",
                            "left_mouse_up",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "hscroll",
                            "screenshot",
                            "wait",
                            "terminate",
                            "call_user",
                        ],
                    },
                    "keys": {
                        "type": "array",
                        "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Required only by `action=type` and `action=call_user`.",
                    },
                    "coordinate": {
                        "type": "array",
                        "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=left_click_drag`, optional for `action=left_mouse_down` and `action=left_mouse_up`.",
                    },
                    "pixels": {
                        "type": "number",
                        "description": "Scroll amount. Required only by `action=scroll` or `action=hscroll`.",
                    },
                    "time": {
                        "type": "number",
                        "description": "Seconds to wait. Required only by `action=wait`.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def build_internal_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def, ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "For normal UI interaction steps:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "For terminal steps, you may either:\n"
        "- output a final natural-language response with no tool call, or\n"
        "- use a terminal tool call such as call_user or terminate.\n\n"
        "Rules:\n"
        "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything after a tool call.\n"
        "- Use call_user when you need user information or confirmation.\n"
        "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
        "- If the task is infeasible, say so explicitly in the response."
    )


def build_instruction_prompt(instruction: str, previous_actions_str: str) -> str:
    return (
        "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n"
        f"{previous_actions_str}"
    )


# ---------------------------------------------------------------------------
# qwen/parser.py
# ---------------------------------------------------------------------------

def parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
    params: Dict = {}
    func_match = re.search(r"<function=([^>]+)>", xml_content)
    if not func_match or func_match.group(1) != "computer_use":
        return None

    for match in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml_content, re.DOTALL):
        name = match.group(1)
        value = match.group(2).strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value
    return params


def iter_tool_call_params(response: str):
    for tool_call_match in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        params = parse_xml_tool_call(tool_call_match.group(1))
        if params:
            yield params


def parse_keys(raw_keys, *, lowercase: bool = False) -> List[str]:
    if isinstance(raw_keys, str):
        try:
            raw_keys = json.loads(raw_keys)
        except Exception:
            try:
                raw_keys = ast.literal_eval(raw_keys)
            except Exception:
                pass

    def clean_key_token(key: object) -> str:
        token = str(key).strip()
        token = token.strip(" \t\r\n[](){}\"'")
        token = token.rstrip(" \t\r\n]")
        token = token.lstrip(" \t\r\n[")
        return token.strip()

    def flatten(keys_obj) -> List[str]:
        if keys_obj is None:
            return []
        if isinstance(keys_obj, list):
            values: List[str] = []
            for item in keys_obj:
                values.extend(flatten(item))
            return values
        values = []
        for part in re.split(r"\s*\+\s*", str(keys_obj).strip()):
            cleaned = clean_key_token(part)
            if cleaned:
                values.append(cleaned)
        return values

    keys = flatten(raw_keys)
    return [key.lower() for key in keys] if lowercase else keys


def parse_coordinate(raw_coord):
    if isinstance(raw_coord, str):
        try:
            raw_coord = json.loads(raw_coord)
        except Exception:
            return None
    if isinstance(raw_coord, list) and len(raw_coord) >= 2:
        return raw_coord[0], raw_coord[1]
    return None


def parse_number(raw_value, default=0.0) -> float:
    try:
        return float(raw_value)
    except Exception:
        return float(default)


def extract_action_line(response: str, *, preserve_base_split_bug: bool = False) -> str:
    for line in response.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            if preserve_base_split_bug:
                return stripped.split("Action:", 1)[-1].strip()
            return stripped.split(":", 1)[-1].strip()
    return ""


def looks_infeasible_response(text: str) -> bool:
    lowered = text.lower()
    if "[infeasible]" in lowered or "infeasible" in lowered:
        return True

    literal_patterns = [
        "not possible",
        "impossible",
        "not feasible",
        "cannot be completed",
        "can't be completed",
        "cannot be done",
        "cannot complete",
        "can't complete",
        "unable to complete",
        "cannot do this task",
        "can't do this task",
        "cannot complete this task as described",
        "cannot be completed as specified",
        "can't be completed as specified",
        "not available in your country",
        "not available",
        "unavailable",
        "not supported",
        "does not support",
        "doesn't support",
        "cannot natively",
        "does not have a built-in",
        "doesn't have a built-in",
        "does not include",
        "is not among the natively built-in",
        "will fall back to english",
        "requires the official",
        "no bluetooth found",
        "plug in a dongle",
        "folder is empty",
        "downloads folder is empty",
        "do not have the credentials",
        "don't have the credentials",
        "do not have the account credentials",
        "don't have the account credentials",
        "need the user's google account credentials",
        "requires a language pack extension",
        "requires email verification",
        "requires a sign-up",
        "requires sign-up",
        "requires google account credentials",
        "requires a google account",
        "sign in to the google account",
        "drm-protected",
        "drm protection",
        "cannot directly play",
        "no legitimate way",
        "requires a plugin",
        "requires an extension",
        "requires extension",
        "requires plugin",
        "requires a valid account",
        "requires purchase",
        "requires a purchased",
        "no valid account",
        "hidden audio",
        "could you clarify",
    ]
    if any(pattern in lowered for pattern in literal_patterns):
        return True

    regex_patterns = [
        r"\bthere is no [a-z0-9 _-]+\b",
        r"\bno [a-z0-9 _-]+ in [a-z0-9 _-]+ list\b",
        r"\brequires? (an? )?(extension|plugin|account|credentials|hardware|language pack)\b",
        r"\bneed(?:s)? (an? )?(extension|plugin|account|credentials|hardware|language pack)\b",
        r"\b(without|no) (extensions?|plugins?|terminal|ffmpeg|other apps?).{0,120}\\b(cannot|can't|not possible|not feasible)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in regex_patterns)


# ---------------------------------------------------------------------------
# qwen/actions.py (internal variant only — parse_internal_response)
# ---------------------------------------------------------------------------

def py_string(text: str) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)


def _termination_code(status: object) -> str:
    normalized = str(status or "success").strip().lower()
    return "FAIL" if normalized in {"fail", "failed", "failure", "error", "infeasible"} else "DONE"


def _coord_adjuster(
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
):
    def adjust(x: float, y: float) -> Tuple[int, int]:
        return adjust_coordinates(
            x,
            y,
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    return adjust


def _instruction_from_first_code(first_code: str) -> str:
    if first_code == "DONE":
        return "Task completed"
    if first_code == "WAIT":
        return "Waiting"
    if "." in first_code:
        return f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
    return "Performing action"


def parse_internal_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[str, List[str]]:
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if not response or not response.strip():
        return low_level_instruction, pyautogui_code

    adjust = _coord_adjuster(
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )
    infeasible_response = looks_infeasible_response(response)

    def process_tool_call_params(params: Dict) -> None:
        action = params.get("action")
        if not action:
            return

        coordinate = parse_coordinate(params.get("coordinate"))

        if action == "left_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.click({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.click()")
        elif action == "right_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.rightClick()")
        elif action == "middle_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.middleClick()")
        elif action == "double_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.doubleClick()")
        elif action == "triple_click":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.tripleClick({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.tripleClick()")
        elif action == "type":
            text = "" if params.get("text") is None else str(params.get("text", ""))
            normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
            if "\n" not in normalized_text:
                pyautogui_code.append(f"pyautogui.typewrite({py_string(normalized_text)})")
            else:
                chunks = normalized_text.split("\n")
                for idx, chunk in enumerate(chunks):
                    if chunk:
                        pyautogui_code.append(f"pyautogui.typewrite({py_string(chunk)})")
                    if idx < len(chunks) - 1:
                        pyautogui_code.append(f"pyautogui.press({py_string('enter')})")
        elif action == "key":
            keys = parse_keys(params.get("keys", []), lowercase=True)
            if not keys:
                return
            keys_str = ", ".join(py_string(key) for key in keys)
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            else:
                pyautogui_code.append(f"pyautogui.press({keys_str})")
        elif action == "key_down":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                pyautogui_code.append(f"pyautogui.keyDown({py_string(key)})")
        elif action == "key_up":
            for key in parse_keys(params.get("keys", []), lowercase=True):
                pyautogui_code.append(f"pyautogui.keyUp({py_string(key)})")
        elif action == "scroll":
            pixels = int(parse_number(params.get("pixels", 0), default=0))
            pyautogui_code.append(f"pyautogui.scroll({pixels})")
        elif action == "hscroll":
            pixels = int(parse_number(params.get("pixels", 0), default=0))
            pyautogui_code.append(f"pyautogui.hscroll({pixels})")
        elif action == "wait":
            pyautogui_code.append("WAIT")
        elif action == "terminate":
            pyautogui_code.append(_termination_code(params.get("status", "success")))
        elif action == "call_user":
            pyautogui_code.append("FAIL" if infeasible_response else "DONE")
        elif action == "screenshot":
            pyautogui_code.append("WAIT")
        elif action == "mouse_move":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            else:
                pyautogui_code.append("pyautogui.moveTo(0, 0)")
        elif action == "left_click_drag":
            if coordinate:
                x, y = adjust(*coordinate)
                duration = parse_number(params.get("duration", 0.5), default=0.5)
                pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
            else:
                pyautogui_code.append("pyautogui.dragTo(0, 0)")
        elif action == "left_mouse_down":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            pyautogui_code.append("pyautogui.mouseDown(button='left')")
        elif action == "left_mouse_up":
            if coordinate:
                x, y = adjust(*coordinate)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
            pyautogui_code.append("pyautogui.mouseUp(button='left')")

    low_level_instruction = extract_action_line(response)

    for params in iter_tool_call_params(response):
        process_tool_call_params(params)

    if not pyautogui_code:
        pyautogui_code.append("FAIL" if infeasible_response else "DONE")

    if not low_level_instruction and pyautogui_code:
        first_code = pyautogui_code[0]
        if first_code == "FAIL":
            low_level_instruction = "Need user input"
        else:
            low_level_instruction = _instruction_from_first_code(first_code)

    return low_level_instruction, pyautogui_code


# ---------------------------------------------------------------------------
# qwen/history.py
# ---------------------------------------------------------------------------

def update_folding_state(total_screenshots: int, folded_prefix_k: int, image_max: int, fold_size: int) -> int:
    while (total_screenshots - folded_prefix_k) > image_max:
        folded_prefix_k += fold_size
    if folded_prefix_k > total_screenshots:
        folded_prefix_k = total_screenshots
    return folded_prefix_k


def should_collapse_step(step_num_1based: int, folded_prefix_k: int) -> bool:
    return step_num_1based <= folded_prefix_k


def previous_actions_text(actions: List[str], start_step: int) -> str:
    previous_actions = [
        f"Step {i + 1}: {actions[i]}"
        for i in range(0, min(start_step - 1, len(actions)))
    ]
    return "\n".join(previous_actions) if previous_actions else "None"


def wrap_tool_response(parts: List[Dict]) -> List[Dict]:
    return (
        [{"type": "text", "text": "<tool_response>\n"}]
        + parts
        + [{"type": "text", "text": "\n</tool_response>"}]
    )


def build_messages(
    *,
    system_prompt: str,
    instruction_prompt: str,
    screenshots: List[str],
    responses: List[str],
    start_step: int,
    total_steps: int,
    folded_prefix_k: int,
    collapse_text: str,
    response_transform: Callable[[str], str] = lambda text: text,
) -> List[Dict]:
    messages: List[Dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = should_collapse_step(step_num, folded_prefix_k)

        if is_collapsed:
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = wrap_tool_response([{"type": "text", "text": collapse_text}])
            messages.append({"role": "user", "content": user_content})
        else:
            img_url = f"data:image/png;base64,{screenshots[step_num - 1]}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}]
                )
            messages.append({"role": "user", "content": user_content})

        if step_num <= total_steps - 1 and (step_num - 1) < len(responses):
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response_transform(responses[step_num - 1]),
                        }
                    ],
                }
            )

    return messages


def ensure_empty_think_prefix(response: str) -> str:
    text = response or ""
    if re.match(r"^\s*<think>.*?</think>\s*", text, re.DOTALL):
        return text
    return "<think>\n\n</think>\n\n" + text.lstrip("\n")


# ---------------------------------------------------------------------------
# qwen/client.py
# ---------------------------------------------------------------------------

MAX_RETRY_TIMES = int(os.getenv("OSWORLD_MAX_RETRY_TIMES", "5"))


def extract_content_text(content) -> str:
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


def extract_message_field(message, field: str):
    value = getattr(message, field, None)
    if value is not None:
        return value

    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        return dumped.get(field)

    if isinstance(message, dict):
        return message.get(field)

    return None


def merge_reasoning_content(content, reasoning_content) -> str:
    content_text = extract_content_text(content)
    reasoning_text = extract_content_text(reasoning_content).strip()
    if not reasoning_text:
        return content_text
    return f"<think>\n{reasoning_text}\n</think>\n\n{content_text.lstrip()}"


def call_openai_compatible(
    payload: Dict,
    model: str,
    *,
    base_url: Optional[str],
    api_key: Optional[str],
    default_max_tokens: int,
    default_temperature: float,
    default_top_p: float,
) -> str:
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    default_timeout = str(
        float(os.environ.get("OSWORLD_HTTP_CONNECT_TIMEOUT", "10"))
        + float(os.environ.get("OSWORLD_HTTP_READ_TIMEOUT", "120"))
    )
    timeout_s = float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", default_timeout))

    try:
        client = openai.OpenAI(base_url=resolved_base_url, api_key=resolved_api_key, timeout=timeout_s)
    except TypeError:
        client = openai.OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

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
            create_kwargs = dict(
                model=model,
                messages=payload["messages"],
                max_tokens=payload.get("max_tokens", default_max_tokens),
                temperature=payload.get("temperature", default_temperature),
                top_p=payload.get("top_p", default_top_p),
            )
            extra_body = payload.get("extra_body")
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            content = extract_message_field(message, "content")
            reasoning_content = extract_message_field(message, "reasoning_content")
            return merge_reasoning_content(content, reasoning_content)
        except retryable_types as exc:
            last_err = exc
            print(f"  [WARN] QwenAgent call_llm failed attempt {attempt}/{MAX_RETRY_TIMES}: {exc}")
            time.sleep(min(5.0 * attempt, 30.0))

    if last_err is not None:
        raise last_err
    return ""


# ---------------------------------------------------------------------------
# qwen/main.py — QwenAgent (internal-tool variant; the class
# mm_agents.qwen.QwenAgent actually exports and README recommends)
# ---------------------------------------------------------------------------

class QwenAgent:
    """OSWorld's Qwen computer-use agent, ported as-is.

    Stateful like the OSWorld original: call reset() once per task, then
    predict(instruction, obs) once per step, where obs == {"screenshot": <png bytes>}.
    Returns (response_text, pyautogui_code) — pyautogui_code is a list whose
    items are either real pyautogui source (already executable) or one of
    the special tokens "DONE" / "FAIL" / "WAIT".
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = None,
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        history_n: int = 100,
        coordinate_type: str = "relative",
        image_max: int = 20,
        fold_size: int = 10,
        collapse_text: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = False,
    ):
        self.platform = platform
        self.model = model or QWEN_MODEL
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.history_n = history_n
        self.coordinate_type = coordinate_type
        self.image_max = int(image_max)
        self.fold_size = int(fold_size)
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT
        self.base_url = (base_url or QWEN_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set (required for --model-provider qwen)")
        self.enable_thinking = enable_thinking

        if self.image_max < 1:
            raise ValueError("image_max must be >= 1")
        if self.fold_size < 1:
            raise ValueError("fold_size must be >= 1")

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []
        self.folded_prefix_k = 0

    def reset(self, *args, **kwargs) -> None:
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []
        self.folded_prefix_k = 0

    def _build_payload(self, messages: List[Dict]) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "")
        if "dashscope" in base_url.lower():
            extra_body = dict(payload.get("extra_body") or {})
            extra_body["enable_thinking"] = bool(self.enable_thinking)
            payload["extra_body"] = extra_body
        return payload

    def call_llm(self, payload: Dict) -> str:
        return call_openai_compatible(
            payload,
            self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            default_max_tokens=self.max_tokens,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
        )

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]

        original_width, original_height = image_size_from_bytes(screenshot_bytes)
        processed_b64 = process_image(screenshot_bytes)
        processed_width, processed_height = image_size_from_base64(processed_b64)

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)
        self.folded_prefix_k = update_folding_state(
            total_steps, self.folded_prefix_k, self.image_max, self.fold_size
        )

        start_step = max(1, total_steps - self.history_n)
        previous_actions_str = previous_actions_text(self.actions, start_step)

        tools_def = build_internal_tools_def(processed_width, processed_height, self.coordinate_type)
        system_prompt = build_internal_system_prompt(tools_def, self.collapse_text)
        instruction_prompt = build_instruction_prompt(instruction, previous_actions_str)

        self.observations.append({"screenshot": processed_b64})
        messages = build_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=self.screenshots,
            responses=self.responses,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
            response_transform=ensure_empty_think_prefix,
        )

        response = self.call_llm(self._build_payload(messages))
        self.responses.append(response or "")

        low_level_instruction, pyautogui_code = parse_internal_response(
            response or "",
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

        self.actions.append(low_level_instruction)
        return response or "", pyautogui_code
