"""Shared logic for computer-use-agent model backends.

scripts/holo_model.py and scripts/minimax_model.py each implement a thin
`_call_api` / `_call_summary_api` pair for their own chat/completions
endpoint and subclass BaseAgent here for everything else: the action-space
system prompt, the JSON step schema, message assembly (screenshot pruning +
history-summary blocks), and the context-exceeded retry loop. Keeping this
in one place means both backends behave identically except for the actual
HTTP call.
"""

import json
import re

AGENT_SYSTEM_PROMPT = (
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
    "Think step by step, then output the best action.\n\n"
    "IMPORTANT: Before calling `answer`, make sure all file changes are saved to "
    "disk (e.g. File → Export As / Ctrl+Shift+E in GIMP, Ctrl+S in most apps). "
    "Unsaved changes will not be evaluated.\n\n"
    "IMPORTANT — LibreOffice file format: When working with LibreOffice Calc, "
    "Impress, or Writer, always save files in their ORIGINAL Microsoft Office "
    "format (.xlsx, .pptx, .docx). Use Ctrl+S to save. If a dialog appears "
    "with the title 'Keep Current Format?' or asking whether to keep the "
    "Microsoft format, ALWAYS click the 'Keep Current Format!' button (do NOT "
    "choose 'Use ODF Format!'). Saving in ODF format will cause evaluation to "
    "fail because the grader expects the original file format."
)

# ---------------------------------------------------------------------------
# OSWorld-parity system prompt (prompt_style == "osworld")
# ---------------------------------------------------------------------------
# Verbatim OSWorld SYS_PROMPT_IN_SCREENSHOT_OUT_CODE (mm_agents/prompts.py in
# https://github.com/xlang-ai/OSWorld) — the prompt OSWorld's baseline uses for
# the screenshot-observation / pyautogui-action-space setting. The model
# returns raw pyautogui code in a fenced code block (or the bare commands
# WAIT / DONE / FAIL) instead of our structured tool_call JSON — see
# parse_pyautogui_response() below for how that's parsed back out.
#
# Two differences from the OSWorld original:
#   - Dropped "My computer's password is '{CLIENT_PASSWORD}' ...": none of our
#     sandbox backends (e2b, Aliyun, Docker) document a fixed sudo password for
#     their desktop user — e2b's docs (https://e2b.dev/docs/template/user-and-workdir)
#     name the default user `user` but say nothing about sudo credentials — and
#     a made-up password in the prompt would just be a lie the model could act on.
#   - Appended our two dataset-specific reminders (save-before-done, LibreOffice
#     "Keep Current Format!"), the same two paragraphs tacked onto
#     AGENT_SYSTEM_PROMPT above, since CUA-Gym's reward scripts read files off
#     disk in their original Microsoft Office format.
OSWORLD_SYSTEM_PROMPT = (
    "You are an agent which follow my instruction and perform desktop computer "
    "tasks as instructed. You have good knowledge of computer and good internet "
    "connection and assume your code will run on a computer for controlling the "
    "mouse and keyboard. For each step, you will get an observation of an image, "
    "which is the screenshot of the computer screen and you will predict the "
    "action of the computer based on the image.\n\n"
    "You are required to use `pyautogui` to perform the action grounded to the "
    "observation, but DONOT use the `pyautogui.locateCenterOnScreen` function to "
    "locate the element you want to operate with since we have no image of the "
    "element you want to operate with. DONOT USE `pyautogui.screenshot()` to "
    "make screenshot. Return one line or multiple lines of python code to "
    "perform the action each time, be time efficient. When predicting multiple "
    "lines of code, make some small sleep like `time.sleep(0.5);` interval so "
    "that the machine could take; Each time you need to predict a complete "
    "code, no variables or function can be shared from history You need to to "
    "specify the coordinates of by yourself based on your observation of "
    "current observation, but you should be careful to ensure that the "
    "coordinates are correct. You ONLY need to return the code inside a code "
    "block, like this:\n"
    "```python\n"
    "# your code here\n"
    "```\n"
    "Specially, it is also allowed to return the following special code:\n"
    "When you think you have to wait for some time, return ```WAIT```;\n"
    "When you think the task can not be done, return ```FAIL```, don't easily "
    "say ```FAIL```, try your best to do the task;\n"
    "When you think the task is done, return ```DONE```.\n\n"
    "First give the current screenshot and previous things we did a short "
    "reflection, then RETURN ME THE CODE OR SPECIAL CODE I ASKED FOR. NEVER "
    "EVER RETURN ME ANYTHING ELSE.\n\n"
    "IMPORTANT: Before returning ```DONE```, make sure all file changes are "
    "saved to disk (e.g. File → Export As / Ctrl+Shift+E in GIMP, Ctrl+S in "
    "most apps). Unsaved changes will not be evaluated.\n\n"
    "IMPORTANT — LibreOffice file format: When working with LibreOffice Calc, "
    "Impress, or Writer, always save files in their ORIGINAL Microsoft Office "
    "format (.xlsx, .pptx, .docx). Use Ctrl+S to save. If a dialog appears "
    "with the title 'Keep Current Format?' or asking whether to keep the "
    "Microsoft format, ALWAYS click the 'Keep Current Format!' button (do NOT "
    "choose 'Use ODF Format!'). Saving in ODF format will cause evaluation to "
    "fail because the grader expects the original file format."
)

PROMPT_STYLES = ("cua_gym", "osworld")

STEP_SCHEMA = {
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

DONE_TOOL_NAMES = {"answer", "done", "finish", "complete", "terminate", "success", "fail", "failure"}

JSON_SCHEMA_HINT = (
    "\n\nYou MUST respond with a single JSON object matching this schema, and "
    "nothing else (no prose, no markdown code fences):\n" + json.dumps(STEP_SCHEMA)
)


def resolve_system_prompt(prompt_style: str, needs_json_hint: bool) -> str:
    """Resolve the system prompt for a prompt_style ("cua_gym" or "osworld").

    needs_json_hint: append JSON_SCHEMA_HINT for cua_gym backends that lack
    guided decoding (Minimax, Qwen) and so must ask for JSON in the prompt
    itself. Ignored for "osworld", whose whole point is free-text/code output.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style: {prompt_style!r} (expected one of {PROMPT_STYLES})")
    if prompt_style == "osworld":
        return OSWORLD_SYSTEM_PROMPT
    return AGENT_SYSTEM_PROMPT + JSON_SCHEMA_HINT if needs_json_hint else AGENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# OSWorld-style code-action parsing (prompt_style == "osworld")
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:\w+\s+)?(.*?)```", re.DOTALL)
_TERMINAL_COMMANDS = {"WAIT", "DONE", "FAIL"}


def parse_pyautogui_response(text: str) -> list[str]:
    """Extract pyautogui code / terminal commands from an OSWorld-style raw response.

    Mirrors OSWorld's mm_agents.agent.parse_code_from_string(): pulls fenced code
    blocks out of the model's free-text response. A block that is exactly WAIT/
    DONE/FAIL is a terminal command; a block whose last line is one of those
    commands is split into (code, command). Returns a list mixing python source
    strings and the literal strings "WAIT"/"DONE"/"FAIL", in emission order.
    """
    codes: list[str] = []
    for raw in _CODE_BLOCK_RE.findall(text or ""):
        block = raw.strip()
        if not block:
            continue
        if block in _TERMINAL_COMMANDS:
            codes.append(block)
            continue
        lines = block.split("\n")
        if lines[-1].strip() in _TERMINAL_COMMANDS:
            body = "\n".join(lines[:-1]).strip()
            if body:
                codes.append(body)
            codes.append(lines[-1].strip())
        else:
            codes.append(block)
    return codes


# ---------------------------------------------------------------------------
# History management helpers
# ---------------------------------------------------------------------------

def describe_tool_fallback(tool: dict) -> str:
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


def _truncate_raw_response(raw: str, limit: int = 150) -> str:
    """Collapse a raw (possibly osworld-style, non-JSON) model response to one line."""
    text = " ".join((raw or "").split())
    if not text:
        return "(no data)"
    return (text[:limit] + "…") if len(text) > limit else text


def make_summary_block_rules(start: int, end: int, responses: list) -> str:
    lines = [f"Steps {start + 1}–{end}:"]
    for i in range(start, end):
        try:
            step = json.loads(responses[i] or "{}")
            desc = describe_tool_fallback(step.get("tool_call") or {})
        except Exception:
            # Not JSON — e.g. prompt_style=="osworld", where the raw response is
            # a free-text reflection + fenced pyautogui code. Fall back to a
            # truncated copy of the raw text rather than a placeholder.
            desc = _truncate_raw_response(responses[i])
        lines.append(f"  Step {i + 1}: {desc}")
    return "\n".join(lines)


def build_summary_prompt(start: int, end: int, responses: list) -> str:
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
            step_lines.append(f"Step {i + 1}:\n  Raw response: {_truncate_raw_response(responses[i])}")

    return (
        "You are summarizing a GUI agent's action history.\n"
        "For each step below, write 1–2 sentences that capture:\n"
        "  1. What was observed on screen and why the action was chosen.\n"
        "  2. What action was taken.\n"
        "Be concise (≤ 30 words per step). Do not include any preamble.\n"
        "Format your entire response as:\n"
        "Step N: <1–2 sentences>\n\n"
        + "\n\n".join(step_lines)
    )


def parse_summary_response(raw: str, start: int, end: int, responses: list) -> str:
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
                desc = describe_tool_fallback(tool)
            except Exception:
                desc = _truncate_raw_response(responses[i])
        lines.append(f"  Step {step_num}: {desc}")
    return "\n".join(lines)


def build_messages(
    instruction: str,
    current_b64: str,
    state: dict,
    effective_n: int,
    system_prompt: str,
) -> list:
    """Build messages with summary blocks in system prompt and image pruning."""
    summary_blocks = state["summary_blocks"]
    summary_coverage = state["summary_coverage"]
    screenshot_b64s = state["screenshot_b64s"]
    responses = state["responses"]

    sys_parts = [f"{system_prompt}\n\nTask: {instruction}"]
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


def is_context_exceeded(text: str) -> bool:
    low = text.lower()
    return "context length" in low or "input length" in low or "maximum context" in low


def is_retryable_api_error(text: str) -> bool:
    """Transient server-side generation failures the provider itself says to retry.

    E.g. DashScope aborting mid-generation while enforcing response_format=json_object:
    "Model output became abnormal ... Please retry the request". Distinct from
    is_context_exceeded (which needs image-pruning, not a plain retry) and from
    genuinely fatal 4xx errors (bad request/auth) that a retry can't fix.
    """
    low = text.lower()
    return (
        "model output became abnormal" in low
        or "generation was aborted" in low
        or "please retry the request" in low
    )


# ---------------------------------------------------------------------------
# Base agent — shared history-summarization + retry loop
# ---------------------------------------------------------------------------

class BaseAgent:
    """Common driver for a model backend.

    Subclasses implement `_call_api(messages) -> (text, context_exceeded)`
    and `_call_summary_api(prompt) -> text`; everything else (message
    assembly, screenshot pruning on context-exceeded, history summarization)
    is shared so behavior is identical across backends.
    """

    history_n = 3          # steps that keep their screenshot in conversation turns
    summary_interval = 10  # compress this many completed steps into a summary block
    max_retry = 3          # API call retries
    enable_history_summary = True
    system_prompt = AGENT_SYSTEM_PROMPT
    prompt_style = "cua_gym"   # "cua_gym" (our tool_call JSON) or "osworld" (raw pyautogui code)
    action_format = "json"     # "json" -> tool_call dict; "code" -> fenced pyautogui/WAIT/DONE/FAIL

    def _configure_prompt(self, prompt_style: str, needs_json_hint: bool) -> None:
        """Resolve system_prompt/action_format from prompt_style; call from subclass __init__."""
        prompt_style = prompt_style or "cua_gym"
        self.prompt_style = prompt_style
        self.action_format = "code" if prompt_style == "osworld" else "json"
        self.system_prompt = resolve_system_prompt(prompt_style, needs_json_hint)

    def initial_state(self) -> dict:
        return {
            "screenshot_b64s": [],
            "responses": [],
            "summary_blocks": [],
            "summary_coverage": 0,
        }

    def _call_api(self, messages: list) -> tuple:
        raise NotImplementedError

    def _call_summary_api(self, prompt: str) -> str:
        raise NotImplementedError

    def _make_summary_block(self, start: int, end: int, responses: list) -> str:
        try:
            prompt = build_summary_prompt(start, end, responses)
            raw = self._call_summary_api(prompt)
            return parse_summary_response(raw, start, end, responses)
        except Exception as exc:
            print(f"  [WARN] LLM summarization failed ({exc}); using rule-based fallback")
            return make_summary_block_rules(start, end, responses)

    def call(self, instruction: str, current_b64: str, state: dict) -> tuple:
        """Call the model with image pruning and optional history summarization.

        state keys: screenshot_b64s, responses, summary_blocks, summary_coverage.
        Returns (action_text, updated_state).
        """
        response_text = ""
        for effective_n in range(self.history_n, 0, -1):
            messages = build_messages(instruction, current_b64, state, effective_n, self.system_prompt)
            response_text, context_exceeded = self._call_api(messages)
            if not context_exceeded:
                break
            print(f"  [WARN] context exceeded with {effective_n} image(s); retrying with {effective_n - 1}")

        new_screenshot_b64s = state["screenshot_b64s"] + [current_b64]
        new_responses = state["responses"] + [response_text]
        new_summary_blocks = list(state["summary_blocks"])
        new_summary_coverage = state["summary_coverage"]

        unsummarized = len(new_responses) - new_summary_coverage
        if self.enable_history_summary and unsummarized >= self.summary_interval:
            start = new_summary_coverage
            end = start + self.summary_interval
            block = self._make_summary_block(start, end, new_responses)
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
