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


def make_summary_block_rules(start: int, end: int, responses: list) -> str:
    lines = [f"Steps {start + 1}–{end}:"]
    for i in range(start, end):
        try:
            step = json.loads(responses[i] or "{}")
            desc = describe_tool_fallback(step.get("tool_call") or {})
        except Exception:
            desc = "(action performed)"
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
            step_lines.append(f"Step {i + 1}: (no data)")

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
            except Exception:
                tool = {}
            desc = describe_tool_fallback(tool)
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
