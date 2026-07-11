"""Holo-3.1-35B-A3B model backend for run_trajectories.py.

Talks to a local vLLM OpenAI-compatible `/chat/completions` endpoint.

prompt_style == "cua_gym" (default) implements H Company's official agent
harness (https://hub.hcompany.ai/agent-loop):
  - system prompt + Task + <output_format> schema block;
  - user turns are <observation>-wrapped screenshots, plus a
    <tool_output tool="..."> message after each executed action
    (appended via add_tool_output(), called by run_trajectories.py);
  - assistant turns replay only the parsed step JSON — reasoning is never
    spliced back into history;
  - at most the last 3 screenshots stay inline; older ones become
    "[screenshot evicted]" placeholders (trim_to_last_n_images);
  - JSON is enforced server-side via vLLM structured outputs
    (discriminated per-tool union in STEP_SCHEMA) and thinking is enabled
    through chat_template_kwargs;
  - NO history summarization — all text turns stay in context; the
    model's `note` field is its durable memory. Serve vLLM with the
    official context length (see slurm/start_vllm.sh: --max-model-len
    65537 --reasoning-parser qwen3).

prompt_style == "osworld" keeps the OSWorld screenshot+pyautogui-code
baseline driven by BaseAgent (including its history summarization).

Env vars:
    HOLO_BASE_URL   vLLM endpoint, e.g. http://nlpgpu06:8000/v1 (default: http://localhost:8000/v1)
    HOLO_MODEL      model name as registered with vLLM (default: holo-3.1)
    HOLO_API_KEY    bearer token (default: "token" — vLLM ignores it unless configured)
"""

import json
import os
import re
import time

import requests
from agent_common import (
    AGENT_SYSTEM_PROMPT,
    OSWORLD_SYSTEM_PROMPT,
    OUTPUT_FORMAT_BLOCK,
    PROMPT_STYLES,
    STEP_SCHEMA,
    BaseAgent,
    is_context_exceeded,
    is_retryable_api_error,
    trim_to_last_n_images,
)

HOLO_BASE_URL = os.getenv("HOLO_BASE_URL", "http://localhost:8000/v1")
HOLO_MODEL = os.getenv("HOLO_MODEL", "holo-3.1")


class HoloAgent(BaseAgent):
    history_n = 3          # official image budget: keep at most the last 3 screenshots
    max_retry = 3

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None,
                 enable_history_summary: bool = None, prompt_style: str = "cua_gym"):
        self.base_url = base_url or HOLO_BASE_URL
        self.model = model or HOLO_MODEL
        self.api_key = api_key or os.getenv("HOLO_API_KEY", "token")
        prompt_style = prompt_style or "cua_gym"
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"Unknown prompt_style: {prompt_style!r} (expected one of {PROMPT_STYLES})")
        self.prompt_style = prompt_style
        self.action_format = "code" if prompt_style == "osworld" else "json"
        if prompt_style == "osworld":
            self.system_prompt = OSWORLD_SYSTEM_PROMPT
            if enable_history_summary is not None:
                self.enable_history_summary = enable_history_summary
        else:
            self.system_prompt = AGENT_SYSTEM_PROMPT
            # The official harness has no history summarization: all text turns
            # stay in context, only old screenshots are evicted.
            self.enable_history_summary = False

    # ------------------------------------------------------------------
    # Official agent loop (prompt_style == "cua_gym")
    # ------------------------------------------------------------------

    def initial_state(self) -> dict:
        if self.prompt_style == "osworld":
            return super().initial_state()
        return {"messages": [], "last_tool_name": None}

    def call(self, instruction: str, current_b64: str, state: dict) -> tuple:
        if self.prompt_style == "osworld":
            return super().call(instruction, current_b64, state)

        messages = state["messages"]
        if not messages:
            messages.append({
                "role": "system",
                "content": f"{self.system_prompt}\n\nTask: {instruction}{OUTPUT_FORMAT_BLOCK}",
            })

        messages.append({"role": "user", "content": [
            {"type": "text", "text": "<observation>\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
            {"type": "text", "text": "\n</observation>"},
        ]})
        trim_to_last_n_images(messages, n=self.history_n)

        text, context_exceeded = self._call_api(messages)
        if context_exceeded:
            # The official harness has no summarization fallback; shed all but
            # the current screenshot and retry once.
            print("  [WARN] context exceeded; evicting all but the current screenshot")
            trim_to_last_n_images(messages, n=1)
            text, context_exceeded = self._call_api(messages)
            if context_exceeded:
                return "", state

        step = self._parse_step(text)
        if step is None:
            # Shouldn't happen under guided decoding — keep the raw text so the
            # trajectory shows what the model actually said.
            messages.append({"role": "assistant", "content": text})
            state["last_tool_name"] = None
            return text, state

        # Push only the parsed output back into history (official rule:
        # "do not splice the reasoning back in").
        clean = json.dumps(step, ensure_ascii=False)
        messages.append({"role": "assistant", "content": clean})
        tool = step.get("tool_call") or {}
        state["last_tool_name"] = tool.get("tool_name") if isinstance(tool, dict) else None
        return clean, state

    def add_tool_output(self, state: dict, result: str) -> None:
        """Append the official <tool_output> acknowledgement for the last action.

        Called by run_trajectories.py after executing the tool_call, so the
        message order matches the official loop: assistant -> tool_output ->
        next observation. No-op for the osworld prompt style.
        """
        if self.prompt_style == "osworld":
            return
        tool_name = state.get("last_tool_name")
        if not tool_name or not state["messages"]:
            return
        state["messages"].append({
            "role": "user",
            "content": f'<tool_output tool="{tool_name}">\n{result or ""}\n</tool_output>',
        })

    @staticmethod
    def _parse_step(text: str):
        """Parse the model content into a step dict, or None if not valid JSON.

        With --reasoning-parser qwen3 the content field already excludes the
        thinking block; the <think> strip is a defensive fallback for servers
        launched without it.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    # ------------------------------------------------------------------
    # HTTP calls
    # ------------------------------------------------------------------

    def _call_api(self, messages: list) -> tuple:
        for attempt in range(self.max_retry):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.8,
                    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
                }
                if self.action_format == "json":
                    payload["structured_outputs"] = {"json": STEP_SCHEMA}
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=120,
                )
                if not resp.ok:
                    err_text = resp.text[:500]
                    if is_context_exceeded(err_text):
                        return "", True
                    if is_retryable_api_error(err_text) and attempt < self.max_retry - 1:
                        time.sleep(5)
                        continue
                    raise RuntimeError(f"{resp.status_code} {resp.reason}: {err_text}")
                text = resp.json()["choices"][0]["message"]["content"] or ""
                return text, False
            except RuntimeError:
                raise
            except Exception as exc:
                if is_context_exceeded(str(exc)):
                    return "", True
                if attempt < self.max_retry - 1:
                    time.sleep(5)
        return "", False

    def _call_summary_api(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.0,
            },
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(f"summary LLM call failed: {resp.status_code} {resp.text[:200]}")
        raw = resp.json()["choices"][0]["message"]["content"] or ""
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
