"""Holo-3.1-35B-A3B model backend for run_trajectories.py.

Talks to a local vLLM OpenAI-compatible `/chat/completions` endpoint.
Structured JSON output is enforced via vLLM's `structured_outputs`
guided-decoding parameter, and extended thinking is toggled through
`chat_template_kwargs`.

Env vars:
    HOLO_BASE_URL   vLLM endpoint, e.g. http://nlpgpu06:8000/v1 (default: http://localhost:8000/v1)
    HOLO_MODEL      model name as registered with vLLM (default: holo-3.1)
    HOLO_API_KEY    bearer token (default: "token" — vLLM ignores it unless configured)
"""

import os
import re
import time

import requests
from agent_common import STEP_SCHEMA, BaseAgent, is_context_exceeded, is_retryable_api_error

HOLO_BASE_URL = os.getenv("HOLO_BASE_URL", "http://localhost:8000/v1")
HOLO_MODEL = os.getenv("HOLO_MODEL", "holo-3.1")


class HoloAgent(BaseAgent):
    history_n = 3          # small context model — keep few screenshots inline
    summary_interval = 10
    max_retry = 3
    enable_history_summary = True

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None,
                 enable_history_summary: bool = None, prompt_style: str = "cua_gym"):
        self.base_url = base_url or HOLO_BASE_URL
        self.model = model or HOLO_MODEL
        self.api_key = api_key or os.getenv("HOLO_API_KEY", "token")
        if enable_history_summary is not None:
            self.enable_history_summary = enable_history_summary
        # Holo enforces the JSON schema via vLLM guided decoding (structured_outputs
        # below), not a prompt hint, so needs_json_hint=False even in "cua_gym" mode.
        self._configure_prompt(prompt_style, needs_json_hint=False)

    def _call_api(self, messages: list) -> tuple:
        for attempt in range(self.max_retry):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.8,
                    "chat_template_kwargs": {"enable_thinking": True},
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
