"""Qwen (Aliyun DashScope) model backend for run_trajectories.py.

Talks to Aliyun DashScope's OpenAI-compatible `/chat/completions` endpoint
(https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope).
Like MiniMax, DashScope's compatible-mode endpoint does not expose vLLM-style
guided decoding, so the JSON action schema is enforced through the system
prompt plus `response_format={"type": "json_object"}`.

Env vars:
    DASHSCOPE_API_KEY  bearer token (required) — get one at
                       https://bailian.console.aliyun.com/
    QWEN_BASE_URL      API base (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    QWEN_MODEL         model name (default: qwen3.7-plus)
"""

import json
import os
import re
import time

import requests

from agent_common import (
    AGENT_SYSTEM_PROMPT,
    BaseAgent,
    STEP_SCHEMA,
    is_context_exceeded,
    is_retryable_api_error,
)

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.7-plus")

_JSON_SCHEMA_HINT = (
    "\n\nYou MUST respond with a single JSON object matching this schema, and "
    "nothing else (no prose, no markdown code fences):\n" + json.dumps(STEP_SCHEMA)
)


class QwenAgent(BaseAgent):
    history_n = 5             # Qwen3.7-plus has a large context window, like MiniMax M3
    summary_interval = 10
    max_retry = 3
    enable_history_summary = False
    system_prompt = AGENT_SYSTEM_PROMPT + _JSON_SCHEMA_HINT

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None,
                 enable_history_summary: bool = None):
        self.base_url = (base_url or QWEN_BASE_URL).rstrip("/")
        self.model = model or QWEN_MODEL
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set (required for --model-provider qwen)")
        if enable_history_summary is not None:
            self.enable_history_summary = enable_history_summary

    def _call_api(self, messages: list) -> tuple:
        for attempt in range(self.max_retry):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 4096,
                        "temperature": 0.8,
                        "response_format": {"type": "json_object"},
                    },
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
