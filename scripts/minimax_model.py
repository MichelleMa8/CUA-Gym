"""MiniMax M3 model backend for run_trajectories.py.

Talks to MiniMax's OpenAI-compatible `/chat/completions` endpoint. Unlike
the local vLLM server used for Holo, MiniMax M3 does not expose vLLM-style
guided decoding (`structured_outputs` / `chat_template_kwargs`), so the JSON
action schema is enforced through the system prompt plus
`response_format={"type": "json_object"}`.

Confirmed against https://platform.minimax.io/docs (2026-07): the
OpenAI-compatible base URL is https://api.minimax.io/v1, chat completions
support `image_url` content parts the same way OpenAI does, and pricing for
the standard (<=512k context) tier is $0.30 / M input tokens and
$1.20 / M output tokens. MiniMax does not publish an image-tokenization
formula, so per-image token counts here are an estimate, not official.

Env vars:
    MINIMAX_API_KEY   bearer token (required)
    MINIMAX_BASE_URL  API base (default: https://api.minimax.io/v1)
    MINIMAX_MODEL     model name (default: MiniMax-M3)
"""

import json
import os
import re
import time

import requests

from agent_common import AGENT_SYSTEM_PROMPT, BaseAgent, STEP_SCHEMA, is_context_exceeded

MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M3")

_JSON_SCHEMA_HINT = (
    "\n\nYou MUST respond with a single JSON object matching this schema, and "
    "nothing else (no prose, no markdown code fences):\n" + json.dumps(STEP_SCHEMA)
)


class MinimaxAgent(BaseAgent):
    history_n = 5             # MiniMax M3 has a much larger context window than Holo
    summary_interval = 10
    max_retry = 3
    enable_history_summary = False
    system_prompt = AGENT_SYSTEM_PROMPT + _JSON_SCHEMA_HINT

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None,
                 enable_history_summary: bool = None):
        self.base_url = (base_url or MINIMAX_BASE_URL).rstrip("/")
        self.model = model or MINIMAX_MODEL
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set (required for --model-provider minimax_m3)")
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
