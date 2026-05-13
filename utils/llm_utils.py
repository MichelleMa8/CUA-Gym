"""
Unified LLM calling utilities for OpenRLVR pipeline.

Now split into provider-specific callers (OpenAI vs Claude) under a shared base.
"""

import os
import json
import hashlib
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union, Type, List
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import anthropic
import logging


class BaseLLMCaller(ABC):
    """Base class handling cache, stats, and retry orchestration."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str],
        temperature: float,
        use_cache: bool,
        cache_dir: Optional[Union[str, Path]],
        cache_persistence: bool,
        retry_attempts: int,
        api_delay: float,
        stats: Optional[Dict[str, Any]],
        stats_lock: Optional[threading.Lock],
    ):
        self.model = model
        self.temperature = temperature
        self.use_cache = use_cache
        self.cache_persistence = cache_persistence
        self.retry_attempts = retry_attempts
        self.api_delay = api_delay
        self.stats = stats or {}
        self.stats_lock = stats_lock or threading.Lock()

        if use_cache:
            self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / f"llm_cache_{model.replace('/', '_')}.json"
            self.llm_cache = self._load_cache()
            self.cache_lock = threading.Lock()
            self._async_cache_lock = None
        else:
            self.cache_dir = None
            self.cache_file = None
            self.llm_cache = None
            self.cache_lock = None
            self._async_cache_lock = None

        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(self.__class__.__name__)

    # ----- cache helpers -----
    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_file or not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.logger.debug(f"Loaded {len(data)} cached responses from {self.cache_file}")
                    return data
        except Exception as e:
            self.logger.warning(f"Could not load cache from {self.cache_file}: {e}")
        return {}

    def _save_cache(self):
        if not self.cache_file or not self.llm_cache:
            return
        try:
            with self.cache_lock:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_cache, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved {len(self.llm_cache)} cached responses to {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Could not save cache to {self.cache_file}: {e}")

    def _get_cache_key(self, prompt: str, system_content: str = "", temperature: Optional[float] = None) -> str:
        content_to_hash = f"{self.model}:{system_content}:{prompt}:{temperature or self.temperature}"
        return hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()

    def _update_stats(self, **kwargs):
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value

    # ----- provider-specific invocation -----
    @abstractmethod
    def _invoke_sync(self, **kwargs) -> str:
        ...

    @abstractmethod
    async def _invoke_async(self, **kwargs) -> str:
        ...

    # ----- public API -----
    def call_llm(
        self,
        prompt: str,
        system_content: str = "",
        temperature: Optional[float] = None,
        retry_attempts: Optional[int] = None,
        api_delay: Optional[float] = None,
        schema: Optional[Type[BaseModel]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Optional[str]:
        retry_attempts = retry_attempts or self.retry_attempts
        api_delay = api_delay or self.api_delay
        temperature = temperature or self.temperature

        cache_key = None
        if self.use_cache and self.llm_cache is not None:
            cache_key = self._get_cache_key(prompt, system_content, temperature)
            with self.cache_lock:
                if cache_key in self.llm_cache:
                    self._update_stats(cache_hits=1)
                    self.logger.debug("Using cached response")
                    return self.llm_cache[cache_key]

        for attempt in range(retry_attempts):
            try:
                time.sleep(api_delay)
                self._update_stats(llm_calls=1)

                content = self._invoke_sync(
                    prompt=prompt,
                    system_content=system_content,
                    temperature=temperature,
                    schema=schema,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                if cache_key and self.llm_cache is not None:
                    with self.cache_lock:
                        self.llm_cache[cache_key] = content
                    if self.cache_persistence:
                        self._save_cache()

                self.logger.debug(f"Response received from {self.model}")
                return content

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retry_attempts - 1:
                    wait_time = (2 ** attempt) * api_delay
                    self.logger.debug(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {retry_attempts} attempts failed")
                    self._update_stats(errors=1)
                    return None
        return None

    async def call_llm_async(
        self,
        prompt: str,
        system_content: str = "",
        temperature: Optional[float] = None,
        schema: Optional[Type[BaseModel]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Optional[str]:
        temperature = temperature or self.temperature

        cache_key = None
        if self.use_cache and self.llm_cache is not None:
            cache_key = self._get_cache_key(prompt, system_content, temperature)
            if self._async_cache_lock is None:
                self._async_cache_lock = asyncio.Lock()
            async with self._async_cache_lock:
                if cache_key in self.llm_cache:
                    self._update_stats(cache_hits=1)
                    return self.llm_cache[cache_key]

        try:
            self._update_stats(llm_calls=1)
            content = await self._invoke_async(
                prompt=prompt,
                system_content=system_content,
                temperature=temperature,
                schema=schema,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            if cache_key and self.llm_cache is not None:
                async with self._async_cache_lock:
                    self.llm_cache[cache_key] = content
                if self.cache_persistence:
                    self._save_cache()
            return content
        except Exception as e:
            self.logger.error(f"Async LLM call failed: {e}")
            self._update_stats(errors=1)
            return None


class OpenAILLMCaller(BaseLLMCaller):
    """OpenAI provider: supports structured outputs and function calling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url or None)

    def _invoke_sync(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        system_content = kwargs.get("system_content", "")
        temperature = kwargs.get("temperature", self.temperature)
        schema = kwargs.get("schema")
        messages = kwargs.get("messages")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        # structured outputs (responses API) when no tools/messages
        if schema and not tools and messages is None:
            msg_payload: List[Dict[str, Any]] = []
            if system_content:
                msg_payload.append({"role": "system", "content": system_content})
            msg_payload.append({"role": "user", "content": prompt})
            response = self.client.responses.parse(  # type: ignore[attr-defined]
                model=self.model,
                input=msg_payload,
                temperature=temperature,
                text_format=schema,
            )
            parsed = response.output_parsed
            return json.dumps(parsed.model_dump(), ensure_ascii=False)

        # function calling
        if tools or messages is not None:
            msg_payload = messages or []
            if not messages:
                if system_content:
                    msg_payload.append({"role": "system", "content": system_content})
                msg_payload.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msg_payload,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice or "auto",
            )
            return response.choices[0].message.content

        # default chat
        msg_payload: List[Dict[str, Any]] = []
        if system_content:
            msg_payload.append({"role": "system", "content": system_content})
        msg_payload.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msg_payload,
            temperature=temperature,
        )
        return response.choices[0].message.content

    async def _invoke_async(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        system_content = kwargs.get("system_content", "")
        temperature = kwargs.get("temperature", self.temperature)
        schema = kwargs.get("schema")
        messages = kwargs.get("messages")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        if schema and not tools and messages is None:
            msg_payload: List[Dict[str, Any]] = []
            if system_content:
                msg_payload.append({"role": "system", "content": system_content})
            msg_payload.append({"role": "user", "content": prompt})
            response = await self.async_client.responses.parse(  # type: ignore[attr-defined]
                model=self.model,
                input=msg_payload,
                temperature=temperature,
                text_format=schema,
            )
            parsed = response.output_parsed
            return json.dumps(parsed.model_dump(), ensure_ascii=False)

        if tools or messages is not None:
            msg_payload = messages or []
            if not messages:
                if system_content:
                    msg_payload.append({"role": "system", "content": system_content})
                msg_payload.append({"role": "user", "content": prompt})
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=msg_payload,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice or "auto",
            )
            return response.choices[0].message.content

        msg_payload: List[Dict[str, Any]] = []
        if system_content:
            msg_payload.append({"role": "system", "content": system_content})
        msg_payload.append({"role": "user", "content": prompt})
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=msg_payload,
            temperature=temperature,
        )
        return response.choices[0].message.content


class ClaudeLLMCaller(BaseLLMCaller):
    """Claude provider: simple messages.create; async falls back to sync."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        self.async_client = None  # not implemented

    def _invoke_sync(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        system_content = kwargs.get("system_content", "")
        temperature = kwargs.get("temperature", self.temperature)

        messages = []
        prompt_content = [
            {"type": "text", "cache_control": {"type": "ephemeral"}, "text": prompt}
        ]
        messages.append({"role": "user", "content": prompt_content})

        response = self.client.messages.create(
            system=[{"type": "text", "text": system_content}] if system_content else None,
            model=self.model,
            messages=messages,
            max_tokens=12000,
            temperature=temperature,
        )
        return response.content[0].text

    async def _invoke_async(self, **kwargs) -> str:
        # Claude async not supported; fallback to sync
        return self._invoke_sync(**kwargs)


# Convenience factory
def create_llm_caller(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    use_cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_persistence: bool = True,
    retry_attempts: int = 3,
    api_delay: float = 0.5,
    stats: Optional[Dict[str, Any]] = None,
    stats_lock: Optional[threading.Lock] = None,
):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set OPENAI_API_KEY environment variable.")

    common_kwargs = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        use_cache=use_cache,
        cache_dir=cache_dir,
        cache_persistence=cache_persistence,
        retry_attempts=retry_attempts,
        api_delay=api_delay,
        stats=stats,
        stats_lock=stats_lock,
    )

    if "claude" in model.lower():
        return ClaudeLLMCaller(**common_kwargs)
    return OpenAILLMCaller(**common_kwargs)


# Backward compatibility alias
def LLMCaller(*args, **kwargs):
    """Backward compatible factory returning appropriate caller."""
    return create_llm_caller(*args, **kwargs)
