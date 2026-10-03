"""LLM providers (Groq, Gemini) with timeouts, retries and runtime (re)configuration.

The API key normally comes from the desktop app's settings screen (stored
encrypted by Electron and passed in at launch) or, when developing, from a
local .env file. It is never bundled into the installer.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

PROVIDERS = {
    "groq": {"label": "Groq", "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
             "default_model": "openai/gpt-oss-120b"},
    "gemini": {"label": "Gemini", "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
               "default_model": "gemini-2.5-flash"},
}


class LLMError(RuntimeError):
    """A user-presentable LLM failure (bad key, rate limit, outage...)."""


class LLMNotConfigured(LLMError):
    pass


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> Optional["LLMConfig"]:
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        candidates = [provider] if provider in PROVIDERS else list(PROVIDERS)
        for name in candidates:
            spec = PROVIDERS[name]
            key = os.environ.get(spec["key_env"], "").strip()
            if key:
                model = os.environ.get(spec["model_env"], "").strip() or spec["default_model"]
                return cls(provider=name, api_key=key, model=model)
        return None


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    try:
        return float(headers.get("retry-after")) if headers and headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


def _is_transient(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 429 or status >= 500
    return type(exc).__name__ in {
        "APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout",
        "ConnectError", "TimeoutException", "RemoteProtocolError",
    }


class _GroqBackend:
    def __init__(self, config: LLMConfig, timeout: float):
        from groq import Groq

        self.config = config
        self.client = Groq(api_key=config.api_key, timeout=timeout, max_retries=0)

    def complete(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        extra = {}
        if "gpt-oss" in self.config.model:
            # Reasoning models spend completion tokens thinking before they answer;
            # keep that short and leave headroom so small calls don't come back empty.
            extra["reasoning_effort"] = "low"
            max_tokens = max(max_tokens, 1024)
        response = self.client.chat.completions.create(
            model=self.config.model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, **extra,
        )
        return response.choices[0].message.content or ""

    def check(self) -> None:
        self.client.models.retrieve(self.config.model)  # validates key + model, costs nothing


class _GeminiBackend:
    def __init__(self, config: LLMConfig, timeout: float):
        from google import genai
        from google.genai import types

        self.config = config
        self.types = types
        self.client = genai.Client(
            api_key=config.api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    def complete(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                # Leave room for "thinking" models, which spend output tokens before answering.
                max_output_tokens=max(max_tokens, 2048),
            ),
        )
        return response.text or ""

    def check(self) -> None:
        self.client.models.get(model=self.config.model)


_BACKENDS = {"groq": _GroqBackend, "gemini": _GeminiBackend}


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None, timeout: float = 60.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._backend = None
        self._config: Optional[LLMConfig] = None
        if config:
            self._set(config)

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(LLMConfig.from_env())

    @property
    def configured(self) -> bool:
        return self._backend is not None

    def describe(self) -> dict:
        cfg = self._config
        return {"configured": self.configured, "provider": cfg.provider if cfg else None,
                "model": cfg.model if cfg else None}

    def _set(self, config: LLMConfig) -> None:
        if config.provider not in _BACKENDS:
            raise LLMError(f"Unknown provider '{config.provider}'. Use one of: {', '.join(_BACKENDS)}.")
        backend = _BACKENDS[config.provider](config, self.timeout)
        with self._lock:
            self._backend, self._config = backend, config

    def configure(self, provider: str, api_key: str, model: Optional[str] = None, validate: bool = True) -> dict:
        provider = provider.strip().lower()
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}.")
        if not api_key.strip():
            raise LLMError("API key is empty.")
        config = LLMConfig(provider, api_key.strip(), (model or "").strip() or PROVIDERS[provider]["default_model"])
        backend = _BACKENDS[provider](config, self.timeout)
        if validate:
            self._call(backend.check, provider=PROVIDERS[provider]["label"])
        with self._lock:
            self._backend, self._config = backend, config
        logger.info("LLM configured: %s / %s", config.provider, config.model)
        return self.describe()

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1500, temperature: float = 0.2) -> str:
        backend = self._backend
        if backend is None:
            raise LLMNotConfigured("No LLM API key configured. Add one in Settings.")
        text = self._call(backend.complete, prompt, system, max_tokens, temperature)
        if not text.strip():
            raise LLMError("The model returned an empty response.")
        return text.strip()

    def _call(self, fn, *args, provider: Optional[str] = None):
        if provider is None:
            provider = PROVIDERS[self._config.provider]["label"] if self._config else "The provider"
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args)
            except Exception as e:
                status = _status_code(e)
                if status in (401, 403):
                    raise LLMError(f"{provider} rejected the API key ({status}). Check it in Settings.") from e
                if status == 404:
                    raise LLMError(f"Model not found on {provider}: {e}") from e
                if not _is_transient(e) or attempt == self.max_retries:
                    if status == 429:
                        raise LLMError(f"{provider} rate limit reached. Wait a minute and try again.") from e
                    raise LLMError(f"{provider} request failed: {e}") from e
                delay = _retry_after(e) or (2 ** attempt + random.random())
                delay = min(delay, 30)
                logger.warning("LLM call failed (%s), retrying in %.1fs", e, delay)
                time.sleep(delay)
