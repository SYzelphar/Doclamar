from __future__ import annotations
import os
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system: str = "") -> str:
        pass


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        logger.info(f"GeminiProvider initialized: {model}")

    def generate(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self.model.generate_content(full_prompt)
        return response.text.strip()

    def generate_json(self, prompt: str, system: str = "") -> str:
        import google.generativeai as genai
        sys_instruction = (system + "\n\nRespond ONLY with valid JSON. No markdown fences. No explanation.") if system else "Respond ONLY with valid JSON. No markdown fences. No explanation."
        model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        full_prompt = f"{sys_instruction}\n\n{prompt}"
        response = model.generate_content(full_prompt)
        return response.text.strip()


class GroqProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment.")
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"GroqProvider initialized: {model}")

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def generate_json(self, prompt: str, system: str = "") -> str:
        sys_msg = (system + "\n\nRespond ONLY with valid JSON. No markdown. No explanation.") if system else "Respond ONLY with valid JSON. No markdown. No explanation."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()


_provider_instance: BaseLLMProvider | None = None


def get_llm() -> BaseLLMProvider:
    global _provider_instance
    if _provider_instance is None:
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        if provider == "gemini":
            _provider_instance = GeminiProvider()
        elif provider == "groq":
            _provider_instance = GroqProvider()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'gemini' or 'groq'.")
    return _provider_instance


def set_llm(provider: BaseLLMProvider):
    global _provider_instance
    _provider_instance = provider
