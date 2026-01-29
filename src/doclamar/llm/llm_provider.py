import os
from typing import Optional
from .base import BaseLLM
from .gemini_llm import GeminiLLM

_llm_instance: Optional[BaseLLM] = None


def get_llm() -> BaseLLM:
    """
    Returns a singleton LLM instance.
    Switching providers here does NOT affect agents.
    """
    global _llm_instance

    if _llm_instance is None:
        provider = os.getenv("LLM_PROVIDER", "gemini")

        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")

            _llm_instance = GeminiLLM(api_key=api_key)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    return _llm_instance
