from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract interface for all LLM providers.
    Agents must depend ONLY on this interface.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
