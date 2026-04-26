"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float


class BaseLLMClient(ABC):
    """Base class all LLM clients must implement."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        """Send a prompt to the model and return a standardized response."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai')."""
