"""LLM model client implementations.

Client classes are imported lazily to avoid requiring all SDK packages
to be installed when only using a subset of providers.
"""

from .base import BaseLLMClient


def _get_client_class(provider: str):
    """Lazily import and return the client class for a provider."""
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient
    elif provider == "google":
        from .gemini_client import GeminiClient
        return GeminiClient
    elif provider == "together":
        from .together_client import TogetherClient
        return TogetherClient
    else:
        raise ValueError(f"Unknown provider: {provider}. Available: openai, anthropic, google, together")


CLIENT_REGISTRY = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "together": "together",
}


def get_client(provider: str, **kwargs) -> BaseLLMClient:
    """Factory to get the right client by provider name."""
    if provider not in CLIENT_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(CLIENT_REGISTRY.keys())}")
    cls = _get_client_class(provider)
    return cls(**kwargs)
