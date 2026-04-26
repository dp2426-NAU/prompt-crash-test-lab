"""Anthropic Claude client implementation."""

import time

from anthropic import Anthropic

from .base import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str = "", model_id: str = "claude-3-5-sonnet-20241022", **kwargs):
        from config.settings import ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        self.model_id = model_id

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.content[0].text if response.content else "",
            model=self.model_id,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency,
        )
