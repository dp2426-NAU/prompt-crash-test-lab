"""Together AI client for Llama models."""

import time

from together import Together

from .base import BaseLLMClient, LLMResponse


class TogetherClient(BaseLLMClient):
    def __init__(self, api_key: str = "", model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", **kwargs):
        from config.settings import TOGETHER_API_KEY
        self.client = Together(api_key=api_key or TOGETHER_API_KEY)
        self.model_id = model_id

    @property
    def provider_name(self) -> str:
        return "together"

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.perf_counter() - start) * 1000

        usage = response.usage
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            model=self.model_id,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )
