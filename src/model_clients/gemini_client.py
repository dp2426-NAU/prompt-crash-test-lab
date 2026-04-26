"""Google Gemini client implementation."""

import time

import google.generativeai as genai

from .base import BaseLLMClient, LLMResponse


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str = "", model_id: str = "gemini-1.5-pro", **kwargs):
        from config.settings import GOOGLE_API_KEY
        genai.configure(api_key=api_key or GOOGLE_API_KEY)
        self.model_id = model_id
        self.model = genai.GenerativeModel(model_id)

    @property
    def provider_name(self) -> str:
        return "google"

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1024)

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        start = time.perf_counter()
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        latency = (time.perf_counter() - start) * 1000

        tokens_in = 0
        tokens_out = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            text=response.text if response.parts else "",
            model=self.model_id,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency,
        )
