"""
LLM Client — async interface to vLLM OpenAI-compatible endpoint.

Handles batched generation, retries, and rate limiting for high-throughput
synthetic data generation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class LLMConfig:
    """Configuration for vLLM connection."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3-235B-A22B-GPTQ-Int4"
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    max_concurrent: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0


class LLMClient:
    """
    Async client for vLLM OpenAI-compatible API.

    Supports batched generation with concurrency control.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._total_tokens = 0
        self._total_requests = 0
        self._start_time: Optional[float] = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single completion."""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        async with self._semaphore:
            return await self._generate_with_retry(messages, max_tokens, temperature)

    async def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Generate completions for a batch of prompts concurrently."""
        if self._start_time is None:
            self._start_time = time.time()

        tasks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            tasks.append(self.generate(messages, max_tokens=max_tokens, system_prompt=system_prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for r in results:
            if isinstance(r, Exception):
                outputs.append(f"[ERROR: {type(r).__name__}: {r}]")
            else:
                outputs.append(r)

        return outputs

    async def _generate_with_retry(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> str:
        """Generate with exponential backoff retry."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                )

                self._total_requests += 1
                if response.usage:
                    self._total_tokens += response.usage.total_tokens

                content = response.choices[0].message.content or ""
                return content

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise RuntimeError(f"Failed after {self.config.max_retries} attempts: {last_error}")

    @property
    def stats(self) -> dict:
        """Return generation statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "elapsed_seconds": round(elapsed, 1),
            "tokens_per_second": round(self._total_tokens / elapsed, 1) if elapsed > 0 else 0,
            "requests_per_minute": round(self._total_requests / (elapsed / 60), 1) if elapsed > 0 else 0,
        }

    async def health_check(self) -> bool:
        """Check if the vLLM server is responsive."""
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False
