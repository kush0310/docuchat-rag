"""LLM clients. Like embedders, anything with `generate(system, prompt) -> str` works."""
from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        import anthropic

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")
