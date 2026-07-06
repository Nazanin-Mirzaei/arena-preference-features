"""LLM judge API client."""

from __future__ import annotations

from openai import OpenAI

from .config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


class OpenAIJudge:
    """OpenAI-backed LLM judge."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    def evaluate_raw(self, judge_prompt: str) -> str:
        """Send a judge prompt to the model and return raw text output."""
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": judge_prompt}],
            temperature=self.temperature,
        )
        return response.output_text

    def evaluate(self, judge_prompt: str) -> str:
        """Alias for evaluate_raw."""
        return self.evaluate_raw(judge_prompt)

