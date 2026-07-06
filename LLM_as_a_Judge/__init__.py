"""LLM-as-a-Judge package for independent row-level correctness evaluation."""

from .prompt import build_judge_prompt
from .parser import parse_json_output

__all__ = ["build_judge_prompt", "parse_json_output"]
