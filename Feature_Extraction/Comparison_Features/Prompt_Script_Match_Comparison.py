"""
This module extracts pairwise comparison features capturing whether each
response's dominant Unicode script matches the prompt's dominant script.
This is a cheap, dependency-free fallback for prompt-language-match
signals (see Prompt_Language_Match_Comparison) - useful when full
language detection is unavailable, misfires, or returns "unknown", since
a script mismatch (e.g. prompt in Arabic, response in Latin) is still a
strong indicator that a response answered in the wrong language.

Currently supports:
- prompt script match (per side + agreement signal)
"""

from typing import Any
import math


# ---------------------------------------------------------
# Helper: robust string normalization
# ---------------------------------------------------------
def _normalize_script(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    if text in ("", "nan", "none", "other"):
        return ""
    return text


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_prompt_script_match_comparison(
    prompt_script: Any,
    a_script: Any,
    b_script: Any,
) -> dict:
    """
    Compare each response's detected script against the prompt's script.

    Args:
        prompt_script: detect_script output for user_prompt
        a_script: detect_script output for response_a
        b_script: detect_script output for response_b

    Returns:
        {
            "a_prompt_script_match": bool,     # response_a script == prompt script
            "b_prompt_script_match": bool,     # response_b script == prompt script
            "only_one_matched_script": bool,   # exactly one side matched, the other didn't
        }
    """

    prompt = _normalize_script(prompt_script)
    a = _normalize_script(a_script)
    b = _normalize_script(b_script)

    a_match = bool(prompt) and a == prompt
    b_match = bool(prompt) and b == prompt

    return {
        "a_prompt_script_match": a_match,
        "b_prompt_script_match": b_match,
        "only_one_matched_script": a_match != b_match,
    }


__all__ = ["extract_prompt_script_match_comparison"]
