"""
This module extracts pairwise comparison features capturing whether each
response answered in the same language as the user prompt. Answering in
the wrong language is a strong quality/loser signal independent of
content, and when only one side matches the prompt language, that side
has a decisive advantage.

Currently supports:
- prompt language match (per side + agreement signal)
"""

from typing import Any
import math


# ---------------------------------------------------------
# Helper: robust string normalization
# ---------------------------------------------------------
def _normalize_lang(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    if text in ("", "nan", "none", "unknown"):
        return ""
    return text


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_prompt_language_match_comparison(
    prompt_language: Any,
    a_language: Any,
    b_language: Any,
) -> dict:
    """
    Compare each response's detected language against the prompt's language.

    Args:
        prompt_language: primary_language detected for user_prompt
        a_language: primary_language detected for response_a
        b_language: primary_language detected for response_b

    Returns:
        {
            "a_prompt_lang_match": bool,     # response_a language == prompt language
            "b_prompt_lang_match": bool,     # response_b language == prompt language
            "only_one_matched_lang": bool,   # exactly one side matched, the other didn't
        }
    """

    prompt_lang = _normalize_lang(prompt_language)
    a_lang = _normalize_lang(a_language)
    b_lang = _normalize_lang(b_language)

    a_match = bool(prompt_lang) and a_lang == prompt_lang
    b_match = bool(prompt_lang) and b_lang == prompt_lang

    return {
        "a_prompt_lang_match": a_match,
        "b_prompt_lang_match": b_match,
        "only_one_matched_lang": a_match != b_match,
    }


__all__ = ["extract_prompt_language_match_comparison"]
