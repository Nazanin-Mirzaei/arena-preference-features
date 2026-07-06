"""
This Module provides a robust, language-aware token count estimator suitable for
LLM evaluation pipelines. This implementation performs adaptive token
estimation without requiring heavy dependencies (e.g., SentencePiece,
TikToken, transformers).
"""

from typing import Any
import math
import re
# import warnings

DEFAULT_AVG_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------
# Utility: Robust NaN Detection
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ---------------------------------------------------------
# Language Group Detection
# ---------------------------------------------------------
def _detect_language_group(s: str) -> str:
    if re.search(r'[\u0600-\u06FF]', s):
        return "persian_arabic"
    if re.search(r'[\u4E00-\u9FFF]', s):
        return "chinese"
    if re.search(r'[A-Za-z]', s):
        return "latin"
    return "other"


def _adaptive_token_ratio(s: str, fallback: float) -> float:
    lang = _detect_language_group(s)

    if lang == "latin":
        return 3.8
    if lang == "persian_arabic":
        return 2.4
    if lang == "chinese":
        return 1.2

    return fallback


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def count_tokens(text: Any, average_chars_per_token: float = DEFAULT_AVG_CHARS_PER_TOKEN) -> int:

    if _is_nan(text):
        return 0

    # FIXED BLOCK 👇
    if not isinstance(text, str):
        text = str(text)
        # optional:
        # warnings.warn("Non-string input detected; coercing to string.")

    s = text.strip()

    if not s:
        return 0

    if average_chars_per_token <= 0:
        raise ValueError("average_chars_per_token must be greater than zero")

    s = re.sub(r"\s+", " ", s)

    ratio = _adaptive_token_ratio(s, average_chars_per_token)

    est = len(s) / ratio
    return max(1, round(est))


__all__ = ["count_tokens"]
