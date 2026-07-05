"""
Provides a robust, language-aware token count estimator suitable for
LLM evaluation pipelines. This implementation performs adaptive token
estimation without requiring heavy dependencies (e.g., SentencePiece,
TikToken, transformers).

It supports:
    - Unicode-based language detection
    - Adaptive chars-per-token ratios
    - Whitespace normalization
    - Safe handling of None / NaN / non-string inputs
"""

from typing import Any
import math
import re
#import warnings

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
# Language Group Detection (Unicode-based)
# ---------------------------------------------------------
def _detect_language_group(s: str) -> str:
    """
    Rough classification of script types based on Unicode ranges.
    Helps select more realistic chars-per-token ratios.
    """
    if re.search(r'[\u0600-\u06FF]', s):       # Arabic / Persian scripts
        return "persian_arabic"
    if re.search(r'[\u4E00-\u9FFF]', s):       # CJK (Chinese)
        return "chinese"
    if re.search(r'[A-Za-z]', s):              # Latin scripts
        return "latin"
    return "other"


def _adaptive_token_ratio(s: str, fallback: float) -> float:
    """
    Selects a chars/token ratio based on script heuristics.
    """
    lang = _detect_language_group(s)

    if lang == "latin":
        return 3.8
    if lang == "persian_arabic":
        return 2.4
    if lang == "chinese":
        return 1.2

    return fallback


# ---------------------------------------------------------
# Main Function: Token Estimator
# ---------------------------------------------------------
def count_tokens(text: Any, average_chars_per_token: float = DEFAULT_AVG_CHARS_PER_TOKEN) -> int:
    """
    Estimate the number of tokens in a text using an adaptive heuristic.
    This is NOT an exact tokenizer — it is designed for large-scale,
    dependency-free statistical analysis.

    Parameters
    ----------
    text : Any
        Input text (converted to string if needed)
    average_chars_per_token : float
        Fallback heuristic (default = 4)

    Returns
    -------
    int
        Estimated token count
    """
    if _is_nan(text):
        return 0

    if not isinstance(text, str):
        #warnings.warn("countTokens: Non-string input detected; coercing to string.")
    s = str(text).strip()

    if not s:
        return 0

    if average_chars_per_token <= 0:
        raise ValueError("average_chars_per_token must be greater than zero")

    # Normalize whitespace to avoid overcounting
    s = re.sub(r"\s+", " ", s)

    # Select adaptive token ratio
    ratio = _adaptive_token_ratio(s, average_chars_per_token)

    # Round instead of floor to reduce bias
    est = len(s) / ratio
    return max(1, round(est))


__all__ = ["count_tokens"]
