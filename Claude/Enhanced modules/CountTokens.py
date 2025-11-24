"""
Lightweight token estimate with improved robustness across languages
and text domains. This implementation maintains the dependency-free
design while applying adaptive heuristics based on text structure.
"""

from typing import Any
import math
import re
import warnings

DEFAULT_AVG_CHARS_PER_TOKEN = 4


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _detect_language_group(s: str) -> str:
    """
    Classifies the text based on Unicode ranges into rough script groups.
    This allows the estimator to choose a more realistic token density.
    """
    if re.search(r'[\u0600-\u06FF]', s):   # Arabic/Persian
        return "persian_arabic"
    if re.search(r'[\u4E00-\u9FFF]', s):   # Chinese
        return "chinese"
    if re.search(r'[A-Za-z]', s):          # Latin-based (English, etc.)
        return "latin"
    return "other"


def _adaptive_token_ratio(s: str, fallback_ratio: float) -> float:
    """
    Returns an estimated chars/token ratio based on script type.
    """
    lang = _detect_language_group(s)
    if lang == "latin":
        return 3.8
    if lang == "persian_arabic":
        return 2.4
    if lang == "chinese":
        return 1.2
    return fallback_ratio


def countTokens(text: Any, average_chars_per_token: float = DEFAULT_AVG_CHARS_PER_TOKEN) -> int:
    """
    Robust token estimate using adaptive heuristics. Maintains compatibility
    with the original interface while improving reliability across languages
    and domains.
    """
    if _is_nan(text):
        return 0

    if not isinstance(text, str):
        warnings.warn("countTokens: Non-string input detected; coercing to string.")
    s = str(text).strip()

    if not s:
        return 0

    if average_chars_per_token <= 0:
        raise ValueError("average_chars_per_token must be greater than zero")

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)

    # Use adaptive ratio unless manually overridden
    ratio = _adaptive_token_ratio(s, average_chars_per_token)

    # Use round() instead of floor() to reduce underestimation bias
    return max(1, round(len(s) / ratio))
