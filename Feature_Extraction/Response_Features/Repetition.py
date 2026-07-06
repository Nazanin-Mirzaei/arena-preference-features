"""
This Module provides lightweight estimation of lexical repetition in LLM
responses using heuristic word-frequency analysis.

Supports:
    - Unicode-aware word extraction
    - Case-insensitive lexical normalization
    - Inverse lexical diversity estimation
    - Robust handling of None / NaN / empty inputs

This feature is useful for:
    - Measuring lexical diversity
    - Detecting repetitive responses
    - Comparing writing variation across model outputs
"""

from typing import Any
import math
import re


# ---------------------------------------------------------
# Utility: NaN-safe checking
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    """Return True if the input should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def compute_repetition_density(text: Any) -> float:
    if _is_nan(text):
        return 0.0

    clean_text = str(text).strip()

    if not clean_text or clean_text.lower() == "nan":
        return 0.0

    words = [
        word.lower()
        for word in _WORD_RE.findall(clean_text)
    ]

    if not words:
        return 0.0

    repeated_tokens = len(words) - len(set(words))

    return float(repeated_tokens / len(words))


__all__ = ["compute_repetition_density"]
