"""Comprehensive Punctuation Counting Feature Module

This module extracts a lightweight structural feature from text by counting
common punctuation marks. It is useful for measuring writing style complexity,
formality, and sentence structure density.
"""

import math
import re
from typing import Any


# ---------------------------------------------------------
# Text Cleaning Utility
# ---------------------------------------------------------
def _clean_text(text: Any) -> str:
    """
    Normalize input text into a clean string.

    Handles:
        - None values
        - NaN floats
        - string "nan"
        - whitespace trimming
    """

    if text is None:
        return ""

    if isinstance(text, float) and math.isnan(text):
        return ""

    value = str(text).strip()

    if value.lower() == "nan":
        return ""

    return value


# ---------------------------------------------------------
# Main Feature: Punctuation Counter
# ---------------------------------------------------------
def count_punctuation(text: Any) -> int:
    """
    Count occurrences of common punctuation marks in text.

    This feature provides a simple proxy for writing structure density.
    """

    clean = _clean_text(text)

    # -------------------------
    # Edge case: empty input
    # -------------------------
    if not clean:
        return 0

    # -------------------------
    # Punctuation pattern
    # -------------------------
    punct_re = re.compile(r"[.,!?;:()\[\]{}\"'`…]")

    # -------------------------
    # Feature computation
    # -------------------------
    return len(punct_re.findall(clean))

__all__ = ["count_punctuation"]
