"""Robust Sentence Counting Feature Module

This module estimates the number of sentences in a text using lightweight
heuristics based on punctuation and newline structure.
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
# Main Feature: Sentence Counter
# ---------------------------------------------------------
def count_sentences(text: Any) -> int:
    """
    Estimate number of sentences in a text using simple heuristics.

    Splits on:
        - sentence-ending punctuation (. ! ?)
        - newline characters
    """

    clean = _clean_text(text)

    # -------------------------
    # Edge case: empty input
    # -------------------------
    if not clean:
        return 0

    # -------------------------
    # Sentence splitting rules
    # -------------------------
    sentence_re = re.compile(r"(?<=[.!?])\s+|\n+")
    word_re = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

    sentences = [
        part.strip()
        for part in sentence_re.split(clean)
        if part.strip()
    ]

    # -------------------------
    # Output logic
    # -------------------------
    if sentences:
        return len(sentences)

    return 1 if word_re.search(clean) else 0

__all__ = ["count_sentences"]
