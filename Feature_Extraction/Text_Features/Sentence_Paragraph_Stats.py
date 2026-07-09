"""
This module computes the population standard deviation of sentence counts
across paragraphs in a text.

It uses lightweight heuristics for both sentence and paragraph segmentation.
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
# Paragraphs are separated by blank lines (\n\n or more)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def compute_sentence_per_paragraph_std(text: Any) -> float:
    """
    Compute population standard deviation of sentence counts per paragraph.
    """

    # -------------------------
    # Input validation
    # -------------------------
    if _is_nan(text):
        return 0.0

    clean_text = str(text).strip()

    if not clean_text:
        return 0.0

    if clean_text.lower() == "nan":
        return 0.0

    # -------------------------
    # Paragraph segmentation
    # -------------------------
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SPLIT_RE.split(clean_text)
        if paragraph.strip()
    ]

    # -------------------------
    # Sentence counting per paragraph
    # -------------------------
    sentence_counts = []

    for paragraph in paragraphs:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(paragraph)
            if sentence.strip()
        ]
        sentence_counts.append(len(sentences))

    # -------------------------
    # Edge case
    # -------------------------
    if len(sentence_counts) <= 1:
        return 0.0

    # -------------------------
    # Statistics (population variance → std)
    # -------------------------
    mean_value = sum(sentence_counts) / len(sentence_counts)

    variance = sum(
        (count - mean_value) ** 2
        for count in sentence_counts
    ) / len(sentence_counts)

    return math.sqrt(variance)


__all__ = ["compute_sentence_per_paragraph_std"]
