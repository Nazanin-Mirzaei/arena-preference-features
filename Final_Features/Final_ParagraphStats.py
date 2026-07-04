"""
Paragraph Length Statistics Module

Provides lightweight paragraph-length statistics for LLM responses
using newline-based paragraph segmentation.

Supports:
    - Mean paragraph length
    - Population standard deviation of paragraph length
    - Unicode-aware word counting
    - Robust handling of None / NaN / empty inputs

This feature is useful for:
    - Measuring document structure
    - Quantifying response organization
    - Comparing formatting consistency across model outputs
"""

from __future__ import annotations

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
_PARAGRAPH_SPLIT_RE = re.compile(r"\n+")
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def paragraph_length_stats(text: Any) -> dict[str, float]:
    """
    Compute paragraph-length statistics for a text response.

    Paragraph length is measured as the number of words in each
    newline-separated paragraph.

    Parameters
    ----------
    text : Any
        Input response text.

    Returns
    -------
    dict[str, float]
        Dictionary containing:

        - mean_paragraph_length
        - sd_paragraph_length
    """

    # -------------------------
    # Input validation
    # -------------------------
    if _is_nan(text):
        return {
            "mean_paragraph_length": 0.0,
            "sd_paragraph_length": 0.0,
        }

    clean_text = str(text).strip()

    if not clean_text:
        return {
            "mean_paragraph_length": 0.0,
            "sd_paragraph_length": 0.0,
        }

    if clean_text.lower() == "nan":
        return {
            "mean_paragraph_length": 0.0,
            "sd_paragraph_length": 0.0,
        }

    # -------------------------
    # Paragraph segmentation
    # -------------------------
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SPLIT_RE.split(clean_text)
        if paragraph.strip()
    ]

    # -------------------------
    # Paragraph length calculation
    # -------------------------
    paragraph_lengths = [
        len(_WORD_RE.findall(paragraph))
        for paragraph in paragraphs
    ]

    if not paragraph_lengths:
        return {
            "mean_paragraph_length": 0.0,
            "sd_paragraph_length": 0.0,
        }

    # -------------------------
    # Statistics
    # -------------------------
    mean_value = sum(paragraph_lengths) / len(paragraph_lengths)

    variance = sum(
        (length - mean_value) ** 2
        for length in paragraph_lengths
    ) / len(paragraph_lengths)

    return {
        "mean_paragraph_length": float(mean_value),
        "sd_paragraph_length": (
            float(math.sqrt(variance))
            if len(paragraph_lengths) > 1
            else 0.0
        ),
    }


__all__ = ["paragraph_length_stats"]
