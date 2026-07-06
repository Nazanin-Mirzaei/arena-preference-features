"""
This Module provides lightweight paragraph-length statistics for LLM responses
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

from typing import Any
import math
import re


# ---------------------------------------------------------
# Utility: NaN-safe checking
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    """
    Return True if the input should be treated as missing.
    """

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
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_paragraph_statistics(text: Any) -> dict[str, float]:
    """
    Compute paragraph-length statistics for a text response.
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
    # Length computation
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
        "paragraph_length_mean": float(mean_value),
        "paragraph_length_std": (
            float(math.sqrt(variance))
            if len(paragraph_lengths) > 1
            else 0.0
        ),
    }


__all__ = ["extract_paragraph_statistics"]
