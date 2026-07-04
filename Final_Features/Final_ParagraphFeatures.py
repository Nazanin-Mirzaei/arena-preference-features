"""
Paragraph Count Feature Module

Provides lightweight paragraph counting for LLM responses using
newline-based segmentation.

Supports:
    - Newline-based paragraph detection
    - Removal of empty paragraphs
    - Robust handling of None / NaN / empty inputs

This feature is useful for:
    - Measuring document structure
    - Quantifying response organization
    - Comparing formatting styles across model outputs
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


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def num_paragraphs(text: Any) -> int:
    """
    Count the number of non-empty paragraphs in a text response.

    Paragraphs are defined as text blocks separated by one or more
    newline characters.

    Parameters
    ----------
    text : Any
        Input response text.

    Returns
    -------
    int
        Number of non-empty paragraphs.
    """

    # -------------------------
    # Input validation
    # -------------------------
    if _is_nan(text):
        return 0

    clean_text = str(text).strip()

    if not clean_text:
        return 0

    if clean_text.lower() == "nan":
        return 0

    # -------------------------
    # Paragraph segmentation
    # -------------------------
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SPLIT_RE.split(clean_text)
        if paragraph.strip()
    ]

    return int(len(paragraphs))


__all__ = ["num_paragraphs"]
