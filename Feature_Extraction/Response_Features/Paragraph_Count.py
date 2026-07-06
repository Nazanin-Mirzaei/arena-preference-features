"""
This Module provides lightweight paragraph counting for LLM responses using
newline-based segmentation.

Supports:
    - Paragraph detection based on double newlines
    - Removal of empty paragraphs
    - Robust handling of None / NaN / empty inputs

This feature is useful for:
    - Measuring document structure
    - Quantifying response organization
    - Comparing formatting styles across model outputs
"""

from typing import Any
import math
import re


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def extract_paragraph_features(text: Any) -> dict:
    """
    Count the number of non-empty paragraphs in a text response.
    """

    if _is_nan(text):
        return {"paragraph_count": 0}

    clean_text = str(text).strip()

    if not clean_text or clean_text.lower() == "nan":
        return {"paragraph_count": 0}

    paragraphs = [
        p.strip()
        for p in _PARAGRAPH_SPLIT_RE.split(clean_text)
        if p.strip()
    ]

    return {
        "paragraph_count": len(paragraphs)
    }


__all__ = ["extract_paragraph_features"]
