"""Paragraph length statistics based on newline-separated paragraphs."""

from __future__ import annotations

import math
import re
from typing import Any


_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def paragraph_length_stats(text: Any) -> dict:
    """Return mean and population standard deviation of paragraph word lengths."""
    if text is None:
        return {"mean_paragraph_length": 0.0, "sd_paragraph_length": 0.0}
    if isinstance(text, float) and math.isnan(text):
        return {"mean_paragraph_length": 0.0, "sd_paragraph_length": 0.0}

    clean_text = str(text).strip()
    if not clean_text:
        return {"mean_paragraph_length": 0.0, "sd_paragraph_length": 0.0}

    if clean_text.lower() == "nan":
        return {"mean_paragraph_length": 0.0, "sd_paragraph_length": 0.0}

    paragraphs = [part.strip() for part in re.split(r"\n+", clean_text) if part.strip()]
    lengths = [len(_WORD_RE.findall(paragraph)) for paragraph in paragraphs]
    if not lengths:
        return {"mean_paragraph_length": 0.0, "sd_paragraph_length": 0.0}

    mean_value = sum(lengths) / len(lengths)
    variance = sum((value - mean_value) ** 2 for value in lengths) / len(lengths)
    return {
        "mean_paragraph_length": float(mean_value),
        "sd_paragraph_length": float(math.sqrt(variance)) if len(lengths) > 1 else 0.0,
    }
