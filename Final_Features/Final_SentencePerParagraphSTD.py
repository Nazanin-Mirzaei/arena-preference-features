"""Sentence-per-paragraph dispersion feature."""

from __future__ import annotations

import math
import re
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sd_sent_per_paragraph(text: Any) -> float:
    """Return population standard deviation of sentence counts per paragraph."""
    if text is None:
        return 0.0
    if isinstance(text, float) and math.isnan(text):
        return 0.0

    clean_text = str(text).strip()
    if not clean_text:
        return 0.0

    if clean_text.lower() == "nan":
        return 0.0

    paragraphs = [part.strip() for part in re.split(r"\n+", clean_text) if part.strip()]
    counts = []
    for paragraph in paragraphs:
        sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(paragraph) if part.strip()]
        counts.append(len(sentences))

    if len(counts) <= 1:
        return 0.0

    mean_value = sum(counts) / len(counts)
    variance = sum((value - mean_value) ** 2 for value in counts) / len(counts)
    return float(math.sqrt(variance))
