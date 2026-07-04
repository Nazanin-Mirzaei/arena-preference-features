"""Sentence length dispersion feature."""

from __future__ import annotations

import math
import re
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def sd_sent_len(text: Any) -> float:
    """Return population standard deviation of sentence lengths in words."""
    if text is None:
        return 0.0
    if isinstance(text, float) and math.isnan(text):
        return 0.0

    clean_text = str(text).strip()
    if not clean_text:
        return 0.0

    if clean_text.lower() == "nan":
        return 0.0

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(clean_text) if part.strip()]
    lengths = [len(_WORD_RE.findall(sentence)) for sentence in sentences]
    if len(lengths) <= 1:
        return 0.0

    mean_value = sum(lengths) / len(lengths)
    variance = sum((value - mean_value) ** 2 for value in lengths) / len(lengths)
    return float(math.sqrt(variance))
