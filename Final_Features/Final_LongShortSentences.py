"""Heuristic long and short sentence count features."""

from __future__ import annotations

import math
import re
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_SHORT_SENTENCE_MAX_WORDS = 5
_LONG_SENTENCE_MIN_WORDS = 25


def long_short_sentences(text: Any) -> dict:
    """Return counts of long and short sentences in text."""
    if text is None:
        return {"long_sents": 0, "short_sents": 0}
    if isinstance(text, float) and math.isnan(text):
        return {"long_sents": 0, "short_sents": 0}

    clean_text = str(text).strip()
    if not clean_text:
        return {"long_sents": 0, "short_sents": 0}

    if clean_text.lower() == "nan":
        return {"long_sents": 0, "short_sents": 0}

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(clean_text) if part.strip()]
    lengths = [len(_WORD_RE.findall(sentence)) for sentence in sentences]
    long_sents = sum(length >= _LONG_SENTENCE_MIN_WORDS for length in lengths)
    short_sents = sum(0 < length <= _SHORT_SENTENCE_MAX_WORDS for length in lengths)
    return {"long_sents": int(long_sents), "short_sents": int(short_sents)}
