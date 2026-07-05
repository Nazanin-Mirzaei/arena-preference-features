"""
Long and Short Sentence Detection Module

Provides lightweight detection of sentence-length characteristics in
LLM responses using heuristic rules.

Supports:
    - Regex-based sentence segmentation
    - Unicode-aware word counting
    - Long sentence counting
    - Short sentence counting
    - Robust handling of None / NaN / empty inputs

This feature is useful for:
    - Measuring writing style
    - Comparing response verbosity
    - Quantifying structural characteristics of LLM outputs
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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

# Sentence-length thresholds (measured in words)
_SHORT_SENTENCE_MAX_WORDS = 10
_LONG_SENTENCE_MIN_WORDS = 30


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def long_short_sentences(text: Any) -> dict[str, int]:
    """
    Count long and short sentences in a text response.

    A short sentence contains at most `_SHORT_SENTENCE_MAX_WORDS`
    words, while a long sentence contains at least
    `_LONG_SENTENCE_MIN_WORDS` words.

    Parameters
    ----------
    text : Any
        Input response text.

    Returns
    -------
    dict[str, int]
        Dictionary containing:

        - long_sents
        - short_sents
    """

    # -------------------------
    # Input validation
    # -------------------------
    if _is_nan(text):
        return {"long_sents": 0, "short_sents": 0}

    clean_text = str(text).strip()

    if not clean_text:
        return {"long_sents": 0, "short_sents": 0}

    if clean_text.lower() == "nan":
        return {"long_sents": 0, "short_sents": 0}

    # -------------------------
    # Sentence segmentation
    # -------------------------
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(clean_text)
        if sentence.strip()
    ]

    # -------------------------
    # Word counting
    # -------------------------
    sentence_lengths = [
        len(_WORD_RE.findall(sentence))
        for sentence in sentences
    ]

    # -------------------------
    # Long sentence count
    # -------------------------
    long_sents = sum(
        length >= _LONG_SENTENCE_MIN_WORDS
        for length in sentence_lengths
    )

    # -------------------------
    # Short sentence count
    # -------------------------
    short_sents = sum(
        0 < length <= _SHORT_SENTENCE_MAX_WORDS
        for length in sentence_lengths
    )

    return {
        "long_sents": int(long_sents),
        "short_sents": int(short_sents),
    }


__all__ = ["long_short_sentences"]