"""
This Module provides lightweight sentence-length dispersion statistics for
LLM responses using heuristic sentence segmentation.

Supports:
    - Regex-based sentence segmentation
    - Unicode-aware word counting
    - Population standard deviation of sentence lengths
    - Robust handling of None / NaN / empty inputs

This feature is useful for:
    - Measuring writing consistency
    - Quantifying sentence-length variation
    - Comparing structural characteristics of LLM outputs
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


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def compute_sentence_length_std(text: Any) -> float:
    """
    Compute the population standard deviation of sentence lengths.

    Sentence length is measured as the number of words in each
    sentence.

    Parameters
    ----------
    text : Any
        Input response text.

    Returns
    -------
    float
        Population standard deviation of sentence lengths.
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
    # Sentence Segmentation
    # -------------------------
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(clean_text)
        if sentence.strip()
    ]

    # -------------------------
    # Sentence Length Calculation
    # -------------------------
    sentence_lengths = [
        len(_WORD_RE.findall(sentence))
        for sentence in sentences
    ]

    if len(sentence_lengths) <= 1:
        return 0.0

    # -------------------------
    # Statistics
    # -------------------------
    mean_value = sum(sentence_lengths) / len(sentence_lengths)

    variance = sum(
        (length - mean_value) ** 2
        for length in sentence_lengths
    ) / len(sentence_lengths)

    return {
        "sentence_length_std": float(math.sqrt(variance))
    }


__all__ = ["compute_sentence_length_std"]
