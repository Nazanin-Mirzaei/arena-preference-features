"""
Writing style analysis module for evaluating structural and linguistic
characteristics of LLM-generated responses.

This module provides lightweight, dependency-free tools for computing
textual features relevant to model evaluation pipelines. The focus is on
capturing stylistic attributes such as elaboration, reasoning format,
and sentence-level structure without performing semantic interpretation.

The extracted features are suitable for downstream statistical analysis,
comparison of model outputs, or meta-evaluation frameworks.
"""

from typing import Any, Dict
import re
import math


def _is_nan(value: Any) -> bool:
    """
    Robustly identifies null-like values. Supports None, numpy.nan,
    pandas.NA, and other float-based missing-value representations,
    without requiring external libraries.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def analyzeWritingStyle(text: Any) -> Dict[str, Any]:
    """
    Computes structural writing-style features from a text response.

    Extracted metrics:
        - is_detailed:
            Indicates whether the response exceeds both length and
            complexity thresholds (heuristic-based).
        - has_step_by_step:
            Indicates whether the text contains explicit reasoning or
            procedural markers often used in chain-of-thought or
            structured explanations.
        - word_count:
            Total number of space-delimited tokens.
        - sentence_count:
            Number of sentences determined via punctuation segmentation.
        - avg_words_per_sentence:
            Mean number of words per sentence.

    Parameters
    ----------
    text : Any
        Input text. Non-string values are coerced to string.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed linguistic metrics. Returns default
        values for invalid or empty input.
    """
    if _is_nan(text):
        return {
            "is_detailed": False,
            "has_step_by_step": False,
            "word_count": 0,
            "sentence_count": 0,
            "avg_words_per_sentence": 0.0,
        }

    s = str(text)
    if not s.strip():
        return {
            "is_detailed": False,
            "has_step_by_step": False,
            "word_count": 0,
            "sentence_count": 0,
            "avg_words_per_sentence": 0.0,
        }

    # Tokenization
    words = s.split()
    word_count = len(words)

    # Sentence segmentation by common punctuation markers
    sentences = re.split(r'[.!?]+\s*', s)
    sentences = [seg.strip() for seg in sentences if seg.strip()]
    sentence_count = len(sentences) if sentences else 1

    avg_words_per_sentence = (
        word_count / sentence_count if sentence_count > 0 else 0.0
    )

    # Expanded set of reasoning-pattern indicators.
    # Covers explicit enumeration, logical transitions, procedural signals,
    # and formulation markers frequently used in structured LLM reasoning.
    step_patterns = [
        # Explicit enumerations
        r'\bstep\s*\d+\b',
        r'\bpart\s*\d+\b',
        r'\bsection\s*\d+\b',
        r'\bphase\s*\d+\b',
        r'\bstage\s*\d+\b',

        # Ordinals commonly used in structured explanations
        r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfourth\b',
        r'\bfifth\b', r'\binitially\b', r'\bsubsequently\b',

        # Procedural transitions
        r'\bnext\b', r'\bthen\b', r'\bafter that\b', r'\bfinally\b',
        r'\bthe following steps\b', r'\bthe process involves\b',

        # Logical or causal connectors
        r'\btherefore\b', r'\bthus\b', r'\bconsequently\b',
        r'\bas a result\b', r'\bthis implies\b', r'\bthis means\b',

        # Explanatory structuring signals
        r'\blet us consider\b', r'\bwe can observe\b',
        r'\bwe can see\b', r'\bto summarize\b', r'\bin conclusion\b',
        r'\bthe key idea is\b', r'\bthe reasoning is\b',

        # “Reasoning mode” markers characteristic of chain-of-thought
        r'\bto break this down\b', r'\bthe steps are as follows\b',
        r'\bwe proceed as follows\b', r'\bthe approach is\b',
        r'\banalysis shows\b', r'\bwe analyze\b',
    ]

    has_step_by_step = any(
        re.search(pattern, s.lower()) for pattern in step_patterns
    )

    # Heuristic for detailed responses:
    # Requires both a substantial total length and above-average sentence complexity.
    is_detailed = (avg_words_per_sentence > 15) and (word_count > 50)

    return {
        "is_detailed": is_detailed,
        "has_step_by_step": has_step_by_step,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words_per_sentence,
    }
