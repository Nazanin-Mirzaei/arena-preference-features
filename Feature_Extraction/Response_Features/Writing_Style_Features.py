"""
This module extracts structural and linguistic features from text responses.
It is designed for LLM evaluation pipelines and supports:

- word_count
- sentence_count
- avg_words_per_sentence
- is_detailed (heuristic)
- has_step_by_step (reasoning / procedural structure detection)

The module is dependency-free, fast, and NaN-safe.
"""

from typing import Any, Dict
import re
import math


# ---------------------------------------------------------
# Helper: robust NaN checking
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ---------------------------------------------------------
# Main writing-style analysis function
# ---------------------------------------------------------
def extract_writing_style_features(text: Any) -> Dict[str, Any]:
    """
    Extract structural writing-style features from text.

    Returns:
        {
            "is_detailed": bool,
            "has_step_by_step": bool,
            "word_count": int,
            "sentence_count": int,
            "avg_words_per_sentence": float
        }
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

    # ---------------------------------------------
    # Token and sentence counting
    # ---------------------------------------------
    words = s.split()
    word_count = len(words)

    # Split by . ! ? + possible whitespace
    sentences = re.split(r"[.!?]+\s*", s)
    sentences = [seg.strip() for seg in sentences if seg.strip()]
    sentence_count = len(sentences) if sentences else 1

    avg_words_per_sentence = (
        word_count / sentence_count if sentence_count > 0 else 0.0
    )

    # ---------------------------------------------
    # Detect structured reasoning / step-by-step
    # ---------------------------------------------
    step_by_step_patterns = [
        # Explicit enumerations
        r"\bstep\s*\d+\b",
        r"\bpart\s*\d+\b",
        r"\bsection\s*\d+\b",
        r"\bphase\s*\d+\b",
        r"\bstage\s*\d+\b",

        # Ordinals for structured reasoning
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b",
        r"\bfifth\b", r"\binitially\b", r"\bsubsequently\b",

        # Procedural transitions
        r"\bnext\b", r"\bthen\b", r"\bafter that\b", r"\bfinally\b",
        r"\bthe following steps\b", r"\bthe process involves\b",

        # Logical / causal connectors
        r"\btherefore\b", r"\bthus\b", r"\bconsequently\b",
        r"\bas a result\b", r"\bthis implies\b", r"\bthis means\b",

        # Reasoning / explanation markers
        r"\blet us consider\b", r"\bwe can observe\b",
        r"\bwe analyze\b", r"\bthe reasoning is\b",
        r"\bto summarize\b", r"\bin conclusion\b",
        r"\bthe key idea is\b",

        # Chain-of-thought style markers
        r"\bto break this down\b",
        r"\bthe steps are as follows\b",
        r"\bwe proceed as follows\b",
    ]

    has_step_by_step = any(
        re.search(pattern, s.lower()) for pattern in step_by_step_patterns
    )

    # ---------------------------------------------
    # Heuristic: detailed response?
    # ---------------------------------------------
    is_detailed = (avg_words_per_sentence > 15) and (word_count > 50)

    return {
        "is_detailed": is_detailed,
        "has_step_by_step": has_step_by_step,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words_per_sentence,
    }


__all__ = ["extract_writing_style_features"]
