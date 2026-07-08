"""
This module extracts interaction-related behavioral signals directly from raw text.
It detects conversational engagement patterns such as conclusions, next-step suggestions,
interaction prompts, and end-of-question usage.

All features are computed using lightweight regex-based heuristics.
"""

import math
import re
from typing import Any


# ---------------------------------------------------------
# Text Cleaning Utility
# ---------------------------------------------------------
def _clean_text(text: Any) -> str:
    """
    Normalize input text into a clean string.

    Handles:
        - None values
        - NaN floats
        - string "nan"
        - leading/trailing whitespace
    """

    if text is None:
        return ""

    if isinstance(text, float) and math.isnan(text):
        return ""

    value = str(text).strip()

    if value.lower() == "nan":
        return ""

    return value


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_interaction_features(text: Any) -> dict:
    """
    Extract interaction behavior features from a text response.

    Returns a dictionary of binary features and a composite interaction score.
    """

    clean = _clean_text(text)

    # -------------------------
    # Edge case: empty input
    # -------------------------
    if not clean:
        return {
            "has_question_at_end": False,
            "has_conclusion": False,
            "has_next_steps": False,
            "has_interaction_prompt": False,
            "interaction_score": 0,
        }

    # -------------------------
    # Regex patterns
    # -------------------------
    conclusion_re = re.compile(
        r"\b(?:in conclusion|to conclude|overall|therefore|thus|in summary|"
        r"to summarize|finally|bottom line)\b",
        flags=re.IGNORECASE,
    )

    next_steps_re = re.compile(
        r"\b(?:next steps?|following steps?|follow-up steps?|you can now|from here|"
        r"going forward|recommended steps?|action items?)\b",
        flags=re.IGNORECASE,
    )

    interaction_prompt_re = re.compile(
        r"\b(?:let me know|tell me|would you like|do you want|if you want|"
        r"feel free to|share|ask me|can you provide|please provide)\b",
        flags=re.IGNORECASE,
    )

    # -------------------------
    # Feature extraction
    # -------------------------
    has_question = bool(re.search(r"\?\s*$", clean.rstrip()))
    has_conclusion = bool(conclusion_re.search(clean))
    has_next_steps = bool(next_steps_re.search(clean))
    has_interaction_prompt = bool(interaction_prompt_re.search(clean))

    # -------------------------
    # Feature aggregation
    # -------------------------
    return {
        "has_question_at_end": has_question,
        "has_conclusion": has_conclusion,
        "has_next_steps": has_next_steps,
        "has_interaction_prompt": has_interaction_prompt,
        "interaction_score": (
            int(has_question)
            + int(has_conclusion)
            + int(has_next_steps)
            + int(has_interaction_prompt)
        ),
    }

__all__ = ["extract_interaction_features"]
