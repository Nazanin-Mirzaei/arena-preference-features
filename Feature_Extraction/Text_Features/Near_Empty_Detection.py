"""
This module detects near-empty responses using lightweight length
heuristics. A response that is empty, whitespace-only, or reduced to a
handful of words is a strong quality/loser signal independent of
content correctness - it typically indicates a truncated, degenerate,
or failed generation rather than a legitimately terse answer.
"""

import math
import re
from typing import Any

DEFAULT_WORD_COUNT_THRESHOLD = 3
DEFAULT_CHAR_COUNT_THRESHOLD = 10

_ALNUM_RE = re.compile(r"\w", flags=re.UNICODE)


# ---------------------------------------------------------
# Text Cleaning Utility
# ---------------------------------------------------------
def _clean_text(text: Any) -> str:
    """
    Normalize input text into a clean string.
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
# Main Feature: Near-Empty Detector
# ---------------------------------------------------------
def is_near_empty(
    text: Any,
    word_count_threshold: int = DEFAULT_WORD_COUNT_THRESHOLD,
    char_count_threshold: int = DEFAULT_CHAR_COUNT_THRESHOLD,
) -> bool:
    """
    Detect whether a response is empty, whitespace-only, or too short to
    plausibly be a genuine answer.

    A response counts as near-empty if it is empty/whitespace-only, has
    no alphanumeric content at all (e.g. a long run of punctuation), OR
    if it falls at or below both the word-count and character-count
    thresholds. Requiring both count thresholds avoids misclassifying
    legitimately short but complete answers (e.g. a single long word,
    or a short numeric answer with several words) as near-empty.
    """

    clean = _clean_text(text)

    if not clean:
        return True

    if not _ALNUM_RE.search(clean):
        return True

    word_count = len(clean.split())
    char_count = len(clean)

    return word_count <= word_count_threshold and char_count <= char_count_threshold


__all__ = ["is_near_empty"]
