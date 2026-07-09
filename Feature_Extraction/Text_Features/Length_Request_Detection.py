"""
This module infers whether a user prompt explicitly requests a brief
or a detailed response, using lightweight regex heuristics over the
prompt text. Used to check whether a response's actual length (see
Writing_Style_Features.word_count) complies with what was asked for -
a verbose answer to a "briefly" request, or a terse answer to an
"explain in detail" request, are both format-compliance failures
independent of content correctness.
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
# Request Patterns
# ---------------------------------------------------------
_BRIEF_REQUEST_RE = re.compile(
    r"\b(?:briefly|in brief|be brief|concisely|be concise|"
    r"in a (?:few|couple of) (?:words|sentences)|"
    r"in one (?:word|sentence)|in a sentence|"
    r"short(?:ly)? answer|keep it short|keep (?:it|this) brief|"
    r"quick(?:ly)? (?:summary|summarize)|tl;?dr|"
    r"in \d+ words? or (?:less|fewer)|no more than \d+ words?|"
    r"without (?:going into|explaining) detail|"
    r"don'?t (?:go into|elaborate on) detail)\b",
    flags=re.IGNORECASE,
)

_DETAILED_REQUEST_RE = re.compile(
    r"\b(?:in detail|in-depth|in depth|thorough(?:ly)?|comprehensive(?:ly)?|"
    r"elaborate(?: on)?|go into detail|"
    r"explain (?:fully|thoroughly|in detail)|"
    r"detailed (?:explanation|answer|response|breakdown)|"
    r"(?:thorough|comprehensive|detailed) (?:\w+\s+){0,2}"
    r"(?:breakdown|explanation|analysis|overview|summary)|"
    r"step[- ]by[- ]step (?:explanation|guide|breakdown)|"
    r"as much detail as possible|extensively|"
    r"don'?t leave anything out|leave no stone unturned|"
    r"walk me through)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_length_request_features(text: Any) -> dict:
    """
    Detect whether a prompt explicitly requests a brief or a detailed
    response.

    Returns:
        {
            "requests_brief": bool,
            "requests_detailed": bool,
        }

    Note:
        A prompt could in principle match both patterns (e.g. "briefly
        explain in detail why..." is contradictory but not impossible
        in noisy real-world data); both flags are independent booleans
        rather than a mutually exclusive category.
    """

    clean = _clean_text(text)

    if not clean:
        return {
            "requests_brief": False,
            "requests_detailed": False,
        }

    return {
        "requests_brief": bool(_BRIEF_REQUEST_RE.search(clean)),
        "requests_detailed": bool(_DETAILED_REQUEST_RE.search(clean)),
    }


__all__ = ["extract_length_request_features"]
