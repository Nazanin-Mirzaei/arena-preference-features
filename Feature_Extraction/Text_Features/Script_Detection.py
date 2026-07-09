"""
This module classifies text by dominant Unicode script (writing system)
using lightweight character-range heuristics. It serves as a cheap,
dependency-free fallback for language comparison when full language
detection (e.g. MediaPipe) is unavailable, misfires, or returns "unknown" -
script mismatches are still a strong signal that a response answered in
the wrong language relative to the prompt.
"""

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

    if isinstance(text, float):
        import math
        if math.isnan(text):
            return ""

    value = str(text).strip()

    if value.lower() == "nan":
        return ""

    return value


# ---------------------------------------------------------
# Script Unicode Ranges
# ---------------------------------------------------------
_SCRIPT_PATTERNS = [
    ("arabic", re.compile(r'[؀-ۿݐ-ݿ]')),
    ("cjk", re.compile(r'[一-鿿぀-ヿ가-힯]')),
    ("cyrillic", re.compile(r'[Ѐ-ӿ]')),
    ("latin", re.compile(r'[A-Za-z]')),
]


# ---------------------------------------------------------
# Main Feature: Script Detector
# ---------------------------------------------------------
def detect_script(text: Any) -> str:
    """
    Detect the dominant Unicode script of a text using character-range
    heuristics.

    Returns one of: "arabic", "cjk", "cyrillic", "latin", "other".

    Detection order (first match wins): Arabic/Persian, CJK (Chinese/
    Japanese/Korean), Cyrillic, Latin. Non-Latin scripts are checked
    first since Latin punctuation/digits can appear alongside them.
    """

    clean = _clean_text(text)

    if not clean:
        return "other"

    for script_name, pattern in _SCRIPT_PATTERNS:
        if pattern.search(clean):
            return script_name

    return "other"


__all__ = ["detect_script"]
