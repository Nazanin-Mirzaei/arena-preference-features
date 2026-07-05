"""
This module extracts structural signals related to code presence in text
by counting different types of code delimiters.

It captures:
    - Markdown code fences (``` and ~~~)
    - Inline backtick code
    - HTML code-related tags

The resulting feature reflects how "code-heavy" a response is.
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
        - whitespace trimming
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
# Main Feature: Code Delimiter Count
# ---------------------------------------------------------
def count_code_delimiters(text: Any) -> int:
    """
    Count occurrences of code-related delimiters in text.

    Includes:
        - Markdown fenced code blocks
        - Inline backtick code
        - HTML code-related tags
    """

    clean = _clean_text(text)

    # -------------------------
    # Edge case: empty input
    # -------------------------
    if not clean:
        return 0

    # -------------------------
    # Regex patterns
    # -------------------------
    code_fence_re = re.compile(r"```|~~~")
    inline_code_re = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
    html_code_re = re.compile(r"</?(?:code|pre|script|style)\b", flags=re.IGNORECASE)

    # -------------------------
    # Feature computation
    # -------------------------
    return int(
        len(code_fence_re.findall(clean))
        + len(inline_code_re.findall(clean))
        + len(html_code_re.findall(clean))
    )

__all__ = ["count_code_delimiters"]
