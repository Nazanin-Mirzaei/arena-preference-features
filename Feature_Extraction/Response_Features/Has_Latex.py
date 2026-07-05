"""LaTeX / Mathematical Notation Detection Module

This module detects whether a text contains LaTeX or mathematical expressions
using regex-based heuristics. It covers inline math, display math, and
common LaTeX commands/environments.
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
# Main Feature: LaTeX Detector
# ---------------------------------------------------------
def has_latex(text: Any) -> bool:
    """
    Detect whether text contains LaTeX or mathematical notation.

    Recognizes:
        - Inline math ($...$)
        - Display math ($$...$$)
        - LaTeX environments (\begin...\end)
        - Common math commands (\frac, \sum, etc.)
    """

    clean = _clean_text(text)

    # -------------------------
    # Edge case: empty input
    # -------------------------
    if not clean:
        return False

    # -------------------------
    # LaTeX detection pattern
    # -------------------------
    latex_re = re.compile(
        r"(\$\$.*?\$\$|\$[^$\n]+\$|\\\\\(.+?\\\\\)|\\\\\[.+?\\\\\]|"
        r"\\\\begin\{[a-zA-Z*]+\}.*?\\\\end\{[a-zA-Z*]+\}|"
        r"\\\\(?:frac|sum|int|sqrt|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|infty)\b)",
        flags=re.DOTALL,
    )

    # -------------------------
    # Feature output
    # -------------------------
    return bool(latex_re.search(clean))

__all__ = ["has_latex"]
