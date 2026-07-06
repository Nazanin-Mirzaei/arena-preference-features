"""
This module determines whether a given input is more likely to be natural
language (prose) or code/markup-like content using lightweight heuristics.

It combines lexical, structural, and ratio-based signals.
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
# Main Classifier: Likely Natural Text
# ---------------------------------------------------------
def is_natural_text(text: Any) -> bool:
    """
    Determine whether input is likely natural language rather than code.

    Returns:
        bool: True if text resembles prose, False if it resembles code/markup.
    """

    clean = _clean_text(text)

    # -------------------------
    # Edge case: empty / too short
    # -------------------------
    if not clean:
        return False

    word_re = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

    code_fence_re = re.compile(r"```|~~~")
    inline_code_re = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
    html_code_re = re.compile(r"</?(?:code|pre|script|style)\b", flags=re.IGNORECASE)

    code_hint_re = re.compile(
        r"(^|\n)\s*(?:def|class|import|from|for|while|if|elif|else|try|except|return|"
        r"function|const|let|var|public|private|SELECT|INSERT|UPDATE|DELETE)\b|"
        r"[{};]\s*$|</?[a-z][^>]*>",
        flags=re.IGNORECASE,
    )

    words = word_re.findall(clean)

    # too little linguistic content
    if len(words) < 3:
        return False

    # -------------------------
    # Code delimiter signals
    # -------------------------
    fenced_code_delimiters = len(code_fence_re.findall(clean))
    code_delimiters = (
        fenced_code_delimiters
        + len(inline_code_re.findall(clean))
        + len(html_code_re.findall(clean))
    )

    # -------------------------
    # Structural signals
    # -------------------------
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    code_like_lines = sum(bool(code_hint_re.search(line)) for line in lines)

    code_line_ratio = code_like_lines / max(len(lines), 1)

    alpha_chars = sum(ch.isalpha() for ch in clean)
    visible_chars = max(sum(not ch.isspace() for ch in clean), 1)

    alpha_ratio = alpha_chars / visible_chars

    # -------------------------
    # Decision rules
    # -------------------------
    if fenced_code_delimiters >= 2:
        return False

    if code_line_ratio >= 0.4 and len(lines) >= 2:
        return False

    return bool(alpha_ratio >= 0.45 and code_delimiters <= 2)

__all__ = ["is_natural_text"]
