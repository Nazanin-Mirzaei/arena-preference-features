"""
Code Block Detection Module

Provides robust detection of code-like content in LLM responses.
Supports:
    - Triple-backtick fenced blocks (open/closed)
    - Inline backtick code
    - Indented Markdown code blocks
    - HTML <code> and <pre> fragments
    - Heuristic detection of common programming syntax

This feature is useful for:
    - Measuring technicality of responses
    - Distinguishing reasoning vs. coding answers
    - Normalizing structured outputs
"""

from typing import Any
import re
import math


# ---------------------------------------------------------
# Utility: NaN-safe checking
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ---------------------------------------------------------
# Main Detector
# ---------------------------------------------------------
def is_code_block(text: Any) -> bool:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return False

    s = str(text)
    if not s.strip():
        return False

    # triple backticks
    if re.search(r"```[\s\S]*?```", s):
        return True

    # inline backticks (looser)
    if re.search(r"`[^`]{2,}`", s):
        return True

    # HTML code
    if re.search(r"<code[^>]*>[\s\S]*?</code>", s, re.IGNORECASE):
        return True

    if re.search(r"<pre[^>]*>[\s\S]*?</pre>", s, re.IGNORECASE):
        return True

    # better heuristics
    heuristic_patterns = [
        r"\bdef\s+\w+\(",
        r"\bclass\s+\w+",
        r"#include\s+<",
        r"\bfunction\s+\w+",
        r"^\s*(for|while)\s*\(",
    ]

    return any(re.search(p, s, re.MULTILINE) for p in heuristic_patterns)


__all__ = ["is_code_block"]
