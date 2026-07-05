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
def detectCodeBlock(text: Any) -> bool:
    """
    Detects presence of code-like blocks in a text response.
    Covers Markdown, HTML, and heuristic code patterns.
    """
    if _is_nan(text):
        return False

    s = str(text)
    if not s.strip():
        return False

    # -------------------------
    # 1. Triple-backtick blocks
    # -------------------------
    # Closed block
    if re.search(r"```[\s\S]*?```", s, re.IGNORECASE):
        return True

    # Inline triple-backtick
    if re.search(r"```[^`]+```", s, re.IGNORECASE):
        return True

    # Unclosed block (common in LLM output)
    if re.search(r"```[\s\S]*", s, re.IGNORECASE):
        return True

    # -------------------------
    # 2. Inline backtick code
    # -------------------------
    if re.search(r'`[A-Za-z0-9_=\-\+\*/\(\)\{\};\[\]]+`', s):
        return True

    # -------------------------
    # 3. Markdown-indented code
    # -------------------------
    if re.search(r"^( {4}|\t).+", s, re.MULTILINE):
        return True

    # -------------------------
    # 4. HTML code tags
    # -------------------------
    if re.search(r"<code[^>]*>[\s\S]*?</code>", s, re.IGNORECASE):
        return True

    if re.search(r"<pre[^>]*>[\s\S]*?</pre>", s, re.IGNORECASE):
        return True

    # -------------------------
    # 5. Heuristic patterns
    # -------------------------
    heuristic_patterns = [
        r"\bdef\s+\w+\(",       # Python
        r"\bclass\s+\w+",       # class
        r"#include\s+<",        # C/C++
        r"\bfunction\s+\w+",    # JS
        r"\bfor\s*\(",          # loops
        r"\bwhile\s*\(",        # loops
        r"=\s*[^=]",            # assignment
        r";\s*$",               # semicolon-at-end lines
    ]

    for pat in heuristic_patterns:
        if re.search(pat, s, re.MULTILINE):
            return True

    return False


__all__ = ["detectCodeBlock"]
