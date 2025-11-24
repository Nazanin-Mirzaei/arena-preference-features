"""
Enhanced code-block detection utility for LLM response analysis.

This version expands support for:
  - Multiline and single-line triple-backtick blocks
  - Unclosed code blocks (common in LLM output)
  - Markdown-indented code (4 spaces or tabs)
  - Generic HTML code/pre tags
  - Heuristic detection of code-like syntax in unformatted text

The implementation remains dependency-free and optimized for
robust detection across diverse LLM answer styles.
"""

from typing import Any
import re
import math


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def detectCodeBlock(text: Any) -> bool:
    if _is_nan(text):
        return False

    s = str(text)
    if not s.strip():
        return False

    # 1. Flexible triple-backtick detection

    # Multiline: ```lang?\n content \n```
    if re.search(r'```[\s\S]*?```', s, re.IGNORECASE):
        return True

    # Single-line: ```code```
    if re.search(r'```[^`]+```', s, re.IGNORECASE):
        return True

    # Unclosed block: ```python\n... (no closing backticks)
    if re.search(r'```[\s\S]*', s, re.IGNORECASE):
        return True

    # 2. Inline backtick code (more conservative, word-like)

    # Require alphanumeric or symbols typical of code inside the backticks
    if re.search(r'`[A-Za-z0-9_=\-\+\*/\(\)\{\};\[\]]+`', s):
        return True

    # 3. Markdown-indented code blocks (4 spaces or tabs)
    if re.search(r'^( {4}|\t).+', s, re.MULTILINE):
        return True

    # 4. HTML code fragments
    if re.search(r'<code[^>]*>[\s\S]*?</code>', s, re.IGNORECASE):
        return True

    if re.search(r'<pre[^>]*>[\s\S]*?</pre>', s, re.IGNORECASE):
        return True

    # 5. Heuristic code syntax detection
    heuristic_patterns = [
        r'\bdef\s+\w+\(',          # Python function
        r'\bclass\s+\w+',          # Class definitions
        r'{\s*}',                  # Empty braces
        r'=\s*[^=]',               # Assignment
        r';\s*$',                  # Semicolon line ends
        r'\bfunction\s+\w+',       # JS function
        r'#include\s+<',           # C include
        r'\bfor\s*\(',             # Loops
        r'\bwhile\s*\(',           # Loops
    ]

    for pat in heuristic_patterns:
        if re.search(pat, s, re.MULTILINE):
            return True

    return False
