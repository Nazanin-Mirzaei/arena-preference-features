"""
Structural / Format-Based Text Features

This module provides lightweight, dependency-free structural features
for analyzing LLM-generated responses. These features do not interpret
meaning or writing style; instead, they focus purely on the surface form
of the text.

Included features:
    - Markdown code block count
    - Regex-based sentence count
    - Punctuation count
"""

from typing import Any
import re
import string
import math


# ---------------------------------------------------------
# Utility: Robust NaN Detection
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    """
    Lightweight NaN detector without external dependencies.
    Supports: None, float('nan'), numpy.nan (if passed as float).
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ---------------------------------------------------------
# 1. Markdown Code Block Counter
# ---------------------------------------------------------
def count_code_blocks(text: Any) -> int:
    """
    Count the number of Markdown triple-backtick code block delimiters.

    Example of detected pattern:
        ```python
        print("hi")
        ```

    Parameters:
        text: Any input (converted to string)

    Returns:
        int — number of code block delimiters found
    """
    if _is_nan(text):
        return 0
    s = str(text)
    return s.count("```")


# ---------------------------------------------------------
# 2. Regex-Based Sentence Counter
# ---------------------------------------------------------
def count_sentences(text: Any) -> int:
    """
    Count sentences based on punctuation marks (., !, ?).

    This is a lightweight alternative to nltk.sent_tokenize and avoids
    external dependencies or downloads.

    Parameters:
        text: Input text

    Returns:
        int — number of detected sentence-ending punctuation groups
    """
    if _is_nan(text):
        return 0
    s = str(text).strip()
    if not s:
        return 0

    matches = re.findall(r"[.!?]+", s)
    return len(matches)


# ---------------------------------------------------------
# 3. Punctuation Count
# ---------------------------------------------------------
def count_punctuation(text: Any) -> int:
    """
    Count punctuation characters using the built-in string.punctuation list.

    Includes characters:
        !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    Parameters:
        text: Input text

    Returns:
        int — total number of punctuation characters
    """
    if _is_nan(text):
        return 0
    s = str(text)

    return sum(1 for c in s if c in string.punctuation)


__all__ = [
    "count_code_blocks",
    "count_sentences",
    "count_punctuation",
]
