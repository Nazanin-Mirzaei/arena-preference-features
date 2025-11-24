"""
Emoji detection utility for textual content analysis.

This module identifies the presence of Unicode emoji characters in text
using a comprehensive range of emoji code points defined in the Unicode
Standard. It is designed to detect visual or emotional cues in responses
that may influence human preference in LLM evaluations.

The pattern covers emoji from emoticons, symbols, transport, flags,
and enclosed characters as defined in Unicode 15.1. It does not attempt
to classify or interpret emoji meaning, only to detect their presence.

The function returns a boolean and is optimized for speed and reliability
in large-scale analysis.

Usage:
    has_emoji = detectEmojis(response_text)
"""

from typing import Any
import re
import math

def _is_nan(value: Any) -> bool:
    """
    Detects null or undefined values in a robust manner.
    Handles None, numpy.nan, and pandas.NA without requiring external libraries.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False

# Unicode emoji ranges as defined in Unicode 15.1 (covering emoji from U+1F600 to U+1F9FF,
# plus additional symbols, flags, and enclosed characters)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]",
    flags=re.UNICODE
)

def detectEmojis(text: Any) -> bool:
    """
    Determines whether a text string contains any Unicode emoji characters.
    
    Parameters:
        text: Input text of any type. Non-string inputs are converted to string.
    
    Returns:
        True if one or more emoji characters are detected, False otherwise.
        Returns False for null, empty, or non-string inputs.
    """
    if _is_nan(text):
        return False
    s = str(text)
    if not s.strip():
        return False
    return bool(_EMOJI_PATTERN.search(s))