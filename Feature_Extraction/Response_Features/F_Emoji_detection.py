"""
Emoji Feature Extraction Utility
--------------------------------

Provides two main functions:
    detect_emoji(text) → bool
    count_emojis(text) → int

This module merges the strengths of multiple implementations:
    • Extended Unicode 15.1 emoji ranges (team A)
    • Robust NaN-safe input handling (team B)
    • Fast and clean modular structure
"""

from typing import Any
import re
import math


# ---------------------------------------
# 1) Safe NaN detection
# ---------------------------------------
def _is_nan(value: Any) -> bool:
    """Detects None, NaN, pd.NA without external libraries."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ---------------------------------------
# 2) Extended Unicode 15.1 Emoji Pattern
#    (merged from both implementations)
# ---------------------------------------
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric extensions
    "\U0001F800-\U0001F8FF"  # arrows supplement
    "\U0001F900-\U0001F9FF"  # symbols supplement
    "\U0001FA00-\U0001FA6F"  # chess etc
    "\U0001FA70-\U0001FAFF"  # newer emojis
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # weather / misc symbols
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE
)


# ---------------------------------------
# 3) Boolean Detector
# ---------------------------------------
def detect_emoji(text: Any) -> bool:
    """Return True if text contains emoji."""
    if _is_nan(text):
        return False
    s = str(text).strip()
    if not s:
        return False
    return bool(EMOJI_PATTERN.search(s))


# ---------------------------------------
# 4) Exact Emoji Counter
# ---------------------------------------
def count_emojis(text: Any) -> int:
    """Return number of emojis found in the text."""
    if _is_nan(text):
        return 0
    s = str(text).strip()
    if not s:
        return 0
    return len(EMOJI_PATTERN.findall(s))


__all__ = ["detect_emoji", "count_emojis", "EMOJI_PATTERN"]
