"""
List detection utility for structured response analysis.

Detects Markdown-style ordered and unordered list items via
line-prefix heuristics, independent of the dataset-provided
conv_metadata list counts (see Metadata_Features/Lists.py), so it can
be applied directly to raw prompt/response text such as user_prompt,
where no metadata list counts exist.
"""

from typing import Any
import re
import math


# -------------------------------
# Helper: robust NaN detection
# -------------------------------
def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ------------------------------------------
# Regex patterns
# ------------------------------------------
# Unordered: line starts with -, *, or + followed by a space
_UNORDERED_ITEM_PATTERN = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)

# Ordered: line starts with a number followed by . or ) and a space
_ORDERED_ITEM_PATTERN = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)


# ------------------------------------------
# Feature 1: detect if text contains a list
# ------------------------------------------
def detect_list(text: Any) -> bool:
    """
    Returns True if text contains a Markdown-style ordered or
    unordered list (at least one list item line).
    """
    if _is_nan(text):
        return False

    s = str(text)
    if not s.strip():
        return False

    if _UNORDERED_ITEM_PATTERN.search(s):
        return True
    if _ORDERED_ITEM_PATTERN.search(s):
        return True

    return False


# ------------------------------------------
# Feature 2: count number of list items
# ------------------------------------------
def count_list_items(text: Any) -> int:
    """
    Count total number of Markdown-style list item lines
    (ordered + unordered).
    """
    if _is_nan(text):
        return 0

    s = str(text)
    if not s.strip():
        return 0

    unordered = len(_UNORDERED_ITEM_PATTERN.findall(s))
    ordered = len(_ORDERED_ITEM_PATTERN.findall(s))

    return unordered + ordered


__all__ = ["detect_list", "count_list_items"]
