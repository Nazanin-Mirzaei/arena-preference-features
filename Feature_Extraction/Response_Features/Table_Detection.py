"""
Table detection utility for structured response analysis.

This module provides two features:
1) detect_tables(text)  → True/False  
2) count_tables(text)   → number of Markdown/HTML tables

Supported formats:
- Markdown tables using pipe syntax (| col1 | col2 | ...)
- HTML table tags (<table>, <tr>, <td>, <th>)

The module is optimized for large-scale LLM evaluation datasets.
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
# Markdown tables: at least 2 pipes on a line → | col1 | col2 | col3 |
_MARKDOWN_TABLE_PATTERN = re.compile(r"\|.*\|.*\|", re.MULTILINE)

# HTML tables: look for structural tags
_HTML_TABLE_PATTERN = re.compile(r"<table|<tr|<td|<th", re.IGNORECASE)


# ------------------------------------------
# Feature 1: detect if text contains a table
# ------------------------------------------
def detect_tables(text: Any) -> bool:
    """
    Returns True if text contains a Markdown or HTML-style table.
    """
    if _is_nan(text):
        return False

    s = str(text)
    if not s.strip():
        return False

    if _MARKDOWN_TABLE_PATTERN.search(s):
        return True
    if _HTML_TABLE_PATTERN.search(s):
        return True

    return False


# ------------------------------------------
# Feature 2: count number of tables
# ------------------------------------------
def count_tables(text: Any) -> int:
    """
    Count number of Markdown/HTML tables in the input text.
    """
    if not detect_tables(text):
        return 0

    s = str(text)

    md_tables = len(_MARKDOWN_TABLE_PATTERN.findall(s))
    html_tables = len(_HTML_TABLE_PATTERN.findall(s))

    return md_tables + html_tables


__all__ = ["detect_tables", "count_tables"]
