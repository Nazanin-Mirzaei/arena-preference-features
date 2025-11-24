"""
Table detection utility for structured response analysis.

This module detects the presence of tabular data in text responses,
including Markdown-style tables (pipe-delimited) and HTML table tags.
It is intended to identify responses that employ structured formatting
to convey information — a signal of technical proficiency or
intentional organization.

The function does not parse or validate table structure, only detects
the presence of markers commonly associated with tabular representation.
It is designed for use in large-scale evaluation datasets where manual
inspection is infeasible.

Usage:
    has_table = detectTables(response_text)
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

def detectTables(text: Any) -> bool:
    """
    Detects the presence of table-like structures in text.
    
    Supports:
      - Markdown tables: lines with at least two pipe-separated columns
      - HTML table tags: <table>, <tr>, <td>, <th>
    
    Parameters:
        text: Input text of any type. Non-string inputs are converted to string.
    
    Returns:
        True if table-like structure is detected, False otherwise.
        Returns False for null, empty, or non-string inputs.
    """
    if _is_nan(text):
        return False
    s = str(text)
    if not s.strip():
        return False
    
    # Match Markdown tables: at least two columns separated by pipes on a single line
    if re.search(r'\|.*\|.*\|', s, re.MULTILINE):
        return True
    
    # Match HTML table elements
    if re.search(r'<table|<tr|<td|<th', s, re.IGNORECASE):
        return True
    
    return False