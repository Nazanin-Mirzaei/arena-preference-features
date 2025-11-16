"""
Table detection utility.

This module provides `detectTables` to detect table-like structures in text,
including Markdown tables and HTML tables.
"""

from typing import Any
import re
import math


def _is_nan(value: Any) -> bool:
	"""
	Best-effort NaN detection without heavy dependencies.
	"""
	if value is None:
		return True
	if isinstance(value, float):
		return math.isnan(value)
	return False


def detectTables(text: Any) -> bool:
	"""
	Detect table-like structures: Markdown and HTML.
	"""
	if _is_nan(text):
		return False
	s = str(text)
	if not s:
		return False
	# Markdown tables: at least three columns separated by pipes on a line
	if re.search(r"\|.*\|.*\|", s, re.MULTILINE):
		return True
	# HTML tables: presence of typical tags
	if re.search(r"<table|<tr|<td|<th", s, re.IGNORECASE):
		return True
	return False


__all__ = ["detectTables"]


