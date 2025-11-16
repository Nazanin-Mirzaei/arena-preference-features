"""
Approximate token counting utility.

This module provides a single function `countTokens` that estimates the number
of tokens in a piece of text using a simple heuristic: ~4 characters per token.
This is fast, dependency-free, and sufficient for coarse analysis where exact
tokenization is not required.
"""

from typing import Any
import math

AVERAGE_CHARS_PER_TOKEN: int = 4


def _is_nan(value: Any) -> bool:
	"""
	Best-effort NaN detection without introducing heavy dependencies.
	Handles None, float('nan'), and numpy/pandas NaN where possible.
	"""
	if value is None:
		return True
	# Float NaN check (covers numpy.nan as well because it's a float)
	if isinstance(value, float):
		return math.isnan(value)
	return False


def countTokens(text: Any, average_chars_per_token: int = AVERAGE_CHARS_PER_TOKEN) -> int:
	"""
	Estimate token count using a characters-per-token heuristic.

	Parameters:
	- text: Any
	  The input to measure. Non-string inputs are converted with str().
	- average_chars_per_token: int
	  Heuristic divisor; default is 4 chars per token.

	Returns:
	- int: Estimated token count (floor(len(str(text)) / average_chars_per_token)).
	"""
	if _is_nan(text):
		return 0
	s = str(text)
	if not s:
		return 0
	if average_chars_per_token <= 0:
		raise ValueError("average_chars_per_token must be > 0")
	return math.floor(len(s) / average_chars_per_token)


__all__ = ["countTokens", "AVERAGE_CHARS_PER_TOKEN"]


