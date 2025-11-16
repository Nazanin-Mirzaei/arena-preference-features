"""
Emoji detection utility.

This module provides `detectEmojis` to check if a text contains any emoji
characters using a broad Unicode regex range.
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


_EMOJI_PATTERN = re.compile(
	"["
	"\U0001F600-\U0001F64F"  # emoticons
	"\U0001F300-\U0001F5FF"  # symbols & pictographs
	"\U0001F680-\U0001F6FF"  # transport & map symbols
	"\U0001F1E0-\U0001F1FF"  # flags
	"\U00002702-\U000027B0"  # dingbats
	"\U000024C2-\U0001F251"  # enclosed characters
	"]",
	flags=re.UNICODE,
)


def detectEmojis(text: Any) -> bool:
	"""
	Return True if `text` contains any emoji, else False.
	"""
	if _is_nan(text):
		return False
	s = str(text)
	if not s:
		return False
	return bool(_EMOJI_PATTERN.search(s))


__all__ = ["detectEmojis"]


