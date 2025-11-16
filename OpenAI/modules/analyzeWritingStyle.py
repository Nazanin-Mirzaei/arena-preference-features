"""
Writing style analysis utility.

This module provides `analyzeWritingStyle`, which computes simple text metrics:
- word_count
- sentence_count
- avg_words_per_sentence
- is_detailed (heuristic: avg_words_per_sentence > 15 and word_count > 50)
"""

from typing import Any, Dict
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


def analyzeWritingStyle(text: Any) -> Dict[str, float]:
	"""
	Analyze writing style and return metrics.
	"""
	if _is_nan(text):
		return {
			"is_detailed": False,
			"word_count": 0,
			"sentence_count": 0,
			"avg_words_per_sentence": 0.0,
		}
	s = str(text)
	if not s:
		return {
			"is_detailed": False,
			"word_count": 0,
			"sentence_count": 0,
			"avg_words_per_sentence": 0.0,
		}
	words = s.split()
	sentences = re.split(r"[.!?]+\s+", s)
	sentences = [seg for seg in sentences if seg.strip()]
	word_count = len(words)
	sentence_count = len(sentences) if sentences else 1
	avg_wps = (word_count / sentence_count) if sentence_count > 0 else 0.0
	is_detailed = (avg_wps > 15) and (word_count > 50)
	return {
		"is_detailed": is_detailed,
		"word_count": word_count,
		"sentence_count": sentence_count,
		"avg_words_per_sentence": avg_wps,
	}


__all__ = ["analyzeWritingStyle"]


