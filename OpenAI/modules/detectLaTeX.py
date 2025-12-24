"""
LaTeX and math expression detection utility.

This module provides `detectLaTeX`, which scans text for common LaTeX math
delimiters (inline and display), environment blocks, and math-focused macros.
It returns True when any LaTeX/math pattern is detected, otherwise False.
"""

from typing import Any, Dict, List, Set
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


# Precompiled regex patterns for performance and clarity.
_INLINE_MATH = re.compile(r"\$(?:\\.|[^\$\\])+\$", flags=re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$(?:\\.|[^\$\\])+\$\$", flags=re.DOTALL)
_BRACKET_DISPLAY = re.compile(r"\\\[(?:.|\n)*?\\\]", flags=re.DOTALL)
_BRACKET_INLINE = re.compile(r"\\\((?:.|\n)*?\\\)", flags=re.DOTALL)
_ENVIRONMENT = re.compile(r"\\begin\{([^\}]+)\}[\s\S]*?\\end\{\1\}", flags=re.DOTALL)

# Common math macros to catch math-like content even without explicit delimiters.
_MATH_MACROS = [
	"frac",
	"sum",
	"int",
	"sqrt",
	"alpha",
	"beta",
	"gamma",
	"theta",
	"lambda",
	"infty",
	"cdot",
	"times",
	"leq",
	"geq",
	"neq",
	"approx",
	"pm",
	"mp",
	"left",
	"right",
	"mathrm",
	"mathbb",
	"mathbf",
]
_MACRO_PATTERN = re.compile(r"\\(" + "|".join(_MATH_MACROS) + r")\b")


def detectLaTeX(text: Any) -> bool:
	"""
	Detect LaTeX/math patterns in text.
	Returns True if any pattern is found, else False.
	"""
	if _is_nan(text):
		return False

	s = str(text)
	if not s or len(s.strip()) < 2:
		return False

	inline_math = _INLINE_MATH.findall(s)
	display_math = _DISPLAY_MATH.findall(s)
	bracket_display = _BRACKET_DISPLAY.findall(s)
	bracket_inline = _BRACKET_INLINE.findall(s)
	environments = _ENVIRONMENT.findall(s)

	macros: Set[str] = set(match for match in _MACRO_PATTERN.findall(s))

	return any([
		inline_math,
		display_math,
		bracket_display,
		bracket_inline,
		environments,
		macros,
	])


__all__ = ["detectLaTeX"]



