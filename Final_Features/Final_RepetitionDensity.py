"""Lexical repetition density feature."""

from __future__ import annotations

import math
import re
from typing import Any


_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def repetition_density(text: Any) -> float:
    """Return inverse lexical diversity: repeated tokens divided by all tokens."""
    if text is None:
        return 0.0
    if isinstance(text, float) and math.isnan(text):
        return 0.0

    clean_text = str(text).strip()
    if not clean_text:
        return 0.0

    if clean_text.lower() == "nan":
        return 0.0

    words = [word.lower() for word in _WORD_RE.findall(clean_text)]
    if not words:
        return 0.0

    repeated_tokens = len(words) - len(set(words))
    return float(repeated_tokens / len(words))
