"""Basic newline-based paragraph features."""

from __future__ import annotations

import math
import re
from typing import Any


_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def num_paragraphs(text: Any) -> int:
    """Return the number of non-empty newline-separated paragraphs."""
    if text is None:
        return 0
    if isinstance(text, float) and math.isnan(text):
        return 0

    clean_text = str(text).strip()
    if not clean_text:
        return 0

    if clean_text.lower() == "nan":
        return 0

    paragraphs = [part.strip() for part in re.split(r"\n+", clean_text) if part.strip()]
    return int(len(paragraphs))
