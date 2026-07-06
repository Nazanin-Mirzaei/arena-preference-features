"""
This Module provides:
    - detect_language: primary language detection
    - detect_multi_language: chunk-based multilingual analysis
    - detect_response_language: tuned for long LLM responses
    - detect_prompt_language: tuned for short prompts

Dependency:
    - langdetect (optional fallback to "unknown")
"""

from typing import Any, Dict, List
import re
import math
from collections import Counter

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# ---------------------------------------------------------
# Safe NaN check
# ---------------------------------------------------------
def _is_nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ---------------------------------------------------------
# Text cleaner (removes noise)
# ---------------------------------------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"<code>[\s\S]*?</code>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# Chunking utility
# ---------------------------------------------------------
def _get_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    i = 0

    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# Core language detection
# ---------------------------------------------------------
def detect_language(text: Any) -> str:
    if _is_nan(text):
        return "unknown"

    s = str(text).strip()
    if len(s) < 3:
        return "unknown"

    cleaned = _clean_text(s)
    if len(cleaned) < 3 or not LANGDETECT_AVAILABLE:
        return "unknown"

    try:
        langs = detect_langs(cleaned)
        if langs and langs[0].prob >= 0.5:
            return langs[0].lang
    except Exception:
        pass

    try:
        return detect(cleaned)
    except Exception:
        return "unknown"


# ---------------------------------------------------------
# Multi-language detection
# ---------------------------------------------------------
def detect_multi_language(
    text: Any,
    *,
    chunk_size: int = 200,
    overlap: int = 50
) -> Dict[str, Any]:

    if _is_nan(text):
        return _empty()

    s = str(text).strip()
    if len(s) < 10 or not LANGDETECT_AVAILABLE:
        return _empty()

    cleaned = _clean_text(s)
    chunks = _get_chunks(cleaned, chunk_size, overlap)

    languages = []
    for c in chunks:
        lang = detect_language(c)
        if lang != "unknown":
            languages.append(lang)

    if not languages:
        return _empty()

    counts = Counter(languages)
    total = len(languages)

    dist = {k: v / total for k, v in counts.items()}
    primary = max(dist, key=dist.get)

    significant = [k for k, v in dist.items() if v >= 0.15]

    return {
        "is_multi_language": len(significant) > 1,
        "languages": significant if len(significant) > 1 else [primary],
        "language_distribution": dist,
        "primary_language": primary,
        "secondary_languages": [k for k in significant if k != primary],
    }


# ---------------------------------------------------------
# Response / Prompt wrappers
# ---------------------------------------------------------
def detect_response_language(text: Any) -> Dict[str, Any]:
    return detect_multi_language(text, chunk_size=300, overlap=100)


def detect_prompt_language(text: Any) -> Dict[str, Any]:
    return detect_multi_language(text, chunk_size=50, overlap=15)


# ---------------------------------------------------------
# Empty fallback
# ---------------------------------------------------------
def _empty() -> Dict[str, Any]:
    return {
        "is_multi_language": False,
        "languages": [],
        "language_distribution": {},
        "primary_language": "unknown",
        "secondary_languages": []
    }


__all__ = [
    "detect_language",
    "detect_multi_language",
    "detect_response_language",
    "detect_prompt_language"
]
