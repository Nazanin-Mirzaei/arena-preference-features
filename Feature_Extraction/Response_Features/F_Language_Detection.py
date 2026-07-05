"""
Robust Language Detection Module

Provides:
    - detectLanguage(text): primary language detection
    - detectMultiLanguage(text): chunk-based multilingual analysis
    - detectResponseLanguage(text): tuned for long LLM responses
    - detectPromptLanguage(text): tuned for short user prompts

Dependencies:
    Requires `langdetect` for full functionality.
    Falls back to "unknown" when unavailable.
"""

from typing import Any, Dict, List
import re
import math
from collections import Counter

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# -------------------------------------------
# Utility: Safe NaN checking
# -------------------------------------------
def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# -------------------------------------------
# Utility: Clean text for better detection
# -------------------------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"<code>[\s\S]*?</code>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------------------
# Utility: Split into overlapping chunks
# -------------------------------------------
def _get_text_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


# -------------------------------------------
# Primary language detection
# -------------------------------------------
def detectLanguage(text: Any) -> str:
    if _is_nan(text):
        return "unknown"

    s = str(text).strip()
    if len(s) < 3:
        return "unknown"

    cleaned = _clean_text(s)
    if len(cleaned) < 3:
        return "unknown"

    if not LANGDETECT_AVAILABLE:
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


# -------------------------------------------
# Multi-language detection
# -------------------------------------------
def detectMultiLanguage(
    text: Any,
    *,
    chunk_size: int = 200,
    overlap: int = 50
) -> Dict[str, Any]:

    if _is_nan(text):
        return _empty_multilang()

    s = str(text).strip()
    if len(s) < 10:
        return _empty_multilang()

    cleaned = _clean_text(s)
    if len(cleaned) < 10:
        return _empty_multilang()

    if not LANGDETECT_AVAILABLE:
        return _empty_multilang()

    chunks = _get_text_chunks(cleaned, chunk_size=chunk_size, overlap=overlap)
    chunk_languages = [detectLanguage(chunk) for chunk in chunks if detectLanguage(chunk) != "unknown"]

    if not chunk_languages:
        return _empty_multilang()

    counts = Counter(chunk_languages)
    total = len(chunk_languages)

    dist = {lang: c / total for lang, c in counts.items()}
    significant = [lang for lang, r in dist.items() if r >= 0.15]

    primary = max(dist, key=dist.get)
    secondary = [lang for lang in significant if lang != primary]

    return {
        "is_multi_language": len(significant) > 1,
        "languages": significant if len(significant) > 1 else [primary],
        "language_distribution": dist,
        "primary_language": primary,
        "secondary_languages": secondary,
    }


# Utility helper
def _empty_multilang():
    return {
        "is_multi_language": False,
        "languages": [],
        "language_distribution": {},
        "primary_language": "unknown",
        "secondary_languages": []
    }


# -------------------------------------------
# Convenience wrappers
# -------------------------------------------
def detectResponseLanguage(text: Any) -> Dict[str, Any]:
    return detectMultiLanguage(text, chunk_size=300, overlap=100)


def detectPromptLanguage(text: Any) -> Dict[str, Any]:
    return detectMultiLanguage(text, chunk_size=50, overlap=15)


__all__ = [
    "detectLanguage",
    "detectMultiLanguage",
    "detectResponseLanguage",
    "detectPromptLanguage"
]
