"""
Robust language detection utility.

This module provides `detectLanguage` to detect the primary language of a text,
`detectMultiLanguage` with configurable chunking controls, and convenience
wrappers (`detectResponseLanguage` / `detectPromptLanguage`) that bake in
recommended defaults for long responses vs. short prompts.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import math
from collections import Counter

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    # Fallback: basic heuristics if langdetect is not available
    pass


def _is_nan(value: Any) -> bool:
    """
    Best-effort NaN detection without heavy dependencies.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _clean_text(text: str) -> str:
    """
    Clean text for better language detection by removing code blocks,
    URLs, and excessive whitespace.
    """
    # Remove code blocks (markdown and HTML)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'<code>[\s\S]*?</code>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _get_text_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for multi-language detection.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks


def detectLanguage(text: Any) -> str:
    """
    Detect the primary language of a text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Primary language code (e.g., 'en', 'zh', 'fr') or 'unknown' if detection fails
    """
    if _is_nan(text):
        return 'unknown'
    
    s = str(text)
    if not s or len(s.strip()) < 3:
        return 'unknown'
    
    # Clean the text
    cleaned = _clean_text(s)
    
    if not cleaned or len(cleaned.strip()) < 3:
        return 'unknown'
    
    if LANGDETECT_AVAILABLE:
        try:
            # Try to detect with confidence scores
            langs = detect_langs(cleaned)
            if langs:
                primary = langs[0]
                # Use minimum confidence of 0.5 for reliability
                if primary.prob >= 0.5:
                    return primary.lang
        except LangDetectException:
            pass
        except Exception:
            # Fallback to simple detect if detect_langs fails
            try:
                return detect(cleaned)
            except Exception:
                pass
    
    # Fallback: return unknown
    return 'unknown'


def detectMultiLanguage(
    text: Any,
    *,
    chunk_size: int = 200,
    overlap: int = 50
) -> Dict[str, Any]:
    """
    Detect if a text contains multiple languages.
    
    Args:
        text: Input text to analyze
        chunk_size: Number of words per chunk for multi-language detection
        overlap: Word overlap between consecutive chunks
        
    Returns:
        Dictionary with:
        - 'is_multi_language': Boolean indicating if multiple languages detected
        - 'languages': List of detected language codes
        - 'language_distribution': Dict mapping language codes to their ratios
        - 'primary_language': Most common language
        - 'secondary_languages': List of other languages found
    """
    # Internal defaults
    min_lang_ratio = 0.15
    min_confidence = 0.3
    
    if _is_nan(text):
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }
    
    s = str(text)
    if not s or len(s.strip()) < 10:
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }
    
    # Clean the text
    cleaned = _clean_text(s)
    
    if not cleaned or len(cleaned.strip()) < 10:
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }
    
    if not LANGDETECT_AVAILABLE:
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }
    
    # Split text into chunks
    chunks = _get_text_chunks(cleaned, chunk_size=chunk_size, overlap=overlap)
    
    if len(chunks) < 2:
        # Too short for multi-language detection, use single detection
        primary_lang = detectLanguage(text)
        return {
            'is_multi_language': False,
            'languages': [primary_lang] if primary_lang != 'unknown' else [],
            'language_distribution': {primary_lang: 1.0} if primary_lang != 'unknown' else {},
            'primary_language': primary_lang,
            'secondary_languages': []
        }
    
    # Detect language for each chunk
    chunk_languages = []
    for chunk in chunks:
        try:
            lang = detectLanguage(chunk)
            if lang != 'unknown':
                chunk_languages.append(lang)
        except Exception:
            continue
    
    if not chunk_languages:
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }
    
    # Count language occurrences
    lang_counts = Counter(chunk_languages)
    total_chunks = len(chunk_languages)
    
    # Calculate ratios
    language_distribution = {
        lang: count / total_chunks
        for lang, count in lang_counts.items()
    }
    
    # Filter languages that meet minimum ratio threshold
    significant_languages = [
        lang for lang, ratio in language_distribution.items()
        if ratio >= min_lang_ratio
    ]
    
    # Determine if multi-language
    is_multi_language = len(significant_languages) > 1
    
    # Get primary and secondary languages
    sorted_langs = sorted(
        language_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    primary_language = sorted_langs[0][0] if sorted_langs else 'unknown'
    secondary_languages = [
        lang for lang, _ in sorted_langs[1:]
        if lang in significant_languages
    ]
    
    return {
        'is_multi_language': is_multi_language,
        'languages': significant_languages if is_multi_language else [primary_language] if primary_language != 'unknown' else [],
        'language_distribution': language_distribution,
        'primary_language': primary_language,
        'secondary_languages': secondary_languages
    }


def detectResponseLanguage(
    text: Any,
    *,
    chunk_size: int = 300,
    overlap: int = 100
) -> Dict[str, Any]:
    """
    Convenience wrapper using response-tuned chunking defaults.
    """
    return detectMultiLanguage(
        text,
        chunk_size=chunk_size,
        overlap=overlap
    )


def detectPromptLanguage(
    text: Any,
    *,
    chunk_size: int = 50,
    overlap: int = 15
) -> Dict[str, Any]:
    """
    Convenience wrapper using user prompt chunking defaults.
    """
    return detectMultiLanguage(
        text,
        chunk_size=chunk_size,
        overlap=overlap
    )


__all__ = [
    "detectLanguage",
    "detectMultiLanguage",
    "detectResponseLanguage",
    "detectPromptLanguage",
]

