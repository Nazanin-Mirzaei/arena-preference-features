"""
Public API: detectLanguage for single-label detection, and
detectMultiLanguage for code-switching / mixed-language detection.
"""

from typing import Any, Dict
from collections import Counter

from .text_cleaning import _is_nan, _clean_text, _normalize_lang
from .detector import _detect
from .chunking import _split_into_word_chunks, _is_low_signal_chunk


def detectLanguage(text: Any) -> str:
    """
    Detect the primary language of a text.

    Args:
        text: Input text to analyze

    Returns:
        Primary language code (e.g., 'en', 'zh', 'fr') or 'unknown' if detection fails
    """
    detections = _detect(text)
    if not detections:
        return 'unknown'

    return _normalize_lang(detections[0].language_code)


def detectMultiLanguage(
    text: Any,
    *,
    chunk_size: int = 20,
    min_lang_ratio: float = 0.05
) -> Dict[str, Any]:
    """
    Detect if a text contains multiple languages by running MediaPipe's
    LanguageDetector on word chunks of the text and looking at the resulting
    language distribution across chunks.

    A single detect() call only returns the model's overall best-guess
    language (plus its residual uncertainty over other candidates) for the
    whole input; it does not localize which language appears in which part
    of the text. Chunking is what actually finds code-switched/mixed-language
    content, since MediaPipe stays reliable even on short chunks.

    Chunks that are too short, an unfenced code line, or mostly code/symbols
    rather than natural language are excluded from the vote entirely rather
    than forced into a language guess (see _is_low_signal_chunk) - this
    prevents long, code-heavy documents from being over-reported as
    multi-language just because a pure-code chunk got misdetected.

    Known remaining gap: this only catches chunks with no real language
    content. A chunk that MIXES code syntax with genuine natural-language
    text (e.g. a VBA subroutine with Portuguese string literals and comments
    interleaved throughout) still has real language content, so it isn't
    excluded - but the surrounding code syntax can dilute the signal enough
    that MediaPipe scatters its guess across several near-miss languages
    (e.g. Portuguese content misread as pt/es/ca/gl/it across chunks) instead
    of settling on the one correct language. Closing this gap would need
    per-chunk code-line stripping (like text_cleaning._strip_code_lines), not
    just a code-vs-no-code classification.

    Defaults (chunk_size=20, min_lang_ratio=0.05) were chosen by a grid search
    that maximized F1 against a hand-labeled validation set of 40 real dataset
    prompts (9 judged genuinely multi-language, 31 single-language), scoring
    precision/recall at each (chunk_size, min_lang_ratio) combination. This
    setting reached F1=0.76 (precision=0.67, recall=0.89), and was a stable
    local optimum across neighboring values, not a single-cell spike - but 40
    examples is still a small validation set for this parameter, so treat it
    as the best available empirical estimate rather than a fully converged
    optimum.

    Args:
        text: Input text to analyze
        chunk_size: Number of words per chunk
        min_lang_ratio: Minimum fraction of chunks a language must appear as
            the primary language in, to be considered significant

    Returns:
        Dictionary with:
        - 'is_multi_language': Boolean indicating if multiple languages detected
        - 'languages': List of detected language codes
        - 'language_distribution': Dict mapping language codes to their chunk ratio
        - 'primary_language': Most common language across chunks
        - 'secondary_languages': List of other significant languages found
    """
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

    cleaned = _clean_text(s)
    if not cleaned or len(cleaned.strip()) < 10:
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }

    chunks = _split_into_word_chunks(cleaned, chunk_size)

    chunk_languages = [
        _normalize_lang(detections[0].language_code)
        for detections in (_detect(chunk) for chunk in chunks if not _is_low_signal_chunk(chunk))
        if detections
    ]

    if not chunk_languages:
        return {
            'is_multi_language': False,
            'languages': [],
            'language_distribution': {},
            'primary_language': 'unknown',
            'secondary_languages': []
        }

    lang_counts = Counter(chunk_languages)
    total_chunks = len(chunk_languages)
    language_distribution = {
        lang: count / total_chunks
        for lang, count in lang_counts.items()
    }

    significant_languages = [
        lang for lang, ratio in language_distribution.items()
        if ratio >= min_lang_ratio
    ]

    sorted_langs = sorted(language_distribution.items(), key=lambda x: x[1], reverse=True)
    primary_language = sorted_langs[0][0]
    is_multi_language = len(significant_languages) > 1
    secondary_languages = [
        lang for lang, _ in sorted_langs[1:]
        if lang in significant_languages
    ]

    return {
        'is_multi_language': is_multi_language,
        'languages': significant_languages if is_multi_language else [primary_language],
        'language_distribution': language_distribution,
        'primary_language': primary_language,
        'secondary_languages': secondary_languages
    }
