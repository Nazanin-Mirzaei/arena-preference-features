"""
Language detection utility built on Google MediaPipe's LanguageDetector task.

This module provides `detectLanguage` to detect the primary language of a text,
and `detectMultiLanguage` to detect whether a text mixes multiple languages,
using MediaPipe's native ranked probability distribution over 110 languages.
"""

from typing import Any, Dict, List, Optional
import math
import re
from collections import Counter
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import text as mp_text

_MODEL_PATH = Path(__file__).parent / "models" / "language_detector.tflite"

# Old/alias ISO codes some datasets or consumers may still expect
_LANG_ALIAS = {
    "iw": "he",  # Hebrew
    "in": "id",  # Indonesian
    "ji": "yi",  # Yiddish
}

_detector = None


def _get_detector():
    """
    Lazily construct the MediaPipe LanguageDetector (loads the .tflite model on first use).
    """
    global _detector
    if _detector is None:
        base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp_text.LanguageDetectorOptions(base_options=base_options)
        _detector = mp_text.LanguageDetector.create_from_options(options)
    return _detector


def _is_nan(value: Any) -> bool:
    """
    Best-effort NaN detection without heavy dependencies.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# Lines that look like code even when not fenced in markdown (unfenced pastes,
# or fences left unclosed by the user). Matched per-line so surrounding prose
# on the same block is preserved.
_CODE_LINE_RE = re.compile(
    r'^\s*('
    r'import\s+\w|from\s+[\w.]+\s+import|'
    r'def\s+\w+\s*\(|function\s*\w*\s*\(|'
    r'class\s+\w+[\s:({]|#include\s*[<"]|'
    r'</?(?:template|div|body|html|span|button|script|style|section|h1|p|meta)\b|'
    r'[\w.]+\s*=\s*[\{\[]\s*$|'
    r'^[\}\]\);]+\s*$|'
    r'v-if=|v-for=|v-model=|:class=|:key=|@click=|@close|'
    r'public\s+\w+|private\s+\w+|const\s+\w+\s*=|let\s+\w+\s*=|'
    r'#!/usr/bin/|<\?php|<%|SELECT\s+.*FROM|<!DOCTYPE'
    r')',
    re.MULTILINE | re.IGNORECASE,
)
# Long runs of only 0/1 characters (e.g. ASCII-encoded binary), not natural language.
_BINARY_LINE_RE = re.compile(r'^[01\s]{20,}$')

# Bare LaTeX commands with no wrapping delimiter (no $...$, \[...\], or
# \begin{}...\end{}) - e.g. "\frac{a}{b}" or "\lambda" typed directly.
# Whitelisted to known math macros (same list as modules/detectLaTeX.py) rather
# than matching any backslash-word sequence, to avoid stripping non-math
# backslash sequences this module doesn't recognize as LaTeX.
# Matches 1+ leading backslashes since some sources (e.g. text extracted from
# repr()-like serialized data) carry doubled-up escaping.
_LATEX_MACROS = [
    "frac", "sum", "int", "sqrt", "alpha", "beta", "gamma", "theta",
    "lambda", "mu", "infty", "cdot", "times", "leq", "geq", "neq", "approx",
    "pm", "mp", "left", "right", "mathrm", "mathbb", "mathbf", "partial",
    "nabla", "begin", "end", "text",
]
_LATEX_BRACE_ARG = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
_LATEX_BARE_RE = re.compile(
    r'\\+(' + '|'.join(_LATEX_MACROS) + r')\b'
    + r'(?:' + _LATEX_BRACE_ARG + r'|\[[^\[\]]*\])*'
)


# Minimum fraction of a chunk's non-whitespace characters that must be
# alphabetic for the chunk to be worth a language vote. Chunks below this are
# presumed to be code/symbols/punctuation with no real natural-language
# content (e.g. a Pine Script parameter block, a JSON fragment, a run of
# operators) and are excluded from detectMultiLanguage's vote rather than
# forcing MediaPipe to guess a language for them.
_MIN_ALPHA_RATIO = 0.5


def _is_low_signal_chunk(chunk: str) -> bool:
    """
    True if a detectMultiLanguage chunk should be excluded from the language
    vote: too short to classify reliably (below _MIN_CHUNK_CHARS - the same
    threshold _split_into_word_chunks uses for merging micro-chunks), an
    unfenced code line (_CODE_LINE_RE), or below _MIN_ALPHA_RATIO alphabetic
    density (symbols/punctuation/code with no real language content).
    """
    stripped = chunk.strip()
    if len(stripped) < _MIN_CHUNK_CHARS:
        return True
    if _CODE_LINE_RE.match(stripped):
        return True
    alpha_count = sum(1 for c in stripped if c.isalpha())
    non_space_count = sum(1 for c in stripped if not c.isspace())
    if non_space_count == 0:
        return True
    return (alpha_count / non_space_count) < _MIN_ALPHA_RATIO


def _strip_code_lines(text: str) -> str:
    # Split on real newlines or literal backslash-n (some sources, e.g. text
    # extracted from repr()-like serialized data, carry the escaped form).
    lines = re.split(r'\n|\\n', text)
    kept = [
        ln for ln in lines
        if not _CODE_LINE_RE.match(ln) and not _BINARY_LINE_RE.match(ln.strip())
    ]
    stripped = '\n'.join(kept)
    # Guard against stripping a short, single-line code snippet down to nothing:
    # only apply the result if meaningful text actually remains.
    if len(stripped.strip()) < 3:
        return text
    return stripped


def _clean_text(text: str) -> str:
    """
    Clean text for better language detection by removing code blocks,
    LaTeX/math expressions, URLs, and excessive whitespace.
    """
    # Remove code blocks (markdown and HTML)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'<code>[\s\S]*?</code>', ' ', text)

    # Strip remaining code-like lines that weren't fenced, or whose fence was
    # left unclosed (a stray leftover backtick after this point is harmless text noise)
    text = _strip_code_lines(text)

    # Remove LaTeX/math expressions
    # Inline math: $...$
    text = re.sub(r'\$(?:\\.|[^\$\\])+\$', ' ', text)
    # Display math: $$...$$
    text = re.sub(r'\$\$(?:\\.|[^\$\\])+\$\$', ' ', text)
    # \[ ... \]
    text = re.sub(r'\\\[(?:.|\n)*?\\\]', ' ', text)
    # \(...\)
    text = re.sub(r'\\\((?:.|\n)*?\\\)', ' ', text)
    # LaTeX environments: \begin{...}...\end{...}
    text = re.sub(r'\\begin\{[^\}]+\}[\s\S]*?\\end\{[^\}]+\}', ' ', text)

    # Remove any remaining bare (undelimited) LaTeX commands
    text = _LATEX_BARE_RE.sub(' ', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def _normalize_lang(code: str) -> str:
    return _LANG_ALIAS.get(code, code)


def _detect(text: str):
    """
    Run the MediaPipe detector on cleaned text. Returns the ranked list of
    detections (language_code, probability), sorted descending by probability,
    or an empty list if the text is unusable.
    """
    if _is_nan(text):
        return []

    s = str(text)
    if not s or len(s.strip()) < 3:
        return []

    cleaned = _clean_text(s)
    if not cleaned or len(cleaned.strip()) < 3:
        return []

    result = _get_detector().detect(cleaned)
    return result.detections


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


# Unicode ranges for CJK scripts with NO spaces between words (Han, Hiragana,
# Katakana), plus CJK punctuation/fullwidth forms so punctuation doesn't
# fragment these spans into tiny useless chunks. Hangul (Korean) is
# deliberately excluded here: unlike Chinese/Japanese, Korean text uses spaces
# between words, so it's chunked by word count like Latin script rather than
# by raw character count (see _split_into_word_chunks).
_CJK_NO_SPACE_RE = re.compile(
    r'[぀-ゟ゠-ヿ一-鿿'
    r'　-〿＀-￯、-〿]+'
)

# Minimum characters for a chunk to be worth running through the detector;
# below this, per-chunk predictions are too unreliable to be useful signal.
# Used both by _split_into_word_chunks (merges undersized chunks into a
# neighbor rather than detecting them alone) and by _is_low_signal_chunk
# (excludes undersized chunks from detectMultiLanguage's vote entirely).
_MIN_CHUNK_CHARS = 8


def _split_into_word_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Split text into chunks small enough for reliable per-chunk language
    detection. Word-based splitting (str.split()) works for space-delimited
    scripts but fails for Chinese/Japanese text, which has no spaces between
    words: an entire sentence collapses into a single "word" token, and
    adjacent CJK/Latin text with no space between them gets glued together
    into one chunk, making code-switching between CJK and other languages
    undetectable.

    To handle this, text is first split into runs of "no-space CJK" (Han,
    Hiragana, Katakana) vs. everything else. Everything else - including
    Hangul (Korean), which does use spaces between words - is chunked by word
    count as before. No-space CJK runs are chunked by character count (using
    roughly the same chars-per-chunk budget as the word-based chunk_size,
    since these scripts have no word boundaries to split on).
    Chunks below _MIN_CHUNK_CHARS are merged into a neighboring chunk rather
    than detected on their own, since very short chunks are unreliable.

    Note: this function only splits text into chunks - it doesn't decide
    which chunks are worth detecting. detectMultiLanguage separately filters
    the returned chunks through _is_low_signal_chunk before running detection,
    to exclude chunks that are mostly code/symbols.
    """
    # CJK characters average roughly 2x the "information density" of a
    # whitespace-delimited word, so use a comparable character budget per chunk.
    cjk_chunk_chars = chunk_size * 2

    spans = []
    last_end = 0
    for m in _CJK_NO_SPACE_RE.finditer(text):
        if m.start() > last_end:
            spans.append(('other', text[last_end:m.start()]))
        spans.append(('cjk', m.group()))
        last_end = m.end()
    if last_end < len(text):
        spans.append(('other', text[last_end:]))

    raw_chunks = []
    for kind, span in spans:
        if not span.strip():
            continue
        if kind == 'cjk':
            for i in range(0, len(span), cjk_chunk_chars):
                piece = span[i:i + cjk_chunk_chars]
                if piece.strip():
                    raw_chunks.append(piece)
        else:
            words = span.split()
            for i in range(0, len(words), chunk_size):
                piece = ' '.join(words[i:i + chunk_size])
                if piece.strip():
                    raw_chunks.append(piece)

    if not raw_chunks:
        return [text] if text.strip() else []

    # Merge chunks shorter than _MIN_CHUNK_CHARS into a neighbor so no
    # unreliable micro-chunk is sent to the detector on its own.
    merged = []
    pending = ''
    for chunk in raw_chunks:
        combined = (pending + ' ' + chunk).strip() if pending else chunk
        if len(combined) < _MIN_CHUNK_CHARS:
            pending = combined
        else:
            merged.append(combined)
            pending = ''
    if pending:
        if merged:
            merged[-1] = (merged[-1] + ' ' + pending).strip()
        else:
            merged.append(pending)

    return merged


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
    per-chunk code-line stripping (like detectLanguage's _strip_code_lines),
    not just a code-vs-no-code classification.

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


__all__ = [
    "detectLanguage",
    "detectMultiLanguage",
]
