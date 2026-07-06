"""
Chunk splitting for multi-language detection: word/CJK-aware chunking of
text into pieces small enough for reliable per-chunk language detection,
plus the low-signal filter that excludes code/symbol-heavy chunks from
the language vote.
"""

from typing import List

from .config import _CJK_NO_SPACE_RE, _MIN_CHUNK_CHARS, _MIN_ALPHA_RATIO, _CODE_LINE_RE


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
