"""
Shared constants and configuration for the Language_Detection package:
model path, language code aliases, regex patterns, and tunable thresholds
used across the cleaning, chunking, and detection modules.
"""

import re
from pathlib import Path

_MODEL_PATH = Path(__file__).parent / "models" / "language_detector.tflite"

# Old/alias ISO codes some datasets or consumers may still expect
_LANG_ALIAS = {
    "iw": "he",  # Hebrew
    "in": "id",  # Indonesian
    "ji": "yi",  # Yiddish
}

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

# Unicode ranges for CJK scripts with NO spaces between words (Han, Hiragana,
# Katakana), plus CJK punctuation/fullwidth forms so punctuation doesn't
# fragment these spans into tiny useless chunks. Hangul (Korean) is
# deliberately excluded here: unlike Chinese/Japanese, Korean text uses spaces
# between words, so it's chunked by word count like Latin script rather than
# by raw character count (see chunking._split_into_word_chunks).
_CJK_NO_SPACE_RE = re.compile(
    r'[぀-ゟ゠-ヿ一-鿿'
    r'　-〿＀-￯、-〿]+'
)

# Minimum characters for a chunk to be worth running through the detector;
# below this, per-chunk predictions are too unreliable to be useful signal.
# Used both by chunking._split_into_word_chunks (merges undersized chunks into
# a neighbor rather than detecting them alone) and by chunking._is_low_signal_chunk
# (excludes undersized chunks from detectMultiLanguage's vote entirely).
_MIN_CHUNK_CHARS = 8

# Minimum fraction of a chunk's non-whitespace characters that must be
# alphabetic for the chunk to be worth a language vote. Chunks below this are
# presumed to be code/symbols/punctuation with no real natural-language
# content (e.g. a Pine Script parameter block, a JSON fragment, a run of
# operators) and are excluded from detectMultiLanguage's vote rather than
# forcing MediaPipe to guess a language for them.
_MIN_ALPHA_RATIO = 0.5
