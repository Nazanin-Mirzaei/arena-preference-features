"""
Text sanitation helpers shared by the public API: NaN-safe input handling,
stripping of code/LaTeX/URLs before detection, and language code normalization.
"""

from typing import Any
import math
import re

from .config import _LANG_ALIAS, _CODE_LINE_RE, _BINARY_LINE_RE, _LATEX_BARE_RE


def _is_nan(value: Any) -> bool:
    """
    Best-effort NaN detection without heavy dependencies.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


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
