# Language Detection Module

This module provides language identification for the LM Arena 140K dataset preprocessing pipeline. It wraps Google MediaPipe's `LanguageDetector` task to detect the natural language(s) present in model prompts and responses, handling edge cases common in real-world LLM conversation data — code blocks, LaTeX expressions, URL fragments, CJK scripts, and NaN values.

## Overview

The module exposes two public functions:

| Function | Purpose |
|---|---|
| `detectLanguage(text)` | Detect the single primary language of a text. Returns an ISO 639-1 code (e.g., `"en"`, `"zh"`, `"fr"`) or `"unknown"`. |
| `detectMultiLanguage(text, chunk_size, min_lang_ratio)` | Detect whether a text contains multiple languages. Returns a dictionary with per-language chunk ratios, primary and secondary language labels, and a Boolean `is_multi_language` flag. |

## Architecture

The pipeline proceeds through four stages:

```
Input text
    ↓
text_cleaning._clean_text()    — Strip code fences, LaTeX, URLs, code-like lines
    ↓
detector._detect()             — MediaPipe LanguageDetector inference
    ↓                              (for detectLanguage, stops here)
chunking._split_into_word_chunks() — Split into word/CJK-aware chunks
    ↓
chunking._is_low_signal_chunk()    — Filter out code-heavy / undersized chunks
    ↓
collections.Counter vote          — Aggregate per-chunk language predictions
    ↓
detectMultiLanguage()             — Return language distribution
```

## Key Design Decisions

### Code and LaTeX Stripping

Raw LLM conversations mix natural language with substantial non-linguistic content. The `_clean_text` function (`text_cleaning.py:40`) removes:

- **Fenced code blocks**: ` ```…``` `, inline `` `…` ``, `<code>…</code>`
- **Unfenced code lines**: Lines matching common code patterns (imports, function definitions, HTML tags, SQL queries, etc.) via `_CODE_LINE_RE` (config.py:22)
- **Binary lines**: Long runs of only `0`/`1` characters via `_BINARY_LINE_RE` (config.py:37)
- **LaTeX**: Inline `$…$`, display `$$…$$`, `\[…\]`, `\(…\)`, `\begin{…}…\end{…}`, and bare macros (e.g., `\frac{a}{b}`) via a whitelist of known math commands (config.py:46)
- **URLs**: `http(s)://...` patterns

A guard in `_strip_code_lines` (text_cleaning.py:35) prevents stripping a short single-line code snippet down to nothing: if the result is under 3 characters, the original text is preserved.

### CJK-Aware Chunking

Standard word-based splitting (`str.split()`) fails for Chinese and Japanese text, which uses no spaces between words — an entire sentence collapses into a single token, making code-switching between CJK and other scripts undetectable.

The `_split_into_word_chunks` function (chunking.py:33) separates text into runs of "no-space CJK" (Han, Hiragana, Katakana) vs. everything else (Latin, Hangul, etc.). CJK runs are chunked by character count (2× the word budget, since CJK characters carry higher information density), while everything else is chunked by word count. Hangul (Korean) is deliberately grouped with the "other" category since Korean uses spaces between words.

### Low-Signal Filtering

Not every chunk should contribute to the language vote. `_is_low_signal_chunk` (chunking.py:13) excludes:

1. **Micro-chunks** below `_MIN_CHUNK_CHARS` (8 characters) — too short for reliable detection
2. **Code lines** matched by `_CODE_LINE_RE` — unfenced code that was not caught during cleaning
3. **Symbol-heavy chunks** below `_MIN_ALPHA_RATIO` (0.5 alphabetic density) — pure punctuation, operators, or JSON fragments with no natural-language content

This prevents long, code-heavy documents from being over-reported as multi-language due to misdetected code chunks.

### Multi-Language Detection

While `detectLanguage` runs MediaPipe on the entire cleaned text at once and returns the top candidate, `detectMultiLanguage` (api.py:31) works by:

1. Cleaning the text
2. Splitting into chunks (default 20 words or 40 CJK characters)
3. Running `_detect` on each chunk that passes the low-signal filter
4. Aggregating per-chunk predictions via `Counter`
5. Computing language ratios relative to total non-excluded chunks
6. Marking as multi-language if two or more languages each meet `min_lang_ratio` (default 0.05, i.e., ≥5% of chunks)

The default parameters were selected by a grid search maximizing F1 against a hand-labeled validation set of 40 dataset prompts (9 multi-language, 31 single-language), achieving F1 = 0.76 (precision 0.67, recall 0.89).

## Language Code Normalization

Obsolete ISO 639-1 codes are remapped via `_LANG_ALIAS` (config.py:13):

| Old Code | Normalized |
|---|---|
| `iw` (Hebrew) | `he` |
| `in` (Indonesian) | `id` |
| `ji` (Yiddish) | `yi` |

## Dependencies

- `mediapipe` (Google MediaPipe Tasks — Text)
- A TFLite model at `models/language_detector.tflite`
- Python ≥ 3.10

## Usage

```python
from Language_Detection import detectLanguage, detectMultiLanguage

# Single-language detection
detectLanguage("Hello, how are you?")            # 'en'
detectLanguage("Bonjour, comment allez-vous ?")  # 'fr'
detectLanguage(float("nan"))                     # 'unknown'

# Multi-language detection
result = detectMultiLanguage(
    "Hello world. C'est un texte mélangé. Как дела?",
    chunk_size=20,
    min_lang_ratio=0.05
)
# result == {
#     'is_multi_language': True,
#     'languages': ['en', 'fr', 'ru'],
#     'language_distribution': {'en': 0.5, 'fr': 0.25, 'ru': 0.25},
#     'primary_language': 'en',
#     'secondary_languages': ['fr', 'ru']
# }
```

## References

- MediaPipe LanguageDetector: https://ai.google.dev/edge/mediapipe/solutions/text/language_detector
