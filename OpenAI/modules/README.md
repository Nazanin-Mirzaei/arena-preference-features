## Module Overview

- `analyzeWritingStyle(text)` — basic readability metrics (word/sentence counts, avg words per sentence, and a coarse `is_detailed` flag) with NaN-safe handling.
- `countTokens(text, average_chars_per_token=4)` — fast heuristic token estimator that avoids heavy tokenizer dependencies.
- `detectEmojis(text)` — boolean check for common emoji ranges via a Unicode regex.
- `detectLaTeX(text)` — boolean check for LaTeX/math content: inline/display math delimiters (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`), `\begin{...}...\end{...}` environments, and common math macros (`\frac`, `\sum`, `\alpha`, etc.).
- `detectTables(text)` — detects markdown or HTML table patterns for quick structural tagging.
- `is_code.py` — code-block detection and language classification for fenced ```` ``` ```` blocks:
  - `detect_code(text)` — finds fenced code blocks and classifies each one's language via explicit fence tag, Pygments lexer analysis, regex patterns, then keyword matching (in that priority order), returning a list of classification labels.
  - `is_likely_text(content)` — heuristic check for prose/log-style content (bullet/numbered lists, `Step`/`Log:`/`Error:`/`[INFO]` markers) as opposed to code.
- Language detection from `detectLanguage.py`, built on Google MediaPipe's `LanguageDetector` (110-language model):
  - `detectLanguage(text)` — primary language code for a text, or `'unknown'` if detection fails.
  - `detectMultiLanguage(text, chunk_size=20, min_lang_ratio=0.05)` — splits text into word chunks (with CJK-aware chunking for scripts with no word spacing), runs per-chunk detection, and reports whether multiple languages are present along with their distribution.
  - Both functions strip code blocks, LaTeX/math, and URLs before detection to avoid contaminating the language signal.

All helpers sanitize inputs, guard against very short/empty samples, and return conservative defaults (e.g., `0`, `False`, or `'unknown'`) when the signal is weak. Mix and match modules based on your study column; import the specific helper you need or wrap them in your own pipelines.
