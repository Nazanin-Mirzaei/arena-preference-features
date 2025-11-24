## Module Overview

- `analyzeWritingStyle(text)` — basic readability metrics (word/sentence counts, avg words per sentence, and a coarse `is_detailed` flag) with NaN-safe handling.
- `countTokens(text, average_chars_per_token=4)` — fast heuristic token estimator that avoids heavy tokenizer dependencies.
- `detectEmojis(text)` — boolean check for common emoji ranges via a Unicode regex.
- `detectTables(text)` — detects markdown or HTML table patterns for quick structural tagging.
- Language helpers from `detectLanguage.py`:
  - `detectLanguage(text)` — primary language code for short inputs.
  - `detectMultiLanguage(text, chunk_size=200, overlap=50)` — scans overlapping windows to surface multiple languages.
  - `detectResponseLanguage(text, chunk_size=300, overlap=100)` — tuned for longer assistant-style responses.
  - `detectPromptLanguage(text, chunk_size=50, overlap=15)` — tuned for short user prompts.

All helpers sanitize inputs, guard against very short/empty samples, and return conservative defaults (e.g., `0`, `False`, or `'unknown'`) when the signal is weak. Mix and match modules based on your study column; import the specific helper you need or wrap them in your own pipelines.
