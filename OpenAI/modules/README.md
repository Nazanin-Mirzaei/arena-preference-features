## Language Detection Helpers

- `detectLanguage(text)` returns the primary language code for short snippets.
- `detectMultiLanguage(text, chunk_size=200, overlap=50)` inspects overlapping chunks; adjust chunking when text lengths change.
- `detectResponseLanguage(text, chunk_size=300, overlap=100)` is tuned for long assistant responses (~300-word chunks with 100-word overlap); override the defaults if your responses differ significantly.
- `detectPromptLanguage(text, chunk_size=50, overlap=15)` is tuned for short user prompts (~50-word chunks with 15-word overlap).
- All helpers clean the text, skip very short inputs, and fall back to `'unknown'` if detection is unreliable.

When integrating into new studies, import the wrapper that matches your column type, or call `detectMultiLanguage` directly if you need custom chunking.
