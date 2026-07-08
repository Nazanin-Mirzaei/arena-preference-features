# Response Features Extraction Module

## Overview

This module provides a **comprehensive library of lightweight, dependency-free feature extractors** for analyzing raw LLM-generated response text. Each extractor operates on a single text input and returns interpretable, ML-ready features capturing structural, linguistic, stylistic, and formatting properties of model outputs.

Designed for large-scale evaluation pipelines, these features enable pairwise response comparison (A/B), behavioral profiling, and supervised preference modeling without requiring heavy NLP infrastructure.

---

## Module Structure

```
Response_Features/
│
├── Code_Delimiter_Count.py       # Code delimiter frequency
├── Emoji_Features.py             # Emoji presence and count
├── Has_Latex.py                  # LaTeX / mathematical expression detection
├── Interaction_Features.py       # Conversational engagement signals
├── Is_Code_Block.py              # Code block and programming syntax detection
├── Is_Natural_Text.py            # Natural language vs. code classification
├── Paragraph_Count.py            # Paragraph segmentation count
├── Paragraph_Statistics.py       # Paragraph length mean and standard deviation
├── Punctuation_Count.py          # Punctuation mark frequency
├── Repetition.py                 # Lexical repetition density
├── Sentence_Count.py             # Sentence count estimation
├── Sentence_Length.py            # Short and long sentence classification
├── Sentence_Length_Stats.py      # Sentence length standard deviation
├── Sentence_Paragraph_Stats.py   # Sentence-per-paragraph dispersion
├── Table_Detection.py            # Markdown / HTML table detection and count
├── Token_Features.py             # Character-based adaptive token estimation
├── Writing_Style_Features.py     # Structural style, reasoning, and detail signals
│
└── README.md
```

---

## Input Format

All feature extractors accept a single raw text input of type `Any`:

```python
text: Any
```

Each function independently handles `None`, `NaN`, empty strings, and non-string coercions by returning a zero-initialized or neutral default. No preprocessing or text normalization is required before calling any extractor.

---

## Feature Descriptions

### 1. Code Delimiter Count (`Code_Delimiter_Count.py`)

**Function:** `count_code_delimiters(text) -> int`

Counts occurrences of code-related delimiters as a proxy for how code-heavy a response is.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `int` | Sum of Markdown code fences (`` ``` ``, `~~~`), inline backtick sequences, and HTML code-related tags (`<code>`, `<pre>`, `<script>`, `<style>`) |

---

### 2. Emoji Features (`Emoji_Features.py`)

**Functions:** `detect_emoji(text) -> bool`, `count_emojis(text) -> int`

Detects and counts emoji characters using an extended Unicode 15.1 pattern covering emoticons, symbols, flags, dingbats, and pictographs.

| Feature | Type | Description |
|---------|------|-------------|
| `detect_emoji` | `bool` | Whether the text contains any emoji characters |
| `count_emojis` | `int` | Total number of emoji occurrences |

---

### 3. LaTeX Detection (`Has_Latex.py`)

**Function:** `has_latex(text) -> bool`

Detects the presence of LaTeX or mathematical notation using regex heuristics.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `bool` | Whether text contains inline math (`$...$`), display math (`$$...$$`), LaTeX environments (`\begin...\end`), or common math commands (`\frac`, `\sum`, `\int`, etc.) |

---

### 4. Interaction Features (`Interaction_Features.py`)

**Function:** `extract_interaction_features(text) -> dict`

Captures conversational engagement patterns through lexical indicators of discourse structure.

| Feature | Type | Description |
|---------|------|-------------|
| `has_question_at_end` | `bool` | Text ends with a question mark |
| `has_conclusion` | `bool` | Contains conclusive phrases (e.g., "in conclusion", "therefore") |
| `has_next_steps` | `bool` | Contains forward-looking phrases (e.g., "next steps", "going forward") |
| `has_interaction_prompt` | `bool` | Contains user-engagement prompts (e.g., "let me know", "feel free to") |
| `interaction_score` | `int` | Composite score: sum of the four binary indicators `[0, 4]` |

---

### 5. Code Block Detection (`Is_Code_Block.py`)

**Function:** `is_code_block(text) -> bool`

Determines whether a response contains code-like content through multi-signal detection.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `bool` | `True` if text contains fenced code blocks, inline backtick code, HTML code tags, or heuristic programming syntax patterns (function definitions, imports, control flow) |

---

### 6. Natural Text Classification (`Is_Natural_Text.py`)

**Function:** `is_natural_text(text) -> bool`

Classifies input as natural prose or code/markup using lexical, structural, and ratio-based signals.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `bool` | `True` if text resembles natural language, `False` if it resembles code or structured markup |

Decision criteria include fenced code delimiter prevalence, code-like line ratio, alphabetic character proportion, and total code delimiter count.

---

### 7. Paragraph Count (`Paragraph_Count.py`)

**Function:** `extract_paragraph_features(text) -> dict`

Segments text by double-newline boundaries and counts non-empty paragraphs.

| Feature | Type | Description |
|---------|------|-------------|
| `paragraph_count` | `int` | Number of paragraphs delimited by blank lines |

---

### 8. Paragraph Statistics (`Paragraph_Statistics.py`)

**Function:** `extract_paragraph_statistics(text) -> dict`

Computes paragraph-length distribution statistics using word-count-based paragraph segmentation.

| Feature | Type | Description |
|---------|------|-------------|
| `paragraph_length_mean` | `float` | Mean words per paragraph |
| `paragraph_length_std` | `float` | Population standard deviation of paragraph word counts |

---

### 9. Punctuation Count (`Punctuation_Count.py`)

**Function:** `count_punctuation(text) -> int`

Counts common punctuation marks as a lightweight proxy for writing structure density.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `int` | Frequency of `.`, `,`, `!`, `?`, `;`, `:`, `(`, `)`, `[`, `]`, `{`, `}`, `"`, `'`, `` ` ``, `…` |

---

### 10. Repetition Density (`Repetition.py`)

**Function:** `compute_repetition_density(text) -> float`

Estimates lexical repetition using inverse type-token ratio.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `float` | Proportion of repeated tokens: `(total_words - unique_words) / total_words` in range `[0, 1]` |

Computed case-insensitively with Unicode-aware word boundaries.

---

### 11. Sentence Count (`Sentence_Count.py`)

**Function:** `count_sentences(text) -> int`

Estimates sentence count by splitting on sentence-ending punctuation and newlines.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `int` | Number of sentences detected |

---

### 12. Sentence Length Features (`Sentence_Length.py`)

**Function:** `extract_sentence_length_features(text) -> dict`

Classifies sentences into short (≤10 words) and long (≥30 words) categories.

| Feature | Type | Description |
|---------|------|-------------|
| `long_sentence_count` | `int` | Sentences with 30 or more words |
| `short_sentence_count` | `int` | Sentences with 10 or fewer words |

---

### 13. Sentence Length Standard Deviation (`Sentence_Length_Stats.py`)

**Function:** `compute_sentence_length_std(text) -> dict`

Computes the population standard deviation of sentence lengths (in words) across all sentences.

| Feature | Type | Description |
|---------|------|-------------|
| `sentence_length_std` | `float` | Dispersion measure of sentence word counts |

---

### 14. Sentence-Per-Paragraph Dispersion (`Sentence_Paragraph_Stats.py`)

**Function:** `compute_sentence_per_paragraph_std(text) -> dict`

Computes the population standard deviation of sentence counts across paragraphs, capturing structural inconsistency.

| Feature | Type | Description |
|---------|------|-------------|
| `sentence_per_paragraph_std` | `float` | Dispersion of sentences-per-paragraph across the document |

---

### 15. Table Detection (`Table_Detection.py`)

**Functions:** `detect_tables(text) -> bool`, `count_tables(text) -> int`

Detects and counts structured tabular content in Markdown pipe syntax and HTML table markup.

| Feature | Type | Description |
|---------|------|-------------|
| `detect_tables` | `bool` | Whether any Markdown or HTML table structure is present |
| `count_tables` | `int` | Total table occurrences (Markdown pipes + HTML table tags) |

---

### 16. Token Features (`Token_Features.py`)

**Function:** `count_tokens(text, average_chars_per_token=4) -> int`

Provides adaptive token count estimation without external tokenizers, using language-group-specific character-to-token ratios.

| Feature | Type | Description |
|---------|------|-------------|
| `return` | `int` | Estimated token count (rounded, minimum 1) |

Character-to-token ratios are calibrated per language group:
- **Latin:** 3.8 characters per token
- **Persian/Arabic:** 2.4 characters per token
- **Chinese:** 1.2 characters per token
- **Other:** configurable fallback (default 4.0)

---

### 17. Writing Style Features (`Writing_Style_Features.py`)

**Function:** `extract_writing_style_features(text) -> dict`

Extracts high-level structural and reasoning-related writing style signals.

| Feature | Type | Description |
|---------|------|-------------|
| `word_count` | `int` | Whitespace-delimited word count |
| `sentence_count` | `int` | Sentence count via punctuation splitting |
| `avg_words_per_sentence` | `float` | Mean words per sentence |
| `is_detailed` | `bool` | Heuristic: `True` if `avg_words_per_sentence > 15` and `word_count > 50` |
| `has_step_by_step` | `bool` | Detects structured reasoning via enumerations, ordinals, procedural transitions, logical connectors, and chain-of-thought markers |

---

## Category Summary

| Category | Modules |
|----------|---------|
| **Structural** | `Paragraph_Count`, `Paragraph_Statistics`, `Sentence_Count`, `Sentence_Length`, `Sentence_Length_Stats`, `Sentence_Paragraph_Stats` |
| **Code & Technical** | `Code_Delimiter_Count`, `Is_Code_Block`, `Table_Detection`, `Has_Latex` |
| **Linguistic & Lexical** | `Repetition`, `Token_Features`, `Is_Natural_Text` |
| **Stylistic** | `Writing_Style_Features`, `Sentence_Length`, `Sentence_Length_Stats`, `Paragraph_Statistics`, `Punctuation_Count` |
| **Discourse & Interaction** | `Interaction_Features`, `Writing_Style_Features` |
| **Symbolic** | `Emoji_Features` |

---

## Usage Examples

### Single Feature Extraction

```python
from Response_Features.Code_Delimiter_Count import count_code_delimiters
from Response_Features.Emoji_Features import count_emojis, detect_emoji

text = df.loc[0, "response_a"]

code_density = count_code_delimiters(text)
emoji_count = count_emojis(text)
has_emoji = detect_emoji(text)
```

### Composite Feature Extraction

```python
from Response_Features.Interaction_Features import extract_interaction_features
from Response_Features.Writing_Style_Features import extract_writing_style_features

text = df.loc[0, "response_a"]

interaction = extract_interaction_features(text)
style = extract_writing_style_features(text)
```

### Batch Processing with Pandas

```python
from Response_Features.Repetition import compute_repetition_density
from Response_Features.Token_Features import count_tokens

df["a_repetition_density"] = df["response_a"].apply(compute_repetition_density)
df["b_repetition_density"] = df["response_b"].apply(compute_repetition_density)
df["a_tokens"] = df["response_a"].apply(count_tokens)
df["b_tokens"] = df["response_b"].apply(count_tokens)
```

### Full Pipeline via Feature Runner

```python
from Feature_Extraction.Feature_Runners.Run_Response_Features import run_all_response_features

df = run_all_response_features(df)
```

---

## Design Principles

### Dependency-Free Implementation

All extractors rely exclusively on the Python standard library (`re`, `math`, `typing`). No external NLP frameworks, language models, or tokenization libraries are required.

### Robust Input Handling

Every function defensively guards against:
- `None` / `NaN` / `pd.NA` inputs
- Empty or whitespace-only strings
- Non-string types via safe coercion
- String `"nan"` representations

### Stateless and Parallelizable

Each extractor is a pure function with no side effects, enabling straightforward unit testing, distributed processing, and vectorized application via `pandas.Series.apply`.

### ML-Ready Output

All features are returned as primitive types (`int`, `float`, `bool`) or flat dictionaries thereof, suitable for direct ingestion into gradient-boosted trees, linear models, or neural classifiers without additional encoding.

---

## Intended Use Cases

- **Pairwise response comparison (A/B):** Extract features independently from `response_a` and `response_b` to model preference, quality deltas, and stylistic divergence.
- **Behavioral profiling:** Characterize model outputs along axes of verbosity, structure, code usage, reasoning style, and engagement.
- **Reward model feature engineering:** Provide high-level structural priors for preference and reward modeling in RLHF pipelines.
- **Style vs. reasoning classification:** Distinguish between formatting-heavy responses and structured analytical outputs.

---

## Notes

- Token counts in `Token_Features.py` are heuristic estimates based on character-to-token ratios calibrated per script group, not exact tokenizer outputs.
- Sentence segmentation across all modules uses regex heuristics (punctuation + newline splitting) and may diverge from ground-truth parse trees for edge cases such as abbreviations or nested quotations.
- Language group detection in `Token_Features.py` is script-based and not a full language identification system; it primarily distinguishes Latin, Persian/Arabic, and Chinese scripts.
- All regex patterns are designed for speed and interpretability over exhaustive coverage; some edge cases (e.g., nested code blocks, multi-line LaTeX) may be underrepresented.
