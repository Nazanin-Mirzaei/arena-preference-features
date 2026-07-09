# Text Features Extraction Module

## Overview

This module provides a **comprehensive library of lightweight feature extractors** for analyzing raw LLM-generated text inputs. Each extractor operates on a single text input and returns interpretable, ML-ready features capturing structural, linguistic, stylistic, formatting, and quality/failure properties.

The same feature extraction framework can be applied to different text sources, including user prompts and model-generated responses.

Designed for large-scale evaluation pipelines, these features enable prompt-response analysis, pairwise response comparison (A/B), behavioral profiling, format-compliance checking, and supervised preference modeling without requiring heavy NLP infrastructure.

---

## Module Structure

```

Text_Features/
│
├── Code_Delimiter_Count.py       # Code delimiter frequency
├── Emoji_Features.py             # Emoji presence and count
├── Format_Request_Detection.py   # Prompt-side requested output format (list/table/json/code)
├── Has_Latex.py                  # LaTeX / mathematical expression detection
├── Interaction_Features.py       # Conversational and instructional interaction signals
├── Is_Code_Block.py              # Code block and programming syntax detection
├── Is_Natural_Text.py            # Natural language vs. code classification
├── Json_Detection.py             # JSON object/array detection
├── Length_Request_Detection.py   # Prompt-side requested response length (brief/detailed)
├── List_Detection.py             # Markdown list detection and item count
├── Near_Empty_Detection.py       # Empty / degenerate response detection
├── Paragraph_Count.py            # Paragraph segmentation count
├── Paragraph_Statistics.py       # Paragraph length mean and standard deviation
├── Punctuation_Count.py          # Punctuation mark frequency
├── Refusal_Detection.py          # Refusal / apology / inability language detection
├── Repetition.py                 # Lexical repetition density
├── Script_Detection.py           # Dominant Unicode script classification
├── Sentence_Count.py             # Sentence count estimation
├── Sentence_Length.py            # Short and long sentence classification
├── Sentence_Length_Stats.py      # Sentence length standard deviation
├── Sentence_Paragraph_Stats.py   # Sentence-per-paragraph dispersion
├── Table_Detection.py            # Markdown / HTML table detection and count
├── Token_Features.py             # Character-based adaptive token estimation
├── Truncation_Detection.py       # Cut-off / unfinished response detection
├── Writing_Style_Features.py     # Structural style, reasoning, and detail signals
│
└── README.md

````

---

## Input Format

All feature extractors accept a single raw text input of type `Any`:

```python
text: Any
````

Each function independently handles:

* `None`
* `NaN`
* empty strings
* whitespace-only inputs
* non-string values

by returning a zero-initialized or neutral default.

No preprocessing or text normalization is required before calling any extractor.

The same extractor functions are used for:

* `user_prompt`
* `response_a`
* `response_b`

`Format_Request_Detection.py` and `Length_Request_Detection.py` are prompt-only extractors: they infer what the user *asked for*, and are only meaningful against `user_prompt`, not responses.

---

## Feature Descriptions

### 1. Code Delimiter Count (`Code_Delimiter_Count.py`)

**Function:** `count_code_delimiters(text) -> int`

Counts occurrences of code-related delimiters as a proxy for how code-heavy a text input is.

| Feature | Type  | Description                                                                                                                                    |
| ------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| return  | `int` | Sum of Markdown code fences (` ``` `, `~~~`), inline backtick sequences, and HTML code-related tags (`<code>`, `<pre>`, `<script>`, `<style>`) |

---

### 2. Emoji Features (`Emoji_Features.py`)

**Functions:**
`detect_emoji(text) -> bool`
`count_emojis(text) -> int`

Detects and counts emoji characters using an extended Unicode 15.1 pattern covering emoticons, symbols, flags, dingbats, and pictographs.

| Feature      | Type   | Description                                    |
| ------------ | ------ | ----------------------------------------------- |
| detect_emoji | `bool` | Whether the text contains any emoji characters |
| count_emojis | `int`  | Total number of emoji occurrences              |

---

### 3. Format Request Detection (`Format_Request_Detection.py`)

**Function:** `extract_format_request_features(text) -> dict`

Infers which output format(s) a user prompt explicitly requests, using regex heuristics. Used to later check whether a response's actual format (`Is_Code_Block`, `Table_Detection`, `List_Detection`, `Json_Detection`) complies with what was asked for. **Prompt-only** — not meaningful on responses.

| Feature            | Type   | Description                                                             |
| ------------------- | ------ | ------------------------------------------------------------------------ |
| `requests_list`     | `bool` | Prompt asks for a bulleted/numbered list ("as a list", "enumerate")     |
| `requests_table`    | `bool` | Prompt asks for tabular output ("as a table", "tabulate")               |
| `requests_json`     | `bool` | Prompt asks for JSON output ("in json", "return json")                  |
| `requests_code`     | `bool` | Prompt asks for code ("write a function", "implement ... in Python")    |

---

### 4. LaTeX Detection (`Has_Latex.py`)

**Function:** `has_latex(text) -> bool`

Detects the presence of LaTeX or mathematical notation using regex heuristics.

| Feature | Type   | Description                                                                                                                                                          |
| ------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| return  | `bool` | Whether text contains inline math (`$...$`), display math (`$$...$$`), LaTeX environments (`\begin...\end`), or common math commands (`\frac`, `\sum`, `\int`, etc.) |

---

### 5. Interaction Features (`Interaction_Features.py`)

**Function:** `extract_interaction_features(text) -> dict`

Captures conversational and instructional interaction patterns through lexical indicators of discourse structure.

| Feature                  | Type   | Description                                                            |
| ------------------------ | ------ | ------------------------------------------------------------------------ |
| `has_question_at_end`    | `bool` | Text ends with a question mark                                         |
| `has_conclusion`         | `bool` | Contains conclusive phrases (e.g., "in conclusion", "therefore")       |
| `has_next_steps`         | `bool` | Contains forward-looking phrases (e.g., "next steps", "going forward") |
| `has_interaction_prompt` | `bool` | Contains engagement prompts (e.g., "let me know", "feel free to")      |
| `interaction_score`      | `int`  | Composite score: sum of the four binary indicators `[0, 4]`            |

---

### 6. Code Block Detection (`Is_Code_Block.py`)

**Function:** `is_code_block(text) -> bool`

Determines whether a text input contains code-like content through multi-signal detection.

| Feature | Type   | Description                                                                                                                |
| ------- | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| return  | `bool` | `True` if text contains fenced code blocks, inline backtick code, HTML code tags, or heuristic programming syntax patterns |

---

### 7. Natural Text Classification (`Is_Natural_Text.py`)

**Function:** `is_natural_text(text) -> bool`

Classifies input as natural prose or code/markup using lexical, structural, and ratio-based signals.

| Feature | Type | Description |
|---------|------|-------------|
| return | `bool` | `True` if text resembles natural language, `False` if it resembles code or structured markup |

Decision criteria include:

- fenced code delimiter prevalence
- code-like line ratio
- alphabetic character proportion
- total code delimiter count

---

### 8. JSON Detection (`Json_Detection.py`)

**Function:** `has_json(text) -> bool`

Detects whether text contains a JSON object/array, either fenced in a ` ```json ` code block or as raw brace/bracket-delimited text that parses successfully. Used to check format compliance when a prompt explicitly requests JSON output.

| Feature | Type   | Description                                                                       |
| ------- | ------ | ------------------------------------------------------------------------------------ |
| return  | `bool` | `True` if a fenced or raw JSON object/array is present and parses to a dict/list |

A bare string, number, or boolean parses as valid JSON but does not count — only structured objects/arrays qualify.

---

### 9. Length Request Detection (`Length_Request_Detection.py`)

**Function:** `extract_length_request_features(text) -> dict`

Infers whether a user prompt explicitly requests a brief or a detailed response, using regex heuristics. Used to check whether a response's actual length (`Writing_Style_Features.word_count`) complies with what was asked for. **Prompt-only** — not meaningful on responses.

| Feature              | Type   | Description                                                                    |
| --------------------- | ------ | -------------------------------------------------------------------------------- |
| `requests_brief`      | `bool` | Prompt asks for brevity ("briefly", "tl;dr", "in one sentence")                |
| `requests_detailed`   | `bool` | Prompt asks for depth ("in detail", "comprehensive", "walk me through")        |

The two flags are independent booleans (not mutually exclusive) since a prompt can, in noisy real-world data, contain contradictory or overlapping phrasing.

---

### 10. List Detection (`List_Detection.py`)

**Functions:**
`detect_list(text) -> bool`
`count_list_items(text) -> int`

Detects Markdown-style ordered and unordered list items via line-prefix heuristics. Independent of the dataset-provided `conv_metadata` list counts (see `Metadata_Features/Lists.py`), so it can be applied directly to raw text such as `user_prompt`, where no metadata list counts exist.

| Feature           | Type   | Description                                                     |
| ------------------ | ------ | ------------------------------------------------------------------ |
| `detect_list`      | `bool` | Whether text contains at least one Markdown list item line       |
| `count_list_items` | `int`  | Total number of ordered + unordered list item lines              |

---

### 11. Near-Empty Detection (`Near_Empty_Detection.py`)

**Function:** `is_near_empty(text, word_count_threshold=3, char_count_threshold=10) -> bool`

Detects empty, whitespace-only, or degenerate responses reduced to a handful of words — a strong quality/loser signal independent of content correctness, typically indicating a truncated or failed generation rather than a legitimately terse answer.

| Feature | Type   | Description                                                                                                                          |
| ------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| return  | `bool` | `True` if text is empty/whitespace-only, has no alphanumeric content, or falls at/below both the word- and character-count thresholds |

Requiring both thresholds avoids misclassifying legitimately short but complete answers (a single long word, a short numeric answer) as near-empty.

---

### 12. Paragraph Count (`Paragraph_Count.py`)

**Function:** `extract_paragraph_features(text) -> dict`

Segments text by double-newline boundaries and counts non-empty paragraphs.

| Feature | Type | Description |
|---------|------|-------------|
| `paragraph_count` | `int` | Number of paragraphs delimited by blank lines |

---

### 13. Paragraph Statistics (`Paragraph_Statistics.py`)

**Function:** `extract_paragraph_statistics(text) -> dict`

Computes paragraph-length distribution statistics using word-count-based paragraph segmentation.

| Feature | Type | Description |
|---------|------|-------------|
| `paragraph_length_mean` | `float` | Mean words per paragraph |
| `paragraph_length_std` | `float` | Population standard deviation of paragraph word counts |

---

### 14. Punctuation Count (`Punctuation_Count.py`)

**Function:** `count_punctuation(text) -> int`

Counts common punctuation marks as a lightweight proxy for writing structure density.

| Feature | Type | Description |
|---------|------|-------------|
| return | `int` | Frequency of `.`, `,`, `!`, `?`, `;`, `:`, `(`, `)`, `[`, `]`, `{`, `}`, `"`, `'`, `` ` ``, `…` |

---

### 15. Refusal Detection (`Refusal_Detection.py`)

**Function:** `has_refusal(text) -> bool`

Detects refusal, apology, or inability-to-comply language in model responses using regex heuristics gated on refusal-object context (to avoid false positives on hedging like "I can't stress enough..."). Refusals are a strong quality signal independent of content correctness — a response that declines the request is a common loser in pairwise human preference, and both sides refusing is a signal for "both_bad" outcomes.

| Feature | Type   | Description                                                                                                                            |
| ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| return  | `bool` | `True` if text contains refusal phrasing: inability statements ("I can't help with that"), explicit declines, AI self-identification disclaimers, apology-prefaced refusals, access/permission denial, or policy/comfort framing |

---

### 16. Repetition Density (`Repetition.py`)

**Function:** `compute_repetition_density(text) -> float`

Estimates lexical repetition using inverse type-token ratio.

| Feature | Type | Description |
|---------|------|-------------|
| return | `float` | Proportion of repeated tokens: `(total_words - unique_words) / total_words`, in range `[0, 1]` |

Computed case-insensitively with Unicode-aware word boundaries.

---

### 17. Script Detection (`Script_Detection.py`)

**Function:** `detect_script(text) -> str`

Classifies text by dominant Unicode script (writing system) using character-range heuristics. Serves as a cheap, dependency-free fallback for language comparison when full language detection is unavailable, misfires, or returns "unknown" — script mismatches are still a strong signal that a response answered in the wrong language relative to the prompt.

| Feature | Type  | Description                                                                                    |
| ------- | ----- | -------------------------------------------------------------------------------------------------- |
| return  | `str` | One of `"arabic"`, `"cjk"`, `"cyrillic"`, `"latin"`, `"other"` — first matching script, checked in that order |

---

### 18. Sentence Count (`Sentence_Count.py`)

**Function:** `count_sentences(text) -> int`

Estimates sentence count by splitting on sentence-ending punctuation and newlines.

| Feature | Type | Description |
|---------|------|-------------|
| return | `int` | Number of detected sentences |

---

### 19. Sentence Length Features (`Sentence_Length.py`)

**Function:** `extract_sentence_length_features(text) -> dict`

Classifies sentences into short and long categories.

| Feature | Type | Description |
|---------|------|-------------|
| `long_sentence_count` | `int` | Number of sentences with 30 or more words |
| `short_sentence_count` | `int` | Number of sentences with 10 or fewer words |

---

### 20. Sentence Length Standard Deviation (`Sentence_Length_Stats.py`)

**Function:** `compute_sentence_length_std(text) -> float`

Computes the population standard deviation of sentence lengths.

| Feature | Type | Description |
|---------|------|-------------|
| `sentence_length_std` | `float` | Dispersion measure of sentence word counts |

---

### 21. Sentence-Per-Paragraph Dispersion (`Sentence_Paragraph_Stats.py`)

**Function:** `compute_sentence_per_paragraph_std(text) -> float`

Computes the population standard deviation of sentence counts across paragraphs.

| Feature | Type | Description |
|---------|------|-------------|
| `sentence_per_paragraph_std` | `float` | Structural consistency measure across paragraphs |

---

### 22. Table Detection (`Table_Detection.py`)

**Functions:**

```python
detect_tables(text) -> bool
count_tables(text) -> int
````

Detects and counts structured tabular content in Markdown and HTML formats.

| Feature         | Type   | Description                                             |
| --------------- | ------ | --------------------------------------------------------- |
| `detect_tables` | `bool` | Whether any Markdown or HTML table structure is present |
| `count_tables`  | `int`  | Total number of detected table structures               |

---

### 23. Token Features (`Token_Features.py`)

**Function:** `count_tokens(text, average_chars_per_token=4) -> int`

Provides adaptive token count estimation without external tokenizers.

The estimator uses language-group-specific character-to-token ratios.

| Feature | Type  | Description                                |
| ------- | ----- | ------------------------------------------- |
| return  | `int` | Estimated token count (rounded, minimum 1) |

Character-to-token ratios:

* **Latin:** 3.8 characters per token
* **Persian/Arabic:** 2.4 characters per token
* **Chinese:** 1.2 characters per token
* **Other:** configurable fallback (default 4.0)

---

### 24. Truncation Detection (`Truncation_Detection.py`)

**Function:** `is_truncated(text) -> bool`

Detects responses that appear cut off before completion — distinct from near-emptiness, since a long, otherwise-substantive answer can still be truncated (unclosed code fence, sentence stopped mid-word). Detection is deliberately conservative to avoid flagging legitimately complete responses that don't end in a period (code blocks, tables, lists, headers, questions).

| Feature | Type   | Description                                                                                                                                                           |
| ------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| return  | `bool` | `True` if the text has an unclosed fenced code block (odd `` ``` ``/`~~~` count), or its final line ends mid-sentence on a plausibly-cut alphabetic word rather than punctuation or a recognized structural line |

Empty/whitespace-only input is not considered truncated (that is near-emptiness, a separate signal — see `Near_Empty_Detection`).

---

### 25. Writing Style Features (`Writing_Style_Features.py`)

**Function:** `extract_writing_style_features(text) -> dict`

Extracts high-level structural and reasoning-related writing style signals.

| Feature                  | Type    | Description                                                                                |
| ------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `word_count`             | `int`   | Whitespace-delimited word count                                                            |
| `sentence_count`         | `int`   | Sentence count estimated using punctuation splitting                                       |
| `avg_words_per_sentence` | `float` | Mean words per sentence                                                                    |
| `is_detailed`            | `bool`  | `True` if `avg_words_per_sentence > 15` and `word_count > 50`                              |
| `has_step_by_step`       | `bool`  | Detects structured explanations through enumerations, transitions, and procedural patterns |

---

## Category Summary

| Category                    | Modules                                                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Structural**               | `Paragraph_Count`, `Paragraph_Statistics`, `Sentence_Count`, `Sentence_Length`, `Sentence_Length_Stats`, `Sentence_Paragraph_Stats`, `List_Detection`      |
| **Code & Technical**         | `Code_Delimiter_Count`, `Is_Code_Block`, `Table_Detection`, `Has_Latex`, `Json_Detection`                                                                  |
| **Linguistic & Lexical**     | `Repetition`, `Token_Features`, `Is_Natural_Text`, `Script_Detection`                                                                                      |
| **Stylistic**                | `Writing_Style_Features`, `Sentence_Length`, `Sentence_Length_Stats`, `Paragraph_Statistics`, `Punctuation_Count`                                          |
| **Discourse & Interaction**  | `Interaction_Features`, `Writing_Style_Features`                                                                                                            |
| **Symbolic**                 | `Emoji_Features`                                                                                                                                            |
| **Format Compliance (prompt)** | `Format_Request_Detection`, `Length_Request_Detection`                                                                                                   |
| **Quality & Failure Signals** | `Refusal_Detection`, `Near_Empty_Detection`, `Truncation_Detection`                                                                                       |

---

## Usage Examples

### Single Text Feature Extraction

The same extractor functions can be applied to prompts or responses.

```python
from Text_Features.Code_Delimiter_Count import count_code_delimiters
from Text_Features.Emoji_Features import count_emojis, detect_emoji

text = df.loc[0, "user_prompt"]

code_density = count_code_delimiters(text)
emoji_count = count_emojis(text)
has_emoji = detect_emoji(text)
````

---

### Composite Feature Extraction

Multiple feature groups can be extracted from a single text input.

```python
from Text_Features.Interaction_Features import extract_interaction_features
from Text_Features.Writing_Style_Features import extract_writing_style_features

text = df.loc[0, "response_a"]

interaction = extract_interaction_features(text)
style = extract_writing_style_features(text)
```

---

### Quality / Failure Signal Extraction

Refusal, near-emptiness, and truncation are checked directly on model responses.

```python
from Text_Features.Refusal_Detection import has_refusal
from Text_Features.Near_Empty_Detection import is_near_empty
from Text_Features.Truncation_Detection import is_truncated

text = df.loc[0, "response_a"]

refused = has_refusal(text)
degenerate = is_near_empty(text)
cut_off = is_truncated(text)
```

---

### Prompt-Only Format-Compliance Extraction

Format and length requests are inferred from the prompt, then compared against actual response formatting/length.

```python
from Text_Features.Format_Request_Detection import extract_format_request_features
from Text_Features.Length_Request_Detection import extract_length_request_features

prompt = df.loc[0, "user_prompt"]

format_request = extract_format_request_features(prompt)
length_request = extract_length_request_features(prompt)
```

---

### Batch Processing with Pandas

Feature extraction can be independently applied to prompts and both model responses.

```python
from Text_Features.Repetition import compute_repetition_density
from Text_Features.Token_Features import count_tokens

df["prompt_repetition_density"] = (
    df["user_prompt"]
    .apply(compute_repetition_density)
)

df["a_repetition_density"] = (
    df["response_a"]
    .apply(compute_repetition_density)
)

df["b_repetition_density"] = (
    df["response_b"]
    .apply(compute_repetition_density)
)
```

---

## Integration with Feature Runner

For full pipeline execution, use the unified feature runner:

```python
from Feature_Extraction.Feature_Runners.Run_Text_Features import run_all_Text_Features

df = run_all_Text_Features(df)
```

The runner automatically extracts features from:

* `user_prompt`
* `response_a`
* `response_b`

and generates separate prefixed feature groups. It additionally attaches `primary_language` and `is_multilingual` (via `Language_Detection.api`) alongside `script` (via `Script_Detection`) for each source. The prompt-only `Format_Request_Detection` and `Length_Request_Detection` outputs are computed once, against `user_prompt`, and prefixed `prompt_` — they have no `a_`/`b_` counterparts.

---

# Output Schema

After running the complete text feature extraction pipeline, the resulting DataFrame contains the original columns plus extracted features.

The naming convention follows:

```
{source}_{feature_name}
```

where:

* `source = prompt` for `user_prompt`
* `source = a` for `response_a`
* `source = b` for `response_b`

---

## Prompt Features (`prompt_` prefix)

Examples:

```
prompt_primary_language
prompt_is_multilingual
prompt_script
prompt_code_delimiters
prompt_punctuation_count
prompt_has_latex
prompt_has_emoji
prompt_emoji_count
prompt_token_count
prompt_repetition_density
prompt_is_code_block
prompt_is_natural_text
prompt_sentence_count
prompt_has_table
prompt_table_count
prompt_has_list
prompt_list_item_count
prompt_has_json
prompt_paragraph_count
prompt_paragraph_length_mean
prompt_paragraph_length_std
prompt_long_sentence_count
prompt_short_sentence_count
prompt_sentence_length_std
prompt_sentence_per_paragraph_std
prompt_has_question_at_end
prompt_has_conclusion
prompt_has_next_steps
prompt_has_interaction_prompt
prompt_interaction_score
prompt_word_count
prompt_avg_words_per_sentence
prompt_is_detailed
prompt_has_step_by_step
prompt_has_refusal
prompt_is_near_empty
prompt_is_truncated
prompt_requests_list
prompt_requests_table
prompt_requests_json
prompt_requests_code
prompt_requests_brief
prompt_requests_detailed
```

---

## Response A Features (`a_` prefix)

Examples:

```
a_primary_language
a_is_multilingual
a_script
a_code_delimiters
a_punctuation_count
a_has_latex
a_has_emoji
a_emoji_count
a_token_count
a_repetition_density
a_is_code_block
a_is_natural_text
a_sentence_count
a_has_table
a_table_count
a_has_list
a_list_item_count
a_has_json
a_paragraph_count
a_paragraph_length_mean
a_paragraph_length_std
a_long_sentence_count
a_short_sentence_count
a_sentence_length_std
a_sentence_per_paragraph_std
a_has_question_at_end
a_has_conclusion
a_has_next_steps
a_has_interaction_prompt
a_interaction_score
a_word_count
a_avg_words_per_sentence
a_is_detailed
a_has_step_by_step
a_has_refusal
a_is_near_empty
a_is_truncated
```

`a_` has no `requests_*` fields — those are prompt-only.

---

## Response B Features (`b_` prefix)

Response B follows the identical schema:

```
b_primary_language
b_is_multilingual
b_script
b_code_delimiters
b_punctuation_count
b_has_latex
b_has_emoji
b_emoji_count
b_token_count
b_repetition_density
b_is_code_block
b_is_natural_text
b_sentence_count
b_has_table
b_table_count
b_has_list
b_list_item_count
b_has_json
b_has_refusal
b_is_near_empty
b_is_truncated
...
```

---

# Design Principles

## Dependency-Light Implementation

All extractors rely primarily on lightweight Python libraries:

* `re`
* `math`
* `json`
* `typing`
* standard Python utilities

No transformer models, embedding APIs, or heavy NLP frameworks are required.

The module is designed for large-scale processing of datasets such as LM-Arena.

---

## Robust Input Handling

Every extractor defensively handles:

* `None`
* `NaN`
* `pd.NA`
* empty strings
* whitespace-only strings
* non-string input types

All functions return stable outputs to prevent pipeline failures during large-scale extraction.

---

## Stateless and Parallelizable

Each extractor is implemented as an independent function:

* no shared state
* no persistent memory
* no dependency between extractors

This enables:

* unit testing
* multiprocessing
* distributed processing
* efficient `pandas.apply` execution

---

## Source-Agnostic Feature Extraction

The extraction logic is independent of the text source.

The same feature extractor can analyze:

* user instructions
* model-generated responses
* retrieved documents
* generated summaries
* other textual artifacts

Only the output prefix changes according to the input source, with the exception of the two prompt-only format-compliance extractors (`Format_Request_Detection`, `Length_Request_Detection`), which are semantically meaningful only against `user_prompt`.

---

## ML-Ready Output

All generated features are returned as:

* `int`
* `float`
* `bool`
* `str`

or flat dictionaries containing these primitive values.

The output can directly be used in:

* LightGBM
* XGBoost
* linear models
* ranking models
* neural classifiers

without additional feature encoding.

---

# Intended Use Cases

## Pairwise Preference Modeling

Extract features independently from:

* `response_a`
* `response_b`

to model:

* human preference
* response quality difference
* stylistic divergence
* structural differences

---

## Prompt-Response Analysis

Analyze relationships between:

* user request complexity
* response structure
* response style
* model behavior

by comparing:

```
prompt_*  →  a_*
prompt_*  →  b_*
```

---

## Format & Length Compliance Checking

Compare what a prompt explicitly requested against what a response actually delivered:

```
prompt_requests_code    →  a_is_code_block / b_is_code_block
prompt_requests_table   →  a_has_table / b_has_table
prompt_requests_list    →  a_has_list / b_has_list
prompt_requests_json    →  a_has_json / b_has_json
prompt_requests_brief   →  a_word_count / b_word_count
prompt_requests_detailed → a_word_count / b_word_count
```

---

## Behavioral Profiling

Characterize LLM outputs according to:

* verbosity
* formatting style
* code usage
* mathematical content
* conversational engagement
* structural organization

---

## Failure / Quality Signal Detection

Flag responses that are likely losers in pairwise comparison independent of content correctness:

* `has_refusal` — the model declined the request
* `is_near_empty` — the response is empty or degenerate
* `is_truncated` — the response was cut off before completion

---

## Reward Model Feature Engineering

Provide interpretable signals for preference models and reward modeling pipelines.

Examples:

* response length difference
* structural complexity difference
* interaction score difference
* formatting preference signals
* format/length compliance signals
* refusal / truncation / near-empty signals

---

# Notes

* Token counts are heuristic estimates based on character-to-token ratios calibrated by script groups and are not exact tokenizer outputs.
* Sentence and paragraph segmentation rely on regex-based heuristics and may differ from linguistic parsers in edge cases.
* Language detection in token estimation is script-based and does not represent full semantic language identification.
* `Script_Detection` is a lightweight Unicode-range heuristic intended as a fallback signal alongside full language detection (`Language_Detection.api`), not a replacement for it.
* Regex-based detectors prioritize speed and interpretability over exhaustive linguistic coverage.
* `Refusal_Detection`, `Format_Request_Detection`, and `Length_Request_Detection` gate common phrases on surrounding context (e.g. "I can't" only counts as refusal near a refusal object) to reduce false positives from hedging or code-related language.
* `Truncation_Detection` is deliberately conservative: it will not flag legitimately complete responses that end on code fences, list items, table rows, headings, or terminal punctuation.
* The feature extraction pipeline operates on copies of DataFrames and does not mutate the original input.
* The unified runner applies the same feature extraction logic consistently across prompts and responses, except for the prompt-only format-compliance extractors.
