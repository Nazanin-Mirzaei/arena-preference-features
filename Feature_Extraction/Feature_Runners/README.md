# Feature Runners: Execution Pipeline

## Overview

This module provides the **orchestration layer** for the feature extraction system, responsible for coordinating the application of individual feature extractors across DataFrames. Each runner implements a batch-processing pipeline that parses raw inputs, applies a registry of extractors, and merges the resulting features into a flat, columnar format suitable for downstream ML workflows.

The runners abstract away the complexity of input parsing, nested output flattening, and column-naming conventions, exposing a clean API over the underlying extractor modules.

The pipeline currently supports three feature groups:

- **Metadata Features:** Extracted from conversation-level metadata (`conv_metadata`)
- **Text Features:** Extracted from raw textual content (`user_prompt`, `response_a`, `response_b`)
- **Unified Feature Pipeline:** Combines metadata and text-based features into a single DataFrame

---

## Module Structure

```

Feature_Runners/
│
├── Run_Metadata_Features.py     # Metadata feature extraction pipeline
├── Run_Text_Features.py         # Text feature extraction pipeline (prompt + responses)
├── Run_All_Features.py          # Unified pipeline (metadata + text)
│
└── README.md

````

---

# Pipeline Components

## 1. Run Metadata Features (`Run_Metadata_Features.py`)

**Function:** `run_all_metadata_features(df) -> pd.DataFrame`

Orchestrates the extraction of all conversation-level metadata features from the `conv_metadata` column.

### Architecture

The pipeline follows a **parse-once, apply-all** pattern for efficiency:

1. **Parsing:** Each row's `conv_metadata` is parsed once via `ast.literal_eval` (if stored as a string).
2. **Registry dispatch:** A feature registry applies all metadata extractors to the parsed dictionary.
3. **Flattening:** Dictionary outputs are expanded into individual DataFrame columns.
4. **Concatenation:** Extracted features are merged with the original DataFrame.

### Registered Extractors

| Extractor | Source Module | Output Features |
|-----------|--------------|----------------|
| `extract_bold_features` | `Metadata_Features.Bold` | Bold usage statistics |
| `extract_conversation_dynamics` | `Metadata_Features.Conversation` | Conversation depth and turn statistics |
| `extract_header_features` | `Metadata_Features.Headers` | Markdown heading structure |
| `extract_list_features` | `Metadata_Features.Lists` | Ordered/unordered list statistics |
| `extract_token_features` | `Metadata_Features.Tokens` | Token allocation statistics |
| `extract_dataset_format_features` | `Metadata_Features.Dataset_Baseline` | Dataset-provided formatting fields |

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
````

---

# 2. Run Text Features (`Run_Text_Features.py`)

**Function:** `run_all_text_features(df) -> pd.DataFrame`

Extracts linguistic, structural, formatting, and language-related features from:

* `user_prompt`
* `response_a`
* `response_b`

The same feature extractor is applied independently to each text field.

---

## Architecture

The pipeline follows a **multi-column apply and prefix-merge** pattern:

1. A composite extractor (`extract_all_text_features`) combines all text-based feature extractors.
2. The extractor is applied separately to:

   * `user_prompt`
   * `response_a`
   * `response_b`
3. Generated features receive prefixes:

   * `prompt_`
   * `a_`
   * `b_`
4. All features are concatenated with the original DataFrame.

---

## Composite Feature Set

The `extract_all_text_features()` function extracts:

| Feature                      | Type    | Description                                              |
| ---------------------------- | ------- | -------------------------------------------------------- |
| `primary_language`           | `str`   | Detected dominant language of the text                   |
| `is_multilingual`            | `bool`  | Whether multiple languages are detected                  |
| `code_delimiters`            | `int`   | Count of code delimiters such as backticks and code tags |
| `punctuation_count`          | `int`   | Number of punctuation symbols                            |
| `sentence_count`             | `int`   | Estimated number of sentences                            |
| `has_latex`                  | `bool`  | Presence of mathematical LaTeX notation                  |
| `has_emoji`                  | `bool`  | Whether emojis exist                                     |
| `emoji_count`                | `int`   | Number of emojis                                         |
| `token_count`                | `int`   | Estimated token count                                    |
| `repetition_density`         | `float` | Lexical repetition ratio                                 |
| `is_code_block`              | `bool`  | Presence of code-like blocks                             |
| `is_natural_text`            | `bool`  | Natural language vs code classification                  |
| `has_table`                  | `bool`  | Presence of tables                                       |
| `table_count`                | `int`   | Number of detected tables                                |
| `paragraph_count`            | `int`   | Number of paragraphs                                     |
| `paragraph_length_mean`      | `float` | Average paragraph size                                   |
| `paragraph_length_std`       | `float` | Paragraph size variation                                 |
| `long_sentence_count`        | `int`   | Number of long sentences                                 |
| `short_sentence_count`       | `int`   | Number of short sentences                                |
| `sentence_length_std`        | `float` | Sentence length variation                                |
| `sentence_per_paragraph_std` | `float` | Sentence distribution variation                          |
| `has_question_at_end`        | `bool`  | Ends with a question                                     |
| `has_conclusion`             | `bool`  | Contains conclusion indicators                           |
| `has_next_steps`             | `bool`  | Contains future-action indicators                        |
| `has_interaction_prompt`     | `bool`  | Contains user engagement phrases                         |
| `interaction_score`          | `int`   | Interaction indicator score                              |
| `word_count`                 | `int`   | Number of words                                          |
| `avg_words_per_sentence`     | `float` | Average sentence length                                  |
| `is_detailed`                | `bool`  | Detailed response indicator                              |
| `has_step_by_step`           | `bool`  | Step-by-step structure detection                         |

---

## Output Naming Convention

The extracted features are prefixed based on their source:

### User Prompt Features

```
prompt_primary_language
prompt_token_count
prompt_sentence_count
prompt_is_multilingual
...
```

### Response A Features

```
a_primary_language
a_token_count
a_sentence_count
a_is_multilingual
...
```

### Response B Features

```
b_primary_language
b_token_count
b_sentence_count
b_is_multilingual
...
```

This naming convention allows direct comparison between:

* Prompt and response characteristics
* Response A and response B differences
* Model behavior patterns

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_Text_Features import run_all_text_features

df = run_all_text_features(df)
```

---

# 3. Run All Features (`Run_All_Features.py`)

**Function:** `run_all_features(df) -> pd.DataFrame`

A unified pipeline that executes metadata and text feature extraction based on available columns.

---

## Behavior

1. If `conv_metadata` exists:

   * Metadata features are extracted.

2. If any of the following columns exist:

   * `user_prompt`
   * `response_a`
   * `response_b`

   Text features are extracted.

3. Available feature groups are merged into a single DataFrame.

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = run_all_features(df)
```

---

# Output Schema

After running the full pipeline:

## Metadata Features

```
bold_total
bold_style_preference
emphasis_intensity
turns
is_multi_turn
turn_density
total_headers
header_depth_score
total_list_items
ordered_ratio
unordered_ratio
assistant_tokens_total
user_tokens
context_tokens_total
assistant_token_ratio
...
```

---

## Prompt Features

```
prompt_primary_language
prompt_is_multilingual
prompt_token_count
prompt_sentence_count
prompt_word_count
prompt_is_code_block
prompt_has_latex
prompt_repetition_density
...
```

---

## Response A Features

```
a_primary_language
a_is_multilingual
a_token_count
a_sentence_count
a_word_count
a_is_code_block
a_has_latex
a_repetition_density
...
```

---

## Response B Features

```
b_primary_language
b_is_multilingual
b_token_count
b_sentence_count
b_word_count
b_is_code_block
b_has_latex
b_repetition_density
...
```

---

# Design Principles

## Column-Gated Execution

Each runner checks the required input columns before execution. This allows processing datasets with different available fields without raising unnecessary errors.

---

## Modular Feature Expansion

Each feature group is isolated:

* Metadata features analyze dataset-provided information.
* Text features analyze raw textual content.
* Additional feature groups can be integrated without modifying the entire pipeline.

---

## Safe Feature Aggregation

Dictionary-based outputs are safely merged into flat DataFrame columns, allowing different extractors with different output formats to coexist.

---

## Consistent Naming Convention

Feature prefixes preserve feature provenance:

* `prompt_` → user prompt features
* `a_` → response A features
* `b_` → response B features

This enables:

* Pairwise model training
* Feature difference analysis
* Behavioral comparison between prompts and responses

---

# Intended Use Cases

* **End-to-end feature extraction:** Convert raw LM-Arena data into a structured ML-ready dataset.
* **Prompt-response analysis:** Study relationships between user prompts and generated answers.
* **Pairwise response modeling:** Compare behavioral and structural differences between responses.
* **Reward model feature engineering:** Provide interpretable signals for preference modeling.
* **LLM behavior analysis:** Analyze response style, structure, language, and formatting patterns.

---

# Notes

* Text feature extraction depends on `Language_Detection` for language identification.
* Language detection is applied independently to prompts and responses.
* Metadata extraction operates only on pre-parsed conversation metadata.
* All runners operate on a copy of the input DataFrame and do not mutate the original data.
* Feature extraction is designed to be lightweight and scalable for large datasets.

