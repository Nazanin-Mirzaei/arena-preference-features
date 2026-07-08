# Feature Extraction System

## Overview

This module implements a **modular, scalable, and dependency-light feature extraction system** for large-scale analysis of LLM conversational data. It is designed for the LM-Arena 140K dataset and extracts structural, linguistic, formatting, language, and metadata-level signals from conversational data.

The system extracts features from three main sources:

- **Conversation Metadata (`conv_metadata`)**
- **User Prompts (`user_prompt`)**
- **Assistant Responses (`response_a`, `response_b`)**

These features provide interpretable ML-ready representations for downstream tasks such as pairwise preference modeling, response quality analysis, reward modeling, and LLM behavioral profiling.

The system consists of three coordinated layers:

1. Metadata feature extractors
2. Text feature extractors
3. Feature orchestration runners

Together, they transform raw conversational data into a flat feature representation.

---

## Repository Structure

```

Feature_Extraction/
│
├── Metadata_Features/              # Conversation-level metadata extractors
│   ├── Bold.py
│   ├── Conversation.py
│   ├── Headers.py
│   ├── Lists.py
│   ├── Tokens.py
│   ├── Dataset_Baseline.py
│   └── README.md
│
├── Text_Features/                  # Raw text feature extractors
│   ├── Code_Delimiter_Count.py
│   ├── Emoji_Features.py
│   ├── Has_Latex.py
│   ├── Interaction_Features.py
│   ├── Is_Code_Block.py
│   ├── Is_Natural_Text.py
│   ├── Paragraph_Count.py
│   ├── Paragraph_Statistics.py
│   ├── Punctuation_Count.py
│   ├── Repetition.py
│   ├── Sentence_Count.py
│   ├── Sentence_Length.py
│   ├── Sentence_Length_Stats.py
│   ├── Sentence_Paragraph_Stats.py
│   ├── Table_Detection.py
│   ├── Token_Features.py
│   ├── Writing_Style_Features.py
│   └── README.md
│
├── Feature_Runners/                # Execution pipelines
│   ├── Run_Metadata_Features.py
│   ├── Run_Text_Features.py
│   ├── Run_All_Features.py
│   └── README.md
│
└── README.md

```

---

# System Architecture

The feature extraction system is organized into three hierarchical layers:

```

┌─────────────────────────────────────────────────────────┐
│                  Feature_Runners                        │
│ Batch processing, column handling, feature merging      │
├──────────────────────┬──────────────────────────────────┤
│ Metadata_Features    │        Text_Features             │
│  (conv_metadata)     │ (prompt + response texts)         │
├──────────────────────┴──────────────────────────────────┤
│              Individual Feature Extractors              │
│       Stateless functions operating on single input     │
└─────────────────────────────────────────────────────────┘

````

---

# Architectural Principles

## Layer 1 — Extractors

Each module inside `Metadata_Features` and `Text_Features` provides one or more pure feature extraction functions.

Inputs:

- Metadata dictionary (`conv_metadata`)
- Raw text (`user_prompt`, `response_a`, `response_b`)

Outputs:

- Scalar values
- Flat dictionaries of interpretable features

All extractors are stateless and independently testable.

---

## Layer 2 — Pipelines (`Feature_Runners`)

Runner modules apply feature extractors across DataFrames.

Responsibilities:

- Input validation
- Missing value handling
- Batch feature extraction
- Output flattening
- Feature naming conventions

---

## Layer 3 — Unified Entry Point (`Run_All_Features`)

A single pipeline that executes available feature extraction modules depending on existing DataFrame columns.

Supported inputs:

- `conv_metadata`
- `user_prompt`
- `response_a`
- `response_b`

---

# Feature Categories

## Metadata Features

Extracted from `conv_metadata`.

| Category | Example Features | Source |
|---|---|---|
| Formatting | `bold_total`, `bold_style_preference`, `emphasis_intensity` | Bold.py |
| Conversation Dynamics | `turns`, `is_multi_turn`, `turn_density` | Conversation.py |
| Document Structure | `total_headers`, `header_depth_score` | Headers.py |
| List Usage | `total_list_items`, `ordered_ratio`, `unordered_ratio` | Lists.py |
| Token Allocation | `assistant_tokens_total`, `user_tokens`, `context_tokens_total` | Tokens.py |
| Dataset Baseline | Dataset-provided formatting/token statistics | Dataset_Baseline.py |

---

# Text Features

Extracted independently from:

- `user_prompt`
- `response_a`
- `response_b`

The same feature extraction pipeline is applied to all text fields.

Generated columns use prefixes:

- `prompt_`
- `a_`
- `b_`

---

## Text Feature Categories

| Category | Example Features |
|---|---|
| Language | `primary_language`, `is_multilingual` |
| Structural | `paragraph_count`, `sentence_count`, `sentence_length_std` |
| Code & Technical | `code_delimiters`, `is_code_block`, `has_latex` |
| Lexical | `token_count`, `repetition_density`, `is_natural_text` |
| Style | `word_count`, `avg_words_per_sentence`, `is_detailed` |
| Interaction | `interaction_score`, `has_conclusion`, `has_next_steps` |
| Symbolic | `has_emoji`, `emoji_count` |

---

# Usage

## Full Pipeline

```python
from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = run_all_features(df)
````

---

## Metadata Features Only

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
```

---

## Text Features Only

```python
from Feature_Extraction.Feature_Runners.Run_Text_Features import run_all_text_features

df = run_all_text_features(df)
```

---

## Single Text Feature Extraction

Example:

```python
from Feature_Extraction.Text_Features.Repetition import compute_repetition_density

score = compute_repetition_density(text)
```

---

# Output Schema

After running `run_all_features`, the DataFrame contains:

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

Prefix:

```
prompt_
```

Example:

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

Prefix:

```
a_
```

Example:

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

Prefix:

```
b_
```

Same schema as Response A.

---

# Design Principles

## Dependency-Light Architecture

The system mainly relies on:

* Python standard library
* pandas
* mediapipe (only for language detection)

No transformer models or embedding APIs are required.

---

## Modular and Extensible

New features can be added by:

1. Creating a standalone extractor
2. Returning a scalar or dictionary output
3. Registering it in the corresponding runner

---

## Robust Input Handling

All extractors handle:

* None values
* NaN values
* Empty strings
* Missing metadata fields
* Invalid input formats

---

## Consistent Naming Convention

Feature prefixes indicate the source:

| Prefix    | Source      |
| --------- | ----------- |
| `prompt_` | User prompt |
| `a_`      | Response A  |
| `b_`      | Response B  |

This enables direct feature comparison and pairwise modeling.

---

# Intended Use Cases

* Pairwise preference modeling
* Reward model feature engineering
* Response quality analysis
* Prompt-response behavior analysis
* LLM output profiling
* Feature-based ML pipelines

---

# Notes

* Text features operate on raw text inputs and are shared between prompts and responses.
* Language detection is applied independently to prompts and responses.
* Metadata features operate only on `conv_metadata`.
* Token estimation is heuristic and not based on exact model tokenizers.
* Sentence and paragraph segmentation use regex-based heuristics.
* All runners operate on copies of DataFrames and do not mutate original inputs.
