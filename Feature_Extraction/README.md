# Feature Extraction System

## Overview

This module implements a **modular, scalable, and dependency-light feature extraction system** for large-scale analysis of LLM-generated conversational data. It is designed for the LM-Arena 140K dataset and extracts structural, linguistic, formatting, and metadata-level signals from model responses to support downstream tasks such as pairwise preference modeling, response quality scoring, and behavioral analysis.

The system comprises three coordinated layers — metadata extractors, response text extractors, and orchestration runners — that together produce a flat, ML-ready feature representation from raw conversational data.

---

## Repository Structure

```
Feature_Extraction/
│
├── Metadata_Features/          # Conversation-level metadata extractors
│   ├── Bold.py                 # Bold formatting frequency and style
│   ├── Conversation.py         # Turn-taking dynamics and density
│   ├── Headers.py              # Markdown heading depth and distribution
│   ├── Lists.py                # Ordered / unordered list segmentation
│   ├── Tokens.py               # Assistant, user, and context token counts
│   ├── Dataset_Baseline.py     # Unified baseline: all dataset-provided fields
│   └── README.md
│
├── Response_Features/          # Raw text feature extractors (A/B responses)
│   ├── Code_Delimiter_Count.py # Code delimiter frequency
│   ├── Emoji_Features.py       # Emoji presence and count
│   ├── Has_Latex.py            # LaTeX / mathematical expression detection
│   ├── Interaction_Features.py # Conversational engagement signals
│   ├── Is_Code_Block.py        # Code block and programming syntax detection
│   ├── Is_Natural_Text.py      # Natural language vs. code classification
│   ├── Paragraph_Count.py      # Paragraph segmentation count
│   ├── Paragraph_Statistics.py # Paragraph length mean and standard deviation
│   ├── Punctuation_Count.py    # Punctuation mark frequency
│   ├── Repetition.py           # Lexical repetition density
│   ├── Sentence_Count.py       # Sentence count estimation
│   ├── Sentence_Length.py      # Short and long sentence classification
│   ├── Sentence_Length_Stats.py# Sentence length standard deviation
│   ├── Sentence_Paragraph_Stats.py # Sentence-per-paragraph dispersion
│   ├── Table_Detection.py      # Markdown / HTML table detection and count
│   ├── Token_Features.py       # Character-based adaptive token estimation
│   ├── Writing_Style_Features.py # Structural style and reasoning signals
│   └── README.md
│
├── Feature_Runners/            # Orchestration and execution pipelines
│   ├── Run_Metadata_Features.py # Metadata extraction pipeline
│   ├── Run_Response_Features.py # Response A/B extraction pipeline
│   ├── Run_All_Features.py      # Unified extraction pipeline
│   └── README.md
│
└── README.md                   # This file
```

---

## System Architecture

The feature extraction system is organized into three hierarchical layers:

```
┌─────────────────────────────────────────────────────────┐
│                   Feature_Runners                        │
│   Orchestration layer: batch processing, input parsing,  │
│   output flattening, column naming conventions           │
├─────────────────────┬───────────────────────────────────┤
│  Metadata_Features  │         Response_Features          │
│  (conv_metadata)    │     (response_a / response_b)      │
├─────────────────────┴───────────────────────────────────┤
│               Individual Feature Extractors                │
│   Stateless, pure functions operating on single inputs     │
└─────────────────────────────────────────────────────────┘
```

### Architectural Principles

**Layer 1 — Extractors (Stateless Functions):** Each Python module in `Metadata_Features` and `Response_Features` exposes one or more pure functions that accept a single input (a metadata dictionary or a raw text string) and return a dictionary or scalar of extracted features. These functions have no side effects, no shared state, and no dependencies beyond the Python standard library (and, for response features, the `Language_Detection` module).

**Layer 2 — Pipelines (`Feature_Runners`):** Orchestration scripts iterate over DataFrames, parse raw inputs, dispatch extractors, and merge nested outputs into flat columnar structures. They handle input validation, missing-value imputation, and naming conventions (e.g., A/B prefixing for response features).

**Layer 3 — Unified Entry Point (`Run_All_Features`):** A single function that conditionally executes metadata and response pipelines based on the presence of required columns in the input DataFrame.

---

## Feature Categories

### Metadata Features

Extracted from the `conv_metadata` dictionary, these features capture conversation-level structural properties:

| Category | Example Features | Source |
|----------|-----------------|--------|
| Formatting | `bold_total`, `bold_style_preference`, `emphasis_intensity` | `Bold.py` |
| Discourse | `turns`, `is_multi_turn`, `turn_density` | `Conversation.py` |
| Document Structure | `total_headers`, `header_depth_score` | `Headers.py` |
| List Usage | `total_list_items`, `ordered_ratio`, `unordered_ratio` | `Lists.py` |
| Token Allocation | `assistant_tokens_total`, `user_tokens`, `context_tokens_total`, `assistant_token_ratio` | `Tokens.py` |
| Dataset Baseline | `dataset_a_bold`, `dataset_b_bold`, per-response header/list/token counts | `Dataset_Baseline.py` |

### Response Features

Extracted independently from `response_a` and `response_b` raw text, these features (each prefixed with `a_` or `b_`) capture textual and structural properties:

| Category | Features | Source Modules |
|----------|----------|---------------|
| Structural | `paragraph_count`, `paragraph_length_mean`, `paragraph_length_std`, `sentence_count`, `sentence_length_std`, `sentence_per_paragraph_std` | `Paragraph_Count`, `Paragraph_Statistics`, `Sentence_Count`, `Sentence_Length_Stats`, `Sentence_Paragraph_Stats` |
| Code & Technical | `code_delimiters`, `is_code_block`, `has_table`, `table_count`, `has_latex` | `Code_Delimiter_Count`, `Is_Code_Block`, `Table_Detection`, `Has_Latex` |
| Lexical | `repetition_density`, `token_count`, `is_natural_text` | `Repetition`, `Token_Features`, `Is_Natural_Text` |
| Stylistic | `word_count`, `avg_words_per_sentence`, `is_detailed`, `has_step_by_step`, `punctuation_count`, `long_sentence_count`, `short_sentence_count` | `Writing_Style_Features`, `Punctuation_Count`, `Sentence_Length` |
| Discourse | `has_question_at_end`, `has_conclusion`, `has_next_steps`, `has_interaction_prompt`, `interaction_score` | `Interaction_Features` |
| Symbolic | `has_emoji`, `emoji_count` | `Emoji_Features` |
| Language | `primary_language`, `is_multilingual` | `Language_Detection.api` |

---

## Usage

### Full Pipeline (Recommended)

```python
import pandas as pd
from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = pd.read_csv("lm_arena_140k.csv")
df = run_all_features(df)

print(df.shape)
print([col for col in df.columns if col.startswith(("a_", "b_", "bold", "turns", "total_"))])
```

### Metadata Features Only

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
```

### Response Features Only

```python
from Feature_Extraction.Feature_Runners.Run_Response_Features import run_all_response_features

df = run_all_response_features(df)
```

### Single Input Extraction

```python
from Feature_Extraction.Response_Features.Repetition import compute_repetition_density
from Feature_Extraction.Response_Features.Token_Features import count_tokens

text = "This is a sample LLM response with some repeated content."

rep = compute_repetition_density(text)   # float in [0, 1]
tok = count_tokens(text)                 # int
```

---

## Output Schema

After running `run_all_features`, the resulting DataFrame contains the original columns plus:

### Metadata Columns (no prefix)

```
bold_total, bold_style_preference, emphasis_intensity,
turns, is_multi_turn, turn_density,
total_headers, header_depth_score,
total_list_items, ordered_ratio, unordered_ratio,
assistant_tokens_total, user_tokens, context_tokens_total, assistant_token_ratio,
dataset_a_bold, dataset_b_bold, dataset_a_list_ordered, dataset_b_list_ordered,
dataset_a_list_unordered, dataset_b_list_unordered,
dataset_a_header_h1 ... dataset_b_header_h6,
dataset_a_tokens, dataset_b_tokens,
dataset_a_context_tokens, dataset_b_context_tokens
```

### Response A Columns (`a_` prefix)

```
a_primary_language, a_is_multilingual,
a_code_delimiters, a_punctuation_count, a_sentence_count, a_has_latex,
a_has_emoji, a_emoji_count,
a_token_count, a_repetition_density,
a_is_code_block, a_is_natural_text,
a_has_table, a_table_count,
a_paragraph_count, a_paragraph_length_mean, a_paragraph_length_std,
a_long_sentence_count, a_short_sentence_count,
a_sentence_length_std, a_sentence_per_paragraph_std,
a_has_question_at_end, a_has_conclusion, a_has_next_steps, a_has_interaction_prompt, a_interaction_score,
a_word_count, a_avg_words_per_sentence, a_is_detailed, a_has_step_by_step
```

### Response B Columns (`b_` prefix)

Identical schema to response A, with `b_` prefix.

---

## Design Principles

### Dependency-Light Architecture

The entire system relies only on `pandas`, the Python standard library (`re`, `math`, `ast`, `typing`), and, for language detection, the `mediapipe` package and `Language_Detection` module. No transformer models, embedding APIs, or heavy NLP frameworks are required.

### Modular and Extensible

Each feature extractor is an independent, self-contained function with a consistent interface. New extractors can be added by:
1. Implementing a function that accepts a single input and returns a dictionary
2. Registering it in `METADATA_FEATURES` (for metadata) or adding it to `extract_all_response_features` (for responses)

### Robust Input Handling

Every extractor defensively guards against `None`, `NaN`, empty strings, malformed data, and type mismatches, returning zero-initialized or neutral defaults to ensure pipeline stability at scale.

### Dual-Response Design

Response features are extracted and named independently for `response_a` and `response_b` using a consistent prefix convention (`a_` / `b_`). This enables direct pairwise comparison, differential analysis, and feature-level preference modeling.

### ML-Ready Output

All features are returned as primitive types (`int`, `float`, `bool`, `str`) or flat dictionaries thereof. No encoding, normalization, or additional transformation is required before ingestion into tree-based models, linear classifiers, or neural networks.

---

## Dependencies

| Dependency | Scope | Required |
|-----------|-------|----------|
| `pandas` | DataFrames, batch processing | Yes |
| Python `re` | Regex-based text analysis | Yes (stdlib) |
| Python `math` | Numerical transformations | Yes (stdlib) |
| Python `ast` | Metadata string parsing | Yes (stdlib) |
| `mediapipe` | Language detection | Optional (only Response Features) |
| `Language_Detection` | Language identification wrapper | Optional (only Response Features) |

---

## Intended Use Cases

- **Pairwise preference modeling:** Extract A/B features to train reward models or classifiers that predict human preference judgments.
- **Response quality scoring:** Use structural and linguistic features as proxies for response quality, verbosity, or formatting effort.
- **Behavioral profiling:** Characterize LLM outputs along dimensions of code usage, reasoning structure, conversational engagement, and stylistic variation.
- **Feature-based ML pipelines:** Serve as input features for LightGBM, XGBoost, or linear models for ranking, classification, or regression tasks.
- **Baseline validation:** Compare module-extracted features against dataset-provided metadata fields using `Dataset_Baseline.py` to ensure consistency.

---

## Notes

- Token count estimates in `Token_Features.py` are heuristic (character-to-token ratios calibrated per script group) and do not reflect exact model tokenizer outputs.
- Sentence and paragraph segmentation uses regex heuristics that may diverge from ground truth for edge cases such as abbreviations, nested quotations, or irregular line breaks.
- The `Language_Detection` dependency is required only for the response features pipeline; metadata features can run independently without it.
- All pipelines operate on copies of the input DataFrame and do not mutate the caller's data.
- The `safe_update` function in `Run_Response_Features.py` ensures safe merging of dictionary-returning extractors; a duplicate definition exists in the source and the second silently overrides the first with identical behavior.
