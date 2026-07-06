# Feature Runners: Execution Pipeline

## Overview

This module provides the **orchestration layer** for the feature extraction system, responsible for coordinating the application of individual feature extractors across DataFrames. Each runner implements a batch-processing pipeline that parses raw inputs, applies a registry of extractors, and merges the resulting features into a flat, columnar format suitable for downstream ML workflows.

The runners abstract away the complexity of input parsing, nested output flattening, and column-naming conventions, exposing a clean API over the underlying extractor modules.

---

## Module Structure

```
Feature_Runners/
│
├── Run_Metadata_Features.py     # Metadata feature extraction pipeline
├── Run_Response_Features.py     # Response A/B feature extraction pipeline
├── Run_All_Features.py          # Unified pipeline (metadata + response)
│
└── README.md
```

---

## Pipeline Components

### 1. Run Metadata Features (`Run_Metadata_Features.py`)

**Function:** `run_all_metadata_features(df) -> pd.DataFrame`

Orchestrates the extraction of all conversation-level metadata features from the `conv_metadata` column.

#### Architecture

The pipeline follows a **parse-once, apply-all** pattern for efficiency:

1. **Parsing:** Each row's `conv_metadata` is parsed once via `ast.literal_eval` (if stored as a string) and cached.
2. **Registry dispatch:** A feature registry (`METADATA_FEATURES` list) iterates over all registered extractors, applying each to the parsed metadata.
3. **Flattening:** Each extractor's dictionary output is expanded into individual columns via `pd.Series`.
4. **Concatenation:** All feature DataFrames are merged with the original DataFrame along the column axis.

#### Registered Extractors

| Extractor | Source Module | Output Columns |
|-----------|--------------|----------------|
| `extract_bold_features` | `Metadata_Features.Bold` | `bold_total`, `bold_style_preference`, `emphasis_intensity` |
| `extract_conversation_dynamics` | `Metadata_Features.Conversation` | `turns`, `is_multi_turn`, `turn_density` |
| `extract_header_features` | `Metadata_Features.Headers` | `total_headers`, `header_depth_score` |
| `extract_list_features` | `Metadata_Features.Lists` | `total_list_items`, `ordered_ratio`, `unordered_ratio` |
| `extract_token_features` | `Metadata_Features.Tokens` | `assistant_tokens_total`, `user_tokens`, `context_tokens_total`, `assistant_token_ratio` |
| `extract_dataset_format_features` | `Metadata_Features.Dataset_Baseline` | `dataset_a_bold`, `dataset_b_bold`, `dataset_a_list_ordered`, … (18 columns) |

#### Usage

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
```

---

### 2. Run Response Features (`Run_Response_Features.py`)

**Function:** `run_all_response_features(df) -> pd.DataFrame`

Orchestrates the extraction of all textual and structural features from both `response_a` and `response_b` columns, producing a parallel set of A-prefixed and B-prefixed feature columns.

#### Architecture

The pipeline follows a **dual-apply, prefix-merge** pattern:

1. **Composite extractor:** A single `extract_all_response_features(text)` function aggregates all 17 individual extractors into one dictionary-returning callable.
2. **Dual dispatch:** The composite extractor is applied independently to `response_a` and `response_b`.
3. **Prefix assignment:** Each feature set receives a prefix (`a_` or `b_`) to enable pairwise comparison.
4. **Concatenation:** Both feature sets are merged with the original DataFrame.

#### Composite Feature Set

The `extract_all_response_features` function produces the following dictionary (after `safe_update` merges from dict-returning extractors):

| Key | Type | Source |
|-----|------|--------|
| `primary_language` | `str` | `Language_Detection.api.detectLanguage` |
| `is_multilingual` | `bool` | `Language_Detection.api.detectMultiLanguage` |
| `code_delimiters` | `int` | `Code_Delimiter_Count.count_code_delimiters` |
| `punctuation_count` | `int` | `Punctuation_Count.count_punctuation` |
| `sentence_count` | `int` | `Sentence_Count.count_sentences` |
| `has_latex` | `bool` | `Has_Latex.has_latex` |
| `has_emoji` | `bool` | `Emoji_Features.detect_emoji` |
| `emoji_count` | `int` | `Emoji_Features.count_emojis` |
| `token_count` | `int` | `Token_Features.count_tokens` |
| `repetition_density` | `float` | `Repetition.compute_repetition_density` |
| `is_code_block` | `bool` | `Is_Code_Block.is_code_block` |
| `is_natural_text` | `bool` | `Is_Natural_Text.is_natural_text` |
| `has_table` | `bool` | `Table_Detection.detect_tables` |
| `table_count` | `int` | `Table_Detection.count_tables` |
| `paragraph_count` | `int` | `Paragraph_Count.extract_paragraph_features` |
| `paragraph_length_mean` | `float` | `Paragraph_Statistics.extract_paragraph_statistics` |
| `paragraph_length_std` | `float` | `Paragraph_Statistics.extract_paragraph_statistics` |
| `long_sentence_count` | `int` | `Sentence_Length.extract_sentence_length_features` |
| `short_sentence_count` | `int` | `Sentence_Length.extract_sentence_length_features` |
| `sentence_length_std` | `float` | `Sentence_Length_Stats.compute_sentence_length_std` |
| `sentence_per_paragraph_std` | `float` | `Sentence_Paragraph_Stats.compute_sentence_per_paragraph_std` |
| `has_question_at_end` | `bool` | `Interaction_Features.extract_interaction_features` |
| `has_conclusion` | `bool` | `Interaction_Features.extract_interaction_features` |
| `has_next_steps` | `bool` | `Interaction_Features.extract_interaction_features` |
| `has_interaction_prompt` | `bool` | `Interaction_Features.extract_interaction_features` |
| `interaction_score` | `int` | `Interaction_Features.extract_interaction_features` |
| `word_count` | `int` | `Writing_Style_Features.extract_writing_style_features` |
| `avg_words_per_sentence` | `float` | `Writing_Style_Features.extract_writing_style_features` |
| `is_detailed` | `bool` | `Writing_Style_Features.extract_writing_style_features` |
| `has_step_by_step` | `bool` | `Writing_Style_Features.extract_writing_style_features` |

After prefixing, each column appears as `a_<name>` and `b_<name>`, enabling direct differential analysis.

#### Usage

```python
from Feature_Extraction.Feature_Runners.Run_Response_Features import run_all_response_features

df = run_all_response_features(df)
```

---

### 3. Run All Features (`Run_All_Features.py`)

**Function:** `run_all_features(df) -> pd.DataFrame`

A unified pipeline that conditionally executes both metadata and response feature extractors based on column availability.

#### Behavior

1. If the DataFrame contains a `conv_metadata` column, metadata features are extracted via `run_all_metadata_features`.
2. If the DataFrame contains both `response_a` and `response_b` columns, response features are extracted via `run_all_response_features`.
3. Both conditions can be satisfied simultaneously, producing a fully featured DataFrame.

#### Usage

```python
from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = run_all_features(df)
```

---

## Output Schema

After running the full pipeline via `run_all_features`, the output DataFrame contains:

### Metadata Features (no prefix)

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
dataset_a_bold
dataset_b_bold
...
```

### Response A Features (prefixed `a_`)

```
a_primary_language
a_is_multilingual
a_code_delimiters
a_punctuation_count
a_sentence_count
a_has_latex
a_has_emoji
a_emoji_count
a_token_count
a_repetition_density
a_is_code_block
a_is_natural_text
a_has_table
a_table_count
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
```

### Response B Features (prefixed `b_`)

Identical schema to response A, with `b_` prefix.

---

## Design Principles

### Column-Gated Execution

Each runner checks for the existence of required input columns before applying extraction. This enables the unified pipeline to gracefully handle DataFrames with partial column availability without raising errors.

### Parse-Once Optimization

Metadata features share a common parsing step (`ast.literal_eval`). The pipeline performs this parsing exactly once per row, then dispatches the parsed dictionary to all registered extractors, avoiding redundant deserialization.

### Extensible Registry Pattern

Metadata features are registered in a list (`METADATA_FEATURES`), making it straightforward to add or remove extractors without modifying pipeline logic. Response features use a composite function pattern where new extractors can be incorporated into `extract_all_response_features`.

### Safe Nested Output Handling

The `safe_update` utility in `Run_Response_Features.py` ensures that extractors returning dictionaries are safely merged into the feature collection, while extractors returning scalars are assigned directly. This provides robustness against inconsistent return types across extractors.

### Dual-Response Naming Convention

Response features are systematically prefixed with `a_` and `b_`, preserving a consistent naming convention that enables:
- Column-wise differential analysis (`df["a_token_count"] - df["b_token_count"]`)
- Automatic feature pairing for pairwise models
- Clear provenance tracking for each feature

---

## Intended Use Cases

- **End-to-end feature extraction:** Apply the full pipeline to a raw LM-Arena dataset and produce a feature-rich DataFrame for downstream modeling.
- **Modular extraction:** Choose between metadata-only, response-only, or combined extraction depending on the analytical target.
- **Batch processing at scale:** Each runner operates entirely via `pandas.apply` and is compatible with parallelization backends (e.g., `pandarallel`, `swifter`, `dask`).
- **Pipeline integration:** The uniform return type (`pd.DataFrame`) allows these runners to be composed into larger data processing workflows.

---

## Notes

- The `extract_all_response_features` function in `Run_Response_Features.py` depends on `Language_Detection.api` for language identification. This is the only external module dependency in the response pipeline.
- `Run_Response_Features.py` contains a duplicate definition of `safe_update` (lines 24–28 and 33–35); the second definition silently overrides the first at module load time. Both implementations are functionally identical.
- Metadata parsing uses `ast.literal_eval`, which safely evaluates string representations of Python dictionaries. It is restricted to literal expressions and does not execute arbitrary code.
- All runners operate on a copy of the input DataFrame to avoid mutating the caller's data.
