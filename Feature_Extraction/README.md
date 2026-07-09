# Feature Extraction System

## Overview

This module implements a **modular, scalable, and dependency-light feature extraction system** for large-scale analysis of LLM conversational data. It is designed for the LM-Arena 140K dataset and extracts structural, linguistic, formatting, language, quality/failure, and metadata-level signals from conversational data.

The system extracts features from four sources:

- **Conversation Metadata (`conv_metadata`, `category_tag`)**
- **User Prompts (`user_prompt`)**
- **Assistant Responses (`response_a`, `response_b`)**
- **Pairwise A-vs-B Relationships** (derived from the three sources above)

These features provide interpretable ML-ready representations for downstream tasks such as pairwise preference modeling, response quality analysis, reward modeling, format/length compliance checking, and LLM behavioral profiling.

The system consists of four coordinated layers:

1. Metadata feature extractors
2. Text feature extractors
3. Comparison feature extractors (pairwise A vs. B)
4. Feature orchestration runners

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
│   ├── Category_Tag.py             # category_tag column (not conv_metadata)
│   └── README.md
│
├── Text_Features/                  # Raw text feature extractors
│   ├── Code_Delimiter_Count.py
│   ├── Emoji_Features.py
│   ├── Format_Request_Detection.py # prompt-only: requested output format
│   ├── Has_Latex.py
│   ├── Interaction_Features.py
│   ├── Is_Code_Block.py
│   ├── Is_Natural_Text.py
│   ├── Json_Detection.py
│   ├── Length_Request_Detection.py # prompt-only: requested response length
│   ├── List_Detection.py
│   ├── Near_Empty_Detection.py
│   ├── Paragraph_Count.py
│   ├── Paragraph_Statistics.py
│   ├── Punctuation_Count.py
│   ├── Refusal_Detection.py
│   ├── Repetition.py
│   ├── Script_Detection.py
│   ├── Sentence_Count.py
│   ├── Sentence_Length.py
│   ├── Sentence_Length_Stats.py
│   ├── Sentence_Paragraph_Stats.py
│   ├── Table_Detection.py
│   ├── Token_Features.py
│   ├── Truncation_Detection.py
│   ├── Writing_Style_Features.py
│   └── README.md
│
├── Comparison_Features/            # Pairwise response_a vs. response_b extractors
│   ├── Avg_Words_Per_Sentence_Comparison.py
│   ├── Code_Delimiters_Comparison.py
│   ├── Emoji_Count_Comparison.py
│   ├── Format_Compliance_Comparison.py
│   ├── Format_Richness_Comparison.py
│   ├── Has_Conclusion_Comparison.py
│   ├── Has_Interaction_Prompt_Comparison.py
│   ├── Has_Latex_Comparison.py
│   ├── Has_Next_Steps_Comparison.py
│   ├── Has_Question_At_End_Comparison.py
│   ├── Has_Step_By_Step_Comparison.py
│   ├── Interaction_Score_Comparison.py
│   ├── Is_Detailed_Comparison.py
│   ├── Is_Multilingual_Comparison.py
│   ├── Is_Natural_Text_Comparison.py
│   ├── Length_Compliance_Comparison.py
│   ├── List_Item_Count_Comparison.py
│   ├── Long_Sentence_Count_Comparison.py
│   ├── Near_Empty_Comparison.py
│   ├── Paragraph_Count_Comparison.py
│   ├── Paragraph_Length_Mean_Comparison.py
│   ├── Paragraph_Length_Std_Comparison.py
│   ├── Prompt_Language_Match_Comparison.py
│   ├── Prompt_Script_Match_Comparison.py
│   ├── Punctuation_Count_Comparison.py
│   ├── Refusal_Comparison.py
│   ├── Repetition_Density_Comparison.py
│   ├── Sentence_Count_Comparison.py
│   ├── Sentence_Length_Std_Comparison.py
│   ├── Short_Sentence_Count_Comparison.py
│   ├── Table_Count_Comparison.py
│   ├── Token_Count_Comparison.py
│   ├── Truncation_Comparison.py
│   ├── Word_Count_Comparison.py
│   └── README.md
│
├── Feature_Runners/                # Execution pipelines
│   ├── Run_Metadata_Features.py
│   ├── Run_Text_Features.py
│   ├── Run_Comparison_Features.py
│   ├── Run_All_Features.py
│   └── README.md
│
└── README.md

```

---

# System Architecture

The feature extraction system is organized into four hierarchical layers:

```

┌──────────────────────────────────────────────────────────────────┐
│                        Feature_Runners                           │
│      Batch processing, column-gating, feature merging            │
├───────────────────┬───────────────────┬──────────────────────────┤
│ Metadata_Features │   Text_Features   │   Comparison_Features    │
│ (conv_metadata,   │ (prompt + response│ (a_* vs b_* outputs of   │
│  category_tag)    │  texts)           │  the two stages at left) │
├───────────────────┴───────────────────┴──────────────────────────┤
│                   Individual Feature Extractors                  │
│           Stateless functions operating on single input          │
└──────────────────────────────────────────────────────────────────┘

```

`Comparison_Features` is a **second-stage** layer: its extractors don't read raw `conv_metadata`/text at all — they consume the columns that `Metadata_Features` and `Text_Features` already produced. It must run after those two layers.

---

# Architectural Principles

## Layer 1 — Extractors (`Metadata_Features`, `Text_Features`)

Each module inside `Metadata_Features` and `Text_Features` provides one or more pure feature extraction functions.

Inputs:

- Metadata dictionary (`conv_metadata`) or the `category_tag` column
- Raw text (`user_prompt`, `response_a`, `response_b`)

Outputs:

- Scalar values
- Flat dictionaries of interpretable features

All extractors are stateless and independently testable.

---

## Layer 2 — Extractors (`Comparison_Features`)

Each module inside `Comparison_Features` provides one or more pure comparison functions that take the already-extracted `a_*`/`b_*` (and occasionally `prompt_*`/`dataset_*`) scalar feature values as arguments — not raw text or metadata.

Outputs are one of two recurring shapes:

- **Numeric comparison:** `{metric}_diff`, `{metric}_ratio`, `{metric}_a_gt_b`
- **Boolean agreement:** `both_{signal}`, `only_one_{signal}`, `neither_{signal}`

plus a handful of custom-logic extractors (format richness, prompt language/script match, format/length compliance) that combine multiple inputs before comparing.

---

## Layer 3 — Pipelines (`Feature_Runners`)

Runner modules apply feature extractors across DataFrames.

Responsibilities:

- Input validation
- Missing value handling
- Batch feature extraction
- Output flattening
- Feature naming conventions
- Column-gated execution (skip a stage, or an individual comparator, if its required input columns aren't present)

---

## Layer 4 — Unified Entry Point (`Run_All_Features`)

A single pipeline that sequentially executes metadata, text, and comparison feature extraction depending on which columns already exist in the DataFrame.

Supported inputs:

- `conv_metadata`
- `category_tag`
- `user_prompt`
- `response_a`
- `response_b`

Execution order is fixed (metadata → text → comparison) because the comparison layer depends on columns the other two produce.

---

# Feature Categories

## Metadata Features

Extracted from `conv_metadata` and `category_tag`.

| Category | Example Features | Source |
|---|---|---|
| Formatting | `bold_total`, `bold_style_preference`, `emphasis_intensity` | Bold.py |
| Conversation Dynamics | `turns`, `is_multi_turn`, `turn_density` | Conversation.py |
| Document Structure | `total_headers`, `header_depth_score` | Headers.py |
| List Usage | `total_list_items`, `ordered_ratio`, `unordered_ratio` | Lists.py |
| Token Allocation | `assistant_tokens_total`, `user_tokens`, `context_tokens_total` | Tokens.py |
| Dataset Baseline | Dataset-provided formatting/token statistics (`dataset_a_*`/`dataset_b_*`) | Dataset_Baseline.py |
| Interaction-Type Labels | `cat_complexity`, `cat_math`, `cat_creative_writing`, `cat_if`, `cat_if_score` | Category_Tag.py |

See the [Metadata_Features README](Metadata_Features/README.md) for full details.

---

# Text Features

Extracted independently from:

- `user_prompt`
- `response_a`
- `response_b`

The same feature extraction pipeline is applied to all text fields. Two extractors (`Format_Request_Detection`, `Length_Request_Detection`) are prompt-only, since they infer what the prompt *asked for* rather than describing a text's own properties.

Generated columns use prefixes:

- `prompt_`
- `a_`
- `b_`

---

## Text Feature Categories

| Category | Example Features |
|---|---|
| Language & Script | `primary_language`, `is_multilingual`, `script` |
| Structural | `paragraph_count`, `sentence_count`, `sentence_length_std`, `sentence_per_paragraph_std` |
| Code & Technical | `code_delimiters`, `is_code_block`, `has_latex`, `has_json`, `has_table`, `has_list` |
| Lexical | `token_count`, `repetition_density`, `is_natural_text` |
| Style | `word_count`, `avg_words_per_sentence`, `is_detailed` |
| Interaction | `interaction_score`, `has_conclusion`, `has_next_steps`, `has_question_at_end`, `has_interaction_prompt` |
| Symbolic | `has_emoji`, `emoji_count` |
| Format/Length Requests (prompt-only) | `requests_list`, `requests_table`, `requests_json`, `requests_code`, `requests_brief`, `requests_detailed` |
| Quality & Failure Signals | `has_refusal`, `is_near_empty`, `is_truncated` |

See the [Text_Features README](Text_Features/README.md) for full details.

---

# Comparison Features

Derived from the `Text_Features`/`Metadata_Features` outputs for `response_a` and `response_b` — not from raw text. Describes the *relationship* between the two responses on the same prompt, which is typically more predictive of pairwise human preference than either response's absolute value alone.

Generated columns are generally **unprefixed** (e.g. `word_count_diff`, `both_refused`), since the column name itself already encodes the A-vs-B comparison. A few custom-logic extractors use `a_`/`b_` to expose each side's own computed value alongside the comparison (e.g. `a_format_richness`).

## Comparison Feature Categories

| Category | Example Features |
|---|---|
| Length & Volume | `word_count_diff`, `token_count_diff`, `sentence_count_diff` |
| Structural Distribution | `avg_words_per_sentence_diff`, `paragraph_length_std_diff` |
| Code & Formatting Density | `code_delimiters_diff`, `table_count_diff`, `format_richness_diff` |
| Language & Script | `a_prompt_lang_match`, `a_prompt_script_match`, `both_multilingual` |
| Discourse & Interaction | `interaction_score_diff`, `both_conclusion`, `both_question_at_end` |
| Format & Length Compliance | `a_format_matches_request`, `a_length_appropriate` |
| Quality & Failure Signals | `both_refused`, `both_near_empty`, `both_truncated` |

See the [Comparison_Features README](Comparison_Features/README.md) for full details.

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
from Feature_Extraction.Feature_Runners.Run_Text_Features import run_all_Text_Features

df = run_all_Text_Features(df)
```

---

## Comparison Features Only

Requires `Text_Features` (and, for a few comparators, `Metadata_Features`) to already be present on `df`.

```python
from Feature_Extraction.Feature_Runners.Run_Comparison_Features import run_all_comparison_features

df = run_all_comparison_features(df)
```

---

## Single Text Feature Extraction

Example:

```python
from Feature_Extraction.Text_Features.Repetition import compute_repetition_density

score = compute_repetition_density(text)
```

---

## Single Comparison Feature Extraction

Example:

```python
from Feature_Extraction.Comparison_Features.Word_Count_Comparison import extract_word_count_comparison

result = extract_word_count_comparison(a_word_count=42, b_word_count=30)
# {"word_count_diff": 12.0, "word_count_ratio": ..., "word_count_a_gt_b": True}
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
dataset_a_bold / dataset_b_bold
dataset_a_header_h1..h6 / dataset_b_header_h1..h6
cat_complexity
cat_math
cat_if
cat_if_score
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
prompt_script
prompt_token_count
prompt_sentence_count
prompt_word_count
prompt_is_code_block
prompt_has_latex
prompt_repetition_density
prompt_requests_list
prompt_requests_brief
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
a_script
a_token_count
a_sentence_count
a_word_count
a_is_code_block
a_has_latex
a_repetition_density
a_has_refusal
a_is_near_empty
a_is_truncated
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

## Comparison Features

No fixed prefix — column names encode the A-vs-B relationship directly.

Example:

```
word_count_diff
word_count_ratio
word_count_a_gt_b
token_count_gap_magnitude
both_refused
only_one_refused
neither_refused
a_format_matches_request
b_format_matches_request
only_one_matched_format
a_prompt_lang_match
b_prompt_lang_match
only_one_matched_lang
...
```

---

# Design Principles

## Dependency-Light Architecture

The system mainly relies on:

* Python standard library (`re`, `math`, `json`, `ast`, `typing`)
* pandas
* mediapipe (only for language detection, via `Language_Detection.api`)

No transformer models or embedding APIs are required anywhere in the pipeline.

---

## Modular and Extensible

New features can be added by:

1. Creating a standalone extractor (in `Metadata_Features`, `Text_Features`, or `Comparison_Features`)
2. Returning a scalar or dictionary output
3. Registering it in the corresponding runner

---

## Robust Input Handling

All extractors handle:

* `None` values
* `NaN` values
* Empty strings
* Missing metadata fields
* Invalid input formats / types

without raising — falling back to zero-initialized or neutral defaults.

---

## Column-Gated Execution

Every runner checks required input columns before executing a stage (or, in `Run_Comparison_Features`, before executing each individual comparator). This allows partial pipelines — e.g. only metadata features computed, or only some `Text_Features` present — to run safely, producing a reduced but valid output schema rather than raising an error.

---

## Consistent Naming Convention

Feature prefixes indicate the source:

| Prefix    | Source      |
| --------- | ----------- |
| `prompt_` | User prompt |
| `a_`      | Response A  |
| `b_`      | Response B  |
| `dataset_a_` / `dataset_b_` | Dataset-native metadata fields, mirrored as-is |
| `cat_`    | `category_tag`-derived interaction-type labels |
| *(none)*  | Comparison features — column name already encodes the A-vs-B relationship |

This enables direct feature comparison and pairwise modeling.

---

# Intended Use Cases

* Pairwise preference modeling
* Reward model feature engineering
* Response quality and failure-mode analysis (refusals, near-empty responses, truncation)
* Format and length compliance checking against explicit prompt requests
* Prompt-response behavior analysis
* LLM output profiling
* Feature-based ML pipelines

---

# Notes

* Text features operate on raw text inputs and are shared between prompts and responses, except the prompt-only format/length-request extractors.
* Language and script detection are applied independently to prompts and responses.
* Metadata features operate on `conv_metadata` (parsed once per row) and, separately, on the `category_tag` column.
* Comparison features are a second-stage layer: they require `Text_Features`/`Metadata_Features` columns to already exist on `df`, and are column-gated per comparator so partial upstream data doesn't break the run.
* Token estimation is heuristic and not based on exact model tokenizers.
* Sentence and paragraph segmentation use regex-based heuristics.
* Refusal, near-empty, and truncation detection are conservative regex/length heuristics, not model-based classifiers.
* All runners operate on copies of DataFrames and do not mutate original inputs.
