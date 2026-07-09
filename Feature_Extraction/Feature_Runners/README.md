# Feature Runners: Execution Pipeline

## Overview

This module provides the **orchestration layer** for the feature extraction system, responsible for coordinating the application of individual feature extractors across DataFrames. Each runner implements a batch-processing pipeline that parses raw inputs, applies a registry of extractors, and merges the resulting features into a flat, columnar format suitable for downstream ML workflows.

The runners abstract away the complexity of input parsing, nested output flattening, and column-naming conventions, exposing a clean API over the underlying extractor modules.

The pipeline has three stages, run in sequence:

- **Metadata Features** (`Run_Metadata_Features.py`) — extracted from `conv_metadata` and `category_tag`
- **Text Features** (`Run_Text_Features.py`) — extracted from raw textual content (`user_prompt`, `response_a`, `response_b`)
- **Comparison Features** (`Run_Comparison_Features.py`) — pairwise `response_a` vs. `response_b` comparisons derived from the Text/Metadata outputs above

`Run_All_Features.py` chains all three into a single call.

---

## Module Structure

```

Feature_Runners/
│
├── Run_Metadata_Features.py     # Metadata feature extraction pipeline (conv_metadata + category_tag)
├── Run_Text_Features.py         # Text feature extraction pipeline (prompt + responses)
├── Run_Comparison_Features.py   # Pairwise A-vs-B comparison pipeline (built on the two above)
├── Run_All_Features.py          # Unified pipeline (metadata -> text -> comparison)
│
└── README.md

````

---

# Pipeline Components

## 1. Run Metadata Features (`Run_Metadata_Features.py`)

**Function:** `run_all_metadata_features(df) -> pd.DataFrame`

Orchestrates the extraction of all conversation-level metadata features from the `conv_metadata` column, plus a separate pass over the `category_tag` column.

### Architecture

The pipeline follows a **parse-once, apply-all** pattern for efficiency:

1. **Parsing:** Each row's `conv_metadata` is parsed once via `parse_metadata` (`ast.literal_eval` if stored as a string, passed through if already a dict, `{}` on failure).
2. **Registry dispatch:** The parsed dict is passed through a fixed list of extractor functions (`METADATA_FEATURES`).
3. **Flattening:** Each extractor's dict output is expanded into individual DataFrame columns via `.apply(pd.Series)`.
4. **Concatenation:** All extracted feature frames are concatenated with the original DataFrame in one `pd.concat`.
5. **Category tag pass:** If a `category_tag` column exists, `extract_category_tag_features` is applied to it independently (it is not part of `conv_metadata` and is not run through the parse-once step above) and concatenated onto `df`.

### Registered Extractors (`conv_metadata`)

| Extractor | Source Module | Output Features |
|-----------|--------------|----------------|
| `extract_bold_features` | `Metadata_Features.Bold` | `bold_total`, `bold_style_preference`, `emphasis_intensity` |
| `extract_conversation_dynamics` | `Metadata_Features.Conversation` | `turns`, `is_multi_turn`, `turn_density` |
| `extract_header_features` | `Metadata_Features.Headers` | `total_headers`, `header_depth_score` |
| `extract_list_features` | `Metadata_Features.Lists` | `total_list_items`, `ordered_ratio`, `unordered_ratio` |
| `extract_token_features` | `Metadata_Features.Tokens` | `assistant_tokens_total`, `user_tokens`, `context_tokens_total`, `assistant_token_ratio` |
| `extract_dataset_format_features` | `Metadata_Features.Dataset_Baseline` | `dataset_a_*`/`dataset_b_*` raw formatting fields (bold, list, header h1-h6, tokens, context tokens) |

### Separate Extractor (`category_tag`)

| Extractor | Source Module | Trigger Column | Output Features |
|-----------|--------------|-----------------|------------------|
| `extract_category_tag_features` | `Metadata_Features.Category_Tag` | `category_tag` (not `conv_metadata`) | `cat_complexity`, `cat_creativity`, `cat_domain_knowledge`, `cat_problem_solving`, `cat_real_world`, `cat_specificity`, `cat_technical_accuracy`, `cat_creative_writing`, `cat_math`, `cat_if`, `cat_if_score` |

See the [Metadata_Features README](../Metadata_Features/README.md) for full field-level documentation of every extractor above.

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
````

---

# 2. Run Text Features (`Run_Text_Features.py`)

**Function:** `run_all_Text_Features(df) -> pd.DataFrame`

Extracts linguistic, structural, formatting, language, and quality/failure-signal features from:

* `user_prompt`
* `response_a`
* `response_b`

The same composite extractor is applied independently to each text field.

---

## Architecture

The pipeline follows a **multi-column apply and prefix-merge** pattern:

1. A composite extractor (`extract_all_Text_Features`) combines every per-text `Text_Features` extractor (plus `Language_Detection.api` and `Script_Detection`) into one dict per text field.
2. The composite extractor is applied separately to:

   * `user_prompt` → prefixed `prompt_`
   * `response_a` → prefixed `a_`
   * `response_b` → prefixed `b_`
3. Two additional extractors are applied **only** to `user_prompt` (prefixed `prompt_`), since they infer what the prompt *requested* rather than describing a text's own properties:

   * `extract_format_request_features` (`Text_Features.Format_Request_Detection`)
   * `extract_length_request_features` (`Text_Features.Length_Request_Detection`)
4. All resulting feature frames are concatenated with the original DataFrame in one `pd.concat`.

---

## Composite Feature Set (`extract_all_Text_Features`)

Applied identically to `user_prompt`, `response_a`, and `response_b`:

| Feature                      | Type    | Source Module |
| ----------------------------- | ------- | ---------------- |
| `primary_language`           | `str`   | `Language_Detection.api.detectLanguage` |
| `is_multilingual`            | `bool`  | `Language_Detection.api.detectMultiLanguage` |
| `script`                     | `str`   | `Text_Features.Script_Detection` |
| `code_delimiters`            | `int`   | `Text_Features.Code_Delimiter_Count` |
| `punctuation_count`          | `int`   | `Text_Features.Punctuation_Count` |
| `has_latex`                  | `bool`  | `Text_Features.Has_Latex` |
| `has_emoji`                  | `bool`  | `Text_Features.Emoji_Features` |
| `emoji_count`                | `int`   | `Text_Features.Emoji_Features` |
| `token_count`                | `int`   | `Text_Features.Token_Features` |
| `repetition_density`         | `float` | `Text_Features.Repetition` |
| `is_code_block`              | `bool`  | `Text_Features.Is_Code_Block` |
| `is_natural_text`            | `bool`  | `Text_Features.Is_Natural_Text` |
| `sentence_count`             | `int`   | `Text_Features.Sentence_Count` |
| `has_table`                  | `bool`  | `Text_Features.Table_Detection` |
| `table_count`                | `int`   | `Text_Features.Table_Detection` |
| `has_list`                   | `bool`  | `Text_Features.List_Detection` |
| `list_item_count`            | `int`   | `Text_Features.List_Detection` |
| `has_json`                   | `bool`  | `Text_Features.Json_Detection` |
| `paragraph_count`            | `int`   | `Text_Features.Paragraph_Count` |
| `paragraph_length_mean`      | `float` | `Text_Features.Paragraph_Statistics` |
| `paragraph_length_std`       | `float` | `Text_Features.Paragraph_Statistics` |
| `long_sentence_count`        | `int`   | `Text_Features.Sentence_Length` |
| `short_sentence_count`       | `int`   | `Text_Features.Sentence_Length` |
| `sentence_length_std`        | `float` | `Text_Features.Sentence_Length_Stats` |
| `sentence_per_paragraph_std` | `float` | `Text_Features.Sentence_Paragraph_Stats` |
| `has_question_at_end`        | `bool`  | `Text_Features.Interaction_Features` |
| `has_conclusion`             | `bool`  | `Text_Features.Interaction_Features` |
| `has_next_steps`             | `bool`  | `Text_Features.Interaction_Features` |
| `has_interaction_prompt`     | `bool`  | `Text_Features.Interaction_Features` |
| `interaction_score`          | `int`   | `Text_Features.Interaction_Features` |
| `word_count`                 | `int`   | `Text_Features.Writing_Style_Features` |
| `avg_words_per_sentence`     | `float` | `Text_Features.Writing_Style_Features` |
| `is_detailed`                | `bool`  | `Text_Features.Writing_Style_Features` |
| `has_step_by_step`           | `bool`  | `Text_Features.Writing_Style_Features` |
| `has_refusal`                | `bool`  | `Text_Features.Refusal_Detection` |
| `is_near_empty`              | `bool`  | `Text_Features.Near_Empty_Detection` |
| `is_truncated`               | `bool`  | `Text_Features.Truncation_Detection` |

## Prompt-Only Feature Set

Applied only to `user_prompt`, prefixed `prompt_` (no `a_`/`b_` equivalents):

| Feature | Type | Source Module |
|---------|------|------------------|
| `requests_list` | `bool` | `Text_Features.Format_Request_Detection` |
| `requests_table` | `bool` | `Text_Features.Format_Request_Detection` |
| `requests_json` | `bool` | `Text_Features.Format_Request_Detection` |
| `requests_code` | `bool` | `Text_Features.Format_Request_Detection` |
| `requests_brief` | `bool` | `Text_Features.Length_Request_Detection` |
| `requests_detailed` | `bool` | `Text_Features.Length_Request_Detection` |

See the [Text_Features README](../Text_Features/README.md) for full field-level documentation of every extractor above.

---

## Output Naming Convention

The extracted features are prefixed based on their source:

### User Prompt Features

```
prompt_primary_language
prompt_script
prompt_token_count
prompt_sentence_count
prompt_is_multilingual
prompt_requests_list
prompt_requests_brief
...
```

### Response A Features

```
a_primary_language
a_script
a_token_count
a_sentence_count
a_is_multilingual
a_has_refusal
a_is_near_empty
a_is_truncated
...
```

### Response B Features

```
b_primary_language
b_script
b_token_count
b_sentence_count
b_is_multilingual
b_has_refusal
b_is_near_empty
b_is_truncated
...
```

This naming convention allows direct comparison between:

* Prompt and response characteristics
* Response A and response B differences
* Model behavior patterns

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_Text_Features import run_all_Text_Features

df = run_all_Text_Features(df)
```

---

# 3. Run Comparison Features (`Run_Comparison_Features.py`)

**Function:** `run_all_comparison_features(df) -> pd.DataFrame`

Converts already-extracted `Text_Features`/`Metadata_Features` columns for `response_a` and `response_b` into pairwise comparison signals — differences, ratios, and agreement flags — that are typically more predictive of pairwise human preference than either side's absolute value alone.

This runner does **not** touch raw text; it operates entirely on the `prompt_*`/`a_*`/`b_*`/`dataset_*` columns produced by the two runners above.

---

## Architecture

Unlike the metadata and text runners, this one has **no shared registry loop** — each of the ~30 comparators is wired individually:

1. For each comparator, the runner defines the list of input columns it needs.
2. It checks `if all(col in df.columns for col in [...])` before running that comparator at all.
3. If the check passes, the comparator is applied row-wise via `df.apply(..., axis=1)` and the resulting columns are concatenated onto `df`.
4. If the check fails (an upstream column is missing), that comparator is **silently skipped** — no error is raised, and the corresponding output columns simply don't appear.

This makes the runner safe to call on partially-featurized DataFrames (e.g. if only metadata features were run, `format_richness` comparisons still work but `word_count` comparisons are skipped).

Comparison feature names are **not** prefixed with `prompt_`/`a_`/`b_` in general, since the column name itself already encodes the A-vs-B relationship (e.g. `word_count_diff`, `both_refused`). A handful of custom-logic comparators do use `a_`/`b_` to expose each side's own computed value alongside the comparison (e.g. `a_format_richness`, `a_prompt_lang_match`).

See the [Comparison_Features README](../Comparison_Features/README.md) for the full list of ~30 comparators, their input columns, and output schemas.

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_Comparison_Features import run_all_comparison_features

df = run_all_comparison_features(df)
```

---

# 4. Run All Features (`Run_All_Features.py`)

**Function:** `run_all_features(df) -> pd.DataFrame`

Chains all three pipeline stages against a single DataFrame.

---

## Behavior

```python
df = df.copy()

if "conv_metadata" in df.columns:
    df = run_all_metadata_features(df)

if all(col in df.columns for col in ["user_prompt", "response_a", "response_b"]):
    df = run_all_Text_Features(df)

df = run_all_comparison_features(df)

return df
```

1. **Metadata stage:** runs only if `conv_metadata` is present.
2. **Text stage:** runs only if `user_prompt`, `response_a`, and `response_b` are **all** present.
3. **Comparison stage:** always invoked. Its internal per-comparator column checks (see above) mean it silently produces only the comparisons whose inputs actually exist — if neither prior stage ran, it is effectively a no-op that returns `df` unchanged.

Each stage builds on the DataFrame produced by the previous one — this is a sequential pipeline, not an independent branch-and-merge: comparison features depend on columns the text/metadata stages produce, so stage order matters and is fixed.

---

## Usage

```python
from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = run_all_features(df)
```

---

# Output Schema

After running the full pipeline (`run_all_features`), the resulting DataFrame contains the original columns plus all features from every stage that had its required inputs available.

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
dataset_a_list_ordered / dataset_b_list_ordered
dataset_a_header_h1..h6 / dataset_b_header_h1..h6
dataset_a_tokens / dataset_b_tokens
cat_complexity
cat_creativity
cat_math
cat_if
cat_if_score
...
```

---

## Prompt Features

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

## Response A / Response B Features

```
a_primary_language / b_primary_language
a_is_multilingual / b_is_multilingual
a_token_count / b_token_count
a_sentence_count / b_sentence_count
a_word_count / b_word_count
a_is_code_block / b_is_code_block
a_has_latex / b_has_latex
a_repetition_density / b_repetition_density
a_has_refusal / b_has_refusal
a_is_near_empty / b_is_near_empty
a_is_truncated / b_is_truncated
...
```

---

## Comparison Features (unprefixed — describe A vs. B)

```
word_count_diff / word_count_ratio / word_count_a_gt_b
token_count_diff / token_count_ratio / token_count_a_gt_b / token_count_gap_magnitude
sentence_count_diff / sentence_count_ratio / sentence_count_a_gt_b
repetition_density_diff / repetition_density_ratio / repetition_density_a_gt_b
format_richness_diff / format_richness_ratio / format_richness_a_gt_b
both_refused / only_one_refused / neither_refused
both_near_empty / only_one_near_empty / neither_near_empty
both_truncated / only_one_truncated / neither_truncated
a_format_matches_request / b_format_matches_request / only_one_matched_format
a_length_appropriate / b_length_appropriate / only_one_length_appropriate
a_prompt_lang_match / b_prompt_lang_match / only_one_matched_lang
a_prompt_script_match / b_prompt_script_match / only_one_matched_script
...
```

See the [Comparison_Features README](../Comparison_Features/README.md) for the complete list.

---

# Design Principles

## Column-Gated Execution

Every runner checks required input columns before executing a stage (or, for `Run_Comparison_Features`, before executing each individual comparator). This allows processing datasets with different available fields — or datasets that only have some upstream features already computed — without raising unnecessary errors. Missing inputs simply mean fewer output columns, not a failed run.

---

## Modular Feature Expansion

Each feature group is isolated and layered:

* Metadata features analyze dataset-provided information (`conv_metadata`, `category_tag`).
* Text features analyze raw textual content (`user_prompt`, `response_a`, `response_b`).
* Comparison features analyze the *relationship* between `a_*` and `b_*` outputs of the two stages above.
* Additional feature groups can be integrated by adding a new runner and wiring it into `Run_All_Features.py`, without modifying the existing stages.

---

## Safe Feature Aggregation

Dictionary-based outputs are safely merged into flat DataFrame columns via `.apply(pd.Series)` and `pd.concat`, allowing different extractors with different output shapes to coexist in a single pass.

---

## Consistent Naming Convention

Feature prefixes preserve feature provenance:

* `prompt_` → user prompt features
* `a_` → response A features
* `b_` → response B features
* `dataset_a_` / `dataset_b_` → dataset-native metadata fields, mirrored as-is (`Dataset_Baseline`)
* `cat_` → `category_tag`-derived interaction-type labels
* *(unprefixed)* → comparison features, since the column name already encodes the A-vs-B relationship

This enables:

* Pairwise model training
* Feature difference analysis
* Behavioral comparison between prompts and responses

---

# Intended Use Cases

* **End-to-end feature extraction:** Convert raw LM-Arena data into a structured ML-ready dataset in a single `run_all_features(df)` call.
* **Prompt-response analysis:** Study relationships between user prompts and generated answers, including whether responses comply with explicit format/length requests.
* **Pairwise response modeling:** Compare behavioral, structural, and quality differences between responses via the comparison stage.
* **Reward model feature engineering:** Provide interpretable signals for preference modeling, including refusal/truncation/near-empty failure signals.
* **LLM behavior analysis:** Analyze response style, structure, language, formatting, and interaction-type-conditional patterns.

---

# Notes

* Text feature extraction depends on `Language_Detection.api` for `primary_language`/`is_multilingual`, and on `Text_Features.Script_Detection` as a lightweight fallback (`script`) when full language detection is unavailable or unreliable.
* Language and script detection are applied independently to the prompt and each response.
* Metadata extraction from `conv_metadata` operates on a dict parsed once per row (`ast.literal_eval` for string-encoded cells); `category_tag` is parsed and extracted separately since it is a distinct dataset column, not a `conv_metadata` field.
* Comparison features require the corresponding `Text_Features`/`Metadata_Features` columns to already exist on `df` — call `run_all_metadata_features`/`run_all_Text_Features` (or `run_all_features`, which sequences everything correctly) before relying on any comparison output.
* All runners operate on a copy of the input DataFrame (`df.copy()`) and do not mutate the original data.
* Feature extraction is designed to be lightweight and scalable for large datasets — no transformer models or heavy NLP frameworks are used anywhere in the pipeline.
