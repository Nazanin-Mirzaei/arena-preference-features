# Comparison Features Extraction Module

## Overview

This module provides **pairwise comparison feature extractors** that operate on already-computed `Text_Features` (and select `Metadata_Features`) values for `response_a` and `response_b` to produce relative, comparative signals. Where `Text_Features` describes a single response in isolation, `Comparison_Features` describes the *relationship* between two responses on the same prompt — which is typically more predictive of pairwise human preference than either response's absolute value alone.

Every extractor in this module is a **second-stage** feature: its inputs are the outputs of `Text_Features` (and a few `Metadata_Features` columns), not raw prompt/response text. It does not re-read `user_prompt`, `response_a`, or `response_b` directly.

---

## Module Structure

```

Comparison_Features/
│
├── Avg_Words_Per_Sentence_Comparison.py   # avg_words_per_sentence diff/ratio/gt
├── Code_Delimiters_Comparison.py          # code_delimiters diff/ratio/gt
├── Emoji_Count_Comparison.py              # emoji_count diff/ratio/gt
├── Format_Compliance_Comparison.py        # prompt format-request compliance (list/table/json/code)
├── Format_Richness_Comparison.py          # weighted bold+header+list+table composite score
├── Has_Conclusion_Comparison.py           # has_conclusion agreement
├── Has_Interaction_Prompt_Comparison.py   # has_interaction_prompt agreement
├── Has_Latex_Comparison.py                # has_latex agreement
├── Has_Next_Steps_Comparison.py           # has_next_steps agreement
├── Has_Question_At_End_Comparison.py      # has_question_at_end agreement
├── Has_Step_By_Step_Comparison.py         # has_step_by_step agreement
├── Interaction_Score_Comparison.py        # interaction_score diff/ratio/gt
├── Is_Detailed_Comparison.py              # is_detailed agreement
├── Is_Multilingual_Comparison.py          # is_multilingual (code-switching) agreement
├── Is_Natural_Text_Comparison.py          # is_natural_text agreement
├── Length_Compliance_Comparison.py        # prompt length-request compliance (brief/detailed)
├── List_Item_Count_Comparison.py          # list_item_count diff/ratio/gt
├── Long_Sentence_Count_Comparison.py      # long_sentence_count diff/ratio/gt
├── Near_Empty_Comparison.py               # is_near_empty agreement
├── Paragraph_Count_Comparison.py          # paragraph_count diff/ratio/gt
├── Paragraph_Length_Mean_Comparison.py    # paragraph_length_mean diff/ratio/gt
├── Paragraph_Length_Std_Comparison.py     # paragraph_length_std diff/ratio/gt
├── Prompt_Language_Match_Comparison.py    # detected language vs. prompt language
├── Prompt_Script_Match_Comparison.py      # detected script vs. prompt script
├── Punctuation_Count_Comparison.py        # punctuation_count diff/ratio/gt
├── Refusal_Comparison.py                  # has_refusal agreement
├── Repetition_Density_Comparison.py       # repetition_density diff/ratio/gt
├── Sentence_Count_Comparison.py           # sentence_count diff/ratio/gt
├── Sentence_Length_Std_Comparison.py      # sentence_length_std diff/ratio/gt
├── Short_Sentence_Count_Comparison.py     # short_sentence_count diff/ratio/gt
├── Table_Count_Comparison.py              # table_count diff/ratio/gt
├── Token_Count_Comparison.py              # token_count diff/ratio/gt/gap_magnitude
├── Truncation_Comparison.py               # is_truncated agreement
├── Word_Count_Comparison.py               # word_count diff/ratio/gt
│
└── README.md

````

---

## Input Format

Unlike `Text_Features`, these extractors do **not** accept raw text. Each takes the already-extracted `a_*` / `b_*` (and sometimes `prompt_*`) feature values as scalar arguments:

```python
extract_word_count_comparison(a_word_count: Any, b_word_count: Any) -> dict
````

Every extractor independently and defensively handles:

* `None`
* `NaN`
* non-numeric / non-boolean input types

Numeric inputs are coerced via a shared `_to_number` helper (invalid values fall back to `0.0`). Boolean inputs are coerced via a shared `_to_bool` helper (`None`/`NaN` fall back to `False`).

These extractors are always run **after** `Text_Features` (and, for a few, after relevant `Metadata_Features`) have populated the DataFrame, since they consume those columns as input.

---

## Two Recurring Templates

Most modules in this package follow one of two fixed output shapes, so they are documented in groups rather than one-by-one where the pattern is identical.

### A. Numeric Comparison Template (`_diff` / `_ratio` / `_a_gt_b`)

Applies to continuous or count-valued `Text_Features` outputs (word counts, sentence counts, densities, etc.). Given `a` and `b` coerced to `float`:

| Feature       | Type    | Description                          |
| ------------- | ------- | ------------------------------------- |
| `{metric}_diff`    | `float` | `a - b`                          |
| `{metric}_ratio`   | `float` | `a / (b + 1e-6)`                 |
| `{metric}_a_gt_b`  | `bool`  | `a > b`                          |

`EPSILON = 1e-6` guards against division by zero when `b == 0`.

### B. Boolean Agreement Template (`both_*` / `only_one_*` / `neither_*`)

Applies to binary `Text_Features` flags (refusal, near-empty, truncation, latex, interaction sub-signals, etc.). Given `a` and `b` coerced to `bool`:

| Feature         | Type   | Description        |
| ---------------- | ------ | -------------------- |
| `both_{signal}`     | `bool` | `a and b`        |
| `only_one_{signal}` | `bool` | `a != b`          |
| `neither_{signal}`  | `bool` | `not a and not b` |

These modules deliberately omit `_diff`/`_ratio` outputs. Per the modules' own docstrings: a diff on a boolean pair is redundant with the two source columns (fully reconstructible from `a_*`/`b_*`) and would split an already-rare or already-narrow signal by an arbitrary A/B slot direction (model_a/model_b is not a stable axis); a ratio is degenerate for 0/1 values.

---

## Feature Descriptions

### Numeric Template Modules

| # | Module | Function | Underlying `Text_Features` metric | Extra fields |
|---|--------|----------|-------------------------------------|--------------|
| 1 | `Word_Count_Comparison.py` | `extract_word_count_comparison(a_word_count, b_word_count)` | `word_count` | — |
| 2 | `Token_Count_Comparison.py` | `extract_token_count_comparison(a_token_count, b_token_count)` | `token_count` | `token_count_gap_magnitude` (`\|a - b\|`) |
| 3 | `Sentence_Count_Comparison.py` | `extract_sentence_count_comparison(a_sentence_count, b_sentence_count)` | `sentence_count` | — |
| 4 | `Paragraph_Count_Comparison.py` | `extract_paragraph_count_comparison(a_paragraph_count, b_paragraph_count)` | `paragraph_count` | — |
| 5 | `Code_Delimiters_Comparison.py` | `extract_code_delimiters_comparison(a_code_delimiters, b_code_delimiters)` | `code_delimiters` | — |
| 6 | `Punctuation_Count_Comparison.py` | `extract_punctuation_count_comparison(a_punctuation_count, b_punctuation_count)` | `punctuation_count` | — |
| 7 | `Repetition_Density_Comparison.py` | `extract_repetition_density_comparison(a_repetition_density, b_repetition_density)` | `repetition_density` | — |
| 8 | `Emoji_Count_Comparison.py` | `extract_emoji_count_comparison(a_emoji_count, b_emoji_count)` | `emoji_count` | — |
| 9 | `Table_Count_Comparison.py` | `extract_table_count_comparison(a_table_count, b_table_count)` | `table_count` | — |
| 10 | `List_Item_Count_Comparison.py` | `extract_list_item_count_comparison(a_list_item_count, b_list_item_count)` | `list_item_count` | — |
| 11 | `Avg_Words_Per_Sentence_Comparison.py` | `extract_avg_words_per_sentence_comparison(a_avg_words_per_sentence, b_avg_words_per_sentence)` | `avg_words_per_sentence` | — |
| 12 | `Sentence_Length_Std_Comparison.py` | `extract_sentence_length_std_comparison(a_sentence_length_std, b_sentence_length_std)` | `sentence_length_std` | — |
| 13 | `Paragraph_Length_Mean_Comparison.py` | `extract_paragraph_length_mean_comparison(a_paragraph_length_mean, b_paragraph_length_mean)` | `paragraph_length_mean` | — |
| 14 | `Paragraph_Length_Std_Comparison.py` | `extract_paragraph_length_std_comparison(a_paragraph_length_std, b_paragraph_length_std)` | `paragraph_length_std` | — |
| 15 | `Interaction_Score_Comparison.py` | `extract_interaction_score_comparison(a_interaction_score, b_interaction_score)` | `interaction_score` | — |
| 16 | `Long_Sentence_Count_Comparison.py` | `extract_long_sentence_count_comparison(a_long_sentence_count, b_long_sentence_count)` | `long_sentence_count` | — |
| 17 | `Short_Sentence_Count_Comparison.py` | `extract_short_sentence_count_comparison(a_short_sentence_count, b_short_sentence_count)` | `short_sentence_count` | — |

Each produces exactly `{metric}_diff`, `{metric}_ratio`, `{metric}_a_gt_b` (see [Template A](#a-numeric-comparison-template-_diff--_ratio--_a_gt_b)), except `Token_Count_Comparison` which adds `token_count_gap_magnitude`.

---

### Boolean Agreement Template Modules

| # | Module | Function | Underlying `Text_Features` flag | Output prefix |
|---|--------|----------|-------------------------------------|----------------|
| 18 | `Refusal_Comparison.py` | `extract_refusal_comparison(a_has_refusal, b_has_refusal)` | `has_refusal` | `*_refused` |
| 19 | `Near_Empty_Comparison.py` | `extract_near_empty_comparison(a_is_near_empty, b_is_near_empty)` | `is_near_empty` | `*_near_empty` |
| 20 | `Truncation_Comparison.py` | `extract_truncation_comparison(a_is_truncated, b_is_truncated)` | `is_truncated` | `*_truncated` |
| 21 | `Has_Latex_Comparison.py` | `extract_has_latex_comparison(a_has_latex, b_has_latex)` | `has_latex` | `*_has_latex` |
| 22 | `Is_Detailed_Comparison.py` | `extract_is_detailed_comparison(a_is_detailed, b_is_detailed)` | `is_detailed` | `*_detailed` |
| 23 | `Has_Step_By_Step_Comparison.py` | `extract_has_step_by_step_comparison(a_has_step_by_step, b_has_step_by_step)` | `has_step_by_step` | `*_step_by_step` |
| 24 | `Is_Natural_Text_Comparison.py` | `extract_is_natural_text_comparison(a_is_natural_text, b_is_natural_text)` | `is_natural_text` | `*_natural_text` |
| 25 | `Has_Conclusion_Comparison.py` | `extract_has_conclusion_comparison(a_has_conclusion, b_has_conclusion)` | `has_conclusion` | `*_conclusion` |
| 26 | `Has_Next_Steps_Comparison.py` | `extract_has_next_steps_comparison(a_has_next_steps, b_has_next_steps)` | `has_next_steps` | `*_next_steps` |
| 27 | `Has_Question_At_End_Comparison.py` | `extract_has_question_at_end_comparison(a_has_question_at_end, b_has_question_at_end)` | `has_question_at_end` | `*_question_at_end` |
| 28 | `Has_Interaction_Prompt_Comparison.py` | `extract_has_interaction_prompt_comparison(a_has_interaction_prompt, b_has_interaction_prompt)` | `has_interaction_prompt` | `*_interaction_prompt` |
| 29 | `Is_Multilingual_Comparison.py` | `extract_is_multilingual_comparison(a_is_multilingual, b_is_multilingual)` | `is_multilingual` | `*_multilingual` |

Each produces exactly `both_{X}`, `only_one_{X}`, `neither_{X}` (see [Template B](#b-boolean-agreement-template-both_--only_one_--neither_)), where `{X}` is the output-prefix stem above.

Several of these (`Has_Conclusion`, `Has_Next_Steps`, `Has_Question_At_End`, `Has_Interaction_Prompt`) decompose the single composite `interaction_score` into its four individual binary sub-signals, letting a model weigh each discourse pattern independently rather than only seeing the summed score.

---

### Custom-Logic Modules

The remaining modules don't fit either template — they combine multiple inputs, apply weights, thresholds, or normalization before comparing.

#### 30. Format Richness Comparison (`Format_Richness_Comparison.py`)

**Functions:**
`compute_format_richness(bold, headers, list_items, tables) -> float`
`extract_format_richness_comparison(a_bold, a_headers, a_list_items, a_tables, b_bold, b_headers, b_list_items, b_tables) -> dict`

Combines bold-emphasis count, header count (all levels), list item count, and table count into a single weighted per-side "richness" score (`bold×1.0 + headers×2.0 + list_items×1.0 + tables×3.0`), then compares the two scores. Richer formatting is a documented human-preference signal independent of underlying content quality; headers and tables are weighted higher since they are rarer and more visually structuring than bold spans or list items.

| Feature | Type | Description |
|---------|------|-------------|
| `a_format_richness` | `float` | Weighted richness score for response A |
| `b_format_richness` | `float` | Weighted richness score for response B |
| `format_richness_diff` | `float` | `a_format_richness - b_format_richness` |
| `format_richness_ratio` | `float` | `a_format_richness / (b_format_richness + 1e-6)` |
| `format_richness_a_gt_b` | `bool` | `a_format_richness > b_format_richness` |

Inputs are sourced from `Metadata_Features` (`dataset_a_bold`, `dataset_a_header_h1..h6`, `dataset_a_list_ordered`/`list_unordered`) and `Text_Features` (`a_table_count`), summed/combined by the runner before calling this extractor.

---

#### 31. Prompt Language Match Comparison (`Prompt_Language_Match_Comparison.py`)

**Function:** `extract_prompt_language_match_comparison(prompt_language, a_language, b_language) -> dict`

Compares each response's detected `primary_language` (from `Language_Detection.api`, surfaced as `Text_Features.primary_language`) against the prompt's detected language. Answering in the wrong language is a strong quality/loser signal independent of content; when only one side matches, that side has a decisive advantage.

| Feature | Type | Description |
|---------|------|-------------|
| `a_prompt_lang_match` | `bool` | `a_language == prompt_language` (both normalized, case-insensitive; `False` if prompt language is empty/unknown) |
| `b_prompt_lang_match` | `bool` | `b_language == prompt_language` |
| `only_one_matched_lang` | `bool` | Exactly one side matched |

---

#### 32. Prompt Script Match Comparison (`Prompt_Script_Match_Comparison.py`)

**Function:** `extract_prompt_script_match_comparison(prompt_script, a_script, b_script) -> dict`

Compares each response's dominant Unicode script (`Text_Features.Script_Detection.detect_script`) against the prompt's dominant script. A cheap, dependency-free fallback for language-match signals — useful when full language detection is unavailable, misfires, or returns "unknown," since a script mismatch (e.g. prompt in Arabic, response in Latin) is still a strong indicator of a wrong-language response.

| Feature | Type | Description |
|---------|------|-------------|
| `a_prompt_script_match` | `bool` | `a_script == prompt_script` (`"other"`/empty treated as no match) |
| `b_prompt_script_match` | `bool` | `b_script == prompt_script` |
| `only_one_matched_script` | `bool` | Exactly one side matched |

---

#### 33. Format Compliance Comparison (`Format_Compliance_Comparison.py`)

**Functions:**
`compute_format_compliance(requests_list, requests_table, requests_json, requests_code, has_list, has_table, has_json, is_code_block) -> bool`
`extract_format_compliance_comparison(requests_list, requests_table, requests_json, requests_code, a_has_list, a_has_table, a_has_json, a_is_code_block, b_has_list, b_has_table, b_has_json, b_is_code_block) -> dict`

Checks whether each response delivers every output format the prompt explicitly requested (`Text_Features.Format_Request_Detection`), against what the response actually contains (`Text_Features` list/table/json/code detectors). Vacuously `True` if the prompt made no explicit format request.

| Feature | Type | Description |
|---------|------|-------------|
| `a_format_matches_request` | `bool` | Response A satisfies every requested format |
| `b_format_matches_request` | `bool` | Response B satisfies every requested format |
| `only_one_matched_format` | `bool` | Exactly one side complied |

---

#### 34. Length Compliance Comparison (`Length_Compliance_Comparison.py`)

**Functions:**
`compute_length_compliance(requests_brief, requests_detailed, word_count, brief_threshold=60, detailed_threshold=50) -> bool`
`extract_length_compliance_comparison(requests_brief, requests_detailed, a_word_count, b_word_count) -> dict`

Checks whether each response's actual `word_count` satisfies an explicit brevity/detail request in the prompt (`Text_Features.Length_Request_Detection`). A response must be `≤ 60` words to satisfy "briefly," and `≥ 50` words to satisfy "in detail" (mirroring the word-count half of `Writing_Style_Features.is_detailed`). If the prompt contradictorily requests both, both thresholds must be satisfied (possible only in the narrow band between them). Vacuously `True` if the prompt made no explicit length request.

| Feature | Type | Description |
|---------|------|-------------|
| `a_length_appropriate` | `bool` | Response A's length satisfies the prompt's length request(s) |
| `b_length_appropriate` | `bool` | Response B's length satisfies the prompt's length request(s) |
| `only_one_length_appropriate` | `bool` | Exactly one side complied |

---

## Category Summary

| Category | Modules |
|----------|---------|
| **Length & Volume** | `Word_Count_Comparison`, `Token_Count_Comparison`, `Sentence_Count_Comparison`, `Paragraph_Count_Comparison`, `Long_Sentence_Count_Comparison`, `Short_Sentence_Count_Comparison` |
| **Structural Distribution** | `Avg_Words_Per_Sentence_Comparison`, `Sentence_Length_Std_Comparison`, `Paragraph_Length_Mean_Comparison`, `Paragraph_Length_Std_Comparison` |
| **Code & Formatting Density** | `Code_Delimiters_Comparison`, `Punctuation_Count_Comparison`, `Table_Count_Comparison`, `List_Item_Count_Comparison`, `Format_Richness_Comparison` |
| **Symbolic & Notation** | `Emoji_Count_Comparison`, `Has_Latex_Comparison` |
| **Lexical Quality** | `Repetition_Density_Comparison` |
| **Language & Script** | `Prompt_Language_Match_Comparison`, `Prompt_Script_Match_Comparison`, `Is_Multilingual_Comparison` |
| **Discourse & Interaction** | `Interaction_Score_Comparison`, `Has_Conclusion_Comparison`, `Has_Next_Steps_Comparison`, `Has_Question_At_End_Comparison`, `Has_Interaction_Prompt_Comparison` |
| **Style Composites** | `Is_Detailed_Comparison`, `Has_Step_By_Step_Comparison`, `Is_Natural_Text_Comparison` |
| **Format & Length Compliance** | `Format_Compliance_Comparison`, `Length_Compliance_Comparison` |
| **Quality & Failure Signals** | `Refusal_Comparison`, `Near_Empty_Comparison`, `Truncation_Comparison` |

---

## Usage Examples

### Single Comparison

```python
from Comparison_Features.Word_Count_Comparison import extract_word_count_comparison

result = extract_word_count_comparison(
    a_word_count=df.loc[0, "a_word_count"],
    b_word_count=df.loc[0, "b_word_count"],
)
# {"word_count_diff": ..., "word_count_ratio": ..., "word_count_a_gt_b": ...}
```

---

### Boolean Agreement Comparison

```python
from Comparison_Features.Refusal_Comparison import extract_refusal_comparison

result = extract_refusal_comparison(
    a_has_refusal=df.loc[0, "a_has_refusal"],
    b_has_refusal=df.loc[0, "b_has_refusal"],
)
# {"both_refused": ..., "only_one_refused": ..., "neither_refused": ...}
```

---

### Compliance Comparison (Prompt + Both Responses)

```python
from Comparison_Features.Format_Compliance_Comparison import extract_format_compliance_comparison

row = df.loc[0]
result = extract_format_compliance_comparison(
    requests_list=row["prompt_requests_list"],
    requests_table=row["prompt_requests_table"],
    requests_json=row["prompt_requests_json"],
    requests_code=row["prompt_requests_code"],
    a_has_list=row["a_has_list"], a_has_table=row["a_has_table"],
    a_has_json=row["a_has_json"], a_is_code_block=row["a_is_code_block"],
    b_has_list=row["b_has_list"], b_has_table=row["b_has_table"],
    b_has_json=row["b_has_json"], b_is_code_block=row["b_is_code_block"],
)
```

---

## Integration with Feature Runner

For full pipeline execution, use the unified comparison feature runner:

```python
from Feature_Extraction.Feature_Runners.Run_Comparison_Features import run_all_comparison_features

df = run_all_comparison_features(df)
```

The runner:

* Assumes `Text_Features` (via `Run_Text_Features.run_all_Text_Features`) and relevant `Metadata_Features` have already populated the DataFrame with `prompt_*`, `a_*`, and `b_*` columns.
* **Gates every comparator behind a column-existence check.** Each block only runs `if all(col in df.columns for col in [...])`, so if an upstream feature (e.g. `a_word_count`) is missing, that comparison is silently skipped rather than raising an error. This makes the runner safe to call on partially-featurized DataFrames.
* Applies each extractor row-wise via `df.apply(..., axis=1)` and concatenates the resulting columns onto `df`.
* Does not add a source prefix — comparison feature names are already globally unique (`word_count_diff`, `both_refused`, etc.) since they describe a relationship, not a single source.

---

# Output Schema

Comparison features have **no `prompt_`/`a_`/`b_` prefix** in general — column names already encode that they compare A vs. B (e.g. `word_count_diff`, `both_refused`). The exceptions are the per-side outputs of a few custom-logic modules, which use `a_`/`b_` to report each side's own computed value before the comparison (e.g. `a_format_richness`, `a_prompt_lang_match`, `a_format_matches_request`).

Examples of generated columns:

```
word_count_diff
word_count_ratio
word_count_a_gt_b
token_count_diff
token_count_ratio
token_count_a_gt_b
token_count_gap_magnitude
sentence_count_diff
sentence_count_ratio
sentence_count_a_gt_b
paragraph_count_diff
paragraph_count_ratio
paragraph_count_a_gt_b
code_delimiters_diff
code_delimiters_ratio
code_delimiters_a_gt_b
punctuation_count_diff
punctuation_count_ratio
punctuation_count_a_gt_b
repetition_density_diff
repetition_density_ratio
repetition_density_a_gt_b
emoji_count_diff
emoji_count_ratio
emoji_count_a_gt_b
table_count_diff
table_count_ratio
table_count_a_gt_b
list_item_count_diff
list_item_count_ratio
list_item_count_a_gt_b
avg_words_per_sentence_diff
avg_words_per_sentence_ratio
avg_words_per_sentence_a_gt_b
sentence_length_std_diff
sentence_length_std_ratio
sentence_length_std_a_gt_b
paragraph_length_mean_diff
paragraph_length_mean_ratio
paragraph_length_mean_a_gt_b
paragraph_length_std_diff
paragraph_length_std_ratio
paragraph_length_std_a_gt_b
interaction_score_diff
interaction_score_ratio
interaction_score_a_gt_b
long_sentence_count_diff
long_sentence_count_ratio
long_sentence_count_a_gt_b
short_sentence_count_diff
short_sentence_count_ratio
short_sentence_count_a_gt_b
a_format_richness
b_format_richness
format_richness_diff
format_richness_ratio
format_richness_a_gt_b
a_prompt_lang_match
b_prompt_lang_match
only_one_matched_lang
a_prompt_script_match
b_prompt_script_match
only_one_matched_script
both_refused
only_one_refused
neither_refused
both_near_empty
only_one_near_empty
neither_near_empty
a_format_matches_request
b_format_matches_request
only_one_matched_format
a_length_appropriate
b_length_appropriate
only_one_length_appropriate
both_truncated
only_one_truncated
neither_truncated
both_has_latex
only_one_has_latex
neither_has_latex
both_detailed
only_one_detailed
neither_detailed
both_step_by_step
only_one_step_by_step
neither_step_by_step
both_natural_text
only_one_natural_text
neither_natural_text
both_conclusion
only_one_conclusion
neither_conclusion
both_next_steps
only_one_next_steps
neither_next_steps
both_question_at_end
only_one_question_at_end
neither_question_at_end
both_interaction_prompt
only_one_interaction_prompt
neither_interaction_prompt
both_multilingual
only_one_multilingual
neither_multilingual
```

---

# Design Principles

## Dependency-Light Implementation

All extractors rely only on:

* `typing`
* `math`

No transformer models, embedding APIs, or heavy NLP frameworks are required — comparisons are pure arithmetic/boolean logic over already-extracted scalar features.

---

## Robust Input Handling

Every extractor defensively coerces:

* `None` → `0.0` (numeric) or `False` (boolean)
* `NaN` → `0.0` (numeric) or `False` (boolean)
* non-numeric / non-boolean types → safely coerced or defaulted, never raising

All functions return stable dict outputs to prevent pipeline failures during large-scale extraction.

---

## Stateless and Parallelizable

Each extractor is an independent pure function:

* no shared state
* no persistent memory
* no dependency between extractors

This enables unit testing, multiprocessing, and efficient `pandas.apply` execution, matching the design principles of `Text_Features`.

---

## Second-Stage, Not Source-Agnostic

Unlike `Text_Features`, these extractors are **not** source-agnostic — they specifically model the `response_a` vs. `response_b` relationship (plus, for a few modules, the `user_prompt` vs. response relationship). They must run after the feature stage(s) that produce their inputs, and the `Run_Comparison_Features` runner enforces this via column-existence gating rather than a hard dependency error.

---

## Prefer Asymmetry Signals Over Fixed Preference

For binary style/content flags whose value is conditional on prompt category (e.g. `has_latex`, `is_detailed`, `has_step_by_step`, `is_natural_text`), the corresponding comparator does not encode "having the flag is better." It only reports whether the two sides agree or disagree — the direction of preference is left for a downstream model to learn conditionally on prompt category, rather than being baked into the feature.

---

## ML-Ready Output

All generated features are returned as:

* `float`
* `bool`

flat dictionaries, directly usable in LightGBM, XGBoost, linear models, ranking models, or neural classifiers without additional encoding.

---

# Intended Use Cases

## Pairwise Preference Modeling

The primary purpose of this module: convert independent `Text_Features` for `response_a` and `response_b` into direct comparison signals suitable as inputs to a reward model or preference classifier.

## Format & Length Compliance Scoring

`Format_Compliance_Comparison` and `Length_Compliance_Comparison` connect prompt-side requests (`Text_Features.Format_Request_Detection`, `Text_Features.Length_Request_Detection`) to response-side delivery, producing a compliance signal per side.

## Failure/Quality Asymmetry Detection

`Refusal_Comparison`, `Near_Empty_Comparison`, and `Truncation_Comparison` flag the common "one side failed outright" and "both sides failed" (`both_bad`) cases that are decisive in pairwise human preference regardless of content quality.

## Language/Script Consistency Checking

`Prompt_Language_Match_Comparison` and `Prompt_Script_Match_Comparison` flag responses that answered in the wrong language relative to the prompt, with the script-based version serving as a fallback when full language detection is unavailable or unreliable.

---

# Notes

* All comparison modules assume `a_*`/`b_*` inputs were produced by the corresponding `Text_Features` extractor of the same base name (e.g. `Word_Count_Comparison` expects `Text_Features.Writing_Style_Features.word_count`) — see the [Text_Features README](../Text_Features/README.md) for those definitions.
* `EPSILON = 1e-6` is used uniformly across all ratio computations to avoid division-by-zero when the denominator side's metric is `0`.
* Boolean comparators intentionally omit `_diff`/`_ratio` fields; see [Template B](#b-boolean-agreement-template-both_--only_one_--neither_) for the rationale documented in each module's docstring.
* `Format_Richness_Comparison` is the only module that also computes and exposes each side's raw composite score (`a_format_richness`/`b_format_richness`), not just the comparison — because that composite doesn't exist anywhere upstream.
* The comparison feature pipeline operates on copies of DataFrames and does not mutate the original input.
* The runner's column-existence gating means comparison features are additive and non-fatal: missing upstream columns reduce the output schema rather than raising an exception.
