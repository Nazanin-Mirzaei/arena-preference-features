# Metadata Features Extraction Module

## Overview

This module provides a **comprehensive suite of feature extractors** designed to capture structural, formatting, conversational, and interaction-type properties from two dataset-provided metadata columns in the LM-Arena dataset: conversation-level metadata (`conv_metadata`) and interaction-category labels (`category_tag`). Each extractor operates on a pre-parsed dictionary and returns a flat dictionary of interpretable, ML-ready features.

These features capture latent signals about LLM response behavior that are not directly observable from raw text alone, including formatting tendencies, conversational depth, token allocation patterns, and the interaction type of the prompt itself (creative writing, math, instruction-following, etc.).

---

## Module Structure

```
Metadata_Features/
│
├── Bold.py                 # Bold text frequency and style preference
├── Headers.py              # Markdown heading depth and distribution
├── Lists.py                # Ordered / unordered list segmentation
├── Tokens.py                # Assistant, user, and context token counts
├── Conversation.py         # Turn-taking dynamics and density
├── Dataset_Baseline.py     # Unified baseline extractor (all formatting fields)
├── Category_Tag.py         # Flattened interaction-type labels (category_tag column)
│
└── README.md
```

---

## Input Format

Most feature extractors in this module accept a single `conv_metadata` dictionary extracted from the LM-Arena dataset:

```python
meta: Dict[str, Any]
```

The metadata dictionary is expected to contain fields such as:

- `bold_count_a`, `bold_count_b`
- `list_count_a`, `list_count_b`
- `header_count_a`, `header_count_b`
- `sum_assistant_a_tokens`, `sum_assistant_b_tokens`
- `sum_user_tokens`, `context_a_tokens`, `context_b_tokens`
- `turns`

Each extractor gracefully handles `None`, `NaN`, missing keys, and non-dict inputs by returning a default zero-initialized feature vector.

`Category_Tag.py` is the one exception: it reads from a **separate** dataset column, `category_tag`, not `conv_metadata`. See [Category Tag Features](#7-category-tag-features-category_tagpy) below.

---

## Feature Descriptions

### 1. Bold Text Features (`Bold.py`)

**Function:** `extract_bold_features(meta)`

Analyzes bold formatting usage across both responses (A and B) using Markdown conventions (`**text**` and `__text__`).

| Feature | Type | Description |
|---------|------|-------------|
| `bold_total` | `int` | Total count of bold spans across both responses |
| `bold_style_preference` | `str` | Preferred bold syntax: `"**"`, `"__"`, `"equal"`, or `"none"` |
| `emphasis_intensity` | `float` | Log-scaled emphasis score: `log(1 + bold_total)` |

---

### 2. Header Features (`Headers.py`)

**Function:** `extract_header_features(meta)`

Extracts Markdown heading structure (H1–H6) from both responses.

| Feature | Type | Description |
|---------|------|-------------|
| `total_headers` | `int` | Total number of heading occurrences across both responses |
| `header_depth_score` | `int` | Weighted structural depth: sum of `level × count` per heading level |

The depth score assigns greater weight to deeper headings (e.g., H4 > H1), providing a proxy for hierarchical document complexity.

---

### 3. List Features (`Lists.py`)

**Function:** `extract_list_features(meta)`

Captures the prevalence and distribution of ordered (numbered) and unordered (bulleted) lists.

| Feature | Type | Description |
|---------|------|-------------|
| `total_list_items` | `int` | Total list items across both responses |
| `ordered_ratio` | `float` | Proportion of list items that are ordered `[0, 1]` |
| `unordered_ratio` | `float` | Proportion of list items that are unordered `[0, 1]` |

---

### 4. Token Features (`Tokens.py`)

**Function:** `extract_token_features(meta)`

Provides token-level analytics that capture response verbosity and input-output balance.

| Feature | Type | Description |
|---------|------|-------------|
| `assistant_tokens_total` | `int` | Sum of assistant A and B tokens |
| `user_tokens` | `int` | Total user-provided tokens |
| `context_tokens_total` | `int` | Sum of context tokens for both responses |
| `assistant_token_ratio` | `float` | Assistant-to-user token ratio: `assistant_total / max(user_tok, 1)` |

---

### 5. Conversation Dynamics (`Conversation.py`)

**Function:** `extract_conversation_dynamics(meta)`

Captures the interactive structure of multi-turn conversations.

| Feature | Type | Description |
|---------|------|-------------|
| `turns` | `int` | Total number of conversational turns |
| `is_multi_turn` | `bool` | Flag indicating more than one turn |
| `turn_density` | `float` | Mean assistant tokens per turn: `assistant_tokens / turns` |

---

### 6. Dataset Baseline Extractor (`Dataset_Baseline.py`)

**Function:** `extract_dataset_format_features(meta)`

A unified extractor that mirrors all formatting-related fields natively present in the LM-Arena dataset's `conv_metadata`. Designed for validation and comparison against module-level extractors.

| Feature | Type | Description |
|---------|------|-------------|
| `dataset_a_bold` / `dataset_b_bold` | `int` | Bold count per response (star-style only) |
| `dataset_a_list_ordered` / `dataset_b_list_ordered` | `int` | Ordered list items per response |
| `dataset_a_list_unordered` / `dataset_b_list_unordered` | `int` | Unordered list items per response |
| `dataset_a_header_h1`–`h6` / `dataset_b_header_h1`–`h6` | `int` | Header counts per heading level per response |
| `dataset_a_tokens` / `dataset_b_tokens` | `int` | Assistant token counts per response |
| `dataset_a_context_tokens` / `dataset_b_context_tokens` | `int` | Context token counts per response |

---

### 7. Category Tag Features (`Category_Tag.py`)

**Function:** `extract_category_tag_features(value)`

Flattens the dataset-provided `category_tag` column into individual interpretable feature columns. Unlike every other extractor in this module, its input is the raw `category_tag` cell (a dict or its string representation), **not** a parsed `conv_metadata` dictionary — `Run_Metadata_Features.py` applies it to the `category_tag` column directly rather than the shared `conv_metadata` registry.

`category_tag` contains human/model labels describing the interaction type of the prompt, grouped into versioned sub-objects (e.g. `criteria_v0.1`, `if_v0.1`). Group lookup strips the version suffix (`criteria` matches `criteria_v0.1`, `criteria_v0.2`, ...), so extraction survives dataset version bumps, and output feature names are version-free for long-term stability.

| Feature | Type | Description |
|---------|------|-------------|
| `cat_complexity` | `bool` | Prompt requires multi-step or non-trivial reasoning |
| `cat_creativity` | `bool` | Prompt rewards creative/original output |
| `cat_domain_knowledge` | `bool` | Prompt requires specialized domain knowledge |
| `cat_problem_solving` | `bool` | Prompt is a problem-solving task |
| `cat_real_world` | `bool` | Prompt is grounded in a real-world scenario |
| `cat_specificity` | `bool` | Prompt demands a specific, precise answer |
| `cat_technical_accuracy` | `bool` | Prompt requires technically accurate output |
| `cat_creative_writing` | `bool` | Prompt is a creative writing task |
| `cat_math` | `bool` | Prompt is a math task |
| `cat_if` | `bool` | Prompt contains explicit instruction-following constraints |
| `cat_if_score` | `int` | Instruction-following complexity score (0–4) |

These are free, pre-labeled interaction-type signals — no heuristics or NLP involved, just structural flattening. They let a downstream classifier learn **conditional** preferences: e.g. verbosity or step-by-step structure may help on `cat_complexity`/`cat_problem_solving` prompts but hurt on prompts asking for a short, precise answer.

Fully defensive: `None`, `NaN`, malformed strings, missing sub-groups, and unexpected value types all fall back to `False` / `0` per field rather than raising.

---

## Usage Examples

### Single Feature Extraction

```python
from Metadata_Features.Bold import extract_bold_features
from Metadata_Features.Headers import extract_header_features

meta = df.loc[0, "conv_metadata"]

bold_feats = extract_bold_features(meta)
header_feats = extract_header_features(meta)
```

### Batch Processing with Pandas

```python
from Metadata_Features.Conversation import extract_conversation_dynamics
from Metadata_Features.Tokens import extract_token_features

df["turns"] = df["conv_metadata"].apply(
    lambda m: extract_conversation_dynamics(m)["turns"]
)
df["assistant_tokens"] = df["conv_metadata"].apply(
    lambda m: extract_token_features(m)["assistant_tokens_total"]
)
```

### Category Tag Extraction

```python
from Metadata_Features.Category_Tag import extract_category_tag_features

category_tag = df.loc[0, "category_tag"]

cat_feats = extract_category_tag_features(category_tag)
# {"cat_complexity": True, "cat_creativity": False, ..., "cat_if_score": 2}
```

### Full Metadata Pipeline via Feature Runner

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
```

`run_all_metadata_features` applies the `conv_metadata`-based extractors first, then separately applies `Category_Tag.py` to the `category_tag` column if present — both are column-gated, so either can be safely absent from the input DataFrame.

---

## Design Principles

### Lightweight Implementation

All extractors rely solely on the Python standard library and basic arithmetic operations. No external NLP dependencies (transformers, spaCy, NLTK) are required.

### Robust Input Handling

Every function defensively guards against:
- `None` / `NaN` inputs
- Missing dictionary keys
- Non-dict or malformed metadata
- Type inconsistencies (e.g., string values where integers are expected)

### Stateless and Testable

Each extractor is a pure function with no side effects: same input always produces identical output, enabling straightforward unit testing and parallelization via `pandas.apply`.

### ML-Ready Output

All features are returned as primitive types (`int`, `float`, `bool`, `str`) suitable for direct ingestion into ML pipelines (LightGBM, XGBoost, linear models, or neural networks).

---

## Intended Use Cases

- **Pairwise response modeling:** Features such as `bold_total`, `header_depth_score`, and `ordered_ratio` capture stylistic differences between responses A and B.
- **Conversation-level quality analysis:** `turns`, `turn_density`, and `assistant_token_ratio` provide insight into conversational depth and engagement.
- **Baseline validation:** `Dataset_Baseline.py` is used to verify consistency between extracted features and dataset-provided fields.
- **Feature engineering for reward models:** Metadata features serve as high-level structural priors for preference and reward modeling tasks.
- **Conditional preference modeling:** `Category_Tag.py`'s `cat_*` features let downstream models (or precomputed interaction terms) condition other features — length, formatting, structure — on the type of prompt being answered, rather than treating those signals as universally good or bad.

---

## Notes

- Bold counts in `Dataset_Baseline.py` only track star-style (`**`) bold syntax, while `Bold.py` also captures underscore-style (`__`).
- Token features are heuristic estimates derived from the dataset's metadata token counters, not from a standalone tokenizer.
- All extractors in this module except `Category_Tag.py` operate on **pre-parsed `conv_metadata` dictionaries**, not raw JSON strings. Use `ast.literal_eval` or `json.loads` if metadata is stored as a string — or pass the raw string directly, since every extractor's own parsing helper (or the runner's `parse_metadata`) handles that internally.
- `Category_Tag.py` operates on the separate `category_tag` column and handles its own parsing (`parse_category_tag`) independently of `conv_metadata` parsing.
- `Category_Tag.py`'s group lookup is version-suffix-tolerant (`criteria` matches `criteria_v0.1`, `criteria_v0.2`, ...), so it does not need to be updated when the dataset's `category_tag` schema version changes, as long as the sub-group and field names stay the same.
