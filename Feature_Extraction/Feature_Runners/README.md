# 📊 Feature Extraction Pipeline

## Overview

This module provides a **lightweight, scalable, and dependency-free feature extraction pipeline** for analyzing LLM responses and metadata in large-scale datasets.

It is designed for:

* LLM evaluation datasets (e.g., Arena-style comparisons)
* Response quality analysis (A/B comparisons)
* Structural, linguistic, and statistical feature extraction
* Fast batch processing with pandas

---

# 🧠 Pipeline Architecture

The system is divided into three main layers:

```
Feature_Extraction/
│
├── Response_Features/        # Low-level feature extractors
├── Metadata_Features/        # Metadata feature extractors
└── Feature_Runners/          # High-level execution pipelines
```

---

# ⚙️ Feature Runners

## 1. Run_Response_Features.py

### Purpose

Extracts **textual and structural features** from:

* `response_a`
* `response_b`

### Output format

Each feature is duplicated with prefixes:

```
a_feature_name
b_feature_name
```

### Example features

* token_count
* sentence_count
* repetition_density
* paragraph_count
* code_delimiters
* table_count
* writing style metrics

### Usage

```python
from Feature_Extraction.Feature_Runners.Run_Response_Features import run_all_response_features

df = run_all_response_features(df)
```

---

## 2. Run_Metadata_Features.py

### Purpose

Extracts structured features from conversation metadata (`conv_metadata`).

### Key characteristics

* Parses raw dict / string metadata
* Applies multiple feature extractors
* Flattens nested outputs automatically

### Example features

* bold usage
* headers
* lists
* token structure
* conversation dynamics

### Usage

```python
from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features

df = run_all_metadata_features(df)
```

---

## 3. Run_All_Features.py

### Purpose

Unified pipeline that runs:

* Metadata features
* Response A/B features

### Behavior

Automatically checks for required columns:

* `conv_metadata`
* `response_a`
* `response_b`

### Usage

```python
from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = run_all_features(df)
```

---

# 🧩 Feature Design Principles

## 1. Stateless Functions

Each feature function:

* takes `text` as input
* returns `dict` or scalar
* has no side effects

---

## 2. Safe Aggregation

All nested outputs are handled using:

* `safe_update()`
* `_flatten()` where needed

This ensures:

* no crashes from nested dicts
* consistent output schema

---

## 3. Robust Input Handling

All features support:

* `None`
* `NaN`
* empty strings
* non-string inputs

---

## 4. Dual Response Design

Response features are always split into:

* `response_a → a_*`
* `response_b → b_*`

This enables:

* direct A/B comparison
* model ranking features
* difference analysis

---

# 🚀 Output Example

After running the full pipeline:

```python
df = run_all_features(df)
```

You will get columns like:

### Response A

```
a_token_count
a_sentence_count
a_repetition_density
a_paragraph_count
```

### Response B

```
b_token_count
b_sentence_count
b_repetition_density
b_paragraph_count
```

### Metadata

```
bold_count
header_depth
conversation_turns
list_density
```

---

# ⚠️ Known Design Constraints

* Some features return nested dictionaries → must be flattened
* Feature consistency is required (dict vs scalar issues must be avoided)
* Column naming must remain stable for downstream ML pipelines

---

# 🧪 Recommended Usage Pattern

```python
import pandas as pd

from Feature_Extraction.Feature_Runners.Run_All_Features import run_all_features

df = pd.read_csv("data.csv")

df = run_all_features(df)

print(df.shape)
print(df.columns[:20])
```

---

# 📌 Summary

This pipeline provides:

✔ scalable feature extraction
✔ dual-response analysis (A/B)
✔ metadata + text integration
✔ safe handling of nested outputs
✔ clean modular architecture
