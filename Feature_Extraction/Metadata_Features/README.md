# 📊 Metadata Feature Extraction

This module provides a **clean, modular, and extensible feature extraction pipeline** for the `conv_metadata` field in the LM-Arena dataset.

It extracts structured features that describe:

* Formatting behavior
* Conversation structure
* Token usage
* Text organization patterns

These features are designed for downstream tasks such as:

* Response quality modeling
* Pairwise ranking
* Behavioral analysis of LLM outputs
* Feature-based ML models (LightGBM / XGBoost)

---

# 📁 Module Structure

```text
metadata/
│
├── bold.py                  # Bold/Emphasis statistics
├── conversation.py         # Turn-level conversation dynamics
├── headers.py              # Markdown heading structure
├── lists.py                # Ordered / unordered list patterns
├── tokens.py               # Token usage statistics
├── dataset_baseline.py     # Ground-truth dataset-provided features
```

---

# ⚙️ Input Format

All feature functions operate on a single object:

```python
meta: dict
```

This comes from parsing the `conv_metadata` column:

```python
import ast

def parse_metadata(meta):
    try:
        return ast.literal_eval(meta)
    except Exception:
        return {}
```

---

# 🔄 Example Usage

## Single feature extraction

```python
from metadata.bold import extract_bold_features

features = extract_bold_features(meta_dict)
print(features)
```

Output:

```python
{
    "bold_total": 5,
    "bold_style_preference": "**",
    "emphasis_intensity": 0.005
}
```

---

## Full pipeline usage (recommended)

```python
import pandas as pd
from metadata.bold import extract_bold_features
from metadata.conversation import extract_conversation_dynamics

def run_features(meta):
    return {
        **extract_bold_features(meta),
        **extract_conversation_dynamics(meta),
    }
```

---

# 📌 Feature Descriptions

## 🔹 Bold Features (`bold.py`)

Extracts emphasis patterns:

* total bold occurrences
* style preference (`**` vs `__`)
* normalized emphasis intensity

---

## 🔹 Conversation Dynamics (`conversation.py`)

Captures structure of dialogue:

* number of turns
* multi-turn flag
* token density per turn

---

## 🔹 Headers (`headers.py`)

Measures markdown structure:

* total headings (H1–H6)
* weighted depth score

---

## 🔹 Lists (`lists.py`)

Captures list formatting behavior:

* ordered vs unordered ratio
* total list items

---

## 🔹 Tokens (`tokens.py`)

Measures token usage:

* assistant / user / context tokens
* token ratios and totals

---

## 🔹 Dataset Baseline (`dataset_baseline.py`)

Replicates dataset-provided engineered features:

* bold counts (a/b)
* list counts
* header distributions
* token statistics

Used for:

> validation and feature consistency checks

---

# 🧠 Design Philosophy

This module is built with:

* **Single responsibility per file**
* **Pure function design**
* **No side effects**
* **Fast vectorizable pipeline compatibility**
* **Easy extension for new features**

---

# 🚀 Pipeline Integration

Recommended usage:

```python
meta = df["conv_metadata"].apply(parse_metadata)
features = meta.apply(extract_bold_features)
```

Or full pipeline:

```python
df = run_all_metadata_features(df)
```

---

# ⚠️ Notes

* All functions expect a **parsed dict**, not raw string
* Missing or corrupted metadata is safely handled
* Designed for large-scale datasets (100K+ rows)

---

# 📈 Future Extensions

Planned additions:

* sentiment-based metadata features
* coherence / repetition metrics
* hallucination indicators
* response entropy features
