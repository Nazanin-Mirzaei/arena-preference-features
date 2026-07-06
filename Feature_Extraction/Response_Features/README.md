# 📊 Response Features Extraction Module

This module provides a **comprehensive, modular, and dependency-light feature extraction pipeline** for analyzing LLM-generated responses.

It focuses on extracting **structural, linguistic, formatting, and behavioral signals** directly from raw model outputs.

These features are designed for downstream tasks such as:

* Response quality modeling
* Pairwise ranking (LM-Arena style evaluation)
* LLM behavioral analysis
* Supervised ML models (LightGBM / XGBoost / Neural classifiers)
* Style vs reasoning classification

---

# 📁 Module Structure

```
Response_Features/
│
├── Code_Delimiter_Count.py        # Code delimiter frequency signals
├── Emoji_Features.py              # Emoji presence and counts
├── Has_Latex.py                   # LaTeX / math expression detection
├── Interaction_Features.py        # Conversational interaction patterns
├── Is_Code_Block.py               # Code block detection (Markdown/HTML)
├── Is_Natural_Text.py             # Natural text vs code classifier
├── Language_Detection.py         # Language + multilingual detection
├── Paragraph_Count.py            # Paragraph segmentation count
├── Paragraph_Statistics.py       # Paragraph length statistics
├── Punctuation_Count.py          # Punctuation density
├── Repetition.py                 # Lexical repetition density
├── Sentence_Count.py             # Sentence counting
├── Sentence_Length.py            # Sentence length distribution
├── Sentence_Length_Stats.py      # Sentence length statistics (STD)
├── Sentence_Paragraph_Stats.py   # Sentence-per-paragraph variation
├── Table_Detection.py            # Markdown / HTML table detection
├── Token_Features.py             # Token estimation features
├── Writing_Style_Features.py     # Writing style + structure analysis
│
└── README.md
```

---

# ⚙️ Input Format

All feature extractors operate on a **single raw text input**:

```python
text: Any
```

Typical usage:

```python
response_text = df["response_a"]
```

or:

```python
response_text = df["response_b"]
```

Each feature function is applied independently per response.

---

# 🔄 Example Usage

## Single feature

```python
from Response_Features.Code_Delimiter_Count import count_code_delimiters

value = count_code_delimiters(text)
```

---

## Full feature extraction pipeline

```python
from Response_Features import *

features = {
    "code_delimiters": count_code_delimiters(text),
    "emoji_count": count_emojis(text),
    "has_latex": detect_latex(text),
    "is_code_block": detect_code_block(text),
    "repetition": repetition_density(text),
}
```

---

## Batch usage (Pandas)

```python
df["emoji_count"] = df["response"].apply(count_emojis)
df["has_code"] = df["response"].apply(detect_code_block)
```

---

# 📌 Feature Categories

## 🧱 1. Structural Features

Capture formatting and document structure:

* Paragraph_Count
* Sentence_Count
* Sentence_Length
* Sentence_Length_Stats
* Sentence_Paragraph_Stats
* Punctuation_Count

---

## 💻 2. Code & Technical Content

Detect programming or structured outputs:

* Code_Delimiter_Count
* Is_Code_Block
* Table_Detection
* Has_Latex

---

## 🧠 3. Linguistic & Language Features

Measure language properties:

* Language_Detection
* Token_Features
* Repetition
* Is_Natural_Text

---

## 💬 4. Interaction & Discourse Features

Capture conversational behavior:

* Interaction_Features
* Writing_Style_Features

Includes:

* step-by-step reasoning detection
* conclusion patterns
* prompt engagement signals

---

## 🎭 5. Stylistic Features

Capture writing style and complexity:

* Writing_Style_Features
* Sentence_Length_Stats
* Paragraph_Statistics

---

## 😊 6. Emotional / Symbolic Features

* Emoji_Features

Includes:

* emoji presence
* emoji count
* symbolic expression signals

---

# 🧠 Design Philosophy

This module is built around:

### ✔ Lightweight Design

* No heavy NLP dependencies (except optional language detection)

### ✔ Robustness

* Handles:

  * None
  * NaN
  * empty strings
  * noisy LLM outputs

### ✔ Independence

Each feature is:

* Stateless
* Independent
* Parallelizable

### ✔ ML-Ready Output

All outputs are:

* scalar values
* boolean flags
* normalized statistics

---

# ⚙️ Typical Pipeline Usage

```python
def extract_response_features(text):
    return {
        "emoji": count_emojis(text),
        "latex": detect_latex(text),
        "code_block": detect_code_block(text),
        "repetition": repetition_density(text),
        "sentences": count_sentences(text),
        "paragraphs": count_paragraphs(text),
    }
```

---

# 📈 Expected Use Cases

This module is especially useful for:

### 🔹 LLM Evaluation (LM-Arena style)

* Pairwise response comparison
* Win/loss prediction

### 🔹 Quality Scoring

* verbosity vs conciseness
* structure vs chaos detection

### 🔹 Behavioral Analysis

* reasoning style detection
* formatting tendencies

### 🔹 Feature-based ML models

* LightGBM / XGBoost classifiers
* ranking models
* reward models

---

# ⚠️ Notes

* All functions assume **raw text input**
* No preprocessing required before calling functions
* Language detection may require `langdetect`
* Token features are heuristic (not exact tokenizer-based)
* Some regex-based features may be sensitive to noisy text

---
