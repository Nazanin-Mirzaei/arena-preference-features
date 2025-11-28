# Unified Feature Extraction Module – Final Version (6 Features)  
LLM Comparative Evaluation Framework –

Supervisor Instruction:  
"All groups must use a single shared feature module and clearly document what has been done."

Tasks Completed  
- Collected all feature implementations from all groups  
- Merged into one clean, robust, and fully tested module  
- Fixed language detection priority bug (Persian/Arabic now correctly prioritized)  
- Added 6th public feature: `detectLanguage()` (previously internal, now exposed)  
- All functions documented, type-hinted, and production-ready  

### Repository Structure

### Available Features (All Groups Must Use These)

| # | Feature                    | Function                        | Return Type       | Key Highlights                                      |
|---|----------------------------|----------------------------------|-------------------|-----------------------------------------------------|
| 1 | Writing Style Analysis     | `analyzeWritingStyle(text)`      | `dict`            | `is_detailed`, `has_step_by_step`, word/sentence stats |
| 2 | Language Detection         | `detectLanguage(text)`           | `str`             | Returns: `persian_arabic` / `chinese` / `latin` / `other` / `unknown` |
| 3 | Token Counting             | `countTokens(text)`              | `int`             | Language-adaptive: Persian (~2.4), Latin (~3.8), Chinese (~1.2) |
| 4 | Code Block Detection       | `detectCodeBlock(text)`          | `bool`            | Supports fenced ```, indentation, inline `, HTML, heuristics |
| 5 | Emoji Detection            | `detectEmojis(text)`             | `bool`            | Full Unicode range support                          |
| 6 | Table Detection            | `detectTables(text)`             | `bool`            | Markdown tables + HTML tables                       |

### Usage Example (Mandatory for All Groups)
```python
from analysis_utils import (
    analyzeWritingStyle, detectLanguage, countTokens,
    detectCodeBlock, detectEmojis, detectTables
)

response = "متن پاسخ مدل اینجا قرار می‌گیرد..."

print("Language:", detectLanguage(response))                    # → persian_arabic
print("Detailed:", analyzeWritingStyle(response)["is_detailed"])
print("Step-by-step:", analyzeWritingStyle(response)["has_step_by_step"])
print("Token count ≈", countTokens(response))
print("Has code:", detectCodeBlock(response))
print("Has emoji:", detectEmojis(response))
print("Has table:", detectTables(response))
