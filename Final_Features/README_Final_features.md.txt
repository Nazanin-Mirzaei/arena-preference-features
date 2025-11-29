LLM Evaluation – Unified Feature Extraction

All modules have been merged, cleaned, optimized, and normalized from multiple team submissions.


The toolkit includes the following final feature modules:

| Module Name                        | Purpose                                                    | Dependencies           |
| ---------------------------------- | ---------------------------------------------------------- | ---------------------- |
| `Final_Emoji_Detection`            | Detects emoji presence and counts emojis                   | None                   |
| `Final_Table_Detection`            | Detects Markdown/HTML tables                               | None                   |
| `Final_Writing_Style_Analysis`     | Extracts writing-style metrics                             | None                   |
| `Final_CodeBlock_Detection`        | Detects code blocks in multiple formats                    | None                   |
| `Final_Format-Based_Text_Features` | Counts punctuation, robust sentence count, code delimiters | None                   |
| `Final_Feature_Token`              | Language-adaptive token estimation                         | None                   |
| `Final_Language_Detection`         | Single & multi-language detection                          | Optional: `langdetect` |






1. Installation Requirements

All modules are dependency-free except language detection.


If you want language detection:

pip install langdetect


Everything else works with Python standard library only.




2. How to Use Each Module

Every module exposes one main function that can be called on any string.
You can use them independently or integrate them inside your DataFrame pipeline.

Emoji Detection
from Final_Emoji_Detection import detectEmojis, count_emojis

has_emoji = detectEmojis(response_text)
emoji_count = count_emojis(response_text)


Table Detection
from Final_Table_Detection import detectTables

contains_table = detectTables(response_text)


Writing Style Analysis
from Final_Writing_Style_Analysis import analyzeWritingStyle

metrics = analyzeWritingStyle(response_text)

# metrics = {
#    "is_detailed": ...,
#    "has_step_by_step": ...,
#    "word_count": ...,
#    "sentence_count": ...,
#    "avg_words_per_sentence": ...
# }



Code Block Detection
from Final_CodeBlock_Detection import detectCodeBlock

has_code = detectCodeBlock(response_text)



Format-Based Text Features
from Final_Format-Based_Text_Features import (
    count_punctuation,
    count_code_delimiters,
    count_sentences_robust
)

punct = count_punctuation(response_text)
code_blocks = count_code_delimiters(response_text)
sentences = count_sentences_robust(response_text)



Token Estimation
from Final_Feature_Token import countTokens

estimated_tokens = countTokens(response_text)



Language Detection 
from Final_Language_Detection import detectLanguage, detectMultiLanguage

lang = detectLanguage(response_text)
ml = detectMultiLanguage(response_text)



3. Example: Applying All Features to a DataFrame
def extract_all_features(df, col="response"):
    df["emoji_count"] = df[col].apply(count_emojis)
    df["has_emoji"] = df[col].apply(detectEmojis)

    df["has_table"] = df[col].apply(detectTables)

    style = df[col].apply(analyzeWritingStyle)
    df["word_count"] = style.apply(lambda x: x["word_count"])
    df["sentence_count"] = style.apply(lambda x: x["sentence_count"])
    df["avg_wps"] = style.apply(lambda x: x["avg_words_per_sentence"])
    df["is_detailed"] = style.apply(lambda x: x["is_detailed"])
    df["has_step_by_step"] = style.apply(lambda x: x["has_step_by_step"])

    df["has_codeblock"] = df[col].apply(detectCodeBlock)

    df["punctuation_count"] = df[col].apply(count_punctuation)
    df["code_delimiters"] = df[col].apply(count_code_delimiters)
    df["sentences_robust"] = df[col].apply(count_sentences_robust)

    df["token_estimate"] = df[col].apply(countTokens)

    
    df["predicted_language"] = df[col].apply(detectLanguage)

    return df


