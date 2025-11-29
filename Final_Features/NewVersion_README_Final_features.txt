LLM Evaluation – Unified Feature Extraction

All modules have been merged, cleaned, optimized, and normalized from multiple team submissions.


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



