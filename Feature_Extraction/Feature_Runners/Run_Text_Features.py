import pandas as pd

from Language_Detection.api import detectLanguage
from Language_Detection.api import detectMultiLanguage

from Feature_Extraction.Text_Features.Script_Detection import detect_script

from Feature_Extraction.Text_Features.Interaction_Features import extract_interaction_features
from Feature_Extraction.Text_Features.Refusal_Detection import has_refusal
from Feature_Extraction.Text_Features.Near_Empty_Detection import is_near_empty
from Feature_Extraction.Text_Features.Is_Code_Block import is_code_block
from Feature_Extraction.Text_Features.Code_Delimiter_Count import count_code_delimiters
from Feature_Extraction.Text_Features.Punctuation_Count import count_punctuation
from Feature_Extraction.Text_Features.Sentence_Count import count_sentences
from Feature_Extraction.Text_Features.Has_Latex import has_latex
from Feature_Extraction.Text_Features.Emoji_Features import detect_emoji, count_emojis
from Feature_Extraction.Text_Features.Token_Features import count_tokens
from Feature_Extraction.Text_Features.Is_Natural_Text import is_natural_text
from Feature_Extraction.Text_Features.Sentence_Length import extract_sentence_length_features
from Feature_Extraction.Text_Features.Paragraph_Count import extract_paragraph_features
from Feature_Extraction.Text_Features.Paragraph_Statistics import extract_paragraph_statistics
from Feature_Extraction.Text_Features.Repetition import compute_repetition_density
from Feature_Extraction.Text_Features.Sentence_Length_Stats import compute_sentence_length_std
from Feature_Extraction.Text_Features.Sentence_Paragraph_Stats import compute_sentence_per_paragraph_std
from Feature_Extraction.Text_Features.Table_Detection import detect_tables, count_tables
from Feature_Extraction.Text_Features.Writing_Style_Features import extract_writing_style_features


# -------------------------------------------------
# Safe dictionary update
# -------------------------------------------------
def safe_update(features, output):
    if isinstance(output, dict):
        features.update(output)


# -------------------------------------------------
# Extract all text features from a single text
# -------------------------------------------------
def extract_all_Text_Features(text: str) -> dict:

    features = {}

    # Language features
    features["primary_language"] = detectLanguage(text)
    features["is_multilingual"] = detectMultiLanguage(text)["is_multi_language"]
    features["script"] = detect_script(text)

    # Code / formatting features
    features["code_delimiters"] = count_code_delimiters(text)
    features["punctuation_count"] = count_punctuation(text)
    features["has_latex"] = has_latex(text)

    # Emoji features
    features["has_emoji"] = detect_emoji(text)
    features["emoji_count"] = count_emojis(text)

    # Token and lexical features
    features["token_count"] = count_tokens(text)
    features["repetition_density"] = compute_repetition_density(text)

    # Content type features
    features["is_code_block"] = is_code_block(text)
    features["is_natural_text"] = is_natural_text(text)

    # Structure features
    features["sentence_count"] = count_sentences(text)

    features["has_table"] = detect_tables(text)
    features["table_count"] = count_tables(text)

    safe_update(features, extract_paragraph_features(text))
    safe_update(features, extract_paragraph_statistics(text))
    safe_update(features, extract_sentence_length_features(text))

    features["sentence_length_std"] = compute_sentence_length_std(text)
    features["sentence_per_paragraph_std"] = compute_sentence_per_paragraph_std(text)

    # Style and interaction features
    safe_update(features, extract_interaction_features(text))
    safe_update(features, extract_writing_style_features(text))

    # Refusal / failure features
    features["has_refusal"] = has_refusal(text)
    features["is_near_empty"] = is_near_empty(text)

    return features


# -------------------------------------------------
# Apply text features to prompt and responses
# -------------------------------------------------
def run_all_Text_Features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # User prompt features
    prompt_features = (
        df["user_prompt"]
        .apply(extract_all_Text_Features)
        .apply(pd.Series)
        .add_prefix("prompt_")
    )

    # Response A features
    response_a_features = (
        df["response_a"]
        .apply(extract_all_Text_Features)
        .apply(pd.Series)
        .add_prefix("a_")
    )

    # Response B features
    response_b_features = (
        df["response_b"]
        .apply(extract_all_Text_Features)
        .apply(pd.Series)
        .add_prefix("b_")
    )

    df = pd.concat(
        [
            df,
            prompt_features,
            response_a_features,
            response_b_features
        ],
        axis=1
    )

    return df


__all__ = [
    "extract_all_Text_Features",
    "run_all_Text_Features"
]
