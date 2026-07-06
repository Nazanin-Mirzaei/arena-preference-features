import pandas as pd

from Feature_Extraction.Response_Features.Is_Code_Block import is_code_block
from Feature_Extraction.Response_Features.Interaction_Features import extract_interaction_features
from Feature_Extraction.Response_Features.Code_Delimiter_Count import count_code_delimiters
from Feature_Extraction.Response_Features.Punctuation_Count import count_punctuation
from Feature_Extraction.Response_Features.Sentence_Count import count_sentences
from Feature_Extraction.Response_Features.Has_Latex import has_latex
from Feature_Extraction.Response_Features.Emoji_Features import detect_emoji, count_emojis
from Feature_Extraction.Response_Features.Token_Features import count_tokens
from Feature_Extraction.Response_Features.Is_Natural_Text import is_natural_text
# from Feature_Extraction.Response_Features.Language_Detection import (
#     detect_language,
#     detect_multi_language,
#     detect_response_language,
#     detect_prompt_language,
# )
from Feature_Extraction.Response_Features.Sentence_Length import extract_sentence_length_features
from Feature_Extraction.Response_Features.Paragraph_Count import extract_paragraph_features
from Feature_Extraction.Response_Features.Paragraph_Statistics import extract_paragraph_statistics
from Feature_Extraction.Response_Features.Repetition import compute_repetition_density
from Feature_Extraction.Response_Features.Sentence_Length_Stats import compute_sentence_length_std
from Feature_Extraction.Response_Features.Sentence_Paragraph_Stats import compute_sentence_per_paragraph_std
from Feature_Extraction.Response_Features.Table_Detection import detect_tables, count_tables
from Feature_Extraction.Response_Features.Writing_Style_Features import extract_writing_style_features


# -------------------------------------------------
# Main Feature Function (safe wrapper)
# -------------------------------------------------
def extract_all_response_features(text: str) -> dict:
    """
    Extract all response-level features from a single text.
    """

    features = {}

    # --- simple scalar features ---
    features["code_delimiters"] = count_code_delimiters(text)
    features["punctuation_count"] = count_punctuation(text)
    features["sentence_count"] = count_sentences(text)
    features["has_latex"] = has_latex(text)

    # emoji (two outputs)
    features["has_emoji"] = detect_emoji(text)
    features["emoji_count"] = count_emojis(text)

    # token + repetition
    features["token_count"] = count_tokens(text)
    features["repetition_density"] = compute_repetition_density(text)

    # structure
    features["is_code_block"] = is_code_block(text)
    features["is_natural_text"] = is_natural_text(text)

    # tables
    features["has_table"] = detect_tables(text)
    features["table_count"] = count_tables(text)

    # paragraph features (dict outputs)
    features.update(extract_paragraph_features(text))
    features.update(extract_paragraph_statistics(text))

    # sentence features (dict outputs)
    features.update(extract_sentence_length_features(text))
    features["sentence_length_std"] = compute_sentence_length_std(text)
    features["sentence_per_paragraph_std"] = compute_sentence_per_paragraph_std(text)

    # interaction / writing style (dict outputs)
    features.update(extract_interaction_features(text))
    features.update(extract_writing_style_features(text))

    return features


# -------------------------------------------------
# Pipeline for DataFrame
# -------------------------------------------------
def run_all_response_features(
    df: pd.DataFrame,
    text_col: str = "response"
) -> pd.DataFrame:

    df = df.copy()

    feature_df = df[text_col].apply(extract_all_response_features).apply(pd.Series)

    df = pd.concat([df, feature_df], axis=1)

    return df


__all__ = ["extract_all_response_features", "run_all_response_features"]
