import pandas as pd

from Feature_Extraction.Comparison_Features.Word_Count_Comparison import extract_word_count_comparison
from Feature_Extraction.Comparison_Features.Token_Count_Comparison import extract_token_count_comparison
from Feature_Extraction.Comparison_Features.Sentence_Count_Comparison import extract_sentence_count_comparison
from Feature_Extraction.Comparison_Features.Paragraph_Count_Comparison import extract_paragraph_count_comparison
from Feature_Extraction.Comparison_Features.Code_Delimiters_Comparison import extract_code_delimiters_comparison
from Feature_Extraction.Comparison_Features.Punctuation_Count_Comparison import extract_punctuation_count_comparison
from Feature_Extraction.Comparison_Features.Repetition_Density_Comparison import extract_repetition_density_comparison
from Feature_Extraction.Comparison_Features.Format_Richness_Comparison import extract_format_richness_comparison
from Feature_Extraction.Comparison_Features.Prompt_Language_Match_Comparison import extract_prompt_language_match_comparison
from Feature_Extraction.Comparison_Features.Prompt_Script_Match_Comparison import extract_prompt_script_match_comparison
from Feature_Extraction.Comparison_Features.Refusal_Comparison import extract_refusal_comparison
from Feature_Extraction.Comparison_Features.Near_Empty_Comparison import extract_near_empty_comparison


# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------
def run_all_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    word_count_columns = ["a_word_count", "b_word_count"]

    if all(col in df.columns for col in word_count_columns):
        comparison_features = df.apply(
            lambda row: extract_word_count_comparison(
                row["a_word_count"], row["b_word_count"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    token_count_columns = ["a_token_count", "b_token_count"]

    if all(col in df.columns for col in token_count_columns):
        comparison_features = df.apply(
            lambda row: extract_token_count_comparison(
                row["a_token_count"], row["b_token_count"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    sentence_count_columns = ["a_sentence_count", "b_sentence_count"]

    if all(col in df.columns for col in sentence_count_columns):
        comparison_features = df.apply(
            lambda row: extract_sentence_count_comparison(
                row["a_sentence_count"], row["b_sentence_count"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    paragraph_count_columns = ["a_paragraph_count", "b_paragraph_count"]

    if all(col in df.columns for col in paragraph_count_columns):
        comparison_features = df.apply(
            lambda row: extract_paragraph_count_comparison(
                row["a_paragraph_count"], row["b_paragraph_count"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    code_delimiters_columns = ["a_code_delimiters", "b_code_delimiters"]

    if all(col in df.columns for col in code_delimiters_columns):
        comparison_features = df.apply(
            lambda row: extract_code_delimiters_comparison(
                row["a_code_delimiters"], row["b_code_delimiters"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    punctuation_count_columns = ["a_punctuation_count", "b_punctuation_count"]

    if all(col in df.columns for col in punctuation_count_columns):
        comparison_features = df.apply(
            lambda row: extract_punctuation_count_comparison(
                row["a_punctuation_count"], row["b_punctuation_count"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    repetition_density_columns = ["a_repetition_density", "b_repetition_density"]

    if all(col in df.columns for col in repetition_density_columns):
        comparison_features = df.apply(
            lambda row: extract_repetition_density_comparison(
                row["a_repetition_density"], row["b_repetition_density"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    header_level_columns = [f"header_h{level}" for level in range(1, 7)]
    format_richness_columns = (
        ["dataset_a_bold", "dataset_b_bold"]
        + [f"dataset_a_{col}" for col in header_level_columns]
        + [f"dataset_b_{col}" for col in header_level_columns]
        + ["dataset_a_list_ordered", "dataset_a_list_unordered"]
        + ["dataset_b_list_ordered", "dataset_b_list_unordered"]
        + ["a_table_count", "b_table_count"]
    )

    if all(col in df.columns for col in format_richness_columns):
        comparison_features = df.apply(
            lambda row: extract_format_richness_comparison(
                a_bold=row["dataset_a_bold"],
                a_headers=sum(row[f"dataset_a_{col}"] for col in header_level_columns),
                a_list_items=row["dataset_a_list_ordered"] + row["dataset_a_list_unordered"],
                a_tables=row["a_table_count"],
                b_bold=row["dataset_b_bold"],
                b_headers=sum(row[f"dataset_b_{col}"] for col in header_level_columns),
                b_list_items=row["dataset_b_list_ordered"] + row["dataset_b_list_unordered"],
                b_tables=row["b_table_count"],
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    prompt_lang_match_columns = ["prompt_primary_language", "a_primary_language", "b_primary_language"]

    if all(col in df.columns for col in prompt_lang_match_columns):
        comparison_features = df.apply(
            lambda row: extract_prompt_language_match_comparison(
                row["prompt_primary_language"],
                row["a_primary_language"],
                row["b_primary_language"],
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    prompt_script_match_columns = ["prompt_script", "a_script", "b_script"]

    if all(col in df.columns for col in prompt_script_match_columns):
        comparison_features = df.apply(
            lambda row: extract_prompt_script_match_comparison(
                row["prompt_script"],
                row["a_script"],
                row["b_script"],
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    refusal_columns = ["a_has_refusal", "b_has_refusal"]

    if all(col in df.columns for col in refusal_columns):
        comparison_features = df.apply(
            lambda row: extract_refusal_comparison(
                row["a_has_refusal"], row["b_has_refusal"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    near_empty_columns = ["a_is_near_empty", "b_is_near_empty"]

    if all(col in df.columns for col in near_empty_columns):
        comparison_features = df.apply(
            lambda row: extract_near_empty_comparison(
                row["a_is_near_empty"], row["b_is_near_empty"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    return df


__all__ = ["run_all_comparison_features"]
