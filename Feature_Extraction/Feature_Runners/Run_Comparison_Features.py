import pandas as pd

from Feature_Extraction.Comparison_Features.Word_Count_Comparison import extract_word_count_comparison
from Feature_Extraction.Comparison_Features.Token_Count_Comparison import extract_token_count_comparison


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

    return df


__all__ = ["run_all_comparison_features"]
