import pandas as pd

from Feature_Extraction.Comparison_Features.Word_Count_Comparison import extract_word_count_comparison


# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------
def run_all_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = ["a_word_count", "b_word_count"]

    if all(col in df.columns for col in required_columns):
        comparison_features = df.apply(
            lambda row: extract_word_count_comparison(
                row["a_word_count"], row["b_word_count"]
            ),
            axis=1,
        ).apply(pd.Series)

        df = pd.concat([df, comparison_features], axis=1)

    return df


__all__ = ["run_all_comparison_features"]
