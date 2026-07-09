import pandas as pd

from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features
from Feature_Extraction.Feature_Runners.Run_Text_Features import run_all_Text_Features
from Feature_Extraction.Feature_Runners.Run_Comparison_Features import run_all_comparison_features


def run_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------------------------
    # Metadata features
    # -------------------------
    if "conv_metadata" in df.columns:
        df = run_all_metadata_features(df)

    # -------------------------
    # Text features
    # (user_prompt + response_a + response_b)
    # -------------------------
    required_text_columns = [
        "user_prompt",
        "response_a",
        "response_b"
    ]

    if all(col in df.columns for col in required_text_columns):
        df = run_all_Text_Features(df)

    # -------------------------
    # Comparison features
    # (a_* vs b_* pairwise metrics, requires text features above)
    # -------------------------
    df = run_all_comparison_features(df)

    return df


__all__ = ["run_all_features"]
