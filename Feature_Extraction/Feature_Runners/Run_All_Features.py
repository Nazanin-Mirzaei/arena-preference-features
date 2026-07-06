import pandas as pd

from Feature_Extraction.Feature_Runners.Run_Metadata_Features import run_all_metadata_features
from Feature_Extraction.Feature_Runners.Run_Response_Features import run_all_response_features


def run_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------------------------
    # Metadata features
    # -------------------------
    if "conv_metadata" in df.columns:
        df = run_all_metadata_features(df)

    # -------------------------
    # Response features (A / B)
    # -------------------------
    if "response_a" in df.columns and "response_b" in df.columns:
        df = run_all_response_features(df)

    return df


__all__ = ["run_all_features"]
