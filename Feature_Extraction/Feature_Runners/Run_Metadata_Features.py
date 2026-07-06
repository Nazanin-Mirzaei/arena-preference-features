import pandas as pd
import ast

from Feature_Extraction.Metadata_Features.Bold import extract_bold_features
from Feature_Extraction.Metadata_Features.Conversation import extract_conversation_dynamics
from Feature_Extraction.Metadata_Features.Headers import extract_header_features
from Feature_Extraction.Metadata_Features.Lists import extract_list_features
from Feature_Extraction.Metadata_Features.Tokens import extract_token_features
from Feature_Extraction.Metadata_Features.Dataset_Baseline import extract_dataset_format_features


# -------------------------------------------------
# Parse once (CRITICAL for speed)
# -------------------------------------------------
def parse_metadata(meta):
    if isinstance(meta, dict):
        return meta
    try:
        return ast.literal_eval(meta)
    except Exception:
        return {}


# -------------------------------------------------
# Feature registry (clean + extensible design)
# -------------------------------------------------
METADATA_FEATURES = [
    extract_bold_features,
    extract_conversation_dynamics,
    extract_header_features,
    extract_list_features,
    extract_token_features,
    extract_dataset_format_features,
]


# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------
def run_all_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse once
    parsed = df["conv_metadata"].apply(parse_metadata)

    feature_frames = []

    # Apply all features
    for func in METADATA_FEATURES:
        feat_df = parsed.apply(func).apply(pd.Series)
        feature_frames.append(feat_df)

    # Merge all feature outputs
    df = pd.concat([df] + feature_frames, axis=1)

    return df


__all__ = ["run_all_metadata_features"]
