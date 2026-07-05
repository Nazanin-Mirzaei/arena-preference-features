"""
Unified Dataset Feature Extractor
---------------------------------
Extracts ALL formatting-related features that already exist inside the
LM-Arena dataset's `conv_metadata` column.

Includes:
- Bold Count
- Ordered / Unordered Lists
- Header Counts (H1–H6)
- Assistant Token Counts
- Context Token Counts

This module is meant to mirror the dataset-provided features so they can
be compared against module-extracted features for validation.
"""

import ast
import pandas as pd


def parse_metadata(meta):
    """Safely parses the metadata dict stored as string."""
    try:
        return ast.literal_eval(meta)
    except Exception:
        return {}


def get_value(meta, path, default=0):
    """Traverse nested dict paths in conv_metadata."""
    cur = meta
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default


def extract_dataset_format_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts dataset-provided formatting features from conv_metadata.

    Output columns include:
        dataset_a_bold, dataset_b_bold
        dataset_a_list_ordered, dataset_a_list_unordered
        dataset_b_list_ordered, dataset_b_list_unordered
        dataset_a_header_h1 ... h6
        dataset_b_header_h1 ... h6
        dataset_a_tokens, dataset_b_tokens
        dataset_a_context_tokens, dataset_b_context_tokens
    """
    df = df.copy()

    meta_parsed = df["conv_metadata"].apply(parse_metadata)

    # ---- BOLD COUNT ----
    df["dataset_a_bold"] = meta_parsed.apply(lambda m: get_value(m, ["bold_count_a", "**"], 0))
    df["dataset_b_bold"] = meta_parsed.apply(lambda m: get_value(m, ["bold_count_b", "**"], 0))

    # ---- LIST COUNTS ----
    df["dataset_a_list_ordered"]   = meta_parsed.apply(lambda m: get_value(m, ["list_count_a", "ordered"], 0))
    df["dataset_a_list_unordered"] = meta_parsed.apply(lambda m: get_value(m, ["list_count_a", "unordered"], 0))

    df["dataset_b_list_ordered"]   = meta_parsed.apply(lambda m: get_value(m, ["list_count_b", "ordered"], 0))
    df["dataset_b_list_unordered"] = meta_parsed.apply(lambda m: get_value(m, ["list_count_b", "unordered"], 0))

    # ---- HEADER COUNTS ----
    for h in ["h1","h2","h3","h4","h5","h6"]:
        df[f"dataset_a_header_{h}"] = meta_parsed.apply(lambda m: get_value(m, ["header_count_a", h], 0))
        df[f"dataset_b_header_{h}"] = meta_parsed.apply(lambda m: get_value(m, ["header_count_b", h], 0))

    # ---- TOKEN COUNTS ----
    df["dataset_a_tokens"] = meta_parsed.apply(lambda m: m.get("sum_assistant_a_tokens", 0))
    df["dataset_b_tokens"] = meta_parsed.apply(lambda m: m.get("sum_assistant_b_tokens", 0))

    # ---- CONTEXT TOKEN COUNTS ----
    df["dataset_a_context_tokens"] = meta_parsed.apply(lambda m: m.get("context_a_tokens", 0))
    df["dataset_b_context_tokens"] = meta_parsed.apply(lambda m: m.get("context_b_tokens", 0))

    return df


__all__ = ["extract_dataset_format_features"]
