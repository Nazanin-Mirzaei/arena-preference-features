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
import ast


# -------------------------
# Safe parser (optional now)
# -------------------------
def parse_metadata(meta):
    if isinstance(meta, dict):
        return meta
    try:
        return ast.literal_eval(meta)
    except Exception:
        return {}


# -------------------------
# Helper
# -------------------------
def get_value(meta, path, default=0):
    cur = meta
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default


# -------------------------
# Main Feature Function (REFACTORED)
# -------------------------
def extract_dataset_format_features(meta: dict) -> dict:
    """
    Extract dataset formatting features from a single conv_metadata dict.
    """

    if not isinstance(meta, dict):
        meta = {}

    features = {}

    # ---- BOLD COUNT ----
    features["dataset_a_bold"] = get_value(meta, ["bold_count_a", "**"], 0)
    features["dataset_b_bold"] = get_value(meta, ["bold_count_b", "**"], 0)

    # ---- LIST COUNTS ----
    features["dataset_a_list_ordered"] = get_value(meta, ["list_count_a", "ordered"], 0)
    features["dataset_a_list_unordered"] = get_value(meta, ["list_count_a", "unordered"], 0)

    features["dataset_b_list_ordered"] = get_value(meta, ["list_count_b", "ordered"], 0)
    features["dataset_b_list_unordered"] = get_value(meta, ["list_count_b", "unordered"], 0)

    # ---- HEADER COUNTS ----
    for h in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        features[f"dataset_a_header_{h}"] = get_value(meta, ["header_count_a", h], 0)
        features[f"dataset_b_header_{h}"] = get_value(meta, ["header_count_b", h], 0)

    # ---- TOKEN COUNTS ----
    features["dataset_a_tokens"] = meta.get("sum_assistant_a_tokens", 0)
    features["dataset_b_tokens"] = meta.get("sum_assistant_b_tokens", 0)

    # ---- CONTEXT TOKEN COUNTS ----
    features["dataset_a_context_tokens"] = meta.get("context_a_tokens", 0)
    features["dataset_b_context_tokens"] = meta.get("context_b_tokens", 0)

    return features


__all__ = ["extract_dataset_format_features"]
