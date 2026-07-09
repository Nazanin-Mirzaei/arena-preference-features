"""
This module extracts pairwise comparison features between response A and
response B for formatting richness — a weighted composite of bold usage,
headers, list items, and tables. Richer formatting is a documented human
preference signal independent of underlying content quality.

Currently supports:
- format_richness (bold + headers + list items + tables, weighted)
"""

from typing import Any
import math

EPSILON = 1e-6

# ---------------------------------------------------------
# Composite weights
# ---------------------------------------------------------
# Headers and tables are rarer and more visually structuring than bold
# spans or list items, so they carry more weight in the composite.
BOLD_WEIGHT = 1.0
HEADER_WEIGHT = 2.0
LIST_ITEM_WEIGHT = 1.0
TABLE_WEIGHT = 3.0


# ---------------------------------------------------------
# Helper: robust numeric coercion
# ---------------------------------------------------------
def _to_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------
# Per-side richness score
# ---------------------------------------------------------
def compute_format_richness(
    bold: Any,
    headers: Any,
    list_items: Any,
    tables: Any,
) -> float:
    """
    Compute a single weighted formatting-richness score for one response side.

    Args:
        bold: total bold-emphasis count
        headers: total header count (all levels combined)
        list_items: total list item count (ordered + unordered)
        tables: total detected table count

    Returns:
        float: weighted sum of formatting signals
    """

    return (
        _to_number(bold) * BOLD_WEIGHT
        + _to_number(headers) * HEADER_WEIGHT
        + _to_number(list_items) * LIST_ITEM_WEIGHT
        + _to_number(tables) * TABLE_WEIGHT
    )


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_format_richness_comparison(
    a_bold: Any,
    a_headers: Any,
    a_list_items: Any,
    a_tables: Any,
    b_bold: Any,
    b_headers: Any,
    b_list_items: Any,
    b_tables: Any,
) -> dict:
    """
    Compare formatting richness between response A and response B.

    Returns:
        {
            "a_format_richness": float,
            "b_format_richness": float,
            "format_richness_diff": float,      # a - b
            "format_richness_ratio": float,     # a / (b + epsilon)
            "format_richness_a_gt_b": bool,     # a > b
        }
    """

    a_richness = compute_format_richness(a_bold, a_headers, a_list_items, a_tables)
    b_richness = compute_format_richness(b_bold, b_headers, b_list_items, b_tables)

    return {
        "a_format_richness": a_richness,
        "b_format_richness": b_richness,
        "format_richness_diff": a_richness - b_richness,
        "format_richness_ratio": a_richness / (b_richness + EPSILON),
        "format_richness_a_gt_b": a_richness > b_richness,
    }


__all__ = ["compute_format_richness", "extract_format_richness_comparison"]
