"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- list_item_count
"""

from typing import Any
import math

EPSILON = 1e-6


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
# Main Feature Extractor
# ---------------------------------------------------------
def extract_list_item_count_comparison(a_list_item_count: Any, b_list_item_count: Any) -> dict:
    """
    Compare list_item_count between response A and response B.

    Args:
        a_list_item_count: list_item_count computed for response_a
        b_list_item_count: list_item_count computed for response_b

    Returns:
        {
            "list_item_count_diff": float,      # a - b
            "list_item_count_ratio": float,     # a / (b + epsilon)
            "list_item_count_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_list_item_count)
    b = _to_number(b_list_item_count)

    return {
        "list_item_count_diff": a - b,
        "list_item_count_ratio": a / (b + EPSILON),
        "list_item_count_a_gt_b": a > b,
    }


__all__ = ["extract_list_item_count_comparison"]
