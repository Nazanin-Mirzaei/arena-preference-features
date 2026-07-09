"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- table_count
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
def extract_table_count_comparison(a_table_count: Any, b_table_count: Any) -> dict:
    """
    Compare table_count between response A and response B.

    Args:
        a_table_count: table_count computed for response_a
        b_table_count: table_count computed for response_b

    Returns:
        {
            "table_count_diff": float,      # a - b
            "table_count_ratio": float,     # a / (b + epsilon)
            "table_count_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_table_count)
    b = _to_number(b_table_count)

    return {
        "table_count_diff": a - b,
        "table_count_ratio": a / (b + EPSILON),
        "table_count_a_gt_b": a > b,
    }


__all__ = ["extract_table_count_comparison"]
