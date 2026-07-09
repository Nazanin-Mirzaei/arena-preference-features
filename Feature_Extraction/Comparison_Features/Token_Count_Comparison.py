"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- token_count
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
def extract_token_count_comparison(a_token_count: Any, b_token_count: Any) -> dict:
    """
    Compare token_count between response A and response B.

    Args:
        a_token_count: token_count computed for response_a
        b_token_count: token_count computed for response_b

    Returns:
        {
            "token_count_diff": float,           # a - b
            "token_count_ratio": float,          # a / (b + epsilon)
            "token_count_a_gt_b": bool,          # a > b
            "token_count_gap_magnitude": float,  # |a - b|
        }
    """

    a = _to_number(a_token_count)
    b = _to_number(b_token_count)

    return {
        "token_count_diff": a - b,
        "token_count_ratio": a / (b + EPSILON),
        "token_count_a_gt_b": a > b,
        "token_count_gap_magnitude": abs(a - b),
    }


__all__ = ["extract_token_count_comparison"]
