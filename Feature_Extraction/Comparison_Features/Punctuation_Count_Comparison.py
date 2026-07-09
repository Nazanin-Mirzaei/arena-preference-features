"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- punctuation_count
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
def extract_punctuation_count_comparison(a_punctuation_count: Any, b_punctuation_count: Any) -> dict:
    """
    Compare punctuation_count between response A and response B.

    Args:
        a_punctuation_count: punctuation_count computed for response_a
        b_punctuation_count: punctuation_count computed for response_b

    Returns:
        {
            "punctuation_count_diff": float,      # a - b
            "punctuation_count_ratio": float,     # a / (b + epsilon)
            "punctuation_count_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_punctuation_count)
    b = _to_number(b_punctuation_count)

    return {
        "punctuation_count_diff": a - b,
        "punctuation_count_ratio": a / (b + EPSILON),
        "punctuation_count_a_gt_b": a > b,
    }


__all__ = ["extract_punctuation_count_comparison"]
