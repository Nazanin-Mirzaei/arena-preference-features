"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- emoji_count
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
def extract_emoji_count_comparison(a_emoji_count: Any, b_emoji_count: Any) -> dict:
    """
    Compare emoji_count between response A and response B.

    Args:
        a_emoji_count: emoji_count computed for response_a
        b_emoji_count: emoji_count computed for response_b

    Returns:
        {
            "emoji_count_diff": float,      # a - b
            "emoji_count_ratio": float,     # a / (b + epsilon)
            "emoji_count_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_emoji_count)
    b = _to_number(b_emoji_count)

    return {
        "emoji_count_diff": a - b,
        "emoji_count_ratio": a / (b + EPSILON),
        "emoji_count_a_gt_b": a > b,
    }


__all__ = ["extract_emoji_count_comparison"]
