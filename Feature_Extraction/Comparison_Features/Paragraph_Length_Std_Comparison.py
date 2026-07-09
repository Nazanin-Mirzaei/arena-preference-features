"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- paragraph_length_std
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
def extract_paragraph_length_std_comparison(
    a_paragraph_length_std: Any, b_paragraph_length_std: Any
) -> dict:
    """
    Compare paragraph_length_std between response A and response B.

    Args:
        a_paragraph_length_std: paragraph_length_std computed for response_a
        b_paragraph_length_std: paragraph_length_std computed for response_b

    Returns:
        {
            "paragraph_length_std_diff": float,      # a - b
            "paragraph_length_std_ratio": float,     # a / (b + epsilon)
            "paragraph_length_std_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_paragraph_length_std)
    b = _to_number(b_paragraph_length_std)

    return {
        "paragraph_length_std_diff": a - b,
        "paragraph_length_std_ratio": a / (b + EPSILON),
        "paragraph_length_std_a_gt_b": a > b,
    }


__all__ = ["extract_paragraph_length_std_comparison"]
