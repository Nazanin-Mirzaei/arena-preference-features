"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- code_delimiters
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
def extract_code_delimiters_comparison(a_code_delimiters: Any, b_code_delimiters: Any) -> dict:
    """
    Compare code_delimiters between response A and response B.

    Args:
        a_code_delimiters: code_delimiters computed for response_a
        b_code_delimiters: code_delimiters computed for response_b

    Returns:
        {
            "code_delimiters_diff": float,      # a - b
            "code_delimiters_ratio": float,     # a / (b + epsilon)
            "code_delimiters_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_code_delimiters)
    b = _to_number(b_code_delimiters)

    return {
        "code_delimiters_diff": a - b,
        "code_delimiters_ratio": a / (b + EPSILON),
        "code_delimiters_a_gt_b": a > b,
    }


__all__ = ["extract_code_delimiters_comparison"]
