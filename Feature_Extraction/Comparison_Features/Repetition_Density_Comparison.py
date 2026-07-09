"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- repetition_density
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
def extract_repetition_density_comparison(a_repetition_density: Any, b_repetition_density: Any) -> dict:
    """
    Compare repetition_density between response A and response B.

    Args:
        a_repetition_density: repetition_density computed for response_a
        b_repetition_density: repetition_density computed for response_b

    Returns:
        {
            "repetition_density_diff": float,      # a - b
            "repetition_density_ratio": float,     # a / (b + epsilon)
            "repetition_density_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_repetition_density)
    b = _to_number(b_repetition_density)

    return {
        "repetition_density_diff": a - b,
        "repetition_density_ratio": a / (b + EPSILON),
        "repetition_density_a_gt_b": a > b,
    }


__all__ = ["extract_repetition_density_comparison"]
