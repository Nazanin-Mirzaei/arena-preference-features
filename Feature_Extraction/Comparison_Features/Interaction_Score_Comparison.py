"""
This module extracts pairwise comparison features between response A and
response B for a single underlying metric. Comparative features capture
the relative difference between two responses, which is typically more
predictive of pairwise human preference than either response's absolute
value alone.

Currently supports:
- interaction_score
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
def extract_interaction_score_comparison(
    a_interaction_score: Any, b_interaction_score: Any
) -> dict:
    """
    Compare interaction_score between response A and response B.

    Args:
        a_interaction_score: interaction_score computed for response_a
        b_interaction_score: interaction_score computed for response_b

    Returns:
        {
            "interaction_score_diff": float,      # a - b
            "interaction_score_ratio": float,     # a / (b + epsilon)
            "interaction_score_a_gt_b": bool,     # a > b
        }
    """

    a = _to_number(a_interaction_score)
    b = _to_number(b_interaction_score)

    return {
        "interaction_score_diff": a - b,
        "interaction_score_ratio": a / (b + EPSILON),
        "interaction_score_a_gt_b": a > b,
    }


__all__ = ["extract_interaction_score_comparison"]
