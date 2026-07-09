"""
This module extracts pairwise comparison features capturing truncation
asymmetry between response A and response B. A response that is
truncated while the other is complete is a decisive loser signal;
both sides being truncated is a signal for a "both_bad" outcome.

Currently supports:
- is_truncated (per side + agreement signals)
"""

from typing import Any


# ---------------------------------------------------------
# Helper: robust boolean coercion
# ---------------------------------------------------------
def _to_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN check without importing math
        return False
    return bool(value)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_truncation_comparison(a_is_truncated: Any, b_is_truncated: Any) -> dict:
    """
    Compare truncation status between response A and response B.

    Args:
        a_is_truncated: is_truncated computed for response_a
        b_is_truncated: is_truncated computed for response_b

    Returns:
        {
            "both_truncated": bool,       # both sides are truncated
            "only_one_truncated": bool,   # exactly one side is truncated
            "neither_truncated": bool,    # neither side is truncated
        }

    Note:
        No truncation_diff or truncation_ratio is provided, for the
        same reasons as Refusal_Comparison / Near_Empty_Comparison:
        diff is redundant with a_is_truncated/b_is_truncated and splits
        the signal by an arbitrary A/B slot direction, and ratio is
        degenerate for booleans.
    """

    a = _to_bool(a_is_truncated)
    b = _to_bool(b_is_truncated)

    return {
        "both_truncated": a and b,
        "only_one_truncated": a != b,
        "neither_truncated": not a and not b,
    }


__all__ = ["extract_truncation_comparison"]
