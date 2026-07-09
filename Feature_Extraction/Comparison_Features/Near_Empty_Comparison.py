"""
This module extracts pairwise comparison features capturing near-empty
asymmetry between response A and response B. A response that is
near-empty while the other is substantive is a decisive loser signal;
both sides being near-empty is a strong indicator of a "both_bad"
outcome.

Currently supports:
- is_near_empty (per side + agreement signals)
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
def extract_near_empty_comparison(a_is_near_empty: Any, b_is_near_empty: Any) -> dict:
    """
    Compare near-empty status between response A and response B.

    Args:
        a_is_near_empty: is_near_empty computed for response_a
        b_is_near_empty: is_near_empty computed for response_b

    Returns:
        {
            "both_near_empty": bool,       # both sides are near-empty
            "only_one_near_empty": bool,   # exactly one side is near-empty
            "neither_near_empty": bool,    # neither side is near-empty
        }

    Note:
        No near_empty_diff or near_empty_ratio is provided, for the
        same reasons as Refusal_Comparison: diff is redundant with
        a_is_near_empty/b_is_near_empty and splits the signal by an
        arbitrary A/B slot direction, and ratio is degenerate for
        booleans.
    """

    a = _to_bool(a_is_near_empty)
    b = _to_bool(b_is_near_empty)

    return {
        "both_near_empty": a and b,
        "only_one_near_empty": a != b,
        "neither_near_empty": not a and not b,
    }


__all__ = ["extract_near_empty_comparison"]
