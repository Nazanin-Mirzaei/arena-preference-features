"""
This module extracts pairwise comparison features capturing
"conclusion phrasing" asymmetry between response A and response B.
has_conclusion (see Text_Features.Interaction_Features) detects
conclusive phrases ("in conclusion", "to summarize", "therefore",
etc.) - one component of the broader interaction_score composite.
A response that wraps up with a clear conclusion while the other
doesn't is a minor discourse-structure signal.

Currently supports:
- has_conclusion (per side + agreement signals)
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
def extract_has_conclusion_comparison(a_has_conclusion: Any, b_has_conclusion: Any) -> dict:
    """
    Compare has_conclusion status between response A and response B.

    Args:
        a_has_conclusion: has_conclusion computed for response_a
        b_has_conclusion: has_conclusion computed for response_b

    Returns:
        {
            "both_conclusion": bool,       # both sides have conclusion phrasing
            "only_one_conclusion": bool,   # exactly one side does
            "neither_conclusion": bool,    # neither side does
        }

    Note:
        No has_conclusion_diff or _ratio is provided, for the same
        reasons as the other boolean comparisons in this package: diff
        is redundant with a_has_conclusion/b_has_conclusion and splits
        the signal by an arbitrary A/B slot direction, and ratio is
        degenerate for booleans.
    """

    a = _to_bool(a_has_conclusion)
    b = _to_bool(b_has_conclusion)

    return {
        "both_conclusion": a and b,
        "only_one_conclusion": a != b,
        "neither_conclusion": not a and not b,
    }


__all__ = ["extract_has_conclusion_comparison"]
