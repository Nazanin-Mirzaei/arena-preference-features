"""
This module extracts pairwise comparison features capturing
"next-steps phrasing" asymmetry between response A and response B.
has_next_steps (see Text_Features.Interaction_Features) detects
forward-looking phrases ("next steps", "going forward", "you can now",
etc.) - one component of the broader interaction_score composite.
A response that offers forward-looking guidance while the other
doesn't is a minor discourse-structure signal.

Currently supports:
- has_next_steps (per side + agreement signals)
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
def extract_has_next_steps_comparison(a_has_next_steps: Any, b_has_next_steps: Any) -> dict:
    """
    Compare has_next_steps status between response A and response B.

    Args:
        a_has_next_steps: has_next_steps computed for response_a
        b_has_next_steps: has_next_steps computed for response_b

    Returns:
        {
            "both_next_steps": bool,       # both sides have next-steps phrasing
            "only_one_next_steps": bool,   # exactly one side does
            "neither_next_steps": bool,    # neither side does
        }
    """

    a = _to_bool(a_has_next_steps)
    b = _to_bool(b_has_next_steps)

    return {
        "both_next_steps": a and b,
        "only_one_next_steps": a != b,
        "neither_next_steps": not a and not b,
    }


__all__ = ["extract_has_next_steps_comparison"]
