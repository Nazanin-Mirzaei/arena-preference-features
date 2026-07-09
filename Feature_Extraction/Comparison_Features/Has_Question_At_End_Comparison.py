"""
This module extracts pairwise comparison features capturing
"ends with a question" asymmetry between response A and response B.
has_question_at_end (see Text_Features.Interaction_Features) detects
whether a response ends on a question mark - one component of the
broader interaction_score composite. A response that invites
continued engagement while the other doesn't is a minor
discourse-structure signal.

Currently supports:
- has_question_at_end (per side + agreement signals)
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
def extract_has_question_at_end_comparison(
    a_has_question_at_end: Any, b_has_question_at_end: Any
) -> dict:
    """
    Compare has_question_at_end status between response A and response B.

    Args:
        a_has_question_at_end: has_question_at_end computed for response_a
        b_has_question_at_end: has_question_at_end computed for response_b

    Returns:
        {
            "both_question_at_end": bool,       # both sides end with a question
            "only_one_question_at_end": bool,   # exactly one side does
            "neither_question_at_end": bool,    # neither side does
        }
    """

    a = _to_bool(a_has_question_at_end)
    b = _to_bool(b_has_question_at_end)

    return {
        "both_question_at_end": a and b,
        "only_one_question_at_end": a != b,
        "neither_question_at_end": not a and not b,
    }


__all__ = ["extract_has_question_at_end_comparison"]
