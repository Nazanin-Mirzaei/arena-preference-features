"""
This module extracts pairwise comparison features capturing
"engagement prompt phrasing" asymmetry between response A and
response B. has_interaction_prompt (see
Text_Features.Interaction_Features) detects user-engagement phrases
("let me know", "feel free to", "would you like", etc.) - one
component of the broader interaction_score composite. A response that
invites further engagement while the other doesn't is a minor
discourse-structure signal.

Currently supports:
- has_interaction_prompt (per side + agreement signals)
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
def extract_has_interaction_prompt_comparison(
    a_has_interaction_prompt: Any, b_has_interaction_prompt: Any
) -> dict:
    """
    Compare has_interaction_prompt status between response A and response B.

    Args:
        a_has_interaction_prompt: has_interaction_prompt computed for response_a
        b_has_interaction_prompt: has_interaction_prompt computed for response_b

    Returns:
        {
            "both_interaction_prompt": bool,       # both sides have engagement phrasing
            "only_one_interaction_prompt": bool,   # exactly one side does
            "neither_interaction_prompt": bool,    # neither side does
        }
    """

    a = _to_bool(a_has_interaction_prompt)
    b = _to_bool(b_has_interaction_prompt)

    return {
        "both_interaction_prompt": a and b,
        "only_one_interaction_prompt": a != b,
        "neither_interaction_prompt": not a and not b,
    }


__all__ = ["extract_has_interaction_prompt_comparison"]
