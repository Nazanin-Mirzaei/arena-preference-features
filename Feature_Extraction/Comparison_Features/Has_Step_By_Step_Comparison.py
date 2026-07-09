"""
This module extracts pairwise comparison features capturing "step-by-
step structure" asymmetry between response A and response B.
has_step_by_step (see Text_Features.Writing_Style_Features) detects
enumerations, ordinals, procedural transitions, and reasoning markers.
Like is_detailed and has_latex, its value is conditional on prompt
type: structured step-by-step reasoning helps on procedural/
problem-solving prompts (cat_problem_solving) but can read as
padding on simple factual prompts.

Currently supports:
- has_step_by_step (per side + agreement signals)
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
def extract_has_step_by_step_comparison(a_has_step_by_step: Any, b_has_step_by_step: Any) -> dict:
    """
    Compare has_step_by_step status between response A and response B.

    Args:
        a_has_step_by_step: has_step_by_step computed for response_a
        b_has_step_by_step: has_step_by_step computed for response_b

    Returns:
        {
            "both_step_by_step": bool,       # both sides use step-by-step structure
            "only_one_step_by_step": bool,   # exactly one side does
            "neither_step_by_step": bool,    # neither side does
        }

    Note:
        No has_step_by_step_diff or _ratio is provided, for the same
        reasons as the other boolean comparisons in this package: diff
        is redundant with a_has_step_by_step/b_has_step_by_step and
        splits the signal by an arbitrary A/B slot direction, and
        ratio is degenerate for booleans.
    """

    a = _to_bool(a_has_step_by_step)
    b = _to_bool(b_has_step_by_step)

    return {
        "both_step_by_step": a and b,
        "only_one_step_by_step": a != b,
        "neither_step_by_step": not a and not b,
    }


__all__ = ["extract_has_step_by_step_comparison"]
