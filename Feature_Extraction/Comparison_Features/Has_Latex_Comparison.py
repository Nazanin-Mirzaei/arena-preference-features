"""
This module extracts pairwise comparison features capturing LaTeX /
math-notation usage asymmetry between response A and response B.
Unlike failure signals (refusal, near-empty, truncation), LaTeX
presence is not inherently good or bad on its own - its value is
conditional on whether the prompt calls for math/technical notation
(see Metadata_Features.Category_Tag.cat_math). What is informative
regardless of context is asymmetry: one side using proper math
notation while the other doesn't, on the same prompt, is a signal a
classifier can learn to weigh differently depending on the prompt
category.

Currently supports:
- has_latex (per side + agreement signals)
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
def extract_has_latex_comparison(a_has_latex: Any, b_has_latex: Any) -> dict:
    """
    Compare LaTeX/math-notation usage between response A and response B.

    Args:
        a_has_latex: has_latex computed for response_a
        b_has_latex: has_latex computed for response_b

    Returns:
        {
            "both_has_latex": bool,       # both sides use LaTeX/math notation
            "only_one_has_latex": bool,   # exactly one side uses it
            "neither_has_latex": bool,    # neither side uses it
        }

    Note:
        No has_latex_diff or has_latex_ratio is provided, for the same
        reasons as Refusal_Comparison / Near_Empty_Comparison: diff is
        redundant with a_has_latex/b_has_latex and splits the signal by
        an arbitrary A/B slot direction, and ratio is degenerate for
        booleans.
    """

    a = _to_bool(a_has_latex)
    b = _to_bool(b_has_latex)

    return {
        "both_has_latex": a and b,
        "only_one_has_latex": a != b,
        "neither_has_latex": not a and not b,
    }


__all__ = ["extract_has_latex_comparison"]
