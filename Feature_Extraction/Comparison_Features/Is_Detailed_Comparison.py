"""
This module extracts pairwise comparison features capturing "detailed
response" asymmetry between response A and response B. is_detailed
(see Text_Features.Writing_Style_Features) is a heuristic composite of
sentence length and word count. One side being detailed while the
other isn't, on the same prompt, is informative for preference
modeling - though like format_richness, its value is conditional on
prompt category (H4): detail helps on complex/technical prompts and
can be a liability on prompts that explicitly request brevity (see
Length_Compliance_Comparison).

Currently supports:
- is_detailed (per side + agreement signals)
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
def extract_is_detailed_comparison(a_is_detailed: Any, b_is_detailed: Any) -> dict:
    """
    Compare is_detailed status between response A and response B.

    Args:
        a_is_detailed: is_detailed computed for response_a
        b_is_detailed: is_detailed computed for response_b

    Returns:
        {
            "both_detailed": bool,       # both sides are detailed
            "only_one_detailed": bool,   # exactly one side is detailed
            "neither_detailed": bool,    # neither side is detailed
        }

    Note:
        No is_detailed_diff or is_detailed_ratio is provided, for the
        same reasons as the other boolean comparisons in this package:
        diff is redundant with a_is_detailed/b_is_detailed and splits
        the signal by an arbitrary A/B slot direction, and ratio is
        degenerate for booleans.
    """

    a = _to_bool(a_is_detailed)
    b = _to_bool(b_is_detailed)

    return {
        "both_detailed": a and b,
        "only_one_detailed": a != b,
        "neither_detailed": not a and not b,
    }


__all__ = ["extract_is_detailed_comparison"]
