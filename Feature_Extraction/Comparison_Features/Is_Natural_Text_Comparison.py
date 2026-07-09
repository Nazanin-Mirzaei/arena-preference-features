"""
This module extracts pairwise comparison features capturing "natural
prose vs. code/markup" asymmetry between response A and response B.
is_natural_text (see Text_Features.Is_Natural_Text) classifies a
response as natural language versus code or structured markup. Its
value is conditional on prompt type: the more code-like response may
be preferred on a coding prompt (cat_math / technical prompts), while
the more natural-language response may be preferred on a writing
prompt (cat_creative_writing) - so asymmetry between A and B is the
informative signal, not a fixed preference for either side.

Currently supports:
- is_natural_text (per side + agreement signals)
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
def extract_is_natural_text_comparison(a_is_natural_text: Any, b_is_natural_text: Any) -> dict:
    """
    Compare is_natural_text status between response A and response B.

    Args:
        a_is_natural_text: is_natural_text computed for response_a
        b_is_natural_text: is_natural_text computed for response_b

    Returns:
        {
            "both_natural_text": bool,       # both sides are natural prose
            "only_one_natural_text": bool,   # exactly one side is
            "neither_natural_text": bool,    # neither side is (both code/markup-like)
        }

    Note:
        No is_natural_text_diff or _ratio is provided, for the same
        reasons as the other boolean comparisons in this package: diff
        is redundant with a_is_natural_text/b_is_natural_text and
        splits the signal by an arbitrary A/B slot direction, and
        ratio is degenerate for booleans.
    """

    a = _to_bool(a_is_natural_text)
    b = _to_bool(b_is_natural_text)

    return {
        "both_natural_text": a and b,
        "only_one_natural_text": a != b,
        "neither_natural_text": not a and not b,
    }


__all__ = ["extract_is_natural_text_comparison"]
