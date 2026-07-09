"""
This module extracts pairwise comparison features capturing
code-switching / mixed-language asymmetry between response A and
response B. is_multilingual (see Language_Detection.api) detects
whether a text mixes multiple languages within word chunks. One side
code-switching while the other doesn't is informative independent of
which language was requested - unintentional language mixing is
typically a quality defect, while intentional mixing (e.g. quoting a
foreign term) is comparatively rare.

Currently supports:
- is_multilingual (per side + agreement signals)
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
def extract_is_multilingual_comparison(a_is_multilingual: Any, b_is_multilingual: Any) -> dict:
    """
    Compare is_multilingual status between response A and response B.

    Args:
        a_is_multilingual: is_multilingual computed for response_a
        b_is_multilingual: is_multilingual computed for response_b

    Returns:
        {
            "both_multilingual": bool,       # both sides mix languages
            "only_one_multilingual": bool,   # exactly one side does
            "neither_multilingual": bool,    # neither side does
        }
    """

    a = _to_bool(a_is_multilingual)
    b = _to_bool(b_is_multilingual)

    return {
        "both_multilingual": a and b,
        "only_one_multilingual": a != b,
        "neither_multilingual": not a and not b,
    }


__all__ = ["extract_is_multilingual_comparison"]
