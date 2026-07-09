"""
This module extracts pairwise comparison features capturing refusal
asymmetry between response A and response B. A response that refuses
while the other complies is a decisive loser signal; both sides
refusing is a strong indicator of a "both_bad" outcome.

Currently supports:
- has_refusal (per side + agreement signals)
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
def extract_refusal_comparison(a_has_refusal: Any, b_has_refusal: Any) -> dict:
    """
    Compare refusal behavior between response A and response B.

    Args:
        a_has_refusal: has_refusal computed for response_a
        b_has_refusal: has_refusal computed for response_b

    Returns:
        {
            "both_refused": bool,       # both sides refused
            "only_one_refused": bool,   # exactly one side refused
            "neither_refused": bool,    # neither side refused
        }

    Note:
        No refusal_diff or refusal_ratio is provided. diff (in {-1,0,1})
        is redundant with a_has_refusal/b_has_refusal (fully
        reconstructible from them) and splits an already-rare signal by
        direction, which is meaningless here since model_a/model_b is
        an arbitrary per-row slot, not a stable axis. ratio is
        degenerate for booleans (see is_code_block_ratio, removed for
        the same reason).
    """

    a = _to_bool(a_has_refusal)
    b = _to_bool(b_has_refusal)

    return {
        "both_refused": a and b,
        "only_one_refused": a != b,
        "neither_refused": not a and not b,
    }


__all__ = ["extract_refusal_comparison"]
