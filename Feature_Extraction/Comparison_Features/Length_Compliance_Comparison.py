"""
This module checks whether each response's actual length complies with
an explicit brevity/detail request in the prompt ("briefly" vs.
"explain in detail"), and compares compliance between response A and
response B. A verbose answer to a "briefly" request, or a terse answer
to a "detailed" request, is a quality/loser signal independent of
content correctness.

Currently supports:
- length compliance via word_count thresholds against requests_brief /
  requests_detailed
"""

from typing import Any
import math

# A response at or below this word count satisfies a "briefly" request.
DEFAULT_BRIEF_WORD_COUNT_THRESHOLD = 60

# A response at or above this word count satisfies a "detailed" request.
# Mirrors the word_count half of Writing_Style_Features.is_detailed.
DEFAULT_DETAILED_WORD_COUNT_THRESHOLD = 50


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
# Helper: robust numeric coercion
# ---------------------------------------------------------
def _to_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------
# Per-side length compliance
# ---------------------------------------------------------
def compute_length_compliance(
    requests_brief: Any,
    requests_detailed: Any,
    word_count: Any,
    brief_threshold: int = DEFAULT_BRIEF_WORD_COUNT_THRESHOLD,
    detailed_threshold: int = DEFAULT_DETAILED_WORD_COUNT_THRESHOLD,
) -> bool:
    """
    Check whether a single response side's length satisfies the
    prompt's explicit brevity/detail request.

    Returns:
        bool: True if the response length satisfies every requested
            length constraint, or True (vacuously) if the prompt made
            no explicit length request. If the prompt contradictorily
            requests both brief and detailed, both thresholds must be
            satisfied (which is possible only in the narrow band
            between them, mirroring the contradiction in the prompt
            itself).
    """

    brief_requested = _to_bool(requests_brief)
    detailed_requested = _to_bool(requests_detailed)

    if not brief_requested and not detailed_requested:
        return True

    count = _to_number(word_count)

    compliant = True
    if brief_requested:
        compliant = compliant and count <= brief_threshold
    if detailed_requested:
        compliant = compliant and count >= detailed_threshold

    return compliant


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_length_compliance_comparison(
    requests_brief: Any,
    requests_detailed: Any,
    a_word_count: Any,
    b_word_count: Any,
) -> dict:
    """
    Compare length-request compliance between response A and response B.

    Returns:
        {
            "a_length_appropriate": bool,
            "b_length_appropriate": bool,
            "only_one_length_appropriate": bool,   # exactly one side complied
        }

    Note:
        No length_compliance_diff or _ratio is provided, for the same
        reasons as Format_Compliance_Comparison: diff on a boolean
        pair is redundant with the two match columns and splits an
        already-narrow signal (prompts with explicit length requests
        are a minority) by an arbitrary A/B slot direction; ratio is
        degenerate for booleans.
    """

    a_match = compute_length_compliance(requests_brief, requests_detailed, a_word_count)
    b_match = compute_length_compliance(requests_brief, requests_detailed, b_word_count)

    return {
        "a_length_appropriate": a_match,
        "b_length_appropriate": b_match,
        "only_one_length_appropriate": a_match != b_match,
    }


__all__ = ["compute_length_compliance", "extract_length_compliance_comparison"]
