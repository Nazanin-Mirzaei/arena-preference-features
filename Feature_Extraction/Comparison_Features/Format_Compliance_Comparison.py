"""
This module checks whether each response's format complies with what
the prompt explicitly requested (list, table, JSON, or code), and
compares compliance between response A and response B. A response
that ignores an explicit format request is a quality/loser signal
independent of content correctness.

Currently supports:
- format compliance across list, table, JSON, and code requests
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
# Per-side compliance score
# ---------------------------------------------------------
def compute_format_compliance(
    requests_list: Any,
    requests_table: Any,
    requests_json: Any,
    requests_code: Any,
    has_list: Any,
    has_table: Any,
    has_json: Any,
    is_code_block: Any,
) -> Any:
    """
    Check whether a single response side satisfies every format the
    prompt explicitly requested.

    Returns:
        bool: True if the response delivers every requested format,
            or True (vacuously) if the prompt requested no format.
    """

    requested = {
        "list": _to_bool(requests_list),
        "table": _to_bool(requests_table),
        "json": _to_bool(requests_json),
        "code": _to_bool(requests_code),
    }

    delivered = {
        "list": _to_bool(has_list),
        "table": _to_bool(has_table),
        "json": _to_bool(has_json),
        "code": _to_bool(is_code_block),
    }

    if not any(requested.values()):
        return True

    return all(delivered[fmt] for fmt, asked in requested.items() if asked)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_format_compliance_comparison(
    requests_list: Any,
    requests_table: Any,
    requests_json: Any,
    requests_code: Any,
    a_has_list: Any,
    a_has_table: Any,
    a_has_json: Any,
    a_is_code_block: Any,
    b_has_list: Any,
    b_has_table: Any,
    b_has_json: Any,
    b_is_code_block: Any,
) -> dict:
    """
    Compare format-request compliance between response A and response B.

    Returns:
        {
            "a_format_matches_request": bool,
            "b_format_matches_request": bool,
            "only_one_matched_format": bool,   # exactly one side complied
        }

    Note:
        No format_compliance_diff or _ratio is provided, for the same
        reasons as Refusal_Comparison / Near_Empty_Comparison: diff on
        a boolean pair is redundant with the two match columns and
        splits an already-narrow signal (prompts with explicit format
        requests are a minority) by an arbitrary A/B slot direction;
        ratio is degenerate for booleans.
    """

    a_match = compute_format_compliance(
        requests_list, requests_table, requests_json, requests_code,
        a_has_list, a_has_table, a_has_json, a_is_code_block,
    )
    b_match = compute_format_compliance(
        requests_list, requests_table, requests_json, requests_code,
        b_has_list, b_has_table, b_has_json, b_is_code_block,
    )

    return {
        "a_format_matches_request": a_match,
        "b_format_matches_request": b_match,
        "only_one_matched_format": a_match != b_match,
    }


__all__ = ["compute_format_compliance", "extract_format_compliance_comparison"]
