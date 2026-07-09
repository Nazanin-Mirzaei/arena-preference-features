"""
JSON detection utility for structured response analysis.

Detects whether text contains a JSON object/array, either fenced in a
```json code block or as raw brace/bracket-delimited structured text
that parses successfully. Used to check format compliance when a
prompt explicitly requests JSON output.
"""

from typing import Any
import json
import re
import math


# -------------------------------
# Helper: robust NaN detection
# -------------------------------
def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


# ------------------------------------------
# Regex patterns
# ------------------------------------------
_JSON_FENCE_PATTERN = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Candidate raw JSON blocks: outermost {...} or [...] spans
_RAW_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_RAW_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def _is_valid_json(candidate: str) -> bool:
    candidate = candidate.strip()
    if not candidate:
        return False
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return False
    # A bare string/number/bool parses as valid JSON but isn't
    # structured content - require an object or array.
    return isinstance(parsed, (dict, list))


# ------------------------------------------
# Feature: detect if text contains JSON
# ------------------------------------------
def has_json(text: Any) -> bool:
    """
    Returns True if text contains a parseable JSON object or array,
    either inside a ```json fenced code block or as raw text.
    """
    if _is_nan(text):
        return False

    s = str(text)
    if not s.strip():
        return False

    for match in _JSON_FENCE_PATTERN.findall(s):
        if _is_valid_json(match):
            return True

    obj_match = _RAW_OBJECT_PATTERN.search(s)
    if obj_match and _is_valid_json(obj_match.group(0)):
        return True

    arr_match = _RAW_ARRAY_PATTERN.search(s)
    if arr_match and _is_valid_json(arr_match.group(0)):
        return True

    return False


__all__ = ["has_json"]
