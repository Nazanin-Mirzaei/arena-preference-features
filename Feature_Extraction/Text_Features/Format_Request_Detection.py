"""
This module infers which output format(s) a user prompt explicitly
requests - list, table, JSON, or code - using lightweight regex
heuristics over the prompt text. Used to check whether a response's
actual format (see Is_Code_Block, Table_Detection, List_Detection,
Json_Detection) complies with what was asked for.
"""

import math
import re
from typing import Any


# ---------------------------------------------------------
# Text Cleaning Utility
# ---------------------------------------------------------
def _clean_text(text: Any) -> str:
    """
    Normalize input text into a clean string.
    """

    if text is None:
        return ""

    if isinstance(text, float) and math.isnan(text):
        return ""

    value = str(text).strip()

    if value.lower() == "nan":
        return ""

    return value


# ---------------------------------------------------------
# Request Patterns
# ---------------------------------------------------------
_LIST_REQUEST_RE = re.compile(
    r"\b(?:as a list|in list form|bullet(?:ed)? (?:list|points?)|"
    r"numbered list|list (?:of|out|the|\d)|give me a list|"
    r"^list\b|"
    r"enumerate|itemi[sz]e)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)

_TABLE_REQUEST_RE = re.compile(
    r"\b(?:as a table|in table form|in a table|tabulate|"
    r"table (?:of|comparing|with)|markdown table|comparison table|"
    r"(?:into|as|to) a table)\b",
    flags=re.IGNORECASE,
)

_JSON_REQUEST_RE = re.compile(
    r"\b(?:as json|in json|json format|json object|json array|"
    r"return json|output json|valid json)\b",
    flags=re.IGNORECASE,
)

# Programming language names, used both to gate "in <language>" (avoid
# matching plain language references like "the difference in Python")
# and to recognize casual phrasing like "code this up ... in C++".
_PROGRAMMING_LANGUAGES = (
    r"python|javascript|java|c\+\+|c#|typescript|go|rust|sql|bash"
)

_CODE_REQUEST_RE = re.compile(
    r"\b(?:write (?:a |the )?(?:\w+\s+){0,2}(?:function|script|program|code|class|method|algorithm)|"
    r"give me (?:the |a )?(?:\w+\s+){0,2}code|show me (?:the |a )?(?:\w+\s+){0,2}code|"
    r"code (?:this|it|that) up|"
    r"(?:write|code|implement|debug|fix)(?:\s+\w+){0,4}\s+in (?:" + _PROGRAMMING_LANGUAGES + r")\b|"
    r"code (?:for|to|that)|implement (?:a |the )?(?:\w+\s+){0,2}(?:function|class|method|algorithm)|"
    r"write me a (?:\w+\s+){0,2}(?:function|script|program))\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------
# Main Feature Extractor
# ---------------------------------------------------------
def extract_format_request_features(text: Any) -> dict:
    """
    Detect which structured output formats a prompt explicitly requests.

    Returns:
        {
            "requests_list": bool,
            "requests_table": bool,
            "requests_json": bool,
            "requests_code": bool,
        }
    """

    clean = _clean_text(text)

    if not clean:
        return {
            "requests_list": False,
            "requests_table": False,
            "requests_json": False,
            "requests_code": False,
        }

    return {
        "requests_list": bool(_LIST_REQUEST_RE.search(clean)),
        "requests_table": bool(_TABLE_REQUEST_RE.search(clean)),
        "requests_json": bool(_JSON_REQUEST_RE.search(clean)),
        "requests_code": bool(_CODE_REQUEST_RE.search(clean)),
    }


__all__ = ["extract_format_request_features"]
