"""
This module detects refusal / apology / inability language in model
responses using lightweight regex heuristics. Refusals are a strong
quality signal independent of content correctness - a response that
declines the request is a common loser in pairwise human preference,
and both sides refusing is a signal for "both_bad" outcomes.
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
# Refusal Patterns
# ---------------------------------------------------------
# "I cannot" / "I can't" alone are too common in non-refusal contexts
# (emphasis, code explanations, hedging: "I can't stress enough...",
# "the function can't return null"). They only count as a refusal
# signal when followed by a refusal object (help, provide, comply,
# do that, etc.) within a short distance.
_REFUSAL_OBJECT = (
    r"(?:help|assist|provide|answer|comply|complete|do that|do this|"
    r"fulfill|generate|write|create|continue|proceed|share|disclose|"
    r"engage in|participate in|give you)"
)

_REFUSAL_RE = re.compile(
    r"\b(?:"
    # Inability phrasing gated on a refusal object
    r"(?:i cannot|i can't|i can not|i'm unable to|i am unable to|"
    r"i'm not able to|i am not able to|i don't have the ability to|"
    r"i do not have the ability to|i won't be able to|"
    r"i will not be able to)\s+(?:\w+\s+){0,3}" + _REFUSAL_OBJECT + r"|"
    # Explicit decline / refusal, no gating needed
    r"i must decline|i have to decline|i'm going to have to decline|"
    r"i won't|i will not|i refuse to|"
    # Model self-identification disclaimers
    r"as an ai(?:\s+language\s+model|\s+assistant)?|"
    r"as a language model|as a large language model|"
    r"i'm just an ai|i am just an ai|i'm only a language model|"
    r"i am only a language model|"
    # Apology-prefaced refusal
    r"i'm sorry,? but|i am sorry,? but|i apologize,? but|"
    r"unfortunately,? i(?:'m| am)? (?:unable|not able|can't|cannot)|"
    # Access / permission denial
    r"i don't have access to|i do not have access to|"
    r"i'm not permitted to|i am not permitted to|"
    r"i'm not allowed to|i am not allowed to|"
    # Policy / guideline / appropriateness framing
    r"against my (?:guidelines|programming|policies)|"
    r"goes against (?:my|the) (?:guidelines|policy|policies)|"
    r"violates? (?:my|the) (?:guidelines|policy|policies)|"
    r"it would be inappropriate for me to|"
    r"i don't feel comfortable|i do not feel comfortable|"
    r"i'm not comfortable|i am not comfortable"
    r")\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------
# Main Feature: Refusal Detector
# ---------------------------------------------------------
def has_refusal(text: Any) -> bool:
    """
    Detect whether a response contains refusal, apology, or
    inability-to-comply language.

    Recognizes phrases such as:
        - "I cannot" / "I can't"
        - "As an AI"
        - "I'm unable to" / "I am unable to"
        - "I apologize, but"
    """

    clean = _clean_text(text)

    if not clean:
        return False

    return bool(_REFUSAL_RE.search(clean))


__all__ = ["has_refusal"]
