"""
This module detects truncated responses using lightweight structural
heuristics. Truncation - a response cut off before it finished
generating - is a strong quality/loser signal that is distinct from
near-emptiness: a long, otherwise-substantive answer can still be
truncated (unclosed code fence, sentence stopped mid-word).

Detection is deliberately conservative to avoid flagging legitimately
complete responses that happen not to end in a period (code blocks,
tables, lists, headers, questions).
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
# Patterns
# ---------------------------------------------------------
# Count fenced code-block markers (``` or ~~~). An odd count means a
# code block was opened but never closed -> truncated.
_CODE_FENCE_RE = re.compile(r"(?:```|~~~)")

# Characters/tokens that indicate a legitimately complete ending.
# Sentence terminators, closing brackets/quotes, and colon (intro to
# something that may itself be a complete structure).
_COMPLETE_ENDING_CHARS = set(".!?…。！？:;)]}\"'`*_>|")

# A final line that is a structural element (list item, table row,
# heading, blockquote, horizontal rule) is treated as a complete
# ending even without terminal punctuation.
_STRUCTURAL_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-*+]\s+\S|"          # unordered list item
    r"\d+[.)]\s+\S|"        # ordered list item
    r"#{1,6}\s+\S|"         # heading
    r">\s+\S|"              # blockquote
    r"\|.*\||"              # table row
    r"[-=*]{3,}\s*"         # horizontal rule
    r")",
)


# ---------------------------------------------------------
# Main Feature: Truncation Detector
# ---------------------------------------------------------
def is_truncated(text: Any) -> bool:
    """
    Detect whether a response appears to be cut off before completion.

    Flags a response as truncated if:
      1. It contains an unclosed fenced code block (odd number of
         ``` / ~~~ markers), OR
      2. Its final line ends mid-sentence: not on sentence-ending
         punctuation / a closing bracket, not on a recognized
         structural line (list, table, heading, blockquote, rule),
         and the last "word" looks like a plausibly-cut token.

    Empty / whitespace-only input is not considered truncated here
    (that is near-emptiness, a separate signal).
    """

    clean = _clean_text(text)

    if not clean:
        return False

    # -------------------------
    # 1. Unclosed code fence
    # -------------------------
    if len(_CODE_FENCE_RE.findall(clean)) % 2 == 1:
        return True

    # -------------------------
    # 2. Ends mid-sentence
    # -------------------------
    # Work from the last non-empty line.
    lines = [ln for ln in clean.splitlines() if ln.strip()]
    if not lines:
        return False

    last_line = lines[-1].rstrip()

    # Structural final lines are complete endings.
    if _STRUCTURAL_LINE_RE.match(last_line):
        return False

    last_char = last_line[-1]
    if last_char in _COMPLETE_ENDING_CHARS:
        return False

    # At this point the text ends on a non-terminal character. Treat it
    # as truncated only if the final token is an ordinary alphabetic
    # word (mid-sentence / mid-word cutoff), not a bare number, symbol,
    # or abbreviation-like token.
    last_token = last_line.split()[-1] if last_line.split() else ""
    if re.fullmatch(r"[A-Za-z]{2,}", last_token):
        return True

    return False


__all__ = ["is_truncated"]
