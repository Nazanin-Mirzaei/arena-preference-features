"""Parsing helpers for LLM judge outputs."""

from __future__ import annotations

import json
from typing import Any


def parse_json_output(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a model output into JSON.

    Returns:
        A tuple of (parsed_json, parse_error).
    """
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1]), None
            except json.JSONDecodeError as error:
                return None, str(error)
        return None, "No JSON object found in model output."

