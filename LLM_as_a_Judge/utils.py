"""Utility functions for CSV and JSONL IO."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def clean_text(value: Any, max_chars: int = 20000) -> str:
    """Convert a value to a clean text string."""
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def read_csv_dataset(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV dataset as a list of dictionaries."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records to JSONL."""
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    """Write records to CSV."""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({column: row.get(column) for column in columns})

