"""
Header Feature Extraction (Markdown Headings).

This module extracts:
- Total number of headings (h1–h6)
- Heading depth distribution
- Structural richness

Requires metadata fields:
    header_count_a
    header_count_b
"""

from typing import Dict, Any

def extract_header_features(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {"total_headers": 0, "header_depth_score": 0}

    h_a = meta.get("header_count_a", {})
    h_b = meta.get("header_count_b", {})

    total_headers = sum(h_a.values()) + sum(h_b.values())

    # Weighted depth score: deeper headings add more “structure”
    depth_score = (
        sum(level * h_a.get(f"h{level}", 0) for level in range(1, 7)) +
        sum(level * h_b.get(f"h{level}", 0) for level in range(1, 7))
    )

    return {
        "total_headers": total_headers,
        "header_depth_score": depth_score,
    }

__all__ = ["extract_header_features"]
