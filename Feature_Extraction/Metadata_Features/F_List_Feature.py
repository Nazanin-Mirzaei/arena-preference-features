"""
List Feature Extraction (Ordered & Unordered Lists).

Uses metadata fields:
- list_count_a
- list_count_b

Provides:
- total_list_items
- ordered_ratio
- unordered_ratio
"""

from typing import Dict, Any

def extract_list_features(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {"total_list_items": 0, "ordered_ratio": 0.0, "unordered_ratio": 0.0}

    a = meta.get("list_count_a", {})
    b = meta.get("list_count_b", {})

    total_ordered = a.get("ordered", 0) + b.get("ordered", 0)
    total_unordered = a.get("unordered", 0) + b.get("unordered", 0)
    total = total_ordered + total_unordered

    return {
        "total_list_items": total,
        "ordered_ratio": total_ordered / max(total, 1),
        "unordered_ratio": total_unordered / max(total, 1),
    }

__all__ = ["extract_list_features"]
