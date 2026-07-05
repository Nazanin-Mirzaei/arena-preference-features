"""
Bold Text Feature Extraction.

Uses:
- bold_count_a: {"**": X, "__": Y}
- bold_count_b: same format

Provides:
- bold_total
- bold_style_preference
- emphasis_intensity
"""

from typing import Dict, Any

def extract_bold_features(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {"bold_total": 0, "bold_style_preference": None, "emphasis_intensity": 0.0}

    a = meta.get("bold_count_a", {})
    b = meta.get("bold_count_b", {})

    total_stars = a.get("**", 0) + b.get("**", 0)
    total_underscore = a.get("__", 0) + b.get("__", 0)
    bold_total = total_stars + total_underscore

    # Style preference: star-style or underscore-style
    if total_stars > total_underscore:
        preference = "**"
    elif total_underscore > total_stars:
        preference = "__"
    else:
        preference = "equal"

    return {
        "bold_total": bold_total,
        "bold_style_preference": preference,
        "emphasis_intensity": bold_total / 1000.0,  # normalized heuristic
    }

__all__ = ["extract_bold_features"]
