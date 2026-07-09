"""
Category Tag Feature Extraction.

Flattens the dataset-provided `category_tag` column into individual
interpretable feature columns. `category_tag` contains human/model
labels describing the interaction type of each prompt, grouped into
versioned sub-objects:

- creative_writing_v0.1: {creative_writing: bool, score: str}
- criteria_v0.1: {complexity, creativity, domain_knowledge,
      problem_solving, real_world, specificity, technical_accuracy}
      (all bool)
- if_v0.1: {if: bool, score: int}   # instruction-following
- math_v0.1: {math: bool}

These are free, pre-labeled interaction-type signals that let a
downstream classifier learn conditional preferences (e.g. verbosity
helping on creative writing but hurting on math). Output feature names
drop the version suffix so they stay stable across version bumps.
"""

import ast
from typing import Any, Dict


# -------------------------
# Safe parser
# -------------------------
def parse_category_tag(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


# -------------------------
# Helpers
# -------------------------
def _find_group(meta: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Return the sub-dict for a versioned group by prefix (e.g. "criteria"
    matches "criteria_v0.1"), so feature extraction survives version
    bumps. Falls back to an exact match if present.
    """
    if prefix in meta and isinstance(meta[prefix], dict):
        return meta[prefix]
    for key, value in meta.items():
        if key.split("_v")[0] == prefix and isinstance(value, dict):
            return value
    return {}


def _to_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, (bool, int, float)) else False


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# -------------------------
# Feature keys (stable, version-suffix stripped)
# -------------------------
_CRITERIA_KEYS = [
    "complexity",
    "creativity",
    "domain_knowledge",
    "problem_solving",
    "real_world",
    "specificity",
    "technical_accuracy",
]


def _empty_features() -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for key in _CRITERIA_KEYS:
        features[f"cat_{key}"] = False
    features["cat_creative_writing"] = False
    features["cat_math"] = False
    features["cat_if"] = False
    features["cat_if_score"] = 0
    return features


# -------------------------
# Main Feature Function
# -------------------------
def extract_category_tag_features(value: Any) -> Dict[str, Any]:
    meta = parse_category_tag(value)
    if not meta:
        return _empty_features()

    features: Dict[str, Any] = {}

    criteria = _find_group(meta, "criteria")
    for key in _CRITERIA_KEYS:
        features[f"cat_{key}"] = _to_bool(criteria.get(key, False))

    creative = _find_group(meta, "creative_writing")
    features["cat_creative_writing"] = _to_bool(creative.get("creative_writing", False))

    math = _find_group(meta, "math")
    features["cat_math"] = _to_bool(math.get("math", False))

    instruction = _find_group(meta, "if")
    features["cat_if"] = _to_bool(instruction.get("if", False))
    features["cat_if_score"] = _to_int(instruction.get("score", 0), default=0)

    return features


__all__ = ["extract_category_tag_features", "parse_category_tag"]
