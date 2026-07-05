"""
Token Feature Extraction from LM-Arena conv_metadata.

This module extracts token-based features from metadata fields such as:
- sum_assistant_a_tokens
- sum_assistant_b_tokens
- context_a_tokens
- context_b_tokens
- sum_user_tokens

The goal is to provide a consistent interface for token-level analytics
that can be used across all teams.

Usage:
    from token_features_from_metadata import extract_token_features
    feats = extract_token_features(conv_metadata_dict)
"""

from typing import Dict, Any

def extract_token_features(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract token-related features from 'conv_metadata'.

    Parameters
    ----------
    meta : dict
        conv_metadata field extracted from the dataset.

    Returns
    -------
    dict
        Dictionary with token statistics.
    """
    if not isinstance(meta, dict):
        return {
            "assistant_tokens_total": 0,
            "user_tokens": 0,
            "context_tokens_total": 0,
            "assistant_token_ratio": 0.0,
        }

    a_tok = meta.get("sum_assistant_a_tokens") or 0
    b_tok = meta.get("sum_assistant_b_tokens") or 0
    user_tok = meta.get("sum_user_tokens") or 0
    ctx_a = meta.get("context_a_tokens") or 0
    ctx_b = meta.get("context_b_tokens") or 0

    assistant_total = a_tok + b_tok
    context_total = ctx_a + ctx_b

    return {
        "assistant_tokens_total": assistant_total,
        "user_tokens": user_tok,
        "context_tokens_total": context_total,
        "assistant_token_ratio": assistant_total / max(user_tok, 1),
    }

__all__ = ["extract_token_features"]
