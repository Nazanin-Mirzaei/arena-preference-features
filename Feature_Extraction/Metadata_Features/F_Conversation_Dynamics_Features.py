"""
Conversation Dynamics Features.

Extracts:
- number of turns
- multi_turn flag
- turn_density (tokens per turn)
"""

from typing import Dict, Any

def extract_conversation_dynamics(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {"turns": 0, "is_multi_turn": False, "turn_density": 0.0}

    turns = meta.get("turns") or 0
    turns = int(turns) if isinstance(turns, (int, float)) else 0

    a = meta.get("sum_assistant_a_tokens") or 0
    b = meta.get("sum_assistant_b_tokens") or 0
    assistant_tokens = a + b

    denom = turns if turns > 0 else 1

    return {
        "turns": turns,
        "is_multi_turn": turns > 1,
        "turn_density": assistant_tokens / denom,
    }

__all__ = ["extract_conversation_dynamics"]
