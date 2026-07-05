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

    turns = meta.get("turns", 0)
    assistant_tokens = meta.get("sum_assistant_a_tokens", 0) + meta.get("sum_assistant_b_tokens", 0)

    return {
        "turns": turns,
        "is_multi_turn": turns > 1,
        "turn_density": assistant_tokens / max(turns, 1),
    }

__all__ = ["extract_conversation_dynamics"]
