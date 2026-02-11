"""Shared utility functions for EKM core."""
import numpy as np
from typing import Union, List

def cosine_similarity(
    v1: Union[np.ndarray, List[float]], 
    v2: Union[np.ndarray, List[float]]
) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        # Fallback for empty or malformed inputs
        return 0.0

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    # Shift x by subtracting max to prevent overflow
    if x.size == 0:
        return x
    
    # Handle the case where x might be a list
    x_arr = np.array(x) if not isinstance(x, np.ndarray) else x
    
    try:
        exp_x = np.exp(x_arr - np.max(x_arr, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    except Exception:
        # Fallback for edge cases
        return x_arr

def estimate_tokens(text: str) -> int:
    """
    Very basic heuristic for token estimation when tiktoken is unavailable.
    Rule of thumb: 1 token ~= 4 characters for English.
    """
    if not text:
        return 0
    return len(text) // 4

def truncate_text_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to stay within a token limit using heuristic estimation.
    """
    if not text:
        return ""
    
    curr_tokens = estimate_tokens(text)
    if curr_tokens <= max_tokens:
        return text
    
    # Truncate based on character count (tokens * 4) as a safe approximation
    max_chars = max_tokens * 4
    return text[:max_chars] + "... [TRUNCATED]"
