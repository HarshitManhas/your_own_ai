"""
distance_metrics.py — Hand-rolled distance functions.

Implements Cosine, Euclidean, and Manhattan distance.
All functions operate on plain Python lists of floats — no NumPy.

Equivalent to DistanceMetrics.java in the original Java project.
"""

import math
from typing import Callable, List

# Type alias for a distance function
DistFn = Callable[[List[float], List[float]], float]


def euclidean(a: List[float], b: List[float]) -> float:
    """
    Euclidean (L2) distance.
    Formula: sqrt(sum((a_i - b_i)^2))
    Best for: geometric distances where magnitude matters.
    """
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def manhattan(a: List[float], b: List[float]) -> float:
    """
    Manhattan (L1) distance.
    Formula: sum(|a_i - b_i|)
    Best for: grid-like spaces, robust to outliers.
    """
    return sum(abs(x - y) for x, y in zip(a, b))


def cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine distance = 1 - cosine_similarity.
    Formula: 1 - (A·B) / (||A|| × ||B||)
    Best for: text/NLP — cares about direction, not magnitude.
    Returns 0 for identical vectors, ~2 for opposite vectors.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 1.0
    return 1.0 - dot / denom


def get_dist_fn(metric: str) -> DistFn:
    """
    Return the distance function for the given metric name.
    Supported: 'cosine', 'euclidean', 'manhattan'.
    Defaults to cosine for unknown names.
    """
    metric = metric.lower()
    if metric == "euclidean":
        return euclidean
    if metric == "manhattan":
        return manhattan
    return cosine  # default
