"""
brute_force.py — Brute Force K-NN search.

Complexity: O(N × d) — linear in both vector count and dimensions.
Always returns 100% correct (exact) results.
Used as the baseline for benchmarking against KD-Tree and HNSW.

Equivalent to BruteForce.java in the original Java project.
"""

from typing import List, Tuple

from .distance_metrics import DistFn
from .models import VectorItem


class BruteForce:
    """
    Exact K-NN search by exhaustive scan.

    insert() / remove() are O(1) amortized.
    knn()    is O(N × d).
    """

    def __init__(self) -> None:
        self._items: List[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        """Add a vector to the store."""
        self._items.append(item)

    def remove(self, item_id: int) -> bool:
        """Remove a vector by ID. Returns True if found."""
        before = len(self._items)
        self._items = [v for v in self._items if v.id != item_id]
        return len(self._items) < before

    def knn(
        self,
        query: List[float],
        k: int,
        dist_fn: DistFn,
    ) -> List[Tuple[float, int]]:
        """
        Return the k nearest neighbors as a list of (distance, id) tuples,
        sorted by distance ascending.
        """
        if not self._items:
            return []

        scored = [(dist_fn(query, v.emb), v.id) for v in self._items]
        scored.sort(key=lambda x: x[0])
        return scored[:k]

    def size(self) -> int:
        return len(self._items)
