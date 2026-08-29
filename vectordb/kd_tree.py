"""
kd_tree.py — KD-Tree for K-NN search.

Complexity: O(log N) average for low dimensions.
Binary space partitioning that cycles through dimensions.
Degrades to ~O(N) at high dimensions (curse of dimensionality) — good for
16D demo vectors, bad for 768D document embeddings (use HNSW for those).

Equivalent to KDTree.java in the original Java project.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .distance_metrics import DistFn
from .models import VectorItem


@dataclass
class _KDNode:
    item: VectorItem
    left: Optional["_KDNode"] = field(default=None, repr=False)
    right: Optional["_KDNode"] = field(default=None, repr=False)


class KDTree:
    """
    KD-Tree K-NN search.

    insert() is O(log N) average (tree stays unbalanced after many inserts).
    knn()    is O(log N) average, O(N) worst case at high dimensions.

    Note: KD-Trees do NOT support efficient deletion — vectors removed from
    VectorDB are simply marked deleted and filtered during search.
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._root: Optional[_KDNode] = None
        self._deleted_ids: set = set()

    # ── Insertion ──────────────────────────────────────────────────────────────

    def _insert(self, node: Optional[_KDNode], item: VectorItem, depth: int) -> _KDNode:
        if node is None:
            return _KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem) -> None:
        self._root = self._insert(self._root, item, 0)
        # Un-mark if re-inserted after deletion
        self._deleted_ids.discard(item.id)

    def mark_deleted(self, item_id: int) -> None:
        """Mark a node as deleted (soft delete — tree structure unchanged)."""
        self._deleted_ids.add(item_id)

    # ── Search ─────────────────────────────────────────────────────────────────

    def knn(
        self,
        query: List[float],
        k: int,
        dist_fn: DistFn,
    ) -> List[Tuple[float, int]]:
        """
        Return up to k nearest neighbors as (distance, id) tuples,
        sorted by distance ascending.

        Uses a max-heap of size k and the hyperplane-intersection pruning rule
        to skip entire subtrees.
        """
        # heap stores (-distance, id) so we can efficiently pop the farthest
        heap: List[Tuple[float, int]] = []  # max-heap (negated distances)

        def _search(node: Optional[_KDNode], depth: int) -> None:
            if node is None:
                return

            axis = depth % self.dims
            q_val = query[axis]
            n_val = node.item.emb[axis]

            # Recurse into the closer subtree first
            if q_val < n_val:
                closer, farther = node.left, node.right
            else:
                closer, farther = node.right, node.left

            _search(closer, depth + 1)

            # Evaluate current node (skip soft-deleted)
            if node.item.id not in self._deleted_ids:
                d = dist_fn(query, node.item.emb)
                if len(heap) < k:
                    heapq.heappush(heap, (-d, node.item.id))
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, node.item.id))

            # Hyperplane intersection check:
            # Only visit the farther subtree if it could contain a closer point.
            axis_dist = abs(q_val - n_val)
            if len(heap) < k or axis_dist < -heap[0][0]:
                _search(farther, depth + 1)

        _search(self._root, 0)

        # Convert max-heap to sorted ascending list
        return sorted((-neg_d, nid) for neg_d, nid in heap)
