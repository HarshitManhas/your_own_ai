"""
hnsw.py — Hierarchical Navigable Small World graph.

The same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.
Builds a multi-layer graph where each layer is progressively sparser.
Searches start at the top layer and zoom in, achieving O(log N) complexity
at any number of dimensions — unlike KD-Trees which degrade at high dims.

Key parameters:
    M        = 16   — max connections per node per layer (more = better recall, more memory)
    M0       = 2*M  — max connections at layer 0 (denser base)
    ef_build = 200  — beam width during construction (more = better quality, slower build)
    mL       = 1/ln(M) — level generation factor

Equivalent to HNSW.java in the original Java project.
"""

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .distance_metrics import DistFn
from .models import VectorItem


@dataclass
class _Node:
    item: VectorItem
    max_layer: int
    # neighbors[layer] = list of neighbor node IDs
    neighbors: List[List[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.neighbors = [[] for _ in range(self.max_layer + 1)]


class HNSW:
    """
    Hierarchical Navigable Small World graph index.

    insert() is O(log N)
    knn()    is O(log N) approximate — works well at any dimension
    remove() is soft delete (marks node, skips during search)
    """

    def __init__(self, M: int = 16, ef_build: int = 200) -> None:
        self.M = M
        self.M0 = 2 * M
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(M)

        self._graph: Dict[int, _Node] = {}  # node_id → _Node
        self._top_layer: int = -1
        self._entry_point: int = -1
        self._deleted_ids: Set[int] = set()
        self._rng = random.Random(42)

    # ── Level Generation ───────────────────────────────────────────────────────

    def _random_level(self) -> int:
        """Sample a layer level using the HNSW exponential distribution."""
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    # ── Layer Search (beam search within one layer) ────────────────────────────

    def _search_layer(
        self,
        query: List[float],
        ep_id: int,
        ef: int,
        layer: int,
        dist_fn: DistFn,
    ) -> List[Tuple[float, int]]:
        """
        Beam search within a single layer.
        Returns up to ef candidates as (distance, id) sorted ascending.
        """
        visited: Set[int] = {ep_id}

        ep_node = self._graph[ep_id]
        d0 = dist_fn(query, ep_node.item.emb)

        # candidates: min-heap (closest first)  [dist, id]
        candidates: List[Tuple[float, int]] = [(d0, ep_id)]
        heapq.heapify(candidates)

        # found: max-heap (farthest first) stored as (-dist, id)
        found: List[Tuple[float, int]] = [(-d0, ep_id)]
        heapq.heapify(found)

        while candidates:
            cd, cid = heapq.heappop(candidates)

            # If closest candidate is farther than the worst in found → stop
            if found and cd > -found[0][0] and len(found) >= ef:
                break

            cnode = self._graph.get(cid)
            if cnode is None or layer >= len(cnode.neighbors):
                continue

            for nid in cnode.neighbors[layer]:
                if nid in visited:
                    continue
                visited.add(nid)

                if nid in self._deleted_ids:
                    continue

                nnode = self._graph.get(nid)
                if nnode is None:
                    continue

                nd = dist_fn(query, nnode.item.emb)
                worst_found = -found[0][0] if found else float("inf")

                if nd < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)  # evict farthest

        # Convert to sorted ascending list
        return sorted((-neg_d, nid) for neg_d, nid in found)

    # ── Neighbor Selection (Heuristic) ─────────────────────────────────────────

    def _select_neighbors(
        self,
        candidates: List[Tuple[float, int]],
        max_neighbors: int,
    ) -> List[int]:
        """
        Simple greedy selection: keep the max_neighbors closest candidates.
        (Full HNSW paper uses a diversity heuristic; simple greedy works well.)
        """
        return [nid for _, nid in candidates[:max_neighbors]]

    # ── Insert ─────────────────────────────────────────────────────────────────

    def insert(self, item: VectorItem, dist_fn: DistFn) -> None:
        """
        Insert a new vector into the HNSW graph.

        1. Sample a random insertion level.
        2. Descend from the top layer to level+1 greedily.
        3. Run beam search at each level from the insertion level down to 0.
        4. Connect the new node to its neighbors at each level.
        5. Prune neighbor lists that exceed M / M0.
        """
        node_id = item.id
        node_level = self._random_level()
        new_node = _Node(item, node_level)
        self._graph[node_id] = new_node
        self._deleted_ids.discard(node_id)

        if self._entry_point == -1:
            # First node — it becomes the graph entry point
            self._entry_point = node_id
            self._top_layer = node_level
            return

        ep = self._entry_point
        current_top = self._top_layer

        # Phase 1: descend from top layer to node_level + 1 (greedy, ef=1)
        for layer in range(current_top, node_level, -1):
            candidates = self._search_layer(item.emb, ep, 1, layer, dist_fn)
            if candidates:
                ep = candidates[0][1]

        # Phase 2: insert at each layer from node_level down to 0
        for layer in range(min(node_level, current_top), -1, -1):
            max_c = self.M0 if layer == 0 else self.M
            candidates = self._search_layer(
                item.emb, ep, self.ef_build, layer, dist_fn
            )
            neighbors = self._select_neighbors(candidates, max_c)
            new_node.neighbors[layer] = neighbors

            # Bidirectional connections
            for nid in neighbors:
                nnode = self._graph.get(nid)
                if nnode is None or layer >= len(nnode.neighbors):
                    continue
                nnode.neighbors[layer].append(node_id)
                # Prune if over limit
                if len(nnode.neighbors[layer]) > max_c:
                    # Keep only the max_c closest
                    scored = [
                        (dist_fn(nnode.item.emb, self._graph[x].item.emb), x)
                        for x in nnode.neighbors[layer]
                        if x in self._graph
                    ]
                    scored.sort()
                    nnode.neighbors[layer] = [x for _, x in scored[:max_c]]

            if candidates:
                ep = candidates[0][1]

        # Update entry point if new node is on a higher layer
        if node_level > current_top:
            self._entry_point = node_id
            self._top_layer = node_level

    # ── Search ─────────────────────────────────────────────────────────────────

    def knn(
        self,
        query: List[float],
        k: int,
        dist_fn: DistFn,
        ef: int = 50,
    ) -> List[Tuple[float, int]]:
        """
        Approximate K-NN search.

        1. Descend from top layer to layer 1 greedily (ef=1).
        2. At layer 0 run full beam search with ef candidates.
        3. Return top k.
        """
        if self._entry_point == -1:
            return []

        ep = self._entry_point
        ef = max(ef, k)

        # Greedy descent through upper layers
        for layer in range(self._top_layer, 0, -1):
            candidates = self._search_layer(query, ep, 1, layer, dist_fn)
            if candidates:
                ep = candidates[0][1]

        # Full beam search at layer 0
        candidates = self._search_layer(query, ep, ef, 0, dist_fn)

        # Filter deleted nodes and return top k
        result = [(d, nid) for d, nid in candidates if nid not in self._deleted_ids]
        return result[:k]

    # ── Remove ─────────────────────────────────────────────────────────────────

    def remove(self, item_id: int) -> bool:
        """Soft-delete a node (mark as deleted, skip during search)."""
        if item_id not in self._graph:
            return False
        self._deleted_ids.add(item_id)
        return True

    # ── Graph Info (for API) ───────────────────────────────────────────────────

    def graph_info(self) -> dict:
        """Return graph statistics for the /vectors endpoint."""
        active = [n for nid, n in self._graph.items() if nid not in self._deleted_ids]
        edges = sum(len(n.neighbors[0]) for n in active) // 2 if active else 0
        return {
            "nodes": len(active),
            "edges": edges,
            "layers": self._top_layer + 1,
            "entry_point": self._entry_point,
        }

    def size(self) -> int:
        return len(self._graph) - len(self._deleted_ids)
