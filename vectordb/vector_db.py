"""
vector_db.py — Unified VectorDB interface over BruteForce, KD-Tree, and HNSW.

Manages the 16D demo vector index.  Every insert() call adds the vector to
all three indexes simultaneously so they stay in sync and can be compared.

Thread-safe via threading.Lock (equivalent to Java's ReentrantLock).

Equivalent to VectorDB.java in the original Java project.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .brute_force import BruteForce
from .distance_metrics import get_dist_fn
from .hnsw import HNSW
from .kd_tree import KDTree
from .models import VectorItem


@dataclass
class Hit:
    id: int
    metadata: str
    category: str
    distance: float
    emb: List[float]


@dataclass
class SearchResult:
    hits: List[Hit]
    microseconds: int
    algo: str
    metric: str


@dataclass
class BenchResult:
    algo: str
    microseconds: int
    results: List[Hit]


class VectorDB:
    """
    Unified vector store over three indexes.

    All three indexes are kept in sync:
        BruteForce — exact, O(N·d)
        KD-Tree    — approximate for high dims, O(log N) for low dims
        HNSW       — approximate nearest neighbor, O(log N) any dimension
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(M=16, ef_build=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(
        self,
        metadata: str,
        category: str,
        emb: List[float],
        metric: str = "cosine",
    ) -> int:
        dist_fn = get_dist_fn(metric)
        with self._lock:
            item = VectorItem(
                id=self._next_id,
                metadata=metadata,
                category=category,
                emb=emb,
            )
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._kdt.mark_deleted(item_id)
            self._hnsw.remove(item_id)
            return True

    def search(
        self,
        query: List[float],
        k: int,
        metric: str = "cosine",
        algo: str = "hnsw",
    ) -> SearchResult:
        dist_fn = get_dist_fn(metric)
        with self._lock:
            t0 = time.perf_counter()
            raw = self._run_search(query, k, dist_fn, algo)
            elapsed_us = int((time.perf_counter() - t0) * 1_000_000)

        hits = self._to_hits(raw)
        return SearchResult(hits=hits, microseconds=elapsed_us, algo=algo, metric=metric)

    def benchmark(
        self,
        query: List[float],
        k: int,
        metric: str = "cosine",
    ) -> List[BenchResult]:
        dist_fn = get_dist_fn(metric)
        results: List[BenchResult] = []

        with self._lock:
            for algo in ("hnsw", "kdtree", "bruteforce"):
                t0 = time.perf_counter()
                raw = self._run_search(query, k, dist_fn, algo)
                elapsed_us = int((time.perf_counter() - t0) * 1_000_000)
                results.append(
                    BenchResult(
                        algo=algo,
                        microseconds=elapsed_us,
                        results=self._to_hits(raw),
                    )
                )

        return results

    def _run_search(
        self,
        query: List[float],
        k: int,
        dist_fn,
        algo: str,
    ) -> List[Tuple[float, int]]:
        algo = algo.lower()
        if algo == "bruteforce":
            return self._bf.knn(query, k, dist_fn)
        elif algo == "kdtree":
            return self._kdt.knn(query, k, dist_fn)
        else:  # default: hnsw
            if self._hnsw.size() < 10:
                # Fall back to brute force for very small datasets
                return self._bf.knn(query, k, dist_fn)
            return self._hnsw.knn(query, k, dist_fn)

    def _to_hits(self, raw: List[Tuple[float, int]]) -> List[Hit]:
        hits: List[Hit] = []
        for dist, nid in raw:
            item = self._store.get(nid)
            if item is None:
                continue
            hits.append(
                Hit(
                    id=item.id,
                    metadata=item.metadata,
                    category=item.category,
                    distance=dist,
                    emb=item.emb,
                )
            )
        return hits

    def all_items(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)
