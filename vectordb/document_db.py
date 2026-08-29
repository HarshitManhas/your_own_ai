"""
document_db.py — HNSW index over real Ollama embeddings (768D).

Handles the RAG document storage: stores document chunks with their 768D
embeddings and supports similarity search for the RAG pipeline.
Uses HNSW for large datasets and falls back to BruteForce for < 10 chunks.

Thread-safe via threading.Lock.
Equivalent to DocumentDB.java in the original Java project.
"""

import threading
from typing import Dict, List, Optional, Tuple

from .brute_force import BruteForce
from .distance_metrics import cosine, get_dist_fn
from .hnsw import HNSW
from .models import DocItem, VectorItem


class DocumentDB:
    """
    HNSW-backed store for document chunks with 768D Ollama embeddings.

    insert()   — add a chunk (pre-embedded)
    remove()   — remove a chunk by id
    search()   — return top-k similar chunks
    list_all() — return all stored chunks
    """

    def __init__(self) -> None:
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(M=16, ef_build=200)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        """Insert a pre-embedded document chunk. Returns the assigned ID."""
        dist_fn = cosine
        with self._lock:
            item_id = self._next_id
            self._next_id += 1
            self._dims = len(emb)

            doc = DocItem(id=item_id, title=title, text=text, emb=emb)
            self._store[item_id] = doc

            # Build a VectorItem wrapper for the generic indexes
            vi = VectorItem(id=item_id, metadata=title, category="doc", emb=emb)
            self._bf.insert(vi)
            self._hnsw.insert(vi, dist_fn)
            return item_id

    def remove(self, item_id: int) -> bool:
        """Remove a document chunk by ID."""
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            return True

    def search(self, query_emb: List[float], k: int) -> List[DocItem]:
        """
        Return the top-k most similar document chunks for the given embedding.
        Uses HNSW for ≥10 chunks, brute force otherwise.
        """
        dist_fn = cosine
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(query_emb, k, dist_fn)
            else:
                raw = self._hnsw.knn(query_emb, k, dist_fn)

            results: List[DocItem] = []
            for _dist, nid in raw:
                doc = self._store.get(nid)
                if doc is not None:
                    results.append(doc)
            return results

    def list_all(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)
