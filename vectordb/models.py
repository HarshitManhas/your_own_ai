"""
models.py — Core data classes.

VectorItem  : a single stored vector (id, metadata, category, embedding).
DocItem     : a document chunk stored in DocumentDB (id, title, text, embedding).

Equivalent to VectorItem.java and DocumentDB.DocItem in the original Java project.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VectorItem:
    """Represents a single vector stored in the database."""
    id: int
    metadata: str
    category: str
    emb: List[float]


@dataclass
class DocItem:
    """Represents a document chunk stored in DocumentDB (768D embedding)."""
    id: int
    title: str
    text: str
    emb: List[float]
