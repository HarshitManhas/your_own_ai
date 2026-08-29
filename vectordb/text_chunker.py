"""
text_chunker.py — Splits long documents into overlapping word-based chunks.

Default: ~250 words per chunk, 30-word overlap between consecutive chunks.
Overlap ensures context is not lost at chunk boundaries when searching.

Equivalent to TextChunker.java in the original Java project.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_words: int = 250,
    overlap_words: int = 30,
) -> List[str]:
    """
    Split a document into overlapping word-based chunks.

    Args:
        text:         The full document text.
        chunk_words:  Words per chunk (default: 250).
        overlap_words: Overlapping words between consecutive chunks (default: 30).

    Returns:
        List of text chunk strings. Always contains at least one element.
    """
    words = text.split()
    if not words:
        return [""]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words  # slide window with overlap

    return chunks if chunks else [text]
