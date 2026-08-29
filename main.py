"""
main.py — FastAPI HTTP server entry point.

Wires together all components and exposes the same REST API as the
original Java/Javalin project. All routes are identical so the original
frontend HTML (index.html) works without any modification.

Run with:
    python main.py
    # or: uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Routes:
    GET  /                     → serves index.html
    GET  /search               → vector similarity search
    POST /insert               → insert a new demo vector
    DELETE /delete/{id}        → remove a demo vector
    POST /benchmark            → run all 3 algos and compare timing
    GET  /vectors              → list all demo vectors (for PCA scatter plot)
    GET  /ollama/status        → check if Ollama is running
    GET  /doc/list             → list stored document chunks
    POST /doc/add              → embed + store document via Ollama
    DELETE /doc/{id}           → remove a document chunk
    POST /doc/ask              → RAG: find relevant chunks + stream LLM answer
"""

import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vectordb import demo_data
from vectordb.document_db import DocumentDB
from vectordb.ollama_client import OllamaClient
from vectordb.text_chunker import chunk_text
from vectordb.vector_db import VectorDB
from vectordb.distance_metrics import get_dist_fn

# ── Constants ─────────────────────────────────────────────────────────────────
DIMS = 16
STATIC_DIR = Path(__file__).parent / "static"

# ── Global state ──────────────────────────────────────────────────────────────
db: VectorDB
doc_db: DocumentDB
ollama: OllamaClient


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize all components on startup."""
    global db, doc_db, ollama

    db = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()

    demo_data.load(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    yield  # server runs here


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Your Own AI — Vector Database Engine (Python)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static file serving ────────────────────────────────────────────────────────
@app.get("/")
async def root() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

# Mount static files directory for any other assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic request models ────────────────────────────────────────────────────

class InsertRequest(BaseModel):
    metadata: str
    category: Optional[str] = "default"
    embedding: List[float]


class BenchmarkRequest(BaseModel):
    v: Optional[str] = None
    k: Optional[int] = 5
    metric: Optional[str] = "cosine"


class DocAddRequest(BaseModel):
    title: str
    text: str


class DocAskRequest(BaseModel):
    question: str
    k: Optional[int] = 5


# ── Helper functions ───────────────────────────────────────────────────────────

def parse_vec(v_param: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated float vector string."""
    if not v_param:
        return None
    try:
        return [float(x.strip()) for x in v_param.split(",")]
    except ValueError:
        return None


def pca_2d(vectors: List[List[float]]) -> List[List[float]]:
    """
    Reduce N×D vectors to N×2 using simple PCA (first 2 principal components).
    Pure Python implementation — no NumPy required.
    """
    if not vectors or len(vectors[0]) < 2:
        return [[0.0, 0.0] for _ in vectors]

    n = len(vectors)
    d = len(vectors[0])

    # Center the data
    means = [sum(v[i] for v in vectors) / n for i in range(d)]
    centered = [[v[i] - means[i] for i in range(d)] for v in vectors]

    # Covariance matrix (d×d)
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / max(n - 1, 1)

    # Power iteration to find first two principal components
    def power_iter(matrix: List[List[float]], deflate_vec: Optional[List[float]] = None) -> List[float]:
        import random as _r
        vec = [_r.gauss(0, 1) for _ in range(d)]
        # Deflate away from first component if provided
        for _ in range(100):
            # Matrix-vector multiply
            new_vec = [sum(matrix[i][j] * vec[j] for j in range(d)) for i in range(d)]
            # Deflate
            if deflate_vec is not None:
                dot = sum(new_vec[i] * deflate_vec[i] for i in range(d))
                new_vec = [new_vec[i] - dot * deflate_vec[i] for i in range(d)]
            # Normalize
            norm = math.sqrt(sum(x * x for x in new_vec)) or 1.0
            new_vec = [x / norm for x in new_vec]
            vec = new_vec
        return vec

    pc1 = power_iter(cov)
    pc2 = power_iter(cov, deflate_vec=pc1)

    # Project onto first 2 PCs
    projected = [
        [sum(v[i] * pc1[i] for i in range(d)), sum(v[i] * pc2[i] for i in range(d))]
        for v in centered
    ]
    return projected


# ── Demo Vector Endpoints ──────────────────────────────────────────────────────

@app.get("/search")
async def search(
    v: Optional[str] = None,
    k: int = 5,
    metric: str = "cosine",
    algo: str = "hnsw",
) -> JSONResponse:
    query = parse_vec(v)
    if query is None or len(query) != DIMS:
        raise HTTPException(status_code=400, detail=f"need {DIMS}D vector as ?v=f1,f2,...,f{DIMS}")

    out = db.search(query, k, metric, algo)

    results = [
        {
            "id": h.id,
            "metadata": h.metadata,
            "category": h.category,
            "distance": h.distance,
            "embedding": h.emb,
        }
        for h in out.hits
    ]
    return JSONResponse({
        "results": results,
        "latencyUs": out.microseconds,
        "algo": out.algo,
        "metric": out.metric,
    })


@app.post("/insert")
async def insert(req: InsertRequest) -> JSONResponse:
    if len(req.embedding) != DIMS:
        raise HTTPException(status_code=400, detail=f"embedding must be {DIMS}D")
    item_id = db.insert(
        metadata=req.metadata,
        category=req.category or "default",
        emb=req.embedding,
        metric="cosine",
    )
    return JSONResponse({"id": item_id})


@app.delete("/delete/{item_id}")
async def delete(item_id: int) -> JSONResponse:
    ok = db.remove(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"ID {item_id} not found")
    return JSONResponse({"deleted": item_id})


@app.post("/benchmark")
async def benchmark(request: Request) -> JSONResponse:
    body = await request.json()
    v_param = body.get("v")
    k = int(body.get("k", 5))
    metric = body.get("metric", "cosine")

    query = parse_vec(v_param)
    if query is None or len(query) != DIMS:
        # Use a neutral query if none provided
        query = [1.0 / math.sqrt(DIMS)] * DIMS

    results = db.benchmark(query, k, metric)
    return JSONResponse([
        {
            "algo": r.algo,
            "latencyUs": r.microseconds,
            "results": [
                {"id": h.id, "metadata": h.metadata, "category": h.category, "distance": h.distance}
                for h in r.results
            ],
        }
        for r in results
    ])


@app.get("/vectors")
async def list_vectors() -> JSONResponse:
    """Return all demo vectors for the PCA scatter plot."""
    items = db.all_items()
    if not items:
        return JSONResponse({"vectors": [], "pca": []})

    raw_embs = [item.emb for item in items]
    pca_coords = pca_2d(raw_embs)

    vectors = [
        {
            "id": item.id,
            "metadata": item.metadata,
            "category": item.category,
            "embedding": item.emb,
            "pca": pca_coords[i],
        }
        for i, item in enumerate(items)
    ]
    return JSONResponse({"vectors": vectors, "count": len(vectors)})


# ── Ollama Status ──────────────────────────────────────────────────────────────

@app.get("/ollama/status")
async def ollama_status() -> JSONResponse:
    up = ollama.is_available()
    return JSONResponse({
        "available": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "url": ollama.base_url,
    })


# ── Document / RAG Endpoints ───────────────────────────────────────────────────

@app.get("/doc/list")
async def doc_list() -> JSONResponse:
    docs = doc_db.list_all()
    return JSONResponse({
        "documents": [
            {"id": d.id, "title": d.title, "text": d.text[:200] + "..." if len(d.text) > 200 else d.text}
            for d in docs
        ],
        "count": len(docs),
    })


@app.post("/doc/add")
async def doc_add(req: DocAddRequest) -> JSONResponse:
    if not ollama.is_available():
        raise HTTPException(status_code=503, detail="Ollama is offline. Install from ollama.com")

    chunks = chunk_text(req.text)
    inserted_ids: List[int] = []

    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if emb is None:
            raise HTTPException(status_code=502, detail=f"Ollama embedding failed for chunk {i}")
        chunk_title = req.title if len(chunks) == 1 else f"{req.title} (chunk {i+1}/{len(chunks)})"
        doc_id = doc_db.insert(chunk_title, chunk, emb)
        inserted_ids.append(doc_id)

    return JSONResponse({
        "inserted": inserted_ids,
        "chunks": len(chunks),
        "title": req.title,
    })


@app.delete("/doc/{doc_id}")
async def doc_delete(doc_id: int) -> JSONResponse:
    ok = doc_db.remove(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found")
    return JSONResponse({"deleted": doc_id})


@app.post("/doc/ask")
async def doc_ask(req: DocAskRequest) -> StreamingResponse:
    """
    RAG pipeline:
    1. Embed the question via Ollama.
    2. Find the top-k most similar document chunks (HNSW search).
    3. Assemble a context prompt.
    4. Stream the LLM's answer back to the client.
    """
    if not ollama.is_available():
        raise HTTPException(status_code=503, detail="Ollama is offline")

    if doc_db.size() == 0:
        raise HTTPException(status_code=400, detail="No documents stored. Add some documents first.")

    # Step 1: Embed the question
    q_emb = ollama.embed(req.question)
    if q_emb is None:
        raise HTTPException(status_code=502, detail="Failed to embed question via Ollama")

    # Step 2: Retrieve top-k relevant chunks
    top_chunks = doc_db.search(q_emb, req.k or 5)
    if not top_chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Step 3: Build context prompt
    context_parts = [f"[{i+1}] {chunk.title}\n{chunk.text}" for i, chunk in enumerate(top_chunks)]
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"You are a helpful assistant. Use ONLY the following context to answer the question. "
        f"If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}\n\n"
        f"Answer:"
    )

    # Step 4: Stream the LLM response
    def generate_stream():
        for token in ollama.stream_generate(prompt):
            yield token

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"X-Sources": str(len(top_chunks))},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
