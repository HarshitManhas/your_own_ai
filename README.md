# Your Own AI — Vector Database from Scratch (Python)

A **production-grade Vector Database** built entirely from scratch in Python, with a rich web UI.  
No third-party vector DB libraries — every algorithm is hand-rolled, educational, and readable.

> **This is a Python port** of the [original Java project](https://github.com/HarshitManhas/your_own_ai).  
> **TL;DR:** You paste a document, a local LLM (Ollama) converts it to a high-dimensional vector, it gets stored in three different index structures simultaneously, and you can query it with natural language. All of this happens on your machine, privately.

---

## Features

- **Three search algorithms** implemented from scratch and benchmarked side-by-side:
  - **Brute Force** — exact O(N·d) baseline
  - **KD-Tree** — O(log N) for low dimensions
  - **HNSW** — O(log N) at any dimension (same algorithm as Pinecone, Weaviate, Chroma)
- **RAG pipeline** — paste documents, ask questions, get answers from a local LLM
- **PCA scatter plot** — visualize all vectors in 2D (pure Python, no NumPy)
- **100% local** — no cloud, no API keys, no data leaves your machine
- **Three distance metrics** — Cosine, Euclidean, Manhattan

## Tech Stack

| Component | Technology |
|---|---|
| HTTP Server | FastAPI + Uvicorn |
| Search Indexes | Hand-rolled (BruteForce, KD-Tree, HNSW) |
| Embeddings | Ollama `nomic-embed-text` (768D) |
| Text Generation | Ollama `llama3.2` |
| Frontend | Vanilla HTML/CSS/JS (original from Java project) |

## Project Structure

```
your-own-ai/
├── main.py                  # FastAPI server — all REST endpoints
├── vectordb/
│   ├── models.py            # VectorItem, DocItem dataclasses
│   ├── distance_metrics.py  # Cosine, Euclidean, Manhattan
│   ├── brute_force.py       # O(N·d) exhaustive scan
│   ├── kd_tree.py           # Binary space partitioning tree
│   ├── hnsw.py              # Hierarchical Navigable Small World graph
│   ├── vector_db.py         # Unified index (all 3 kept in sync)
│   ├── document_db.py       # HNSW index for 768D document embeddings
│   ├── text_chunker.py      # Sliding-window document chunker
│   ├── ollama_client.py     # Ollama REST API client
│   └── demo_data.py         # 20 pre-built 16D semantic vectors
├── static/
│   └── index.html           # Web UI (PCA plot, search, benchmark, RAG chat)
└── requirements.txt         # fastapi, uvicorn, httpx
```

## Prerequisites & Setup

### 1. Python 3.10+

```bash
pip install -r requirements.txt
```

### 2. Ollama (optional — needed for RAG / document Q&A)

Install from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull nomic-embed-text   # for embeddings (768D)
ollama pull llama3.2           # for text generation
```

## Running

```bash
python main.py
```

Then open **http://localhost:8080** in your browser.

The server starts with 20 pre-loaded demo vectors so the visualization panel works immediately — even without Ollama.

## REST API

| Method | Route | Description |
|---|---|---|
| `GET` | `/search?v=f1,...,f16&k=5&metric=cosine&algo=hnsw` | Vector similarity search |
| `POST` | `/insert` | Insert a new 16D demo vector |
| `DELETE` | `/delete/{id}` | Remove a demo vector |
| `POST` | `/benchmark` | Run all 3 algos and compare timing |
| `GET` | `/vectors` | All demo vectors with PCA coords |
| `GET` | `/ollama/status` | Check if Ollama is running |
| `GET` | `/doc/list` | List stored document chunks |
| `POST` | `/doc/add` | Embed and store a document |
| `DELETE` | `/doc/{id}` | Remove a document chunk |
| `POST` | `/doc/ask` | RAG: stream an LLM answer using your documents |

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web UI (index.html)                         │
│    PCA Scatter Plot  │  Chat / RAG  │  Benchmark Panel          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP REST (port 8080)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     main.py  (FastAPI Server)                   │
│  Routes: /search  /insert  /delete  /benchmark  /doc/ask ...   │
└────────────┬──────────────────────────────┬────────────────────┘
             │                              │
             ▼                              ▼
┌────────────────────────┐      ┌──────────────────────────────┐
│   VectorDB (16D demo)  │      │  DocumentDB (768D real docs) │
│   BruteForce + KDTree  │      │  HNSW + BruteForce fallback  │
│   + HNSW (all synced)  │      │                              │
└────────────────────────┘      └──────────┬───────────────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │   OllamaClient        │
                               │  /api/embeddings      │  ← nomic-embed-text (768D)
                               │  /api/generate        │  ← llama3.2 (text answer)
                               └───────────────────────┘
```
