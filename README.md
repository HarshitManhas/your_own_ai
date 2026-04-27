# Your Own AI — Vector Database from Scratch

A **production-grade Vector Database** built entirely from scratch in Java, with a rich web UI.  
No third-party vector DB libraries — every algorithm is hand-rolled, educational, and readable.

> **TL;DR:** You paste a document, a local LLM (Ollama) converts it to a high-dimensional vector, it gets stored in three different index structures simultaneously, and you can query it with natural language. All of this happens on your machine, privately.

---

## Table of Contents

- [What is this project?](#what-is-this-project)
- [How does it work? (End-to-end flow)](#how-does-it-work-end-to-end-flow)
- [Core Concepts Explained](#core-concepts-explained)
  - [Vector Embeddings](#1-vector-embeddings)
  - [Similarity Search (K-NN)](#2-similarity-search-k-nn)
  - [Distance Metrics](#3-distance-metrics)
  - [Brute Force Search](#4-brute-force-search)
  - [KD-Tree](#5-kd-tree)
  - [HNSW — Hierarchical Navigable Small World](#6-hnsw--hierarchical-navigable-small-world)
  - [RAG — Retrieval-Augmented Generation](#7-rag--retrieval-augmented-generation)
  - [Text Chunking](#8-text-chunking)
  - [Thread Safety](#9-thread-safety)
  - [PCA — Principal Component Analysis](#10-pca--principal-component-analysis)
- [Project Architecture](#project-architecture)
- [File-by-File Breakdown](#file-by-file-breakdown)
- [Features](#features)
- [Prerequisites & Setup](#prerequisites--setup)
- [Build & Run](#build--run)
- [REST API Reference](#rest-api-reference)
- [Use a Smaller LLM](#use-a-smallerfaster-llm)

---

## What is this project?

This project is a **vector database** — the same kind of database that powers AI applications like ChatGPT's memory, Notion AI, GitHub Copilot, and document search in tools like Perplexity.

Instead of storing and querying rows of data like a traditional SQL database, a vector database stores **mathematical representations of meaning** (called embeddings) and answers questions like:
- *"What other sentences are semantically similar to this one?"*
- *"Find the document most relevant to my question."*
- *"Which food items cluster with 'pizza'?"*

This project implements **three different search algorithms** side-by-side so you can see exactly how they work and compare their performance. It also ships a full **RAG pipeline** — paste in your own documents, ask questions in plain English, and a local LLM answers using your documents as context.

Everything runs **100% locally** on your machine. No cloud. No API keys. No data leaves your computer.

---

## How does it work? (End-to-end flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (index.html)                     │
│    PCA Scatter Plot  │  Chat / RAG  │  Benchmark Panel          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP REST (port 8080)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Main.java  (Javalin HTTP Server)            │
│  Routes: /search  /insert  /delete  /benchmark  /doc/ask ...    │
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
                               │  /api/embeddings      │  ←── nomic-embed-text (768D)
                               │  /api/generate        │  ←── llama3.2 (text answer)
                               └───────────────────────┘
```

**Step-by-step for the RAG pipeline:**

1. You paste a document into the web UI and click "Add Document".
2. `Main.java` receives the POST request and calls `TextChunker` to split the document into overlapping ~250-word chunks.
3. Each chunk is sent to **Ollama** (`OllamaClient.embed()`), which runs the `nomic-embed-text` model locally and returns a **768-dimensional float array** — the chunk's meaning encoded as a vector.
4. Each chunk+embedding is stored in `DocumentDB`, which inserts it into an **HNSW graph** (or brute force if < 10 documents).
5. When you ask a question, the question text is also embedded into a 768D vector.
6. The HNSW index finds the **top-k most similar chunks** (nearest neighbors in 768D space).
7. The matched chunks are assembled into a context prompt and sent to **Ollama's generation API** (`llama3.2`).
8. The LLM reads the context and writes an answer. This is streamed back to the chat UI.

---

## Core Concepts Explained

### 1. Vector Embeddings

A vector embedding is how AI models convert text (or images, audio, etc.) into numbers.

```
"Pizza is delicious"  →  [0.12, -0.45, 0.88, 0.03, ..., 0.67]  (768 numbers)
"I love pasta"        →  [0.11, -0.43, 0.85, 0.05, ..., 0.65]  (768 numbers, very similar!)
"The stock market"    →  [-0.72, 0.31, -0.12, 0.94, ..., -0.33] (768 numbers, very different)
```

Sentences with similar meaning produce vectors that are **close together in space**. This is the core magic — mathematical closeness = semantic similarity.

In this project, **16D demo vectors** are hand-crafted for visualization, and **768D vectors** come from the `nomic-embed-text` model via Ollama.

---

### 2. Similarity Search (K-NN)

**K-Nearest Neighbor (K-NN)** search answers: *"What are the K vectors closest to my query vector?"*

Given a query vector `q`, find the K stored vectors with the smallest distance to `q`.

The challenge: when you have millions of vectors in 768 dimensions, computing the distance to every single stored vector is too slow. This project shows three different approaches to solving that problem.

---

### 3. Distance Metrics

How you measure "closeness" between two vectors matters. This project supports three metrics, all implemented in `DistanceMetrics.java`:

| Metric | Formula | Best For |
|---|---|---|
| **Cosine** | `1 - (A·B) / (‖A‖ × ‖B‖)` | Text/NLP — cares about direction, not magnitude |
| **Euclidean** | `√Σ(aᵢ - bᵢ)²` | Geometric distance — cares about magnitude |
| **Manhattan** | `Σ|aᵢ - bᵢ|` | Grid-like spaces, robust to outliers |

For text embeddings (the RAG pipeline), **cosine distance** is always used. For the 16D demo, you can switch at runtime.

---

### 4. Brute Force Search

**File:** `BruteForce.java`  
**Complexity:** O(N × d) — linear in both count and dimensions

The simplest possible approach: compute the distance from the query to **every single stored vector**, then sort and return the top K.

```
For each stored vector v:
    compute distance(query, v)
Sort all results by distance
Return top K
```

✅ Always 100% correct (exact results)  
✅ Simple to implement  
❌ Slow at large scale — gets slower the more vectors you store

This is the **baseline** against which the other algorithms are benchmarked.

---

### 5. KD-Tree

**File:** `KDTree.java`  
**Complexity:** O(log N) average — degrades at high dimensions

A KD-Tree (K-Dimensional Tree) is a binary search tree that partitions space along alternating dimensions.

**How insertion works:**
1. At depth 0, split on dimension 0 (x-axis). Vectors with smaller x go left.
2. At depth 1, split on dimension 1 (y-axis). Vectors with smaller y go left.
3. Repeat, cycling through all dimensions.

**How search works:**
1. Descend the tree like a normal BST to find the nearest leaf.
2. Track the best distance seen so far.
3. Backtrack and check sibling subtrees **only if they could possibly contain a closer point** (the "hyperplane intersection" check).
4. This pruning makes most searches skip most of the tree.

**The curse of dimensionality:** At high dimensions (like 768D), almost every subtree *could* contain a closer point, so backtracking visits nearly all nodes — degenerating to O(N). That's why KD-Trees work great for 16D but not for 768D.

---

### 6. HNSW — Hierarchical Navigable Small World

**File:** `HNSW.java`  
**Complexity:** O(log N) — works at any dimension  
**Used by:** Pinecone, Weaviate, Chroma, Milvus, pgvector

HNSW is the industry-standard algorithm for approximate nearest-neighbor (ANN) search. It builds a **multi-layer graph** where:

- **Layer 0** (bottom): contains ALL vectors with many connections
- **Layer 1**: contains a random ~1/e subset of vectors  
- **Layer 2**: an even sparser subset...
- And so on upward

Think of it like a city's transport network: highways (top layers) let you skip large distances quickly, local roads (bottom layer) give fine-grained navigation.

**Search process:**
1. Start at the entry point of the **top layer**.
2. Greedily move to the neighbor that minimizes distance to the query.
3. When stuck (no neighbor is closer), drop down one layer.
4. At layer 0, run a full **beam search** with `ef` candidates to find the K best results.

**Key parameters:**
- `M = 16` — max connections per node per layer (more = better recall, more memory)
- `M0 = 32` — max connections at layer 0 (2×M)
- `efBuild = 200` — beam width during construction (more = better graph quality, slower inserts)
- `mL = 1/ln(M)` — layer assignment probability factor

**Node assignment to layers** uses a randomized exponential distribution:
```
level = floor(-ln(random()) × mL)
```
Most nodes land at layer 0, a few at layer 1, even fewer at layer 2, etc.

**Why bidirectional links?** When a new node `n` connects to neighbor `x`, `x` also connects back to `n`. This ensures the graph stays navigable from any starting point.

---

### 7. RAG — Retrieval-Augmented Generation

**Files:** `DocumentDB.java`, `OllamaClient.java`, `TextChunker.java`

RAG solves the problem of making an LLM "know" about your private documents without retraining it.

```
Question → Embed → HNSW Search → Top-K Chunks → Prompt + Context → LLM → Answer
```

**Why not just send the whole document to the LLM?**
- LLMs have context-length limits (can't fit a 100-page PDF)
- Retrieving only the relevant chunks is faster and produces more focused answers
- The LLM doesn't hallucinate facts that aren't in the context

**Why HNSW for documents?** Document embeddings are 768-dimensional. KD-Trees degrade at this dimension (curse of dimensionality), so HNSW's graph-based approach is used exclusively for documents. The brute force fallback is used when fewer than 10 chunks are stored (where the overhead of HNSW isn't worth it).

---

### 8. Text Chunking

**File:** `TextChunker.java`

Long documents are split into overlapping chunks before embedding. This is critical because:
1. Embedding models have a max token limit — they can't encode a whole book at once.
2. Smaller, focused chunks produce more precise embeddings.
3. Overlap (30 words by default) prevents important sentences from being cut at chunk boundaries.

```
Document: [word1 word2 ... word500]
Chunk 1:  [word1  ... word250]          (words 1-250)
Chunk 2:  [word221 ... word470]         (words 221-470, 30-word overlap with chunk 1)
Chunk 3:  [word441 ... word500]         (words 441-500)
```

Default settings: **250 words per chunk**, **30 words overlap**.

---

### 9. Thread Safety

**Files:** `VectorDB.java`, `DocumentDB.java`

Both database classes are thread-safe. Multiple HTTP requests can arrive simultaneously (Javalin runs on a thread pool), so all read and write operations are protected with a `ReentrantLock`.

```java
lock.lock();
try {
    // critical section — only one thread at a time
} finally {
    lock.unlock();  // always released, even on exception
}
```

This is equivalent to `std::mutex` in C++. The `finally` block guarantees the lock is always released, preventing deadlocks.

---

### 10. PCA — Principal Component Analysis

**File:** `index.html` (JavaScript, client-side)

The web UI visualizes 16D vectors on a 2D scatter plot using **PCA**, a dimensionality reduction technique.

PCA finds the two directions in 16D space that capture the most variance (spread) among all your vectors, then projects every vector onto those two directions. The result: similar vectors still cluster together, but you can see it in 2D.

This runs entirely in the browser — no server computation needed.

---

## Project Architecture

```
your_own_ai/
├── pom.xml                              ← Maven build config (dependencies, fat JAR)
├── index.html                           ← Frontend (served as static file)
└── src/main/java/com/vectordb/
    ├── Main.java                        ← HTTP server + all REST route handlers
    ├── VectorItem.java                  ← Core data model (id, metadata, category, emb[])
    ├── DistanceMetrics.java             ← Cosine / Euclidean / Manhattan distance functions
    ├── BruteForce.java                  ← O(N·d) exact search baseline
    ├── KDTree.java                      ← O(log N) binary space partitioning tree
    ├── HNSW.java                        ← O(log N) multilayer graph (production ANN)
    ├── VectorDB.java                    ← Unified 16D index: BruteForce + KDTree + HNSW
    ├── DocumentDB.java                  ← 768D document index for RAG (HNSW + BF fallback)
    ├── OllamaClient.java                ← HTTP client to Ollama: embed + generate
    ├── TextChunker.java                 ← Splits documents into overlapping word chunks
    └── DemoData.java                    ← 20 pre-built 16D demo vectors (4 categories)
```

**Data flow overview:**

```
Demo vectors (16D)                     Real documents (768D)
      │                                        │
      ▼                                        ▼
 VectorDB.insert()                    DocumentDB.insert()
      │                                        │
      ├─► BruteForce.insert()         OllamaClient.embed()
      ├─► KDTree.insert()                      │
      └─► HNSW.insert()              HNSW.insert() + BruteForce.insert()
                                               │
                                     HNSW.knn() → top-K chunks
                                               │
                                     OllamaClient.generate(context+question)
                                               │
                                           Answer
```

---

## File-by-File Breakdown

### `VectorItem.java`
The fundamental unit of data in the database. Every stored vector — whether a demo vector or a document chunk embedding — is a `VectorItem`.

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Unique auto-incremented identifier |
| `metadata` | `String` | Human-readable label (e.g., "Binary Search") |
| `category` | `String` | Group label (e.g., "CS", "Math", "Food") |
| `emb` | `float[]` | The embedding vector (16D or 768D) |

Immutable by design — all fields are `final`. No setters.

---

### `DistanceMetrics.java`
A utility class containing the three distance functions and a factory method.

- Uses a `@FunctionalInterface` (`DistFn`) so any metric can be passed around as a lambda.
- `getDistFn(String metric)` uses Java 21's `switch` expression to map `"cosine"` / `"manhattan"` / default → the corresponding method reference.
- Cosine distance is `1 - cosine_similarity` so that "more similar = smaller number", consistent with the other metrics.

---

### `BruteForce.java`
The simplest possible implementation. Stores all items in an `ArrayList` and scans every element on every search. Uses a `PriorityQueue`-less approach — just creates all distances, sorts, and slices.

- `insert(VectorItem)` — O(1) append
- `knn(query, k, dist)` — O(N·d) full scan
- `remove(int id)` — O(N) scan to filter

---

### `KDTree.java`
Recursive binary tree. The axis to split on cycles through dimensions: `axis = depth % dims`.

Key method: `knnSearch()` — descends to the closest leaf first (like a BST), then backtracks to check sibling subtrees only when `|diff| < bestDistanceSoFar` (the hyperplane may intersect the best-ball).

**Note on deletions:** KD-Trees don't support efficient single-node deletion while maintaining balance. Like the C++ original, this implementation rebuilds the entire tree from scratch on every delete — acceptable for demo-scale data.

---

### `HNSW.java`
The most complex file in the project. Implements the full HNSW algorithm from the 2016 paper by Malkov & Yashunin.

**Key internal classes:**
- `Node` — stores the `VectorItem` and a list-of-lists for neighbors per layer: `neighbors.get(layer)` = list of node IDs connected at that layer.

**Key methods:**
- `randomLevel()` — generates insertion layer using `floor(-ln(random) × mL)`. This exponential distribution is what creates the hierarchical structure.
- `searchLayer(query, ep, ef, layer, dist)` — beam search within one layer. Uses two heaps: a min-heap of candidates to explore and a max-heap of the best `ef` results found so far. Stops when the closest unexplored candidate is farther than the worst result in the found set.
- `insert(item, dist)` — assigns a random layer, descends greedily from the top to find a good entry point at the insertion level, then runs `searchLayer` at each layer from insertion level down to 0, connects neighbors bidirectionally, and prunes connections that exceed `M` (or `M0` at layer 0).
- `knn(query, k, ef, dist)` — greedy descent to layer 1, then full beam search at layer 0.
- `remove(id)` — removes the node and cleans up all references from neighbors. Doesn't re-link orphaned nodes (approximation acceptable for this scale).
- `getInfo()` — returns graph statistics and node/edge lists for the frontend visualization.

---

### `VectorDB.java`
The orchestration layer for the **16D demo index**. Keeps all three algorithms perfectly in sync — every insert goes to BruteForce, KDTree, and HNSW simultaneously.

- Thread-safe via `ReentrantLock`.
- `search()` dispatches to the correct algorithm based on the `algo` parameter.
- `benchmark()` runs all three algorithms on the same query and returns their execution times in microseconds.
- `remove()` rebuilds the KD-Tree after every deletion (consistent with how KD-Trees work).

---

### `DocumentDB.java`
The document index for the **RAG pipeline** — stores 768D Ollama embeddings.

Key behavior:
- If fewer than 10 chunks are stored, uses `BruteForce` (HNSW has overhead that isn't worth it for tiny datasets).
- Once 10+ chunks are stored, uses `HNSW` for all queries.
- Always cosine distance (standard for text embeddings).
- `search()` filters results by `maxDist = 0.7` — chunks more than 0.7 cosine-distance away are considered irrelevant and discarded.

---

### `OllamaClient.java`
An HTTP client wrapper around the local Ollama REST API. Uses Java 11's built-in `java.net.http.HttpClient`.

| Method | Ollama Endpoint | Description |
|---|---|---|
| `isAvailable()` | `GET /api/tags` | Health check — is Ollama running? |
| `embed(text)` | `POST /api/embeddings` | Converts text → float[] via `nomic-embed-text` |
| `generate(prompt)` | `POST /api/generate` | Generates a text answer via `llama3.2` |

- Uses **Gson** to build JSON request bodies and parse responses.
- `embed()` has a 30s timeout (embedding is fast). `generate()` has a 180s timeout (LLM generation can be slow on CPU).
- Returns safe defaults (`new float[0]`, error strings) instead of throwing exceptions — the caller handles failures gracefully.

---

### `TextChunker.java`
A stateless utility class (no instances, only static methods).

**Algorithm:**
1. Split the document on whitespace → `words[]`.
2. If total words ≤ `chunkWords`, return the whole text as one chunk.
3. Otherwise, slide a window of `chunkWords` words, stepping by `chunkWords - overlapWords` each time.
4. Each window becomes one chunk string.

Default: 250 words/chunk, 30-word overlap → step size = 220 words.

---

### `DemoData.java`
Pre-built 16-dimensional demo vectors covering four semantic categories:
- **CS** — algorithms, data structures, machine learning
- **Math** — calculus, linear algebra, statistics
- **Food** — pizza, sushi, tacos, ramen
- **Sports** — basketball, soccer, tennis, swimming

The vectors are hand-crafted so that items in the same category have high cosine similarity (dimension values cluster by topic). This makes the PCA scatter plot immediately show visible clusters when the app starts.

---

### `Main.java`
The application entry point and HTTP server. Uses **Javalin 6** (backed by Jetty).

On startup:
1. Creates a `VectorDB` (16D) and loads all 20 demo vectors from `DemoData`.
2. Creates a `DocumentDB` (768D) for real document embeddings.
3. Creates an `OllamaClient` and checks if Ollama is running.
4. Registers all REST endpoints.
5. Starts the server on port 8080.
6. Serves `index.html` as a static resource.

All endpoints are synchronous. JSON serialization uses **Jackson** (via Javalin's `ctx.json()`) for responses and **Gson** for parsing request bodies.

---

### `index.html`
A single-file frontend with no build step required. Contains:
- **PCA Scatter Plot** — Canvas-based 2D projection of all stored vectors, colored by category, with hover labels.
- **Chat UI** — Submit a question, see retrieved document chunks, read the LLM's answer.
- **Benchmark Panel** — Search with all three algorithms simultaneously, see execution times side-by-side.
- **Algorithm Selector** — Switch between HNSW / KD-Tree / Brute Force for demo searches.
- **Document Management** — Add / list / delete documents for the RAG pipeline.

All HTTP calls use the browser's native `fetch()` API.

---

## Features

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — compare speed live |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with nomic-embed-text (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD: insert, delete, search, benchmark, hnsw-info |
| **Thread-Safe** | ReentrantLock protects all concurrent reads/writes |
| **100% Local** | No cloud, no API keys, no telemetry |

---

## Prerequisites & Setup

1. **Java 21** (also works with Java 17+)
2. **Maven 3.6+**
3. **Ollama** — [https://ollama.com](https://ollama.com)

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB  — language model

# Start Ollama server (keep this running)
ollama serve
```

---

## Build & Run

```bash
# Build a fat JAR (all dependencies bundled)
mvn package -q

# Run the server
java -jar target/vectordb-1.0-SNAPSHOT.jar
```

Open **http://localhost:8080** in your browser.

Or run directly with Maven (no JAR needed):
```bash
mvn exec:java
```

---

## REST API Reference

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| POST | `/insert` | Insert a demo vector `{"meta":"...","category":"...","emb":[...]}` |
| DELETE | `/delete/:id` | Delete by ID |
| GET | `/items` | List all demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| GET | `/hnsw-info` | HNSW graph structure and layer stats |
| GET | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| GET | `/doc/list` | — | List all stored document chunks |
| DELETE | `/doc/delete/:id` | — | Delete document chunk |
| POST | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| GET | `/status` | — | Ollama status and model info |

### Example: Search via curl
```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl
```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Use a Smaller/Faster LLM

If `llama3.2` is too slow on your hardware:

```bash
ollama pull llama3.2:1b    # ~800 MB, much faster
```

Then edit `OllamaClient.java`:
```java
public String genModel = "llama3.2:1b";
```

Recompile:
```bash
mvn package -q
```

---

## Algorithm Complexity Summary

| Algorithm | Search | Insert | Space | Dimensions |
|---|---|---|---|---|
| Brute Force | O(N·d) | O(1) | O(N·d) | Any |
| KD-Tree | O(log N) avg | O(log N) | O(N·d) | Low (≤ 20D) |
| HNSW | O(log N) | O(log N) | O(N·M·layers·d) | Any |

**N** = number of stored vectors, **d** = dimensions, **M** = HNSW max connections (16 here).

---

## Java vs C++ Notes

This project is a port of an original C++ implementation. The Java equivalents are:

| Aspect | C++ | Java |
|---|---|---|
| HTTP Server | cpp-httplib (header-only) | Javalin 6 (Jetty-backed) |
| JSON | Hand-written serializer | Jackson (Javalin) + Gson (parsing) |
| Threading | `std::mutex` | `ReentrantLock` |
| HTTP Client | cpp-httplib | Java 11 `HttpClient` |
| Build | `g++` single file | Maven fat JAR |
| Run | `./db` | `java -jar vectordb.jar` |
| Random seed | `mt19937(42)` | `new Random(42)` |
