package com.vectordb;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * DocumentDB — HNSW index over real Ollama embeddings (768D).
 * Handles the RAG document storage with HNSW + brute force fallback.
 */
public class DocumentDB {

    public static class DocItem {
        public final int id;
        public final String title;
        public final String text;
        public final float[] emb;

        public DocItem(int id, String title, String text, float[] emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw;
    private final BruteForce bf = new BruteForce();
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    private int dims = 0;

    public DocumentDB() {
        this.hnsw = new HNSW(16, 200);
    }

    /**
     * Insert a pre-embedded document chunk.
     */
    public int insert(String title, String text, float[] emb) {
        lock.lock();
        try {
            if (dims == 0) dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);

            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, DistanceMetrics::cosine);
            bf.insert(vi);

            return item.id;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Semantic search — returns top-k most similar document chunks
     * with their distances, filtered by max_dist threshold.
     */
    public List<Map.Entry<Float, DocItem>> search(float[] query, int k, float maxDist) {
        lock.lock();
        try {
            if (store.isEmpty()) return Collections.emptyList();

            List<float[]> raw;
            if (store.size() < 10) {
                raw = bf.knn(query, k, DistanceMetrics::cosine);
            } else {
                raw = hnsw.knn(query, k, 50, DistanceMetrics::cosine);
            }

            List<Map.Entry<Float, DocItem>> results = new ArrayList<>();
            for (float[] pair : raw) {
                int id = (int) pair[1];
                float dist = pair[0];
                DocItem doc = store.get(id);
                if (doc != null && dist <= maxDist) {
                    results.add(Map.entry(dist, doc));
                }
            }
            return results;
        } finally {
            lock.unlock();
        }
    }

    public List<Map.Entry<Float, DocItem>> search(float[] query, int k) {
        return search(query, k, 0.7f);
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally {
            lock.unlock();
        }
    }

    public List<DocItem> all() {
        lock.lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        try {
            return store.size();
        } finally {
            lock.unlock();
        }
    }

    public int getDims() {
        return dims;
    }
}
