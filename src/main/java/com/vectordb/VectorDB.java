package com.vectordb;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * VectorDB — Unified interface over BruteForce, KD-Tree, and HNSW.
 * Manages the 16D demo vector index.
 * Thread-safe via ReentrantLock (equivalent to the C++ std::mutex).
 */
public class VectorDB {
    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    public final int dims;

    public VectorDB(int dims) {
        this.dims = dims;
        this.kdt = new KDTree(dims);
        this.hnsw = new HNSW(16, 200);
    }

    public int insert(String meta, String category, float[] emb, DistanceMetrics.DistFn dist) {
        lock.lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, category, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally {
            lock.unlock();
        }
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            // Rebuild KD-Tree (same as C++ version)
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally {
            lock.unlock();
        }
    }

    // ── Search result types ──────────────────────────────────────

    public static class Hit {
        public int id;
        public String meta, category;
        public float[] emb;
        public float distance;

        public Hit(int id, String meta, String category, float[] emb, float distance) {
            this.id = id;
            this.meta = meta;
            this.category = category;
            this.emb = emb;
            this.distance = distance;
        }
    }

    public static class SearchResult {
        public List<Hit> hits;
        public long microseconds;
        public String algo, metric;

        public SearchResult(List<Hit> hits, long us, String algo, String metric) {
            this.hits = hits;
            this.microseconds = us;
            this.algo = algo;
            this.metric = metric;
        }
    }

    public SearchResult search(float[] query, int k, String metric, String algo) {
        lock.lock();
        try {
            var distFn = DistanceMetrics.getDistFn(metric);
            long start = System.nanoTime();

            List<float[]> raw = switch (algo) {
                case "bruteforce" -> bf.knn(query, k, distFn);
                case "kdtree"     -> kdt.knn(query, k, distFn);
                default           -> hnsw.knn(query, k, 50, distFn);
            };

            long us = (System.nanoTime() - start) / 1000;

            List<Hit> hits = new ArrayList<>();
            for (float[] pair : raw) {
                int id = (int) pair[1];
                float dist = pair[0];
                VectorItem v = store.get(id);
                if (v != null) {
                    hits.add(new Hit(id, v.metadata, v.category, v.emb, dist));
                }
            }

            return new SearchResult(hits, us, algo, metric);
        } finally {
            lock.unlock();
        }
    }

    // ── Benchmark ──────────────────────────────────────────────

    public static class BenchResult {
        public long bfUs, kdUs, hnswUs;
        public int count;

        public BenchResult(long bfUs, long kdUs, long hnswUs, int count) {
            this.bfUs = bfUs;
            this.kdUs = kdUs;
            this.hnswUs = hnswUs;
            this.count = count;
        }
    }

    public BenchResult benchmark(float[] query, int k, String metric) {
        lock.lock();
        try {
            var distFn = DistanceMetrics.getDistFn(metric);

            long t0 = System.nanoTime();
            bf.knn(query, k, distFn);
            long bfUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            kdt.knn(query, k, distFn);
            long kdUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            hnsw.knn(query, k, 50, distFn);
            long hnswUs = (System.nanoTime() - t0) / 1000;

            return new BenchResult(bfUs, kdUs, hnswUs, store.size());
        } finally {
            lock.unlock();
        }
    }

    public List<VectorItem> all() {
        lock.lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            lock.unlock();
        }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.lock();
        try {
            return hnsw.getInfo();
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
}
