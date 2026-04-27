package com.vectordb;

import java.util.*;

/**
 * Brute Force K-NN search — O(N·d) complexity.
 * Exact baseline, computes distance to every stored vector.
 */
public class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) {
        items.add(v);
    }

    /**
     * K-nearest neighbors via exhaustive scan.
     * Returns list of (distance, id) pairs sorted by distance ascending.
     */
    public List<float[]> knn(float[] query, int k, DistanceMetrics.DistFn dist) {
        List<float[]> results = new ArrayList<>(items.size());
        for (VectorItem v : items) {
            results.add(new float[]{dist.distance(query, v.emb), v.id});
        }
        results.sort(Comparator.comparingDouble(a -> a[0]));
        if (results.size() > k) {
            return new ArrayList<>(results.subList(0, k));
        }
        return results;
    }

    public void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    public List<VectorItem> getItems() {
        return items;
    }
}
