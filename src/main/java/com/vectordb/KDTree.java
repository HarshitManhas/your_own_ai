package com.vectordb;

import java.util.*;

/**
 * KD-Tree for K-NN search — O(log N) average for low dimensions.
 * Binary space partitioning that cycles through dimensions.
 * Degrades to ~O(N) at high dimensions (curse of dimensionality).
 */
public class KDTree {
    private static class KDNode {
        VectorItem item;
        KDNode left;
        KDNode right;

        KDNode(VectorItem item) {
            this.item = item;
        }
    }

    private KDNode root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    private KDNode insert(KDNode node, VectorItem v, int depth) {
        if (node == null) return new KDNode(v);
        int axis = depth % dims;
        if (v.emb[axis] < node.item.emb[axis]) {
            node.left = insert(node.left, v, depth + 1);
        } else {
            node.right = insert(node.right, v, depth + 1);
        }
        return node;
    }

    public void insert(VectorItem v) {
        root = insert(root, v, 0);
    }

    private void knnSearch(KDNode node, float[] query, int k, int depth,
                           DistanceMetrics.DistFn dist,
                           PriorityQueue<float[]> heap) {
        if (node == null) return;

        float dn = dist.distance(query, node.item.emb);
        if (heap.size() < k || dn < heap.peek()[0]) {
            heap.offer(new float[]{dn, node.item.id});
            if (heap.size() > k) heap.poll();
        }

        int axis = depth % dims;
        float diff = query[axis] - node.item.emb[axis];
        KDNode closer  = diff < 0 ? node.left  : node.right;
        KDNode farther = diff < 0 ? node.right : node.left;

        knnSearch(closer, query, k, depth + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek()[0]) {
            knnSearch(farther, query, k, depth + 1, dist, heap);
        }
    }

    /**
     * K-nearest neighbors using the KD-Tree.
     * Returns list of (distance, id) sorted ascending.
     */
    public List<float[]> knn(float[] query, int k, DistanceMetrics.DistFn dist) {
        // Max-heap by distance (so we can evict the farthest candidate)
        PriorityQueue<float[]> heap = new PriorityQueue<>(
                (a, b) -> Float.compare(b[0], a[0])
        );
        knnSearch(root, query, k, 0, dist, heap);

        List<float[]> results = new ArrayList<>();
        while (!heap.isEmpty()) {
            results.add(heap.poll());
        }
        results.sort(Comparator.comparingDouble(a -> a[0]));
        return results;
    }

    /**
     * Rebuild the tree from scratch (used after deletions).
     */
    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) {
            insert(v);
        }
    }
}
