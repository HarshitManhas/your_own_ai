package com.vectordb;

import java.util.*;

/**
 * HNSW — Hierarchical Navigable Small World graph.
 *
 * The same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.
 * Builds a multilayer graph where each layer is progressively sparser.
 * Searches start at the top layer and zoom in, achieving O(log N) complexity.
 *
 * Direct port of the C++ HNSW class from the original project.
 */
public class HNSW {

    private static class Node {
        VectorItem item;
        int maxLayer;
        List<List<Integer>> neighbors; // neighbors[layer] = list of node IDs

        Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            this.neighbors = new ArrayList<>(maxLayer + 1);
            for (int i = 0; i <= maxLayer; i++) {
                this.neighbors.add(new ArrayList<>());
            }
        }
    }

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int M;        // Max connections per layer
    private final int M0;       // Max connections at layer 0 (2*M)
    private final int efBuild;  // ef_construction
    private final float mL;     // Level generation factor
    private int topLayer = -1;
    private int entryPoint = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = 1.0f / (float) Math.log(m);
    }

    public HNSW() {
        this(16, 200);
    }

    private int randomLevel() {
        return (int) Math.floor(-Math.log(rng.nextFloat()) * mL);
    }

    /**
     * Search within a single layer using beam search.
     * Returns candidates sorted by distance ascending.
     */
    private List<float[]> searchLayer(float[] query, int ep, int ef, int layer,
                                       DistanceMetrics.DistFn dist) {
        Set<Integer> visited = new HashSet<>();

        // Min-heap for candidates (closest first)
        PriorityQueue<float[]> candidates = new PriorityQueue<>(
                Comparator.comparingDouble(a -> a[0])
        );
        // Max-heap for found results (farthest first)
        PriorityQueue<float[]> found = new PriorityQueue<>(
                (a, b) -> Float.compare(b[0], a[0])
        );

        float d0 = dist.distance(query, graph.get(ep).item.emb);
        visited.add(ep);
        candidates.offer(new float[]{d0, ep});
        found.offer(new float[]{d0, ep});

        while (!candidates.isEmpty()) {
            float[] closest = candidates.poll();
            float cd = closest[0];
            int cid = (int) closest[1];

            if (found.size() >= ef && cd > found.peek()[0]) break;

            Node cNode = graph.get(cid);
            if (cNode == null || layer >= cNode.neighbors.size()) continue;

            for (int nid : cNode.neighbors.get(layer)) {
                if (visited.contains(nid) || !graph.containsKey(nid)) continue;
                visited.add(nid);

                float nd = dist.distance(query, graph.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek()[0]) {
                    candidates.offer(new float[]{nd, nid});
                    found.offer(new float[]{nd, nid});
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<float[]> result = new ArrayList<>();
        while (!found.isEmpty()) {
            result.add(found.poll());
        }
        result.sort(Comparator.comparingDouble(a -> a[0]));
        return result;
    }

    private List<Integer> selectNeighbors(List<float[]> candidates, int maxM) {
        List<Integer> result = new ArrayList<>();
        int limit = Math.min(candidates.size(), maxM);
        for (int i = 0; i < limit; i++) {
            result.add((int) candidates.get(i)[1]);
        }
        return result;
    }

    /**
     * Insert a vector item into the HNSW graph.
     */
    public void insert(VectorItem item, DistanceMetrics.DistFn dist) {
        int id = item.id;
        int level = randomLevel();
        Node newNode = new Node(item, level);
        graph.put(id, newNode);

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = level;
            return;
        }

        int ep = entryPoint;

        // Greedy descent from top layer to insertion level
        for (int lc = topLayer; lc > level; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.neighbors.size()) {
                List<float[]> w = searchLayer(item.emb, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = (int) w.get(0)[1];
            }
        }

        // Insert at each layer from min(topLayer, level) down to 0
        for (int lc = Math.min(topLayer, level); lc >= 0; lc--) {
            List<float[]> w = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> selected = selectNeighbors(w, maxM);
            newNode.neighbors.get(lc).addAll(selected);

            // Bidirectional connections
            for (int nid : selected) {
                Node neighbor = graph.get(nid);
                if (neighbor == null) continue;

                // Ensure neighbor has enough layer entries
                while (neighbor.neighbors.size() <= lc) {
                    neighbor.neighbors.add(new ArrayList<>());
                }

                List<Integer> conn = neighbor.neighbors.get(lc);
                conn.add(id);

                // Prune if too many connections
                if (conn.size() > maxM) {
                    List<float[]> distList = new ArrayList<>();
                    for (int c : conn) {
                        if (graph.containsKey(c)) {
                            distList.add(new float[]{
                                    dist.distance(neighbor.item.emb, graph.get(c).item.emb), c
                            });
                        }
                    }
                    distList.sort(Comparator.comparingDouble(a -> a[0]));
                    conn.clear();
                    for (int i = 0; i < maxM && i < distList.size(); i++) {
                        conn.add((int) distList.get(i)[1]);
                    }
                }
            }

            if (!w.isEmpty()) ep = (int) w.get(0)[1];
        }

        if (level > topLayer) {
            topLayer = level;
            entryPoint = id;
        }
    }

    /**
     * K-nearest neighbor search.
     */
    public List<float[]> knn(float[] query, int k, int ef, DistanceMetrics.DistFn dist) {
        if (entryPoint == -1) return Collections.emptyList();

        int ep = entryPoint;

        // Greedy descent from top to layer 1
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.neighbors.size()) {
                List<float[]> w = searchLayer(query, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = (int) w.get(0)[1];
            }
        }

        // Full beam search at layer 0
        List<float[]> w = searchLayer(query, ep, Math.max(ef, k), 0, dist);
        if (w.size() > k) {
            w = new ArrayList<>(w.subList(0, k));
        }
        return w;
    }

    /**
     * Remove a node from the graph.
     */
    public void remove(int id) {
        if (!graph.containsKey(id)) return;

        // Remove references to this node from all neighbors
        for (Node node : graph.values()) {
            for (List<Integer> layer : node.neighbors) {
                layer.remove(Integer.valueOf(id));
            }
        }

        // Update entry point if needed
        if (entryPoint == id) {
            entryPoint = -1;
            for (int nid : graph.keySet()) {
                if (nid != id) {
                    entryPoint = nid;
                    break;
                }
            }
        }

        graph.remove(id);
    }

    public int size() {
        return graph.size();
    }

    // ── Graph info for visualization ──────────────────────────────

    public static class GraphInfo {
        public int topLayer;
        public int nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();
        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();

        public static class NodeView {
            public int id;
            public String metadata;
            public String category;
            public int maxLayer;
        }

        public static class EdgeView {
            public int src, dst, layer;
        }
    }

    public GraphInfo getInfo() {
        GraphInfo info = new GraphInfo();
        info.topLayer = topLayer;
        info.nodeCount = graph.size();

        int maxL = Math.max(topLayer + 1, 1);
        info.nodesPerLayer = new ArrayList<>(Collections.nCopies(maxL, 0));
        info.edgesPerLayer = new ArrayList<>(Collections.nCopies(maxL, 0));

        for (var entry : graph.entrySet()) {
            int id = entry.getKey();
            Node node = entry.getValue();

            GraphInfo.NodeView nv = new GraphInfo.NodeView();
            nv.id = id;
            nv.metadata = node.item.metadata;
            nv.category = node.item.category;
            nv.maxLayer = node.maxLayer;
            info.nodes.add(nv);

            for (int lc = 0; lc <= node.maxLayer && lc < maxL; lc++) {
                info.nodesPerLayer.set(lc, info.nodesPerLayer.get(lc) + 1);
                if (lc < node.neighbors.size()) {
                    for (int nid : node.neighbors.get(lc)) {
                        if (id < nid) {
                            info.edgesPerLayer.set(lc, info.edgesPerLayer.get(lc) + 1);
                            GraphInfo.EdgeView ev = new GraphInfo.EdgeView();
                            ev.src = id;
                            ev.dst = nid;
                            ev.layer = lc;
                            info.edges.add(ev);
                        }
                    }
                }
            }
        }

        return info;
    }
}
