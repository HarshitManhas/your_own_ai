package com.vectordb;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.javalin.Javalin;
import io.javalin.http.Context;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Main — HTTP server entry point.
 * Wires together all components and exposes the REST API.
 * Port of the C++ main() function with all its routes.
 */
public class Main {

    static final int DIMS = 16;

    public static void main(String[] args) {
        VectorDB   db     = new VectorDB(DIMS);
        DocumentDB docDB  = new DocumentDB();
        OllamaClient ollama = new OllamaClient();
        Gson gson = new Gson();

        DemoData.load(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        Javalin app = Javalin.create(config -> {
            config.bundledPlugins.enableCors(cors -> cors.addRule(it -> it.anyHost()));
            config.staticFiles.add("/public");
        });

        // ── DEMO VECTOR ENDPOINTS ────────────────────────────────────────

        // GET /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw
        app.get("/search", ctx -> {
            String vParam = ctx.queryParam("v");
            float[] q = parseVec(vParam);
            if (q == null || q.length != DIMS) {
                ctx.status(400).json(Map.of("error", "need " + DIMS + "D vector"));
                return;
            }
            int k = paramInt(ctx, "k", 5);
            String metric = paramStr(ctx, "metric", "cosine");
            String algo   = paramStr(ctx, "algo",   "hnsw");

            var out = db.search(q, k, metric, algo);

            var results = new ArrayList<>();
            for (var h : out.hits) {
                var m = new LinkedHashMap<String, Object>();
                m.put("id",        h.id);
                m.put("metadata",  h.meta);
                m.put("category",  h.category);
                m.put("distance",  h.distance);
                m.put("embedding", floatArrToList(h.emb));
                results.add(m);
            }
            var resp = new LinkedHashMap<String, Object>();
            resp.put("results",   results);
            resp.put("latencyUs", out.microseconds);
            resp.put("algo",      out.algo);
            resp.put("metric",    out.metric);
            ctx.json(resp);
        });

        // POST /insert
        app.post("/insert", ctx -> {
            JsonObject body = gson.fromJson(ctx.body(), JsonObject.class);
            String meta = jsonStr(body, "metadata");
            String cat  = jsonStr(body, "category");
            float[] emb = jsonFloatArr(body, "embedding");
            if (meta == null || emb == null || emb.length != DIMS) {
                ctx.status(400).json(Map.of("error", "invalid body"));
                return;
            }
            int id = db.insert(meta, cat != null ? cat : "default", emb,
                               DistanceMetrics.getDistFn("cosine"));
            ctx.json(Map.of("id", id));
        });

        // DELETE /delete/:id
        app.delete("/delete/{id}", ctx -> {
            int id = Integer.parseInt(ctx.pathParam("id"));
            boolean ok = db.remove(id);
            ctx.json(Map.of("ok", ok));
        });

        // GET /items
        app.get("/items", ctx -> {
            var items = db.all();
            var list = new ArrayList<>();
            for (var v : items) {
                var m = new LinkedHashMap<String, Object>();
                m.put("id",        v.id);
                m.put("metadata",  v.metadata);
                m.put("category",  v.category);
                m.put("embedding", floatArrToList(v.emb));
                list.add(m);
            }
            ctx.json(list);
        });

        // GET /benchmark?v=...&k=5&metric=cosine
        app.get("/benchmark", ctx -> {
            float[] q = parseVec(ctx.queryParam("v"));
            if (q == null || q.length != DIMS) {
                ctx.status(400).json(Map.of("error", "need " + DIMS + "D vector"));
                return;
            }
            int k = paramInt(ctx, "k", 5);
            String metric = paramStr(ctx, "metric", "cosine");
            var b = db.benchmark(q, k, metric);
            ctx.json(Map.of(
                "bruteforceUs", b.bfUs,
                "kdtreeUs",     b.kdUs,
                "hnswUs",       b.hnswUs,
                "itemCount",    b.count
            ));
        });

        // GET /hnsw-info
        app.get("/hnsw-info", ctx -> {
            var gi = db.hnswInfo();
            var resp = new LinkedHashMap<String, Object>();
            resp.put("topLayer",      gi.topLayer);
            resp.put("nodeCount",     gi.nodeCount);
            resp.put("nodesPerLayer", gi.nodesPerLayer);
            resp.put("edgesPerLayer", gi.edgesPerLayer);

            var nodes = gi.nodes.stream().map(n -> {
                var m = new LinkedHashMap<String, Object>();
                m.put("id",       n.id);
                m.put("metadata", n.metadata);
                m.put("category", n.category);
                m.put("maxLyr",   n.maxLayer);
                return m;
            }).collect(Collectors.toList());
            resp.put("nodes", nodes);

            var edges = gi.edges.stream().map(e -> {
                var m = new LinkedHashMap<String, Object>();
                m.put("src", e.src);
                m.put("dst", e.dst);
                m.put("lyr", e.layer);
                return m;
            }).collect(Collectors.toList());
            resp.put("edges", edges);
            ctx.json(resp);
        });

        // GET /stats
        app.get("/stats", ctx -> ctx.json(Map.of(
            "count",      db.size(),
            "dims",       DIMS,
            "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
            "metrics",    List.of("euclidean", "cosine", "manhattan")
        )));

        // ── DOCUMENT + RAG ENDPOINTS ─────────────────────────────────────

        // POST /doc/insert  {"title":"...","text":"..."}
        app.post("/doc/insert", ctx -> {
            JsonObject body = gson.fromJson(ctx.body(), JsonObject.class);
            String title = jsonStr(body, "title");
            String text  = jsonStr(body, "text");
            if (title == null || title.isBlank() || text == null || text.isBlank()) {
                ctx.status(400).json(Map.of("error", "need title and text"));
                return;
            }
            List<String> chunks = TextChunker.chunkText(text);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    ctx.status(503).json(Map.of("error",
                        "Ollama unavailable. Install from https://ollama.com then run: " +
                        "ollama pull nomic-embed-text && ollama pull llama3.2"));
                    return;
                }
                String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }
            ctx.json(Map.of(
                "ids",    ids,
                "chunks", chunks.size(),
                "dims",   docDB.getDims()
            ));
        });

        // DELETE /doc/delete/:id
        app.delete("/doc/delete/{id}", ctx -> {
            int id = Integer.parseInt(ctx.pathParam("id"));
            boolean ok = docDB.remove(id);
            ctx.json(Map.of("ok", ok));
        });

        // GET /doc/list
        app.get("/doc/list", ctx -> {
            var docs = docDB.all();
            var list = docs.stream().map(d -> {
                String preview = d.text.length() > 120
                    ? d.text.substring(0, 120) + "…"
                    : d.text;
                int words = d.text.split("\\s+").length;
                var m = new LinkedHashMap<String, Object>();
                m.put("id",      d.id);
                m.put("title",   d.title);
                m.put("preview", preview);
                m.put("words",   words);
                return m;
            }).collect(Collectors.toList());
            ctx.json(list);
        });

        // POST /doc/search  {"question":"...","k":3}
        app.post("/doc/search", ctx -> {
            JsonObject body = gson.fromJson(ctx.body(), JsonObject.class);
            String question = jsonStr(body, "question");
            int k = body.has("k") ? body.get("k").getAsInt() : 3;
            if (question == null || question.isBlank()) {
                ctx.status(400).json(Map.of("error", "need question"));
                return;
            }
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                ctx.status(503).json(Map.of("error", "Ollama unavailable"));
                return;
            }
            var hits = docDB.search(qEmb, k);
            var contexts = hits.stream().map(e -> {
                var m = new LinkedHashMap<String, Object>();
                m.put("id",       e.getValue().id);
                m.put("title",    e.getValue().title);
                m.put("distance", e.getKey());
                return m;
            }).collect(Collectors.toList());
            ctx.json(Map.of("contexts", contexts));
        });

        // POST /doc/ask  {"question":"...","k":3}   Full RAG pipeline
        app.post("/doc/ask", ctx -> {
            JsonObject body = gson.fromJson(ctx.body(), JsonObject.class);
            String question = jsonStr(body, "question");
            int k = body.has("k") ? body.get("k").getAsInt() : 3;
            if (question == null || question.isBlank()) {
                ctx.status(400).json(Map.of("error", "need question"));
                return;
            }

            // Step 1: embed question
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                ctx.status(503).json(Map.of("error", "Ollama unavailable"));
                return;
            }

            // Step 2: retrieve top-k chunks
            var hits = docDB.search(qEmb, k);

            // Step 3: build prompt
            StringBuilder ctxBuilder = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                var doc = hits.get(i).getValue();
                ctxBuilder.append("[").append(i + 1).append("] ")
                          .append(doc.title).append(":\n")
                          .append(doc.text).append("\n\n");
            }
            String prompt =
                "You are a helpful assistant. Answer the user's question directly. " +
                "Use the provided context if it contains relevant information. " +
                "If it doesn't, just use your own general knowledge. " +
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. " +
                "Just answer the question naturally.\n\n" +
                "Context:\n" + ctxBuilder +
                "Question: " + question + "\n\nAnswer:";

            // Step 4: generate answer
            String answer = ollama.generate(prompt);

            // Step 5: return
            var contexts = hits.stream().map(e -> {
                var m = new LinkedHashMap<String, Object>();
                m.put("id",       e.getValue().id);
                m.put("title",    e.getValue().title);
                m.put("text",     e.getValue().text);
                m.put("distance", e.getKey());
                return m;
            }).collect(Collectors.toList());

            var resp = new LinkedHashMap<String, Object>();
            resp.put("answer",   answer);
            resp.put("model",    ollama.genModel);
            resp.put("contexts", contexts);
            resp.put("docCount", docDB.size());
            ctx.json(resp);
        });

        // GET /status
        app.get("/status", ctx -> {
            boolean up = ollama.isAvailable();
            ctx.json(Map.of(
                "ollamaAvailable", up,
                "embedModel",      ollama.embedModel,
                "genModel",        ollama.genModel,
                "docCount",        docDB.size(),
                "docDims",         docDB.getDims(),
                "demoDims",        DIMS,
                "demoCount",       db.size()
            ));
        });

        // GET / — serve index.html from classpath /public/index.html
        app.get("/", ctx -> {
            try (InputStream is = Main.class.getResourceAsStream("/public/index.html")) {
                if (is == null) { ctx.status(404).result("index.html not found"); return; }
                ctx.html(new String(is.readAllBytes(), StandardCharsets.UTF_8));
            }
        });

        app.start(8080);
    }

    // ── Helpers ────────────────────────────────────────────────────────

    private static float[] parseVec(String s) {
        if (s == null || s.isBlank()) return null;
        String[] parts = s.split(",");
        float[] v = new float[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i].trim());
        } catch (NumberFormatException e) { return null; }
        return v;
    }

    private static int paramInt(Context ctx, String name, int def) {
        String v = ctx.queryParam(name);
        if (v == null) return def;
        try { return Integer.parseInt(v); } catch (NumberFormatException e) { return def; }
    }

    private static String paramStr(Context ctx, String name, String def) {
        String v = ctx.queryParam(name);
        return (v == null || v.isBlank()) ? def : v;
    }

    private static String jsonStr(JsonObject obj, String key) {
        if (obj == null || !obj.has(key) || obj.get(key).isJsonNull()) return null;
        return obj.get(key).getAsString();
    }

    private static float[] jsonFloatArr(JsonObject obj, String key) {
        if (obj == null || !obj.has(key)) return null;
        var arr = obj.getAsJsonArray(key);
        if (arr == null) return null;
        float[] v = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) v[i] = arr.get(i).getAsFloat();
        return v;
    }

    private static List<Float> floatArrToList(float[] arr) {
        List<Float> list = new ArrayList<>(arr.length);
        for (float f : arr) list.add(f);
        return list;
    }
}
