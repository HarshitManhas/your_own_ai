package com.vectordb;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * OllamaClient wraps the local Ollama REST API.
 * Provides embedding (nomic-embed-text) and generation (llama3.2).
 * Port of the C++ OllamaClient class.
 */
public class OllamaClient {
    private final String baseUrl;
    private final HttpClient http;
    private final Gson gson = new Gson();

    public String embedModel = "nomic-embed-text";
    public String genModel   = "llama3.2";

    public OllamaClient(String host, int port) {
        this.baseUrl = "http://" + host + ":" + port;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();
    }

    public OllamaClient() {
        this("127.0.0.1", 11434);
    }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET().build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /** Generate embedding vector. Returns empty array on failure. */
    public float[] embed(String text) {
        try {
            JsonObject body = new JsonObject();
            body.addProperty("model", embedModel);
            body.addProperty("prompt", text);

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                    .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return new float[0];

            JsonObject json = gson.fromJson(resp.body(), JsonObject.class);
            JsonArray arr = json.getAsJsonArray("embedding");
            if (arr == null) return new float[0];

            float[] emb = new float[arr.size()];
            for (int i = 0; i < arr.size(); i++) emb[i] = arr.get(i).getAsFloat();
            return emb;
        } catch (Exception e) {
            return new float[0];
        }
    }

    /** Generate text from a prompt. Returns error string on failure. */
    public String generate(String prompt) {
        try {
            JsonObject body = new JsonObject();
            body.addProperty("model", genModel);
            body.addProperty("prompt", prompt);
            body.addProperty("stream", false);

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/generate"))
                    .timeout(Duration.ofSeconds(180))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                    .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";

            JsonObject json = gson.fromJson(resp.body(), JsonObject.class);
            JsonElement r = json.get("response");
            return r != null ? r.getAsString() : "";
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }
}
