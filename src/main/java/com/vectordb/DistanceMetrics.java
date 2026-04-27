package com.vectordb;

/**
 * Distance metrics: Cosine, Euclidean, Manhattan.
 * Equivalent to the C++ distance functions and getDistFn().
 */
public final class DistanceMetrics {

    private DistanceMetrics() {}

    @FunctionalInterface
    public interface DistFn {
        float distance(float[] a, float[] b);
    }

    public static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (float)(Math.sqrt(na) * Math.sqrt(nb));
    }

    public static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    public static DistFn getDistFn(String metric) {
        return switch (metric) {
            case "cosine"    -> DistanceMetrics::cosine;
            case "manhattan" -> DistanceMetrics::manhattan;
            default          -> DistanceMetrics::euclidean;
        };
    }
}
