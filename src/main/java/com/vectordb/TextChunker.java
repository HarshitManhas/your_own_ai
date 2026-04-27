package com.vectordb;

import java.util.ArrayList;
import java.util.List;

/**
 * Text chunker — splits long documents into overlapping chunks.
 * Matches the C++ chunkText() function.
 */
public final class TextChunker {

    private TextChunker() {}

    /**
     * Split text into overlapping word-based chunks.
     *
     * @param text         The full document text
     * @param chunkWords   Number of words per chunk (default: 250)
     * @param overlapWords Number of overlapping words between chunks (default: 30)
     * @return List of text chunks
     */
    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0 || (words.length == 1 && words[0].isEmpty())) {
            return List.of();
        }
        if (words.length <= chunkWords) {
            return List.of(text);
        }

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;

        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(' ');
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }

        return chunks;
    }

    public static List<String> chunkText(String text) {
        return chunkText(text, 250, 30);
    }
}
