package com.video;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
@RequiredArgsConstructor
@Slf4j
public class SearchRerankService {

    private static final Pattern TOKEN = Pattern.compile("[a-zA-Z][a-zA-Z0-9_-]{2,}");
    private static final long LLM_RERANK_COOLDOWN_MS = 60_000L;
    private static final Set<String> STOP_WORDS = Set.of(
            "the", "and", "for", "with", "from", "that", "this", "there", "their",
            "video", "frame", "frames", "scene", "show", "shows", "showing", "find",
            "search", "what", "where", "when", "person", "people", "image");

    private final ObjectMapper objectMapper;

    @Value("${vlm.url:http://vlm:8001}")
    private String vlmUrl;

    @Value("${vlm.model:qwen2.5vl:7b}")
    private String vlmModel;

    @Value("${search.llm-rerank.enabled:true}")
    private boolean llmRerankEnabled;

    private volatile long llmRerankDisabledUntilMs = 0L;

    public List<Map<String, Object>> rerank(String query, List<Map<String, Object>> candidates, double minConfidence, int limit) {
        for (Map<String, Object> candidate : candidates) {
            applyFallbackScore(query, candidate);
        }
        tryApplyLlmRerank(query, candidates);

        List<Map<String, Object>> ranked = candidates.stream()
                .filter(candidate -> number(candidate.get("confidence")) >= minConfidence)
                .sorted(Comparator
                        .comparingDouble((Map<String, Object> candidate) -> number(candidate.get("confidence"))).reversed()
                        .thenComparing(Comparator.comparingDouble((Map<String, Object> candidate) -> number(candidate.get("initial_score"))).reversed()))
                .toList();

        return diversifyByVideo(ranked, limit);
    }

    private List<Map<String, Object>> diversifyByVideo(List<Map<String, Object>> ranked, int limit) {
        List<Map<String, Object>> diversified = new ArrayList<>();
        Set<String> seenVideos = new HashSet<>();
        for (Map<String, Object> candidate : ranked) {
            String videoKey = resultVideoKey(candidate);
            if (!seenVideos.add(videoKey)) continue;
            diversified.add(candidate);
            if (diversified.size() >= limit) break;
        }
        return diversified;
    }

    private String resultVideoKey(Map<String, Object> candidate) {
        String videoId = string(candidate.get("video_id"));
        if (!videoId.isBlank()) return "video:" + videoId;
        String frameId = string(candidate.get("frame_id"));
        if (!frameId.isBlank()) return "frame:" + frameId;
        return "candidate:" + System.identityHashCode(candidate);
    }

    private void applyFallbackScore(String query, Map<String, Object> candidate) {
        Set<String> queryTokens = tokens(query);
        Set<String> evidenceTokens = tokens(evidenceText(candidate));
        int overlapCount = 0;
        List<String> matched = new ArrayList<>();
        for (String token : queryTokens) {
            if (matchesEvidence(token, evidenceTokens)) {
                overlapCount++;
                matched.add(token);
            }
        }

        double overlap = queryTokens.isEmpty() ? 0 : Math.min(1.0, overlapCount / (double) Math.min(queryTokens.size(), 5));
        double initialScore = number(candidate.get("initial_score"));
        double analysisConfidence = number(candidate.get("analysis_confidence"));
        double actionScore = number(candidate.get("action_score"));
        double feedbackScore = number(candidate.get("feedback_score"));
        double feedbackSignal = number(candidate.get("feedback_signal"));
        if (feedbackSignal == 0.0 && feedbackScore > 0.0) {
            feedbackSignal = (feedbackScore - 0.5) * 2.0;
        }
        double confidence = clamp((0.34 * initialScore) + (0.25 * overlap) + (0.18 * actionScore) + (0.08 * analysisConfidence) + (0.15 * feedbackSignal));
        if (actionScore >= 0.85) {
            confidence = Math.max(confidence, 0.86);
        } else if (actionScore >= 0.65) {
            confidence = Math.max(confidence, 0.78);
        }
        if (overlap >= 0.67 && initialScore >= 0.30) {
            confidence = Math.max(confidence, 0.82);
        } else if (overlap >= 0.50 && initialScore >= 0.45) {
            confidence = Math.max(confidence, 0.80);
        }
        if (feedbackScore >= 0.75) {
            confidence = Math.max(confidence, 0.76 + (feedbackScore * 0.12));
        }
        if (feedbackSignal <= -0.55) {
            confidence = Math.min(confidence, 0.45);
        }

        candidate.put("confidence", round(confidence));
        candidate.put("rerank_source", "hybrid");
        if (!candidate.containsKey("relevance_reason") || String.valueOf(candidate.get("relevance_reason")).isBlank()) {
            candidate.put("relevance_reason", matched.isEmpty()
                    ? "Ranked by visual similarity, chunk context, and available metadata."
                    : "Matched retrieval evidence: " + String.join(", ", matched));
        }
    }

    private void tryApplyLlmRerank(String query, List<Map<String, Object>> candidates) {
        if (candidates.isEmpty()) return;
        if (!llmRerankEnabled) return;
        long now = System.currentTimeMillis();
        if (now < llmRerankDisabledUntilMs) {
            log.debug("Skipping LLM rerank during cooldown");
            return;
        }
        try {
            List<Map<String, Object>> reviewCandidates = candidates.stream()
                    .limit(10)
                    .map(this::llmCandidate)
                    .toList();
            String candidatesJson = objectMapper.writeValueAsString(reviewCandidates);
            String userPrompt = """
                    Rerank video frame search candidates for the user query.
                    Return strict JSON array only. Each item must contain frameId, relevance, reason.
                    relevance must be a number from 0 to 1. Use 0.80 or higher only when the frame clearly satisfies the query.
                    Prefer exact activity/object/scene matches over visual similarity alone.

                    Query: %s
                    Candidates: %s
                    """.formatted(query, candidatesJson);

            Map<String, Object> body = new LinkedHashMap<>();
            body.put("model", vlmModel);
            body.put("stream", false);
            body.put("temperature", 0);
            body.put("max_tokens", 700);
            body.put("messages", List.of(
                    Map.of("role", "system", "content", "You are a strict video search reranker. Output JSON only."),
                    Map.of("role", "user", "content", userPrompt)));

            HttpURLConnection conn = (HttpURLConnection) new URL(vlmUrl + "/v1/chat/completions").openConnection();
            conn.setConnectTimeout(800);
            conn.setReadTimeout(6000);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            try (OutputStream out = conn.getOutputStream()) {
                out.write(objectMapper.writeValueAsBytes(body));
            }

            StringBuilder raw = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) raw.append(line);
            }
            String content = objectMapper.readTree(raw.toString())
                    .path("choices").path(0).path("message").path("content").asText("");
            JsonNode ranking = objectMapper.readTree(extractJsonArray(content));
            if (!ranking.isArray()) return;

            Map<String, JsonNode> byFrameId = new LinkedHashMap<>();
            for (JsonNode item : ranking) {
                String frameId = item.path("frameId").asText("");
                if (!frameId.isBlank()) byFrameId.put(frameId, item);
            }
            int applied = 0;
            for (Map<String, Object> candidate : candidates) {
                String frameId = string(candidate.get("frame_id"));
                JsonNode item = byFrameId.get(frameId);
                if (item == null) continue;
                double relevance = clamp(item.path("relevance").asDouble(number(candidate.get("confidence"))));
                candidate.put("confidence", round(relevance));
                candidate.put("rerank_source", "llm");
                String reason = item.path("reason").asText("");
                if (!reason.isBlank()) candidate.put("relevance_reason", reason);
                applied++;
            }
            if (applied > 0) {
                log.info("Applied LLM rerank to {} search candidates", applied);
            }
        } catch (Exception e) {
            llmRerankDisabledUntilMs = System.currentTimeMillis() + LLM_RERANK_COOLDOWN_MS;
            log.warn("LLM rerank unavailable at {}; using hybrid fallback: {}", vlmUrl, e.getMessage());
        }
    }

    private Map<String, Object> llmCandidate(Map<String, Object> candidate) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("frameId", string(candidate.get("frame_id")));
        item.put("videoId", string(candidate.get("video_id")));
        item.put("timestampMs", Math.round(number(candidate.get("t_ms"))));
        item.put("initialScore", number(candidate.get("initial_score")));
        item.put("caption", string(candidate.get("caption")));
        item.put("mainActivity", string(candidate.get("main_activity")));
        item.put("tags", candidate.get("tags"));
        item.put("objects", candidate.get("objects"));
        item.put("scene", string(candidate.get("scene")));
        item.put("motion", string(candidate.get("motion")));
        item.put("actionTop", string(candidate.get("action_top")));
        item.put("actionLabels", candidate.get("action_labels"));
        item.put("actionScore", number(candidate.get("action_score")));
        item.put("feedbackScore", number(candidate.get("feedback_score")));
        return item;
    }

    private Set<String> tokens(String text) {
        Set<String> out = new HashSet<>();
        Matcher matcher = TOKEN.matcher((text == null ? "" : text).toLowerCase());
        while (matcher.find()) {
            String token = matcher.group();
            if (!STOP_WORDS.contains(token)) {
                out.add(token);
                String singular = singularize(token);
                if (!singular.equals(token) && !STOP_WORDS.contains(singular)) out.add(singular);
            }
        }
        return out;
    }

    private boolean matchesEvidence(String token, Set<String> evidenceTokens) {
        if (evidenceTokens.contains(token)) return true;
        if (token.length() < 4) return false;
        for (String evidence : evidenceTokens) {
            if (evidence.length() >= 4 && (evidence.startsWith(token) || token.startsWith(evidence))) {
                return true;
            }
        }
        return false;
    }

    private String singularize(String token) {
        if (token.length() > 4 && token.endsWith("ies")) {
            return token.substring(0, token.length() - 3) + "y";
        }
        if (token.length() > 3 && token.endsWith("s") && !token.endsWith("ss") && !token.endsWith("is")) {
            return token.substring(0, token.length() - 1);
        }
        return token;
    }

    private String evidenceText(Map<String, Object> candidate) {
        return String.join(" ",
                string(candidate.get("caption")),
                string(candidate.get("chunk_caption")),
                string(candidate.get("main_activity")),
                string(candidate.get("scene")),
                string(candidate.get("motion")),
                string(candidate.get("profile")),
                string(candidate.get("action_top")),
                valueText(candidate.get("tags")),
                valueText(candidate.get("objects")),
                valueText(candidate.get("action_labels")));
    }

    private String valueText(Object value) {
        if (value instanceof Iterable<?> iterable) {
            List<String> parts = new ArrayList<>();
            for (Object item : iterable) parts.add(String.valueOf(item));
            return String.join(" ", parts);
        }
        return value == null ? "" : String.valueOf(value);
    }

    private String extractJsonArray(String text) {
        text = text == null ? "" : text.trim();
        if (text.startsWith("```")) {
            text = text.replaceFirst("^```(?:json)?\\s*", "").replaceFirst("\\s*```$", "");
        }
        int start = text.indexOf('[');
        int end = text.lastIndexOf(']');
        if (start >= 0 && end > start) return text.substring(start, end + 1);
        return text;
    }

    private double number(Object value) {
        if (value instanceof Number number) return number.doubleValue();
        try {
            return value == null ? 0 : Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private String string(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private double clamp(double value) {
        return Math.max(0, Math.min(1, value));
    }

    private double round(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }
}
