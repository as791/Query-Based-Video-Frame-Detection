package com.video;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Map;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/search")
@RequiredArgsConstructor
@Slf4j
public class VectorSearchController {

    private static final Pattern TOKEN = Pattern.compile("[a-zA-Z][a-zA-Z0-9_-]{2,}");
    private static final long FRAME_CONTEXT_CACHE_TTL_MS = 120_000L;
    private static final Set<String> STOP_WORDS = Set.of(
            "the", "and", "for", "with", "from", "that", "this", "there", "their",
            "video", "frame", "frames", "scene", "show", "shows", "showing", "find",
            "search", "what", "where", "when", "person", "people", "image");

    private final QdrantRestClient qdrantClient;
    private final EmbedderRestClient embedderClient;
    private final AwsWrapperService awsWrapperService;
    private final UserRepository userRepository;
    private final SearchRerankService searchRerankService;
    private final Map<String, FrameContextCacheEntry> frameContextCache = new ConcurrentHashMap<>();

    @Value("${S3_BUCKET:stage-video-bucket}")
    private String bucket;

    @PostMapping
    public List<Map<String, Object>> search(
            @RequestBody Map<String, Object> body,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String query = (String) body.get("query");
        if (query == null || query.isBlank()) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "query required");

        String videoId = (String) body.get("videoId"); // optional scope
        double minConfidence = clamp(number(body.getOrDefault("minConfidence", 0.8)), 0.0, 0.99);
        int limit = Math.max(1, Math.min(30, (int) number(body.getOrDefault("limit", 12))));

        List<Double> queryVector = embedderClient.embedText(query);

        // Stage 1: broad chunk retrieval for semantic context.
        List<Map<String, Object>> chunks = qdrantClient.searchChunks(queryVector, user.getId(), videoId, 60);

        Map<String, Map<String, Object>> chunkPayloads = new LinkedHashMap<>();
        Map<String, Double> chunkRankScores = new LinkedHashMap<>();
        int chunkTotal = Math.max(1, chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            Map<String, Object> payload = payload(chunks.get(i));
            String chunkId = string(payload.get("chunk_id"));
            if (chunkId.isBlank()) continue;
            chunkPayloads.put(chunkId, payload);
            chunkRankScores.put(chunkId, 1.0 - (i / (double) chunkTotal));
        }
        List<String> chunkIds = chunks.stream()
                .map(c -> string(payload(c).get("chunk_id")))
                .filter(id -> !id.isBlank())
                .toList();

        // Stage 2: broad frame retrieval scoped to those chunks when possible.
        List<Map<String, Object>> frames = qdrantClient.searchFrames(queryVector, user.getId(), videoId, chunkIds, 80);
        List<Map<String, Object>> candidates = buildCandidates(query, frames, chunkPayloads, chunkRankScores);

        // Stage 3: LLM rerank when available, hybrid fallback otherwise.
        return searchRerankService.rerank(query, candidates, minConfidence, limit);
    }

    @GetMapping("/frames/context")
    public List<Map<String, Object>> frameContext(
            @RequestParam String videoId,
            @RequestParam(required = false) String chunkId,
            @RequestParam long tMs,
            @RequestParam(defaultValue = "12") int window,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String cacheKey = user.getId() + ":" + videoId + ":" + Math.round(tMs / 1000.0) + ":" + window;
        long now = System.currentTimeMillis();
        FrameContextCacheEntry cached = frameContextCache.get(cacheKey);
        if (cached != null && cached.expiresAtMs() > now) {
            return cached.frames();
        }

        int radius = Math.max(1, Math.min(12, window));
        int limit = Math.max(18, Math.min(72, radius * 4));
        long startFromMs = Math.max(0, tMs - (long) radius * 2500L);
        List<Map<String, Object>> payloads = new ArrayList<>(qdrantClient.scrollFramePayloads(user.getId(), videoId, null, startFromMs, limit));
        payloads.sort(Comparator.comparingLong(this::frameTime));

        int center = 0;
        long best = Long.MAX_VALUE;
        for (int i = 0; i < payloads.size(); i++) {
            long distance = Math.abs(frameTime(payloads.get(i)) - tMs);
            if (distance < best) {
                best = distance;
                center = i;
            }
        }

        int start = Math.max(0, center - radius);
        int end = Math.min(payloads.size(), center + radius + 1);
        final long bestDistance = best;
        List<Map<String, Object>> frames = payloads.subList(start, end).stream().map(payload -> {
            String s3Key = (String) payload.get("s3_frame_path");
            String presignedUrl = awsWrapperService.generatePresignedUrl(bucket, s3Key, 10).toString();
            payload.put("url", presignedUrl);
            payload.put("selected", Math.abs(frameTime(payload) - tMs) == bestDistance);
            return payload;
        }).toList();
        if (frameContextCache.size() > 1000) {
            frameContextCache.entrySet().removeIf(entry -> entry.getValue().expiresAtMs() <= now);
        }
        frameContextCache.put(cacheKey, new FrameContextCacheEntry(now + FRAME_CONTEXT_CACHE_TTL_MS, frames));
        return frames;
    }

    private record FrameContextCacheEntry(long expiresAtMs, List<Map<String, Object>> frames) {}

    private long frameTime(Map<String, Object> payload) {
        Object value = payload.get("t_ms");
        if (value instanceof Number number) return number.longValue();
        try {
            return value == null ? 0 : Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private List<Map<String, Object>> buildCandidates(
            String query,
            List<Map<String, Object>> frames,
            Map<String, Map<String, Object>> chunkPayloads,
            Map<String, Double> chunkRankScores) {
        List<Map<String, Object>> candidates = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int total = Math.max(1, frames.size());
        for (int i = 0; i < frames.size(); i++) {
            Map<String, Object> frame = frames.get(i);
            Map<String, Object> payload = new LinkedHashMap<>(payload(frame));
            String key = frameKey(payload);
            if (key.isBlank() || !seen.add(key)) continue;

            String chunkId = string(payload.get("chunk_id"));
            Map<String, Object> chunkPayload = chunkPayloads.getOrDefault(chunkId, Map.of());
            mergeChunkContext(payload, chunkPayload);

            double frameRankScore = 1.0 - (i / (double) total);
            double chunkRankScore = chunkRankScores.getOrDefault(chunkId, 0.0);
            double tagScore = lexicalScore(query, payload);
            double metadataScore = metadataScore(query, payload);
            double initialScore = clamp(
                    (0.35 * frameRankScore)
                            + (0.25 * chunkRankScore)
                            + (0.25 * tagScore)
                            + (0.15 * metadataScore),
                    0.0,
                    1.0);

            String s3Key = string(payload.get("s3_frame_path"));
            if (!s3Key.isBlank()) {
                payload.put("url", awsWrapperService.generatePresignedUrl(bucket, s3Key, 10).toString());
            }
            payload.put("vector_score", round(number(frame.get("score"))));
            payload.put("frame_rank_score", round(frameRankScore));
            payload.put("chunk_rank_score", round(chunkRankScore));
            payload.put("tag_score", round(tagScore));
            payload.put("metadata_score", round(metadataScore));
            payload.put("initial_score", round(initialScore));
            candidates.add(payload);
        }
        return candidates;
    }

    private void mergeChunkContext(Map<String, Object> framePayload, Map<String, Object> chunkPayload) {
        if (chunkPayload.isEmpty()) return;
        putIfMissing(framePayload, "caption", chunkPayload.get("caption"));
        framePayload.put("chunk_caption", string(chunkPayload.get("caption")));
        putIfMissing(framePayload, "main_activity", chunkPayload.get("main_activity"));
        putIfMissing(framePayload, "scene", chunkPayload.get("scene"));
        putIfMissing(framePayload, "motion", chunkPayload.get("motion"));
        putIfMissing(framePayload, "source_file", chunkPayload.get("source_file"));
        putIfMissing(framePayload, "analysis_confidence", chunkPayload.get("analysis_confidence"));
        framePayload.put("tags", mergeLists(framePayload.get("tags"), chunkPayload.get("tags")));
        framePayload.put("objects", mergeLists(framePayload.get("objects"), chunkPayload.get("objects")));
    }

    private void putIfMissing(Map<String, Object> target, String key, Object value) {
        if (value == null) return;
        Object existing = target.get(key);
        if (existing == null || String.valueOf(existing).isBlank()) target.put(key, value);
    }

    private List<String> mergeLists(Object first, Object second) {
        List<String> out = new ArrayList<>();
        addList(out, first);
        addList(out, second);
        return out;
    }

    private void addList(List<String> out, Object value) {
        if (value instanceof Iterable<?> iterable) {
            for (Object item : iterable) addToken(out, item);
        } else if (value != null) {
            for (String item : String.valueOf(value).split("[,;]")) addToken(out, item);
        }
    }

    private void addToken(List<String> out, Object value) {
        String item = string(value).trim().toLowerCase();
        if (!item.isBlank() && !out.contains(item) && out.size() < 16) out.add(item);
    }

    private double lexicalScore(String query, Map<String, Object> payload) {
        Set<String> queryTokens = tokens(query);
        if (queryTokens.isEmpty()) return 0;
        Set<String> evidenceTokens = tokens(String.join(" ",
                string(payload.get("caption")),
                string(payload.get("chunk_caption")),
                string(payload.get("main_activity")),
                string(payload.get("scene")),
                string(payload.get("motion")),
                valueText(payload.get("tags")),
                valueText(payload.get("objects"))));
        int overlap = 0;
        for (String token : queryTokens) {
            if (matchesEvidence(token, evidenceTokens)) overlap++;
        }
        return clamp(overlap / (double) Math.min(queryTokens.size(), 5), 0.0, 1.0);
    }

    private double metadataScore(String query, Map<String, Object> payload) {
        Set<String> queryTokens = tokens(query);
        if (queryTokens.isEmpty()) return 0;
        Set<String> metadataTokens = tokens(String.join(" ",
                string(payload.get("source_file")),
                string(payload.get("profile")),
                string(payload.get("width")),
                string(payload.get("height")),
                string(payload.get("fps"))));
        int overlap = 0;
        for (String token : queryTokens) {
            if (matchesEvidence(token, metadataTokens)) overlap++;
        }
        return clamp(overlap / (double) Math.min(queryTokens.size(), 5), 0.0, 1.0);
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

    private String valueText(Object value) {
        if (value instanceof Iterable<?> iterable) {
            List<String> parts = new ArrayList<>();
            for (Object item : iterable) parts.add(String.valueOf(item));
            return String.join(" ", parts);
        }
        return value == null ? "" : String.valueOf(value);
    }

    private String frameKey(Map<String, Object> payload) {
        String frameId = string(payload.get("frame_id"));
        if (!frameId.isBlank()) return frameId;
        return string(payload.get("video_id")) + ":" + string(payload.get("chunk_id")) + ":" + string(payload.get("t_ms"));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> payload(Map<String, Object> point) {
        Object payload = point.get("payload");
        if (payload instanceof Map<?, ?> map) return (Map<String, Object>) map;
        return Map.of();
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

    private double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }

    private double round(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }
}
