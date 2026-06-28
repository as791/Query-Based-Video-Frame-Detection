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
    private final ActionTaxonomyService actionTaxonomyService;
    private final FewShotLearningService fewShotLearningService;
    private final DomainModelService domainModelService;
    private final Map<String, FrameContextCacheEntry> frameContextCache = new ConcurrentHashMap<>();

    @Value("${S3_BUCKET:stage-video-bucket}")
    private String bucket;

    @Value("${search.candidates.chunk-limit:100}")
    private int chunkCandidateLimit;

    @Value("${search.candidates.frame-limit:160}")
    private int frameCandidateLimit;

    @Value("${search.candidates.action-limit:160}")
    private int actionCandidateLimit;

    @Value("${search.few-shot.min-score:0.56}")
    private double fewShotMinScore;

    @Value("${search.few-shot.top-score:0.72}")
    private double fewShotTopScore;

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
        String benchmarkRunId = (String) body.get("benchmarkRunId"); // optional benchmark scope
        String domainId = domainModelService.normalizeDomainId(String.valueOf(body.getOrDefault("domainId", "")));
        boolean domainScoped = body.containsKey("domainId") && !String.valueOf(body.getOrDefault("domainId", "")).isBlank();
        double minConfidence = clamp(number(body.getOrDefault("minConfidence", 0.8)), 0.0, 0.99);
        int limit = Math.max(1, Math.min(30, (int) number(body.getOrDefault("limit", 12))));
        DomainModelService.DomainState domainState = domainScoped ? domainModelService.state(user.getId(), domainId) : null;
        String fewShotAction = domainScoped
                ? domainModelService.detectAction(user.getId(), domainId, query)
                : fewShotLearningService.detectAction(user.getId(), query);
        String queryAction = fewShotAction;
        if (queryAction.isBlank()) {
            queryAction = actionTaxonomyService.detectAction(query);
        }

        List<Double> queryVector = embedderClient.embedText(query);

        // Stage 1: broad chunk retrieval for semantic context.
        List<Map<String, Object>> chunks = new ArrayList<>(qdrantClient.searchChunks(
                queryVector, user.getId(), videoId, benchmarkRunId, domainScoped ? domainId : "", Math.max(20, chunkCandidateLimit)));
        if (!queryAction.isBlank()) {
            chunks.addAll(qdrantClient.scrollChunksByAction(
                    user.getId(), videoId, benchmarkRunId, domainScoped ? domainId : "", queryAction, Math.max(20, actionCandidateLimit / 2)));
        }

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
                .distinct()
                .toList();

        // Stage 2: broad frame retrieval. Search both inside semantically matching chunks and across
        // the full scope so weak chunk retrieval does not prevent good frames from entering rerank.
        List<Map<String, Object>> frames = new ArrayList<>(qdrantClient.searchFrames(
                queryVector, user.getId(), videoId, benchmarkRunId, domainScoped ? domainId : "", chunkIds, Math.max(30, frameCandidateLimit)));
        if (!chunkIds.isEmpty()) {
            frames.addAll(qdrantClient.searchFrames(
                    queryVector, user.getId(), videoId, benchmarkRunId, domainScoped ? domainId : "", null, Math.max(30, frameCandidateLimit / 2)));
        }
        if (!queryAction.isBlank()) {
            frames.addAll(qdrantClient.scrollFramesByAction(
                    user.getId(), videoId, benchmarkRunId, domainScoped ? domainId : "", queryAction, Math.max(30, actionCandidateLimit)));
        }
        if (!fewShotAction.isBlank()) {
            frames.addAll(searchFewShotPrototypeFrames(
                    user.getId(), videoId, benchmarkRunId, domainScoped ? domainId : "", fewShotAction, Math.max(30, actionCandidateLimit)));
        }
        if (domainState != null) {
            frames.addAll(searchFeedbackPrototypeFrames(
                    user.getId(), videoId, benchmarkRunId, domainId, domainState, query, Math.max(20, actionCandidateLimit / 2)));
        }
        List<Map<String, Object>> candidates = buildCandidates(query, queryAction, domainState, mergeFrameCandidates(frames), chunkPayloads, chunkRankScores);

        // Stage 3: LLM rerank when available, hybrid fallback otherwise.
        return searchRerankService.rerank(query, candidates, minConfidence, limit);
    }

    @PostMapping("/feedback")
    public DomainModelService.DomainState feedback(
            @RequestBody DomainModelService.FeedbackInput input,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        return applyFeedback(user, input);
    }

    @PostMapping("/feedback/undo")
    public DomainModelService.DomainState undoFeedback(
            @RequestBody DomainModelService.FeedbackInput input,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        if (input.domainId() == null || input.domainId().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "domainId required");
        }
        Map<String, Object> point = qdrantClient.findFrameWithVector(user.getId(), input.videoId(), input.frameId(), input.chunkId());
        List<Double> vector = vector(point.get("vector"));
        if (vector.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "frame vector not found");
        }
        return domainModelService.undoFeedback(user.getId(), input, vector);
    }

    @PostMapping("/feedback/batch")
    public Map<String, Object> feedbackBatch(
            @RequestBody List<DomainModelService.FeedbackInput> inputs,
            @AuthenticationPrincipal OAuth2User oAuth2User) {
        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        int count = 0;
        DomainModelService.DomainState latest = null;
        for (DomainModelService.FeedbackInput input : inputs == null ? List.<DomainModelService.FeedbackInput>of() : inputs) {
            latest = applyFeedback(user, input);
            count++;
        }
        return Map.of(
                "count", count,
                "domainId", latest == null ? "" : latest.domainId(),
                "modelVersion", latest == null ? "" : latest.modelVersion());
    }

    private DomainModelService.DomainState applyFeedback(User user, DomainModelService.FeedbackInput input) {
        if (input.domainId() == null || input.domainId().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "domainId required");
        }
        Map<String, Object> point = qdrantClient.findFrameWithVector(user.getId(), input.videoId(), input.frameId(), input.chunkId());
        List<Double> vector = vector(point.get("vector"));
        if (vector.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "frame vector not found");
        }
        return domainModelService.recordFeedback(user.getId(), input, vector);
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

    private List<Map<String, Object>> searchFewShotPrototypeFrames(
            String userId,
            String videoId,
            String benchmarkRunId,
            String domainId,
            String label,
            int limit) {
        List<Map<String, Object>> prototypes = qdrantClient.scrollFewShotExampleFrames(userId, domainId, label, 12);
        if (prototypes.isEmpty()) return List.of();

        List<Map<String, Object>> out = new ArrayList<>();
        List<Double> centroid = meanVector(prototypes.stream()
                .map(item -> vector(item.get("vector")))
                .filter(vector -> !vector.isEmpty())
                .toList());
        if (!centroid.isEmpty()) {
            for (Map<String, Object> hit : qdrantClient.searchFrames(centroid, userId, videoId, benchmarkRunId, domainId, null, Math.max(10, Math.min(40, limit / 2)))) {
                Map<String, Object> payload = new LinkedHashMap<>(payload(hit));
                annotateFewShotMatch(payload, label, number(hit.get("score")), true);
                hit.put("payload", payload);
                out.add(hit);
            }
        }

        int remainingLimit = Math.max(10, limit - out.size());
        int perPrototypeLimit = Math.max(4, Math.min(18, (int) Math.ceil(remainingLimit / (double) prototypes.size())));
        for (Map<String, Object> prototype : prototypes) {
            List<Double> vector = vector(prototype.get("vector"));
            if (vector.isEmpty()) continue;
            for (Map<String, Object> hit : qdrantClient.searchFrames(vector, userId, videoId, benchmarkRunId, domainId, null, perPrototypeLimit)) {
                Map<String, Object> payload = new LinkedHashMap<>(payload(hit));
                annotateFewShotMatch(payload, label, number(hit.get("score")), false);
                hit.put("payload", payload);
                out.add(hit);
            }
            if (out.size() >= limit) break;
        }
        return out;
    }

    private void annotateFewShotMatch(Map<String, Object> payload, String label, double score, boolean centroidMatch) {
        double boundedScore = clamp(score, 0.0, 1.0);
        payload.put("few_shot_score", round(Math.max(number(payload.get("few_shot_score")), boundedScore)));
        payload.put("few_shot_match", boundedScore >= fewShotMinScore);
        payload.put("few_shot_centroid_match", centroidMatch || Boolean.TRUE.equals(payload.get("few_shot_centroid_match")));
        if (boundedScore >= fewShotMinScore) {
            payload.put("action_labels", mergeLists(List.of(label), payload.get("action_labels")));
            payload.put("action_confidence", round(Math.max(number(payload.get("action_confidence")), boundedScore)));
        }
        if (boundedScore >= fewShotTopScore) {
            payload.put("action_top", label);
        }
    }

    private List<Map<String, Object>> searchFeedbackPrototypeFrames(
            String userId,
            String videoId,
            String benchmarkRunId,
            String domainId,
            DomainModelService.DomainState domainState,
            String query,
            int limit) {
        List<List<Double>> vectors = domainModelService.positiveFeedbackVectors(domainState, query, 3);
        if (vectors.isEmpty()) return List.of();
        List<Map<String, Object>> out = new ArrayList<>();
        int perVectorLimit = Math.max(5, Math.min(30, (int) Math.ceil(limit / (double) vectors.size())));
        for (List<Double> vector : vectors) {
            for (Map<String, Object> hit : qdrantClient.searchFrames(vector, userId, videoId, benchmarkRunId, domainId, null, perVectorLimit)) {
                Map<String, Object> payload = new LinkedHashMap<>(payload(hit));
                payload.put("feedback_candidate", true);
                hit.put("payload", payload);
                out.add(hit);
            }
            if (out.size() >= limit) break;
        }
        return out;
    }

    private List<Map<String, Object>> mergeFrameCandidates(List<Map<String, Object>> frames) {
        Map<String, Map<String, Object>> byFrame = new LinkedHashMap<>();
        for (Map<String, Object> frame : frames) {
            Map<String, Object> payload = payload(frame);
            String key = frameKey(payload);
            if (key.isBlank()) continue;
            Map<String, Object> existing = byFrame.get(key);
            if (existing == null) {
                byFrame.put(key, frame);
                continue;
            }
            Map<String, Object> mergedPayload = new LinkedHashMap<>(payload(existing));
            mergeCandidatePayload(mergedPayload, payload(frame));
            existing.put("payload", mergedPayload);
            existing.put("score", Math.max(number(existing.get("score")), number(frame.get("score"))));
            if (!existing.containsKey("vector") || vector(existing.get("vector")).isEmpty()) {
                existing.put("vector", frame.get("vector"));
            }
        }
        return new ArrayList<>(byFrame.values());
    }

    private void mergeCandidatePayload(Map<String, Object> target, Map<String, Object> source) {
        if (source.isEmpty()) return;
        target.put("tags", mergeLists(target.get("tags"), source.get("tags")));
        target.put("objects", mergeLists(target.get("objects"), source.get("objects")));
        target.put("action_labels", mergeLists(target.get("action_labels"), source.get("action_labels")));
        maxNumber(target, source, "action_confidence");
        maxNumber(target, source, "few_shot_score");
        maxNumber(target, source, "feedback_score");
        maxNumber(target, source, "domain_score");
        if (Boolean.TRUE.equals(source.get("few_shot_match"))) target.put("few_shot_match", true);
        if (Boolean.TRUE.equals(source.get("few_shot_centroid_match"))) target.put("few_shot_centroid_match", true);
        if (Boolean.TRUE.equals(source.get("feedback_candidate"))) target.put("feedback_candidate", true);
        if (number(source.get("few_shot_score")) >= fewShotTopScore) {
            target.put("action_top", source.get("action_top"));
        } else {
            putIfMissing(target, "action_top", source.get("action_top"));
        }
    }

    private void maxNumber(Map<String, Object> target, Map<String, Object> source, String key) {
        if (!source.containsKey(key)) return;
        target.put(key, round(Math.max(number(target.get(key)), number(source.get(key)))));
    }

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
            String queryAction,
            DomainModelService.DomainState domainState,
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
            double actionScore = actionScore(queryAction, payload);
            double metadataScore = metadataScore(query, payload);
            double feedbackScore = domainState == null ? 0.0 : domainModelService.feedbackScore(domainState, query, vector(frame.get("vector")));
            double feedbackSignal = domainState == null || feedbackScore == 0.0 ? 0.0 : ((feedbackScore - 0.5) * 2.0);
            double initialScore = clamp(
                    (0.25 * frameRankScore)
                            + (0.20 * chunkRankScore)
                            + (0.20 * tagScore)
                            + (0.30 * actionScore)
                            + (0.05 * metadataScore)
                            + (0.20 * feedbackSignal),
                    0.0,
                    1.0);

            String s3Key = string(payload.get("s3_frame_path"));
            if (!s3Key.isBlank()) {
                payload.put("url", awsWrapperService.generatePresignedUrl(bucket, s3Key, 10).toString());
            }
            String chunkS3Key = string(payload.get("s3_chunk_path"));
            if (!chunkS3Key.isBlank()) {
                payload.put("clip_url", awsWrapperService.generatePresignedUrl(bucket, chunkS3Key, 10).toString());
            }
            payload.put("vector_score", round(number(frame.get("score"))));
            payload.put("frame_rank_score", round(frameRankScore));
            payload.put("chunk_rank_score", round(chunkRankScore));
            payload.put("tag_score", round(tagScore));
            payload.put("action_score", round(actionScore));
            payload.put("query_action", queryAction);
            payload.put("metadata_score", round(metadataScore));
            payload.put("domain_id", domainState == null ? string(payload.get("domain_id")) : domainState.domainId());
            payload.put("domain_score", round(feedbackScore));
            payload.put("feedback_score", round(feedbackScore));
            payload.put("feedback_signal", round(feedbackSignal));
            payload.put("model_version", domainState == null ? "" : domainState.modelVersion());
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
        putIfMissing(framePayload, "s3_chunk_path", chunkPayload.get("s3_chunk_path"));
        putIfMissing(framePayload, "analysis_confidence", chunkPayload.get("analysis_confidence"));
        putIfMissing(framePayload, "action_top", chunkPayload.get("action_top"));
        putIfMissing(framePayload, "action_confidence", chunkPayload.get("action_confidence"));
        putIfMissing(framePayload, "action_scores", chunkPayload.get("action_scores"));
        framePayload.put("tags", mergeLists(framePayload.get("tags"), chunkPayload.get("tags")));
        framePayload.put("objects", mergeLists(framePayload.get("objects"), chunkPayload.get("objects")));
        framePayload.put("action_labels", mergeLists(framePayload.get("action_labels"), chunkPayload.get("action_labels")));
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
                string(payload.get("action_top")),
                valueText(payload.get("tags")),
                valueText(payload.get("objects")),
                valueText(payload.get("action_labels"))));
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

    private double actionScore(String queryAction, Map<String, Object> payload) {
        if (queryAction == null || queryAction.isBlank()) return 0.0;
        double fewShotScore = number(payload.get("few_shot_score"));
        if (queryAction.equals(string(payload.get("action_top")))) {
            if (fewShotScore > 0.0) {
                return fewShotActionScore(fewShotScore);
            }
            return Math.max(0.85, number(payload.get("action_confidence")));
        }
        for (String label : mergeLists(payload.get("action_labels"), null)) {
            if (queryAction.equals(label)) {
                if (fewShotScore > 0.0) {
                    return fewShotActionScore(fewShotScore) * 0.85;
                }
                return Math.max(0.65, number(payload.get("action_confidence")) * 0.85);
            }
        }
        return 0.0;
    }

    private double fewShotActionScore(double score) {
        if (score < fewShotMinScore) return 0.0;
        double normalized = (score - fewShotMinScore) / Math.max(0.0001, 1.0 - fewShotMinScore);
        return clamp(0.50 + (normalized * 0.45), 0.0, 0.95);
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

    private List<Double> vector(Object value) {
        if (!(value instanceof Iterable<?> iterable)) return List.of();
        List<Double> out = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Number number) {
                out.add(number.doubleValue());
            }
        }
        return out;
    }

    private List<Double> meanVector(List<List<Double>> vectors) {
        if (vectors.isEmpty()) return List.of();
        int size = vectors.stream().mapToInt(List::size).min().orElse(0);
        if (size == 0) return List.of();
        List<Double> out = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (List<Double> vector : vectors) {
                sum += vector.get(i);
            }
            out.add(sum / vectors.size());
        }
        return out;
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
