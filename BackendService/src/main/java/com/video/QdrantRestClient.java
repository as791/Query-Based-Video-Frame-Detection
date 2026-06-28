package com.video;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
@RequiredArgsConstructor
public class QdrantRestClient {

    @Value("${QDRANT_URL:http://qdrant:6333}")
    private String qdrantUrl;

    private final RestTemplate restTemplate;

    public List<Map<String, Object>> searchChunks(List<Double> vector, String userId, String videoId, int limit) {
        return searchChunks(vector, userId, videoId, "", limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> searchChunks(List<Double> vector, String userId, String videoId, String benchmarkRunId, int limit) {
        return searchChunks(vector, userId, videoId, benchmarkRunId, "", limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> searchChunks(List<Double> vector, String userId, String videoId, String benchmarkRunId, String domainId, int limit) {
        Map<String, Object> filter = buildUserFilter(userId, videoId, benchmarkRunId, domainId);
        Map<String, Object> body = Map.of(
                "vector", vector,
                "filter", filter,
                "limit", limit,
                "with_payload", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/chunks/points/search", body, Map.class);
        return (List<Map<String, Object>>) resp.get("result");
    }

    public List<Map<String, Object>> searchFrames(List<Double> vector, String userId, String videoId, List<String> chunkIds, int limit) {
        return searchFrames(vector, userId, videoId, "", chunkIds, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> searchFrames(List<Double> vector, String userId, String videoId, String benchmarkRunId, List<String> chunkIds, int limit) {
        return searchFrames(vector, userId, videoId, benchmarkRunId, "", chunkIds, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> searchFrames(List<Double> vector, String userId, String videoId, String benchmarkRunId, String domainId, List<String> chunkIds, int limit) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        if (videoId != null && !videoId.isBlank()) {
            must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        }
        if (benchmarkRunId != null && !benchmarkRunId.isBlank()) {
            must.add(Map.of("key", "benchmark_run_id", "match", Map.of("value", benchmarkRunId)));
        }
        if (domainId != null && !domainId.isBlank()) {
            must.add(Map.of("key", "domain_id", "match", Map.of("value", domainId)));
        }
        if (chunkIds != null && !chunkIds.isEmpty()) {
            must.add(Map.of("key", "chunk_id", "match", Map.of("any", chunkIds)));
        }
        Map<String, Object> body = Map.of(
                "vector", vector,
                "filter", Map.of("must", must),
                "limit", limit,
                "with_payload", true,
                "with_vector", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/frames/points/search", body, Map.class);
        return (List<Map<String, Object>>) resp.get("result");
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollChunksByAction(String userId, String videoId, String benchmarkRunId, String action, int limit) {
        return scrollChunksByAction(userId, videoId, benchmarkRunId, "", action, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollChunksByAction(String userId, String videoId, String benchmarkRunId, String domainId, String action, int limit) {
        return scrollPayloadsByAction("chunks", userId, videoId, benchmarkRunId, domainId, action, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollFramesByAction(String userId, String videoId, String benchmarkRunId, String action, int limit) {
        return scrollFramesByAction(userId, videoId, benchmarkRunId, "", action, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollFramesByAction(String userId, String videoId, String benchmarkRunId, String domainId, String action, int limit) {
        return scrollPayloadsByAction("frames", userId, videoId, benchmarkRunId, domainId, action, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollFewShotExampleFrames(String userId, String label, int limit) {
        return scrollFewShotExampleFrames(userId, "", label, limit);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollFewShotExampleFrames(String userId, String domainId, String label, int limit) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        must.add(Map.of("key", "few_shot_example", "match", Map.of("value", true)));
        must.add(Map.of("key", "few_shot_label", "match", Map.of("value", label)));
        if (domainId != null && !domainId.isBlank()) {
            must.add(Map.of("key", "domain_id", "match", Map.of("value", domainId)));
        }
        Map<String, Object> body = Map.of(
                "filter", Map.of("must", must),
                "limit", Math.max(1, limit),
                "with_payload", true,
                "with_vector", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/frames/points/scroll", body, Map.class);
        Map<String, Object> result = (Map<String, Object>) resp.get("result");
        List<Map<String, Object>> points = (List<Map<String, Object>>) result.get("points");
        return points == null ? List.of() : points;
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> findFrameWithVector(String userId, String videoId, String frameId, String chunkId) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        if (videoId != null && !videoId.isBlank()) {
            must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        }
        if (frameId != null && !frameId.isBlank()) {
            must.add(Map.of("key", "frame_id", "match", Map.of("value", frameId)));
        }
        if (chunkId != null && !chunkId.isBlank()) {
            must.add(Map.of("key", "chunk_id", "match", Map.of("value", chunkId)));
        }
        Map<String, Object> body = Map.of(
                "filter", Map.of("must", must),
                "limit", 1,
                "with_payload", true,
                "with_vector", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/frames/points/scroll", body, Map.class);
        Map<String, Object> result = (Map<String, Object>) resp.get("result");
        List<Map<String, Object>> points = (List<Map<String, Object>>) result.get("points");
        return points == null || points.isEmpty() ? Map.of() : points.get(0);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> scrollFramePayloads(String userId, String videoId, String chunkId, long startFromMs, int limit) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        if (chunkId != null && !chunkId.isBlank()) {
            must.add(Map.of("key", "chunk_id", "match", Map.of("value", chunkId)));
        }
        Map<String, Object> body = Map.of(
                "filter", Map.of("must", must),
                "limit", limit,
                "with_payload", true,
                "order_by", Map.of("key", "t_ms", "start_from", startFromMs));
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/frames/points/scroll", body, Map.class);
        Map<String, Object> result = (Map<String, Object>) resp.get("result");
        List<Map<String, Object>> points = (List<Map<String, Object>>) result.get("points");
        return points.stream()
                .map(point -> (Map<String, Object>) point.get("payload"))
                .toList();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> scrollPayloadsByAction(String collection, String userId, String videoId, String benchmarkRunId, String domainId, String action, int limit) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        if (videoId != null && !videoId.isBlank()) {
            must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        }
        if (benchmarkRunId != null && !benchmarkRunId.isBlank()) {
            must.add(Map.of("key", "benchmark_run_id", "match", Map.of("value", benchmarkRunId)));
        }
        if (domainId != null && !domainId.isBlank()) {
            must.add(Map.of("key", "domain_id", "match", Map.of("value", domainId)));
        }
        must.add(Map.of("key", "action_labels", "match", Map.of("any", List.of(action))));
        Map<String, Object> body = Map.of(
                "filter", Map.of("must", must),
                "limit", Math.max(1, limit),
                "with_payload", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/" + collection + "/points/scroll", body, Map.class);
        Map<String, Object> result = (Map<String, Object>) resp.get("result");
        List<Map<String, Object>> points = (List<Map<String, Object>>) result.get("points");
        return points == null ? List.of() : points;
    }

    private Map<String, Object> buildUserFilter(String userId, String videoId, String benchmarkRunId) {
        return buildUserFilter(userId, videoId, benchmarkRunId, "");
    }

    private Map<String, Object> buildUserFilter(String userId, String videoId, String benchmarkRunId, String domainId) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        if (videoId != null && !videoId.isBlank()) {
            must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        }
        if (benchmarkRunId != null && !benchmarkRunId.isBlank()) {
            must.add(Map.of("key", "benchmark_run_id", "match", Map.of("value", benchmarkRunId)));
        }
        if (domainId != null && !domainId.isBlank()) {
            must.add(Map.of("key", "domain_id", "match", Map.of("value", domainId)));
        }
        return Map.of("must", must);
    }
}
