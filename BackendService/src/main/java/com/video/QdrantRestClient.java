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

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> searchChunks(List<Double> vector, String userId, String videoId, int limit) {
        Map<String, Object> filter = buildUserFilter(userId, videoId);
        Map<String, Object> body = Map.of(
                "vector", vector,
                "filter", filter,
                "limit", limit,
                "with_payload", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/chunks/points/search", body, Map.class);
        return (List<Map<String, Object>>) resp.get("result");
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> searchFrames(List<Double> vector, String userId, String videoId, List<String> chunkIds, int limit) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        if (videoId != null && !videoId.isBlank()) {
            must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        }
        if (chunkIds != null && !chunkIds.isEmpty()) {
            must.add(Map.of("key", "chunk_id", "match", Map.of("any", chunkIds)));
        }
        Map<String, Object> body = Map.of(
                "vector", vector,
                "filter", Map.of("must", must),
                "limit", limit,
                "with_payload", true);
        Map<String, Object> resp = restTemplate.postForObject(
                qdrantUrl + "/collections/frames/points/search", body, Map.class);
        return (List<Map<String, Object>>) resp.get("result");
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

    private Map<String, Object> buildUserFilter(String userId, String videoId) {
        List<Map<String, Object>> must = new ArrayList<>();
        must.add(Map.of("key", "user_id", "match", Map.of("value", userId)));
        if (videoId != null && !videoId.isBlank()) {
            must.add(Map.of("key", "video_id", "match", Map.of("value", videoId)));
        }
        return Map.of("must", must);
    }
}
