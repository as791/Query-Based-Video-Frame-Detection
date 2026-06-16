package com.video;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RStream;
import org.redisson.api.RedissonClient;
import org.redisson.api.StreamMessageId;
import org.redisson.client.codec.StringCodec;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/video")
@RequiredArgsConstructor
@Slf4j
public class VideoStatusController {

    private final RedissonClient redissonClient;
    private final UserRepository userRepository;

    @GetMapping(value = "/{videoId}/status", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter videoStatus(
            @PathVariable String videoId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        SseEmitter emitter = new SseEmitter(5 * 60 * 1000L); // 5 min timeout
        Executors.newSingleThreadExecutor().submit(() -> {
            try {
                RStream<String, String> stream = redissonClient.getStream("ui.status." + videoId, StringCodec.INSTANCE);
                StreamMessageId lastId = StreamMessageId.MIN;
                int idleCount = 0;
                int chunkCount = 0;
                Set<String> normalized = new HashSet<>();
                Set<String> indexed = new HashSet<>();
                Set<String> failed = new HashSet<>();
                while (idleCount < 60) {
                    var messages = stream.read(org.redisson.api.stream.StreamReadArgs
                            .greaterThan(lastId)
                            .count(10)
                            .timeout(java.time.Duration.ofSeconds(2)));
                    if (messages == null || messages.isEmpty()) {
                        idleCount++;
                        continue;
                    }
                    idleCount = 0;
                    for (Map.Entry<StreamMessageId, Map<String, String>> entry : messages.entrySet()) {
                        lastId = entry.getKey();
                        Map<String, String> data = entry.getValue();
                        String stage = data.getOrDefault("stage", "");
                        String chunkId = data.get("chunk_id");
                        if (data.containsKey("chunk_count")) {
                            chunkCount = Math.max(chunkCount, parseInt(data.get("chunk_count")));
                        }
                        if (chunkId != null && !chunkId.isBlank()) {
                            if ("normalized".equals(stage)) normalized.add(chunkId);
                            if ("indexed".equals(stage)) indexed.add(chunkId);
                            if ("failed".equals(stage)) failed.add(chunkId);
                        }

                        Map<String, String> snapshot = new HashMap<>(data);
                        snapshot.put("chunk_count", String.valueOf(chunkCount));
                        snapshot.put("normalized_count", String.valueOf(normalized.size()));
                        snapshot.put("indexed_count", String.valueOf(indexed.size()));
                        snapshot.put("failed_count", String.valueOf(failed.size()));
                        snapshot.put("terminal_count", String.valueOf(indexed.size() + failed.size()));

                        boolean allTerminal = chunkCount > 0 && indexed.size() + failed.size() >= chunkCount;
                        if (allTerminal) {
                            snapshot.put("stage", failed.isEmpty() ? "ready" : "partial");
                        }
                        emitter.send(SseEmitter.event().data(snapshot));
                        if (allTerminal) {
                            emitter.complete();
                            return;
                        }
                    }
                }
                emitter.complete();
            } catch (IOException e) {
                emitter.completeWithError(e);
            }
        });
        return emitter;
    }

    @GetMapping("/{videoId}/status/snapshot")
    public Map<String, String> videoStatusSnapshot(
            @PathVariable String videoId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);
        return aggregateStatus(videoId);
    }

    private Map<String, String> aggregateStatus(String videoId) {
        RStream<String, String> stream = redissonClient.getStream("ui.status." + videoId, StringCodec.INSTANCE);
        int chunkCount = 0;
        Set<String> normalized = new HashSet<>();
        Set<String> indexed = new HashSet<>();
        Set<String> failed = new HashSet<>();
        Map<String, String> latest = new HashMap<>();

        Map<StreamMessageId, Map<String, String>> messages = stream.range(StreamMessageId.MIN, StreamMessageId.MAX);
        if (messages == null || messages.isEmpty()) {
            latest.put("stage", "queued");
            latest.put("chunk_count", "0");
            latest.put("normalized_count", "0");
            latest.put("indexed_count", "0");
            latest.put("failed_count", "0");
            latest.put("terminal_count", "0");
            return latest;
        }

        for (Map<String, String> data : messages.values()) {
            String stage = data.getOrDefault("stage", "");
            String chunkId = data.get("chunk_id");
            if (data.containsKey("chunk_count")) {
                chunkCount = Math.max(chunkCount, parseInt(data.get("chunk_count")));
            }
            if (chunkId != null && !chunkId.isBlank()) {
                if ("normalized".equals(stage)) normalized.add(chunkId);
                if ("indexed".equals(stage)) indexed.add(chunkId);
                if ("failed".equals(stage)) failed.add(chunkId);
            }
            latest = new HashMap<>(data);
        }

        latest.put("chunk_count", String.valueOf(chunkCount));
        latest.put("normalized_count", String.valueOf(normalized.size()));
        latest.put("indexed_count", String.valueOf(indexed.size()));
        latest.put("failed_count", String.valueOf(failed.size()));
        latest.put("terminal_count", String.valueOf(indexed.size() + failed.size()));
        if (chunkCount > 0 && indexed.size() + failed.size() >= chunkCount) {
            latest.put("stage", failed.isEmpty() ? "ready" : "partial");
        }
        return latest;
    }

    private int parseInt(String value) {
        try {
            return value == null ? 0 : Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
