package com.video;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.client.codec.StringCodec;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

@Service
@RequiredArgsConstructor
@Slf4j
public class FewShotLearningService {

    private static final Pattern NON_LABEL = Pattern.compile("[^a-z0-9]");
    private static final TypeReference<FewShotState> STATE_TYPE = new TypeReference<>() {};

    private final RedissonClient redissonClient;
    private final ObjectMapper objectMapper;

    public FewShotState state(String userId) {
        return state(userId, "general");
    }

    public FewShotState state(String userId, String domainId) {
        String raw = bucket(userId, domainId).get();
        if (raw == null || raw.isBlank()) return new FewShotState(List.of(), List.of());
        try {
            FewShotState parsed = objectMapper.readValue(raw, STATE_TYPE);
            return normalize(parsed);
        } catch (Exception e) {
            log.warn("Could not parse few-shot state for user={}: {}", userId, e.getMessage());
            return new FewShotState(List.of(), List.of());
        }
    }

    public FewShotLabel addLabel(String userId, String label, String description) {
        return addLabel(userId, "general", label, description);
    }

    public FewShotLabel addLabel(String userId, String domainId, String label, String description) {
        String normalized = normalizeLabel(label);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("label must contain at least one letter or number");
        }
        FewShotLabel next = new FewShotLabel(normalized, clean(description), 0, "empty");
        FewShotState state = state(userId, domainId);
        Map<String, FewShotLabel> labels = new LinkedHashMap<>();
        for (FewShotLabel item : state.labels()) {
            labels.put(item.label(), item);
        }
        FewShotLabel current = labels.get(normalized);
        labels.put(normalized, new FewShotLabel(
                normalized,
                next.description().isBlank() && current != null ? current.description() : next.description(),
                current == null ? 0 : current.exampleCount(),
                current == null ? "empty" : current.status()
        ));
        FewShotState saved = normalize(new FewShotState(new ArrayList<>(labels.values()), state.examples()));
        save(userId, domainId, saved);
        return saved.labels().stream()
                .filter(item -> item.label().equals(normalized))
                .findFirst()
                .orElse(next);
    }

    public FewShotExample recordExample(String userId, String label, String videoId, String sourceFile, String s3Key) {
        return recordExample(userId, "general", label, videoId, sourceFile, s3Key);
    }

    public FewShotExample recordExample(String userId, String domainId, String label, String videoId, String sourceFile, String s3Key) {
        String normalized = normalizeLabel(label);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("fewShotLabel must contain at least one letter or number");
        }
        FewShotState state = state(userId, domainId);
        String now = Instant.now().toString();
        FewShotExample example = new FewShotExample(videoId, normalized, clean(sourceFile), s3Key, "processing", now);
        List<FewShotExample> examples = new ArrayList<>();
        boolean replaced = false;
        for (FewShotExample item : state.examples()) {
            if (item.videoId().equals(videoId)) {
                examples.add(example);
                replaced = true;
            } else {
                examples.add(item);
            }
        }
        if (!replaced) examples.add(example);

        Map<String, FewShotLabel> labels = new LinkedHashMap<>();
        for (FewShotLabel item : state.labels()) {
            labels.put(item.label(), item);
        }
        labels.putIfAbsent(normalized, new FewShotLabel(normalized, "", 0, "empty"));
        FewShotState saved = normalize(new FewShotState(new ArrayList<>(labels.values()), examples));
        save(userId, domainId, saved);
        return example;
    }

    public String modelId(String userId) {
        return modelId(userId, "general");
    }

    public String modelId(String userId, String domainId) {
        return "fewshot-" + normalizeDomainId(domainId) + "-" + userId;
    }

    public String normalizeLabel(String value) {
        return NON_LABEL.matcher(String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT)).replaceAll("");
    }

    public String detectAction(String userId, String query) {
        return detectAction(userId, "general", query);
    }

    public String detectAction(String userId, String domainId, String query) {
        String compact = normalizeLabel(query);
        FewShotState state = state(userId, domainId);
        for (FewShotLabel label : state.labels()) {
            if (label.label().equals(compact)) return label.label();
        }
        List<String> queryTerms = tokens(query);
        for (FewShotLabel label : state.labels()) {
            if (queryTerms.contains(label.label()) || queryTerms.contains(stem(label.label()))) {
                return label.label();
            }
        }
        for (FewShotLabel label : state.labels()) {
            for (String token : tokens(label.description())) {
                if (queryTerms.contains(token)) return label.label();
            }
        }
        return "";
    }

    private FewShotState normalize(FewShotState input) {
        List<FewShotExample> examples = new ArrayList<>(input == null || input.examples() == null ? List.of() : input.examples());
        examples.sort(Comparator.comparing(FewShotExample::createdAt).reversed());

        Map<String, Integer> counts = new LinkedHashMap<>();
        for (FewShotExample example : examples) {
            String label = normalizeLabel(example.label());
            if (!label.isBlank()) counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        Map<String, FewShotLabel> labels = new LinkedHashMap<>();
        for (FewShotLabel item : input == null || input.labels() == null ? List.<FewShotLabel>of() : input.labels()) {
            String label = normalizeLabel(item.label());
            if (!label.isBlank()) {
                labels.put(label, new FewShotLabel(label, clean(item.description()), counts.getOrDefault(label, 0), status(counts.getOrDefault(label, 0))));
            }
        }
        for (String label : counts.keySet()) {
            labels.putIfAbsent(label, new FewShotLabel(label, "", counts.get(label), status(counts.get(label))));
        }
        return new FewShotState(new ArrayList<>(labels.values()), examples);
    }

    private String status(int exampleCount) {
        if (exampleCount >= 5) return "ready";
        if (exampleCount > 0) return "needs_more_examples";
        return "empty";
    }

    private String clean(String value) {
        return String.valueOf(value == null ? "" : value).trim();
    }

    private List<String> tokens(String value) {
        List<String> out = new ArrayList<>();
        for (String token : String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (token.length() < 3 || out.contains(token)) continue;
            out.add(token);
            String stemmed = stem(token);
            if (!stemmed.equals(token) && !out.contains(stemmed)) out.add(stemmed);
        }
        String compact = normalizeLabel(value);
        if (compact.length() >= 3 && !out.contains(compact)) out.add(compact);
        return out;
    }

    private String stem(String token) {
        if (token == null) return "";
        if (token.length() > 5 && token.endsWith("ing")) return token.substring(0, token.length() - 3);
        if (token.length() > 4 && token.endsWith("ed")) return token.substring(0, token.length() - 2);
        if (token.length() > 4 && token.endsWith("s")) return token.substring(0, token.length() - 1);
        return token;
    }

    private void save(String userId, String domainId, FewShotState state) {
        try {
            bucket(userId, domainId).set(objectMapper.writeValueAsString(normalize(state)));
        } catch (Exception e) {
            throw new IllegalStateException("Could not save few-shot state", e);
        }
    }

    private RBucket<String> bucket(String userId) {
        return bucket(userId, "general");
    }

    private RBucket<String> bucket(String userId, String domainId) {
        return redissonClient.getBucket("fewshot:" + normalizeDomainId(domainId) + ":" + userId + ":state", StringCodec.INSTANCE);
    }

    private String normalizeDomainId(String value) {
        String cleaned = String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT).trim().replaceAll("[^a-z0-9_-]", "-");
        cleaned = cleaned.replaceAll("-+", "-").replaceAll("^-|-$", "");
        return cleaned.isBlank() ? "general" : cleaned;
    }

    public record FewShotState(List<FewShotLabel> labels, List<FewShotExample> examples) {}
    public record FewShotLabel(String label, String description, int exampleCount, String status) {}
    public record FewShotExample(String videoId, String label, String sourceFile, String s3Key, String status, String createdAt) {}
}
