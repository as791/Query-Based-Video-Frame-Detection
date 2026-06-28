package com.video;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.S3Object;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

@Service
@RequiredArgsConstructor
@Slf4j
public class DomainModelService {

    private static final String TENANT_ID = "default";
    private static final Pattern NON_ID = Pattern.compile("[^a-z0-9_-]");
    private static final Pattern NON_LABEL = Pattern.compile("[^a-z0-9]");
    private static final TypeReference<DomainState> STATE_TYPE = new TypeReference<>() {};
    private static final Set<String> STOPWORDS = Set.of(
            "person", "people", "video", "frame", "frames", "show", "shows", "showing",
            "find", "search", "with", "from", "that", "this", "there", "their");

    private final RedissonClient redissonClient;
    private final ObjectMapper objectMapper;
    private final AmazonS3 amazonS3;
    private final ExecutorService feedbackPersistenceExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "domain-feedback-persistence");
        thread.setDaemon(true);
        return thread;
    });

    @Value("${S3_BUCKET:stage-video-bucket}")
    private String bucket;

    public DomainList list(String userId) {
        DomainList current = readDomainList(userId);
        if (current.domains().isEmpty()) {
            DomainSummary general = new DomainSummary("general", "General", "", 0, 0, "", Instant.now().toString());
            current = new DomainList(List.of(general));
            saveDomainList(userId, current);
            state(userId, "general");
        }
        return current;
    }

    public DomainState createDomain(String userId, String rawName, String rawDescription) {
        String name = clean(rawName).isBlank() ? "New domain" : clean(rawName);
        String domainId = normalizeId(name);
        if (domainId.isBlank()) domainId = "domain-" + UUID.randomUUID().toString().substring(0, 8);
        DomainList list = readDomainList(userId);
        String uniqueId = uniqueDomainId(domainId, list.domains());
        DomainState state = initialState(userId, uniqueId, name, clean(rawDescription));
        saveState(state);

        List<DomainSummary> summaries = new ArrayList<>(list.domains());
        summaries.add(summary(state));
        saveDomainList(userId, new DomainList(summaries));
        return state;
    }

    public DomainState state(String userId, String domainId) {
        String normalizedDomainId = normalizeDomainId(domainId);
        String raw = bucket(userId, normalizedDomainId).get();
        if (raw != null && !raw.isBlank()) {
            try {
                return normalize(objectMapper.readValue(raw, STATE_TYPE));
            } catch (Exception e) {
                log.warn("Could not parse cached domain model user={} domain={}: {}", userId, normalizedDomainId, e.getMessage());
            }
        }
        DomainState fromS3 = loadStateFromS3(userId, normalizedDomainId);
        if (fromS3 != null) {
            bucket(userId, normalizedDomainId).set(write(fromS3));
            return fromS3;
        }
        DomainState created = initialState(userId, normalizedDomainId, title(normalizedDomainId), "");
        saveState(created);
        ensureDomainSummary(userId, created);
        return created;
    }

    public DomainLabel addLabel(String userId, String domainId, String label, String description) {
        DomainState state = state(userId, domainId);
        String normalized = normalizeLabel(label);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("label must contain at least one letter or number");
        }
        Map<String, DomainLabel> labels = new LinkedHashMap<>();
        for (DomainLabel item : state.labels()) labels.put(item.label(), item);
        labels.put(normalized, new DomainLabel(normalized, clean(description), 0, "empty"));
        DomainState next = normalize(new DomainState(
                state.schemaVersion(), state.tenantId(), state.userId(), state.domainId(), state.domainName(),
                nextVersion(), new ArrayList<>(labels.values()), state.examples(), state.feedbackStats(),
                state.feedbackCentroids(), state.feedbackEvents(), state.config(), Instant.now().toString()));
        saveState(next);
        ensureDomainSummary(userId, next);
        return next.labels().stream().filter(item -> item.label().equals(normalized)).findFirst().orElseThrow();
    }

    public void recordExample(String userId, String domainId, String label, String videoId, String sourceFile, String s3Key) {
        DomainState state = state(userId, domainId);
        String normalized = normalizeLabel(label);
        if (normalized.isBlank()) return;
        List<DomainExample> examples = new ArrayList<>(state.examples());
        examples.removeIf(item -> item.videoId().equals(videoId));
        examples.add(new DomainExample(videoId, normalized, clean(sourceFile), s3Key, "processing", Instant.now().toString()));
        DomainState next = normalize(new DomainState(
                state.schemaVersion(), state.tenantId(), state.userId(), state.domainId(), state.domainName(),
                nextVersion(), state.labels(), examples, state.feedbackStats(), state.feedbackCentroids(),
                state.feedbackEvents(), state.config(), Instant.now().toString()));
        saveState(next);
    }

    public String detectAction(String userId, String domainId, String query) {
        if (domainId == null || domainId.isBlank()) return "";
        DomainState state = state(userId, domainId);
        String compact = normalizeLabel(query);
        for (DomainLabel label : state.labels()) {
            if (label.label().equals(compact)) return label.label();
        }
        List<String> queryTerms = tokens(query);
        for (DomainLabel label : state.labels()) {
            if (queryTerms.contains(label.label()) || queryTerms.contains(stem(label.label()))) {
                return label.label();
            }
        }
        for (DomainLabel label : state.labels()) {
            for (String token : tokens(label.description())) {
                if (!STOPWORDS.contains(token) && queryTerms.contains(token)) return label.label();
            }
        }
        return "";
    }

    public DomainState recordFeedback(String userId, FeedbackInput input, List<Double> vector) {
        DomainState state = state(userId, input.domainId());
        String queryKey = queryKey(input.query());
        Map<String, Integer> stats = new LinkedHashMap<>(state.feedbackStats());
        FeedbackCentroids centroids = state.feedbackCentroids();
        List<FeedbackEvent> events = new ArrayList<>(state.feedbackEvents() == null ? List.of() : state.feedbackEvents());
        for (int i = events.size() - 1; i >= 0; i--) {
            FeedbackEvent event = events.get(i);
            if (!event.active() || !feedbackMatches(event, input, queryKey)) continue;
            if (event.relevant() == input.relevant()) return state;
            events.set(i, event.withUndo(Instant.now().toString()));
            centroids = removeCentroids(centroids, queryKey, vector, event.relevant());
            decrementStats(stats, event.relevant());
            break;
        }
        centroids = updateCentroids(centroids, queryKey, vector, input.relevant());
        incrementStats(stats, input.relevant());
        events.add(new FeedbackEvent(
                UUID.randomUUID().toString(), queryKey, clean(input.query()), clean(input.videoId()),
                clean(input.frameId()), clean(input.chunkId()), input.relevant(), true,
                Instant.now().toString(), ""));
        DomainState next = normalize(new DomainState(
                state.schemaVersion(), state.tenantId(), state.userId(), state.domainId(), state.domainName(),
                nextVersion(), state.labels(), state.examples(), stats, centroids, events, state.config(), Instant.now().toString()));
        saveStateHot(next);
        persistFeedbackAsync(next, input);
        return next;
    }

    public DomainState undoFeedback(String userId, FeedbackInput input, List<Double> vector) {
        DomainState state = state(userId, input.domainId());
        String queryKey = queryKey(input.query());
        List<FeedbackEvent> events = new ArrayList<>(state.feedbackEvents() == null ? List.of() : state.feedbackEvents());
        int matchIndex = -1;
        for (int i = events.size() - 1; i >= 0; i--) {
            FeedbackEvent event = events.get(i);
            if (event.active() && event.relevant() == input.relevant() && feedbackMatches(event, input, queryKey)) {
                matchIndex = i;
                break;
            }
        }
        if (matchIndex < 0) return state;

        FeedbackEvent event = events.get(matchIndex);
        events.set(matchIndex, event.withUndo(Instant.now().toString()));
        Map<String, Integer> stats = new LinkedHashMap<>(state.feedbackStats());
        decrementStats(stats, event.relevant());
        FeedbackCentroids centroids = removeCentroids(state.feedbackCentroids(), queryKey, vector, event.relevant());
        DomainState next = normalize(new DomainState(
                state.schemaVersion(), state.tenantId(), state.userId(), state.domainId(), state.domainName(),
                nextVersion(), state.labels(), state.examples(), stats, centroids, events, state.config(), Instant.now().toString()));
        saveState(next);
        return next;
    }

    public double feedbackScore(DomainState state, String query, List<Double> vector) {
        if (state == null || vector == null || vector.isEmpty()) return 0.0;
        CentroidPair pair = state.feedbackCentroids().byQueryKey().getOrDefault(queryKey(query), state.feedbackCentroids().global());
        double positive = similarity(pair.positive(), vector);
        double negative = similarity(pair.negative(), vector);
        if (pair.positive().count() == 0 && pair.negative().count() == 0) return 0.0;
        return clamp((positive - negative + 1.0) / 2.0);
    }

    public List<List<Double>> positiveFeedbackVectors(DomainState state, String query, int limit) {
        if (state == null || state.feedbackCentroids() == null) return List.of();
        List<List<Double>> vectors = new ArrayList<>();
        CentroidPair queryPair = state.feedbackCentroids().byQueryKey().get(queryKey(query));
        addPositiveVector(vectors, queryPair, limit);
        addPositiveVector(vectors, state.feedbackCentroids().global(), limit);
        return vectors;
    }

    public String normalizeDomainId(String value) {
        String normalized = normalizeId(value);
        return normalized.isBlank() ? "general" : normalized;
    }

    public String normalizeLabel(String value) {
        return NON_LABEL.matcher(String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT)).replaceAll("");
    }

    private DomainState normalize(DomainState state) {
        List<DomainExample> examples = new ArrayList<>(state.examples() == null ? List.of() : state.examples());
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (DomainExample example : examples) {
            String label = normalizeLabel(example.label());
            if (!label.isBlank()) counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        Map<String, DomainLabel> labels = new LinkedHashMap<>();
        for (DomainLabel item : state.labels() == null ? List.<DomainLabel>of() : state.labels()) {
            String label = normalizeLabel(item.label());
            if (!label.isBlank()) {
                int count = counts.getOrDefault(label, 0);
                labels.put(label, new DomainLabel(label, clean(item.description()), count, labelStatus(count)));
            }
        }
        for (String label : counts.keySet()) {
            labels.putIfAbsent(label, new DomainLabel(label, "", counts.get(label), labelStatus(counts.get(label))));
        }
        return new DomainState(
                1,
                TENANT_ID,
                state.userId(),
                normalizeDomainId(state.domainId()),
                clean(state.domainName()).isBlank() ? title(state.domainId()) : clean(state.domainName()),
                clean(state.modelVersion()).isBlank() ? nextVersion() : state.modelVersion(),
                new ArrayList<>(labels.values()),
                examples,
                state.feedbackStats() == null ? Map.of() : state.feedbackStats(),
                state.feedbackCentroids() == null ? emptyCentroids() : state.feedbackCentroids(),
                state.feedbackEvents() == null ? List.of() : state.feedbackEvents(),
                state.config() == null ? defaultConfig() : state.config(),
                clean(state.updatedAt()).isBlank() ? Instant.now().toString() : state.updatedAt());
    }

    private FeedbackCentroids updateCentroids(FeedbackCentroids existing, String queryKey, List<Double> vector, boolean relevant) {
        FeedbackCentroids base = existing == null ? emptyCentroids() : existing;
        CentroidPair global = updatePair(base.global(), vector, relevant);
        Map<String, CentroidPair> byQuery = new LinkedHashMap<>(base.byQueryKey() == null ? Map.of() : base.byQueryKey());
        byQuery.put(queryKey, updatePair(byQuery.getOrDefault(queryKey, emptyPair()), vector, relevant));
        return new FeedbackCentroids(global, byQuery);
    }

    private CentroidPair updatePair(CentroidPair pair, List<Double> vector, boolean relevant) {
        pair = pair == null ? emptyPair() : pair;
        return relevant
                ? new CentroidPair(updateCentroid(pair.positive(), vector), pair.negative())
                : new CentroidPair(pair.positive(), updateCentroid(pair.negative(), vector));
    }

    private Centroid updateCentroid(Centroid current, List<Double> vector) {
        current = current == null ? new Centroid(0, List.of()) : current;
        int nextCount = current.count() + 1;
        List<Double> out = new ArrayList<>();
        for (int i = 0; i < vector.size(); i++) {
            double prev = i < current.vector().size() ? current.vector().get(i) : 0.0;
            out.add(((prev * current.count()) + vector.get(i)) / nextCount);
        }
        return new Centroid(nextCount, out);
    }

    private FeedbackCentroids removeCentroids(FeedbackCentroids existing, String queryKey, List<Double> vector, boolean relevant) {
        FeedbackCentroids base = existing == null ? emptyCentroids() : existing;
        CentroidPair global = removePair(base.global(), vector, relevant);
        Map<String, CentroidPair> byQuery = new LinkedHashMap<>(base.byQueryKey() == null ? Map.of() : base.byQueryKey());
        byQuery.put(queryKey, removePair(byQuery.getOrDefault(queryKey, emptyPair()), vector, relevant));
        return new FeedbackCentroids(global, byQuery);
    }

    private CentroidPair removePair(CentroidPair pair, List<Double> vector, boolean relevant) {
        pair = pair == null ? emptyPair() : pair;
        return relevant
                ? new CentroidPair(removeCentroid(pair.positive(), vector), pair.negative())
                : new CentroidPair(pair.positive(), removeCentroid(pair.negative(), vector));
    }

    private Centroid removeCentroid(Centroid current, List<Double> vector) {
        if (current == null || current.count() <= 1) return new Centroid(0, List.of());
        int nextCount = current.count() - 1;
        List<Double> out = new ArrayList<>();
        for (int i = 0; i < current.vector().size(); i++) {
            double removed = i < vector.size() ? vector.get(i) : 0.0;
            out.add(((current.vector().get(i) * current.count()) - removed) / nextCount);
        }
        return new Centroid(nextCount, out);
    }

    private void incrementStats(Map<String, Integer> stats, boolean relevant) {
        stats.put("total", stats.getOrDefault("total", 0) + 1);
        String key = relevant ? "positive" : "negative";
        stats.put(key, stats.getOrDefault(key, 0) + 1);
    }

    private void decrementStats(Map<String, Integer> stats, boolean relevant) {
        stats.put("total", Math.max(0, stats.getOrDefault("total", 0) - 1));
        String key = relevant ? "positive" : "negative";
        stats.put(key, Math.max(0, stats.getOrDefault(key, 0) - 1));
    }

    private boolean feedbackMatches(FeedbackEvent event, FeedbackInput input, String queryKey) {
        return clean(event.queryKey()).equals(queryKey)
                && clean(event.videoId()).equals(clean(input.videoId()))
                && clean(event.frameId()).equals(clean(input.frameId()))
                && clean(event.chunkId()).equals(clean(input.chunkId()));
    }

    private double similarity(Centroid centroid, List<Double> vector) {
        if (centroid == null || centroid.count() == 0 || centroid.vector().isEmpty() || vector.isEmpty()) return 0.0;
        int size = Math.min(centroid.vector().size(), vector.size());
        double dot = 0.0;
        double a = 0.0;
        double b = 0.0;
        for (int i = 0; i < size; i++) {
            double av = centroid.vector().get(i);
            double bv = vector.get(i);
            dot += av * bv;
            a += av * av;
            b += bv * bv;
        }
        if (a == 0.0 || b == 0.0) return 0.0;
        return dot / (Math.sqrt(a) * Math.sqrt(b));
    }

    private DomainState initialState(String userId, String domainId, String name, String description) {
        return new DomainState(1, TENANT_ID, userId, normalizeDomainId(domainId), name, nextVersion(),
                List.of(), List.of(), Map.of(), emptyCentroids(), List.of(), defaultConfig(), Instant.now().toString());
    }

    private DomainList readDomainList(String userId) {
        RBucket<String> domainList = redissonClient.getBucket("domains:" + userId, StringCodec.INSTANCE);
        String raw = domainList.get();
        if (raw == null || raw.isBlank()) return new DomainList(List.of());
        try {
            return objectMapper.readValue(raw, DomainList.class);
        } catch (Exception e) {
            log.warn("Could not parse domain list for user={}: {}", userId, e.getMessage());
            return new DomainList(List.of());
        }
    }

    private void saveDomainList(String userId, DomainList list) {
        RBucket<String> domainList = redissonClient.getBucket("domains:" + userId, StringCodec.INSTANCE);
        domainList.set(write(list));
    }

    private void ensureDomainSummary(String userId, DomainState state) {
        DomainList list = readDomainList(userId);
        List<DomainSummary> next = new ArrayList<>();
        boolean replaced = false;
        for (DomainSummary item : list.domains()) {
            if (item.domainId().equals(state.domainId())) {
                next.add(summary(state));
                replaced = true;
            } else {
                next.add(item);
            }
        }
        if (!replaced) next.add(summary(state));
        saveDomainList(userId, new DomainList(next));
    }

    private DomainSummary summary(DomainState state) {
        return new DomainSummary(
                state.domainId(),
                state.domainName(),
                "",
                state.labels().size(),
                state.examples().size(),
                state.modelVersion(),
                state.updatedAt());
    }

    private void saveState(DomainState state) {
        saveStateHot(state);
        persistStateArtifacts(normalize(state));
    }

    private void saveStateHot(DomainState state) {
        DomainState normalized = normalize(state);
        String json = write(normalized);
        bucket(normalized.userId(), normalized.domainId()).set(json);
    }

    private void persistStateArtifacts(DomainState state) {
        String json = write(state);
        putS3(modelKey(state.userId(), state.domainId(), "current/model.json"), json, "application/json");
        putS3(modelKey(state.userId(), state.domainId(), "versions/" + state.modelVersion() + "/model.json"), json, "application/json");
    }

    private void persistFeedbackAsync(DomainState state, FeedbackInput input) {
        feedbackPersistenceExecutor.submit(() -> {
            DomainState normalized = normalize(state);
            persistStateArtifacts(normalized);
            appendFeedback(normalized, input);
        });
    }

    private DomainState loadStateFromS3(String userId, String domainId) {
        String key = modelKey(userId, domainId, "current/model.json");
        try {
            if (!amazonS3.doesObjectExist(bucket, key)) return null;
            S3Object object = amazonS3.getObject(bucket, key);
            String raw = new String(object.getObjectContent().readAllBytes(), StandardCharsets.UTF_8);
            return normalize(objectMapper.readValue(raw, STATE_TYPE));
        } catch (Exception e) {
            log.warn("Could not load domain model from s3://{}/{}: {}", bucket, key, e.getMessage());
            return null;
        }
    }

    private void appendFeedback(DomainState state, FeedbackInput input) {
        String line = write(input) + "\n";
        putS3(modelKey(state.userId(), state.domainId(), "versions/" + state.modelVersion() + "/feedback.jsonl"), line, "application/x-jsonlines");
    }

    private void putS3(String key, String body, String contentType) {
        try {
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            ObjectMetadata metadata = new ObjectMetadata();
            metadata.setContentLength(bytes.length);
            metadata.setContentType(contentType);
            amazonS3.putObject(bucket, key, new ByteArrayInputStream(bytes), metadata);
        } catch (Exception e) {
            log.warn("Could not persist domain model artifact s3://{}/{}: {}", bucket, key, e.getMessage());
        }
    }

    private RBucket<String> bucket(String userId, String domainId) {
        return redissonClient.getBucket("domain-model:" + TENANT_ID + ":" + userId + ":" + normalizeDomainId(domainId), StringCodec.INSTANCE);
    }

    private String modelKey(String userId, String domainId, String suffix) {
        return "user-models/" + TENANT_ID + "/" + userId + "/" + normalizeDomainId(domainId) + "/" + suffix;
    }

    private void addPositiveVector(List<List<Double>> vectors, CentroidPair pair, int limit) {
        if (pair == null || pair.positive() == null || pair.positive().count() == 0 || pair.positive().vector().isEmpty()) return;
        if (vectors.stream().noneMatch(existing -> sameVector(existing, pair.positive().vector()))) {
            vectors.add(pair.positive().vector());
        }
        if (vectors.size() > Math.max(1, limit)) {
            vectors.subList(Math.max(1, limit), vectors.size()).clear();
        }
    }

    private boolean sameVector(List<Double> left, List<Double> right) {
        if (left.size() != right.size()) return false;
        for (int i = 0; i < left.size(); i++) {
            if (Math.abs(left.get(i) - right.get(i)) > 1e-9) return false;
        }
        return true;
    }

    private String write(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            throw new IllegalStateException("Could not serialize domain model", e);
        }
    }

    private String normalizeId(String value) {
        return NON_ID.matcher(String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT).trim().replaceAll("\\s+", "-")).replaceAll("");
    }

    private String uniqueDomainId(String base, List<DomainSummary> existing) {
        String candidate = base;
        int suffix = 2;
        Set<String> taken = existing.stream().map(DomainSummary::domainId).collect(java.util.stream.Collectors.toSet());
        while (taken.contains(candidate)) {
            candidate = base + "-" + suffix++;
        }
        return candidate;
    }

    private String clean(String value) {
        return String.valueOf(value == null ? "" : value).trim();
    }

    private String title(String domainId) {
        String clean = normalizeDomainId(domainId).replace('-', ' ');
        return clean.isBlank() ? "General" : Character.toUpperCase(clean.charAt(0)) + clean.substring(1);
    }

    private String nextVersion() {
        return "v" + Instant.now().toEpochMilli();
    }

    private String labelStatus(int count) {
        if (count >= 5) return "ready";
        if (count > 0) return "needs_more_examples";
        return "empty";
    }

    private FeedbackCentroids emptyCentroids() {
        return new FeedbackCentroids(emptyPair(), Map.of());
    }

    private CentroidPair emptyPair() {
        return new CentroidPair(new Centroid(0, List.of()), new Centroid(0, List.of()));
    }

    private Map<String, Object> defaultConfig() {
        return Map.of("learningMode", "prototype_feedback_ranker", "prototypeWeight", 0.30, "feedbackWeight", 0.20);
    }

    private String queryKey(String query) {
        List<String> tokens = tokens(query);
        return tokens.isEmpty() ? "global" : String.join("-", tokens);
    }

    private List<String> tokens(String value) {
        List<String> out = new ArrayList<>();
        for (String token : String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (token.length() < 3 || STOPWORDS.contains(token) || out.contains(token)) continue;
            out.add(token);
            String stemmed = stem(token);
            if (!stemmed.equals(token) && !out.contains(stemmed)) out.add(stemmed);
            if (out.size() >= 8) break;
        }
        return out;
    }

    private String stem(String token) {
        if (token == null) return "";
        if (token.length() > 5 && token.endsWith("ing")) return token.substring(0, token.length() - 3);
        if (token.length() > 4 && token.endsWith("ed")) return token.substring(0, token.length() - 2);
        if (token.length() > 4 && token.endsWith("s")) return token.substring(0, token.length() - 1);
        return token;
    }

    private double clamp(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    public record DomainList(List<DomainSummary> domains) {}
    public record DomainSummary(
            String domainId,
            String domainName,
            String description,
            int labelCount,
            int exampleCount,
            String modelVersion,
            String updatedAt) {}
    public record DomainState(
            int schemaVersion,
            String tenantId,
            String userId,
            String domainId,
            String domainName,
            String modelVersion,
            List<DomainLabel> labels,
            List<DomainExample> examples,
            Map<String, Integer> feedbackStats,
            FeedbackCentroids feedbackCentroids,
            List<FeedbackEvent> feedbackEvents,
            Map<String, Object> config,
            String updatedAt) {}
    public record DomainLabel(String label, String description, int exampleCount, String status) {}
    public record DomainExample(String videoId, String label, String sourceFile, String s3Key, String status, String createdAt) {}
    public record FeedbackEvent(
            String id,
            String queryKey,
            String query,
            String videoId,
            String frameId,
            String chunkId,
            boolean relevant,
            boolean active,
            String createdAt,
            String undoneAt) {
        FeedbackEvent withUndo(String undoneAt) {
            return new FeedbackEvent(id, queryKey, query, videoId, frameId, chunkId, relevant, false, createdAt, undoneAt);
        }
    }
    public record FeedbackCentroids(CentroidPair global, Map<String, CentroidPair> byQueryKey) {}
    public record CentroidPair(Centroid positive, Centroid negative) {}
    public record Centroid(int count, List<Double> vector) {}
    public record FeedbackInput(String domainId, String query, String videoId, String frameId, String chunkId, boolean relevant) {}
}
