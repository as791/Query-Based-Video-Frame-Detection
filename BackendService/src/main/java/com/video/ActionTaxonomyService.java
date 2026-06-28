package com.video;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

@Service
@RequiredArgsConstructor
@Slf4j
public class ActionTaxonomyService {

    private static final String REDIS_KEY = "taxonomy:action-labels";
    private static final Pattern NON_LABEL = Pattern.compile("[^a-z0-9]");
    private static final TypeReference<List<ActionLabel>> LABEL_LIST = new TypeReference<>() {};
    private static final Set<String> DESCRIPTION_STOPWORDS = Set.of(
            "person", "people", "being", "with", "without", "into", "from", "that", "this",
            "footage", "cctv", "video", "action", "suspicious", "involving", "another",
            "something", "store", "building", "property", "normal"
    );

    private static final List<ActionLabel> DEFAULT_LABELS = List.of(
            new ActionLabel("abuse", "a person being abused or attacked"),
            new ActionLabel("arrest", "a person being arrested"),
            new ActionLabel("arson", "a person starting a fire"),
            new ActionLabel("assault", "a person assaulting another person"),
            new ActionLabel("burglary", "a person breaking into a building"),
            new ActionLabel("explosion", "an explosion or blast"),
            new ActionLabel("fighting", "people fighting"),
            new ActionLabel("normal", "normal CCTV footage with no suspicious action"),
            new ActionLabel("roadaccidents", "a road accident involving vehicles or pedestrians"),
            new ActionLabel("robbery", "a robbery or theft in progress"),
            new ActionLabel("shooting", "a person shooting a weapon"),
            new ActionLabel("shoplifting", "a person shoplifting in a store"),
            new ActionLabel("stealing", "a person stealing something"),
            new ActionLabel("vandalism", "a person vandalizing property")
    );

    private final RedissonClient redissonClient;
    private final ObjectMapper objectMapper;

    public List<ActionLabel> labels() {
        RBucket<String> bucket = bucket();
        String raw = bucket.get();
        if (raw == null || raw.isBlank()) {
            save(DEFAULT_LABELS);
            return DEFAULT_LABELS;
        }
        try {
            List<ActionLabel> parsed = objectMapper.readValue(raw, LABEL_LIST);
            List<ActionLabel> normalized = dedupe(parsed);
            if (normalized.isEmpty()) return DEFAULT_LABELS;
            return normalized;
        } catch (Exception e) {
            log.warn("Could not parse action taxonomy from Redis; using defaults: {}", e.getMessage());
            return DEFAULT_LABELS;
        }
    }

    public ActionLabel add(String label, String description) {
        String normalized = normalize(label);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("label must contain at least one letter or number");
        }
        List<ActionLabel> labels = new ArrayList<>(labels());
        ActionLabel next = new ActionLabel(normalized, cleanDescription(description));
        boolean replaced = false;
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i).label().equals(normalized)) {
                labels.set(i, next);
                replaced = true;
                break;
            }
        }
        if (!replaced) labels.add(next);
        save(labels);
        return next;
    }

    public String detectAction(String query) {
        Map<String, ActionLabel> labels = new LinkedHashMap<>();
        for (ActionLabel label : labels()) {
            labels.put(label.label(), label);
        }
        String compact = normalize(query);
        if (labels.containsKey(compact)) return compact;

        Set<String> queryTerms = terms(query, Set.of());
        for (ActionLabel label : labels.values()) {
            if (queryTerms.contains(label.label()) || queryTerms.contains(stem(label.label()))) {
                return label.label();
            }
        }

        for (ActionLabel label : labels.values()) {
            for (String token : terms(label.description(), DESCRIPTION_STOPWORDS)) {
                if (queryTerms.contains(token)) {
                    return label.label();
                }
            }
        }
        return "";
    }

    private List<ActionLabel> dedupe(List<ActionLabel> input) {
        Map<String, ActionLabel> byLabel = new LinkedHashMap<>();
        for (ActionLabel item : input == null ? List.<ActionLabel>of() : input) {
            String label = normalize(item == null ? "" : item.label());
            if (!label.isBlank()) {
                byLabel.put(label, new ActionLabel(label, cleanDescription(item.description())));
            }
        }
        return new ArrayList<>(byLabel.values());
    }

    private void save(List<ActionLabel> labels) {
        try {
            bucket().set(objectMapper.writeValueAsString(dedupe(labels)));
        } catch (Exception e) {
            throw new IllegalStateException("Could not save action labels", e);
        }
    }

    private RBucket<String> bucket() {
        return redissonClient.getBucket(REDIS_KEY, StringCodec.INSTANCE);
    }

    private String normalize(String value) {
        return NON_LABEL.matcher(String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT)).replaceAll("");
    }

    private String cleanDescription(String value) {
        return String.valueOf(value == null ? "" : value).trim();
    }

    private List<String> tokens(String value) {
        List<String> out = new ArrayList<>();
        for (String token : String.valueOf(value == null ? "" : value).toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (token.length() >= 3 && !out.contains(token)) out.add(token);
        }
        return out;
    }

    private Set<String> terms(String value, Set<String> stopwords) {
        Set<String> out = new HashSet<>();
        for (String token : tokens(value)) {
            if (stopwords.contains(token)) continue;
            out.add(token);
            out.add(stem(token));
        }
        String compact = normalize(value);
        if (compact.length() >= 3) {
            out.add(compact);
            out.add(stem(compact));
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

    public record ActionLabel(String label, String description) {}
}
