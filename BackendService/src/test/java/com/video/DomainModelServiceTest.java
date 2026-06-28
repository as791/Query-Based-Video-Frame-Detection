package com.video;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class DomainModelServiceTest {

    private final DomainModelService service = new DomainModelService(null, new ObjectMapper(), null);

    @Test
    void normalizes_domain_and_label_ids() {
        assertThat(service.normalizeDomainId(" Retail Theft ")).isEqualTo("retail-theft");
        assertThat(service.normalizeDomainId("")).isEqualTo("general");
        assertThat(service.normalizeLabel("lying_down")).isEqualTo("lyingdown");
        assertThat(service.normalizeLabel("Gun!")).isEqualTo("gun");
    }

    @Test
    void feedback_score_rewards_positive_centroid_and_penalizes_negative_centroid() {
        DomainModelService.DomainState state = new DomainModelService.DomainState(
                1,
                "default",
                "user-1",
                "retail",
                "Retail",
                "v1",
                List.of(),
                List.of(),
                Map.of(),
                new DomainModelService.FeedbackCentroids(
                        new DomainModelService.CentroidPair(
                                new DomainModelService.Centroid(2, List.of(1.0, 0.0, 0.0)),
                                new DomainModelService.Centroid(2, List.of(0.0, 1.0, 0.0))),
                        Map.of("person-run", new DomainModelService.CentroidPair(
                                new DomainModelService.Centroid(1, List.of(1.0, 0.0, 0.0)),
                                new DomainModelService.Centroid(1, List.of(0.0, 1.0, 0.0))))),
                Map.of(),
                "2026-06-19T00:00:00Z");

        assertThat(service.feedbackScore(state, "person run", List.of(1.0, 0.0, 0.0))).isGreaterThan(0.80);
        assertThat(service.feedbackScore(state, "person run", List.of(0.0, 1.0, 0.0))).isLessThan(0.25);
    }
}
