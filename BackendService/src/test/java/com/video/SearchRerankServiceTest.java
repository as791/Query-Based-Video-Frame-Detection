package com.video;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class SearchRerankServiceTest {

    private final SearchRerankService service = new SearchRerankService(new ObjectMapper());

    @Test
    void rerank_returns_best_frame_per_video_before_limiting() {
        List<Map<String, Object>> results = service.rerank("person fall", List.of(
                candidate("video-a", "frame-a1", 0.90),
                candidate("video-a", "frame-a2", 0.80),
                candidate("video-b", "frame-b1", 0.70),
                candidate("video-c", "frame-c1", 0.60)
        ), 0.0, 3);

        assertThat(results)
                .extracting(result -> result.get("frame_id"))
                .containsExactly("frame-a1", "frame-b1", "frame-c1");
        assertThat(results)
                .extracting(result -> result.get("video_id"))
                .doesNotHaveDuplicates();
    }

    private Map<String, Object> candidate(String videoId, String frameId, double initialScore) {
        Map<String, Object> candidate = new LinkedHashMap<>();
        candidate.put("video_id", videoId);
        candidate.put("frame_id", frameId);
        candidate.put("caption", "person fall in cctv footage");
        candidate.put("initial_score", initialScore);
        candidate.put("analysis_confidence", 0.50);
        return candidate;
    }
}
