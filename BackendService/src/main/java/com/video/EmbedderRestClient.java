package com.video;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class EmbedderRestClient {

    @Value("${EMBEDDER_URL:http://embedder:8002}")
    private String embedderUrl;

    private final RestTemplate restTemplate;

    @SuppressWarnings("unchecked")
    public List<Double> embedText(String text) {
        Map<String, Object> resp = restTemplate.postForObject(
                embedderUrl + "/embed/text",
                Map.of("text", text),
                Map.class);
        return (List<Double>) resp.get("embedding");
    }
}
