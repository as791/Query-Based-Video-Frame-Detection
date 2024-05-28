package com.video;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class RestTemplateWrapper {
    private RestTemplate restTemplate;

    public RestTemplateWrapper() {
        this.restTemplate = new RestTemplate();
    }

    public <T> ResponseEntity<T> run(String url, HttpMethod method, HttpEntity<SearchRequest> requestEntity, Class<T> classZ) {
        return restTemplate.exchange(url, method, requestEntity, classZ);
    }
}
