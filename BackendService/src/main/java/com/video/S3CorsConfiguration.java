package com.video;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.BucketCrossOriginConfiguration;
import com.amazonaws.services.s3.model.CORSRule;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

@Component
@RequiredArgsConstructor
@Slf4j
public class S3CorsConfiguration {

    private static final String RULE_ID = "videovault-browser-upload";

    private final AmazonS3 amazonS3;

    @Value("${S3_BUCKET:stage-video-bucket}")
    private String bucket;

    @Value("${FRONTEND_ORIGIN:http://localhost:3000}")
    private String frontendOrigin;

    @Value("${S3_CONFIGURE_CORS:true}")
    private boolean configureCors;

    @PostConstruct
    public void configureBucketCors() {
        if (!configureCors) {
            return;
        }

        try {
            BucketCrossOriginConfiguration current = amazonS3.getBucketCrossOriginConfiguration(bucket);
            List<CORSRule> rules = current == null || current.getRules() == null
                    ? new ArrayList<>()
                    : new ArrayList<>(current.getRules());

            rules.removeIf(rule -> RULE_ID.equals(rule.getId()));
            rules.add(browserUploadRule());

            amazonS3.setBucketCrossOriginConfiguration(bucket, new BucketCrossOriginConfiguration(rules));
            log.info("Configured S3 CORS rule {} on bucket {}", RULE_ID, bucket);
        } catch (Exception e) {
            log.warn("Could not configure S3 CORS on bucket {}. Direct browser uploads may fail: {}",
                    bucket, e.getMessage());
        }
    }

    private CORSRule browserUploadRule() {
        Set<String> origins = new LinkedHashSet<>();
        origins.add(frontendOrigin);
        origins.add("http://localhost:3000");
        origins.add("http://127.0.0.1:3000");

        CORSRule rule = new CORSRule()
                .withId(RULE_ID)
                .withAllowedMethods(List.of(
                        CORSRule.AllowedMethods.PUT,
                        CORSRule.AllowedMethods.POST,
                        CORSRule.AllowedMethods.GET,
                        CORSRule.AllowedMethods.HEAD))
                .withAllowedOrigins(new ArrayList<>(origins))
                .withAllowedHeaders(List.of("*"))
                .withExposedHeaders(List.of("ETag", "x-amz-request-id", "x-amz-id-2"))
                .withMaxAgeSeconds(3000);
        return rule;
    }
}
