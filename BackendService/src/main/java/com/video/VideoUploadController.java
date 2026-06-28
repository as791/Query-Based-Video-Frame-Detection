package com.video;

import com.amazonaws.HttpMethod;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.GeneratePresignedUrlRequest;
import com.amazonaws.services.s3.model.AbortMultipartUploadRequest;
import com.amazonaws.services.s3.model.CompleteMultipartUploadRequest;
import com.amazonaws.services.s3.model.InitiateMultipartUploadRequest;
import com.amazonaws.services.s3.model.InitiateMultipartUploadResult;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.PartETag;
import com.amazonaws.services.s3.model.PutObjectRequest;
import com.amazonaws.services.s3.model.SSEAwsKeyManagementParams;
import com.amazonaws.services.s3.model.SSEAlgorithm;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RStream;
import org.redisson.api.RedissonClient;
import org.redisson.api.stream.StreamAddArgs;
import org.redisson.client.codec.StringCodec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@CrossOrigin(originPatterns = {"${FRONTEND_ORIGIN:http://localhost:3000}", "http://localhost:*", "http://127.0.0.1:*"}, allowCredentials = "true")
@RequestMapping("v1/video")
@RequiredArgsConstructor
@Slf4j
public class VideoUploadController {

    private final AmazonS3 amazonS3;
    private final UserRepository userRepository;
    private final RedissonClient redissonClient;
    private final FewShotLearningService fewShotLearningService;
    private final DomainModelService domainModelService;

    @Value("${S3_BUCKET:stage-video-bucket}")
    private String bucket;

    @PostMapping("/presignUpload")
    public Map<String, String> presignUpload(
            @RequestParam(defaultValue = "general") String profile,
            @RequestParam(required = false, defaultValue = "video.mp4") String fileName,
            @RequestParam(required = false, defaultValue = "video/mp4") String contentType,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String videoId = UUID.randomUUID().toString();
        String s3Key = rawVideoKey(user.getId(), videoId);

        Date expiry = new Date(System.currentTimeMillis() + 10L * 60 * 1000); // 10 min
        GeneratePresignedUrlRequest req = new GeneratePresignedUrlRequest(bucket, s3Key)
                .withMethod(HttpMethod.PUT)
                .withExpiration(expiry)
                .withContentType(contentType);

        Map<String, String> uploadHeaders = new java.util.LinkedHashMap<>();
        uploadHeaders.put("Content-Type", contentType);
        if (user.getKmsKeyArn() != null && !user.getKmsKeyArn().isBlank()) {
            req.withSSEAlgorithm(SSEAlgorithm.KMS.getAlgorithm());
            req.withKmsCmkId(user.getKmsKeyArn());
            uploadHeaders.put("x-amz-server-side-encryption", "aws:kms");
            uploadHeaders.put("x-amz-server-side-encryption-aws-kms-key-id", user.getKmsKeyArn());
        }

        URL presignedUrl = amazonS3.generatePresignedUrl(req);

        log.info("Presigned upload URL for user={} video={}", user.getId(), videoId);
        Map<String, String> response = new java.util.LinkedHashMap<>();
        response.put("videoId", videoId);
        response.put("presignedPutUrl", presignedUrl.toString());
        response.put("s3Key", s3Key);
        response.put("expiresIn", "600");
        response.put("fileName", fileName);
        uploadHeaders.forEach((key, value) -> response.put("uploadHeader:" + key, value));
        return response;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, String> upload(
            @RequestParam(defaultValue = "general") String profile,
            @RequestParam(required = false, defaultValue = "") String benchmarkRunId,
            @RequestParam(required = false, defaultValue = "") String fewShotLabel,
            @RequestParam(required = false, defaultValue = "general") String domainId,
            @RequestParam("file") MultipartFile file,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String videoId = UUID.randomUUID().toString();
        String s3Key = rawVideoKey(user.getId(), videoId);

        ObjectMetadata metadata = new ObjectMetadata();
        metadata.setContentLength(file.getSize());
        metadata.setContentType(file.getContentType() == null || file.getContentType().isBlank()
                ? "video/mp4"
                : file.getContentType());

        try {
            PutObjectRequest req = new PutObjectRequest(bucket, s3Key, file.getInputStream(), metadata);
            if (user.getKmsKeyArn() != null && !user.getKmsKeyArn().isBlank()) {
                req.withSSEAwsKeyManagementParams(new SSEAwsKeyManagementParams(user.getKmsKeyArn()));
            }
            amazonS3.putObject(req);
            publishUploadedEvent(videoId, user.getId(), profile, s3Key, file.getOriginalFilename(), benchmarkRunId, fewShotLabel, domainId);
            log.info("Uploaded video via backend for user={} video={}", user.getId(), videoId);
            return Map.of("videoId", videoId, "s3Key", s3Key, "status", "processing");
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Could not read upload", e);
        } catch (Exception e) {
            log.error("Backend upload failed for user={} video={}", user.getId(), videoId, e);
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Could not upload video to object storage", e);
        }
    }

    @PostMapping("/multipart/initiate")
    public Map<String, String> initiateMultipartUpload(
            @RequestParam(defaultValue = "general") String profile,
            @RequestParam(required = false, defaultValue = "video.mp4") String fileName,
            @RequestParam(required = false, defaultValue = "video/mp4") String contentType,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String videoId = UUID.randomUUID().toString();
        String s3Key = rawVideoKey(user.getId(), videoId);
        ObjectMetadata metadata = new ObjectMetadata();
        metadata.setContentType(contentType == null || contentType.isBlank() ? "video/mp4" : contentType);

        InitiateMultipartUploadRequest request = new InitiateMultipartUploadRequest(bucket, s3Key)
                .withObjectMetadata(metadata);
        if (user.getKmsKeyArn() != null && !user.getKmsKeyArn().isBlank()) {
            request.withSSEAwsKeyManagementParams(new SSEAwsKeyManagementParams(user.getKmsKeyArn()));
        }
        InitiateMultipartUploadResult result = amazonS3.initiateMultipartUpload(request);
        return Map.of(
                "videoId", videoId,
                "s3Key", s3Key,
                "uploadId", result.getUploadId(),
                "partSize", String.valueOf(8L * 1024L * 1024L),
                "expiresIn", "600",
                "profile", profile,
                "fileName", fileName);
    }

    @GetMapping("/multipart/{videoId}/part")
    public Map<String, String> presignMultipartPart(
            @PathVariable String videoId,
            @RequestParam String s3Key,
            @RequestParam String uploadId,
            @RequestParam int partNumber,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        validateOwnedKey(user, videoId, s3Key);

        Date expiry = new Date(System.currentTimeMillis() + 10L * 60L * 1000L);
        GeneratePresignedUrlRequest request = new GeneratePresignedUrlRequest(bucket, s3Key)
                .withMethod(HttpMethod.PUT)
                .withExpiration(expiry);
        request.addRequestParameter("partNumber", String.valueOf(partNumber));
        request.addRequestParameter("uploadId", uploadId);
        URL url = amazonS3.generatePresignedUrl(request);
        return Map.of("url", url.toString(), "partNumber", String.valueOf(partNumber), "expiresIn", "600");
    }

    @PostMapping("/multipart/{videoId}/complete")
    public Map<String, String> completeMultipartUpload(
            @PathVariable String videoId,
            @RequestBody Map<String, Object> body,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        String s3Key = String.valueOf(body.get("s3Key"));
        String uploadId = String.valueOf(body.get("uploadId"));
        String profile = String.valueOf(body.getOrDefault("profile", "general"));
        String sourceFile = String.valueOf(body.getOrDefault("sourceFile", "video.mp4"));
        String benchmarkRunId = String.valueOf(body.getOrDefault("benchmarkRunId", ""));
        String fewShotLabel = String.valueOf(body.getOrDefault("fewShotLabel", ""));
        String domainId = String.valueOf(body.getOrDefault("domainId", "general"));
        validateOwnedKey(user, videoId, s3Key);

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> partsBody = (List<Map<String, Object>>) body.get("parts");
        if (partsBody == null || partsBody.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "parts required");
        }
        List<PartETag> parts = new ArrayList<>();
        for (Map<String, Object> part : partsBody) {
            int partNumber = ((Number) part.get("partNumber")).intValue();
            String eTag = String.valueOf(part.get("eTag")).replace("\"", "");
            parts.add(new PartETag(partNumber, eTag));
        }
        parts.sort(java.util.Comparator.comparingInt(PartETag::getPartNumber));
        amazonS3.completeMultipartUpload(new CompleteMultipartUploadRequest(bucket, s3Key, uploadId, parts));
        publishUploadedEvent(videoId, user.getId(), profile, s3Key, sourceFile, benchmarkRunId, fewShotLabel, domainId);
        return Map.of("videoId", videoId, "s3Key", s3Key, "status", "processing");
    }

    @DeleteMapping("/multipart/{videoId}")
    public Map<String, String> abortMultipartUpload(
            @PathVariable String videoId,
            @RequestParam String s3Key,
            @RequestParam String uploadId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        validateOwnedKey(user, videoId, s3Key);

        amazonS3.abortMultipartUpload(new AbortMultipartUploadRequest(bucket, s3Key, uploadId));
        return Map.of("videoId", videoId, "status", "aborted");
    }

    @PostMapping("/{videoId}/finalize")
    public Map<String, String> finalize(
            @PathVariable String videoId,
            @RequestParam(defaultValue = "general") String profile,
            @RequestParam String s3Key,
            @RequestParam(required = false, defaultValue = "video.mp4") String sourceFile,
            @RequestParam(required = false, defaultValue = "") String benchmarkRunId,
            @RequestParam(required = false, defaultValue = "") String fewShotLabel,
            @RequestParam(required = false, defaultValue = "general") String domainId,
            @AuthenticationPrincipal OAuth2User oAuth2User) {

        if (oAuth2User == null) throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        User user = userRepository.findByGoogleSub(oAuth2User.getAttribute("sub"))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));

        validateOwnedKey(user, videoId, s3Key);
        publishUploadedEvent(videoId, user.getId(), profile, s3Key, sourceFile, benchmarkRunId, fewShotLabel, domainId);

        log.info("Published video.uploaded for video={} user={}", videoId, user.getId());
        return Map.of("videoId", videoId, "status", "processing");
    }

    private String rawVideoKey(String userId, String videoId) {
        return "raw/default/" + userId + "/" + videoId + "/original.mp4";
    }

    private void validateOwnedKey(User user, String videoId, String s3Key) {
        String expected = rawVideoKey(user.getId(), videoId);
        if (!expected.equals(s3Key)) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Invalid upload key");
        }
    }

    private void publishUploadedEvent(String videoId, String userId, String profile, String s3Key, String sourceFile, String benchmarkRunId, String fewShotLabel, String domainId) {
        RStream<String, String> stream = redissonClient.getStream("pipeline.events", StringCodec.INSTANCE);
        Map<String, String> entries = new java.util.LinkedHashMap<>();
        String normalizedFewShotLabel = fewShotLearningService.normalizeLabel(fewShotLabel);
        String normalizedDomainId = domainModelService.normalizeDomainId(domainId);
        entries.put("type", "video.uploaded");
        entries.put("video_id", videoId);
        entries.put("user_id", userId);
        entries.put("tenant_id", "default");
        entries.put("domain_id", normalizedDomainId);
        entries.put("profile", profile);
        entries.put("s3_raw_path", s3Key);
        entries.put("source_file", sourceFile == null || sourceFile.isBlank() ? "video.mp4" : sourceFile);
        if (benchmarkRunId != null && !benchmarkRunId.isBlank()) {
            entries.put("benchmark_run_id", benchmarkRunId);
        }
        if (!normalizedFewShotLabel.isBlank()) {
            entries.put("few_shot_example", "true");
            entries.put("few_shot_label", normalizedFewShotLabel);
            entries.put("few_shot_model_id", fewShotLearningService.modelId(userId, normalizedDomainId));
            fewShotLearningService.recordExample(userId, normalizedDomainId, normalizedFewShotLabel, videoId, sourceFile, s3Key);
            domainModelService.recordExample(userId, normalizedDomainId, normalizedFewShotLabel, videoId, sourceFile, s3Key);
        }
        stream.add(StreamAddArgs.entries(entries));
    }
}
