package com.video;

import static org.mockito.Mockito.reset;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.multipart.MultipartFile;

import com.amazonaws.services.s3.model.GeneratePresignedUrlRequest;
import com.amazonaws.services.s3.model.S3ObjectSummary;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class VideoService {

  @Autowired
  private AwsWrapperService s3Client;
  @Autowired
  private VideoMetadataRepository videoMetadataRepository;
  @Autowired
  private RestTemplateWrapper restTemplateWrapper;
  private String bucketName = "stage-video-bucket";
  @Value("${searchServiceUrl}")
  private String searchServiceUrl;

  public VideoMetadata uploadRecordedVideo(MultipartFile file) {
    log.info("File upload started!!");
    String videoId = UUID.randomUUID().toString();
    String fileName = "/tmp/" + videoId + ".mp4";
    try (InputStream inputStream = file.getInputStream();
        FileOutputStream outputStream = new FileOutputStream(fileName, true)) {
      byte[] buffer = new byte[1024];
      int bytesRead;
      while ((bytesRead = inputStream.read(buffer)) != -1) {
        outputStream.write(buffer, 0, bytesRead);
      }
    } catch (IOException e) {
      log.error("Upload failed.", e);
    }
    String s3Key = Util.getS3Key(fileName);
    s3Client.uploadFileToS3(new File(fileName), bucketName, s3Key);
    VideoMetadata videoMetadata = new VideoMetadata(file.getSize(), videoId,
        new S3Path(bucketName, s3Key));
    videoMetadataRepository.insert(videoMetadata);
    log.info("File upload completed!!");
    return videoMetadata;
  }

  @Cacheable(cacheNames = "searchOnVideo", key = "#seachRequestDTO")
  public SearchOutputLinks searchOnVideo(SeachRequestDTO seachRequestDTO) {
    log.info("searching for frame inside video with nearest match started!!");
    List<S3ObjectSummary> s3ObjectSummaries = new ArrayList<>();
    for (long time = seachRequestDTO.getStartTime(); time < seachRequestDTO.getEndTime(); time = time + 60 * 1000) {
      String datePrefix = Util.getPrefixKeyForSearch(time);
      s3ObjectSummaries.addAll(s3Client.s3ListFilesUsingPrefix(bucketName, datePrefix));
    }
    List<String> s3Paths = s3ObjectSummaries.stream()
        .map(s3ObjectSummary -> "s3://" + s3ObjectSummary.getBucketName() + "/" + s3ObjectSummary.getKey())
        .collect(Collectors.toList());

    List<SearchOutput> searchOutputs = new ArrayList<>();

    for (String s3Path : s3Paths) {
      SearchRequest searchRequest = new SearchRequest(s3Path, seachRequestDTO.getQuery());
      HttpEntity<SearchRequest> requestEntity = new HttpEntity<>(searchRequest);
      try {
        ResponseEntity<SearchOutput> responseEntity = restTemplateWrapper.run(searchServiceUrl,
            HttpMethod.POST, requestEntity, SearchOutput.class);
        if (responseEntity.hasBody() && responseEntity.getBody() != null) {
          searchOutputs.add(responseEntity.getBody());
        }
      } catch (RestClientException e) {
        log.error("error while getting respective frame with ex:", e);
      }
    }
    log.info("searching for frame inside video with nearest match completed!!");
    return convertToS3PresignUrls(searchOutputs);
  }

  private SearchOutputLinks convertToS3PresignUrls(List<SearchOutput> searchOutputs) {
    return new SearchOutputLinks(searchOutputs.stream()
        .map(out -> s3Client.generatePresignedUrl(out.getFrame().getBucket(), out.getFrame().getKey(),1).toString())
        .toList());
  }

}
