package com.video;


import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@AllArgsConstructor
@Getter
@Setter
@ToString
@Document("videoMetadata")
public class VideoMetadata {
    private long size;
    @Id
    private String videoId;
    private S3Path s3Path;
}
