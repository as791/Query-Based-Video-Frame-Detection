package com.video;


import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;


@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class SearchOutput {
    @JsonProperty("frame")
    private S3Path frame;
    @JsonProperty("similarityScore")
    private Double similarityScore;
    @JsonProperty("caption")
    private String caption;
}
