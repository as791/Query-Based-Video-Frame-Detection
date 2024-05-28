package com.video;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface VideoMetadataRepository extends MongoRepository<VideoMetadata, String> {
}
