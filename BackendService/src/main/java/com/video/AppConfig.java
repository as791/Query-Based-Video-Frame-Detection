package com.video;
import com.amazonaws.services.s3.transfer.TransferManager;
import com.amazonaws.services.s3.transfer.TransferManagerBuilder;
import org.redisson.Redisson;
import org.redisson.api.RedissonClient;
import org.redisson.codec.JsonJacksonCodec;
import org.redisson.config.Config;
import org.redisson.spring.cache.RedissonSpringCacheManager;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;

import lombok.Getter;
import lombok.Setter;
import org.springframework.context.annotation.Primary;
import org.springframework.data.mongodb.MongoDatabaseFactory;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.SimpleMongoClientDatabaseFactory;
import org.springframework.data.mongodb.core.convert.DefaultDbRefResolver;
import org.springframework.data.mongodb.core.convert.MappingMongoConverter;
import org.springframework.data.mongodb.core.convert.MongoConverter;
import org.springframework.data.mongodb.core.mapping.MongoMappingContext;


@Configuration
@ConfigurationProperties(prefix = "aws")
@Getter
@Setter
public class AppConfig {
    private String accessKey;
    private String secretKey;
    private String region;
    private int maxConnections;
    private int connectionTimeOut;

    @Value("${spring.data.mongodb.uri}")
    private String mongoConnectionUri;

    @Value("${spring.redis.uri}")
    private String redisConnectionUri;


    @Bean
    public AmazonS3 sClient(){
      return AmazonS3ClientBuilder.standard()
            .withClientConfiguration(new ClientConfiguration().withMaxConnections(maxConnections)
                    .withConnectionTimeout(connectionTimeOut))
            .withRegion(region)
            .withCredentials(new AWSStaticCredentialsProvider(new BasicAWSCredentials(accessKey, secretKey)))
            .build();
    }

    @Bean
    public TransferManager transferManager(AmazonS3 amazonS3Client) {
        return TransferManagerBuilder.standard().withS3Client(amazonS3Client).build();
    }

    @Bean
    MongoDatabaseFactory mongoDbFactory() {
        return new SimpleMongoClientDatabaseFactory(mongoConnectionUri);
    }

    @Bean
    MongoConverter mongoConverter(MongoDatabaseFactory mongoDatabaseFactory) {
        MongoMappingContext mongoMappingContext = new MongoMappingContext();
        mongoMappingContext.setAutoIndexCreation(false);
        MappingMongoConverter mongoConverter = new MappingMongoConverter(
            new DefaultDbRefResolver(mongoDatabaseFactory), mongoMappingContext);
        mongoConverter.setMapKeyDotReplacement("-DOT-");
        return mongoConverter;
    }

    @Bean
    MongoTemplate mongoTemplate(MongoDatabaseFactory mongoDatabaseFactory, MongoConverter converter) {
        return new MongoTemplate(mongoDatabaseFactory, converter);
    }


    @Bean(destroyMethod = "shutdown")
    public RedissonClient redissonClient() {
        Config config = new Config();
        config.setCodec(new JsonJacksonCodec());
        config.useSingleServer().setAddress(redisConnectionUri);
        return Redisson.create(config);
    }

    @Bean
    @Primary
    public RedissonSpringCacheManager cacheManager(RedissonClient redissonClient) {
        return new RedissonSpringCacheManager(redissonClient);
    }

}
