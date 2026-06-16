package com.video;

import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.amazonaws.services.kms.AWSKMS;
import com.amazonaws.services.kms.AWSKMSClientBuilder;
import com.amazonaws.services.kms.model.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class KmsService {

    @Value("${kms.key-alias-prefix:alias/vvault/user/}")
    private String keyAliasPrefix;

    @Value("${aws.region:us-east-1}")
    private String region;

    // Dedicated KMS credentials — separate from MinIO's fake AWS_ACCESS_KEY_ID.
    // Leave blank in local dev to fall through to DefaultAWSCredentialsProviderChain
    // (IAM role, ~/.aws/credentials, etc.)
    @Value("${KMS_AWS_ACCESS_KEY_ID:}")
    private String kmsAccessKey;

    @Value("${KMS_AWS_SECRET_ACCESS_KEY:}")
    private String kmsSecretKey;

    private AWSKMS buildClient() {
        AWSKMSClientBuilder builder = AWSKMSClientBuilder.standard().withRegion(region);
        if (!kmsAccessKey.isBlank() && !kmsSecretKey.isBlank()) {
            builder.withCredentials(new AWSStaticCredentialsProvider(
                    new BasicAWSCredentials(kmsAccessKey, kmsSecretKey)));
        } else {
            builder.withCredentials(new DefaultAWSCredentialsProviderChain());
        }
        return builder.build();
    }

    /**
     * Provisions a per-user CMK in AWS KMS.
     * Returns a KmsKeyResult containing the key ID and ARN, both stored in the DB.
     * Idempotent: if the alias already exists the existing key is returned.
     */
    public KmsKeyResult provisionUserKey(String userId) {
        AWSKMS kms = buildClient();
        String alias = keyAliasPrefix + userId;

        // Idempotency: check if alias already exists
        try {
            DescribeKeyResult existing = kms.describeKey(new DescribeKeyRequest().withKeyId(alias));
            KeyMetadata meta = existing.getKeyMetadata();
            log.info("KMS key already exists for user {}: keyId={}", userId, meta.getKeyId());
            return new KmsKeyResult(meta.getKeyId(), meta.getArn());
        } catch (NotFoundException ignored) {
            // Key doesn't exist yet — create it
        }

        // Create the CMK
        CreateKeyResult created = kms.createKey(new CreateKeyRequest()
                .withDescription("Per-user encryption key — userId=" + userId)
                .withKeyUsage(KeyUsageType.ENCRYPT_DECRYPT)
                .withTags(
                        new Tag().withTagKey("app").withTagValue("vvault"),
                        new Tag().withTagKey("userId").withTagValue(userId)));

        KeyMetadata meta = created.getKeyMetadata();
        String keyId = meta.getKeyId();
        String keyArn = meta.getArn();

        // Create a human-readable alias for lookup
        kms.createAlias(new CreateAliasRequest()
                .withAliasName(alias)
                .withTargetKeyId(keyId));

        log.info("Provisioned KMS CMK for user {}: keyId={} alias={}", userId, keyId, alias);
        return new KmsKeyResult(keyId, keyArn);
    }

    public record KmsKeyResult(String keyId, String keyArn) {}
}
