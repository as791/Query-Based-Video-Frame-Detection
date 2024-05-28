package com.video;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Date;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;

import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.GeneratePresignedUrlRequest;
import com.amazonaws.services.s3.model.ObjectListing;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectSummary;
import com.amazonaws.services.s3.model.S3ObjectInputStream;
import com.amazonaws.services.s3.transfer.TransferManager;
import com.amazonaws.services.s3.transfer.TransferManagerBuilder;
import com.amazonaws.services.s3.transfer.Upload;
import com.amazonaws.services.s3.transfer.Transfer.TransferState;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class AwsWrapperService {
    @Autowired
    private AmazonS3 amazonS3Client;

    @Autowired
    private TransferManager transferManager;

    public void uploadFileToS3(File file, String bucketName, String key) {
        Upload upload = transferManager.upload(bucketName, key, file);
        try {
            upload.waitForUploadResult();
        } catch (Exception e) {
            log.error("Error uploading file to S3", e);
        }
    }

    public File downloadFileFromS3(String bucketName, String key, File file) {
        S3Object s3object = amazonS3Client.getObject(bucketName, key);
        S3ObjectInputStream inputStream = s3object.getObjectContent();

        try (FileOutputStream fos = new FileOutputStream(file)) {
            byte[] read_buf = new byte[1024];
            int read_len = 0;
            while ((read_len = inputStream.read(read_buf)) > 0) {
                fos.write(read_buf, 0, read_len);
            }
        } catch (IOException e) {
            log.error("Error downloading file from S3", e);
        }
        return file;
    }

    public List<S3ObjectSummary> s3ListFilesUsingPrefix(String bucketName, String prefix) {
        ObjectListing objectListing = amazonS3Client.listObjects(bucketName, prefix);
        return objectListing.getObjectSummaries();
    }

    public URL generatePresignedUrl(String bucketName, String key, int expirationHour){
        Date expiration = new Date();
        long expTimeMillis = expiration.getTime();
        expTimeMillis += 1000 * 60 * expirationHour; // Add expirationHour hour
        expiration.setTime(expTimeMillis);
        GeneratePresignedUrlRequest generatePresignedUrlRequest = new GeneratePresignedUrlRequest(bucketName, key)
                                                                    .withExpiration(expiration);
        return amazonS3Client.generatePresignedUrl(generatePresignedUrlRequest);
    }
}
