package com.wafer.service;

import io.minio.*;
import io.minio.http.Method;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDate;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class StorageService {

    @Autowired(required = false)
    private MinioClient minioClient;

    @Autowired(required = false)
    private com.wafer.config.MinIOConfig minioConfig;

    @Value("${storage.local.path:./data/uploads}")
    private String localBasePath;

    private boolean useMinio = true;

    @PostConstruct
    public void init() {
        if (minioClient == null || minioConfig == null) {
            useMinio = false;
            try {
                Files.createDirectories(Path.of(localBasePath));
                log.info("StorageService: using local filesystem at {}", Path.of(localBasePath).toAbsolutePath());
            } catch (Exception e) {
                log.error("Failed to create local storage directory", e);
            }
            return;
        }
        try {
            String bucket = minioConfig.getBucket();
            boolean exists = minioClient.bucketExists(
                    BucketExistsArgs.builder().bucket(bucket).build());
            if (!exists) {
                minioClient.makeBucket(
                        MakeBucketArgs.builder().bucket(bucket).build());
                log.info("Created MinIO bucket: {}", bucket);
            }
            log.info("StorageService: using MinIO at {}", minioConfig.getBucket());
        } catch (Exception e) {
            log.warn("MinIO unavailable, falling back to local filesystem", e);
            useMinio = false;
            try {
                Files.createDirectories(Path.of(localBasePath));
            } catch (Exception ex) {
                log.error("Failed to create local storage directory", ex);
            }
        }
    }

    public String upload(MultipartFile file, String dir) {
        if (useMinio) {
            return uploadToMinio(file, dir);
        }
        return uploadToLocal(file, dir);
    }

    private String uploadToMinio(MultipartFile file, String dir) {
        try {
            String filename = UUID.randomUUID().toString() + "_" + file.getOriginalFilename();
            String objectName = dir + "/" + filename;
            minioClient.putObject(
                    PutObjectArgs.builder()
                            .bucket(minioConfig.getBucket())
                            .object(objectName)
                            .stream(file.getInputStream(), file.getSize(), -1)
                            .contentType(file.getContentType())
                            .build());
            log.info("Uploaded to MinIO: {}", objectName);
            return objectName;
        } catch (Exception e) {
            throw new RuntimeException("MinIO upload failed", e);
        }
    }

    private String uploadToLocal(MultipartFile file, String dir) {
        try {
            String filename = UUID.randomUUID().toString() + "_" + file.getOriginalFilename();
            String relativePath = dir + "/" + filename;
            Path targetPath = Path.of(localBasePath, relativePath);
            Files.createDirectories(targetPath.getParent());
            Files.copy(file.getInputStream(), targetPath, StandardCopyOption.REPLACE_EXISTING);
            log.info("Stored locally: {}", targetPath);
            return relativePath;
        } catch (Exception e) {
            throw new RuntimeException("Local storage upload failed", e);
        }
    }

    public String getUrl(String objectName) {
        if (useMinio && minioClient != null) {
            try {
                return minioClient.getPresignedObjectUrl(
                        GetPresignedObjectUrlArgs.builder()
                                .bucket(minioConfig.getBucket())
                                .object(objectName)
                                .method(Method.GET)
                                .expiry(24, TimeUnit.HOURS)
                                .build());
            } catch (Exception e) {
                log.warn("Failed to get presigned URL, fallback to path: {}", objectName);
            }
        }
        return "/api/v1/images/" + objectName;
    }

    public InputStream download(String objectName) {
        if (useMinio && minioClient != null) {
            try {
                return minioClient.getObject(
                        GetObjectArgs.builder()
                                .bucket(minioConfig.getBucket())
                                .object(objectName)
                                .build());
            } catch (Exception e) {
                throw new RuntimeException("MinIO download failed: " + objectName, e);
            }
        }
        try {
            return Files.newInputStream(Path.of(localBasePath, objectName));
        } catch (Exception e) {
            throw new RuntimeException("Local download failed: " + objectName, e);
        }
    }

    public void delete(String objectName) {
        if (useMinio && minioClient != null) {
            try {
                minioClient.removeObject(
                        RemoveObjectArgs.builder()
                                .bucket(minioConfig.getBucket())
                                .object(objectName)
                                .build());
                log.info("Deleted from MinIO: {}", objectName);
                return;
            } catch (Exception e) {
                log.error("Failed to delete from MinIO: {}", objectName, e);
            }
        }
        try {
            Files.deleteIfExists(Path.of(localBasePath, objectName));
        } catch (Exception e) {
            log.error("Failed to delete local: {}", objectName, e);
        }
    }

    public static String generatePath(String prefix) {
        return prefix + "/" + LocalDate.now().toString().replace("-", "/");
    }
}
