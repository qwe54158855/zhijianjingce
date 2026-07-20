package com.wafer.service;

import com.wafer.client.LoraInferenceClient;
import com.wafer.model.dto.InferenceResponse;
import com.wafer.model.dto.LoraInferRequest;
import com.wafer.model.dto.LoraInferResponse;
import com.wafer.model.entity.InferenceTask;
import com.wafer.model.enums.TaskStatus;
import com.wafer.repository.InferenceTaskRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.time.LocalDateTime;
import java.util.Base64;

@Slf4j
@Service
@RequiredArgsConstructor
public class InferenceProcessor {

    private final InferenceTaskRepository taskRepository;
    private final LoraInferenceClient loraClient;
    private final SseService sseService;
    private final CacheService cacheService;
    private final StorageService storageService;
    private final ObjectMapper objectMapper;

    /**
     * Async inference processing — called via AOP proxy so @Async works correctly.
     */
    @Async("taskExecutor")
    public void processInference(Long taskId) {
        InferenceTask task = taskRepository.findById(taskId)
                .orElseThrow(() -> new RuntimeException("Task not found: " + taskId));

        try {
            // Update status
            task.setStatus(TaskStatus.RUNNING);
            taskRepository.save(task);
            sseService.sendProgress(taskId, "running", 10);

            // Build LoRA inference request
            double strength = parseStrength(task.getParams());
            LoraInferRequest loraReq = LoraInferRequest.builder()
                    .mode(task.getType().name().toLowerCase())
                    .imageBase64(encodeImageToBase64(task.getInputUrl()))
                    .strength(strength)
                    .build();

            sseService.sendProgress(taskId, "inferring", 50);

            // Call FastAPI
            LoraInferResponse loraResp = loraClient.infer(loraReq);

            sseService.sendProgress(taskId, "saving", 80);

            // Save result to cache
            cacheService.cacheTaskResult(taskId, loraResp);

            // Update task
            task.setStatus(TaskStatus.DONE);
            task.setDurationMs(loraResp.getDurationMs());
            task.setCompletedAt(LocalDateTime.now());
            taskRepository.save(task);

            sseService.complete(taskId, toResponse(task));
            log.info("Task {} completed in {}ms", taskId, loraResp.getDurationMs());

        } catch (Exception e) {
            log.error("Task {} failed", taskId, e);
            task.setStatus(TaskStatus.FAILED);
            task.setErrorMessage(e.getMessage());
            task.setCompletedAt(LocalDateTime.now());
            taskRepository.save(task);
            sseService.error(taskId, e.getMessage());
        }
    }

    /**
     * Download image from MinIO and encode to Base64 string.
     */
    private String encodeImageToBase64(String objectPath) {
        try (InputStream is = storageService.download(objectPath)) {
            byte[] bytes = is.readAllBytes();
            return Base64.getEncoder().encodeToString(bytes);
        } catch (Exception e) {
            log.error("Failed to encode image to Base64: {}", objectPath, e);
            throw new RuntimeException("Image encoding failed for: " + objectPath, e);
        }
    }

    private double parseStrength(String paramsJson) {
        if (paramsJson == null || paramsJson.isBlank()) {
            return 0.75;
        }
        try {
            JsonNode root = objectMapper.readTree(paramsJson);
            JsonNode strengthNode = root.get("strength");
            if (strengthNode != null && strengthNode.isNumber()) {
                double val = strengthNode.asDouble();
                if (val >= 0.1 && val <= 1.0) {
                    return val;
                }
            }
        } catch (JsonProcessingException e) {
            log.warn("Failed to parse params JSON '{}', falling back to default strength", paramsJson);
        }
        return 0.75;
    }

    private InferenceResponse toResponse(InferenceTask task) {
        return InferenceResponse.builder()
                .taskId(task.getId())
                .type(task.getType())
                .status(task.getStatus())
                .inputUrl(task.getInputUrl() != null ? storageService.getUrl(task.getInputUrl()) : null)
                .outputUrl(task.getOutputUrl())
                .thumbnailUrl(task.getThumbnailUrl())
                .durationMs(task.getDurationMs())
                .errorMessage(task.getErrorMessage())
                .build();
    }
}
