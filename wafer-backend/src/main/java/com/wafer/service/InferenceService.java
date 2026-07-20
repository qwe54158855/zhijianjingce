package com.wafer.service;

import com.wafer.model.dto.InferenceResponse;
import com.wafer.model.entity.InferenceTask;
import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import com.wafer.repository.InferenceTaskRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class InferenceService {

    private final InferenceTaskRepository taskRepository;
    private final StorageService storageService;
    private final InferenceProcessor inferenceProcessor;

    /**
     * Submit a new inference task (synchronous, returns taskId immediately).
     */
    public InferenceResponse submitTask(MultipartFile file, InferenceType type, String paramsJson) {
        // Upload file to MinIO, store the object path (not the presigned URL)
        String path = storageService.upload(file, StorageService.generatePath("uploads"));

        // Create task record
        InferenceTask task = InferenceTask.builder()
                .type(type)
                .status(TaskStatus.PENDING)
                .inputUrl(path)  // store MinIO object path for downstream processing
                .params(paramsJson)
                .build();
        task = taskRepository.save(task);

        // Delegate to the separate processor bean so @Async works via AOP proxy
        inferenceProcessor.processInference(task.getId());

        return toResponse(task);
    }

    /**
     * Get task status and result.
     */
    public Optional<InferenceResponse> getTask(Long taskId) {
        return taskRepository.findById(taskId).map(this::toResponse);
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
