package com.wafer.controller;

import com.wafer.model.dto.InferenceResponse;
import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import com.wafer.service.InferenceService;
import com.wafer.service.SseService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/inference")
@RequiredArgsConstructor
public class InferenceController {

    private final InferenceService inferenceService;
    private final SseService sseService;

    @PostMapping
    public ResponseEntity<InferenceResponse> submit(
            @RequestParam("file") MultipartFile file,
            @RequestParam("type") String type,
            @RequestParam(value = "params", required = false) String params) {

        InferenceType inferenceType;
        try {
            inferenceType = InferenceType.valueOf(type.toUpperCase());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }

        InferenceResponse response = inferenceService.submitTask(file, inferenceType, params);
        return ResponseEntity.accepted().body(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<InferenceResponse> getTask(@PathVariable Long id) {
        return inferenceService.getTask(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping(value = "/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public ResponseEntity<SseEmitter> streamTask(@PathVariable Long id) {
        // Check task existence before creating SSE emitter
        var taskOpt = inferenceService.getTask(id);
        if (taskOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        var response = taskOpt.get();
        // If task is already terminal, return response directly rather than a long-lived emitter
        if (response.getStatus() == TaskStatus.DONE
                || response.getStatus() == TaskStatus.FAILED) {
            SseEmitter immediate = new SseEmitter(0L);
            try {
                if (response.getStatus() == TaskStatus.DONE) {
                    immediate.send(SseEmitter.event().name("complete").data(response));
                } else {
                    immediate.send(SseEmitter.event().name("error").data(
                            Map.of("error", response.getErrorMessage())));
                }
            } catch (IOException e) {
                // swallow
            }
            immediate.complete();
            return ResponseEntity.ok(immediate);
        }
        SseEmitter emitter = sseService.createEmitter(id);
        return ResponseEntity.ok(emitter);
    }

    @PostMapping("/batch")
    public ResponseEntity<Map<String, Object>> batch(
            @RequestBody Map<String, Object> request) {
        // Batch inference for gallery pre-generation
        // Processes multiple images with specified LoRA mode
        return ResponseEntity.ok(Map.of("status", "not implemented"));
    }
}
