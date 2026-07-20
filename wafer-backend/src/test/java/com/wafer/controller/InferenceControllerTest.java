package com.wafer.controller;

import com.wafer.model.dto.InferenceResponse;
import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import com.wafer.service.InferenceService;
import com.wafer.service.SseService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Optional;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(InferenceController.class)
class InferenceControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private InferenceService inferenceService;

    @MockBean
    private SseService sseService;

    @Test
    void submit_shouldReturn202() throws Exception {
        InferenceResponse mockResponse = InferenceResponse.builder()
                .taskId(1L)
                .type(InferenceType.ENHANCE)
                .status(TaskStatus.PENDING)
                .build();

        given(inferenceService.submitTask(
                any(org.springframework.web.multipart.MultipartFile.class),
                eq(InferenceType.ENHANCE),
                isNull()
        )).willReturn(mockResponse);

        MockMultipartFile file = new MockMultipartFile(
                "file", "test.png", "image/png", "test-image-content".getBytes());

        mockMvc.perform(multipart("/api/v1/inference")
                        .file(file)
                        .param("type", "enhance"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.taskId").value(1L))
                .andExpect(jsonPath("$.status").value("PENDING"));
    }

    @Test
    void submit_shouldReturn400ForInvalidType() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file", "test.png", "image/png", "test-image-content".getBytes());

        mockMvc.perform(multipart("/api/v1/inference")
                        .file(file)
                        .param("type", "invalid_type"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void getTask_shouldReturn200WhenFound() throws Exception {
        InferenceResponse mockResponse = InferenceResponse.builder()
                .taskId(1L)
                .type(InferenceType.ENHANCE)
                .status(TaskStatus.DONE)
                .build();

        given(inferenceService.getTask(1L)).willReturn(Optional.of(mockResponse));

        mockMvc.perform(get("/api/v1/inference/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.taskId").value(1L))
                .andExpect(jsonPath("$.status").value("DONE"));
    }

    @Test
    void getTask_shouldReturn404WhenNotFound() throws Exception {
        given(inferenceService.getTask(999L)).willReturn(Optional.empty());

        mockMvc.perform(get("/api/v1/inference/999"))
                .andExpect(status().isNotFound());
    }

    @Test
    void streamTask_shouldReturn404WhenTaskNotFound() throws Exception {
        given(inferenceService.getTask(999L)).willReturn(Optional.empty());

        mockMvc.perform(get("/api/v1/inference/999/stream"))
                .andExpect(status().isNotFound());
    }

    @Test
    void streamTask_shouldReturn200AndEmitterForRunningTask() throws Exception {
        InferenceResponse mockResponse = InferenceResponse.builder()
                .taskId(1L)
                .type(InferenceType.ENHANCE)
                .status(TaskStatus.RUNNING)
                .build();

        given(inferenceService.getTask(1L)).willReturn(Optional.of(mockResponse));
        given(sseService.createEmitter(1L)).willReturn(new SseEmitter(300_000L));

        mockMvc.perform(get("/api/v1/inference/1/stream")
                        .accept(MediaType.TEXT_EVENT_STREAM))
                .andExpect(status().isOk())
                .andExpect(request().asyncStarted());
    }

    @Test
    void streamTask_shouldReturnImmediateCompleteForDoneTask() throws Exception {
        InferenceResponse mockResponse = InferenceResponse.builder()
                .taskId(1L)
                .type(InferenceType.ENHANCE)
                .status(TaskStatus.DONE)
                .inputUrl("http://example.com/input.png")
                .build();

        given(inferenceService.getTask(1L)).willReturn(Optional.of(mockResponse));

        mockMvc.perform(get("/api/v1/inference/1/stream")
                        .accept(MediaType.TEXT_EVENT_STREAM))
                .andExpect(status().isOk());
    }

    @Test
    void streamTask_shouldReturnImmediateErrorForFailedTask() throws Exception {
        InferenceResponse mockResponse = InferenceResponse.builder()
                .taskId(1L)
                .type(InferenceType.ENHANCE)
                .status(TaskStatus.FAILED)
                .errorMessage("inference failed")
                .build();

        given(inferenceService.getTask(1L)).willReturn(Optional.of(mockResponse));

        mockMvc.perform(get("/api/v1/inference/1/stream")
                        .accept(MediaType.TEXT_EVENT_STREAM))
                .andExpect(status().isOk());
    }
}
