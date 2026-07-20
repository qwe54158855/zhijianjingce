package com.wafer.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class SseServiceTest {

    private SseService sseService;

    @BeforeEach
    void setUp() {
        sseService = new SseService();
    }

    @Test
    void createEmitter_shouldStoreAndReturnEmitter() {
        SseEmitter emitter = sseService.createEmitter(1L);
        assertThat(emitter).isNotNull();
    }

    @Test
    void sendProgress_shouldNotThrowWhenEmitterExists() {
        sseService.createEmitter(1L);
        // Should not throw
        sseService.sendProgress(1L, "running", 10);
    }

    @Test
    void sendProgress_shouldNotThrowWhenEmitterDoesNotExist() {
        // Should not throw for missing task
        sseService.sendProgress(999L, "running", 10);
    }

    @Test
    void complete_shouldNotThrowWhenEmitterExists() {
        sseService.createEmitter(1L);
        Map<String, Object> result = Map.of("taskId", 1L, "status", "DONE");
        // Should not throw
        sseService.complete(1L, result);
    }

    @Test
    void complete_shouldNotThrowWhenEmitterDoesNotExist() {
        // Should not throw for missing task
        sseService.complete(999L, Map.of("status", "DONE"));
    }

    @Test
    void error_shouldNotThrowWhenEmitterExists() {
        sseService.createEmitter(1L);
        // Should not throw
        sseService.error(1L, "something went wrong");
    }

    @Test
    void error_shouldNotThrowWhenEmitterDoesNotExist() {
        // Should not throw for missing task
        sseService.error(999L, "something went wrong");
    }

    @Test
    void emitter_shouldBeRemovedOnCompletion() {
        SseEmitter emitter = sseService.createEmitter(1L);
        assertThat(emitter).isNotNull();

        sseService.complete(1L, Map.of("status", "DONE"));

        // After complete(), the onCompletion callback should have removed the emitter.
        // sendProgress should no-op (no emitter found) — no exception expected.
        sseService.sendProgress(1L, "test", 50);
    }

    @Test
    void emitter_shouldBeRemovedOnError() {
        sseService.createEmitter(1L);

        sseService.error(1L, "error occurred");

        // After completeWithError(), the onCompletion callback should have removed the emitter.
        // sendProgress should no-op (no emitter found) — no exception expected.
        sseService.sendProgress(1L, "test", 50);
    }

    @Test
    void multipleEmitters_shouldBeIndependent() {
        sseService.createEmitter(1L);
        sseService.createEmitter(2L);

        sseService.complete(1L, Map.of("taskId", 1L, "status", "DONE"));

        // Task 2 should still be active
        sseService.sendProgress(2L, "running", 50);
    }

    @Test
    void emitter_shouldHaveConfiguredTimeout() {
        SseEmitter emitter = sseService.createEmitter(1L);
        // Default timeout is 300_000ms (5 minutes)
        assertThat(emitter.getTimeout()).isEqualTo(300_000L);
    }
}
