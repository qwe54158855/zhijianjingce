package com.wafer.service;

import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class SseService {

    private final Map<Long, SseEmitter> emitters = new ConcurrentHashMap<>();

    public SseEmitter createEmitter(Long taskId) {
        SseEmitter emitter = new SseEmitter(300_000L); // 5 min timeout
        emitters.put(taskId, emitter);
        emitter.onCompletion(() -> emitters.remove(taskId));
        emitter.onTimeout(() -> emitters.remove(taskId));
        return emitter;
    }

    public void sendProgress(Long taskId, String stage, int progress) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("progress")
                        .data(Map.of("stage", stage, "progress", progress)));
            } catch (Exception e) {
                emitters.remove(taskId);
            }
        }
    }

    public void complete(Long taskId, Object result) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("complete")
                        .data(result));
                emitter.complete();
            } catch (Exception e) {
                // ignore — onCompletion callback already removes the emitter
            }
        }
    }

    public void error(Long taskId, String error) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("error")
                        .data(Map.of("error", error)));
                emitter.completeWithError(new RuntimeException(error));
            } catch (Exception e) {
                // ignore — onCompletion callback already removes the emitter
            }
        }
    }
}
