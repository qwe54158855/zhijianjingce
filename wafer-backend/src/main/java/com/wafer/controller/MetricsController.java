package com.wafer.controller;

import com.wafer.model.enums.TaskStatus;
import com.wafer.repository.InferenceTaskRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/metrics")
@RequiredArgsConstructor
public class MetricsController {

    private final InferenceTaskRepository taskRepository;

    @GetMapping("/overview")
    public ResponseEntity<Map<String, Object>> overview() {
        return ResponseEntity.ok(Map.of(
            "totalTasks", taskRepository.count(),
            "pendingTasks", taskRepository.countByStatus(TaskStatus.PENDING),
            "runningTasks", taskRepository.countByStatus(TaskStatus.RUNNING),
            "doneTasks", taskRepository.countByStatus(TaskStatus.DONE),
            "failedTasks", taskRepository.countByStatus(TaskStatus.FAILED)
        ));
    }
}
