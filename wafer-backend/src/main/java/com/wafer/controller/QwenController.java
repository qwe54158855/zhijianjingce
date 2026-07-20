package com.wafer.controller;

import com.wafer.client.QwenInferenceClient;
import com.wafer.model.dto.QwenEnhanceRequest;
import com.wafer.model.dto.QwenEnhanceResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/qwen")
@RequiredArgsConstructor
public class QwenController {

    private final QwenInferenceClient qwenClient;

    @PostMapping("/enhance")
    public ResponseEntity<QwenEnhanceResponse> enhance(
            @RequestBody QwenEnhanceRequest request) {
        QwenEnhanceResponse response = qwenClient.enhance(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/report")
    public ResponseEntity<Map<String, Object>> report(
            @RequestBody Map<String, Object> request) {
        Map<String, Object> response = qwenClient.report(request);
        return ResponseEntity.ok(response);
    }
}
