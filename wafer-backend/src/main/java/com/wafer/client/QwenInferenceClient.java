package com.wafer.client;

import com.wafer.model.dto.QwenEnhanceRequest;
import com.wafer.model.dto.QwenEnhanceResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.Map;

@FeignClient(
    name = "qwen-service",
    url = "${qwen.service.url}",
    path = "/api/v1/qwen"
)
public interface QwenInferenceClient {

    @PostMapping("/enhance")
    QwenEnhanceResponse enhance(@RequestBody QwenEnhanceRequest request);

    @PostMapping("/report")
    Map<String, Object> report(@RequestBody Map<String, Object> request);
}
