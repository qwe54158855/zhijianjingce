package com.wafer.client;

import com.wafer.model.dto.LoraInferRequest;
import com.wafer.model.dto.LoraInferResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.Map;

@FeignClient(
    name = "lora-service",
    url = "${lora.service.url}",
    path = "/api/v1/lora"
)
public interface LoraInferenceClient {

    @PostMapping("/infer")
    LoraInferResponse infer(@RequestBody LoraInferRequest request);

    @PostMapping("/switch")
    void switchLora(@RequestBody Map<String, Object> request);
}
