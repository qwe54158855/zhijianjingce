package com.wafer.model.dto;

import lombok.Data;
import java.util.List;
import java.util.Map;

@Data
public class QwenEnhanceResponse {
    private boolean success;
    private String enhancedImage;
    private List<Map<String, Object>> detections;
    private Map<String, Object> analysis;
    private int inferenceTimeMs;
}
