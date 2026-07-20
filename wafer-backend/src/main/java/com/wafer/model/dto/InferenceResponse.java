package com.wafer.model.dto;

import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InferenceResponse {
    private Long taskId;
    private InferenceType type;
    private TaskStatus status;
    private String inputUrl;
    private String outputUrl;
    private String thumbnailUrl;
    private Integer durationMs;
    private String errorMessage;
}
