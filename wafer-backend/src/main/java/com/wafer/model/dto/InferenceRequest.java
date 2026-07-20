package com.wafer.model.dto;

import com.wafer.model.enums.InferenceType;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InferenceRequest {
    @NotNull
    private InferenceType type;

    private String params;  // JSON string: {strength: 0.75, steps: 20}
}
