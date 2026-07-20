package com.wafer.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LoraInferRequest {
    @JsonProperty("image_base64")
    private String imageBase64;
    private String mode;         // enhance / wavelength / defect
    private Double strength;     // 0.1 ~ 1.0
    private String prompt;
    @JsonProperty("control_image_base64")
    private String controlImageBase64;
}
