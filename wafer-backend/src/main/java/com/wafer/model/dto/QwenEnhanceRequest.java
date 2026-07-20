package com.wafer.model.dto;

import lombok.Data;

@Data
public class QwenEnhanceRequest {
    private String image;
    private String format = "jpg";
}
