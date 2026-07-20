package com.wafer.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class GalleryItemDTO {
    private Long id;
    private String title;
    private String description;
    private String category;
    private String[] tags;
    private String originalUrl;
    private String resultUrl;
    private String diffUrl;
    private String thumbnailUrl;
    private String metrics;
    private Integer displayOrder;
}
