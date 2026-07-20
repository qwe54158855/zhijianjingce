package com.wafer.model.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "gallery_item", indexes = {
    @Index(name = "idx_gallery_category", columnList = "category"),
    @Index(name = "idx_gallery_order", columnList = "displayOrder")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class GalleryItem {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 200)
    private String title;

    @Column(columnDefinition = "TEXT")
    private String description;

    @Column(nullable = false, length = 50)
    private String category;

    @Column(columnDefinition = "JSONB DEFAULT '[]'")
    private String tags;

    @Column(name = "original_url", nullable = false, length = 500)
    private String originalUrl;

    @Column(name = "result_url", nullable = false, length = 500)
    private String resultUrl;

    @Column(name = "diff_url", length = 500)
    private String diffUrl;

    @Column(name = "thumbnail_url", length = 500)
    private String thumbnailUrl;

    @Column(columnDefinition = "JSONB DEFAULT '{}'")
    private String metrics;

    @Column(name = "display_order")
    @Builder.Default
    private Integer displayOrder = 0;

    @Column(name = "created_at", updatable = false)
    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "updated_at")
    @Builder.Default
    private LocalDateTime updatedAt = LocalDateTime.now();

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
