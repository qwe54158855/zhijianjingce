package com.wafer.controller;

import com.wafer.model.dto.GalleryItemDTO;
import com.wafer.service.GalleryService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class GalleryController {

    private final GalleryService galleryService;

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "UP", "service", "wafer-backend");
    }

    @GetMapping("/gallery")
    public ResponseEntity<Page<GalleryItemDTO>> getGallery(
            @RequestParam(required = false) String category,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(galleryService.getGallery(category, page, size));
    }

    @GetMapping("/gallery/{id}")
    public ResponseEntity<GalleryItemDTO> getById(@PathVariable Long id) {
        return galleryService.getById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/gallery/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        return ResponseEntity.ok(galleryService.getStats());
    }
}
