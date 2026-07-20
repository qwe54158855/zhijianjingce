package com.wafer.service;

import com.wafer.model.dto.GalleryItemDTO;
import com.wafer.model.entity.GalleryItem;
import com.wafer.repository.GalleryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class GalleryService {

    private final GalleryRepository galleryRepository;

    public Page<GalleryItemDTO> getGallery(String category, int page, int size) {
        PageRequest pr = PageRequest.of(page, size, Sort.by(Sort.Direction.ASC, "displayOrder"));
        Page<GalleryItem> items;

        if (category != null && !category.isEmpty()) {
            items = galleryRepository.findByCategory(category, pr);
        } else {
            items = galleryRepository.findAll(pr);
        }

        return items.map(this::toDTO);
    }

    public Optional<GalleryItemDTO> getById(Long id) {
        return galleryRepository.findById(id).map(this::toDTO);
    }

    public Map<String, Object> getStats() {
        long total = galleryRepository.count();
        return Map.of("totalItems", total);
    }

    public GalleryItemDTO create(GalleryItem item) {
        GalleryItem saved = galleryRepository.save(item);
        return toDTO(saved);
    }

    private GalleryItemDTO toDTO(GalleryItem item) {
        return GalleryItemDTO.builder()
                .id(item.getId())
                .title(item.getTitle())
                .description(item.getDescription())
                .category(item.getCategory())
                .tags(item.getTags() != null ? item.getTags().replaceAll("[\\[\\]\" ]", "").split(",") : new String[0])
                .originalUrl(item.getOriginalUrl())
                .resultUrl(item.getResultUrl())
                .diffUrl(item.getDiffUrl())
                .thumbnailUrl(item.getThumbnailUrl())
                .metrics(item.getMetrics())
                .displayOrder(item.getDisplayOrder())
                .build();
    }
}
