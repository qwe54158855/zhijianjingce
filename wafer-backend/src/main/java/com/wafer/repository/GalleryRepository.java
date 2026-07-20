package com.wafer.repository;

import com.wafer.model.entity.GalleryItem;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface GalleryRepository extends JpaRepository<GalleryItem, Long> {
    Page<GalleryItem> findByCategory(String category, Pageable pageable);
    List<GalleryItem> findAllByOrderByDisplayOrderAsc();
}
