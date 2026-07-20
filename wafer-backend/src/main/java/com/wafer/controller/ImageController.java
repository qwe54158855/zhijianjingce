package com.wafer.controller;

import com.wafer.service.StorageService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/images")
@RequiredArgsConstructor
public class ImageController {

    private final StorageService storageService;

    @PostMapping("/upload")
    public ResponseEntity<Map<String, String>> upload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(defaultValue = "uploads") String dir) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "文件为空"));
        }
        String path = storageService.upload(file, StorageService.generatePath(dir));
        String url = storageService.getUrl(path);
        return ResponseEntity.ok(Map.of("path", path, "url", url));
    }

    @GetMapping("/**")
    public void getImage(HttpServletRequest request, HttpServletResponse response) {
        // Extract path from request: /api/v1/images/uploads/2026/07/10/xxx.jpg
        String requestURI = request.getRequestURI();
        String prefix = "/api/v1/images/";
        String path = requestURI.substring(requestURI.indexOf(prefix) + prefix.length());
        // Proxying MinIO images; for simplicity, return presigned URL redirect
        response.setStatus(302);
        response.setHeader("Location", storageService.getUrl(path));
    }
}
