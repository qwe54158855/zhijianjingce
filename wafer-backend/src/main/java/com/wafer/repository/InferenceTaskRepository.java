package com.wafer.repository;

import com.wafer.model.entity.InferenceTask;
import com.wafer.model.enums.TaskStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface InferenceTaskRepository extends JpaRepository<InferenceTask, Long> {
    List<InferenceTask> findByStatusOrderByCreatedAtDesc(TaskStatus status);
    long countByStatus(TaskStatus status);
}
