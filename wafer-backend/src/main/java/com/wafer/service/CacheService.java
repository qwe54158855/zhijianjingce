package com.wafer.service;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class CacheService {

    @Autowired(required = false)
    private RedisTemplate<String, Object> redisTemplate;

    private final Map<String, Object> localCache = new ConcurrentHashMap<>();
    private boolean useRedis = true;

    @PostConstruct
    public void init() {
        if (redisTemplate == null) {
            useRedis = false;
            log.info("CacheService: using in-memory cache (Redis unavailable)");
        } else {
            log.info("CacheService: using Redis cache");
        }
    }

    public void set(String key, Object value, long ttlSeconds) {
        if (useRedis && redisTemplate != null) {
            try {
                redisTemplate.opsForValue().set(key, value, ttlSeconds, TimeUnit.SECONDS);
                return;
            } catch (Exception e) {
                log.warn("Redis set failed, falling back to local cache", e);
            }
        }
        localCache.put(key, value);
    }

    public Object get(String key) {
        if (useRedis && redisTemplate != null) {
            try {
                Object val = redisTemplate.opsForValue().get(key);
                if (val != null) return val;
            } catch (Exception e) {
                log.warn("Redis get failed, falling back to local cache", e);
            }
        }
        return localCache.get(key);
    }

    public void delete(String key) {
        if (useRedis && redisTemplate != null) {
            try {
                redisTemplate.delete(key);
                return;
            } catch (Exception e) {
                log.warn("Redis delete failed", e);
            }
        }
        localCache.remove(key);
    }

    public boolean hasKey(String key) {
        if (useRedis && redisTemplate != null) {
            try {
                return Boolean.TRUE.equals(redisTemplate.hasKey(key));
            } catch (Exception e) {
                log.warn("Redis hasKey failed", e);
            }
        }
        return localCache.containsKey(key);
    }

    public void cacheTaskResult(Long taskId, Object result) {
        set("task:result:" + taskId, result, 3600);
    }

    public Object getTaskResult(Long taskId) {
        return get("task:result:" + taskId);
    }
}
