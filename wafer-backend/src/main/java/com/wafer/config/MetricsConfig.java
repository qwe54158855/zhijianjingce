package com.wafer.config;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MetricsConfig {

    @Bean
    public Counter inferenceCounter(MeterRegistry registry) {
        return Counter.builder("wafer.inference.total")
                .description("Total inference requests")
                .register(registry);
    }

    @Bean
    public Timer inferenceTimer(MeterRegistry registry) {
        return Timer.builder("wafer.inference.duration")
                .description("Inference duration")
                .register(registry);
    }
}
