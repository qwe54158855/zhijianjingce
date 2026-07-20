"""
参数与推理时间基准测试

测量:
  1. 总参数量
  2. 各模块参数分布
  3. 推理时间分解
  4. 检查是否满足 10M / 80ms 约束
"""

import os
import sys
import torch
import time
import numpy as np

# Ensure project root is on sys.path for standalone runs
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.wafer_multitask import WaferMultiTaskModel


def count_parameters(model):
    """统计各模块参数量"""
    table = []
    total = 0

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total += params
        if params > 0:
            table.append((name, params, params / sum(
                p.numel() for p in model.parameters()) * 100))

    return table, total


def measure_inference_time(model, input_tensor, num_warmup=10, num_iterations=100):
    """测量推理时间"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            model(input_tensor)
            times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'mean': np.mean(times),
        'p50': np.percentile(times, 50),
        'p99': np.percentile(times, 99),
        'min': np.min(times),
        'max': np.max(times),
    }


def main():
    print("=" * 60)
    print("晶圆检测模型 — 参数与推理时间基准测试")
    print("=" * 60)
    print()

    # === 参数测量 ===
    print("参数测量")
    print("-" * 40)

    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    table, total = count_parameters(model)

    for name, params, pct in table:
        print(f"  {name:20s}  {params/1e6:.3f}M  ({pct:5.1f}%)")
    print("-" * 40)
    passed_param = total < 10e6
    print(f"  {'Total':20s}  {total/1e6:.3f}M  (10M limit: {'PASS' if passed_param else 'FAIL'})")
    print(f"  {'Headroom':20s}  {(1 - total/10e6)*100:.1f}%")
    print()

    # === 推理时间测量 ===
    print("Inference Time Measurement")
    print("-" * 40)

    input_tensor = torch.randn(1, 1, 512, 512)
    stats = measure_inference_time(model, input_tensor)

    print(f"  Mean: {stats['mean']:.1f} ms")
    print(f"  P50:  {stats['p50']:.1f} ms")
    print(f"  P99:  {stats['p99']:.1f} ms")
    print(f"  Min:  {stats['min']:.1f} ms")
    print(f"  Max:  {stats['max']:.1f} ms")
    passed_time = stats['mean'] < 80
    print(f"  80ms constraint: {'PASS' if passed_time else 'FAIL'}")
    print()

    # === 降级方案对比 ===
    print("Ablation Config Comparison")
    print("-" * 40)

    configs = [
        ('Full (PISM+ASG 13-angle)', True, True),
        ('No ASG (dual wavelength)', True, False),
        ('No physics (original single)', False, False),
    ]

    for name, enable_physics, enable_polar in configs:
        m = WaferMultiTaskModel(in_channels=1, num_classes=4,
                                 enable_physics=enable_physics,
                                 enable_polar=enable_polar)
        params = sum(p.numel() for p in m.parameters())
        t = measure_inference_time(m, input_tensor)
        print(f"  {name:30s}  {params/1e6:.2f}M  {t['mean']:.1f}ms")

    print()
    print("=" * 60)
    print("Benchmark Complete")
    print("=" * 60)

    # Return pass/fail status for programmatic use
    return passed_param and passed_time


if __name__ == '__main__':
    main()
