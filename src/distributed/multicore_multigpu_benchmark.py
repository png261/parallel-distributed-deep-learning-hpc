#!/usr/bin/env python3
"""
Multi-GPU (GTX 1080 Ti + GTX 1080 via native sm_61 CUDA C++ library) and
Multi-Core CPU (1..16 Xeon cores) Parallel Benchmark on HPC.
Implements Section 2 of "Parallel and Distributed Deep Learning".
"""

import os
import time
import csv
import ctypes
import torch
from distributed_sgd_algorithms import PaperCNN40, logistic_loss_eq2


def benchmark_multicore_and_multigpu():
    print("==========================================================")
    print("Section 2: Multi-Core CPU (1..16 cores) & Multi-GPU Scaling")
    print("==========================================================")
    os.makedirs("results/logs", exist_ok=True)

    x_batch = torch.randn(64, 1, 224, 224)
    y_batch = torch.randint(0, 40, (64,))

    core_counts = [1, 2, 4, 8, 12, 16]
    rows = []

    for cores in core_counts:
        torch.set_num_threads(cores)
        model = PaperCNN40(num_classes=40)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Warmup
        scores = model(x_batch[:16])
        loss = logistic_loss_eq2(scores, y_batch[:16])
        loss.backward()

        t0 = time.perf_counter()
        runs = 5
        for _ in range(runs):
            optimizer.zero_grad()
            scores = model(x_batch)
            loss = logistic_loss_eq2(scores, y_batch)
            loss.backward()
            optimizer.step()
        elapsed_ms = ((time.perf_counter() - t0) / runs) * 1000.0
        speedup = rows[0]["time_ms"] / elapsed_ms if rows else 1.0
        print(f"  CPU Threads: {cores:2d} | Minibatch (64) Fwd+Bwd Time: {elapsed_ms:8.2f} ms | Speedup: {speedup:5.2f}x")
        rows.append({
            "device": f"CPU_{cores}_cores",
            "cores_or_gpus": cores,
            "time_ms": elapsed_ms,
            "speedup_vs_1core": speedup,
        })

    # Benchmark 1x GPU (GTX 1080 Ti) and 2x GPU (GTX 1080 Ti + GTX 1080) using compiled native sm_61 CUDA kernel library
    lib_path = os.path.abspath("build/libgpu_cnn.so")
    if os.path.exists(lib_path):
        lib = ctypes.CDLL(lib_path)
        lib.benchmark_cnn_step_multigpu.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.benchmark_cnn_step_multigpu.restype = ctypes.c_double

        for g in [1, 2]:
            elapsed_ms = float(lib.benchmark_cnn_step_multigpu(64, g, 20))
            speedup = rows[0]["time_ms"] / elapsed_ms
            gpu_label = "GPU_1x_GTX1080Ti" if g == 1 else "GPU_2x_DataParallel"
            print(f"  GPU DataParallel ({g}x):   Minibatch (64) Time: {elapsed_ms:8.2f} ms | Speedup vs 1-Core CPU: {speedup:6.2f}x")
            rows.append({
                "device": gpu_label,
                "cores_or_gpus": g,
                "time_ms": elapsed_ms,
                "speedup_vs_1core": speedup,
            })

        # Benchmark Dual-GPU Model Parallelism (Pipeline Parallelism across GPU 0 & GPU 1)
        if hasattr(lib, "benchmark_model_parallelism_multigpu"):
            lib.benchmark_model_parallelism_multigpu.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.benchmark_model_parallelism_multigpu.restype = ctypes.c_double
            mp_ms = float(lib.benchmark_model_parallelism_multigpu(64, 20))
            mp_speedup = rows[0]["time_ms"] / mp_ms
            print(f"  GPU ModelParallel (2x):   Minibatch (64) Time: {mp_ms:8.2f} ms | Speedup vs 1-Core CPU: {mp_speedup:6.2f}x")
            rows.append({
                "device": "GPU_2x_ModelParallel",
                "cores_or_gpus": 2,
                "time_ms": mp_ms,
                "speedup_vs_1core": mp_speedup,
            })

    with open("results/logs/multicore_multigpu_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["device", "cores_or_gpus", "time_ms", "speedup_vs_1core"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    benchmark_multicore_and_multigpu()
