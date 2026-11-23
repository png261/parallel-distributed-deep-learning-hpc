#!/usr/bin/env python3
"""
Direction 2: Heterogeneous Multi-GPU Load Balancing
Evaluates asymmetric compute capabilities of:
- GPU 0: NVIDIA GeForce GTX 1080 Ti (3584 CUDA cores, 11 GB VRAM)
- GPU 1: NVIDIA GeForce GTX 1080 (2560 CUDA cores, 8 GB VRAM)

Benchmarks a total mini-batch of 64 images across different batch allocation splits (B0, B1):
- Uniform 50-50 allocation (32, 32): GPU 0 finishes early and is forced to idle waiting for GPU 1.
- Proportional / Optimal allocation (B0 / B1 ~ 38..42 / 26..22): Balances execution times,
  minimizes GPU idle time, and maximizes cluster training throughput (samples/sec).
"""

import os
import csv
import ctypes


def main():
    os.makedirs("results/logs", exist_ok=True)
    print("==========================================================")
    print("Direction 2: Heterogeneous Multi-GPU Load Balancing")
    print("Hardware: GPU 0 (GTX 1080 Ti, 3584 cores) vs GPU 1 (GTX 1080, 2560 cores)")
    print("==========================================================")

    lib_path = os.path.abspath("build/libgpu_cnn.so")
    if not os.path.exists(lib_path):
        print(f"Error: {lib_path} not found. Please compile libgpu_cnn.so first.")
        return

    lib = ctypes.CDLL(lib_path)
    lib.benchmark_heterogeneous_multigpu.argtypes = [
        ctypes.c_int,  # batch_0
        ctypes.c_int,  # batch_1
        ctypes.c_int,  # num_runs
        ctypes.POINTER(ctypes.c_double),  # out_gpu0_ms
        ctypes.POINTER(ctypes.c_double),  # out_gpu1_ms
        ctypes.POINTER(ctypes.c_double),  # out_total_ms
        ctypes.POINTER(ctypes.c_double),  # out_gpu0_idle_ms
    ]

    total_batch = 64
    splits = [
        (16, 48),
        (24, 40),
        (32, 32),  # Uniform Baseline
        (36, 28),
        (40, 24),  # Near theoretical core ratio 3584 / (3584+2560) ~ 58.3% (37.3 ~ 38..40)
        (44, 20),
        (48, 16),
    ]

    num_runs = 25
    rows = []

    print(f"\nBenchmarking Total Batch Size = {total_batch} (Average over {num_runs} runs):")
    print(f"{'Split (B0/B1)':<15}{'GPU 0 (1080Ti)':<18}{'GPU 1 (1080)':<18}{'Total Step (ms)':<18}{'GPU 0 Idle (ms)':<18}{'Throughput (img/s)':<18}")
    print("-" * 105)

    for b0, b1 in splits:
        c_gpu0 = ctypes.c_double()
        c_gpu1 = ctypes.c_double()
        c_tot = ctypes.c_double()
        c_idle = ctypes.c_double()

        lib.benchmark_heterogeneous_multigpu(
            b0, b1, num_runs,
            ctypes.byref(c_gpu0),
            ctypes.byref(c_gpu1),
            ctypes.byref(c_tot),
            ctypes.byref(c_idle)
        )

        gpu0_ms = c_gpu0.value
        gpu1_ms = c_gpu1.value
        tot_ms = c_tot.value
        idle_ms = c_idle.value
        throughput = (total_batch / (tot_ms / 1000.0))

        ratio_pct = (b0 / total_batch) * 100.0
        label = f"{b0}/{b1} ({ratio_pct:.0f}%)"

        print(f"{label:<15}{gpu0_ms:<18.2f}{gpu1_ms:<18.2f}{tot_ms:<18.2f}{idle_ms:<18.2f}{throughput:<18.1f}")

        rows.append({
            "split_label": label,
            "batch_gpu0": b0,
            "batch_gpu1": b1,
            "gpu0_pct": ratio_pct,
            "gpu0_time_ms": gpu0_ms,
            "gpu1_time_ms": gpu1_ms,
            "total_step_ms": tot_ms,
            "gpu0_idle_ms": idle_ms,
            "throughput_img_per_sec": throughput,
        })

    with open("results/logs/heterogeneous_gpu_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Find optimal split
    best_row = max(rows, key=lambda x: x["throughput_img_per_sec"])
    uniform_row = [r for r in rows if r["batch_gpu0"] == 32][0]
    gain_pct = ((best_row["throughput_img_per_sec"] - uniform_row["throughput_img_per_sec"]) / uniform_row["throughput_img_per_sec"]) * 100.0

    print("-" * 105)
    print(f"Optimal Split: {best_row['split_label']} -> Throughput: {best_row['throughput_img_per_sec']:.1f} img/s")
    print(f"Uniform 50-50 Split: Throughput: {uniform_row['throughput_img_per_sec']:.1f} img/s (GPU 0 Idle Time: {uniform_row['gpu0_idle_ms']:.2f} ms)")
    print(f"Heterogeneous Load Balancing Throughput Gain: +{gain_pct:.1f}%")
    print("\nSaved Direction 2 results to results/logs/heterogeneous_gpu_benchmark.csv!")


if __name__ == "__main__":
    main()
