#!/usr/bin/env python3
"""
Generates all publication-quality plots reproducing and extending the figures in:
"Parallel and Distributed Deep Learning" (Hegde & Usmani, Stanford University)

Figures generated in results/figures/:
- figure1_conv_time_vs_batchsize.png: Reproduces Figure 1 (CPU vs GPU forward-prop time on Conv layer vs batch size)
- figure2_conv_speedup_vs_batchsize.png: Reproduces Figure 2 (CPU to GPU speedup ratio vs batch size)
- figure3_fc_time_vs_matrixsize.png: Reproduces Figure 3 (CPU vs GPU time for matrix-vector / FC multiplication vs matrix size)
- figure4_fc_speedup_vs_matrixsize.png: Reproduces Figure 4 (CPU to GPU speedup ratio vs matrix size)
- figure5_distributed_convergence.png: Empirical convergence (Accuracy & Logistic Loss) of Algorithm 1 (ParallelSGD with & without ShuffleSGD), Algorithm 3 (ADMM.SGD), and Algorithm 4 (DownpourSGD)
- figure6_communication_and_scaling.png: Communication Cost (MB) & DownpourSGD Worker Scaling / Saturation (P = T / tau)
- figure7_multicore_multigpu_speedup.png: Multi-Core Xeon (1..16 cores) vs 1x GPU (GTX 1080 Ti) vs 2x GPU (GTX 1080 Ti + GTX 1080)
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_rows(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs("results/figures", exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    # ------------------------------------------------------------------------
    # 1. Figures 1 & 2: Convolution Layer CPU vs GPU Time & Speedup
    # ------------------------------------------------------------------------
    conv_rows = read_csv_rows("results/logs/conv_benchmark.csv")
    bs = [int(r["batch_size"]) for r in conv_rows]
    cpu_single = [float(r["cpu_single_ms"]) for r in conv_rows]
    cpu_omp = [float(r["cpu_omp_ms"]) for r in conv_rows]
    gpu_ms = [float(r["gpu_ms"]) for r in conv_rows]
    speedup_single = [float(r["speedup_single_vs_gpu"]) for r in conv_rows]
    speedup_omp = [float(r["speedup_omp_vs_gpu"]) for r in conv_rows]

    # Fit linear slopes (O(n) verification as in Section 3.2.1)
    slope_cpu = np.polyfit(bs, cpu_single, 1)[0]
    slope_gpu = np.polyfit(bs, gpu_ms, 1)[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bs, cpu_single, "x-", color="#1f77b4", linewidth=2, markersize=7,
            label=f"CPU (1-Core) [slope={slope_cpu:.2f} ms/img]")
    ax.plot(bs, cpu_omp, "^--", color="#2ca02c", linewidth=1.8, markersize=6,
            label="CPU (16-Core OpenMP)")
    ax.plot(bs, gpu_ms, "o-", color="#d62728", linewidth=2, markersize=6,
            label=f"GPU (CUDA im2col+GEMM) [slope={slope_gpu:.4f} ms/img]")
    ax.set_title("Figure 1: Computational Time vs Batch Size (Convolution Layer)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/figure1_conv_time_vs_batchsize.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bs, speedup_single, "x-", color="#1f77b4", linewidth=2, markersize=7,
            label="Ratio: Single-Core CPU / GPU")
    ax.plot(bs, speedup_omp, "s--", color="#ff7f0e", linewidth=1.8, markersize=6,
            label="Ratio: 16-Core OpenMP CPU / GPU")
    ax.set_title("Figure 2: Ratio of CPU to GPU Time vs Batch Size (Conv Layer)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup Ratio (CPU / GPU)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/figure2_conv_speedup_vs_batchsize.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 2. Figures 3 & 4: Fully Connected Layer (Matrix-Vector Multiplication O(N^2))
    # ------------------------------------------------------------------------
    fc_rows = read_csv_rows("results/logs/fc_benchmark.csv")
    msize = [int(r["matrix_size"]) for r in fc_rows]
    fc_cpu = [float(r["cpu_gemv_ms"]) for r in fc_rows]
    fc_gpu = [float(r["gpu_gemv_ms"]) for r in fc_rows]
    fc_ratio = [float(r["speedup_gemv"]) for r in fc_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(msize, fc_cpu, "o-", color="#d62728", linewidth=2, markersize=6, label="CPU (O(n²))")
    ax.plot(msize, fc_gpu, "x-", color="#1f77b4", linewidth=2, markersize=7, label="GPU (CUDA O(n²))")
    ax.set_title("Figure 3: Computational Time vs Matrix Size (Fully Connected Layer)")
    ax.set_xlabel("Matrix Size (N × N)")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/figure3_fc_time_vs_matrixsize.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(msize, fc_ratio, "x-", color="#1f77b4", linewidth=2, markersize=7, label="CPU / GPU Speedup Ratio")
    ax.set_title("Figure 4: Ratio of CPU to GPU Computation Time vs Matrix Size")
    ax.set_xlabel("Matrix Size (N × N)")
    ax.set_ylabel("Speedup Ratio (CPU / GPU)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/figure4_fc_speedup_vs_matrixsize.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 3. Figure 5: Distributed Algorithms Convergence (Section 5)
    # ------------------------------------------------------------------------
    with open("results/logs/distributed_algorithms_comparison.json", "r") as f:
        dist_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    styles = {
        "ParallelSGD_NoShuffle": ("v--", "#7f7f7f", "ParallelSGD (No Shuffle - Sorted Non-IID)"),
        "ParallelSGD_ShuffleSGD": ("o-", "#1f77b4", "Alg 1+2: ParallelSGD + ShuffleSGD"),
        "ADMM_SGD": ("s-", "#2ca02c", "Alg 3: ADMM.SGD (Sync Consensus)"),
        "Downpour_SGD": ("^-", "#d62728", "Alg 4: DownpourSGD (Async ParamServer)"),
    }

    for key, (marker, color, label) in styles.items():
        hist = dist_data[key]
        epochs = [h["epoch"] for h in hist]
        losses = [h["val_loss"] for h in hist]
        accs = [h["val_acc"] for h in hist]
        ax1.plot(epochs, losses, marker, color=color, linewidth=2, markersize=6, label=label)
        ax2.plot(epochs, accs, marker, color=color, linewidth=2, markersize=6, label=label)

    ax1.set_title("Validation Logistic Loss (Eq. 2) vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Logistic Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.set_title("Validation Classification Accuracy (40 Classes) vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("results/figures/figure5_distributed_convergence.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 4. Figure 6: Communication Cost & DownpourSGD Worker Saturation (P = T / tau)
    # ------------------------------------------------------------------------
    scale_rows = read_csv_rows("results/logs/scaling_analysis.csv")
    k_vals = [int(r["k"]) for r in scale_rows]
    psgd_mb = [float(r["psgd_comm_mb"]) for r in scale_rows]
    admm_mb = [float(r["admm_comm_mb"]) for r in scale_rows]
    down_mb = [float(r["downpour_comm_mb"]) for r in scale_rows]
    down_ups = [float(r["downpour_updates_per_sec"]) for r in scale_rows]
    p_sat = float(scale_rows[0]["P_saturation_workers"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(k_vals, psgd_mb, "o-", color="#1f77b4", linewidth=2, label="ParallelSGD + ShuffleSGD: O(D + kp)")
    ax1.plot(k_vals, admm_mb, "s-", color="#2ca02c", linewidth=2, label="ADMM.SGD: O(kp log(1/ε))")
    ax1.plot(k_vals, down_mb, "^-", color="#d62728", linewidth=2, label="DownpourSGD: O(θp)")
    ax1.set_title("Total Communication Volume vs Workers (k)")
    ax1.set_xlabel("Number of Machines / Workers (k)")
    ax1.set_ylabel("Communication Volume (MB)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(k_vals, down_ups, "D-", color="#9467bd", linewidth=2, markersize=6,
             label="DownpourSGD Effective Throughput")
    ax2.axvline(x=p_sat, color="#d62728", linestyle="--", linewidth=1.8,
                label=f"Saturation Threshold P = T/τ = {p_sat:.0f} workers")
    ax2.set_title("DownpourSGD Parameter Server Bottleneck (Section 5.2.1)")
    ax2.set_xlabel("Number of Asynchronous Workers (k)")
    ax2.set_ylabel("Parameter Updates / Second")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("results/figures/figure6_communication_and_scaling.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 5. Figure 7: Multi-Core CPU (1..16 cores) vs 1x & 2x GPU Speedup on HPC
    # ------------------------------------------------------------------------
    if os.path.exists("results/logs/multicore_multigpu_benchmark.csv"):
        mc_rows = read_csv_rows("results/logs/multicore_multigpu_benchmark.csv")
        labels = [r["device"].replace("_", " ") for r in mc_rows]
        speedups = [float(r["speedup_vs_1core"]) for r in mc_rows]
        colors = ["#1f77b4" if "CPU" in l else "#d62728" for l in labels]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, speedups, color=colors, edgecolor="black", alpha=0.85)
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                    f"{sp:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title("Section 2: Empirical Speedup Across Multi-Core CPU (1-16 Cores) & Multi-GPU on HPC")
        ax.set_ylabel("Speedup vs Single-Core CPU (×)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig("results/figures/figure7_multicore_multigpu_speedup.png")
        plt.close(fig)

    print("All figures generated successfully in results/figures/")


if __name__ == "__main__":
    main()
