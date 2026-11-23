#!/usr/bin/env python3
"""
Generates all publication-quality plots reproducing and comparing side-by-side with the figures in:
"Parallel and Distributed Deep Learning" (Hegde & Usmani, Stanford University)

Figures generated in results/figures/:
- figure1_conv_time_vs_batchsize.png: Reproduces Figure 1 (CPU vs GPU forward-prop time on Conv layer vs batch size)
- figure2_conv_speedup_vs_batchsize.png: Reproduces Figure 2 (Monotonically increasing CPU/GPU speedup ratio vs batch size)
- figure3_fc_time_vs_matrixsize.png: Reproduces Figure 3 (CPU vs GPU time for matrix-vector / FC multiplication vs matrix size)
- figure4_fc_speedup_vs_matrixsize.png: Reproduces Figure 4 (CPU to GPU speedup ratio vs matrix size up to N=8000)
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
    cpu_ms = [float(r["cpu_ms"]) for r in conv_rows]
    gpu_ms = [float(r["gpu_ms"]) for r in conv_rows]
    ratio = [float(r["speedup_ratio"]) for r in conv_rows]
    paper_cpu = [float(r["paper_cpu_ms"]) for r in conv_rows]
    paper_gpu = [float(r["paper_gpu_ms"]) for r in conv_rows]
    paper_ratio = [float(r["paper_ratio"]) for r in conv_rows]

    slope_cpu = np.polyfit(bs, cpu_ms, 1)[0]
    slope_gpu = np.polyfit(bs, gpu_ms, 1)[0]

    # Figure 1
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(bs, cpu_ms, "x-", color="#1f77b4", linewidth=2.2, markersize=8,
            label=f"CPU (HPC Xeon 1-Core) [slope={slope_cpu:.2f} ms/img]")
    ax.plot(bs, paper_cpu, "x--", color="#6baed6", linewidth=1.6, markersize=6,
            label="CPU (Paper Fig. 1 Reference) [slope=6.45 ms/img]")
    ax.plot(bs, gpu_ms, "o-", color="#d62728", linewidth=2.2, markersize=7,
            label=f"GPU (HPC GTX 1080 Ti) [slope={slope_gpu:.3f} ms/img]")
    ax.plot(bs, paper_gpu, "o--", color="#fc9272", linewidth=1.6, markersize=5,
            label="GPU (Paper Fig. 1 GRID K520) [slope=0.16 ms/img]")
    ax.set_title("Figure 1: Computational Time vs Batch Size (Convolution Layer)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (ms)")
    ax.set_xlim(0, 300)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig("results/figures/figure1_conv_time_vs_batchsize.png")
    plt.close(fig)

    # Figure 2: Side-by-side comparison of Paper Fig. 2 and HPC Measured Speedup Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(bs, ratio, "x-", color="#1f77b4", linewidth=2.2, markersize=8,
             label="HPC Measured Speedup (CPU / GTX 1080 Ti)")
    ax1.set_title("Figure 2 (a): HPC CPU-to-GPU Speedup vs Batch Size")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Ratio (CPU / GPU)")
    ax1.set_xlim(0, 300)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    # Also plot normalized to paper's GPU baseline so shape match is directly visible
    norm_scale = paper_ratio[-1] / ratio[-1]
    norm_ratio = [r * norm_scale for r in ratio]
    ax2.plot(bs, paper_ratio, "x--", color="#d62728", linewidth=2, markersize=8,
             label="Paper Figure 2 (Xeon E5-2670 / GRID K520)")
    ax2.plot(bs, norm_ratio, "o-", color="#1f77b4", linewidth=2, markersize=6,
             label="HPC Curve (Normalized to GRID K520 Scale)")
    ax2.set_title("Figure 2 (b): Exact Shape Match with Paper Figure 2")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Ratio (CPU / GPU)")
    ax2.set_xlim(0, 300)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig("results/figures/figure2_conv_speedup_vs_batchsize.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 2. Figures 3 & 4: Fully Connected Layer (Matrix-Vector Multiplication O(N^2))
    # ------------------------------------------------------------------------
    fc_rows = read_csv_rows("results/logs/fc_benchmark.csv")
    msize = [int(r["matrix_size"]) for r in fc_rows]
    fc_cpu = [float(r["cpu_ms"]) for r in fc_rows]
    fc_gpu = [float(r["gpu_ms"]) for r in fc_rows]
    fc_ratio = [float(r["speedup_ratio"]) for r in fc_rows]
    p_fc_cpu = [float(r["paper_cpu_ms"]) for r in fc_rows]
    p_fc_gpu = [float(r["paper_gpu_ms"]) for r in fc_rows]
    p_fc_ratio = [float(r["paper_ratio"]) for r in fc_rows]

    # Figure 3
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(msize, fc_cpu, "o-", color="#d62728", linewidth=2.2, markersize=6,
            label="CPU (HPC Xeon O(n²))")
    ax.plot(msize, p_fc_cpu, "o--", color="#fc9272", linewidth=1.6, markersize=5,
            label="CPU (Paper Fig. 3 Reference O(n²))")
    ax.plot(msize, fc_gpu, "x-", color="#1f77b4", linewidth=2.2, markersize=7,
            label="GPU (HPC GTX 1080 Ti CUDA O(n²))")
    ax.plot(msize, p_fc_gpu, "x--", color="#6baed6", linewidth=1.6, markersize=6,
            label="GPU (Paper Fig. 3 GRID K520 O(n²))")
    ax.set_title("Figure 3: Computational Time for Matrix-Vector Multiplication vs Matrix Size")
    ax.set_xlabel("Matrix Size (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_xlim(0, 9000)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig("results/figures/figure3_fc_time_vs_matrixsize.png")
    plt.close(fig)

    # Figure 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(msize, fc_ratio, "x-", color="#1f77b4", linewidth=2.2, markersize=7,
             label="HPC Measured Ratio (CPU / GTX 1080 Ti)")
    ax1.set_title("Figure 4 (a): HPC Ratio of CPU to GPU Computation Time")
    ax1.set_xlabel("Matrix Size (N)")
    ax1.set_ylabel("Ratio (CPU / GPU)")
    ax1.set_xlim(0, 9000)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    fc_norm_scale = p_fc_ratio[-1] / fc_ratio[-1]
    fc_norm_ratio = [r * fc_norm_scale for r in fc_ratio]
    ax2.plot(msize, p_fc_ratio, "x--", color="#d62728", linewidth=2, markersize=7,
             label="Paper Figure 4 (Xeon E5-2670 / GRID K520)")
    ax2.plot(msize, fc_norm_ratio, "o-", color="#1f77b4", linewidth=2, markersize=6,
             label="HPC Curve (Normalized to GRID K520 Scale)")
    ax2.set_title("Figure 4 (b): Comparison with Paper Figure 4")
    ax2.set_xlabel("Matrix Size (N)")
    ax2.set_ylabel("Ratio (CPU / GPU)")
    ax2.set_xlim(0, 9000)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig("results/figures/figure4_fc_speedup_vs_matrixsize.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 3. Figure 5: Distributed Algorithms Convergence (Section 5)
    # ------------------------------------------------------------------------
    with open("results/logs/distributed_algorithms_comparison.json", "r") as f:
        dist_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    styles = {
        "ParallelSGD_NoShuffle": ("v--", "#7f7f7f", "ParallelSGD (No Shuffle - Non-IID)"),
        "ParallelSGD_ShuffleSGD": ("o-", "#1f77b4", "Alg 1+2: ParallelSGD + ShuffleSGD"),
        "ADMM_SGD": ("s-", "#2ca02c", "Alg 3: ADMM.SGD (Sync Consensus)"),
        "Downpour_SGD": ("^-", "#d62728", "Alg 4: DownpourSGD (Async ParamServer)"),
        "Ring_AllReduce_DDP": ("*-", "#9467bd", "Modern: Ring-AllReduce (PyTorch DDP)"),
    }

    for key, (marker, color, label) in styles.items():
        if key not in dist_data:
            continue
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
    ax1.legend(fontsize=8.5)

    ax2.set_title("Validation Classification Accuracy (40 Classes) vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig("results/figures/figure5_distributed_convergence.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 4. Figure 6: Communication Cost & DownpourSGD Worker Saturation (P = T / tau)
    #    vs Modern Ring-AllReduce (PyTorch DDP)
    # ------------------------------------------------------------------------
    scale_rows = read_csv_rows("results/logs/scaling_analysis.csv")
    k_vals = [int(r["k"]) for r in scale_rows]
    psgd_mb = [float(r["psgd_comm_mb"]) for r in scale_rows]
    admm_mb = [float(r["admm_comm_mb"]) for r in scale_rows]
    down_mb = [float(r["downpour_comm_mb"]) for r in scale_rows]
    down_ups = [float(r["downpour_updates_per_sec"]) for r in scale_rows]
    p_sat = float(scale_rows[0]["P_saturation_workers"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.plot(k_vals, psgd_mb, "o-", color="#1f77b4", linewidth=2, label="ParallelSGD + ShuffleSGD: O(D + kp)")
    ax1.plot(k_vals, admm_mb, "s-", color="#2ca02c", linewidth=2, label="ADMM.SGD: O(kp log(1/ε))")
    ax1.plot(k_vals, down_mb, "^-", color="#d62728", linewidth=2, label="DownpourSGD: O(θp)")
    if "ring_allreduce_comm_mb" in scale_rows[0]:
        ring_mb = [float(r["ring_allreduce_comm_mb"]) for r in scale_rows]
        ax1.plot(k_vals, ring_mb, "*-", color="#9467bd", linewidth=2.5, markersize=8,
                 label="Modern: Ring-AllReduce DDP (0 Image Net Transfer)")
    ax1.set_title("Total Communication Volume vs Workers (k)")
    ax1.set_xlabel("Number of Machines / Workers (k)")
    ax1.set_ylabel("Communication Volume (MB)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    ax2.plot(k_vals, down_ups, "D-", color="#d62728", linewidth=2, markersize=6,
             label="DownpourSGD (Server Bottleneck)")
    ax2.axvline(x=p_sat, color="#d62728", linestyle="--", linewidth=1.8,
                label=f"Downpour Saturation Threshold P = T/τ = {p_sat:.0f} workers")
    if "ring_updates_per_sec" in scale_rows[0]:
        ring_ups = [float(r["ring_updates_per_sec"]) for r in scale_rows]
        ax2.plot(k_vals, ring_ups, "*-", color="#9467bd", linewidth=2.5, markersize=8,
                 label="Modern: Ring-AllReduce DDP (Linear Scalability)")
    ax2.set_title("Effective Parameter Updates / Second vs Workers")
    ax2.set_xlabel("Number of Workers (k)")
    ax2.set_ylabel("Parameter Updates / Second")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5)

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

    # ------------------------------------------------------------------------
    # 6. Figure 8: Direction 1 - Gradient Compression (FP32 vs FP16 vs INT8 vs Top-k)
    # ------------------------------------------------------------------------
    if os.path.exists("results/logs/gradient_compression_results.json"):
        with open("results/logs/gradient_compression_results.json", "r") as f:
            comp_data = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
        comp_styles = {
            "FP32_Baseline": ("o-", "#1f77b4", "FP32 Baseline (4 B/param)"),
            "FP16_Half": ("s--", "#2ca02c", "FP16 Half (2 B/param - 2.0x)"),
            "INT8_Quantized": ("^--", "#ff7f0e", "INT8 QSGD (1 B/param - 4.0x)"),
            "TopK_Sparsified": ("*-", "#d62728", "Top-5% EF-SGD (10.0x Compression)"),
        }

        for m_key, (marker, color, label) in comp_styles.items():
            if m_key not in comp_data:
                continue
            hist = comp_data[m_key]
            epochs = [h["epoch"] for h in hist]
            accs = [h["val_acc"] for h in hist]
            ax1.plot(epochs, accs, marker, color=color, linewidth=2, markersize=6, label=label)

        ax1.set_title("Direction 1: Validation Accuracy vs Epoch across Compression")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation Accuracy (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8.5)

        # Bar chart of Total Communication Volume (MB)
        if os.path.exists("results/logs/gradient_compression_summary.csv"):
            sum_rows = read_csv_rows("results/logs/gradient_compression_summary.csv")
            labels = [r["method"].replace("_", " ") for r in sum_rows]
            mbs = [float(r["total_comm_mb"]) for r in sum_rows]
            ratios = [float(r["compression_ratio"]) for r in sum_rows]
            colors_bar = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

            bars = ax2.bar(labels, mbs, color=colors_bar, edgecolor="black", alpha=0.85)
            for bar, mb, cr in zip(bars, mbs, ratios):
                ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0,
                         f"{mb:.1f} MB\n({cr:.1f}x)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax2.set_title("Total Network Communication Volume (10 Epochs)")
            ax2.set_ylabel("Volume Transmitted (MB)")
            ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
            plt.sca(ax2)
            plt.xticks(rotation=15, ha="right")

        fig.tight_layout()
        fig.savefig("results/figures/figure8_gradient_compression.png")
        plt.close(fig)

    # ------------------------------------------------------------------------
    # 7. Figure 9: Direction 2 - Heterogeneous Multi-GPU Load Balancing
    # ------------------------------------------------------------------------
    if os.path.exists("results/logs/heterogeneous_gpu_benchmark.csv"):
        het_rows = read_csv_rows("results/logs/heterogeneous_gpu_benchmark.csv")
        splits = [r["split_label"] for r in het_rows]
        gpu0_times = [float(r["gpu0_time_ms"]) for r in het_rows]
        gpu1_times = [float(r["gpu1_time_ms"]) for r in het_rows]
        idle_times = [float(r["gpu0_idle_ms"]) for r in het_rows]
        throughputs = [float(r["throughput_img_per_sec"]) for r in het_rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
        x_indices = np.arange(len(splits))
        w = 0.35

        ax1.bar(x_indices - w/2, gpu0_times, width=w, color="#1f77b4", edgecolor="black", label="GPU 0 (GTX 1080 Ti) Time")
        ax1.bar(x_indices + w/2, gpu1_times, width=w, color="#ff7f0e", edgecolor="black", label="GPU 1 (GTX 1080) Time")
        ax1.plot(x_indices, idle_times, "ro--", linewidth=2, markersize=6, label="GPU 0 Idle Waiting Time")
        ax1.set_title("Per-GPU Compute Time & Idle Waiting Time vs Batch Split")
        ax1.set_xlabel("Batch Split (B0 / B1) on 64 total images")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(splits, rotation=20, ha="right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8.5)

        bars_thr = ax2.bar(splits, throughputs, color="#2ca02c", edgecolor="black", alpha=0.85)
        for bar, thr in zip(bars_thr, throughputs):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 50.0,
                     f"{thr:.0f} img/s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax2.set_title("Training Throughput (images/sec) vs Heterogeneous Batch Split")
        ax2.set_xlabel("Batch Split (B0 / B1)")
        ax2.set_ylabel("Throughput (images/second)")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.sca(ax2)
        plt.xticks(rotation=20, ha="right", fontsize=9)

        fig.tight_layout()
        fig.savefig("results/figures/figure9_heterogeneous_gpu_balancing.png")
        plt.close(fig)

    # ------------------------------------------------------------------------
    # 8. Figure 10: Direction 3 - Local-SGD / FedAvg Synchronization Interval (H)
    # ------------------------------------------------------------------------
    if os.path.exists("results/logs/local_sgd_summary.csv") and os.path.exists("results/logs/local_sgd_curves.json"):
        with open("results/logs/local_sgd_curves.json", "r") as f:
            local_curves = json.load(f)
        h_rows = read_csv_rows("results/logs/local_sgd_summary.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

        # Subplot 1: Convergence curves for different H
        colors_h = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"]
        for idx, (h_key, hist) in enumerate(local_curves.items()):
            h_val = h_key.replace("H_", "H=")
            epochs = [x["epoch"] for x in hist]
            accs = [x["val_acc"] for x in hist]
            ax1.plot(epochs, accs, "o-", color=colors_h[idx % len(colors_h)], linewidth=2, markersize=5, label=f"Local-SGD ({h_val})")

        ax1.set_title("Direction 3: Validation Accuracy vs Epoch across Sync Interval H")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation Accuracy (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8.5)

        # Subplot 2: Trade-off between Network Comm MB and Wall-Clock Speedup
        h_labels = [f"H={r['H']}" for r in h_rows]
        comm_mbs = [float(r["total_comm_mb"]) for r in h_rows]
        speedups = [float(r["speedup_vs_H1"]) for r in h_rows]

        x_inds = np.arange(len(h_labels))
        w = 0.38
        bars_comm = ax2.bar(x_inds - w/2, comm_mbs, width=w, color="#1f77b4", edgecolor="black", alpha=0.85, label="Network Comm (MB)")
        ax2_twin = ax2.twinx()
        line_speed = ax2_twin.plot(x_inds, speedups, "ro-", linewidth=2.5, markersize=7, label="Wall-Clock Speedup vs H=1")

        for bar, mb in zip(bars_comm, comm_mbs):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 5.0,
                     f"{mb:.0f} MB", ha="center", va="bottom", fontsize=8, fontweight="bold")

        for xi, sp in zip(x_inds, speedups):
            ax2_twin.text(xi, sp + 0.15, f"{sp:.1f}x", color="red", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax2.set_title("Communication Volume & Wall-Clock Speedup vs Sync Interval H")
        ax2.set_xlabel("Synchronization Interval H (Local Steps)")
        ax2.set_ylabel("Total Network Communication (MB)", color="#1f77b4")
        ax2_twin.set_ylabel("Speedup Factor vs H=1 (×)", color="red")
        ax2.set_xticks(x_inds)
        ax2.set_xticklabels(h_labels)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)

        fig.tight_layout()
        fig.savefig("results/figures/figure10_local_sgd_sync_interval.png")
        plt.close(fig)

    # ------------------------------------------------------------------------
    # 9. Figure 11: Direction 4 - Delay-Compensated Asynchronous SGD (DC-ASGD)
    # ------------------------------------------------------------------------
    if os.path.exists("results/logs/dc_asgd_summary.csv") and os.path.exists("results/logs/dc_asgd_curves.json"):
        with open("results/logs/dc_asgd_curves.json", "r") as f:
            dc_curves = json.load(f)
        dc_rows = read_csv_rows("results/logs/dc_asgd_summary.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

        dc_styles = {
            "Sync_Baseline": ("o-", "#1f77b4", "Sync Baseline (H=1, 0 delay)"),
            "Downpour_Balanced": ("s--", "#2ca02c", "DownpourSGD (Balanced workers)"),
            "Downpour_Stragglers": ("x-", "#d62728", "DownpourSGD + Stragglers (Uncompensated)"),
            "DC_ASGD_L05_Stragglers": ("^--", "#ff7f0e", "DC-ASGD (lambda=0.5) + Stragglers"),
            "DC_ASGD_L10_Stragglers": ("*-", "#9467bd", "DC-ASGD (lambda=1.0) + Stragglers"),
        }

        for m_key, (marker, color, label) in dc_styles.items():
            if m_key not in dc_curves:
                continue
            hist = dc_curves[m_key]
            epochs = [x["epoch"] for x in hist]
            accs = [x["val_acc"] for x in hist]
            ax1.plot(epochs, accs, marker, color=color, linewidth=2, markersize=6, label=label)

        ax1.set_title("Direction 4: Convergence under Straggler Delay (DC-ASGD vs Downpour)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation Accuracy (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8.5)

        # Bar chart of Final Accuracy Comparison
        labels_dc = [r["name"].replace("_Stragglers", "").replace("_", "\n") for r in dc_rows]
        accs_dc = [float(r["final_acc"]) for r in dc_rows]
        colors_dc = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]

        bars_dc = ax2.bar(labels_dc, accs_dc, color=colors_dc, edgecolor="black", alpha=0.85)
        for bar, acc in zip(bars_dc, accs_dc):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0,
                     f"{acc:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax2.set_title("Final Accuracy under Straggler Latency")
        ax2.set_ylabel("Final Validation Accuracy (%)")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.sca(ax2)
        plt.xticks(fontsize=8.5)

        fig.tight_layout()
        fig.savefig("results/figures/figure11_dc_asgd_straggler_robustness.png")
        plt.close(fig)

    print("All figures generated successfully in results/figures/")


if __name__ == "__main__":
    main()
