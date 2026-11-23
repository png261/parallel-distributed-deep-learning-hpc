#!/usr/bin/env python3
"""
Direction 1: Gradient Compression in Distributed Deep Learning
Implements and benchmarks:
1. FP32 Baseline: Standard 32-bit uncompressed gradient transfer (4 bytes/param).
2. FP16 Quantization: 16-bit half-precision gradient transfer (2 bytes/param -> 2x compression).
3. INT8 Quantization (QSGD): 8-bit integer quantization with per-tensor scale (1 byte/param -> 4x compression).
4. Top-k Sparsification with Error Feedback (EF-SGD): Transmits only the top 5% largest magnitude gradients
   with residual error accumulation (e_{t} = e_{t-1} + g_t - TopK(...)), achieving ~10x compression.

Evaluates convergence (Logistic Loss, 40-Class Accuracy) and network volume on PaperCNN40 across 8 workers.
"""

import os
import time
import json
import csv
import math
import numpy as np
import torch
import torch.nn as nn
from distributed_sgd_algorithms import (
    PaperCNN40,
    logistic_loss_eq2,
    generate_sorted_40class_dataset,
    get_flat_params,
    set_flat_params,
    get_flat_grads,
    evaluate_model,
    NetworkCostTracker
)


def quantize_fp16(grad_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Compresses FP32 gradient to FP16. Bytes: 2 * numel."""
    half_grad = grad_tensor.to(torch.float16)
    decompressed = half_grad.to(torch.float32)
    bytes_sent = grad_tensor.numel() * 2
    return decompressed, bytes_sent


def quantize_int8(grad_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Symmetric 8-bit linear quantization (QSGD-style):
    Scale = max(|g|) / 127.0
    Quantized = round(g / Scale) as int8
    Decompressed = Quantized * Scale as float32
    Bytes: 1 * numel + 4 (scale)
    """
    max_val = grad_tensor.abs().max()
    if max_val == 0:
        return grad_tensor.clone(), grad_tensor.numel() + 4
    scale = max_val / 127.0
    int8_grad = torch.clamp(torch.round(grad_tensor / scale), -127, 127).to(torch.int8)
    decompressed = int8_grad.to(torch.float32) * scale
    bytes_sent = grad_tensor.numel() * 1 + 4
    return decompressed, bytes_sent


def topk_sparsify_with_error_feedback(
    grad_tensor: torch.Tensor,
    error_buffer: torch.Tensor,
    sparsity_ratio: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Top-k Sparsification with Error-Feedback (Stich et al. / EF-SGD):
    1. v = grad + error_buffer
    2. k = max(1, int(p * sparsity_ratio))
    3. sparse_v = top-k magnitudes of v
    4. error_buffer = v - sparse_v
    5. Transmitted: (k indices of int32) + (k values of float32) = k * 8 bytes
    """
    v = grad_tensor + error_buffer
    p = v.numel()
    k_num = max(1, int(p * sparsity_ratio))

    abs_v = v.abs()
    threshold = torch.topk(abs_v, k_num).values[-1]
    mask = abs_v >= threshold

    sparse_v = torch.zeros_like(v)
    sparse_v[mask] = v[mask]

    new_error_buffer = v - sparse_v
    bytes_sent = k_num * 8  # 4 bytes for index + 4 bytes for float value
    return sparse_v, new_error_buffer, bytes_sent


def run_compressed_distributed_sgd(
    method_name: str,
    init_state_dict: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    k_workers: int = 8,
    epochs: int = 10,
    batch_size: int = 25,
    lr: float = 0.015,
    sparsity_ratio: float = 0.05,
    device: torch.device = torch.device("cpu")
):
    master_model = PaperCNN40(num_classes=40).to(device)
    master_model.load_state_dict(init_state_dict)
    num_params = sum(p.numel() for p in master_model.parameters())

    tracker = NetworkCostTracker()
    worker_models = [PaperCNN40(num_classes=40).to(device) for _ in range(k_workers)]
    for i in range(k_workers):
        set_flat_params(worker_models[i], get_flat_params(master_model))

    optimizers = [torch.optim.SGD(worker_models[i].parameters(), lr=lr, momentum=0.9) for i in range(k_workers)]
    error_buffers = [torch.zeros(num_params, device=device) for _ in range(k_workers)]

    history = []
    comp_time_total = 0.0
    n_samples = len(y_train)
    n_local = n_samples // k_workers
    steps_per_epoch = max(1, n_local // batch_size)

    total_bytes_transmitted = 0

    for ep in range(1, epochs + 1):
        g_rng = torch.Generator()
        g_rng.manual_seed(ep * 1000 + 42)
        global_perm = torch.randperm(n_samples, generator=g_rng)
        worker_indices = [global_perm[i::k_workers] for i in range(k_workers)]

        for step in range(steps_per_epoch):
            step_worker_times = []
            compressed_grads = []

            for i in range(k_workers):
                worker_models[i].train()
                batch_idx = worker_indices[i][step * batch_size:(step + 1) * batch_size]
                xb = x_train[batch_idx].to(device)
                yb = y_train[batch_idx].to(device)

                t0 = time.perf_counter()
                optimizers[i].zero_grad()
                scores = worker_models[i](xb)
                loss = logistic_loss_eq2(scores, yb)
                loss.backward()
                raw_grad = get_flat_grads(worker_models[i])

                # Apply selected compression method
                if method_name == "FP32_Baseline":
                    c_grad = raw_grad
                    bytes_w = num_params * 4
                elif method_name == "FP16_Half":
                    c_grad, bytes_w = quantize_fp16(raw_grad)
                elif method_name == "INT8_Quantized":
                    c_grad, bytes_w = quantize_int8(raw_grad)
                elif method_name == "TopK_Sparsified":
                    c_grad, error_buffers[i], bytes_w = topk_sparsify_with_error_feedback(
                        raw_grad, error_buffers[i], sparsity_ratio=sparsity_ratio
                    )
                else:
                    raise ValueError(f"Unknown method {method_name}")

                t_w = time.perf_counter() - t0
                step_worker_times.append(t_w)
                compressed_grads.append(c_grad)
                total_bytes_transmitted += bytes_w

            comp_time_total += max(step_worker_times)

            # Ring-AllReduce of the compressed gradients
            avg_grad = torch.stack(compressed_grads, dim=0).mean(dim=0)
            for i in range(k_workers):
                offset = 0
                for p in worker_models[i].parameters():
                    numel = p.numel()
                    p.grad.data.copy_(avg_grad[offset:offset + numel].view_as(p.data))
                    offset += numel
                optimizers[i].step()

        set_flat_params(master_model, get_flat_params(worker_models[0]))
        val_loss, val_acc = evaluate_model(master_model, x_val, y_val, device)
        history.append({
            "epoch": ep,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "comp_time_s": comp_time_total,
            "total_mb_transmitted": total_bytes_transmitted / (1024**2),
        })

    return history


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(12)
    device = torch.device("cpu")

    os.makedirs("results/logs", exist_ok=True)
    print("==========================================================")
    print("Direction 1: Gradient Compression in Distributed Deep Learning")
    print("Methods: FP32 Baseline, FP16 Half, INT8 QSGD, Top-5% EF-SGD")
    print("==========================================================")

    x_train, y_train = generate_sorted_40class_dataset(num_samples=800, num_classes=40, seed=42)
    x_val, y_val = generate_sorted_40class_dataset(num_samples=200, num_classes=40, seed=999)

    base_model = PaperCNN40(num_classes=40).to(device)
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    methods = ["FP32_Baseline", "FP16_Half", "INT8_Quantized", "TopK_Sparsified"]
    results = {}

    summary_rows = []

    for m in methods:
        print(f"\nRunning Distributed SGD with: {m}...")
        hist = run_compressed_distributed_sgd(
            method_name=m,
            init_state_dict=init_state,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            k_workers=8,
            epochs=10,
            batch_size=25,
            lr=0.015,
            sparsity_ratio=0.05,
            device=device
        )
        results[m] = hist
        final_acc = hist[-1]["val_acc"]
        final_loss = hist[-1]["val_loss"]
        final_mb = hist[-1]["total_mb_transmitted"]
        baseline_mb = results["FP32_Baseline"][-1]["total_mb_transmitted"] if "FP32_Baseline" in results else final_mb
        compression_ratio = baseline_mb / final_mb if final_mb > 0 else 1.0

        print(f"  -> Final Val Acc: {final_acc:.2f}%, Loss: {final_loss:.4f}, Comm Volume: {final_mb:.2f} MB (Compression: {compression_ratio:.2f}x)")
        summary_rows.append({
            "method": m,
            "final_acc": final_acc,
            "final_loss": final_loss,
            "total_comm_mb": final_mb,
            "compression_ratio": compression_ratio,
        })

    with open("results/logs/gradient_compression_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("results/logs/gradient_compression_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved Direction 1 results to results/logs/gradient_compression_results.json and summary CSV!")


if __name__ == "__main__":
    main()
