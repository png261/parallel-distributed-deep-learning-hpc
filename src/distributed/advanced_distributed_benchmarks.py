#!/usr/bin/env python3
"""
Advanced Distributed Optimization Benchmarks:
1. Direction 3: Local-SGD / FedAvg with Adaptive Synchronization Interval (H in [1, 2, 4, 8, 16, 32])
   - Analyzes the fundamental trade-off between communication frequency (1/H) vs client drift & convergence.
   - Measures final accuracy, total communication volume, and simulated wall-clock speedup on 1Gbps network.

2. Direction 4: Delay-Compensated Asynchronous SGD (DC-ASGD) vs DownpourSGD under Stragglers
   - Tackles the gradient staleness flaw of DownpourSGD (Section 5.2.1 of the paper).
   - Evaluates:
     * Synchronous SGD (H=1)
     * DownpourSGD (Standard async PS, balanced workers)
     * DownpourSGD (Async PS with 2 straggler workers experiencing 3x delay)
     * DC-ASGD (lambda = 0.5 delay compensation under stragglers)
     * DC-ASGD (lambda = 1.0 delay compensation under stragglers)
"""

import os
import sys
import time
import json
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distributed_sgd_algorithms import (
    PaperCNN40,
    logistic_loss_eq2,
    generate_sorted_40class_dataset,
    get_flat_params,
    set_flat_params,
    evaluate_model,
    NetworkCostTracker
)


# ============================================================================
# Direction 3: Local-SGD / FedAvg with H-Step Synchronization Interval
# ============================================================================

def run_local_sgd_benchmark(
    init_state_dict: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    H_values: list = [1, 2, 4, 8, 16, 32],
    k_workers: int = 8,
    epochs: int = 10,
    batch_size: int = 25,
    base_lr: float = 0.015,
    device: torch.device = torch.device("cpu")
):
    print("=" * 60)
    print("Direction 3: Local-SGD / FedAvg Synchronization Interval Study")
    print(f"H-step values: {H_values} across {k_workers} workers, {epochs} epochs")
    print("=" * 60)

    # Prepare IID stratified shards for the workers
    n_total = len(y_train)
    shard_size = n_total // k_workers
    perm = torch.randperm(n_total)
    worker_shards = []
    for w in range(k_workers):
        idx = perm[w * shard_size : (w + 1) * shard_size]
        worker_shards.append((x_train[idx], y_train[idx]))

    model_dummy = PaperCNN40(num_classes=40)
    p_numel = sum(p.numel() for p in model_dummy.parameters())

    results = []
    curve_data = {}

    for H in H_values:
        print(f"\nRunning Local-SGD with H = {H} (Sync every {H} local steps)...")
        # Initialize master model and worker models
        master_model = PaperCNN40(num_classes=40).to(device)
        master_model.load_state_dict(init_state_dict)
        init_flat = get_flat_params(master_model)

        workers = [PaperCNN40(num_classes=40).to(device) for _ in range(k_workers)]
        optimizers = []
        for w in range(k_workers):
            set_flat_params(workers[w], init_flat)
            # Adjust learning rate with warmup / slight scaling
            opt = torch.optim.SGD(workers[w].parameters(), lr=base_lr, momentum=0.9)
            optimizers.append(opt)

        history = []
        total_comm_rounds = 0
        total_compute_time_s = 0.0

        for ep in range(1, epochs + 1):
            ep_t0 = time.perf_counter()
            # Shuffle local shards per epoch
            worker_loaders = []
            steps_per_epoch = shard_size // batch_size
            for w in range(k_workers):
                xw, yw = worker_shards[w]
                w_perm = torch.randperm(len(yw))
                worker_loaders.append((xw[w_perm], yw[w_perm]))

            # Run steps in lockstep across workers
            for step in range(steps_per_epoch):
                for w in range(k_workers):
                    workers[w].train()
                    xw, yw = worker_loaders[w]
                    s_idx = slice(step * batch_size, (step + 1) * batch_size)
                    xb = xw[s_idx].to(device)
                    yb = yw[s_idx].to(device)

                    optimizers[w].zero_grad()
                    out = workers[w](xb)
                    loss = logistic_loss_eq2(out, yb)
                    loss.backward()
                    optimizers[w].step()

                # Check if synchronization is required at this step
                is_sync_step = ((step + 1) % H == 0) or (step == steps_per_epoch - 1)
                if is_sync_step:
                    total_comm_rounds += 1
                    # Average parameters across all workers
                    worker_flats = [get_flat_params(workers[w]) for w in range(k_workers)]
                    avg_flat = torch.stack(worker_flats, dim=0).mean(dim=0)
                    for w in range(k_workers):
                        set_flat_params(workers[w], avg_flat)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_time = time.perf_counter() - ep_t0
            total_compute_time_s += ep_time

            # Synchronize master model for evaluation
            set_flat_params(master_model, get_flat_params(workers[0]))
            val_loss, val_acc = evaluate_model(master_model, x_val, y_val, device)
            history.append({
                "epoch": ep,
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2)
            })
            print(f"  [H={H}] Epoch {ep:2d}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Communication modeling:
        # Ring-AllReduce communication volume per round = 2 * ((k - 1) / k) * p * 4 bytes
        bytes_per_round = 2 * ((k_workers - 1) / k_workers) * (p_numel * 4)
        total_comm_bytes = total_comm_rounds * bytes_per_round
        total_comm_mb = total_comm_bytes / (1024 * 1024)

        # Network time model (1 Gbps LAN: Latency L = 2ms, Bandwidth B = 125 MB/s)
        L_sec = 0.002
        B_Bps = 125 * 1024 * 1024
        comm_time_per_round = 2 * (k_workers - 1) * L_sec + bytes_per_round / B_Bps
        total_comm_time_s = total_comm_rounds * comm_time_per_round
        total_wall_clock_s = total_compute_time_s + total_comm_time_s

        res_entry = {
            "H": H,
            "final_acc": history[-1]["val_acc"],
            "final_loss": history[-1]["val_loss"],
            "comm_rounds": total_comm_rounds,
            "total_comm_mb": round(total_comm_mb, 2),
            "compute_time_s": round(total_compute_time_s, 2),
            "comm_time_s": round(total_comm_time_s, 2),
            "wall_clock_s": round(total_wall_clock_s, 2),
            "speedup_vs_H1": 1.0  # to be computed relative to H=1
        }
        results.append(res_entry)
        curve_data[f"H_{H}"] = history

    # Compute speedups relative to H=1
    h1_time = results[0]["wall_clock_s"]
    for r in results:
        r["speedup_vs_H1"] = round(h1_time / r["wall_clock_s"], 2)

    return results, curve_data


# ============================================================================
# Direction 4: Delay-Compensated Asynchronous SGD (DC-ASGD) vs DownpourSGD
# ============================================================================

def run_dc_asgd_benchmark(
    init_state_dict: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    k_workers: int = 8,
    epochs: int = 10,
    batch_size: int = 25,
    lr: float = 0.015,
    straggler_delay_multiplier: int = 3,
    device: torch.device = torch.device("cpu")
):
    print("\n" + "=" * 60)
    print("Direction 4: Delay-Compensated Asynchronous SGD (DC-ASGD) vs DownpourSGD")
    print(f"Comparing Async PS with Stragglers (Workers 6, 7 have {straggler_delay_multiplier}x delay)")
    print("=" * 60)

    # Configurations to test:
    # 1. Sync_Baseline: Synchronous update every step (staleness = 0)
    # 2. Downpour_Balanced: Async PS, all workers equal speed
    # 3. Downpour_Stragglers: Async PS, workers 6 & 7 are stragglers (uncompensated)
    # 4. DC_ASGD_L05_Stragglers: Async PS with DC-ASGD (lambda = 0.5)
    # 5. DC_ASGD_L10_Stragglers: Async PS with DC-ASGD (lambda = 1.0)
    configs = [
        {"name": "Sync_Baseline", "mode": "sync", "has_stragglers": False, "lambda": 0.0},
        {"name": "Downpour_Balanced", "mode": "async", "has_stragglers": False, "lambda": 0.0},
        {"name": "Downpour_Stragglers", "mode": "async", "has_stragglers": True, "lambda": 0.0},
        {"name": "DC_ASGD_L05_Stragglers", "mode": "async", "has_stragglers": True, "lambda": 0.5},
        {"name": "DC_ASGD_L10_Stragglers", "mode": "async", "has_stragglers": True, "lambda": 1.0},
    ]

    n_total = len(y_train)
    shard_size = n_total // k_workers
    perm = torch.randperm(n_total)
    worker_shards = []
    for w in range(k_workers):
        idx = perm[w * shard_size : (w + 1) * shard_size]
        worker_shards.append((x_train[idx], y_train[idx]))

    results = []
    curve_data = {}

    for cfg in configs:
        name = cfg["name"]
        mode = cfg["mode"]
        has_stragglers = cfg["has_stragglers"]
        lam = cfg["lambda"]

        print(f"\nRunning {name}...")
        server_model = PaperCNN40(num_classes=40).to(device)
        server_model.load_state_dict(init_state_dict)

        server_params = get_flat_params(server_model)
        worker_models = [PaperCNN40(num_classes=40).to(device) for _ in range(k_workers)]

        # Worker parameter snapshot storage for calculating staleness w_t - w_tau
        worker_snapshots = [server_params.clone() for _ in range(k_workers)]
        worker_pull_steps = [0 for _ in range(k_workers)]

        # Simulate worker processing queue / events
        # Worker speed: 1.0 for normal, 1.0 / multiplier for stragglers
        worker_speeds = [1.0] * k_workers
        if has_stragglers:
            worker_speeds[6] = 1.0 / straggler_delay_multiplier
            worker_speeds[7] = 1.0 / straggler_delay_multiplier

        steps_per_epoch = shard_size // batch_size
        total_server_steps = 0
        total_staleness_sum = 0
        staleness_count = 0

        history = []

        for ep in range(1, epochs + 1):
            worker_loaders = []
            for w in range(k_workers):
                xw, yw = worker_shards[w]
                w_perm = torch.randperm(len(yw))
                worker_loaders.append((xw[w_perm], yw[w_perm]))

            # In each epoch, workers generate gradient updates
            for step in range(steps_per_epoch):
                if mode == "sync":
                    # Synchronous update: all workers compute gradient at current server_params
                    grads = []
                    for w in range(k_workers):
                        set_flat_params(worker_models[w], server_params)
                        worker_models[w].train()
                        xw, yw = worker_loaders[w]
                        s_idx = slice(step * batch_size, (step + 1) * batch_size)
                        xb = xw[s_idx].to(device)
                        yb = yw[s_idx].to(device)

                        worker_models[w].zero_grad()
                        loss = logistic_loss_eq2(worker_models[w](xb), yb)
                        loss.backward()
                        g = torch.cat([p.grad.view(-1) for p in worker_models[w].parameters()])
                        grads.append(g)

                    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
                    server_params = server_params - lr * avg_grad
                    set_flat_params(server_model, server_params)
                    total_server_steps += 1

                else:
                    # Asynchronous Parameter Server
                    for w in range(k_workers):
                        # Stragglers only push every straggler_delay_multiplier steps
                        if has_stragglers and w in [6, 7]:
                            if (step + 1) % straggler_delay_multiplier != 0 and step != steps_per_epoch - 1:
                                continue  # Straggler still computing

                        # Worker computes gradient on its local snapshot w_tau
                        w_tau = worker_snapshots[w]
                        tau = worker_pull_steps[w]
                        staleness = max(0, total_server_steps - tau)
                        total_staleness_sum += staleness
                        staleness_count += 1

                        set_flat_params(worker_models[w], w_tau)
                        worker_models[w].train()
                        xw, yw = worker_loaders[w]
                        s_idx = slice(step * batch_size, (step + 1) * batch_size)
                        xb = xw[s_idx].to(device)
                        yb = yw[s_idx].to(device)

                        worker_models[w].zero_grad()
                        loss = logistic_loss_eq2(worker_models[w](xb), yb)
                        loss.backward()
                        g_tau = torch.cat([p.grad.view(-1) for p in worker_models[w].parameters()])

                        # Apply Delay Compensation (DC-ASGD):
                        # g_comp = g_tau + lambda * (g_tau ** 2) * (w_t - w_tau)
                        if lam > 0.0 and staleness > 0:
                            diff = server_params - w_tau
                            g_comp = g_tau + lam * (g_tau * g_tau) * diff
                        else:
                            g_comp = g_tau

                        # Clip gradient for numerical stability in async regime
                        g_comp = torch.clamp(g_comp, -5.0, 5.0)

                        # Update server parameters
                        server_params = server_params - (lr / k_workers) * g_comp
                        total_server_steps += 1

                        # Worker pulls fresh server parameters
                        worker_snapshots[w] = server_params.clone()
                        worker_pull_steps[w] = total_server_steps

            set_flat_params(server_model, server_params)
            val_loss, val_acc = evaluate_model(server_model, x_val, y_val, device)
            avg_stale = (total_staleness_sum / max(1, staleness_count)) if mode == "async" else 0.0
            history.append({
                "epoch": ep,
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "avg_staleness": round(avg_stale, 2)
            })
            print(f"  [{name}] Epoch {ep:2d}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Avg Staleness: {avg_stale:.1f}")

        avg_stale_final = (total_staleness_sum / max(1, staleness_count)) if mode == "async" else 0.0
        results.append({
            "name": name,
            "final_acc": history[-1]["val_acc"],
            "final_loss": history[-1]["val_loss"],
            "avg_staleness": round(avg_stale_final, 2),
            "description": (
                "Ideal synchronous SGD" if mode == "sync" else
                "Standard Async PS (No stragglers)" if not has_stragglers else
                "Async PS + Stragglers (Uncompensated)" if lam == 0.0 else
                f"DC-ASGD (lambda={lam}) + Stragglers"
            )
        })
        curve_data[name] = history

    return results, curve_data


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(12)
    device = torch.device("cpu")
    print(f"Executing on Device: {device} (12 OpenMP threads)")

    # Generate synthetic dataset matching paper specifications (40 classes, 224x224 greyscale)
    print("Generating 40-Class Dataset (N=800 train, N=200 val)...")
    x_train, y_train = generate_sorted_40class_dataset(num_samples=800, num_classes=40, seed=42)
    x_val, y_val = generate_sorted_40class_dataset(num_samples=200, num_classes=40, seed=999)

    init_model = PaperCNN40(num_classes=40).to(device)
    init_state_dict = {k: v.clone() for k, v in init_model.state_dict().items()}

    # 1. Run Direction 3: Local-SGD Benchmark
    local_sgd_results, local_sgd_curves = run_local_sgd_benchmark(
        init_state_dict=init_state_dict,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        H_values=[1, 2, 4, 8, 16, 32],
        k_workers=8,
        epochs=10,
        batch_size=25,
        base_lr=0.015,
        device=device
    )

    # 2. Run Direction 4: Delay-Compensated Asynchronous SGD Benchmark
    dc_asgd_results, dc_asgd_curves = run_dc_asgd_benchmark(
        init_state_dict=init_state_dict,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        k_workers=8,
        epochs=10,
        batch_size=25,
        lr=0.015,
        straggler_delay_multiplier=3,
        device=device
    )

    # Save Results
    os.makedirs("results/logs", exist_ok=True)

    # Write Local-SGD Summary CSV
    with open("results/logs/local_sgd_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "H", "final_acc", "final_loss", "comm_rounds", "total_comm_mb",
            "compute_time_s", "comm_time_s", "wall_clock_s", "speedup_vs_H1"
        ])
        writer.writeheader()
        writer.writerows(local_sgd_results)

    with open("results/logs/local_sgd_curves.json", "w") as f:
        json.dump(local_sgd_curves, f, indent=2)

    # Write DC-ASGD Summary CSV
    with open("results/logs/dc_asgd_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "final_acc", "final_loss", "avg_staleness", "description"
        ])
        writer.writeheader()
        writer.writerows(dc_asgd_results)

    with open("results/logs/dc_asgd_curves.json", "w") as f:
        json.dump(dc_asgd_curves, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL ADVANCED DISTRIBUTED BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("Files generated:")
    print("  - results/logs/local_sgd_summary.csv")
    print("  - results/logs/local_sgd_curves.json")
    print("  - results/logs/dc_asgd_summary.csv")
    print("  - results/logs/dc_asgd_curves.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
