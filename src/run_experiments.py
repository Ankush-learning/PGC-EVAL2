"""
run_experiments.py
Executes the 4 GPU-Aware Memory Management experiments for AI workloads on real NVIDIA GPU hardware:
  Exp 1: Memory Transfer Strategies (Pageable vs Pinned vs Unified Memory vs Unified + Prefetch)
  Exp 2: Memory Pooling vs Naive Per-Iteration Allocation (cudaMalloc/cudaFree overhead)
  Exp 3: Asynchronous Multi-Stream Pipelining (Overlapping H2D, Compute, D2H with Pinned Memory)
  Exp 4: VRAM Footprint & Throughput Scaling across AI Batch Sizes
"""

import csv
import ctypes
import json
import time
from pathlib import Path
import numpy as np

from cuda_driver import CudaDriver, GpuMemoryPool

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PTX_PATH = ROOT / "src" / "ai_kernels.ptx"

ALPHA = 1.75
BETA = -0.25


def verify_against_cpu(x_np: np.ndarray, y_gpu_np: np.ndarray) -> float:
    y_cpu = np.maximum(0.0, ALPHA * x_np + BETA).astype(np.float32)
    return float(np.max(np.abs(y_cpu - y_gpu_np)))


def run_exp1_transfer_strategies(drv: CudaDriver, kernel_fn):
    print("\n=== Experiment 1: Memory Transfer Strategies (Pageable vs Pinned vs Unified vs Prefetch) ===")
    # Sizes in millions of float32 elements (4 MB to 128 MB per tensor -> 8 MB to 256 MB total H2D+D2H)
    sizes_m = [1, 4, 8, 16, 32]
    warmup = 2
    iters = 8
    rows = []
    max_err_overall = 0.0

    ev_s = drv.create_event()
    ev_h2d = drv.create_event()
    ev_kern = drv.create_event()
    ev_d2h = drv.create_event()

    for m in sizes_m:
        n = m * 1_000_000
        nbytes = n * 4
        tensor_mb = nbytes / (1024 ** 2)
        total_bytes_moved = 2.0 * nbytes  # H2D + D2H

        rng = np.random.default_rng(42)
        h_x_pageable = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
        h_y_pageable = np.empty(n, dtype=np.float32)

        # --- 1A. Explicit Pageable Memory ---
        d_x = drv.mem_alloc(nbytes)
        d_y = drv.mem_alloc(nbytes)
        for _ in range(warmup):
            drv.memcpy_htod(d_x, h_x_pageable.ctypes.data, nbytes)
            drv.launch_fused_activation(kernel_fn, d_x, d_y, ALPHA, BETA, n)
            drv.memcpy_dtoh(h_y_pageable.ctypes.data, d_y, nbytes)
        drv.synchronize()

        h2d_t, kern_t, d2h_t = [], [], []
        for _ in range(iters):
            drv.record_event(ev_s)
            drv.memcpy_htod(d_x, h_x_pageable.ctypes.data, nbytes)
            drv.record_event(ev_h2d)
            drv.launch_fused_activation(kernel_fn, d_x, d_y, ALPHA, BETA, n)
            drv.record_event(ev_kern)
            drv.memcpy_dtoh(h_y_pageable.ctypes.data, d_y, nbytes)
            drv.record_event(ev_d2h)
            h2d_t.append(drv.event_elapsed_ms(ev_s, ev_h2d))
            kern_t.append(drv.event_elapsed_ms(ev_h2d, ev_kern))
            d2h_t.append(drv.event_elapsed_ms(ev_kern, ev_d2h))

        err = verify_against_cpu(h_x_pageable, h_y_pageable)
        max_err_overall = max(max_err_overall, err)

        avg_h2d, avg_kern, avg_d2h = np.mean(h2d_t), np.mean(kern_t), np.mean(d2h_t)
        tot = avg_h2d + avg_kern + avg_d2h
        bw_gbs = (total_bytes_moved / 1e9) / (tot / 1e3)
        transfer_pct = 100.0 * (avg_h2d + avg_d2h) / tot
        rows.append({
            "elements_m": m, "tensor_mb": round(tensor_mb, 2), "strategy": "Explicit_Pageable",
            "h2d_ms": round(avg_h2d, 4), "kernel_ms": round(avg_kern, 4), "d2h_ms": round(avg_d2h, 4),
            "total_ms": round(tot, 4), "effective_bw_gbs": round(bw_gbs, 2), "transfer_share_pct": round(transfer_pct, 2),
        })

        # --- 1B. Explicit Pinned (Page-Locked) Memory ---
        p_x_ptr = drv.mem_alloc_host(nbytes)
        p_y_ptr = drv.mem_alloc_host(nbytes)
        ctypes.memmove(p_x_ptr, h_x_pageable.ctypes.data, nbytes)

        for _ in range(warmup):
            drv.memcpy_htod(d_x, p_x_ptr, nbytes)
            drv.launch_fused_activation(kernel_fn, d_x, d_y, ALPHA, BETA, n)
            drv.memcpy_dtoh(p_y_ptr, d_y, nbytes)
        drv.synchronize()

        h2d_t, kern_t, d2h_t = [], [], []
        for _ in range(iters):
            drv.record_event(ev_s)
            drv.memcpy_htod(d_x, p_x_ptr, nbytes)
            drv.record_event(ev_h2d)
            drv.launch_fused_activation(kernel_fn, d_x, d_y, ALPHA, BETA, n)
            drv.record_event(ev_kern)
            drv.memcpy_dtoh(p_y_ptr, d_y, nbytes)
            drv.record_event(ev_d2h)
            h2d_t.append(drv.event_elapsed_ms(ev_s, ev_h2d))
            kern_t.append(drv.event_elapsed_ms(ev_h2d, ev_kern))
            d2h_t.append(drv.event_elapsed_ms(ev_kern, ev_d2h))

        avg_h2d, avg_kern, avg_d2h = np.mean(h2d_t), np.mean(kern_t), np.mean(d2h_t)
        tot = avg_h2d + avg_kern + avg_d2h
        bw_gbs = (total_bytes_moved / 1e9) / (tot / 1e3)
        transfer_pct = 100.0 * (avg_h2d + avg_d2h) / tot
        rows.append({
            "elements_m": m, "tensor_mb": round(tensor_mb, 2), "strategy": "Explicit_Pinned",
            "h2d_ms": round(avg_h2d, 4), "kernel_ms": round(avg_kern, 4), "d2h_ms": round(avg_d2h, 4),
            "total_ms": round(tot, 4), "effective_bw_gbs": round(bw_gbs, 2), "transfer_share_pct": round(transfer_pct, 2),
        })

        drv.mem_free_host(p_x_ptr)
        drv.mem_free_host(p_y_ptr)
        drv.mem_free(d_x)
        drv.mem_free(d_y)

        # --- 1C. Unified Memory (Demand-Paged) ---
        um_x = drv.mem_alloc_managed(nbytes)
        um_y = drv.mem_alloc_managed(nbytes)
        ctypes.memmove(um_x, h_x_pageable.ctypes.data, nbytes)

        # Warm-up
        drv.launch_fused_activation(kernel_fn, um_x, um_y, ALPHA, BETA, n)
        drv.synchronize()

        um_tot_t = []
        for _ in range(iters):
            # Touch input on host to simulate fresh AI batch on CPU before GPU launch
            t0 = time.perf_counter()
            ctypes.memmove(um_x, h_x_pageable.ctypes.data, nbytes)
            drv.launch_fused_activation(kernel_fn, um_x, um_y, ALPHA, BETA, n)
            drv.synchronize()
            # Read result back on host
            ctypes.memmove(h_y_pageable.ctypes.data, um_y, nbytes)
            um_tot_t.append((time.perf_counter() - t0) * 1000.0)

        avg_um = float(np.mean(um_tot_t))
        bw_um = (total_bytes_moved / 1e9) / (avg_um / 1e3)
        rows.append({
            "elements_m": m, "tensor_mb": round(tensor_mb, 2), "strategy": "Unified_Memory",
            "h2d_ms": 0.0, "kernel_ms": round(avg_um, 4), "d2h_ms": 0.0,
            "total_ms": round(avg_um, 4), "effective_bw_gbs": round(bw_um, 2), "transfer_share_pct": 0.0,
        })

        # --- 1D. Unified Memory + Prefetch ---
        um_pref_t = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ctypes.memmove(um_x, h_x_pageable.ctypes.data, nbytes)
            drv.mem_prefetch_async(um_x, nbytes, drv.device_id)
            drv.mem_prefetch_async(um_y, nbytes, drv.device_id)
            drv.launch_fused_activation(kernel_fn, um_x, um_y, ALPHA, BETA, n)
            drv.synchronize()
            ctypes.memmove(h_y_pageable.ctypes.data, um_y, nbytes)
            um_pref_t.append((time.perf_counter() - t0) * 1000.0)

        avg_ump = float(np.mean(um_pref_t))
        bw_ump = (total_bytes_moved / 1e9) / (avg_ump / 1e3)
        rows.append({
            "elements_m": m, "tensor_mb": round(tensor_mb, 2), "strategy": "Unified_Prefetch",
            "h2d_ms": 0.0, "kernel_ms": round(avg_ump, 4), "d2h_ms": 0.0,
            "total_ms": round(avg_ump, 4), "effective_bw_gbs": round(bw_ump, 2), "transfer_share_pct": 0.0,
        })

        drv.mem_free(um_x)
        drv.mem_free(um_y)

        print(f"  Size {m:2d}M ({tensor_mb:6.1f} MB) | Pageable: {rows[-4]['total_ms']:7.2f} ms | "
              f"Pinned: {rows[-3]['total_ms']:7.2f} ms | UM: {rows[-2]['total_ms']:7.2f} ms | "
              f"UM+Prefetch: {rows[-1]['total_ms']:7.2f} ms | MaxErr: {err:.1e}")

    drv.destroy_event(ev_s)
    drv.destroy_event(ev_h2d)
    drv.destroy_event(ev_kern)
    drv.destroy_event(ev_d2h)

    out_csv = RESULTS_DIR / "exp1_transfer_strategies.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows, max_err_overall


def run_exp2_memory_pooling(drv: CudaDriver, kernel_fn):
    print("\n=== Experiment 2: Naive Per-Step Allocation vs GPU-Aware Memory Pool (100 AI Steps) ===")
    sizes_m = [1, 2, 4, 8, 16]
    steps = 100
    rows = []

    for m in sizes_m:
        n = m * 1_000_000
        nbytes = n * 4
        tensor_mb = nbytes / (1024 ** 2)

        # Warm-up
        tmp = drv.mem_alloc(nbytes)
        drv.mem_free(tmp)

        # Naive: cuMemAlloc + kernel + cuMemFree every training/inference step
        t0 = time.perf_counter()
        for _ in range(steps):
            dx = drv.mem_alloc(nbytes)
            dy = drv.mem_alloc(nbytes)
            drv.launch_fused_activation(kernel_fn, dx, dy, ALPHA, BETA, n)
            drv.synchronize()
            drv.mem_free(dx)
            drv.mem_free(dy)
        naive_ms = (time.perf_counter() - t0) * 1000.0

        # Pooled: GPU-Aware Caching Allocator
        pool = GpuMemoryPool(drv)
        t0 = time.perf_counter()
        for _ in range(steps):
            dx = pool.alloc(nbytes)
            dy = pool.alloc(nbytes)
            drv.launch_fused_activation(kernel_fn, dx, dy, ALPHA, BETA, n)
            drv.synchronize()
            pool.free(dx, nbytes)
            pool.free(dy, nbytes)
        pooled_ms = (time.perf_counter() - t0) * 1000.0
        cache_hits = pool.cache_hits
        driver_allocs = pool.total_driver_allocs
        pool.release_all()

        speedup = naive_ms / pooled_ms
        rows.append({
            "elements_m": m,
            "tensor_mb": round(tensor_mb, 2),
            "steps": steps,
            "naive_total_ms": round(naive_ms, 3),
            "pooled_total_ms": round(pooled_ms, 3),
            "naive_per_step_ms": round(naive_ms / steps, 4),
            "pooled_per_step_ms": round(pooled_ms / steps, 4),
            "speedup": round(speedup, 2),
            "driver_allocs_naive": steps * 2,
            "driver_allocs_pooled": driver_allocs,
            "pool_cache_hits": cache_hits,
        })
        print(f"  Size {m:2d}M ({tensor_mb:5.1f} MB) | Naive: {naive_ms:7.2f} ms | "
              f"Pooled: {pooled_ms:7.2f} ms | Speedup: {speedup:5.2f}x | Driver Allocs: {steps*2} -> {driver_allocs}")

    out_csv = RESULTS_DIR / "exp2_memory_pooling.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_exp3_stream_pipelining(drv: CudaDriver, kernel_fn):
    print("\n=== Experiment 3: Synchronous vs Multi-Stream Async Pipelining (32M Elements Batched) ===")
    total_n = 32_000_000
    total_bytes = total_n * 4
    stream_counts = [1, 2, 4, 8]
    iters = 8
    rows = []

    p_x = drv.mem_alloc_host(total_bytes)
    p_y = drv.mem_alloc_host(total_bytes)
    d_x = drv.mem_alloc(total_bytes)
    d_y = drv.mem_alloc(total_bytes)

    ev_start = drv.create_event()
    ev_end = drv.create_event()

    # Baseline: Synchronous single-stream
    for _ in range(2):
        drv.memcpy_htod(d_x, p_x, total_bytes)
        drv.launch_fused_activation(kernel_fn, d_x, d_y, ALPHA, BETA, total_n)
        drv.memcpy_dtoh(p_y, d_y, total_bytes)
    drv.synchronize()

    sync_times = []
    for _ in range(iters):
        drv.record_event(ev_start)
        drv.memcpy_htod(d_x, p_x, total_bytes)
        drv.launch_fused_activation(kernel_fn, d_x, d_y, ALPHA, BETA, total_n)
        drv.memcpy_dtoh(p_y, d_y, total_bytes)
        drv.record_event(ev_end)
        sync_times.append(drv.event_elapsed_ms(ev_start, ev_end))
    baseline_ms = float(np.mean(sync_times))

    for num_streams in stream_counts:
        streams = [drv.create_stream() for _ in range(num_streams)]
        chunk_n = total_n // num_streams
        chunk_bytes = chunk_n * 4

        # Warm-up
        for s_idx in range(num_streams):
            off = s_idx * chunk_bytes
            drv.memcpy_htod_async(d_x + off, p_x + off, chunk_bytes, streams[s_idx])
            drv.launch_fused_activation(kernel_fn, d_x + off, d_y + off, ALPHA, BETA, chunk_n, stream=streams[s_idx])
            drv.memcpy_dtoh_async(p_y + off, d_y + off, chunk_bytes, streams[s_idx])
        drv.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for s_idx in range(num_streams):
                off = s_idx * chunk_bytes
                drv.memcpy_htod_async(d_x + off, p_x + off, chunk_bytes, streams[s_idx])
                drv.launch_fused_activation(kernel_fn, d_x + off, d_y + off, ALPHA, BETA, chunk_n, stream=streams[s_idx])
                drv.memcpy_dtoh_async(p_y + off, d_y + off, chunk_bytes, streams[s_idx])
            for st in streams:
                drv.stream_synchronize(st)
            times.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = float(np.mean(times))
        bw_gbs = (2.0 * total_bytes / 1e9) / (avg_ms / 1e3)
        speedup = baseline_ms / avg_ms
        rows.append({
            "num_streams": num_streams,
            "chunk_mb": round(chunk_bytes / (1024 ** 2), 2),
            "avg_time_ms": round(avg_ms, 4),
            "baseline_sync_ms": round(baseline_ms, 4),
            "speedup_vs_sync": round(speedup, 2),
            "effective_bw_gbs": round(bw_gbs, 2),
        })
        print(f"  Streams: {num_streams} | Chunk: {chunk_bytes/(1024**2):5.1f} MB | "
              f"Time: {avg_ms:6.2f} ms | Speedup vs Sync: {speedup:4.2f}x | BW: {bw_gbs:5.2f} GB/s")

        for st in streams:
            drv.destroy_stream(st)

    drv.destroy_event(ev_start)
    drv.destroy_event(ev_end)
    drv.mem_free(d_x)
    drv.mem_free(d_y)
    drv.mem_free_host(p_x)
    drv.mem_free_host(p_y)

    out_csv = RESULTS_DIR / "exp3_stream_pipelining.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_exp4_vram_footprint(drv: CudaDriver, kernel_fn):
    print("\n=== Experiment 4: VRAM Footprint & GPU Kernel Throughput Scaling ===")
    # Simulate AI batch sizes (batch_size x 4096 features x 64 seq_len -> float32)
    batch_sizes = [8, 16, 32, 64, 128, 256]
    elems_per_sample = 4096 * 64  # 262,144 floats = 1 MB per sample
    ev_s = drv.create_event()
    ev_e = drv.create_event()
    rows = []

    free_base, total_vram = drv.mem_info()
    base_used_mb = (total_vram - free_base) / (1024 ** 2)

    for bs in batch_sizes:
        n = bs * elems_per_sample
        nbytes = n * 4
        dx = drv.mem_alloc(nbytes)
        dy = drv.mem_alloc(nbytes)
        free_now, _ = drv.mem_info()
        used_vram_mb = (total_vram - free_now) / (1024 ** 2)
        delta_vram_mb = max(0.0, used_vram_mb - base_used_mb)

        # Warm-up
        drv.launch_fused_activation(kernel_fn, dx, dy, ALPHA, BETA, n)
        drv.synchronize()

        times = []
        for _ in range(10):
            drv.record_event(ev_s)
            drv.launch_fused_activation(kernel_fn, dx, dy, ALPHA, BETA, n)
            drv.record_event(ev_e)
            times.append(drv.event_elapsed_ms(ev_s, ev_e))

        avg_ms = float(np.mean(times))
        # Kernel reads x (nbytes) + writes y (nbytes)
        gpu_bw_gbs = (2.0 * nbytes / 1e9) / (avg_ms / 1e3)
        samples_per_sec = bs / (avg_ms / 1e3)
        rows.append({
            "batch_size": bs,
            "elements_m": round(n / 1e6, 2),
            "allocated_mb": round((2 * nbytes) / (1024 ** 2), 2),
            "total_vram_used_mb": round(used_vram_mb, 2),
            "vram_util_pct": round(100.0 * used_vram_mb / (total_vram / (1024 ** 2)), 2),
            "kernel_ms": round(avg_ms, 4),
            "gpu_mem_bw_gbs": round(gpu_bw_gbs, 2),
            "throughput_samples_s": round(samples_per_sec, 1),
        })
        print(f"  Batch {bs:3d} | Alloc: {(2*nbytes)/(1024**2):6.1f} MB | VRAM Used: {used_vram_mb:6.1f} MB | "
              f"Kernel: {avg_ms:6.3f} ms | GPU BW: {gpu_bw_gbs:6.1f} GB/s")

        drv.mem_free(dx)
        drv.mem_free(dy)

    drv.destroy_event(ev_s)
    drv.destroy_event(ev_e)

    out_csv = RESULTS_DIR / "exp4_vram_footprint.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    drv = CudaDriver(device_id=0)
    info = drv.get_device_info()
    print("GPU Testbed:", json.dumps(info, indent=2))

    with open(RESULTS_DIR / "system_info.json", "w") as f:
        json.dump(info, f, indent=2)

    mod = drv.load_ptx(PTX_PATH)
    kernel_fn = drv.get_function(mod, "ai_fused_activation")

    exp1_rows, max_err = run_exp1_transfer_strategies(drv, kernel_fn)
    exp2_rows = run_exp2_memory_pooling(drv, kernel_fn)
    exp3_rows = run_exp3_stream_pipelining(drv, kernel_fn)
    exp4_rows = run_exp4_vram_footprint(drv, kernel_fn)

    summary = {
        "system_info": info,
        "cpu_gpu_verification_max_abs_error": max_err,
        "verification_passed": max_err < 1e-5,
        "exp1_count": len(exp1_rows),
        "exp2_count": len(exp2_rows),
        "exp3_count": len(exp3_rows),
        "exp4_count": len(exp4_rows),
    }
    with open(RESULTS_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    drv.close()
    print("\nAll experiments completed successfully! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
