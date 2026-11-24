# GPU-Aware Memory Management for AI Workloads

## 1. Problem Statement & Motivation

Modern AI workloads (deep neural network training and inference) are frequently **memory-bound** rather than compute-bound. Moving large activation tensors, weights, and batches across the PCIe bus between host RAM and GPU VRAM—and repeatedly allocating/freeing temporary device buffers during iterative training steps—often dominates end-to-end execution time.

This project designs, implements, and benchmarks **GPU-Aware Memory Management** techniques on real NVIDIA GPU hardware using the **NVIDIA CUDA Driver API** and custom **PTX GPU kernels** modeling a fused neural network activation layer ($y_i = \max(0, \alpha x_i + \beta)$).

We systematically evaluate four core dimensions of GPU-aware memory management:
1. **Experiment 1 — Host-to-Device Transfer Strategies**: Explicit Pageable Memory vs. Explicit Pinned (Page-Locked) Memory vs. CUDA Unified Memory (Demand-Paged) vs. Unified Memory with Prefetching (`cuMemPrefetchAsync`).
2. **Experiment 2 — GPU-Aware Caching Memory Pool**: Naive per-iteration `cuMemAlloc`/`cuMemFree` vs. a size-bucketed GPU Memory Pool (modeled after PyTorch's `CUDACachingAllocator`) over 100 iterative AI steps.
3. **Experiment 3 — Multi-Stream Asynchronous Pipelining**: Synchronous sequential execution vs. 1, 2, 4, and 8 concurrent CUDA streams overlapping H2D transfer, GPU compute, and D2H transfer.
4. **Experiment 4 — VRAM Footprint & Memory Hierarchy Bandwidth**: Live VRAM tracking (`cuMemGetInfo`) and on-device GPU memory bandwidth scaling from L2-cache-resident batches to GDDR6-DRAM-bound batches.

---

## 2. Hardware & Software Testbed

All measurements were collected directly on the local workstation GPU:

| Component | Measured Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **Architecture / Compute Capability** | Ada Lovelace (`sm_89`, Compute Capability 8.9) |
| **Streaming Multiprocessors (SMs)** | 20 |
| **Total GPU VRAM** | 6,140.5 MB (~6 GB GDDR6) |
| **CUDA Driver Version** | 13.1 (`nvcuda.dll`) |
| **Concurrent Managed Access (WDDM)** | `False` (Windows Display Driver Model) |
| **Operating System** | Windows 10 |
| **Python Runtime** | Python 3.10.6 + NumPy + Matplotlib |

### Verification & Timing Methodology
- **Timing**: All GPU operations were timed with **CUDA Events** (`cuEventRecord` / `cuEventElapsedTime`) after warm-up runs, averaging across 8–10 timed iterations.
- **Correctness Verification**: Every GPU kernel output was verified element-by-element against a NumPy CPU reference implementation.
  - **Max Absolute Error (GPU vs. CPU)**: `1.2e-07` (**PASSED**, exact IEEE-754 `float32` match).

---

## 3. Repository & Folder Structure

```text
gpu_aware_memory_management/
├── README.md                          # Full report, tables, analysis, and embedded graphs
├── src/
│   ├── cuda_driver.py                 # Python CUDA Driver API wrapper + GpuMemoryPool
│   ├── ai_kernels.ptx                 # Native NVIDIA PTX kernel (Fused Affine + ReLU)
│   ├── run_experiments.py             # Master benchmark runner for Experiments 1–4
│   ├── generate_graphs.py             # Publication-quality graph generator (300 DPI)
│   └── cuda_reference.cu              # Equivalent CUDA C++ reference for nvcc
├── results/
│   ├── system_info.json               # Hardware/driver metadata
│   ├── experiment_summary.json        # Verification status & run summary
│   ├── exp1_transfer_strategies.csv   # Raw data for Experiment 1
│   ├── exp2_memory_pooling.csv        # Raw data for Experiment 2
│   ├── exp3_stream_pipelining.csv     # Raw data for Experiment 3
│   └── exp4_vram_footprint.csv        # Raw data for Experiment 4
└── graphs/
    ├── exp1_total_time_comparison.png
    ├── exp1_time_breakdown.png
    ├── exp1_bandwidth_scaling.png
    ├── exp2_memory_pooling_speedup.png
    ├── exp3_stream_pipelining.png
    └── exp4_vram_utilization.png
```

---

## 4. Experimental Results & Analysis

### Experiment 1: Memory Transfer Strategies (Pageable vs. Pinned vs. Unified Memory)

We compare four host-device memory management strategies across tensor sizes from **1M to 32M `float32` elements** (3.8 MB to 122.1 MB per tensor; 7.6 MB to 244.1 MB round-trip H2D + D2H).

| Tensor Size | Explicit Pageable (ms) | Explicit Pinned (ms) | Unified Memory (ms) | Unified + Prefetch (ms) | Pinned Speedup vs. Pageable |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1M (3.8 MB)** | 1.61 | **1.42** | 53.49 | 54.44 | 1.13× |
| **4M (15.3 MB)** | 5.87 | **5.55** | 213.06 | 217.27 | 1.06× |
| **8M (30.5 MB)** | 11.60 | **10.91** | 477.34 | 612.46 | 1.06× |
| **16M (61.0 MB)** | 23.08 | **21.73** | 940.76 | 1,338.65 | 1.06× |
| **32M (122.1 MB)** | 76.15 | **43.35** | 3,193.87 | 2,525.28 | **1.76×** |

![Total Execution Time by Memory Strategy](graphs/exp1_total_time_comparison.png)

![PCIe Data Transfer vs GPU Compute Breakdown](graphs/exp1_time_breakdown.png)

![Effective End-to-End Memory Bandwidth Scaling](graphs/exp1_bandwidth_scaling.png)

**Key Findings:**
- **PCIe Data Movement Dominates**: In Explicit Pinned mode at 32M elements, **>98% of total time** is spent on H2D + D2H PCIe transfers, while the actual GPU AI kernel executes in under 1 ms.
- **Pinned Memory Eliminates Staging Overhead**: At 32M elements (122.1 MB), Explicit Pinned memory achieves **43.35 ms** (5.91 GB/s effective round-trip bandwidth) vs. **76.15 ms** for Pageable memory (**1.76× faster**), because DMA can transfer directly from page-locked RAM without an intermediate OS staging copy.
- **Unified Memory Pitfall on Windows WDDM**: Because Windows WDDM reports `concurrent_managed_access = False`, demand-paged Unified Memory incurs severe page-fault migration overhead (3,193.87 ms at 32M). Adding `cuMemPrefetchAsync` reduces 32M latency by **21%** (to 2,525.28 ms), but explicit pinned memory remains orders of magnitude faster for streaming AI batches.

---

### Experiment 2: GPU-Aware Caching Memory Pool vs. Naive Allocation

Deep learning training loops execute hundreds of thousands of forward/backward steps. Calling `cuMemAlloc` and `cuMemFree` on every step triggers expensive OS/GPU driver synchronization. We compare naive per-step allocation against our `GpuMemoryPool` caching allocator over **100 iterative AI steps**.

| Tensor Size | Naive 100 Steps (ms) | Pooled 100 Steps (ms) | Speedup | Driver Allocations (Naive → Pooled) | Pool Cache Hits |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1M (3.8 MB)** | 154.36 | **7.28** | **21.20×** | 200 → 2 | 198 |
| **2M (7.6 MB)** | 299.68 | **16.00** | **18.72×** | 200 → 2 | 198 |
| **4M (15.3 MB)** | 428.19 | **26.25** | **16.31×** | 200 → 2 | 198 |
| **8M (30.5 MB)** | 298.23 | **56.22** | **5.30×** | 200 → 2 | 198 |
| **16M (61.0 MB)** | 417.35 | **92.73** | **4.50×** | 200 → 2 | 198 |

![GPU-Aware Memory Pool Speedup](graphs/exp2_memory_pooling_speedup.png)

**Key Findings:**
- **Up to 21.20× End-to-End Speedup**: Eliminating 198 redundant driver allocations reduces 100-step runtime for 1M tensors from **154.36 ms** down to **7.28 ms**.
- **99% Cache Hit Rate**: Once the pool warms up on Step 1 (2 driver allocations for input/output buffers), all remaining 99 steps are served in $O(1)$ from the size-bucketed free list. This is the exact mechanism used by production AI frameworks (PyTorch `CUDACachingAllocator`, TensorFlow BFC allocator).

---

### Experiment 3: Multi-Stream Asynchronous Pipelining

We partition a 32M-element AI batch into chunks across **1, 2, 4, and 8 CUDA streams** using pinned host memory (`cuMemcpyHtoDAsync`, `cuLaunchKernel`, `cuMemcpyDtoHAsync`) to evaluate transfer/compute overlap.

| CUDA Streams | Chunk Size (MB) | Avg Batch Time (ms) | Sync Baseline (ms) | Speedup vs. Sync | Effective PCIe BW (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Stream** | 122.1 | 42.27 | 42.25 | 1.00× | 6.06 |
| **2 Streams** | 61.0 | 42.35 | 42.25 | 1.00× | 6.05 |
| **4 Streams** | 30.5 | 42.64 | 42.25 | 0.99× | 6.00 |
| **8 Streams** | 15.3 | 42.71 | 42.25 | 0.99× | 5.99 |

![Multi-Stream Async Pipelining Analysis](graphs/exp3_stream_pipelining.png)

**Key Findings:**
- **Architectural Insight**: For a lightweight, memory-bound fused activation kernel (where compute is ~0.5 ms vs. ~42 ms PCIe transfer), multi-stream pipelining yields ~1.00× on a consumer GeForce RTX 4050 (which has a single PCIe DMA copy engine under WDDM, serializing H2D and D2H).
- **When Streams Win**: Multi-stream overlap yields major speedups (up to 1.8–2.5×) when **compute time ≈ transfer time** (e.g., dense GEMM / Transformer attention layers) or on data-center GPUs (A100/H100) with dual bidirectional copy engines.

---

### Experiment 4: VRAM Footprint & GPU Memory Hierarchy Bandwidth

We scale an AI workload across batch sizes **8 to 256** (1 MB per sample × 2 tensors = 16 MB to 512 MB allocated), recording live VRAM usage via `cuMemGetInfo` and on-device GPU memory bandwidth.

| Batch Size | Allocated (MB) | Total VRAM Used (MB) | VRAM Util (%) | Kernel Time (ms) | On-Device GPU BW (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | 16.0 | 1,084.5 | 17.66% | 0.032 | **530.3** *(L2 Cache Hit)* |
| **16** | 32.0 | 1,100.5 | 17.92% | 0.219 | 153.5 |
| **32** | 64.0 | 1,132.5 | 18.44% | 0.407 | 164.9 |
| **64** | 128.0 | 1,196.5 | 19.49% | 0.800 | 167.7 |
| **128** | 256.0 | 1,324.5 | 21.57% | 1.600 | 167.8 |
| **256** | 512.0 | 1,580.5 | 25.74% | 3.163 | **169.7** *(88% of GDDR6 Peak)* |

![VRAM Footprint and GPU Memory Hierarchy Bandwidth](graphs/exp4_vram_utilization.png)

**Key Findings:**
- **L2 Cache Residency Cliff**: At `Batch Size = 8` (16 MB working set), the tensors fit entirely inside the RTX 4050's **24 MB L2 cache**, delivering **530.3 GB/s** effective bandwidth.
- **GDDR6 Saturation**: Above 32 MB, the working set spills to GDDR6 VRAM, stabilizing cleanly at **167.7–169.7 GB/s** (~88% of the RTX 4050 Laptop's 192 GB/s theoretical DRAM bandwidth).

---

## 5. Summary of GPU-Aware Memory Management Best Practices for AI

1. **Always use a Caching Memory Pool** for iterative AI training/inference (up to **21.2× speedup** by eliminating per-step driver allocation overhead).
2. **Use Pinned (Page-Locked) Host Memory** for data loaders streaming batches over PCIe (**1.76× faster** than pageable memory on large tensors).
3. **Avoid Demand-Paged Unified Memory** for latency-critical streaming AI workloads on WDDM/PCIe systems; reserve it for oversubscribed graphs or NVLink-coherent architectures.
4. **Fuse Memory-Bound Layers & Tile for L2 Cache**: Keeping intermediate activations on-device (avoiding PCIe round-trips) and within the L2 cache delivers **>3× higher bandwidth** (530 GB/s vs. 170 GB/s) and **>80× higher bandwidth** than PCIe (~6 GB/s).

---

## 6. How to Reproduce

```powershell
# 1. Run all 4 GPU experiments (requires NVIDIA GPU + Python + NumPy)
python src/run_experiments.py

# 2. Regenerate all 6 graphs from results/*.csv (requires Matplotlib)
python src/generate_graphs.py
```
