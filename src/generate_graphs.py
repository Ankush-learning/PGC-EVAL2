"""
generate_graphs.py
Generates publication-quality graphs from the measured GPU experiment CSVs into graphs/.
"""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Explicit_Pageable": "#2563EB",
    "Explicit_Pinned": "#10B981",
    "Unified_Memory": "#EF4444",
    "Unified_Prefetch": "#F59E0B",
}


def load_csv(path: Path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def plot_exp1(rows, gpu_name):
    sizes = sorted(list({int(r["elements_m"]) for r in rows}))
    labels = [f"{s}M\n({s*4/1.048576:.0f} MB)" for s in sizes]
    strategies = ["Explicit_Pageable", "Explicit_Pinned", "Unified_Memory", "Unified_Prefetch"]
    nice = {
        "Explicit_Pageable": "Explicit (Pageable)",
        "Explicit_Pinned": "Explicit (Pinned)",
        "Unified_Memory": "Unified Memory",
        "Unified_Prefetch": "Unified + Prefetch",
    }

    # --- Graph 1: Total Time Comparison (Log Scale) ---
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(sizes))
    w = 0.19
    for idx, strat in enumerate(strategies):
        vals = [float(next(r["total_ms"] for r in rows if int(r["elements_m"]) == s and r["strategy"] == strat)) for s in sizes]
        bars = ax.bar(x + (idx - 1.5) * w, vals, width=w, label=nice[strat], color=COLORS[strat], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.1f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Execution Time (ms, log scale)")
    ax.set_xlabel("AI Tensor Size (Elements & Memory per Tensor)")
    ax.set_title(f"Exp 1: Total Execution Time by Memory Strategy ({gpu_name})")
    ax.legend(frameon=True, facecolor="white", ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "exp1_total_time_comparison.png")
    plt.close(fig)

    # --- Graph 2: Explicit Pageable vs Pinned Time Breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, strat, title in zip(
        axes,
        ["Explicit_Pageable", "Explicit_Pinned"],
        ["Explicit Pageable Memory", "Explicit Pinned (Page-Locked) Memory"],
    ):
        h2d = [float(next(r["h2d_ms"] for r in rows if int(r["elements_m"]) == s and r["strategy"] == strat)) for s in sizes]
        kern = [float(next(r["kernel_ms"] for r in rows if int(r["elements_m"]) == s and r["strategy"] == strat)) for s in sizes]
        d2h = [float(next(r["d2h_ms"] for r in rows if int(r["elements_m"]) == s and r["strategy"] == strat)) for s in sizes]

        ax.bar(x, h2d, width=0.55, label="H2D Copy (PCIe)", color="#3B82F6")
        ax.bar(x, kern, width=0.55, bottom=h2d, label="GPU AI Kernel", color="#10B981")
        ax.bar(x, d2h, width=0.55, bottom=np.array(h2d) + np.array(kern), label="D2H Copy (PCIe)", color="#F59E0B")
        for i, (a, b, c) in enumerate(zip(h2d, kern, d2h)):
            tot = a + b + c
            ax.text(i, tot + 1.2, f"{tot:.1f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}M" for s in sizes])
        ax.set_xlabel("Tensor Size (Million Floats)")
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=True, facecolor="white")

    axes[0].set_ylabel("Execution Time (ms)")
    fig.suptitle(f"Exp 1: PCIe Data Transfer vs GPU Compute Breakdown ({gpu_name})", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "exp1_time_breakdown.png", bbox_inches="tight")
    plt.close(fig)

    # --- Graph 3: Effective End-to-End Bandwidth Scaling ---
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    markers = {"Explicit_Pageable": "o", "Explicit_Pinned": "s", "Unified_Memory": "^", "Unified_Prefetch": "D"}
    for strat in strategies:
        bws = [float(next(r["effective_bw_gbs"] for r in rows if int(r["elements_m"]) == s and r["strategy"] == strat)) for s in sizes]
        ax.plot(sizes, bws, marker=markers[strat], linewidth=2.2, markersize=7, label=nice[strat], color=COLORS[strat])
    ax.set_xlabel("Tensor Size (Million Float32 Elements)")
    ax.set_ylabel("Effective End-to-End Bandwidth (GB/s)")
    ax.set_title(f"Exp 1: Effective End-to-End Memory Bandwidth Scaling ({gpu_name})")
    ax.set_xticks(sizes)
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "exp1_bandwidth_scaling.png")
    plt.close(fig)


def plot_exp2(rows, gpu_name):
    sizes = [int(r["elements_m"]) for r in rows]
    naive = [float(r["naive_total_ms"]) for r in rows]
    pooled = [float(r["pooled_total_ms"]) for r in rows]
    speedup = [float(r["speedup"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(sizes))
    w = 0.36

    b1 = ax1.bar(x - w / 2, naive, width=w, label="Naive (cuMemAlloc/Free per step)", color="#EF4444")
    b2 = ax1.bar(x + w / 2, pooled, width=w, label="GPU-Aware Memory Pool", color="#10B981")
    for b, v in zip(b1, naive):
        ax1.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
    for b, v in zip(b2, pooled):
        ax1.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}M" for s in sizes])
    ax1.set_xlabel("Tensor Size (Million Floats)")
    ax1.set_ylabel("Total Time for 100 AI Steps (ms)")
    ax1.set_title("100-Step Allocation + Compute Time")
    ax1.legend(frameon=True, facecolor="white")

    bars = ax2.bar(x, speedup, width=0.5, color="#6366F1")
    for b, v in zip(bars, speedup):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}M" for s in sizes])
    ax2.set_xlabel("Tensor Size (Million Floats)")
    ax2.set_ylabel("Speedup Factor (Naive / Pooled)")
    ax2.set_title("GPU-Aware Memory Pool Speedup")
    ax2.set_ylim(0, max(speedup) * 1.18)

    fig.suptitle(f"Exp 2: GPU-Aware Caching Memory Pool vs Naive Allocation ({gpu_name})", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "exp2_memory_pooling_speedup.png", bbox_inches="tight")
    plt.close(fig)


def plot_exp3(rows, gpu_name):
    streams = [int(r["num_streams"]) for r in rows]
    times = [float(r["avg_time_ms"]) for r in rows]
    bws = [float(r["effective_bw_gbs"]) for r in rows]
    baseline = float(rows[0]["baseline_sync_ms"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(len(streams))

    bars = ax1.bar(x, times, width=0.5, color="#2563EB", label="Async Multi-Stream")
    ax1.axhline(baseline, color="#EF4444", linestyle="--", linewidth=1.8, label=f"Sync Baseline ({baseline:.2f} ms)")
    for b, v in zip(bars, times):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f} ms", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s} Stream{'s' if s>1 else ''}" for s in streams])
    ax1.set_ylabel("Total Batch Time (ms)")
    ax1.set_title("Latency vs Number of CUDA Streams")
    ax1.set_ylim(0, max(times) * 1.2)
    ax1.legend(frameon=True, facecolor="white")

    ax2.plot(streams, bws, marker="o", linewidth=2.4, markersize=8, color="#10B981")
    for s, bw in zip(streams, bws):
        ax2.text(s, bw + 0.08, f"{bw:.2f} GB/s", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(streams)
    ax2.set_xlabel("Number of CUDA Streams")
    ax2.set_ylabel("Effective PCIe Bandwidth (GB/s)")
    ax2.set_title("Effective Throughput (32M Elements)")
    ax2.set_ylim(5.5, 6.6)

    fig.suptitle(f"Exp 3: Multi-Stream Async Pipelining Analysis ({gpu_name})", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "exp3_stream_pipelining.png", bbox_inches="tight")
    plt.close(fig)


def plot_exp4(rows, gpu_name):
    bs = [int(r["batch_size"]) for r in rows]
    alloc_mb = [float(r["allocated_mb"]) for r in rows]
    vram_mb = [float(r["total_vram_used_mb"]) for r in rows]
    bw = [float(r["gpu_mem_bw_gbs"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(bs))

    ax1.bar(x, vram_mb, width=0.55, color="#8B5CF6", alpha=0.85, label="Total VRAM Used (MB)")
    ax1.bar(x, alloc_mb, width=0.55, color="#10B981", label="Workload Tensor Allocation (MB)")
    for i, (v, a) in enumerate(zip(vram_mb, alloc_mb)):
        ax1.text(i, v + 25, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"BS={b}" for b in bs])
    ax1.set_xlabel("AI Workload Batch Size")
    ax1.set_ylabel("GPU Memory (MB)")
    ax1.set_title("VRAM Footprint Scaling")
    ax1.legend(frameon=True, facecolor="white")

    ax2.plot(bs, bw, marker="s", linewidth=2.4, markersize=8, color="#F59E0B")
    for b, val in zip(bs, bw):
        ax2.annotate(f"{val:.1f} GB/s", (b, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax2.axhline(192.0, color="#64748B", linestyle=":", linewidth=1.6, label="RTX 4050 Theoretical DRAM Peak (~192 GB/s)")
    ax2.set_xlabel("AI Workload Batch Size")
    ax2.set_ylabel("On-Device GPU Memory Bandwidth (GB/s)")
    ax2.set_title("L2 Cache Residency vs GDDR6 DRAM Ceiling")
    ax2.set_ylim(0, max(bw) * 1.18)
    ax2.legend(frameon=True, facecolor="white")

    fig.suptitle(f"Exp 4: VRAM Footprint & GPU Memory Hierarchy Bandwidth ({gpu_name})", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "exp4_vram_utilization.png", bbox_inches="tight")
    plt.close(fig)


def main():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "system_info.json", "r") as f:
        info = json.load(f)
    gpu_name = info["gpu_name"]

    plot_exp1(load_csv(RESULTS_DIR / "exp1_transfer_strategies.csv"), gpu_name)
    plot_exp2(load_csv(RESULTS_DIR / "exp2_memory_pooling.csv"), gpu_name)
    plot_exp3(load_csv(RESULTS_DIR / "exp3_stream_pipelining.csv"), gpu_name)
    plot_exp4(load_csv(RESULTS_DIR / "exp4_vram_footprint.csv"), gpu_name)
    print("Generated all 6 graphs in:", GRAPHS_DIR)


if __name__ == "__main__":
    main()
