"""图表导出 —— 四张核心图（spec §22），Agg backend 无显示环境。

  1. traffic.png             QPS vs time
  2. gpu_alloc.png           Training/Inference GPU vs time
  3. p95.png                 P95 vs time + SLO 阈值线
  4. training_throughput.png Training throughput vs time
  5. network.png             Network 利用率 vs time（阶段二，含 100% 容量线）
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.metrics.collector import TickMetrics  # noqa: E402


def _x(rows: list[TickMetrics]) -> list[float]:
    return [r.time_s for r in rows]


def _fig(name: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.tight_layout()
    return fig, ax


def _save(fig: plt.Figure, out_dir: pathlib.Path, name: str) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_traffic(rows: list[TickMetrics], out_dir: pathlib.Path) -> pathlib.Path:
    fig, ax = _fig("traffic")
    ax.plot(_x(rows), [r.qps for r in rows], color="#1f77b4", lw=1.4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("QPS"); ax.set_title("Inference Traffic (tidal)")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "traffic.png")


def plot_gpu_alloc(rows: list[TickMetrics], out_dir: pathlib.Path) -> pathlib.Path:
    fig, ax = _fig("gpu_alloc")
    ax.plot(_x(rows), [r.training_gpu for r in rows], label="Training", color="#2ca02c", lw=1.6)
    ax.plot(_x(rows), [r.inference_gpu for r in rows], label="Inference", color="#ff7f0e", lw=1.6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("GPU count")
    ax.set_title("GPU Allocation over time")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, out_dir, "gpu_alloc.png")


def plot_p95(rows: list[TickMetrics], slo_ms: float, out_dir: pathlib.Path) -> pathlib.Path:
    fig, ax = _fig("p95")
    ax.plot(_x(rows), [r.inference_p95 for r in rows], color="#d62728", lw=1.4, label="P95")
    ax.axhline(slo_ms, color="#7f7f7f", ls="--", lw=1.2, label=f"SLO {slo_ms:.0f}ms")
    ax.set_xlabel("time (s)"); ax.set_ylabel("latency (ms)")
    ax.set_title("Inference P95 vs SLO")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, out_dir, "p95.png")


def plot_training_throughput(rows: list[TickMetrics], out_dir: pathlib.Path) -> pathlib.Path:
    fig, ax = _fig("training_throughput")
    ax.plot(_x(rows), [r.training_throughput for r in rows], color="#9467bd", lw=1.4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("throughput (work/tick)")
    ax.set_title("Training throughput over time")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "training_throughput.png")


def plot_network(rows: list[TickMetrics], out_dir: pathlib.Path) -> pathlib.Path:
    fig, ax = _fig("network")
    ax.plot(_x(rows), [r.network_utilization for r in rows], color="#17becf", lw=1.4, label="utilization")
    ax.axhline(1.0, color="#7f7f7f", ls="--", lw=1.2, label="capacity (100%)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("network utilization (demand/capacity)")
    ax.set_title("Network utilization over time")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, out_dir, "network.png")


def export_all(rows: list[TickMetrics], out_dir: str | pathlib.Path, slo_ms: float = 300.0) -> list[pathlib.Path]:
    out = pathlib.Path(out_dir)
    return [
        plot_traffic(rows, out),
        plot_gpu_alloc(rows, out),
        plot_p95(rows, slo_ms, out),
        plot_training_throughput(rows, out),
        plot_network(rows, out),
    ]
