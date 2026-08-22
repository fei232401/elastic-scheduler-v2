"""Metrics 采集器 —— 每 tick 采集、汇总、落盘（spec §21/§22/§29）。

产出：
  - metrics_<scheduler>.csv     每 tick 一行，全量指标
  - decisions_<scheduler>.csv   每次配额变化的决策记录
  - summary_<scheduler>.json    汇总：SLO 违约、GPU 利用、切换次数、训练完成等
"""
from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass, field


@dataclass
class TickMetrics:
    tick: int
    time_s: float
    qps: float
    inference_capacity: float
    inference_p50: float
    inference_p95: float
    inference_p99: float
    inference_queue: int
    inference_state: str
    slo_violated: bool
    training_gpu: int
    inference_gpu: int
    training_throughput: float
    training_progress: float
    training_remaining: float
    training_est_completion: float
    training_slowdown: float
    training_state: str
    gpu_utilization: float
    unused_gpu: int
    network_demand: float = 0.0
    network_utilization: float = 0.0
    network_latency_ms: float = 0.0
    network_congested: bool = False
    network_headroom: float = 0.0


@dataclass
class Summary:
    total_ticks: int
    total_seconds: int
    slo_violation_ticks: int
    slo_violation_ratio: float
    resource_switch_count: int
    training_final_progress: float
    training_completed: bool
    training_avg_slowdown: float
    gpu_avg_utilization: float
    gpu_min_utilization: float
    gpu_max_utilization: float
    inference_avg_p95: float
    inference_max_p95: float
    training_final_throughput: float
    network_avg_utilization: float = 0.0
    network_max_utilization: float = 0.0
    network_congested_ticks: int = 0
    details: dict = field(default_factory=dict)


class MetricsCollector:
    CSV_FIELDS = [
        "tick", "time_s", "qps", "inference_capacity", "inference_p50", "inference_p95",
        "inference_p99", "inference_queue", "inference_state", "slo_violated",
        "training_gpu", "inference_gpu", "training_throughput", "training_progress",
        "training_remaining", "training_est_completion", "training_slowdown",
        "training_state", "gpu_utilization", "unused_gpu",
        "network_demand", "network_utilization", "network_latency_ms", "network_congested",
        "network_headroom",
    ]

    def __init__(self, scheduler_name: str, results_dir: str | pathlib.Path = "results") -> None:
        self.scheduler_name = scheduler_name
        self.dir = pathlib.Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[TickMetrics] = []
        self.decisions: list[dict] = []

    def record(self, m: TickMetrics) -> None:
        self.rows.append(m)

    def record_decision(self, decision: dict) -> None:
        self.decisions.append(decision)

    # ---- export ----
    def _csv_path(self, kind: str) -> pathlib.Path:
        return self.dir / f"{kind}_{self.scheduler_name}.csv"

    def write_csv(self) -> pathlib.Path:
        p = self._csv_path("metrics")
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            w.writeheader()
            for r in self.rows:
                w.writerow({
                    "tick": r.tick, "time_s": round(r.time_s, 1), "qps": round(r.qps, 3),
                    "inference_capacity": round(r.inference_capacity, 1),
                    "inference_p50": round(r.inference_p50, 2),
                    "inference_p95": round(r.inference_p95, 2),
                    "inference_p99": round(r.inference_p99, 2),
                    "inference_queue": r.inference_queue, "inference_state": r.inference_state,
                    "slo_violated": int(r.slo_violated), "training_gpu": r.training_gpu,
                    "inference_gpu": r.inference_gpu,
                    "training_throughput": round(r.training_throughput, 3),
                    "training_progress": round(r.training_progress, 1),
                    "training_remaining": round(r.training_remaining, 1),
                    "training_est_completion": round(r.training_est_completion, 1),
                    "training_slowdown": round(r.training_slowdown, 3),
                    "training_state": r.training_state,
                    "gpu_utilization": round(r.gpu_utilization, 3),
                    "unused_gpu": r.unused_gpu,
                    "network_demand": round(r.network_demand, 1),
                    "network_utilization": round(r.network_utilization, 3),
                    "network_latency_ms": round(r.network_latency_ms, 2),
                    "network_congested": int(r.network_congested),
                    "network_headroom": round(r.network_headroom, 1),
                })
        return p

    def write_decisions(self) -> pathlib.Path:
        p = self.dir / f"decisions_{self.scheduler_name}.csv"
        fields = ["time_s", "old_training_gpu", "new_training_gpu", "old_inference_gpu",
                  "new_inference_gpu", "reason", "inference_p95", "network_utilization",
                  "timestamp", "scheduler", "training_state"]
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for d in self.decisions:
                w.writerow(d)
        return p

    def summarize(self) -> Summary:
        if not self.rows:
            raise ValueError("no metrics recorded")
        viol = [r for r in self.rows if r.slo_violated]
        gpu_util = [r.gpu_utilization for r in self.rows]
        p95 = [r.inference_p95 for r in self.rows]
        slow = [r.training_slowdown for r in self.rows if r.training_state in ("RUNNING", "DEGRADED")]
        last = self.rows[-1]
        return Summary(
            total_ticks=len(self.rows),
            total_seconds=int(last.time_s),
            slo_violation_ticks=len(viol),
            slo_violation_ratio=round(len(viol) / len(self.rows), 4),
            resource_switch_count=len(self.decisions),
            training_final_progress=round(last.training_progress, 1),
            training_completed=bool(last.training_state == "COMPLETED"),
            training_avg_slowdown=round(sum(slow) / len(slow), 3) if slow else 0.0,
            gpu_avg_utilization=round(sum(gpu_util) / len(gpu_util), 4),
            gpu_min_utilization=round(min(gpu_util), 4),
            gpu_max_utilization=round(max(gpu_util), 4),
            inference_avg_p95=round(sum(p95) / len(p95), 2),
            inference_max_p95=round(max(p95), 2),
            training_final_throughput=round(last.training_throughput, 3),
            network_avg_utilization=round(
                sum(r.network_utilization for r in self.rows) / len(self.rows), 4
            ),
            network_max_utilization=round(
                max(r.network_utilization for r in self.rows), 4
            ),
            network_congested_ticks=sum(1 for r in self.rows if r.network_congested),
        )

    def write_summary(self, summary: Summary | None = None) -> pathlib.Path:
        s = summary or self.summarize()
        data = {
            "scheduler": self.scheduler_name,
            "total_ticks": s.total_ticks,
            "total_seconds": s.total_seconds,
            "slo_violation_ticks": s.slo_violation_ticks,
            "slo_violation_ratio": s.slo_violation_ratio,
            "resource_switch_count": s.resource_switch_count,
            "training_final_progress": s.training_final_progress,
            "training_completed": s.training_completed,
            "training_avg_slowdown": s.training_avg_slowdown,
            "gpu_avg_utilization": s.gpu_avg_utilization,
            "gpu_min_utilization": s.gpu_min_utilization,
            "gpu_max_utilization": s.gpu_max_utilization,
            "inference_avg_p95": s.inference_avg_p95,
            "inference_max_p95": s.inference_max_p95,
            "training_final_throughput": s.training_final_throughput,
            "network_avg_utilization": s.network_avg_utilization,
            "network_max_utilization": s.network_max_utilization,
            "network_congested_ticks": s.network_congested_ticks,
        }
        p = self.dir / f"summary_{self.scheduler_name}.json"
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def rows_as_lists(self) -> list[list]:
        return [[
            r.tick, r.time_s, r.qps, r.inference_capacity, r.inference_p50, r.inference_p95,
            r.inference_p99, r.inference_queue, r.inference_state, int(r.slo_violated),
            r.training_gpu, r.inference_gpu, r.training_throughput, r.training_progress,
            r.training_remaining, r.training_est_completion, r.training_slowdown,
            r.training_state, r.gpu_utilization, r.unused_gpu,
            round(r.network_demand, 1), round(r.network_utilization, 3),
            round(r.network_latency_ms, 2), int(r.network_congested),
            round(r.network_headroom, 1),
        ] for r in self.rows]
