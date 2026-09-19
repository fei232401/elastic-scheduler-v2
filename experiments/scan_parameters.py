"""Phase 3.3 T4 — 参数扫描：沿 6 个参数找 振荡 / 稳定 / 过度保守 区域（spec §三十五 #4）。

扫描清单（参数 → 基准场景 → 调度器）：
  1. slo_threshold      inference.slo_p95_ms              default（GPU-bound 全潮汐）  elastic
  2. min_hold           scheduler.min_gpu_hold_duration_ticks   default                 elastic
  3. scale_step         scheduler.max_scale_step          default                     elastic
  4. congestion_start   network.congestion_start_util     moderate（网络局部拥塞）    network_aware
  5. max_slowdown       training.max_allowed_slowdown     exp B（网络结构性拥塞）      elastic
  6. ramp               traffic.schedule 上升斜率         default 潮汐骨架             elastic

确定性：Mock，seed 42，840 tick。输出：results/scan/<param>.json + 控制台表。
用法: python -m experiments.scan_parameters
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.simulator.engine import run_experiment

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "scan"

FIELDS = [
    "slo_violation_ratio", "inference_avg_p95", "resource_switch_count",
    "training_final_progress", "training_avg_slowdown", "network_congested_ticks",
]

DEFAULT_TIDE = [
    {"min": 0, "qps": 10}, {"min": 5, "qps": 12}, {"min": 10, "qps": 18},
    {"min": 15, "qps": 25}, {"min": 20, "qps": 35}, {"min": 25, "qps": 45},
    {"min": 30, "qps": 55}, {"min": 35, "qps": 60}, {"min": 40, "qps": 55},
    {"min": 45, "qps": 40}, {"min": 50, "qps": 25}, {"min": 60, "qps": 15},
    {"min": 70, "qps": 10},
]
RAMP_VARIANTS = {
    "fast_15min": [
        {"min": 0, "qps": 10}, {"min": 5, "qps": 25}, {"min": 10, "qps": 45},
        {"min": 15, "qps": 60}, {"min": 20, "qps": 60}, {"min": 40, "qps": 60},
        {"min": 45, "qps": 42}, {"min": 50, "qps": 25}, {"min": 60, "qps": 15},
        {"min": 70, "qps": 10},
    ],
    "med_25min": [
        {"min": 0, "qps": 10}, {"min": 10, "qps": 28}, {"min": 20, "qps": 50},
        {"min": 25, "qps": 60}, {"min": 35, "qps": 60}, {"min": 40, "qps": 48},
        {"min": 45, "qps": 35}, {"min": 50, "qps": 22}, {"min": 60, "qps": 14},
        {"min": 70, "qps": 10},
    ],
    "default_35min": DEFAULT_TIDE,
    "gentle_50min": [
        {"min": 0, "qps": 10}, {"min": 10, "qps": 15}, {"min": 25, "qps": 30},
        {"min": 40, "qps": 50}, {"min": 50, "qps": 60}, {"min": 55, "qps": 58},
        {"min": 60, "qps": 40}, {"min": 65, "qps": 20}, {"min": 70, "qps": 10},
    ],
}


def _set(cfg: dict, path: str, value) -> None:
    node = cfg
    for part in path.split(".")[:-1]:
        node = node.setdefault(part, {})
    node[path.split(".")[-1]] = value


def _run(scheduler: str, base: str, path: str, value) -> dict:
    cfg = load_config(str(ROOT / "config" / base))
    _set(cfg, path, value)
    coll = run_experiment(scheduler, cfg, str(OUT))
    s = coll.summarize()
    return {"value": value} | {
        f: getattr(s, f) for f in FIELDS
    } | {"scheduler": scheduler, "base": base, "param": path}


def _print_table(rows: list[dict], value_label: str) -> None:
    header = (f"{value_label:>14} | SLO违率 | P95均 | 切换 | 训练进度 | 训练sl | 拥塞tick")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['value']):>14} | {r['slo_violation_ratio']:>6.1%} | {r['inference_avg_p95']:>5.0f} | "
            f"{r['resource_switch_count']:>3} | {r['training_final_progress']:>7.0f} | "
            f"{r['training_avg_slowdown']:>6.2f} | {r['network_congested_ticks']:>4}"
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    specs = [
        ("slo_threshold", "elastic", "default.yaml", "inference.slo_p95_ms", [200, 250, 300, 400, 500]),
        ("min_hold", "elastic", "default.yaml", "scheduler.min_gpu_hold_duration_ticks", [0, 6, 12, 24, 48, 96]),
        ("scale_step", "elastic", "default.yaml", "scheduler.max_scale_step", [1, 2, 3, 4]),
        ("congestion_start", "network_aware", "experiment_moderate_network.yaml",
         "network.congestion_start_util", [0.3, 0.5, 0.7, 0.9]),
        ("max_slowdown", "elastic", "experiment_b_network_bound.yaml",
         "training.max_allowed_slowdown", [1.2, 1.5, 2.0, 3.0, 5.0]),
    ]

    all_results = {}
    for name, scheduler, base, path, values in specs:
        print(f"\n=== param: {name}  ({path} @ {base}, scheduler={scheduler}) ===")
        rows = [_run(scheduler, base, path, v) for v in values]
        _print_table(rows, "value")
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2))
        all_results[name] = rows

    print(f"\n=== param: ramp  (traffic.schedule 上升斜率 @ default.yaml, scheduler=elastic) ===")
    ramp_rows = []
    for label, schedule in RAMP_VARIANTS.items():
        cfg = load_config(str(ROOT / "config" / "default.yaml"))
        cfg["traffic"]["schedule"] = schedule
        coll = run_experiment("elastic", cfg, str(OUT))
        s = coll.summarize()
        ramp_rows.append({"value": label} | {f: getattr(s, f) for f in FIELDS})
    _print_table(ramp_rows, "ramp")
    (OUT / "ramp.json").write_text(json.dumps(ramp_rows, indent=2))
    all_results["ramp"] = ramp_rows

    (OUT / "scan_all.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n-> results/scan/scan_all.json")


if __name__ == "__main__":
    main()
