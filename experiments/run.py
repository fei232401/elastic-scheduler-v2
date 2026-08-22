"""实验入口 —— 运行完整潮汐实验并生成图表/统一对比表（spec §29）。

用法：
    python -m experiments.run --scheduler static --config config/default.yaml
    python -m experiments.run --all                                # 某 config 下跑全部策略并对比
    python -m experiments.run --matrix                             # 3 实验 × 4 策略，输出统一表
    python -m experiments.run --matrix --config config/experiment_a_gpu_bound.yaml

输出：results/<exp>/metrics_<s>.csv, decisions_<s>.csv, summary_<s>.json, plots/*.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.metrics.exporter import export_all  # noqa: E402
from src.simulator.engine import SCHEDULERS, run_experiment  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    "A": "experiment_a_gpu_bound.yaml",
    "B": "experiment_b_network_bound.yaml",
    "C": "experiment_c_mixed_bound.yaml",
}

SUMMARY_FIELDS = [
    "slo_violation_ratio", "inference_avg_p95", "inference_max_p95",
    "resource_switch_count", "training_final_progress", "training_avg_slowdown",
    "network_avg_utilization", "network_max_utilization", "network_congested_ticks",
]


def _run_one(scheduler_name: str, cfg_path: str | None, results_dir: str, out: dict) -> None:
    cfg = load_config(cfg_path)
    collector = run_experiment(scheduler_name, cfg, results_dir)
    summary = collector.summarize()
    collector.write_summary(summary)
    export_all(
        collector.rows,
        pathlib.Path(results_dir) / "plots",
        slo_ms=cfg.get("inference", {}).get("slo_p95_ms", 300),
    )
    out[scheduler_name] = {
        f: getattr(summary, f) for f in SUMMARY_FIELDS
    } | {"training_completed": summary.training_completed}
    print(f"  [{scheduler_name}] SLO违率={summary.slo_violation_ratio:.1%} "
          f"P95={summary.inference_avg_p95:.0f}ms 切换={summary.resource_switch_count} "
          f"训练进度={summary.training_final_progress:.0f}")


def _print_table(exp_label: str, results: dict) -> None:
    rows = list(SCHEDULERS)  # static, elastic, hard_preemption, network_aware
    header = f"Experiment {exp_label} | SLO违率 | 平均P95 | 最大P95 | 切换 | 训练进度 | 训练sl | 网利用率均 | 网利用率峰 | 网拥塞tick"
    print(header)
    print("-" * len(header))
    for name in rows:
        r = results.get(name, {})
        print(
            f"{name:16s} | {r.get('slo_violation_ratio', 0):.1%} | "
            f"{r.get('inference_avg_p95', 0):>5.0f} | {r.get('inference_max_p95', 0):>5.0f} | "
            f"{r.get('resource_switch_count', 0):>3} | {r.get('training_final_progress', 0):>7.0f} | "
            f"{r.get('training_avg_slowdown', 0):>6.2f} | {r.get('network_avg_utilization', 0):>6.2f} | "
            f"{r.get('network_max_utilization', 0):>6.2f} | {r.get('network_congested_ticks', 0):>4}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Elastic Training-Inference Scheduler experiment")
    ap.add_argument("--scheduler", choices=sorted(SCHEDULERS), help="scheduler strategy")
    ap.add_argument("--config", default=None, help="path to YAML config (default: config/default.yaml)")
    ap.add_argument("--results-dir", default="results", help="output directory")
    ap.add_argument("--all", action="store_true", help="run all schedulers on one config and compare")
    ap.add_argument("--matrix", action="store_true", help="run 3 experiments x all schedulers, unified table")
    args = ap.parse_args()

    # run.py 恒以 Mock runtime 运行（run_experiment 默认 mode="mock"）。
    # 若误把 Local 配置喂进来，会静默产出 Mock 结果——显式警告，避免结果被误标为 Local。
    if args.config:
        _cfg = load_config(args.config)
        if "local" in _cfg:
            print("WARNING: run.py 恒用 Mock runtime；检测到配置含 local 段。"
                  "真实 workload 实验请用 `python -m experiments.run_local`。",
                  file=sys.stderr)

    if args.matrix:
        # 单实验矩阵：默认跑全部 A/B/C，可 --config 只跑指定一个
        if args.config:
            exps = {"A": pathlib.Path(args.config)}
        else:
            exps = {k: ROOT / "config" / v for k, v in EXPERIMENTS.items()}
        for label, cfg_path in exps.items():
            out_dir = pathlib.Path(args.results_dir) / f"exp_{label}"
            print(f"\n=== Experiment {label}: {cfg_path.name} ===")
            results: dict = {}
            for name in sorted(SCHEDULERS):
                _run_one(name, str(cfg_path), str(out_dir), results)
            _print_table(label, results)
        return

    out: dict = {}
    if args.all:
        for name in sorted(SCHEDULERS):
            _run_one(name, args.config, args.results_dir, out)
        print("\n=== Comparison ===")
        _print_table("(config)", out)
    else:
        if not args.scheduler:
            ap.error("--scheduler required unless --all or --matrix is set")
        _run_one(args.scheduler, args.config, args.results_dir, out)


if __name__ == "__main__":
    main()
