"""Step 10 — Mock vs Local 统一实验（Phase 3.2，spec §16/§18/§24）。

同一 4 策略（static/elastic/hard_preemption/network_aware）分别在：
  - Mock runtime（experiment_b_network_bound.yaml，数学模型，毫秒级）
  - Local runtime（local_experiment.yaml，真实 PyTorch + HTTP + NetemNetwork，wall-clock）
上跑同一形态的潮汐场景，产出同构 metrics/summary → 逐项对比，暴露 Reality Gap。

用法:
  python -m experiments.run_local --scheduler elastic      # 单个策略（Local）
  python -m experiments.run_local --all                    # 4 策略（Local）
  python -m experiments.run_local --compare                # Local(4) + Mock B(4) + 对比表
输出: results/local/metrics_<s>.csv, summary_<s>.json, decisions_<s>.csv
      results/local/mock_vs_local.json（--compare）
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
LOCAL_CFG = ROOT / "config" / "local_experiment.yaml"
MOCK_B_CFG = ROOT / "config" / "experiment_b_network_bound.yaml"
OUT_DIR = ROOT / "results" / "local"

SUMMARY_FIELDS = [
    "slo_violation_ratio", "inference_avg_p95", "inference_max_p95",
    "resource_switch_count", "training_final_progress", "training_avg_slowdown",
    "network_avg_utilization", "network_max_utilization", "network_congested_ticks",
]


def _run_one(scheduler_name: str, cfg_path, results_dir: pathlib.Path,
             mode: str, out: dict) -> None:
    cfg = load_config(str(cfg_path))
    collector = run_experiment(scheduler_name, cfg, str(results_dir), mode=mode)
    summary = collector.summarize()
    collector.write_summary(summary)
    export_all(
        collector.rows,
        results_dir / "plots",
        slo_ms=cfg.get("inference", {}).get("slo_p95_ms", 15),
    )
    out[scheduler_name] = {f: getattr(summary, f) for f in SUMMARY_FIELDS} | {
        "training_completed": summary.training_completed,
        "mode": mode,
        "config": pathlib.Path(cfg_path).name,
    }
    print(f"  [{mode:5s}] {scheduler_name:16s} SLO违率={summary.slo_violation_ratio:.1%} "
          f"P95={summary.inference_avg_p95:.0f}ms 切换={summary.resource_switch_count} "
          f"训练进度={summary.training_final_progress:.0f} "
          f"网拥塞={summary.network_congested_ticks}")


def _print_table(label: str, results: dict) -> None:
    rows = list(SCHEDULERS)
    header = (f"{label} | SLO违率 | 平均P95 | 切换 | 训练进度 | 训练sl | "
              f"网利用率均 | 网拥塞tick")
    print(header)
    print("-" * len(header))
    for name in rows:
        r = results.get(name, {})
        print(
            f"{name:16s} | {r.get('slo_violation_ratio', 0):.1%} | "
            f"{r.get('inference_avg_p95', 0):>5.0f} | {r.get('resource_switch_count', 0):>3} | "
            f"{r.get('training_final_progress', 0):>7.0f} | {r.get('training_avg_slowdown', 0):>6.2f} | "
            f"{r.get('network_avg_utilization', 0):>6.2f} | {r.get('network_congested_ticks', 0):>4}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10: Mock vs Local unified experiment")
    ap.add_argument("--scheduler", choices=sorted(SCHEDULERS))
    ap.add_argument("--all", action="store_true", help="run all 4 schedulers on Local")
    ap.add_argument("--compare", action="store_true",
                    help="Local(4) + Mock B(4) + comparison table")
    ap.add_argument("--config", default=str(LOCAL_CFG))
    ap.add_argument("--mock-config", default=str(MOCK_B_CFG))
    args = ap.parse_args()

    local_cfg = args.config
    mock_cfg = args.mock_config

    if args.compare:
        local_out: dict = {}
        mock_out: dict = {}
        print(f"\n=== Mock runtime (exp B network-bound) — {pathlib.Path(mock_cfg).name} ===")
        for name in sorted(SCHEDULERS):
            _run_one(name, mock_cfg, ROOT / "results" / "mock_B", "mock", mock_out)
        _print_table("Mock B", mock_out)
        print(f"\n=== Local runtime (network-bound, real workload) — {pathlib.Path(local_cfg).name} ===")
        for name in sorted(SCHEDULERS):
            _run_one(name, local_cfg, OUT_DIR, "local", local_out)
        _print_table("Local", local_out)

        print("\n=== Mock vs Local（各 summary 字段比值 / 差异）===")
        print(f"{'field':28s} {'MockB':>12s} {'Local':>12s} {'注释'}")
        for f in SUMMARY_FIELDS[:6]:
            m = mock_out.get("network_aware", {}).get(f, 0)
            l = local_out.get("network_aware", {}).get(f, 0)
            note = ""
            if f == "slo_violation_ratio" and abs(l - m) > 0.15:
                note = "SLO 触发机制不同（Local GPU P95 带窄，过载纯网络驱动）"
            if f == "training_final_progress" and l > 0 and m > 0:
                note = f"量纲不同（Mock 10^4 工作量 / Local 10^8）"
            print(f"{f:28s} {m:>12.4f} {l:>12.4f}  {note}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "mock_vs_local.json").write_text(
            json.dumps({"mock_B": mock_out, "local": local_out}, indent=2))
        print(f"\n对比数据 -> results/local/mock_vs_local.json")
        return

    out: dict = {}
    targets = sorted(SCHEDULERS) if args.all else [args.scheduler]
    if not targets or None in targets:
        ap.error("--scheduler required unless --all/--compare")
    print(f"\n=== Local runtime — {pathlib.Path(local_cfg).name} ===")
    for name in targets:
        _run_one(name, local_cfg, OUT_DIR, "local", out)
    if args.all:
        _print_table("Local", out)


if __name__ == "__main__":
    main()
