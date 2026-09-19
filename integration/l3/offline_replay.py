#!/usr/bin/env python3
r"""L3 M2 对照基线 · 离线回放(理想反馈)—— 同一决策引擎 + 同一配置,但分配即时生效。

与 m2_thin_controller 的唯一差异 = 反馈回路:
  - 现场(L3):每 tick 从真实集群读 actual 分配,决策 patch 后要等真实执行(KWOK/Kueue)。
  - 离线(本脚本):决策 new_* 在同 tick 立即生效(理想假设,同 mock 引擎语义)。

用途:量化"真实反馈滞后"的代价。现场 run_full.json 的 slo_violation_pct 与离线结果的差,
就是理想反馈假设在万卡尺度下的失真。诚实结论:mock/离线 3.5% vs 现场 36.9% —— 不是 bug,
是"决策→执行→生效"真实回路时间常数的暴露。

用法(仓库根):
  python integration/l3/offline_replay.py --config integration/l3/l3_default.yaml --out integration/l3/results/offline_replay.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(_REPO_ROOT))

from src.config import get, load_config
from src.scheduler.elastic import ElasticScheduler
from src.scheduler.base import SchedulerContext
from src.workload.inference import MockInferenceService
from src.workload.training import MockTrainingJob
from src.workload.traffic import TrafficGenerator
from integration.l3.predictive import L3PredictiveScheduler

STRATEGIES = {"elastic", "static", "predictive"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "integration/l3/l3_default.yaml"))
    ap.add_argument("--scheduler", default="elastic", choices=sorted(STRATEGIES))
    ap.add_argument("--out", default=str(_REPO_ROOT / "integration/l3/results/offline_replay.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    ticks = get(cfg, "simulation.duration_ticks", 840)
    tick_seconds = get(cfg, "simulation.tick_seconds", 5)
    seed = get(cfg, "simulation.seed", 42)
    total_gpus = get(cfg, "simulation.total_gpus", 10000)
    init_train = get(cfg, "training.initial_gpu", 8000)
    min_gpu = get(cfg, "training.min_gpu", 4000)
    init_infer = total_gpus - init_train

    if args.scheduler == "static":
        scheduler = None
    elif args.scheduler == "predictive":
        scheduler = L3PredictiveScheduler(cfg)
        scheduler.reset()
    else:
        scheduler = ElasticScheduler(cfg)
        scheduler.reset()
    infer = MockInferenceService(cfg)
    training = MockTrainingJob(cfg)
    traffic = TrafficGenerator(cfg, rng=random.Random(seed))

    cur_train, cur_infer = init_train, init_infer
    infer.set_gpus(cur_infer)

    records = []
    gpu_h_train = gpu_h_infer = 0.0
    slo_violated = 0
    moves_down = moves_up = 0

    for t in range(ticks):
        now_s = t * tick_seconds
        qps = traffic.qps_at_second(now_s)

        infer.set_gpus(cur_infer)
        infer.step(qps)
        training.gpus = cur_train

        ctx = SchedulerContext(
            training_gpus=cur_train,
            inference_gpus=cur_infer,
            training_min_gpu=min_gpu,
            training_initial_gpu=init_train,
            inference_p95_ms=infer.p95_ms,
            slo_p95_ms=infer.slo_p95_ms,
            inference_state=infer.state.value,
            training_throughput=training.throughput,
            training_slowdown=training.slowdown_vs_initial,
            training_would_allow_degradation=training.would_allow_degradation(cur_train),
            inference_overload_counter=infer._overload_counter,
            inference_recovery_counter=infer._recovery_counter,
            now_s=now_s,
            runtime=None,
            extra={"would_allow_degradation": training.would_allow_degradation},
        )
        decision = None
        if scheduler is not None:
            decision = scheduler.step(ctx)
            if decision.changed:
                if decision.new_training_gpu < cur_train:
                    moves_down += 1
                else:
                    moves_up += 1
                cur_train = decision.new_training_gpu
                cur_infer = decision.new_inference_gpu

        gpu_h_train += cur_train * tick_seconds / 3600.0
        gpu_h_infer += cur_infer * tick_seconds / 3600.0
        if infer.p95_ms > infer.slo_p95_ms:
            slo_violated += 1
        records.append({
            "tick": t, "minute": round(now_s / 60.0, 2), "qps": round(qps, 1),
            "p95_ms": round(infer.p95_ms, 1), "state": infer.state.value,
            "train": int(cur_train), "infer": int(cur_infer),
            "changed": bool(decision.changed) if decision else False,
            "reason": decision.reason if decision else "static",
        })

    summary = {
        "config": str(Path(args.config).name),
        "scheduler": args.scheduler,
        "mode": "ideal-feedback (offline)",
        "total_gpus": total_gpus, "ticks": ticks, "seed": seed,
        "final_train": cur_train, "final_infer": cur_infer,
        "gpu_h_train": round(gpu_h_train, 1), "gpu_h_infer": round(gpu_h_infer, 1),
        "gpu_h_total": round(gpu_h_train + gpu_h_infer, 1),
        "slo_violation_ticks": slo_violated,
        "slo_violation_pct": round(100.0 * slo_violated / ticks, 1),
        "moves_down": moves_down, "moves_up": moves_up,
        "peak_p95_ms": max(r["p95_ms"] for r in records),
        "min_train": min(r["train"] for r in records),
        "max_infer": max(r["infer"] for r in records),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
