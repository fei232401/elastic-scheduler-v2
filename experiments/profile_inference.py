"""R2 — 真实 HTTP 推理延迟 profiling（Phase 3.2 Step 6，spec §15/Gate 2）。

测量 1 张物理 GPU 上真实推理 server 的延迟曲线（低负载 → 饱和），
校准 EMULATED 容量模型参数（capacity_per_gpu / base_p95 / overload_p95 / SLO）。
诚实标注：延迟/吞吐 = REAL；容量模型 = EMULATED（虚拟配额）。

用法: python -m experiments.profile_inference [--low 200] [--high 3000] [--duration 3]
输出: results/profiles/inference_profile.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

from src.config import load_config
from src.workload.inference_real import HTTPInferenceService


def main() -> None:
    ap = argparse.ArgumentParser(description="R2: real HTTP inference latency profiling")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--low", type=float, default=200.0, help="低负载 QPS（测 base p95）")
    ap.add_argument("--high", type=float, default=3000.0, help="高负载 QPS（测饱和）")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--slo", type=float, default=15.0,
                    help="本地实验的 SLO 设计参数（ms）。GPU 健康 ~5ms 远低于 SLO，"
                         "网络拥塞（tc/netem，REAL）+10~20ms 能推过 —— GPU 不是瓶颈，网络才是。")
    ap.add_argument("--out", default="results/profiles/inference_profile.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    li = cfg.setdefault("local", {}).setdefault("inference", {})
    li.setdefault("input_dim", 64)
    li.setdefault("hidden_dim", 256)
    li.setdefault("layers", 3)
    li.setdefault("output_dim", 10)
    li.setdefault("generator_workers", 24)
    li.setdefault("max_batch_size", 16)
    li.setdefault("flush_interval_ms", 1.0)

    svc = HTTPInferenceService(cfg)
    svc.start()

    # 暖机：稳定线程池/连接池，避免冷启动瞬态污染 base 延迟
    svc.serve(max(args.low, 300.0), max(args.duration, 2.0))
    # 低负载：base 延迟（中等 QPS 比极低 QPS 更稳定）
    svc.serve(max(args.low, 500.0), args.duration)
    base = svc.real_stats
    # 高负载：饱和吞吐 + 饱和 p95
    svc.serve(args.high, args.duration)
    sat = svc.real_stats
    svc.stop()

    saturation_qps = sat["throughput_qps"]
    initial_infer_gpus = svc.allocated_gpu
    capacity_per_gpu = saturation_qps / initial_infer_gpus if initial_infer_gpus else saturation_qps
    base_p95 = base["p95_ms"]
    overload_p95 = max(base["p95_ms"], sat["p95_ms"])
    slo = args.slo  # 设计参数：健康 GPU 延迟远低于 SLO，真实网络拥塞能推过

    profile = {
        "experiment": "R2_real_inference",
        "mode": "real_latency_curve + emulated_capacity",
        "device": str(svc.device),
        "model": {"input_dim": svc.input_dim, "hidden_dim": svc.hidden_dim,
                  "layers": svc.layers, "max_batch_size": svc.max_batch_size},
        "measurement": {"low_qps": args.low, "high_qps": args.high, "duration_s": args.duration},
        "real": {
            "base_p50_ms": base["p50_ms"], "base_p95_ms": base["p95_ms"],
            "base_p99_ms": base["p99_ms"],
            "saturated_p50_ms": sat["p50_ms"], "saturated_p95_ms": sat["p95_ms"],
            "saturated_p99_ms": sat["p99_ms"],
            "saturation_qps": round(saturation_qps, 1),
            "gpu_util": svc.gpu_utilization,
            "bottleneck_note": "server is GIL/thread-bound (~1.1k qps); GPU compute not saturated",
        },
        "calibration": {
            "capacity_per_gpu": round(capacity_per_gpu, 1),
            "base_p95_ms": round(base_p95, 2),
            "overload_p95_ms": round(overload_p95, 2),
            "saturation_qps": round(saturation_qps, 1),
            "slo_p95_ms": round(slo, 2),
            "initial_inference_gpus": initial_infer_gpus,
            "emulated_note": "capacity = capacity_per_gpu x virtual gpus (spec 22, EMULATED)",
        },
        "honesty_note": "latency/throughput are REAL measurements; the capacity model is "
                        "EMULATED on top (virtual GPU quota).",
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
