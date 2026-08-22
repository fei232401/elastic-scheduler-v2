"""R1 — 真实训练吞吐 profiling（Phase 3.2 Step 4，spec §15/Gate 1）。

测量 1 张物理 GPU 上的真实 samples/sec（多次窗口取 mean/std），
并用当前 EMULATED 缩放曲线给出「1 GPU→X sps / 2 GPU→Y sps」性能表。
诚实标注：单卡 = REAL，多卡 = EMULATED（不冒充真实多卡）。

用法: python -m experiments.profile_training [--windows 5] [--duration 4.0]
输出: results/profiles/training_profile.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

from src.config import load_config
from src.workload.training_real import RealTrainingJob


def main() -> None:
    ap = argparse.ArgumentParser(description="R1: real training throughput profiling")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--windows", type=int, default=5, help="测量窗口数")
    ap.add_argument("--duration", type=float, default=4.0, help="每窗口秒数")
    ap.add_argument("--out", default="results/profiles/training_profile.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lt = cfg.setdefault("local", {}).setdefault("training", {})
    lt.setdefault("input_dim", 256)
    lt.setdefault("hidden_dim", 1024)
    lt.setdefault("layers", 4)
    lt.setdefault("output_dim", 10)
    lt.setdefault("batch_size", 256)
    lt.setdefault("total_work", 50_000_000)
    lt.setdefault("scaling_exponent", 0.85)

    job = RealTrainingJob(cfg)
    job.start()
    job.tick(args.duration)  # 暖机（cuDNN autotune）

    sps: list[float] = []
    utils: list[float] = []
    mems: list[float] = []
    for _ in range(args.windows):
        job.tick(args.duration)
        sps.append(job._last_measured_sps)
        utils.append(job.gpu_utilization)
        mems.append(job.gpu_memory_mb)
    job.stop()

    mean_sps = statistics.mean(sps)
    stdev_sps = statistics.stdev(sps) if len(sps) > 1 else 0.0

    total_gpus = int(cfg.get("simulation", {}).get("total_gpus", 8))
    emulated_curve = {
        g: {"emulated_samples_per_sec": round(mean_sps * job.scaling_at(g), 1),
            "marker": "REAL" if g == 1 else "EMULATED"}
        for g in range(1, total_gpus + 1)
    }

    profile = {
        "experiment": "R1_real_training",
        "mode": "real_single_gpu + emulated_multi_gpu",
        "device": str(job.device),
        "model": {"input_dim": job.input_dim, "hidden_dim": job.hidden_dim,
                  "layers": job.layers, "batch_size": job.batch_size},
        "measurement": {"windows": args.windows, "duration_s": args.duration},
        "real": {
            "mean_samples_per_sec": round(mean_sps, 1),
            "stdev_samples_per_sec": round(stdev_sps, 1),
            "gpu_util_mean": round(statistics.mean(utils), 3),
            "gpu_mem_mb_mean": round(statistics.mean(mems), 1),
        },
        "emulated_curve": emulated_curve,
        "honesty_note": "single-GPU numbers are REAL measurements; multi-GPU are EMULATED "
                        "scaling (n^exponent) on top of the real single-GPU base, per spec 22.",
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
