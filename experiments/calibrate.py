"""Step 9 — Calibration 数据管线（Phase 3.2，spec §22）。

读取 R1/R2/R3 的真实测量（results/profiles/*.json），产出：
  - config/local_calibrated.yaml : 校准配置（= base 配置 + local 段用实测校准值覆盖；
    多卡缩放曲线显式写入，诚实标注）。注意：local 服务/模型形状参数（host/port/input_dim/
    hidden_dim/layers 等）不在此文件内，来自代码默认值；Step 10 的规范入口是
    config/local_experiment.yaml（local 段完整），本文件主要用于审计校准值。
  - results/profiles/calibration.json : 校准审计（逐项 REAL/EMULATED/DESIGN 来源）。

多卡缩放指数的诚实性（见 src/calibration/calibrator.py）：本机只有 1 张物理 GPU，
T1 是 REAL 实测，缩放形状（0.85 幂律）是 spec 假设（EMULATED），绝不伪装真实多卡。

用法: python -m experiments.calibrate [--config config/default.yaml]
           [--out config/local_calibrated.yaml]
先决条件：已跑 R1（profile_training）与 R2（profile_inference）。R3（profile_network）
可选，有则纳入网络校准。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import yaml

from src.calibration.calibrator import build_calibration
from src.config import load_config


def _load_json(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缺少 profile（先跑对应实验）: {p}")
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 9: calibration pipeline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--train-profile", default="results/profiles/training_profile.json")
    ap.add_argument("--inf-profile", default="results/profiles/inference_profile.json")
    ap.add_argument("--net-profile", default="results/profiles/network_profile.json")
    ap.add_argument("--out", default="config/local_calibrated.yaml")
    ap.add_argument("--audit-out", default="results/profiles/calibration.json")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    train_p = _load_json(args.train_profile)
    inf_p = _load_json(args.inf_profile)
    net_p = None
    if pathlib.Path(args.net_profile).exists():
        net_p = _load_json(args.net_profile)

    calibration = build_calibration(base_cfg, train_p, inf_p, net_p)
    runtime_local = calibration["runtime_local"]

    out_cfg = {**base_cfg, **runtime_local}
    out_cfg["local"]["_calibrated_from"] = {
        "R1": calibration["generated_from"]["R1"],
        "R2": calibration["generated_from"]["R2"],
        "R3": calibration["generated_from"]["R3"],
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True))

    audit = pathlib.Path(args.audit_out)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(calibration, indent=2, ensure_ascii=False))

    tr = calibration["training"]
    inf = calibration["inference"]
    print(f"calibration -> {args.out}")
    print(f"  T1 (REAL)        : {tr['base_throughput_real_sps']} sps @ 1 GPU")
    print(f"  scale exponent   : {tr['scaling_exponent']} [EMULATED 假设，非实测]")
    print(f"  curve            : g=1 REAL, g>1 EMULATED {list(tr['calibration_curve'].items())[:3]}...")
    print(f"  inf capacity/GPU : {inf['capacity_per_gpu']} qps [REAL-derived, ×g EMULATED]")
    print(f"  inf base/overload: {inf['base_p95_ms']}/{inf['overload_p95_ms']} ms [REAL]  SLO={inf['slo_p95_ms']} [DESIGN]")
    if calibration["network"]:
        print(f"  netem 越 SLO 阈值 : {calibration['network']['first_netem_delay_crossing_slo_ms']} ms [REAL]")
    print(f"audit -> {args.audit_out}")


if __name__ == "__main__":
    main()
