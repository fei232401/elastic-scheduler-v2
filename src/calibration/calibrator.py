"""Calibration 管线核心（Phase 3.2 Step 9，spec §22）。

把 R1/R2/R3 的真实测量（results/profiles/*.json）转成 Local runtime 的校准参数，
逐项标记来源，杜绝「把模拟当真实」：
  - REAL     ：本机实测 —— T1 单卡吞吐、推理饱和吞吐、真实 P95、真实 netem 抬升；
  - EMULATED ：虚拟多卡配额上的缩放模型 —— capacity×g、吞吐×g^exp。本机只有 1 张
    物理 GPU，多卡行为无法真实测量；缩放形状（0.85 幂律，spec §5）是假设不是测量，
    诚实标注。spec §20 红线：本阶段不做真实多机 distributed training。
  - DESIGN   ：设计参数（SLO）。

多卡缩放指数的诚实性：只有 1 张物理 GPU，无法实测 T(2)/T(4)…。「校准」在这里的
上限 = T1 用真实测量替换默认值，缩放形状显式写成数据工件（calibration_curve，
g=1 标 REAL，g>1 标 EMULATED）—— 未来有真实多卡环境时整体替换曲线即可。不做任何
伪装成真实多卡的产出。

本模块是纯函数（无 I/O、无随机），便于确定性测试；I/O 在 experiments/calibrate.py。
"""
from __future__ import annotations

from src.config import get


def build_scale_curve(exponent: float, n_gpus: int = 8) -> dict[str, float]:
    """EMULATED 多卡缩放系数曲线：scale(g) = g^exponent（spec §5 幂律）。"""
    return {str(g): round(float(g**exponent), 4) for g in range(1, n_gpus + 1)}


def build_calibration(
    base_cfg: dict,
    training_profile: dict,
    inference_profile: dict,
    network_profile: dict | None = None,
) -> dict:
    """从真实测量构建校准审计 dict。

    Args:
        base_cfg: 基础配置（config/default.yaml）—— 提供 total_work 等设计参数。
        training_profile: R1 训练 profile（results/profiles/training_profile.json）。
        inference_profile: R2 推理 profile（results/profiles/inference_profile.json）。
        network_profile: R3 网络 profile（results/profiles/network_profile.json）；可选。

    Returns:
        含 provenance（逐项 REAL/EMULATED/DESIGN）与 runtime_local（Step 10 可用的
        `local:` 段）的审计 dict。
    """
    tr = get(training_profile, "real", {})
    curve = get(training_profile, "emulated_curve", {})
    inf_cal = get(inference_profile, "calibration", {})
    inf_real = get(inference_profile, "real", {})

    exponent = float(get(base_cfg, "training.scaling_exponent", 0.85))
    scale_curve = build_scale_curve(exponent)
    t1 = float(tr.get("mean_samples_per_sec", 0.0))

    calibration = {
        "generated_from": {
            "R1": get(training_profile, "experiment", "R1"),
            "R2": get(inference_profile, "experiment", "R2"),
            "R3": get(network_profile, "experiment", "R3") if network_profile else None,
        },
        "training": {
            "base_throughput_real_sps": round(t1, 1),
            "base_throughput_marker": "REAL (R1 single-GPU measurement)",
            "scaling_exponent": exponent,
            "scaling_marker": "EMULATED: only 1 physical GPU; n^exponent is a spec "
                              "assumption, not a measurement (cannot measure T(2..8) here)",
            "calibration_curve": scale_curve,
            "curve_marker": {str(g): "REAL" if g == 1 else "EMULATED"
                             for g in range(1, len(scale_curve) + 1)},
            "emulated_multigpu_sps": {
                str(g): round(t1 * scale_curve[str(g)], 1) for g in range(1, len(scale_curve) + 1)
            },
            "total_work": float(get(base_cfg, "training.total_work", 100000)),
        },
        "inference": {
            "capacity_per_gpu": inf_cal.get("capacity_per_gpu", 0.0),
            "capacity_marker": "REAL-derived (R2 saturation_qps / initial_inference_gpus); "
                               "x virtual-gpus model is EMULATED",
            "base_p95_ms": inf_cal.get("base_p95_ms", 0.0),
            "overload_p95_ms": inf_cal.get("overload_p95_ms", 0.0),
            "saturation_qps": inf_cal.get("saturation_qps", 0.0),
            "latency_marker": "REAL (R2 real latency curve)",
            "slo_p95_ms": inf_cal.get("slo_p95_ms", 0.0),
            "slo_marker": "DESIGN parameter (not derived from GPU band)",
            "bottleneck_note": inf_real.get("bottleneck_note", ""),
        },
        "network": None,
        "honesty_note": (
            "All REAL numbers are measurements on this 1x RTX 4070 machine; "
            "multi-GPU / capacity scaling is EMULATED (virtual quota) and never "
            "claimed as real multi-GPU data (spec 22)."
        ),
    }

    if network_profile:
        rows = network_profile.get("rows", [])
        slo = network_profile.get("slo_p95_ms", 0.0)
        crossing = None
        for r in rows:
            if r.get("e2e_sat_slo_violated") or r.get("e2e_base_slo_violated"):
                crossing = r.get("netem_delay_ms")
                break
        calibration["network"] = {
            "slo_p95_ms": slo,
            "first_netem_delay_crossing_slo_ms": crossing,
            "marker": "REAL (netem delay on loopback, client-observed e2e p95)",
            "key_findings": network_profile.get("key_findings", []),
        }

    # Step 10 可用的 `local:` 覆盖段（完整配置 = base + 此段）
    runtime_local = {
        "local": {
            "device": get(base_cfg, "local.device", "auto"),
            "training": {
                "total_work": get(base_cfg, "training.total_work", 100000),
                "calibration_curve": scale_curve,
            },
            "inference": {
                "capacity_per_gpu": inf_cal.get("capacity_per_gpu", 0.0),
                "base_p95_ms": inf_cal.get("base_p95_ms", 0.0),
                "overload_p95_ms": inf_cal.get("overload_p95_ms", 0.0),
                "saturation_qps": inf_cal.get("saturation_qps", 0.0),
                "slo_p95_ms": inf_cal.get("slo_p95_ms", inf_cal.get("slo_p95_ms", 0.0)),
            },
        }
    }
    calibration["runtime_local"] = runtime_local
    return calibration
