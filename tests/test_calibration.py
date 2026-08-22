"""Calibration 管线测试（Phase 3.2 Step 9，spec §22）。

用合成 profile（不依赖 GPU / 真实测量缓存）验证校准器的纯函数：
  - T1 用 R1 实测替换默认；
  - 多卡缩放曲线显式写出，g=1 标 REAL、g>1 标 EMULATED（诚实标注）；
  - 推理容量/延迟来自 R2 校准，SLO 标 DESIGN；
  - R3 网络校准提取「越 SLO 的 netem 延迟」；
  - runtime_local 段 = Step 10 可加载的 `local:` 覆盖。
"""
from __future__ import annotations

import copy

from src.calibration.calibrator import build_calibration, build_scale_curve

BASE_CFG = {
    "simulation": {"total_gpus": 8},
    "training": {"total_work": 100000, "initial_gpu": 6, "scaling_exponent": 0.85},
    "inference": {},
    "local": {"device": "auto"},
}

R1 = {
    "experiment": "R1_real_training",
    "real": {"mean_samples_per_sec": 216822.7, "stdev_samples_per_sec": 413.5},
    "emulated_curve": {
        "1": {"emulated_samples_per_sec": 216822.7, "marker": "REAL"},
        "2": {"emulated_samples_per_sec": 390823.2, "marker": "EMULATED"},
    },
}

R2 = {
    "experiment": "R2_real_inference",
    "real": {"bottleneck_note": "server is GIL/thread-bound"},
    "calibration": {
        "capacity_per_gpu": 563.3, "base_p95_ms": 5.27, "overload_p95_ms": 6.46,
        "saturation_qps": 1126.7, "slo_p95_ms": 15.0,
    },
}

R3 = {
    "experiment": "R3_real_network",
    "slo_p95_ms": 15.0,
    "rows": [
        {"netem_delay_ms": 0.0, "e2e_base_slo_violated": False, "e2e_sat_slo_violated": False},
        {"netem_delay_ms": 5.0, "e2e_base_slo_violated": True, "e2e_sat_slo_violated": True},
        {"netem_delay_ms": 10.0, "e2e_base_slo_violated": True, "e2e_sat_slo_violated": True},
    ],
    "key_findings": ["REAL: netem throttles throughput"],
}


def test_scale_curve_is_power_law():
    c = build_scale_curve(0.85)
    assert c["1"] == 1.0
    assert abs(c["2"] - 2 ** 0.85) < 1e-3
    assert len(c) == 8


def test_t1_from_real_measurement_marked_real():
    cal = build_calibration(BASE_CFG, R1, R2, R3)
    tr = cal["training"]
    assert tr["base_throughput_real_sps"] == 216822.7
    assert "REAL" in tr["base_throughput_marker"]
    assert tr["curve_marker"]["1"] == "REAL"
    assert tr["curve_marker"]["2"] == "EMULATED"


def test_scaling_is_honestly_emulated():
    cal = build_calibration(BASE_CFG, R1, R2, R3)
    tr = cal["training"]
    assert "EMULATED" in tr["scaling_marker"]
    assert "not a measurement" in tr["scaling_marker"]
    # 多卡吞吐 = T1 × curve
    assert abs(tr["emulated_multigpu_sps"]["2"] - 216822.7 * (2 ** 0.85)) < 1.0


def test_inference_params_from_r2_with_markers():
    cal = build_calibration(BASE_CFG, R1, R2, R3)
    inf = cal["inference"]
    assert inf["capacity_per_gpu"] == 563.3
    assert inf["base_p95_ms"] == 5.27
    assert "REAL" in inf["latency_marker"]
    assert "EMULATED" in inf["capacity_marker"]
    assert inf["slo_p95_ms"] == 15.0
    assert "DESIGN" in inf["slo_marker"]


def test_network_crossing_extracted():
    cal = build_calibration(BASE_CFG, R1, R2, R3)
    net = cal["network"]
    assert net["first_netem_delay_crossing_slo_ms"] == 5.0
    assert net["marker"] == "REAL (netem delay on loopback, client-observed e2e p95)"
    assert cal["network"]["key_findings"]


def test_network_optional():
    cal = build_calibration(BASE_CFG, R1, R2, None)
    assert cal["network"] is None


def test_runtime_local_section_loadable():
    cal = build_calibration(BASE_CFG, R1, R2, R3)
    rl = cal["runtime_local"]["local"]
    assert rl["training"]["calibration_curve"]["1"] == 1.0
    assert rl["inference"]["capacity_per_gpu"] == 563.3
    assert rl["inference"]["slo_p95_ms"] == 15.0
    # 完整配置 = base + local 覆盖后，training_real 可消费
    full = {**copy.deepcopy(BASE_CFG), **copy.deepcopy(cal["runtime_local"])}
    assert full["local"]["training"]["calibration_curve"]["2"] > 1.0


def test_base_cfg_untouched():
    base = copy.deepcopy(BASE_CFG)
    build_calibration(base, R1, R2, R3)
    assert base == BASE_CFG
