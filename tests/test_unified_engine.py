"""Step 10 统一引擎测试（Phase 3.2，spec §16/§24）。

验证：
  - build_runtime 按 mode 分派 Mock/Local adapter（同一 Runtime Contract）；
  - 同一 tick 循环可驱动 local runtime（真实 workload，短 tick 冒烟）；
  - 4 策略 × 同 config 都能跑通（接线证明）。
"""
from __future__ import annotations

import random

from src.config import load_config
from src.runtime.local import LocalRuntimeAdapter
from src.runtime.mock import MockRuntimeAdapter
from src.simulator.engine import SCHEDULERS, build_runtime, run_experiment


def _tiny_local_cfg():
    cfg = load_config("config/local_experiment.yaml")
    cfg["simulation"]["duration_ticks"] = 2
    cfg["simulation"]["tick_seconds"] = 0.3
    cfg["local"]["inference"]["port"] = 8799
    return cfg


def test_build_runtime_dispatches_mock_and_local():
    cfg = load_config("config/default.yaml")
    rng = random.Random(42)
    assert isinstance(build_runtime("mock", cfg, rng), MockRuntimeAdapter)
    local_cfg = _tiny_local_cfg()
    assert isinstance(build_runtime("local", local_cfg, rng), LocalRuntimeAdapter)


def test_build_runtime_rejects_unknown_mode():
    import pytest
    cfg = load_config("config/default.yaml")
    with pytest.raises(ValueError):
        build_runtime("kubernetes", cfg, random.Random(42))


def test_run_experiment_local_tiny_ticks():
    """真实 workload 短跑：验证 tick 循环经 Local adapter 跑通（真实测量非空）。"""
    cfg = _tiny_local_cfg()
    c = run_experiment("static", cfg, "/tmp/local_engine_test", mode="local")
    s = c.summarize()
    assert s.total_ticks == 2
    assert len(c.rows) == 2
    assert s.slo_violation_ratio > 0.0
    assert s.inference_avg_p95 > cfg["inference"]["slo_p95_ms"]


def test_all_four_schedulers_dispatch_on_local_builder():
    """4 策略都能经 build_runtime 构建 + 同一 engine 驱动（短 tick，接线证明）。"""
    cfg = _tiny_local_cfg()
    for name in SCHEDULERS:
        c = run_experiment(name, cfg, f"/tmp/local_engine_test_{name}", mode="local")
        assert c.summarize().total_ticks == 2
