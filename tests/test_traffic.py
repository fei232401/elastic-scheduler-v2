"""TrafficGenerator 测试 —— 潮汐输入信号（spec §9）。

所有 shipped 配置都用 type=custom（分段线性时间表），是本实验唯一实际生效的
流量模式，此前无直接测试；constant/step/sine/spike 分支仅为完整性覆盖。
"""
from __future__ import annotations

import random

import pytest

from src.workload.traffic import TrafficGenerator


def _cfg(schedule, **kw) -> dict:
    traffic = {"type": "custom", "schedule": schedule, **kw}
    return {"traffic": traffic}


CUSTOM = [
    {"min": 0, "qps": 10},
    {"min": 20, "qps": 60},
    {"min": 40, "qps": 10},
]


class TestCustom:
    def test_before_first_point_uses_first_qps(self):
        gen = TrafficGenerator(_cfg(CUSTOM))
        assert gen.qps_at_min(0) == 10.0
        assert gen.qps_at_min(-5) == 10.0

    def test_linear_interpolation(self):
        gen = TrafficGenerator(_cfg(CUSTOM))
        # 10 -> 60 over 20 min：中点 = 35
        assert gen.qps_at_min(10) == 35.0
        # 60 -> 10 over 20 min：中点 = 35
        assert gen.qps_at_min(30) == 35.0

    def test_after_last_point_uses_last_qps(self):
        gen = TrafficGenerator(_cfg(CUSTOM))
        assert gen.qps_at_min(40) == 10.0
        assert gen.qps_at_min(70) == 10.0

    def test_empty_schedule_returns_zero(self):
        gen = TrafficGenerator(_cfg([]))
        assert gen.qps_at_min(30) == 0.0

    def test_unsorted_schedule_is_sorted(self):
        shuffled = list(reversed(CUSTOM))
        gen = TrafficGenerator(_cfg(shuffled))
        assert gen.qps_at_min(10) == 35.0  # 与排序后一致

    def test_default_config_matches_shipped_tide(self):
        # 与 config/default.yaml 的 70min 潮汐一致：LOW -> RISING -> PEAK -> FALLING -> LOW
        import yaml

        cfg = yaml.safe_load(open("config/default.yaml"))
        gen = TrafficGenerator(cfg)
        assert gen.qps_at_min(0) == 10.0
        assert gen.qps_at_min(20) == 35.0  # 上升段
        assert gen.qps_at_min(35) == 60.0  # 峰值
        assert gen.qps_at_min(55) == 20.0  # 下降段（50->15 之间）
        assert gen.qps_at_min(70) == 10.0  # 回到 LOW


class TestOtherTypes:
    def test_constant(self):
        gen = TrafficGenerator({"traffic": {"type": "constant", "base_qps": 33}})
        assert gen.qps_at_min(0) == 33.0
        assert gen.qps_at_min(99) == 33.0

    def test_step_alternates(self):
        gen = TrafficGenerator({"traffic": {"type": "step", "step_interval_min": 5,
                                            "step_low_qps": 10, "step_high_qps": 60}})
        assert gen.qps_at_min(4.9) == 10.0
        assert gen.qps_at_min(5.0) == 60.0

    def test_sine_peaks_at_quarter_period(self):
        gen = TrafficGenerator({"traffic": {"type": "sine", "sine_period_min": 30,
                                            "sine_base": 35, "sine_amplitude": 25}})
        assert gen.qps_at_min(7.5) == pytest.approx(35.0 + 25.0, abs=1e-9)
        assert gen.qps_at_min(0) == 35.0

    def test_spike_window(self):
        gen = TrafficGenerator({"traffic": {"type": "spike", "spike_min": 30,
                                            "spike_qps": 60, "base_qps": 10}})
        assert gen.qps_at_min(29) == 10.0
        assert gen.qps_at_min(32) == 60.0
        assert gen.qps_at_min(35) == 10.0

    def test_unknown_type_raises(self):
        gen = TrafficGenerator({"traffic": {"type": "bogus"}})
        with pytest.raises(ValueError):
            gen.qps_at_min(0)


class TestJitter:
    def test_jitter_is_deterministic_with_seed(self):
        gen_a = TrafficGenerator(_cfg(CUSTOM), rng=random.Random(42), jitter=0.1)
        gen_b = TrafficGenerator(_cfg(CUSTOM), rng=random.Random(42), jitter=0.1)
        pts = [20, 21, 22, 23, 24]
        assert [gen_a.qps_at_second(t) for t in pts] == [gen_b.qps_at_second(t) for t in pts]

    def test_jitter_never_negative(self):
        gen = TrafficGenerator(_cfg(CUSTOM), rng=random.Random(7), jitter=1.0)
        for t in range(0, 2400, 13):
            assert gen.qps_at_second(float(t)) >= 0.0

    def test_no_jitter_when_ratio_zero(self):
        gen = TrafficGenerator(_cfg(CUSTOM), rng=random.Random(7))
        assert gen.qps_at_second(1200.0) == gen.qps_at_min(20.0)
