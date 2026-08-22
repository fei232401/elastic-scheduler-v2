"""Inference 流量生成器 —— 潮汐周期（spec §9）。

必须支持：constant / step / sine / spike / custom（分段线性时间表）。
默认 scenario = 70min 的 LOW→RISING→PEAK→FALLING→LOW。
确定性抖动由外部传入的 RNG 控制（与 seed 绑定，保证可复现）。
"""
from __future__ import annotations

import math
from typing import Callable

from src.config import get


class TrafficGenerator:
    def __init__(self, cfg: dict, rng=None, jitter: float | None = None) -> None:
        t = get(cfg, "traffic", {})
        self.type = t.get("type", "custom")
        self.schedule = t.get("schedule", [])
        self.step_interval_min = t.get("step_interval_min", 5)
        self.step_low_qps = t.get("step_low_qps", 10)
        self.step_high_qps = t.get("step_high_qps", 60)
        self.sine_period_min = t.get("sine_period_min", 30)
        self.sine_amplitude = t.get("sine_amplitude", 25)
        self.sine_base = t.get("sine_base", 35)
        self.spike_min = t.get("spike_min", 30)
        self.spike_qps = t.get("spike_qps", 60)
        self.base_qps = t.get("base_qps", 10)
        self.jitter_ratio = jitter if jitter is not None else t.get("jitter_ratio", 0.0)
        self.rng = rng

    def qps_at_min(self, minute: float) -> float:
        """返回 minute 时刻的基准 QPS（不含抖动）。"""
        t = self.type
        if t == "constant":
            return float(self.base_qps)
        if t == "step":
            step_idx = int(minute // self.step_interval_min)
            return self.step_low_qps if step_idx % 2 == 0 else self.step_high_qps
        if t == "sine":
            return self.sine_base + self.sine_amplitude * math.sin(
                2 * math.pi * minute / self.sine_period_min
            )
        if t == "spike":
            return self.spike_qps if self.spike_min <= minute < self.spike_min + 5 else self.base_qps
        if t == "custom":
            return self._custom_qps(minute)
        raise ValueError(f"unknown traffic type: {t}")

    def _custom_qps(self, minute: float) -> float:
        pts = sorted(self.schedule, key=lambda p: p["min"])
        if not pts:
            return 0.0
        if minute <= pts[0]["min"]:
            return float(pts[0]["qps"])
        for a, b in zip(pts, pts[1:]):
            if a["min"] <= minute <= b["min"]:
                span = b["min"] - a["min"]
                if span <= 0:
                    return float(b["qps"])
                ratio = (minute - a["min"]) / span
                return a["qps"] + (b["qps"] - a["qps"]) * ratio
        return float(pts[-1]["qps"])

    def qps_at_second(self, seconds: float, apply_jitter: bool = True) -> float:
        """返回秒级 QPS。抖动范围 [-jitter, +jitter] * base（确定性 RNG）。"""
        base = self.qps_at_min(seconds / 60.0)
        if not apply_jitter or self.jitter_ratio <= 0 or self.rng is None:
            return max(0.0, base)
        jitter = self.rng.uniform(-self.jitter_ratio, self.jitter_ratio) * base
        return max(0.0, base + jitter)
