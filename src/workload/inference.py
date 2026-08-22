"""Mock Inference Service —— 在线推理延迟/容量/过载行为的最小模拟（spec §7/§8/§9）。

核心行为：
  - capacity = capacity_per_gpu * gpus（线性容量）
  - incoming_qps <= capacity → P95 在 base 区间内按利用率线性插值
  - incoming_qps > capacity → 排队，P95 非线性上扬（逼近 overload 上限）
  - 防抖（Hysteresis）：连续 overload_trigger_ticks 个 tick 过载才置 OVERLOADED；
    连续 recovery_trigger_ticks 个 tick 健康才置 RECOVERING/RUNNING。
"""
from __future__ import annotations

import math
from enum import Enum

from src.config import get


class InferenceState(str, Enum):
    RUNNING = "RUNNING"
    OVERLOADED = "OVERLOADED"
    RECOVERING = "RECOVERING"


class MockInferenceService:
    def __init__(self, cfg: dict) -> None:
        self.capacity_per_gpu = get(cfg, "inference.capacity_per_gpu", 10)
        self.slo_p95_ms = get(cfg, "inference.slo_p95_ms", 300)
        self.p95_base_ms = get(cfg, "inference.p95_base_ms", 200)
        self.p95_overload_ms = get(cfg, "inference.p95_overload_ms", 900)
        self.overload_trigger_ticks = get(cfg, "inference.overload_trigger_ticks", 3)
        self.recovery_trigger_ticks = get(cfg, "inference.recovery_trigger_ticks", 5)

        self.gpus: int = 2
        self.incoming_qps: float = 0.0
        self.throughput_qps: float = 0.0
        self.queue_length: int = 0
        self.p50_ms: float = 0.0
        self.p95_ms: float = 0.0
        self.p99_ms: float = 0.0
        self.state = InferenceState.RUNNING
        self.slo_violated: bool = False

        self._overload_counter = 0
        self._recovery_counter = 0

    # ---- capacity & latency ----
    @property
    def capacity_qps(self) -> float:
        return self.capacity_per_gpu * self.gpus

    def _compute_latency(self, qps: float) -> tuple[float, float, float, int]:
        """计算 p50/p95/p99 与队列长度。

        qps <= capacity: P95 在 [p95_base*0.75, p95_base] 按利用率插值。
        qps >  capacity: 超容量部分按指数排队，P95 单调升向 overload 上限。
        """
        cap = self.capacity_qps
        if cap <= 0:
            return (self.p95_overload_ms, self.p95_overload_ms, self.p95_overload_ms * 1.1, 1)
        util = min(qps / cap, 2.0)
        if qps <= cap:
            p50 = self.p95_base_ms * 0.6
            p95 = self.p95_base_ms * (0.75 + 0.25 * util)
            queue = 0
        else:
            overload_ratio = qps / cap
            # 非线性：超载越狠 P95 越陡
            p95 = self.p95_base_ms + (self.p95_overload_ms - self.p95_base_ms) * min(
                1.0, (overload_ratio - 1.0) ** 1.5
            )
            p50 = p95 * 0.6
            queue = int(round((qps - cap) * 2.0))
        p99 = p95 * (1.0 + 0.05 * max(0.0, util - 1.0))
        return (p50, min(p95, self.p95_overload_ms * 1.2), p99, queue)

    # ---- lifecycle ----
    def step(self, qps: float, network_latency_ms: float = 0.0) -> None:
        """一个 tick：喂入新 QPS，更新延迟/状态。

        Args:
            qps: 本 tick 到达的推理请求数。
            network_latency_ms: 网络拥塞带来的额外尾延迟（spec §16，回加到 P95/P99）。
        """
        self.incoming_qps = qps
        self.p50_ms, self.p95_ms, self.p99_ms, self.queue_length = self._compute_latency(qps)
        if network_latency_ms > 0:
            self.p95_ms += network_latency_ms
            self.p99_ms += network_latency_ms
            self.p50_ms += network_latency_ms * 0.5
        self.throughput_qps = min(qps, self.capacity_qps)
        self.slo_violated = self.p95_ms > self.slo_p95_ms

        # 防抖计数（spec §9 工程修正）
        if self.slo_violated:
            self._overload_counter += 1
            self._recovery_counter = 0
        else:
            self._recovery_counter += 1
            self._overload_counter = 0

        if (
            self._overload_counter >= self.overload_trigger_ticks
            and self.state != InferenceState.OVERLOADED
        ):
            self.state = InferenceState.OVERLOADED
        elif (
            self._recovery_counter >= self.recovery_trigger_ticks
            and self.state in (InferenceState.OVERLOADED, InferenceState.RECOVERING)
        ):
            self.state = InferenceState.RUNNING
            self._overload_counter = 0
            self._recovery_counter = 0
        elif self.state == InferenceState.OVERLOADED and not self.slo_violated:
            self.state = InferenceState.RECOVERING

    def set_gpus(self, gpus: int) -> None:
        self.gpus = max(gpus, 0)
