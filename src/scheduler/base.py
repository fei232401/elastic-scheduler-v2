"""Scheduler 抽象基类 + 决策数据结构（spec §10/§24，预留插拔位）。

所有调度器实现 `step(ctx)` → `ResourceDecision`。
可插拔：Static / Elastic / Preemptive / Network-aware（均继承本基类）。

Phase 3.1（Runtime Abstraction）：`ResourceDecision` 的定义挪到 `src/runtime/base.py`
（调度决策属于 runtime 契约），本模块 re-export 以兼容现有 import。
`SchedulerContext.runtime` 持有 `RuntimeAdapter` —— Scheduler 只经它读写资源，不直接触碰 Mock 对象。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.runtime.base import ResourceDecision
from src.runtime.base import RuntimeAdapter


@dataclass
class SchedulerContext:
    """调度器每 tick 可见的集群状态快照（由 engine 从 runtime 状态视图组装）。"""

    training_gpus: int
    inference_gpus: int
    training_min_gpu: int
    training_initial_gpu: int
    inference_p95_ms: float
    slo_p95_ms: float
    inference_state: str
    training_throughput: float
    training_slowdown: float
    training_would_allow_degradation: bool
    inference_overload_counter: int
    inference_recovery_counter: int
    now_s: float
    runtime: RuntimeAdapter | None = None
    extra: dict = field(default_factory=dict)


class Scheduler(ABC):
    """调度器基类。子类实现 step()。"""

    name: str = "base"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @abstractmethod
    def step(self, ctx: SchedulerContext) -> ResourceDecision:
        raise NotImplementedError
