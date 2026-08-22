"""Mock Training Job —— 分布式训练行为的最小模拟（spec §5/§6/§12）。

核心行为：
  - throughput = base_throughput * gpu^scaling_exponent（幂律，边际收益递减）
  - 每 tick 按当前 GPU 数推进 progress
  - GPU 减少 → 不停止，progress 继续，但 throughput 下降、完成时间变长
  - 状态机：RUNNING / DEGRADED / PAUSED / COMPLETED / FAILED
    DEGRADED = 仍在跑但 GPU < initial_gpu（被调度器让渡过）
"""
from __future__ import annotations

from enum import Enum

from src.config import get


class TrainingState(str, Enum):
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MockTrainingJob:
    def __init__(self, cfg: dict) -> None:
        self.total_work = get(cfg, "training.total_work", 100000)
        self.initial_gpu = get(cfg, "training.initial_gpu", 6)
        self.min_gpu = get(cfg, "training.min_gpu", 1)
        self.base_throughput = get(cfg, "training.base_throughput", 15)
        self.scaling_exponent = get(cfg, "training.scaling_exponent", 0.85)
        self.max_allowed_slowdown = get(cfg, "training.max_allowed_slowdown", 3.0)

        self.progress: float = 0.0
        self.gpus: int = self.initial_gpu
        self.state = TrainingState.RUNNING
        self._was_degraded: bool = False

    # ---- throughput & timing ----
    def throughput_at(self, gpus: int) -> float:
        """幂律吞吐：throughput = base * gpu^scaling_exponent（spec §5 工程修正）。"""
        if gpus <= 0:
            return 0.0
        return self.base_throughput * (gpus**self.scaling_exponent)

    @property
    def throughput(self) -> float:
        return self.throughput_at(self.gpus)

    @property
    def remaining_work(self) -> float:
        return max(0.0, self.total_work - self.progress)

    @property
    def estimated_completion_time_s(self) -> float:
        """按当前吞吐估算剩余秒数。"""
        tp = self.throughput
        return self.remaining_work / tp if tp > 0 else float("inf")

    @property
    def slowdown_vs_initial(self) -> float:
        """相对 initial_gpu 的完成时间放大倍数（估算）。"""
        tp_initial = self.throughput_at(self.initial_gpu)
        if tp_initial <= 0:
            return float("inf")
        return tp_initial / max(self.throughput, 1e-9)

    # ---- lifecycle ----
    @property
    def is_degraded(self) -> bool:
        return self.state == TrainingState.RUNNING and self.gpus < self.initial_gpu

    def step(self, gpus: int | None = None, throughput_scale: float = 1.0) -> float:
        """推进一个 tick。返回本 tick 完成的工作量。

        Args:
            gpus: 若提供则更新配额（调度器已经改完资源后调用）。
            throughput_scale: 网络拥塞回压导致的吞吐折扣（0~1，spec §16）。
        """
        if gpus is not None:
            self.gpus = max(gpus, 0)

        if self.state in (TrainingState.PAUSED, TrainingState.COMPLETED, TrainingState.FAILED):
            return 0.0
        if self.gpus <= 0:
            self.state = TrainingState.PAUSED
            return 0.0

        # GPU 被让渡过 → DEGRADED（仍 RUNNING）
        if self.gpus < self.initial_gpu:
            self.state = TrainingState.DEGRADED
            self._was_degraded = True
        else:
            self.state = TrainingState.RUNNING

        delta = self.throughput * throughput_scale  # 每个 tick 的吞吐量（tick 视作单位时间）
        self.progress = min(self.total_work, self.progress + delta)
        if self.progress >= self.total_work:
            self.state = TrainingState.COMPLETED
        return delta

    def would_allow_degradation(self, new_gpus: int) -> bool:
        """降级到 new_gpus 是否仍满足 max_allowed_slowdown 约束（spec §12）。"""
        if new_gpus < self.min_gpu:
            return False
        tp_new = self.throughput_at(new_gpus)
        tp_initial = self.throughput_at(self.initial_gpu)
        if tp_new <= 0:
            return False
        return (tp_initial / tp_new) <= self.max_allowed_slowdown
