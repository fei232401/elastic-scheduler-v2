"""资源管理器 —— 训练/推理配额的分配与逐级流转（spec §4/§10）。

调度器决定配额变化，ResourceManager 负责原子执行：
   - Training GPU 减少 n ⇔ Inference GPU 增加 n（总量守恒）
   - 校验 Training 不低于 min_gpu、总量不超过 total_gpus。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.cluster.gpu_pool import GpuPool


@dataclass
class ResourceState:
    training_gpus: int
    inference_gpus: int


class ResourceManager:
    def __init__(self, total_gpus: int, initial_training: int, min_training: int) -> None:
        if not (0 <= min_training <= initial_training <= total_gpus):
            raise ValueError(
                "invalid initial allocation: "
                f"total={total_gpus} initial={initial_training} min={min_training}"
            )
        self._pool = GpuPool(total_gpus)
        self.min_training = min_training
        self.training_gpus = initial_training
        self.inference_gpus = total_gpus - initial_training
        self._pool.alloc_gpus(initial_training)

    @property
    def state(self) -> ResourceState:
        return ResourceState(self.training_gpus, self.inference_gpus)

    @property
    def unused_gpus(self) -> int:
        return self._pool.free_gpus

    def shift(self, delta: int) -> None:
        """训练配额调整 delta 张（负数=让渡给推理，正数=收回）。

        Raises:
            ValueError: 会导致 Training 低于 min_gpu 或总量越界。
        """
        new_training = self.training_gpus + delta
        if new_training < self.min_training:
            raise ValueError(
                f"cannot shift: training would drop to {new_training} < min {self.min_training}"
            )
        new_inference = self.inference_gpus - delta
        if new_inference < 0 or new_inference > self._pool.total:
            raise ValueError(
                f"cannot shift: inference would be {new_inference} (total {self._pool.total})"
            )
        if delta < 0:
            self._pool.free_n(-delta)
            self._pool.alloc_gpus(-delta)
        self.training_gpus = new_training
        self.inference_gpus = new_inference
