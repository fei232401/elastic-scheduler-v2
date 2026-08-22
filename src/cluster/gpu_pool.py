"""GPU 资源池 —— 8 个逻辑 GPU token 的分配/释放（spec §4 Mock Cluster）。

GPU 不需要真实存在，建模为 Resource Token（GPU 0..N-1）。
本模块只负责"哪些卡被谁占用"，不做任何调度决策。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GpuPool:
    total: int
    _allocated: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("total GPUs cannot be negative")
        self._allocated = set()

    @property
    def free_gpus(self) -> int:
        return self.total - len(self._allocated)

    @property
    def allocated_gpus(self) -> int:
        return len(self._allocated)

    def alloc_gpus(self, n: int) -> list[int]:
        """从空闲池分配 n 张卡，返回分配到的 GPU 编号列表。

        Raises:
            ValueError: n 为负或超过空闲数。
        """
        if n < 0:
            raise ValueError("cannot allocate negative GPUs")
        if n > self.free_gpus:
            raise ValueError(
                f"not enough free GPUs: requested {n}, free {self.free_gpus}"
            )
        picked: list[int] = []
        for gpu_id in range(self.total):
            if gpu_id not in self._allocated:
                picked.append(gpu_id)
                self._allocated.add(gpu_id)
            if len(picked) == n:
                break
        return picked

    def free_gpus_by_ids(self, ids: list[int]) -> None:
        """释放指定 GPU 编号。不持有或重复释放视为幂等（忽略）。"""
        for gpu_id in ids:
            self._allocated.discard(gpu_id)

    def free_n(self, n: int) -> None:
        """释放任意 n 张卡（从当前已分配中取）。"""
        ids = list(self._allocated)[:n]
        self.free_gpus_by_ids(ids)
