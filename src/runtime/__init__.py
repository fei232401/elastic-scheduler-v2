"""Runtime 抽象层 —— 让 Scheduler 与具体 Cluster/Workload 解耦（Phase 3.1）。

契约在 `base.py`：统一状态视图 + RuntimeAdapter 接口。
实现：
  - `mock.py`   MockRuntimeAdapter（包装现有 Mock 组件，跑现有实验）
  - `local.py`  LocalRuntimeAdapter（Phase 3.2：真实 workload + tc/netem 网络）
"""
from src.runtime.base import (
    ClusterState,
    InferenceState,
    NetworkState,
    ResourceDecision,
    RuntimeAdapter,
    TrainingState,
)
from src.runtime.local import LocalRuntimeAdapter

__all__ = [
    "ClusterState",
    "TrainingState",
    "InferenceState",
    "NetworkState",
    "ResourceDecision",
    "RuntimeAdapter",
    "LocalRuntimeAdapter",
]
