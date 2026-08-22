"""Runtime Abstraction —— Scheduler 与具体 Cluster/Workload 解耦的契约（Phase 3.1，spec §10/§24）。

设计目标（任务书原则 1/3）：
  - Scheduler 只依赖本模块的接口与状态视图，不直接触碰 Mock/Local/K8s 的具体对象；
  - 只定义真实需要的方法，不为"以后可能用到"堆抽象。

数据流：
    Scheduler ──ResourceDecision──▶ RuntimeAdapter ──scale/apply──▶ Cluster / Workload
                    ◀──状态视图(ClusterState/…)──

本模块同时承载 `ResourceDecision`（§8 统一决策结构），
`scheduler/base.py` 保留 re-export，兼容已有 `from src.scheduler.base import ResourceDecision`。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 统一状态视图 —— Scheduler/engine 只读这些快照，不读具体实现对象
# --------------------------------------------------------------------------- #
@dataclass
class ClusterState:
    """集群资源快照（GPU 分配 + 网络容量占用）。"""

    total_gpu: int
    allocated_training_gpu: int
    allocated_inference_gpu: int
    free_gpu: int
    network_capacity: float
    network_used: float
    network_utilization: float


@dataclass
class TrainingState:
    """训练 workload 状态快照（§5）。

    Phase 3.2 扩展：gpu_utilization / gpu_memory_mb 是 Local Runtime 的真实测量
    （torch 侧采样）；Mock 填语义值（运行中=1.0，显存=0=未建模，Reality Gap 标 N/A）。
    """

    status: str  # RUNNING / DEGRADED / PAUSED / COMPLETED / FAILED
    allocated_gpu: int
    progress: float
    remaining_work: float
    throughput: float
    estimated_completion_time: float
    network_bandwidth: float
    gpu_utilization: float = 0.0  # workload 对其所分配 GPU 的实际使用率（0~1）
    gpu_memory_mb: float = 0.0  # GPU 显存占用 MB；Mock 无显存模型 → 0


@dataclass
class InferenceState:
    """推理 workload 状态快照（§6）。"""

    status: str  # RUNNING / OVERLOADED / RECOVERING
    allocated_gpu: int
    qps: float
    throughput: float
    capacity_qps: float
    p50: float
    p95: float
    p99: float
    queue_length: int
    slo: float
    slo_violation: bool
    network_bandwidth: float
    gpu_utilization: float = 0.0  # 0~1；Mock 按 throughput/capacity 估算
    gpu_memory_mb: float = 0.0  # GPU 显存占用 MB；Mock 无显存模型 → 0


@dataclass
class NetworkState:
    """网络状态快照（§13-17）。"""

    capacity_mbps: float
    demand_mbps: float
    utilization: float
    congestion_latency_ms: float
    throughput_scale: float  # 拥塞回压后训练吞吐的折扣系数（0~1）
    headroom_mbps: float
    congested: bool
    training_bandwidth_mbps: float = 0.0  # 训练占用带宽（Local 为实测，Mock 为公式）
    inference_bandwidth_mbps: float = 0.0  # 推理占用带宽（Local 为实测，Mock 为公式）


# --------------------------------------------------------------------------- #
# ResourceDecision —— 一次调度决策（§8）
# --------------------------------------------------------------------------- #
@dataclass
class ResourceDecision:
    """一次调度决策：新旧 GPU 配额 + 原因（含回放所需上下文）。

    字段：
      old/new_training_gpu、old/new_inference_gpu   目标配额
      reason                                         人类可读原因
      inference_p95 / network_utilization            决策时刻状态快照
      timestamp / scheduler / training_state         决策回放（decisions_*.csv）用
    """

    old_training_gpu: int
    new_training_gpu: int
    old_inference_gpu: int
    new_inference_gpu: int
    reason: str = ""
    inference_p95: float = 0.0
    network_utilization: float = 0.0  # 阶段二 Network-aware 预留
    timestamp: float = 0.0
    scheduler: str = ""
    training_state: str = ""

    @property
    def delta(self) -> int:
        return self.new_training_gpu - self.old_training_gpu

    @property
    def changed(self) -> bool:
        return self.new_training_gpu != self.old_training_gpu

    def to_dict(self) -> dict:
        return {
            "old_training_gpu": self.old_training_gpu,
            "new_training_gpu": self.new_training_gpu,
            "old_inference_gpu": self.old_inference_gpu,
            "new_inference_gpu": self.new_inference_gpu,
            "reason": self.reason,
            "inference_p95": round(self.inference_p95, 2),
            "network_utilization": self.network_utilization,
            "timestamp": self.timestamp,
            "scheduler": self.scheduler,
            "training_state": self.training_state,
        }


# --------------------------------------------------------------------------- #
# RuntimeAdapter —— 抽象接口
# --------------------------------------------------------------------------- #
class RuntimeAdapter(ABC):
    """运行时适配器：把「真实/模拟的集群 + workload + 网络」收敛成一个统一接口。

    调度器不直接操作 GPU/进程/网络对象，只通过本接口：
      读状态  → get_cluster_state / get_training_state / get_inference_state / get_network_state
      下指令  → scale_training / scale_inference / apply_resource_decision
      网络闸门 → network_expansion_feasible（spec §17，供 Network-aware 用）
    """

    # ---- 状态读取 ----
    @abstractmethod
    def get_cluster_state(self) -> ClusterState: ...

    @abstractmethod
    def get_training_state(self) -> TrainingState: ...

    @abstractmethod
    def get_inference_state(self) -> InferenceState: ...

    @abstractmethod
    def get_network_state(self) -> NetworkState: ...

    # ---- 资源控制 ----
    @abstractmethod
    def scale_training(self, target: int) -> None:
        """把训练配额设为 target（推理 = total - target，总量守恒）。"""

    @abstractmethod
    def scale_inference(self, target: int) -> None:
        """把推理配额设为 target（训练 = total - target，总量守恒）。"""

    @abstractmethod
    def apply_resource_decision(self, decision: ResourceDecision) -> None:
        """原子应用一次调度决策（配额变化 + workload 同步）。"""

    # ---- 网络感知（spec §17，Network-aware 调度器用）----
    @abstractmethod
    def network_expansion_feasible(
        self,
        old_training: int,
        new_training: int,
        old_inference: int,
        new_inference: int,
    ) -> tuple[bool, dict]:
        """从 (old_t, old_i) 迁到 (new_t, new_i) 是否不把网络推过容量。

        返回 (feasible, 明细)。明细含 released/delta/net/headroom，供决策日志解释。
        """

    # ---- 训练 workload 参数（scheduler 决策依赖）----
    @property
    @abstractmethod
    def training_min_gpu(self) -> int: ...

    @property
    @abstractmethod
    def training_initial_gpu(self) -> int: ...

    @property
    @abstractmethod
    def training_slowdown(self) -> float: ...

    @abstractmethod
    def would_allow_degradation(self, new_gpus: int) -> bool:
        """降级到 new_gpus 是否仍满足 max_allowed_slowdown（spec §12）。"""

    @property
    @abstractmethod
    def inference_overload_counter(self) -> int: ...

    @property
    @abstractmethod
    def inference_recovery_counter(self) -> int: ...
