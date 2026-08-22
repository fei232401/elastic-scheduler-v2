"""Mock Runtime Adapter —— 把现有 Mock 组件包装成统一 Runtime API（Phase 3.1）。

包装对象：
  - ResourceManager       GPU 配额流转
  - MockTrainingJob       训练行为（幂律吞吐/progress/状态机）
  - MockInferenceService  推理延迟/容量/过载
  - MockNetwork           带宽/拥塞/延迟/回压

Scheduler 只经本 Adapter 读写状态，不直接触碰以上任何对象。
本类不删除/改写任何模拟逻辑 —— 只是加一层接口（原则 2：Mock 实验必须继续工作）。

引擎每 tick 的调用序列（复刻原 engine 时序，保证确定性）：
  1. runtime.advance(qps)            # network.update + inference.step（网络先于推理）
  2. scheduler.step(ctx)             # ctx 从 runtime 状态视图组装
  3. runtime.apply_resource_decision # 应用决策（配额 + workload 同步）
  4. runtime.advance_training()      # training.step（含网络回压）
"""
from __future__ import annotations

from src.cluster.resource_manager import ResourceManager
from src.network.network import MockNetwork
from src.runtime.base import (
    ClusterState,
    InferenceState,
    NetworkState,
    ResourceDecision,
    RuntimeAdapter,
    TrainingState,
)
from src.workload.inference import MockInferenceService
from src.workload.training import MockTrainingJob


class MockRuntimeAdapter(RuntimeAdapter):
    """Mock 运行时：把现有 4 个模拟对象收敛成 RuntimeAdapter 接口。"""

    def __init__(
        self,
        rm: ResourceManager,
        training: MockTrainingJob,
        inference: MockInferenceService,
        network: MockNetwork,
    ) -> None:
        self.rm = rm
        self.training = training
        self.inference = inference
        self.network = network

    # ---------------- 状态读取 ----------------
    def get_cluster_state(self) -> ClusterState:
        return ClusterState(
            total_gpu=self.rm._pool.total,
            allocated_training_gpu=self.rm.training_gpus,
            allocated_inference_gpu=self.rm.inference_gpus,
            # 语义上的空闲卡 = total - 训练 - 推理（Mock 中推理是 total-training 的隐式占位，
            # rm.unused_gpus 只算池内训练登记，不能直接用）
            free_gpu=self.rm._pool.total - self.rm.training_gpus - self.rm.inference_gpus,
            network_capacity=self.network.capacity_mbps,
            network_used=self.network.demand_mbps,
            network_utilization=self.network.utilization,
        )

    def get_training_state(self) -> TrainingState:
        return TrainingState(
            status=self.training.state.value,
            allocated_gpu=self.training.gpus,
            progress=self.training.progress,
            remaining_work=self.training.remaining_work,
            throughput=self.training.throughput,
            estimated_completion_time=self.training.estimated_completion_time_s,
            network_bandwidth=self.network.training_bandwidth(self.training.gpus),
            # Mock 语义：运行中的训练任务占满其分配 GPU；显存未建模 → 0
            gpu_utilization=1.0 if self.training.state.value in ("RUNNING", "DEGRADED") else 0.0,
            gpu_memory_mb=0.0,
        )

    def get_inference_state(self) -> InferenceState:
        return InferenceState(
            status=self.inference.state.value,
            allocated_gpu=self.inference.gpus,
            qps=self.inference.incoming_qps,
            throughput=self.inference.throughput_qps,
            capacity_qps=self.inference.capacity_qps,
            p50=self.inference.p50_ms,
            p95=self.inference.p95_ms,
            p99=self.inference.p99_ms,
            queue_length=self.inference.queue_length,
            slo=self.inference.slo_p95_ms,
            slo_violation=self.inference.slo_violated,
            network_bandwidth=self.network.inference_bandwidth(
                self.inference.gpus, self.inference.incoming_qps
            ),
            # Mock 语义：推理 GPU 使用率 ≈ 吞吐/容量；显存未建模 → 0
            gpu_utilization=(
                min(self.inference.throughput_qps / self.inference.capacity_qps, 1.0)
                if self.inference.capacity_qps > 0
                else 0.0
            ),
            gpu_memory_mb=0.0,
        )

    def get_network_state(self) -> NetworkState:
        return NetworkState(
            capacity_mbps=self.network.capacity_mbps,
            demand_mbps=self.network.demand_mbps,
            utilization=self.network.utilization,
            congestion_latency_ms=self.network.latency_ms,
            throughput_scale=self.network.throughput_scale,
            headroom_mbps=self.network.headroom_mbps,
            congested=self.network.congested,
            # Phase 3.2：显式带宽拆分（§4.4）
            training_bandwidth_mbps=self.network.training_bandwidth(self.rm.training_gpus),
            inference_bandwidth_mbps=self.network.inference_bandwidth(
                self.rm.inference_gpus, self.inference.incoming_qps
            ),
        )

    # ---------------- 资源控制 ----------------
    def scale_training(self, target: int) -> None:
        """训练配额设为 target（推理 = total - target，总量守恒）。"""
        target = max(0, min(target, self.rm._pool.total))
        delta = target - self.rm.training_gpus
        self.rm.shift(delta)
        self.training.gpus = self.rm.training_gpus
        self.inference.set_gpus(self.rm.inference_gpus)

    def scale_inference(self, target: int) -> None:
        """推理配额设为 target（训练 = total - target，总量守恒）。"""
        target = max(0, min(target, self.rm._pool.total))
        new_training = self.rm.training_gpus + self.rm.inference_gpus - target
        self.scale_training(new_training)

    def apply_resource_decision(self, decision: ResourceDecision) -> None:
        if decision.changed:
            self.scale_training(decision.new_training_gpu)

    # ---------------- 网络感知（spec §17）----------------
    def network_expansion_feasible(
        self,
        old_training: int,
        new_training: int,
        old_inference: int,
        new_inference: int,
    ) -> tuple[bool, dict]:
        return self.network.expansion_feasible(
            old_training, new_training, old_inference, new_inference
        )

    # ---------------- 训练 workload 参数 ----------------
    @property
    def training_min_gpu(self) -> int:
        return self.rm.min_training

    @property
    def training_initial_gpu(self) -> int:
        return self.training.initial_gpu

    @property
    def training_slowdown(self) -> float:
        return self.training.slowdown_vs_initial

    def would_allow_degradation(self, new_gpus: int) -> bool:
        return self.training.would_allow_degradation(new_gpus)

    @property
    def inference_overload_counter(self) -> int:
        return self.inference._overload_counter

    @property
    def inference_recovery_counter(self) -> int:
        return self.inference._recovery_counter

    # ---------------- Mock 每 tick 执行（引擎驱动）----------------
    def advance(self, qps: float) -> None:
        """推进一个 tick 的「网络 + 推理」：网络先按当前配额刷新，再喂给推理。"""
        self.network.update(self.rm.training_gpus, self.rm.inference_gpus, qps)
        self.inference.step(qps, network_latency_ms=self.network.latency_ms)

    def advance_training(self) -> None:
        """推进训练一个 tick（用当前配额 + 网络回压折扣吞吐）。"""
        self.training.step(self.rm.training_gpus, throughput_scale=self.network.throughput_scale)
