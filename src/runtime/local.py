"""Local Runtime Adapter —— 同一 Runtime API 的真实实现（Phase 3.2，spec §4/§22，Step 2 骨架）。

与 Mock 的关键区别（诚实标注，spec §22，禁止把模拟当真实）：
  - Mock：advance() 用纯数学模型毫秒级推进一个「虚拟 tick」，无需真实进程；
  - Local：advance()/advance_training() 驱动真实进程（PyTorch 训练 / HTTP 推理 /
    tc+netem 网络整形）跑一个 wall-clock tick（tick_seconds），再采样真实指标。

GPU 模型（本机物理 GPU = 1，RTX 4070 8GB）：
  - 调度器看到的 total_gpus（默认 8）是「虚拟配额」；
  - 真实 workload 始终跑在 1 张物理 GPU 上；配额 > 1 时的吞吐按 EMULATED 缩放模型
    换算（模型由 Step 9 用真实单卡测量校准）。
  - 所有「多 GPU」结果一律标记 EMULATED，不得冒充真实多卡数据。

注入对象（Step 3/5/7 实现，本模块以 duck-type 契约为准，不做无谓抽象）：
  - training : 真实 PyTorch 训练 workload（.tick(seconds) 跑真实计算，属性给真实测量）
  - inference: 真实 HTTP 推理 server + 加载器（.serve(qps, seconds) 驱动真实请求）
  - network  : tc/netem 网络控制器（.update() 施加整形 + 采样）
"""
from __future__ import annotations

from src.cluster.resource_manager import ResourceManager
from src.runtime.base import (
    ClusterState,
    InferenceState,
    NetworkState,
    ResourceDecision,
    RuntimeAdapter,
    TrainingState,
)


class LocalRuntimeAdapter(RuntimeAdapter):
    """本地运行时：配额记账复用 ResourceManager，workload/网络经注入对象驱动。"""

    def __init__(
        self,
        rm: ResourceManager,
        training,
        inference,
        network,
        tick_seconds: float = 5.0,
    ) -> None:
        self.rm = rm
        self.training = training
        self.inference = inference
        self.network = network
        self.tick_seconds = tick_seconds

    # ---------------- 状态读取 ----------------
    def get_cluster_state(self) -> ClusterState:
        return ClusterState(
            total_gpu=self.rm._pool.total,
            allocated_training_gpu=self.rm.training_gpus,
            allocated_inference_gpu=self.rm.inference_gpus,
            # 与 Mock 同一语义：空闲 = total - 训练 - 推理
            free_gpu=self.rm._pool.total - self.rm.training_gpus - self.rm.inference_gpus,
            network_capacity=self.network.capacity_mbps,
            network_used=self.network.demand_mbps,
            network_utilization=self.network.utilization,
        )

    def get_training_state(self) -> TrainingState:
        return TrainingState(
            status=self.training.state,
            allocated_gpu=self.rm.training_gpus,
            progress=self.training.progress,
            remaining_work=self.training.remaining_work,
            throughput=self.training.throughput,
            estimated_completion_time=self.training.estimated_completion_time_s,
            network_bandwidth=self.network.training_bandwidth_mbps,
            gpu_utilization=self.training.gpu_utilization,
            gpu_memory_mb=self.training.gpu_memory_mb,
        )

    def get_inference_state(self) -> InferenceState:
        return InferenceState(
            status=self.inference.state,
            allocated_gpu=self.rm.inference_gpus,
            qps=self.inference.qps,
            throughput=self.inference.throughput_qps,
            capacity_qps=self.inference.capacity_qps,
            p50=self.inference.p50_ms,
            p95=self.inference.p95_ms,
            p99=self.inference.p99_ms,
            queue_length=self.inference.queue_length,
            slo=self.inference.slo_p95_ms,
            slo_violation=self.inference.slo_violated,
            network_bandwidth=self.network.inference_bandwidth_mbps,
            gpu_utilization=self.inference.gpu_utilization,
            gpu_memory_mb=self.inference.gpu_memory_mb,
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
            training_bandwidth_mbps=self.network.training_bandwidth_mbps,
            inference_bandwidth_mbps=self.network.inference_bandwidth_mbps,
        )

    # ---------------- 资源控制 ----------------
    def scale_training(self, target: int) -> None:
        """训练配额设为 target（推理 = total - target，总量守恒）。"""
        target = max(0, min(target, self.rm._pool.total))
        delta = target - self.rm.training_gpus
        self.rm.shift(delta)
        self.training.set_gpus(self.rm.training_gpus)
        self.inference.set_gpus(self.rm.inference_gpus)

    def scale_inference(self, target: int) -> None:
        """推理配额设为 target（训练 = total - target，总量守恒）。"""
        target = max(0, min(target, self.rm._pool.total))
        self.scale_training(self.rm.training_gpus + self.rm.inference_gpus - target)

    def apply_resource_decision(self, decision: ResourceDecision) -> None:
        if decision.changed:
            self.scale_training(decision.new_training_gpu)

    # ---------------- 网络感知（spec §17，Network-aware 调度器用）----------------
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
        return self.inference.overload_counter

    @property
    def inference_recovery_counter(self) -> int:
        return self.inference.recovery_counter

    # ---------------- Local 生命周期（真实时间步进，Step 3-5 接真实 workload）----------------
    def advance(self, qps: float) -> None:
        """一个 tick：先施加网络整形，再驱动真实 HTTP 推理流量（跑 tick_seconds）。

        网络拥塞延迟（tc/netem，REAL）叠加进上报给调度器的推理 P95（镜像 Mock 机制）。
        """
        self.network.update(self.rm.training_gpus, self.rm.inference_gpus, qps)
        self.inference.serve(qps, self.tick_seconds, network_latency_ms=self.network.latency_ms)

    def advance_training(self) -> None:
        """一个 tick：真实 PyTorch 训练推进 tick_seconds。"""
        self.training.tick(self.tick_seconds)

    def start(self) -> None:
        """启动真实进程：网络整形 → 推理 server → 训练。"""
        for obj in (self.network, self.inference, self.training):
            start = getattr(obj, "start", None)
            if start is not None:
                start()

    def stop(self) -> None:
        """停止真实进程：先停训练/推理（停止消费），再移除网络整形。"""
        for obj in (self.training, self.inference, self.network):
            stop = getattr(obj, "stop", None)
            if stop is not None:
                stop()
