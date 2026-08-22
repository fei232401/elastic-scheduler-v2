"""Runtime Abstraction 测试（Phase 3.1）。

覆盖：
  - ResourceDecision 增强字段（timestamp/scheduler/training_state）与 to_dict
  - 统一状态视图（Cluster/Training/Inference/Network State）
  - MockRuntimeAdapter：状态读取、scale_*、apply_resource_decision、网络闸门、训练参数
  - 四策略 × MockRuntime 端到端（任务书 Test 1-4：Static/HardPreemption/Elastic/Network-aware 全部能跑）
"""
from __future__ import annotations

from src.cluster.resource_manager import ResourceManager
from src.network.network import MockNetwork
from src.runtime.base import (
    ClusterState,
    InferenceState,
    NetworkState,
    ResourceDecision,
    TrainingState,
)
from src.runtime.mock import MockRuntimeAdapter
from src.scheduler.base import SchedulerContext
from src.scheduler.elastic import ElasticScheduler
from src.scheduler.network_aware import NetworkAwareElasticScheduler
from src.scheduler.preemptive import HardPreemptionScheduler
from src.scheduler.static import StaticScheduler
from src.workload.inference import MockInferenceService
from src.workload.training import MockTrainingJob

# ---------------- 构造辅助 ----------------

CFG = {
    "simulation": {"tick_seconds": 5},
    "training": {
        "total_work": 100000, "initial_gpu": 6, "min_gpu": 1,
        "base_throughput": 15, "scaling_exponent": 0.85, "max_allowed_slowdown": 3.0,
    },
    "inference": {
        "capacity_per_gpu": 10, "slo_p95_ms": 300, "p95_base_ms": 200,
        "p95_overload_ms": 900, "overload_trigger_ticks": 3, "recovery_trigger_ticks": 5,
    },
    "network": {
        "total_bandwidth_mbps": 30000, "base_latency_ms": 2, "latency_gain_per_overload": 250,
        "congestion_start_util": 0.5, "training_bw_scale": 1.0,
        "train_bw_coeff_a": 50, "train_bw_coeff_b": 150,
        "per_gpu_infer_bw": 100, "bw_per_qps": 20,
        "training_penalty_per_unit": 0.3, "max_training_penalty": 0.7,
    },
}

# 网络受限 + 训练曲线平（实验 B 风格）→ 扩容闸门必拦截
NET_BOUND_CFG = {
    **CFG,
    "training": {**CFG["training"], "initial_gpu": 4, "min_gpu": 1},
    "network": {**CFG["network"],
                "total_bandwidth_mbps": 2200, "training_bw_scale": 0.2,
                "per_gpu_infer_bw": 600, "bw_per_qps": 30},
}

SC_CFG = {
    "simulation": {"tick_seconds": 5},
    "scheduler": {
        "max_scale_step": 1, "min_gpu_hold_duration_ticks": 24,
        "approach_ratio": 0.8, "healthy_ratio": 0.6, "preempt_min_training": 1,
    },
}


def build_adapter(cfg: dict = None) -> MockRuntimeAdapter:
    cfg = cfg or CFG
    t = cfg["training"]
    rm = ResourceManager(total_gpus=8, initial_training=t["initial_gpu"], min_training=t["min_gpu"])
    training = MockTrainingJob(cfg)
    inference = MockInferenceService(cfg)
    network = MockNetwork(cfg)
    return MockRuntimeAdapter(rm, training, inference, network)


def run_ticks(adapter: MockRuntimeAdapter, scheduler, qps_seq: list[float]) -> list[ResourceDecision]:
    """复刻 engine 每 tick 时序：advance → ctx → step → apply → advance_training。"""
    decisions: list[ResourceDecision] = []
    for tick, qps in enumerate(qps_seq):
        adapter.advance(qps)
        cl = adapter.get_cluster_state()
        tr = adapter.get_training_state()
        inf = adapter.get_inference_state()
        ctx = SchedulerContext(
            training_gpus=cl.allocated_training_gpu,
            inference_gpus=cl.allocated_inference_gpu,
            training_min_gpu=adapter.training_min_gpu,
            training_initial_gpu=adapter.training_initial_gpu,
            inference_p95_ms=inf.p95,
            slo_p95_ms=inf.slo,
            inference_state=inf.status,
            training_throughput=tr.throughput,
            training_slowdown=adapter.training_slowdown,
            training_would_allow_degradation=adapter.would_allow_degradation(
                cl.allocated_training_gpu - 1
            ),
            inference_overload_counter=adapter.inference_overload_counter,
            inference_recovery_counter=adapter.inference_recovery_counter,
            now_s=tick * 5.0,
            runtime=adapter,
            extra={"would_allow_degradation": adapter.would_allow_degradation},
        )
        d = scheduler.step(ctx)
        if d.changed:
            d.timestamp = tick * 5.0
            d.scheduler = scheduler.name
            d.training_state = tr.status
            adapter.apply_resource_decision(d)
            decisions.append(d)
        adapter.advance_training()
    return decisions


# ---------------- ResourceDecision 增强 ----------------

class TestResourceDecision:
    def test_to_dict_includes_replay_fields(self):
        d = ResourceDecision(
            old_training_gpu=6, new_training_gpu=5,
            old_inference_gpu=2, new_inference_gpu=3,
            reason="Rule1: reclaim", inference_p95=350.0,
            timestamp=10.0, scheduler="elastic", training_state="DEGRADED",
        )
        dd = d.to_dict()
        assert dd["timestamp"] == 10.0
        assert dd["scheduler"] == "elastic"
        assert dd["training_state"] == "DEGRADED"
        assert dd["old_training_gpu"] == 6 and dd["new_training_gpu"] == 5

    def test_defaults_when_not_set(self):
        d = ResourceDecision(6, 6, 2, 2, reason="noop")
        assert d.timestamp == 0.0 and d.scheduler == "" and d.training_state == ""
        assert d.changed is False and d.delta == 0


# ---------------- 状态视图 ----------------

class TestStateViews:
    def test_state_dataclasses_hold_fields(self):
        cl = ClusterState(8, 6, 2, 0, 10000.0, 500.0, 0.05)
        tr = TrainingState("RUNNING", 6, 100.0, 99900.0, 68.5, 1458.0, 2700.0)
        inf = InferenceState("RUNNING", 2, 10.0, 10.0, 20.0, 150.0, 200.0, 210.0, 0, 300.0, False, 200.0)
        nt = NetworkState(30000.0, 500.0, 0.05, 2.0, 1.0, 29500.0, False)
        assert cl.free_gpu == 0
        assert tr.remaining_work == 99900.0
        assert inf.slo == 300.0 and inf.slo_violation is False
        assert nt.capacity_mbps == 30000.0 and nt.congested is False
        # Phase 3.2 扩展字段默认值（Local Runtime 填充真实测量）
        assert tr.gpu_utilization == 0.0 and tr.gpu_memory_mb == 0.0
        assert inf.gpu_utilization == 0.0 and inf.gpu_memory_mb == 0.0
        assert nt.training_bandwidth_mbps == 0.0 and nt.inference_bandwidth_mbps == 0.0


class TestMockRuntimeState:
    def test_get_cluster_state_reports_allocations(self):
        a = build_adapter()
        cl = a.get_cluster_state()
        assert cl.total_gpu == 8
        assert cl.allocated_training_gpu == 6 and cl.allocated_inference_gpu == 2
        assert cl.free_gpu == 0

    def test_get_training_state_reports_workload(self):
        a = build_adapter()
        tr = a.get_training_state()
        assert tr.status == "RUNNING" and tr.allocated_gpu == 6
        assert tr.progress == 0.0 and tr.remaining_work == 100000.0
        assert tr.network_bandwidth == 2700.0  # train_bw(6) = 50*36+150*6
        # Phase 3.2：Mock 语义 util=1.0（运行中），显存未建模 → 0
        assert tr.gpu_utilization == 1.0 and tr.gpu_memory_mb == 0.0

    def test_get_inference_state_reports_latency(self):
        a = build_adapter()
        a.advance(10.0)  # 低流量，不超容量
        inf = a.get_inference_state()
        assert inf.qps == 10.0 and inf.allocated_gpu == 2
        assert inf.p95 <= 300.0 and inf.slo_violation is False

    def test_get_network_state_reports_congestion(self):
        a = build_adapter()
        a.advance(30.0)  # 高流量，推理带宽挤占
        nt = a.get_network_state()
        assert nt.demand_mbps > 0 and nt.capacity_mbps == 30000.0
        assert nt.congested is False  # 容量富余
        # Phase 3.2：显式带宽拆分（训练 + 推理 = 需求）
        assert nt.training_bandwidth_mbps > 0.0 and nt.inference_bandwidth_mbps > 0.0
        assert abs(
            nt.training_bandwidth_mbps + nt.inference_bandwidth_mbps - nt.demand_mbps
        ) < 1e-6


# ---------------- 资源控制 ----------------

class TestMockRuntimeControl:
    def test_scale_training_preserves_conservation(self):
        a = build_adapter()
        a.scale_training(4)
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 4 and cl.allocated_inference_gpu == 4
        assert a.training.gpus == 4 and a.inference.gpus == 4

    def test_scale_inference_preserves_conservation(self):
        a = build_adapter()
        a.scale_inference(5)
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 3 and cl.allocated_inference_gpu == 5

    def test_apply_resource_decision_noop_when_unchanged(self):
        a = build_adapter()
        d = ResourceDecision(6, 6, 2, 2, reason="noop")
        a.apply_resource_decision(d)
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 6 and cl.allocated_inference_gpu == 2

    def test_apply_resource_decision_applies_shift(self):
        a = build_adapter()
        d = ResourceDecision(6, 5, 2, 3, reason="Rule1")
        a.apply_resource_decision(d)
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 5 and cl.allocated_inference_gpu == 3

    def test_network_expansion_feasible_delegates(self):
        a = build_adapter()
        feasible, detail = a.network_expansion_feasible(6, 5, 2, 3)
        assert feasible is True  # 容量富余
        assert "released_training_bw" in detail

    def test_training_params_exposed(self):
        a = build_adapter()
        assert a.training_min_gpu == 1 and a.training_initial_gpu == 6
        assert a.training_slowdown == 1.0
        assert a.would_allow_degradation(3) is True
        assert a.would_allow_degradation(0) is False  # 低于 min_gpu
        assert a.inference_overload_counter == 0


# ---------------- 四策略 × MockRuntime 端到端（任务书 Test 1-4） ----------------

class TestStaticWithRuntime:
    def test_never_changes_and_training_progresses(self):
        a = build_adapter()
        s = StaticScheduler(SC_CFG)
        qps = [10.0] * 20 + [60.0] * 20
        decisions = run_ticks(a, s, qps)
        assert decisions == []  # static 永不切换
        tr = a.get_training_state()
        assert tr.progress > 0.0
        assert a.get_cluster_state().allocated_training_gpu == 6


class TestElasticWithRuntime:
    def test_reclaims_gpu_under_overload(self):
        a = build_adapter()
        s = ElasticScheduler(SC_CFG)
        decisions = run_ticks(a, s, [60.0] * 10)  # 持续超容量
        assert decisions, "overload 下应发生让渡"
        assert decisions[0].delta == -1
        assert decisions[0].scheduler == "elastic"
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 5 and cl.allocated_inference_gpu == 3

    def test_records_timestamp_and_training_state(self):
        a = build_adapter()
        s = ElasticScheduler(SC_CFG)
        decisions = run_ticks(a, s, [60.0] * 10)
        assert decisions[0].timestamp > 0.0
        assert decisions[0].training_state in ("RUNNING", "DEGRADED")


class TestHardPreemptionWithRuntime:
    def test_slams_to_floor(self):
        a = build_adapter()
        s = HardPreemptionScheduler(SC_CFG)
        decisions = run_ticks(a, s, [60.0] * 10)
        assert decisions, "过载下应发生抢占"
        assert abs(decisions[0].delta) > 1  # 一次性大幅回收
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 1 and cl.allocated_inference_gpu == 7


class TestNetworkAwareWithRuntime:
    def test_gate_blocks_expansion_when_saturated(self):
        a = build_adapter(NET_BOUND_CFG)
        s = NetworkAwareElasticScheduler(SC_CFG)
        run_ticks(a, s, [30.0] * 10)
        # 网络饱和 + 释放带宽 < 新增 → 扩容被闸门拦截，配额不动
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 4 and cl.allocated_inference_gpu == 4
        feasible, _ = a.network_expansion_feasible(4, 3, 4, 5)
        assert feasible is False

    def test_gate_allows_expansion_when_network_abundant(self):
        a = build_adapter(CFG)  # 容量 30000 富余
        s = NetworkAwareElasticScheduler(SC_CFG)
        decisions = run_ticks(a, s, [60.0] * 10)
        assert decisions, "GPU 瓶颈 + 网络富余 → 正常扩容"
        assert decisions[0].delta == -1
