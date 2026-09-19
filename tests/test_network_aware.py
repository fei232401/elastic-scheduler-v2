"""Network-aware Elastic Scheduler 测试（spec §17：扩容前网络闸门）。

Phase 3.1：调度器不再直接持有 MockNetwork，改为经 MockRuntimeAdapter（ctx.runtime）读取
网络状态与扩容闸门 —— 本测试用 adapter 构造上下文。
"""
from src.cluster.resource_manager import ResourceManager
from src.network.network import MockNetwork
from src.runtime.mock import MockRuntimeAdapter
from src.scheduler.base import SchedulerContext
from src.scheduler.network_aware import NetworkAwareElasticScheduler
from src.workload.inference import MockInferenceService
from src.workload.training import MockTrainingJob

CFG = {
    "simulation": {"tick_seconds": 5},
    "scheduler": {
        "max_scale_step": 1,
        "min_gpu_hold_duration_ticks": 24,
        "approach_ratio": 0.8,
        "healthy_ratio": 0.6,
    },
    "training": {"initial_gpu": 4, "min_gpu": 1},
}
NET_CFG = {"network": {
    "total_bandwidth_mbps": 2200,
    "base_latency_ms": 2,
    "latency_gain_per_overload": 250,
    "congestion_start_util": 0.5,
    "training_bw_scale": 0.2,
    "train_bw_coeff_a": 50,
    "train_bw_coeff_b": 150,
    "per_gpu_infer_bw": 600,
    "bw_per_qps": 30,
    "training_penalty_per_unit": 0.3,
    "max_training_penalty": 0.7,
}}


def build_adapter(cfg: dict) -> MockRuntimeAdapter:
    t = cfg.get("training", {})
    rm = ResourceManager(total_gpus=8, initial_training=t.get("initial_gpu", 4),
                         min_training=t.get("min_gpu", 1))
    training = MockTrainingJob(cfg)
    inference = MockInferenceService(cfg)
    network = MockNetwork(cfg)
    return MockRuntimeAdapter(rm, training, inference, network)


def net_ctx(training=4, inference=4, p95=500.0, slo=300.0, state="OVERLOADED",
            now_s=0.0, qps=30.0, cfg=None) -> SchedulerContext:
    adapter = build_adapter(cfg or {**NET_CFG, "training": CFG["training"]})
    adapter.advance(qps)
    return SchedulerContext(
        training_gpus=training,
        inference_gpus=inference,
        training_min_gpu=adapter.training_min_gpu,
        training_initial_gpu=adapter.training_initial_gpu,
        inference_p95_ms=p95,
        slo_p95_ms=slo,
        inference_state=state,
        training_throughput=0.0,
        training_slowdown=1.0,
        training_would_allow_degradation=True,
        inference_overload_counter=0,
        inference_recovery_counter=0,
        now_s=now_s,
        runtime=adapter,
        extra={},
    )


class TestNetworkGate:
    def test_blocks_reclaim_when_network_saturated(self):
        """网络饱和 + 释放带宽 < 新增推理带宽 → 拒绝让渡（spec §17）。"""
        s = NetworkAwareElasticScheduler(CFG)
        d = s.step(net_ctx())
        assert d.changed is False
        assert "BLOCKED" in d.reason
        assert "Network-aware" in d.reason

    def test_allows_reclaim_when_network_has_headroom(self):
        """网络富余 → 正常让渡（与 GPU-only Elastic 一致）。"""
        big_cfg = {
            **NET_CFG,
            "training": CFG["training"],
            "network": {**NET_CFG["network"], "total_bandwidth_mbps": 30000},
        }
        s = NetworkAwareElasticScheduler(CFG)
        ctx = net_ctx(cfg=big_cfg)
        d = s.step(ctx)
        assert d.changed is True
        assert d.new_training_gpu == 3 and d.new_inference_gpu == 5

    def test_blocked_reclaim_does_not_set_hold(self):
        """被拒不算变更 → 下一 tick 网络有余量可立即重试（不等 min_hold）。"""
        s = NetworkAwareElasticScheduler(CFG)
        d1 = s.step(net_ctx(p95=500, now_s=0.0))
        assert d1.changed is False
        assert s._last_change_tick is None

    def test_returns_gpu_when_healthy_and_network_ok(self):
        """健康 + 归还不压网 → 归还 GPU 给训练。"""
        big_cfg = {
            **NET_CFG,
            "training": CFG["training"],
            "network": {**NET_CFG["network"], "total_bandwidth_mbps": 30000},
        }
        ctx = net_ctx(training=3, inference=5, p95=100, slo=300, state="RUNNING",
                      now_s=0.0, qps=10, cfg=big_cfg)
        s = NetworkAwareElasticScheduler(CFG)
        d = s.step(ctx)
        assert d.changed is True
        assert d.new_training_gpu == 4 and d.new_inference_gpu == 4
