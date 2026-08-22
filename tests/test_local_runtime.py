"""LocalRuntimeAdapter 骨架测试（Phase 3.2 Step 2）。

用**最小测试桩**验证 adapter 的接线：状态委托、配额守恒、生命周期时序、四策略可驱动。
桩不是真实 workload —— 真实 PyTorch / HTTP / tc+netem 分别在 Step 3/5/7 接入并替换。
桩的行为刻意接近 Mock 语义，仅用于证明「Local 遵循与 Mock 同一个 Runtime Contract」
（spec §3.2/Gate 4），不含任何「冒充真实多卡」的数据。
"""
from __future__ import annotations

from src.cluster.resource_manager import ResourceManager
from src.runtime.local import LocalRuntimeAdapter
from src.scheduler.base import SchedulerContext
from src.scheduler.elastic import ElasticScheduler
from src.scheduler.network_aware import NetworkAwareElasticScheduler
from src.scheduler.preemptive import HardPreemptionScheduler
from src.scheduler.static import StaticScheduler

# ---------------- 最小测试桩（接线验证用；Step 3/5/7 换真实实现） ----------------

CFG = {
    "simulation": {"total_gpus": 8, "tick_seconds": 5},
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
    "scheduler": {
        "max_scale_step": 1, "min_gpu_hold_duration_ticks": 24,
        "approach_ratio": 0.8, "healthy_ratio": 0.6, "preempt_min_training": 1,
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


class StubTraining:
    """训练桩：幂律吞吐 + progress 累积。仅接线验证，Step 3 换真实 PyTorch。"""

    def __init__(self, cfg: dict) -> None:
        t = cfg["training"]
        self.initial_gpu = t["initial_gpu"]
        self.min_gpu = t["min_gpu"]
        self.total_work = t["total_work"]
        self.base = t["base_throughput"]
        self.exp = t["scaling_exponent"]
        self.gpus = self.initial_gpu
        self.progress = 0.0
        self.state = "RUNNING"
        self.gpu_utilization = 1.0
        self.gpu_memory_mb = 0.0

    def throughput_at(self, g: int) -> float:
        return self.base * (g**self.exp)

    @property
    def throughput(self) -> float:
        return self.throughput_at(self.gpus)

    @property
    def remaining_work(self) -> float:
        return max(0.0, self.total_work - self.progress)

    @property
    def estimated_completion_time_s(self) -> float:
        tp = self.throughput
        return self.remaining_work / tp if tp > 0 else float("inf")

    @property
    def slowdown_vs_initial(self) -> float:
        return self.throughput_at(self.initial_gpu) / max(self.throughput, 1e-9)

    def set_gpus(self, g: int) -> None:
        self.gpus = g
        self.state = "RUNNING" if g >= self.initial_gpu else "DEGRADED"

    def would_allow_degradation(self, new_gpus: int) -> bool:
        return new_gpus >= self.min_gpu

    def tick(self, seconds: float) -> None:
        self.progress = min(self.total_work, self.progress + self.throughput * seconds)
        if self.progress >= self.total_work:
            self.state = "COMPLETED"


class StubInference:
    """推理桩：容量 + 超载延迟 + 防抖计数。仅接线验证，Step 5 换真实 HTTP server。"""

    def __init__(self, cfg: dict) -> None:
        i = cfg["inference"]
        sim = cfg["simulation"]
        t = cfg["training"]
        self.capacity_per_gpu = i["capacity_per_gpu"]
        self.slo_p95_ms = i["slo_p95_ms"]
        self.p95_base_ms = i["p95_base_ms"]
        self.p95_overload_ms = i["p95_overload_ms"]
        self.overload_trigger_ticks = i["overload_trigger_ticks"]
        self.recovery_trigger_ticks = i["recovery_trigger_ticks"]
        self.gpus = sim["total_gpus"] - t["initial_gpu"]
        self.qps = 0.0
        self.throughput_qps = 0.0
        self.p50_ms = 0.0
        self.p95_ms = 0.0
        self.p99_ms = 0.0
        self.queue_length = 0
        self.slo_violated = False
        self.state = "RUNNING"
        self.gpu_utilization = 0.0
        self.gpu_memory_mb = 0.0
        self._oc = 0
        self._rc = 0

    @property
    def capacity_qps(self) -> float:
        return self.capacity_per_gpu * self.gpus

    @property
    def overload_counter(self) -> int:
        return self._oc

    @property
    def recovery_counter(self) -> int:
        return self._rc

    def set_gpus(self, g: int) -> None:
        self.gpus = g

    def serve(self, qps: float, seconds: float, network_latency_ms: float = 0.0) -> None:
        """一个 tick 的简化延迟模型：超容量 → P95 拉高 + 排队（接线验证用）。"""
        self.qps = qps
        cap = self.capacity_qps
        if cap > 0 and qps <= cap:
            util = qps / cap
            self.p95_ms = self.p95_base_ms * (0.75 + 0.25 * util)
            self.p50_ms = self.p95_ms * 0.6
            self.p99_ms = self.p95_ms
            self.queue_length = 0
        else:
            ratio = (qps / cap) if cap > 0 else 2.0
            self.p95_ms = self.p95_base_ms + (self.p95_overload_ms - self.p95_base_ms) * min(
                1.0, (ratio - 1.0) ** 1.5
            )
            self.p50_ms = self.p95_ms * 0.6
            self.p99_ms = self.p95_ms * 1.1
            self.queue_length = int(round((qps - cap) * 2.0)) if cap > 0 else 1
        # 网络拥塞延迟叠加（镜像 adapter 传入的 network_latency_ms）
        self.p95_ms += network_latency_ms
        self.p50_ms += network_latency_ms * 0.5
        self.p99_ms += network_latency_ms
        self.throughput_qps = min(qps, cap)
        self.slo_violated = self.p95_ms > self.slo_p95_ms
        self.gpu_utilization = min(self.throughput_qps / cap, 1.0) if cap > 0 else 0.0

        if self.slo_violated:
            self._oc += 1
            self._rc = 0
        else:
            self._rc += 1
            self._oc = 0
        if self._oc >= self.overload_trigger_ticks and self.state != "OVERLOADED":
            self.state = "OVERLOADED"
        elif self._rc >= self.recovery_trigger_ticks and self.state == "OVERLOADED":
            self.state = "RUNNING"
            self._oc = 0
            self._rc = 0


class StubNetwork:
    """网络桩：带宽核算 + 简化拥塞回压 + §17 闸门。仅接线验证，Step 7 换 tc/netem。"""

    def __init__(self, cfg: dict) -> None:
        n = cfg["network"]
        self.capacity_mbps = n["total_bandwidth_mbps"]
        self.base_latency_ms = n["base_latency_ms"]
        self.training_bw_scale = n["training_bw_scale"]
        self.train_a = n["train_bw_coeff_a"]
        self.train_b = n["train_bw_coeff_b"]
        self.bw_per_qps = n["bw_per_qps"]
        self.per_gpu_infer_bw = n["per_gpu_infer_bw"]
        self.latency_gain_per_overload = n["latency_gain_per_overload"]
        self.training_penalty_per_unit = n["training_penalty_per_unit"]
        self.demand_mbps = 0.0
        self.latency_ms = self.base_latency_ms
        self.throughput_scale = 1.0
        self.headroom_mbps = self.capacity_mbps
        self.congested = False
        self.training_bandwidth_mbps = 0.0
        self.inference_bandwidth_mbps = 0.0

    def training_bw(self, g: int) -> float:
        return self.training_bw_scale * (self.train_a * g * g + self.train_b * g)

    @property
    def utilization(self) -> float:
        return self.demand_mbps / self.capacity_mbps if self.capacity_mbps > 0 else 0.0

    def update(self, training_gpus: int, inference_gpus: int, qps: float) -> None:
        self.training_bandwidth_mbps = self.training_bw(training_gpus)
        self.inference_bandwidth_mbps = qps * self.bw_per_qps + inference_gpus * self.per_gpu_infer_bw
        self.demand_mbps = self.training_bandwidth_mbps + self.inference_bandwidth_mbps
        self.congested = self.utilization > 1.0
        self.headroom_mbps = max(0.0, self.capacity_mbps - self.demand_mbps)
        over = max(0.0, self.utilization - 1.0)
        self.latency_ms = self.base_latency_ms + over * self.latency_gain_per_overload
        self.throughput_scale = max(0.0, min(1.0, 1.0 - over * self.training_penalty_per_unit))

    def expansion_feasible(
        self, old_t: int, new_t: int, old_i: int, new_i: int
    ) -> tuple[bool, dict]:
        """简化 §17：释放训练带宽 vs 新增推理带宽，不把网络推过容量。"""
        released = self.training_bw(old_t) - self.training_bw(new_t)
        added = (new_i - old_i) * self.per_gpu_infer_bw
        net = released - added  # 与 Mock 同构：>0 表示释放带宽不足以覆盖新增推理带宽
        detail = {
            "released_training_bw": round(released, 1),
            "delta_inference_bw": round(added, 1),
            "net_bandwidth_delta": round(net, 1),
            "headroom_mbps": round(self.headroom_mbps, 1),
        }
        # 等价于 Mock 的 delta_infer - released <= headroom（净新增带宽需求 <= 余量）
        return net >= -self.headroom_mbps, detail


def build_adapter(cfg: dict = None) -> LocalRuntimeAdapter:
    cfg = cfg or CFG
    t = cfg["training"]
    sim = cfg["simulation"]
    rm = ResourceManager(
        total_gpus=sim["total_gpus"],
        initial_training=t["initial_gpu"],
        min_training=t["min_gpu"],
    )
    return LocalRuntimeAdapter(
        rm, StubTraining(cfg), StubInference(cfg), StubNetwork(cfg),
        tick_seconds=sim.get("tick_seconds", 5),
    )


def run_ticks(adapter: LocalRuntimeAdapter, scheduler, qps_seq: list[float]) -> list:
    """复刻 engine 每 tick 时序，驱动四策略经 Local adapter（Gate 4 接线证明）。"""
    decisions = []
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
            now_s=tick * adapter.tick_seconds,
            runtime=adapter,
            extra={"would_allow_degradation": adapter.would_allow_degradation},
        )
        d = scheduler.step(ctx)
        if d.changed:
            d.timestamp = tick * adapter.tick_seconds
            d.scheduler = scheduler.name
            d.training_state = tr.status
            adapter.apply_resource_decision(d)
            decisions.append(d)
        adapter.advance_training()
    return decisions


# ---------------- 状态读取委托 ----------------

class TestLocalStateReads:
    def test_cluster_state_reports_allocations(self):
        a = build_adapter()
        cl = a.get_cluster_state()
        assert cl.total_gpu == 8
        assert cl.allocated_training_gpu == 6 and cl.allocated_inference_gpu == 2
        assert cl.free_gpu == 0

    def test_training_state_delegates_to_workload(self):
        a = build_adapter()
        a.advance_training()  # 一个 tick 的训练
        tr = a.get_training_state()
        assert tr.status == "RUNNING" and tr.allocated_gpu == 6
        assert tr.progress > 0.0
        assert tr.gpu_utilization == 1.0 and tr.gpu_memory_mb == 0.0

    def test_inference_state_delegates_after_serve(self):
        a = build_adapter()
        a.advance(60.0)  # 超容量 → P95 拉高
        inf = a.get_inference_state()
        assert inf.qps == 60.0
        assert inf.p95 > inf.slo  # 过载
        assert inf.slo_violation is True

    def test_network_state_reports_demand_and_split(self):
        a = build_adapter()
        a.advance(30.0)
        nt = a.get_network_state()
        assert nt.capacity_mbps == 30000.0
        assert nt.training_bandwidth_mbps > 0.0 and nt.inference_bandwidth_mbps > 0.0
        assert abs(nt.training_bandwidth_mbps + nt.inference_bandwidth_mbps - nt.demand_mbps) < 1e-6


# ---------------- 资源控制 ----------------

class TestLocalControl:
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
        from src.runtime.base import ResourceDecision
        a.apply_resource_decision(ResourceDecision(6, 6, 2, 2, reason="noop"))
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 6 and cl.allocated_inference_gpu == 2

    def test_apply_resource_decision_applies_shift(self):
        a = build_adapter()
        from src.runtime.base import ResourceDecision
        a.apply_resource_decision(ResourceDecision(6, 5, 2, 3, reason="Rule1"))
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
        assert a.would_allow_degradation(0) is False
        assert a.inference_overload_counter == 0

    def test_start_stop_is_safe_without_workload_lifecycle(self):
        a = build_adapter()
        a.start()  # 桩无 start/stop → getattr 默认跳过
        a.stop()


# ---------------- 四策略 × Local adapter 端到端（Gate 4 接线证明） ----------------

class TestStrategiesWithLocalRuntime:
    def test_static_never_changes(self):
        a = build_adapter()
        s = StaticScheduler(CFG)
        decisions = run_ticks(a, s, [10.0] * 10 + [60.0] * 10)
        assert decisions == []
        assert a.get_cluster_state().allocated_training_gpu == 6

    def test_elastic_reclaims_gpu_under_overload(self):
        a = build_adapter()
        s = ElasticScheduler(CFG)
        decisions = run_ticks(a, s, [60.0] * 10)
        assert decisions, "overload 下应发生让渡"
        assert decisions[0].delta == -1
        assert a.get_cluster_state().allocated_training_gpu == 5

    def test_hard_preemption_slams_to_floor(self):
        a = build_adapter()
        s = HardPreemptionScheduler(CFG)
        decisions = run_ticks(a, s, [60.0] * 10)
        assert decisions
        assert abs(decisions[0].delta) > 1
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 1 and cl.allocated_inference_gpu == 7

    def test_network_aware_gate_blocks_when_saturated(self):
        a = build_adapter(NET_BOUND_CFG)
        s = NetworkAwareElasticScheduler(CFG)
        run_ticks(a, s, [30.0] * 10)  # 不过载，配额不动；闸门在富余场景外应拦截
        cl = a.get_cluster_state()
        assert cl.allocated_training_gpu == 4 and cl.allocated_inference_gpu == 4
        feasible, _ = a.network_expansion_feasible(4, 3, 4, 5)
        assert feasible is False  # 释放带宽 < 新增推理带宽 → 拦截

    def test_network_aware_allows_when_network_abundant(self):
        a = build_adapter(CFG)
        s = NetworkAwareElasticScheduler(CFG)
        decisions = run_ticks(a, s, [60.0] * 10)
        assert decisions, "GPU 瓶颈 + 网络富余 → 正常扩容"
        assert decisions[0].delta == -1
