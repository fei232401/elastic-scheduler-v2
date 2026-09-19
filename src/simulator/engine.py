"""仿真引擎 —— 时钟驱动，串联所有模块（spec §10/Step 9.5 确定性种子）。

Phase 3.1（Runtime Abstraction）：引擎不再直接操作 Mock 组件，统一经
RuntimeAdapter 抽象驱动（Scheduler 同样只经 runtime 接口）；`build_runtime(mode)`
按配置分派 MockRuntimeAdapter 或 LocalRuntimeAdapter。

每 tick 流程：
  1. runtime.advance(qps)                    # 网络刷新 + 推理（网络先于推理）
  2. 从 runtime 状态视图组装 SchedulerContext
  3. scheduler.step(ctx) → 决策
  4. runtime.apply_resource_decision(决策)   # 配额流转 + workload 同步
  5. runtime.advance_training()              # 训练推进（含网络回压）
  6. collector 采集指标 + 记录决策
"""
from __future__ import annotations

import pathlib
import random

import numpy as np

from src.cluster.resource_manager import ResourceManager
from src.config import get, load_config
from src.metrics.collector import MetricsCollector, TickMetrics
from src.metrics.exporter import export_all
from src.network.network import MockNetwork
from src.runtime.mock import MockRuntimeAdapter
from src.scheduler.base import Scheduler, SchedulerContext
from src.scheduler.elastic import ElasticScheduler
from src.scheduler.network_aware import NetworkAwareElasticScheduler
from src.scheduler.preemptive import HardPreemptionScheduler
from src.scheduler.static import StaticScheduler
from src.simulator.clock import Clock
from src.workload.inference import MockInferenceService
from src.workload.training import MockTrainingJob
from src.workload.traffic import TrafficGenerator

SCHEDULERS: dict[str, type[Scheduler]] = {
    "static": StaticScheduler,
    "elastic": ElasticScheduler,
    "hard_preemption": HardPreemptionScheduler,
    "network_aware": NetworkAwareElasticScheduler,
}


def build_scheduler(name: str, cfg: dict) -> Scheduler:
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler: {name} (available: {sorted(SCHEDULERS)})")
    return SCHEDULERS[name](cfg)


def build_runtime(mode: str, cfg: dict, rng, seed: int = 42):
    """按 mode 构建 RuntimeAdapter（统一 Runtime Contract，spec §3.2）。

    mock:  纯数学模型，毫秒级推进（Phase 1-2 基线）。
    local: 真实 PyTorch 训练 + 真实 HTTP 推理 + NetemNetwork（EMULATED 带宽核算 +
           真实 tc/netem 整形层），跑真实 wall-clock tick（spec §22）。
    """
    sim = cfg.get("simulation", {})
    total_gpus = sim.get("total_gpus", 8)
    tick_seconds = sim.get("tick_seconds", 5)
    rm = ResourceManager(
        total_gpus=total_gpus,
        initial_training=get(cfg, "training.initial_gpu", 6),
        min_training=get(cfg, "training.min_gpu", 1),
    )
    if mode == "mock":
        return MockRuntimeAdapter(
            rm,
            MockTrainingJob(cfg),
            MockInferenceService(cfg),
            MockNetwork(cfg),
        )
    if mode == "local":
        from src.network.netem import NetemNetwork
        from src.runtime.local import LocalRuntimeAdapter
        from src.workload.inference_real import HTTPInferenceService
        from src.workload.training_real import RealTrainingJob

        return LocalRuntimeAdapter(
            rm=rm,
            training=RealTrainingJob(cfg),
            inference=HTTPInferenceService(cfg),
            network=NetemNetwork(cfg),
            tick_seconds=tick_seconds,
        )
    raise ValueError(f"unknown runtime mode: {mode} (available: mock, local)")


def run_experiment(
    scheduler_name: str,
    cfg: dict,
    results_dir: str | pathlib.Path = "results",
    seed: int | None = None,
    mode: str = "mock",
) -> MetricsCollector:
    """运行一次完整潮汐实验，返回填好数据的 collector。

    Args:
        scheduler_name: static | elastic | hard_preemption | network_aware
        cfg: 已加载的配置 dict
        results_dir: 输出目录（相对或绝对）
        seed: 覆盖配置中的随机种子（确定性，spec Step 9.5）
        mode: mock（数学模型，毫秒级）| local（真实 workload，wall-clock）
    """
    sim = cfg.get("simulation", {})
    seed = seed if seed is not None else sim.get("seed", 42)
    rng = random.Random(seed)
    np.random.seed(seed)
    random.seed(seed)

    total_gpus = sim.get("total_gpus", 8)
    tick_seconds = sim.get("tick_seconds", 5)
    duration_ticks = sim.get("duration_ticks", 840)

    clock = Clock(duration_ticks, tick_seconds)
    runtime = build_runtime(mode, cfg, rng, seed)
    scheduler = build_scheduler(scheduler_name, cfg)
    if hasattr(scheduler, "reset"):
        scheduler.reset()
    collector = MetricsCollector(scheduler_name, results_dir)

    traffic = TrafficGenerator(cfg, rng=rng)
    start = getattr(runtime, "start", None)
    if start is not None:
        start()

    while not clock.done:
        qps = traffic.qps_at_second(clock.now_s)

        runtime.advance(qps)

        cl = runtime.get_cluster_state()
        tr = runtime.get_training_state()
        inf = runtime.get_inference_state()

        ctx = SchedulerContext(
            training_gpus=cl.allocated_training_gpu,
            inference_gpus=cl.allocated_inference_gpu,
            training_min_gpu=runtime.training_min_gpu,
            training_initial_gpu=runtime.training_initial_gpu,
            inference_p95_ms=inf.p95,
            slo_p95_ms=inf.slo,
            inference_state=inf.status,
            training_throughput=tr.throughput,
            training_slowdown=runtime.training_slowdown,
            training_would_allow_degradation=runtime.would_allow_degradation(
                cl.allocated_training_gpu - 1
            ),
            inference_overload_counter=runtime.inference_overload_counter,
            inference_recovery_counter=runtime.inference_recovery_counter,
            now_s=clock.now_s,
            runtime=runtime,
            extra={"would_allow_degradation": runtime.would_allow_degradation},
        )
        decision = scheduler.step(ctx)

        if decision.changed:
            decision.timestamp = clock.now_s
            decision.scheduler = scheduler_name
            decision.training_state = tr.status
            runtime.apply_resource_decision(decision)

        runtime.advance_training()

        cl2 = runtime.get_cluster_state()
        tr2 = runtime.get_training_state()
        inf2 = runtime.get_inference_state()
        net2 = runtime.get_network_state()

        gpu_util = (cl2.allocated_training_gpu + cl2.allocated_inference_gpu) / total_gpus
        collector.record(TickMetrics(
            tick=clock.tick,
            time_s=clock.now_s,
            qps=qps,
            inference_capacity=inf2.capacity_qps,
            inference_p50=inf2.p50,
            inference_p95=inf2.p95,
            inference_p99=inf2.p99,
            inference_queue=inf2.queue_length,
            inference_state=inf2.status,
            slo_violated=inf2.slo_violation,
            training_gpu=cl2.allocated_training_gpu,
            inference_gpu=cl2.allocated_inference_gpu,
            training_throughput=tr2.throughput,
            training_progress=tr2.progress,
            training_remaining=tr2.remaining_work,
            training_est_completion=tr2.estimated_completion_time,
            training_slowdown=runtime.training_slowdown,
            training_state=tr2.status,
            gpu_utilization=gpu_util,
            unused_gpu=cl2.free_gpu,
            network_demand=net2.demand_mbps,
            network_utilization=net2.utilization,
            network_latency_ms=net2.congestion_latency_ms,
            network_congested=net2.congested,
            network_headroom=net2.headroom_mbps,
        ))
        if decision.changed:
            collector.record_decision({**decision.to_dict(), "time_s": clock.now_s})

        clock.advance()

    stop = getattr(runtime, "stop", None)
    if stop is not None:
        stop()

    collector.write_csv()
    collector.write_decisions()
    collector.write_summary()
    return collector
