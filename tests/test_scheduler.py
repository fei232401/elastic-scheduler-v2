"""Scheduler 测试（spec §11/§12：Static 恒定、Elastic 规则 + 防振荡 + max_scale_step）。"""
import pytest

from src.scheduler.base import SchedulerContext
from src.scheduler.elastic import ElasticScheduler
from src.scheduler.static import StaticScheduler

CFG = {
    "simulation": {"tick_seconds": 5},
    "scheduler": {
        "max_scale_step": 1,
        "min_gpu_hold_duration_ticks": 24,
        "approach_ratio": 0.8,
        "healthy_ratio": 0.6,
    },
}


def ctx(training=6, inference=2, p95=150.0, slo=300.0, state="RUNNING",
        min_gpu=1, initial_gpu=6, now_s=0.0, allow_degradation=True) -> SchedulerContext:
    return SchedulerContext(
        training_gpus=training,
        inference_gpus=inference,
        training_min_gpu=min_gpu,
        training_initial_gpu=initial_gpu,
        inference_p95_ms=p95,
        slo_p95_ms=slo,
        inference_state=state,
        training_throughput=0.0,
        training_slowdown=1.0,
        training_would_allow_degradation=allow_degradation,
        inference_overload_counter=0,
        inference_recovery_counter=0,
        now_s=now_s,
        extra={"would_allow_degradation": lambda g: allow_degradation},
    )


class TestStatic:
    def test_never_changes(self):
        s = StaticScheduler(CFG)
        for _ in range(10):
            d = s.step(ctx())
            assert d.changed is False
            assert d.new_training_gpu == 6 and d.new_inference_gpu == 2


class TestElasticRule1:
    def test_overload_reclaims_one(self):
        s = ElasticScheduler(CFG)
        d = s.step(ctx(p95=350, slo=300, state="OVERLOADED", now_s=0.0))
        assert d.changed is True
        assert d.new_training_gpu == 5 and d.new_inference_gpu == 3
        assert "Rule1" in d.reason

    def test_max_scale_step_is_one(self):
        """AC-05：一次最多调整 1 GPU。"""
        s = ElasticScheduler(CFG)
        d = s.step(ctx(p95=800, slo=300, state="OVERLOADED", now_s=0.0))
        assert abs(d.delta) == 1

    def test_reclaim_respects_min_gpu(self):
        s = ElasticScheduler(CFG)
        d = s.step(ctx(training=1, inference=7, p95=350, slo=300, state="OVERLOADED", now_s=0.0))
        assert d.changed is False  # training 已在 min，不能再让

    def test_reclaim_blocked_when_slowdown_exceeded(self):
        s = ElasticScheduler(CFG)
        d = s.step(ctx(p95=350, slo=300, state="OVERLOADED", now_s=0.0, allow_degradation=False))
        assert d.changed is False
        assert "slowdown" in d.reason


class TestElasticRule2:
    def test_approaching_holds(self):
        """P95 在 80%~100% SLO 之间 → 只预警不改配额。"""
        s = ElasticScheduler(CFG)
        d = s.step(ctx(p95=250, slo=300, state="RUNNING", now_s=0.0))  # 250 > 240(80%)
        assert d.changed is False
        assert "Rule2" in d.reason


class TestElasticRule3:
    def test_healthy_returns_gpu_when_degraded(self):
        s = ElasticScheduler(CFG)
        # training 5/initial 6（degraded），P95 低 → 归还 1
        d = s.step(ctx(training=5, inference=3, p95=100, slo=300, state="RUNNING",
                       now_s=0.0, initial_gpu=6))
        assert d.changed is True
        assert d.new_training_gpu == 6 and d.new_inference_gpu == 2
        assert "Rule3" in d.reason

    def test_healthy_but_not_degraded_holds(self):
        s = ElasticScheduler(CFG)
        d = s.step(ctx(training=6, inference=2, p95=100, slo=300, state="RUNNING", now_s=0.0))
        assert d.changed is False


class TestElasticRule4:
    def test_min_hold_blocks_reclaim(self):
        """改配额后 24 tick 内（120s/5s）不再动。"""
        s = ElasticScheduler(CFG)
        s.step(ctx(p95=350, slo=300, state="OVERLOADED", now_s=0.0))  # 触发让渡
        # t=60s (12 tick) 仍在保护期
        d = s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=60.0))
        assert d.changed is False
        assert "hold" in d.reason
        # t=150s (30 tick) 已过保护期 → 再次让渡
        d = s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=150.0))
        assert d.changed is True


class TestFullCycle:
    def test_reclaim_then_return_cycle(self):
        """完整让渡→归还周期（AC-04/AC-07）。"""
        s = ElasticScheduler(CFG)
        # 高峰：逐级让渡（每次把让渡后的配额传给下一次）
        training, inference = 6, 2
        t = 0.0
        for _ in range(3):
            d = s.step(ctx(training=training, inference=inference,
                           p95=350, slo=300, state="OVERLOADED", now_s=t))
            assert d.changed is True and d.delta == -1
            training, inference = d.new_training_gpu, d.new_inference_gpu
            t += 150.0  # 跳过保护期
        assert training == 3
        # 回落：逐级归还
        for _ in range(3):
            d = s.step(ctx(training=training, inference=inference,
                           p95=100, slo=300, state="RUNNING", now_s=t))
            assert d.changed is True and d.delta == 1
            training, inference = d.new_training_gpu, d.new_inference_gpu
            t += 150.0
        assert training == 6
