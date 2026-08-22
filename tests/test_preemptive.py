"""Hard Preemption Scheduler 测试（spec §3.2：一次性大幅回收 / 恢复）。"""
from src.scheduler.base import SchedulerContext
from src.scheduler.preemptive import HardPreemptionScheduler

CFG = {
    "simulation": {"tick_seconds": 5},
    "scheduler": {
        "min_gpu_hold_duration_ticks": 24,
        "healthy_ratio": 0.6,
        "preempt_min_training": 1,
    },
}


def ctx(training=6, inference=2, p95=150.0, slo=300.0, state="RUNNING",
        min_gpu=1, initial_gpu=6, now_s=0.0) -> SchedulerContext:
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
        training_would_allow_degradation=True,
        inference_overload_counter=0,
        inference_recovery_counter=0,
        now_s=now_s,
        extra={},
    )


class TestPreempt:
    def test_overload_slams_to_floor(self):
        """过载确认 → 一次性让渡到地板（6→1，推理 2→7）。"""
        s = HardPreemptionScheduler(CFG)
        d = s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=0.0))
        assert d.changed is True
        assert d.new_training_gpu == 1 and d.new_inference_gpu == 7
        assert "HardPreempt" in d.reason

    def test_min_hold_blocks(self):
        """大幅变动后保护期内不再动。"""
        s = HardPreemptionScheduler(CFG)
        s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=0.0))
        # t=60s 仍在保护期 → 即使仍过载也不动
        d = s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=60.0))
        assert d.changed is False
        # t=150s 过保护期，训练已在地板 → 无动作
        d = s.step(ctx(training=1, inference=7, p95=400, slo=300,
                       state="OVERLOADED", now_s=150.0))
        assert d.changed is False

    def test_restore_to_initial_when_healthy(self):
        """恢复健康 → 一次性归还到 initial。"""
        s = HardPreemptionScheduler(CFG)
        s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=0.0))  # 让渡到 1
        d = s.step(ctx(training=1, inference=7, p95=100, slo=300,
                       state="RUNNING", now_s=150.0))
        assert d.changed is True
        assert d.new_training_gpu == 6 and d.new_inference_gpu == 2
        assert "restor" in d.reason.lower()

    def test_noop_when_not_overloaded_and_at_initial(self):
        s = HardPreemptionScheduler(CFG)
        d = s.step(ctx(p95=100, slo=300, state="RUNNING", now_s=0.0))
        assert d.changed is False

    def test_ignores_max_scale_step(self):
        """Hard Preemption 不遵守 max_scale_step=1（粗粒度就是它的设计）。"""
        s = HardPreemptionScheduler(CFG)
        d = s.step(ctx(p95=400, slo=300, state="OVERLOADED", now_s=0.0))
        assert abs(d.delta) > 1
