"""Static Scheduler —— 固定分配，不调整（spec §3.1，Baseline）。

整个实验过程配额不变：Training=initial，Inference=total-initial。
"""
from __future__ import annotations

from src.scheduler.base import ResourceDecision, Scheduler, SchedulerContext


class StaticScheduler(Scheduler):
    name = "static"

    def step(self, ctx: SchedulerContext) -> ResourceDecision:
        return ResourceDecision(
            old_training_gpu=ctx.training_gpus,
            new_training_gpu=ctx.training_gpus,
            old_inference_gpu=ctx.inference_gpus,
            new_inference_gpu=ctx.inference_gpus,
            reason="static baseline: fixed allocation",
            inference_p95=ctx.inference_p95_ms,
        )
