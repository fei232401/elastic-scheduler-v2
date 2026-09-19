"""Elastic Scheduler —— 主方案，逐级让渡（spec §11 + §12，Rule-based）。

规则：
  Rule 1  SLO violation（防抖确认过载）→ Training 让渡 1 GPU（若有配额且受 slowdown 保护）
  Rule 2  接近 SLO（P95 > 80%*SLO）→ 只预警，不改配额
  Rule 3  健康（P95 < 60%*SLO）且 Training 处于 degraded → 归还 1 GPU
  Rule 4  防振荡：改配额后 min_gpu_hold_duration_ticks 内不再动
  Rule 5  每次最多 ±1 GPU（max_scale_step）
  §12     Training 不得低于 min_gpu；超过 max_allowed_slowdown 时拒绝进一步降级
"""
from __future__ import annotations

from src.scheduler.base import ResourceDecision, Scheduler, SchedulerContext


class ElasticScheduler(Scheduler):
    name = "elastic"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        s = cfg.get("scheduler", {})
        self.max_scale_step = s.get("max_scale_step", 1)
        self.min_hold_ticks = s.get("min_gpu_hold_duration_ticks", 24)
        self.approach_ratio = s.get("approach_ratio", 0.8)
        self.healthy_ratio = s.get("healthy_ratio", 0.6)
        self._last_change_tick: int | None = None

    def reset(self) -> None:
        """重置内部状态（多场景复用时调用）。"""
        self._last_change_tick = None

    def step(self, ctx: SchedulerContext) -> ResourceDecision:
        tick = int(ctx.now_s / (self.cfg.get("simulation", {}).get("tick_seconds", 5)))
        p95, slo = ctx.inference_p95_ms, ctx.slo_p95_ms
        would_allow = ctx.extra.get("would_allow_degradation")

        def noop(reason: str) -> ResourceDecision:
            return ResourceDecision(
                old_training_gpu=ctx.training_gpus,
                new_training_gpu=ctx.training_gpus,
                old_inference_gpu=ctx.inference_gpus,
                new_inference_gpu=ctx.inference_gpus,
                reason=reason,
                inference_p95=p95,
            )

        if self._last_change_tick is not None:
            if tick - self._last_change_tick < self.min_hold_ticks:
                return noop("min_gpu_hold: within protection window after last change")

        if ctx.inference_state == "OVERLOADED" and p95 > slo:
            if ctx.training_gpus > ctx.training_min_gpu:
                new_training = ctx.training_gpus - self.max_scale_step
                if new_training >= ctx.training_min_gpu and (
                    would_allow is None or would_allow(new_training)
                ):
                    self._last_change_tick = tick
                    return ResourceDecision(
                        old_training_gpu=ctx.training_gpus,
                        new_training_gpu=new_training,
                        old_inference_gpu=ctx.inference_gpus,
                        new_inference_gpu=ctx.inference_gpus + self.max_scale_step,
                        reason=(
                            "Rule1: SLO violation (overload confirmed) "
                            f"-> reclaim 1 GPU (Training {ctx.training_gpus}->{new_training})"
                        ),
                        inference_p95=p95,
                    )
            return noop(
                "Rule1 blocked: training at min_gpu or degradation would exceed max_allowed_slowdown"
            )

        if p95 > self.approach_ratio * slo:
            return noop(f"Rule2: approaching SLO (P95={p95:.0f}ms > {self.approach_ratio:.0%}*SLO), hold")

        if p95 < self.healthy_ratio * slo and ctx.training_gpus < ctx.training_initial_gpu:
            self._last_change_tick = tick
            return ResourceDecision(
                old_training_gpu=ctx.training_gpus,
                new_training_gpu=ctx.training_gpus + self.max_scale_step,
                old_inference_gpu=ctx.inference_gpus,
                new_inference_gpu=ctx.inference_gpus - self.max_scale_step,
                reason=(
                    "Rule3: inference healthy, returning 1 GPU to Training "
                    f"({ctx.training_gpus}->{ctx.training_gpus + self.max_scale_step})"
                ),
                inference_p95=p95,
            )

        return noop("no rule triggered: steady state")
