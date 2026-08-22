"""Hard Preemption Scheduler —— 粗糙对比基线（spec §3.2）。

策略（激进，一次性大幅回收）：
  - 高峰确认过载（OVERLOADED）→ Training 一次性让渡到 preempt floor（默认 = min_gpu），
    Inference 一次拿满剩余 GPU。不遵守 max_scale_step / slowdown 保护 —— 这正是它的代价。
  - 流量恢复且推理健康（RUNNING + P95 < healthy_ratio·SLO）→ 一次性归还到 initial。
  - min_hold 保护期内不动（防振荡，规则同 Elastic Rule4）。

与 Elastic 的关键差异：Elastic 一次只让 1 卡、Training 全程不被中断；Hard Preemption
一次砍到地板、Training 被大幅降级/暂停。用同一 workload 对比「粗暴 vs 逐级」。
"""
from __future__ import annotations

from src.scheduler.base import ResourceDecision, Scheduler, SchedulerContext


class HardPreemptionScheduler(Scheduler):
    name = "hard_preemption"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        s = cfg.get("scheduler", {})
        self.healthy_ratio = s.get("healthy_ratio", 0.6)
        self.min_hold_ticks = s.get("min_gpu_hold_duration_ticks", 24)
        # 让渡地板：取配置值与训练 min_gpu 的较大者（配置想压到 0 时 min_gpu 会兜底）
        self.preempt_floor = max(s.get("preempt_min_training", 1), 1)
        self._last_change_tick: int | None = None

    def reset(self) -> None:
        self._last_change_tick = None

    def step(self, ctx: SchedulerContext) -> ResourceDecision:
        tick = int(ctx.now_s / (self.cfg.get("simulation", {}).get("tick_seconds", 5)))
        p95, slo = ctx.inference_p95_ms, ctx.slo_p95_ms
        total = ctx.training_gpus + ctx.inference_gpus

        def noop(reason: str) -> ResourceDecision:
            return ResourceDecision(
                old_training_gpu=ctx.training_gpus,
                new_training_gpu=ctx.training_gpus,
                old_inference_gpu=ctx.inference_gpus,
                new_inference_gpu=ctx.inference_gpus,
                reason=reason,
                inference_p95=p95,
            )

        # min_hold 保护期
        if self._last_change_tick is not None:
            if tick - self._last_change_tick < self.min_hold_ticks:
                return noop("min_gpu_hold: within protection window after last change")

        # 高峰：一次性让渡到地板
        if ctx.inference_state == "OVERLOADED" and p95 > slo:
            floor = max(self.preempt_floor, ctx.training_min_gpu)
            if ctx.training_gpus > floor:
                new_training = floor
                new_inference = total - new_training
                self._last_change_tick = tick
                return ResourceDecision(
                    old_training_gpu=ctx.training_gpus,
                    new_training_gpu=new_training,
                    old_inference_gpu=ctx.inference_gpus,
                    new_inference_gpu=new_inference,
                    reason=(
                        "HardPreempt: overload confirmed -> Training slammed to floor "
                        f"{ctx.training_gpus}->{new_training} (Inference {ctx.inference_gpus}->{new_inference})"
                    ),
                    inference_p95=p95,
                )
            return noop("HardPreempt: training already at floor")

        # 恢复：一次性归还
        if (
            ctx.inference_state == "RUNNING"
            and p95 < self.healthy_ratio * slo
            and ctx.training_gpus < ctx.training_initial_gpu
        ):
            new_training = ctx.training_initial_gpu
            new_inference = total - new_training
            self._last_change_tick = tick
            return ResourceDecision(
                old_training_gpu=ctx.training_gpus,
                new_training_gpu=new_training,
                old_inference_gpu=ctx.inference_gpus,
                new_inference_gpu=new_inference,
                reason=(
                    "HardPreempt: inference healthy, restoring Training to initial "
                    f"{ctx.training_gpus}->{new_training}"
                ),
                inference_p95=p95,
            )

        return noop("no rule triggered: steady state")
