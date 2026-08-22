"""Network-aware Elastic Scheduler —— 在 Elastic 规则上叠加网络闸门（spec §17）。

GPU-only Elastic 只关心「推理要不要更多卡」，不看网络。本调度器在每次改动配额前，
用 spec §17 的关键公式校验网络是否承受得住：

  释放带宽 = 旧训练带宽 - 新训练带宽（缩 Training 释放的带宽）
  新增带宽 = per_gpu_infer_bw * (新推理卡数 - 旧推理卡数)
  允许  iff  (新增带宽 - 释放带宽) <= 剩余余量(headroom)

双向都校验：
  - 让渡给推理（Rule1）：新推理卡要多占带宽，必须由缩训练释放的带宽覆盖；
  - 归还给训练（Rule3）：训练加卡会显著增加训练带宽，同样不能把网络推过容量。

被网络闸门拒绝时，保持配额不动并记录原因（含带宽明细），供实验对比解释。

Phase 3.1（Runtime Abstraction）：不再直接持有 MockNetwork，改经 `ctx.runtime`
（RuntimeAdapter）读取网络状态与扩容闸门 —— Scheduler 只依赖抽象接口。
"""
from __future__ import annotations

from src.scheduler.base import ResourceDecision, SchedulerContext
from src.scheduler.elastic import ElasticScheduler


class NetworkAwareElasticScheduler(ElasticScheduler):
    name = "network_aware"

    def _gated_noop(
        self, ctx: SchedulerContext, base_reason: str, net_detail: dict, utilization: float
    ) -> ResourceDecision:
        return ResourceDecision(
            old_training_gpu=ctx.training_gpus,
            new_training_gpu=ctx.training_gpus,
            old_inference_gpu=ctx.inference_gpus,
            new_inference_gpu=ctx.inference_gpus,
            reason=(
                f"{base_reason} | Network-aware BLOCKED: released "
                f"{net_detail['released_training_bw']}Mbps vs infer +{net_detail['delta_inference_bw']}Mbps"
                f" (headroom {net_detail['headroom_mbps']}Mbps)"
            ),
            inference_p95=ctx.inference_p95_ms,
            network_utilization=utilization,
        )

    def step(self, ctx: SchedulerContext) -> ResourceDecision:
        last_change = self._last_change_tick
        decision = super().step(ctx)

        runtime = ctx.runtime
        if runtime is None:
            # 无 runtime 上下文（单测纯 ctx 场景）→ 退化为 GPU-only Elastic
            return decision

        net = runtime.get_network_state()

        # 记录本 tick 网络利用率到决策（metrics 可回溯）
        decision.network_utilization = net.utilization

        if not decision.changed:
            return decision

        # 网络闸门（spec §17）—— 双向校验（经 RuntimeAdapter，不碰 MockNetwork）
        feasible, detail = runtime.network_expansion_feasible(
            decision.old_training_gpu,
            decision.new_training_gpu,
            decision.old_inference_gpu,
            decision.new_inference_gpu,
        )
        if feasible:
            return decision

        # 被拒：回滚 _last_change_tick，视为「未变更」，下一 tick 网络有余量立即重试
        self._last_change_tick = last_change
        if decision.delta < 0:  # 让渡给推理
            return self._gated_noop(ctx, "Rule1: SLO violation -> reclaim 1 GPU", detail, net.utilization)
        return self._gated_noop(
            ctx, "Rule3: inference healthy -> return 1 GPU to Training", detail, net.utilization
        )
