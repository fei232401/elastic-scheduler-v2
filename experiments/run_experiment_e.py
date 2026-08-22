"""Experiment E —— 预测误差敏感性 (spec 新增量)。

核心问题:调度器依赖对未来负载的预测。预测误差多大时,
"预测辅助调度"仍优于"纯反应式(elastic)"?

方法:
1. 生成潮汐负载(确定性,同 seed)。
2. elastic = 纯反应式(基线,项目一已有)。
3. predictive = 反应式 + 预测器提前分配 GPU(EMA 预测负载,误差受控)。
4. sweep 误差 std:0(完美)、0.1、0.3、0.5 → 看 predictive 退化曲线。

🔍 决策点 (归来后深挖):
- 为什么 predictive 要"提前分配"而不是替代 elastic?
  → 真实调度器不会只用预测,而是预测作为**前瞻信号**叠加在反应式上。
- 误差 std 的量纲:相对负载的比例(0.5 = ±50% 误差)。
- SLO 违约率是主指标,训练进度作为代价(降级牺牲训练)。

## 实测结果 (2026-08-22, config/experiment_a_gpu_bound.yaml, seed=42)

| 配置 | SLO违率 | P95(ms) | 切换次数 | 训练进度 | 训练慢化 |
|---|---|---|---|---|---|
| elastic 基线(纯反应式) | 1.4% | 207 | 6 | 46285 | 1.33× |
| predictive err=0.0(完美) | 0.0% | 184 | 38 | 34557 | 2.42× |
| predictive err=0.1 | 0.0% | 180 | 45 | 27508 | 2.89× |
| predictive err=0.3 | 0.0% | 180 | 44 | 25968 | 3.11× |
| predictive err=0.5 | 0.0% | 180 | 44 | 25846 | 3.14× |

**三个诚实发现**:
1. **预测一定赢 SLO**:0% vs 1.4%,P95 180 vs 207ms——提前让渡确实消除了违约。
2. **误差不敏感(出乎意料)**:err=0.0→0.5 结果几乎不变。原因:本实现只"预测预夺"、
   一旦预夺到训练下限就停在稳态,预测噪声(±50%)翻不动稳态决策。
   → 🔍 说明敏感度不在预夺路径,在**恢复路径**(见发现 3)。
3. **代价是训练被牺牲**:进度掉 25%~44%,切换 6→44 次。因为预夺是**单向**的——
   P95 被压到 180ms(远低于 SLO)后,elastic 反应式恢复逻辑几乎不触发,
   GPU 迟迟不回训练。训练慢化 1.33×→3.1×。

**这直接指向 v2**:预测必须是**双向的**(既预夺也预测恢复),或引入**成本目标函数**
(权衡 SLO 违约的代价 vs 训练进度的代价)。当前 v1 只是最小验证,证明
"预测有用但副作用明显",为归来后的 v2 优化留好了问题。

用法:
    python -m experiments.run_experiment_e
    python -m experiments.run_experiment_e --error 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.simulator.engine import SCHEDULERS, run_experiment

from .predictor import EMAPredictor, EMAPredictorConfig


def _baseline_summary(scheduler: str, cfg: dict) -> dict:
    """跑基线 scheduler,返回 summary 字段。"""
    collector = run_experiment(scheduler, cfg)
    s = collector.summarize()
    return {
        "slo_violation_ratio": s.slo_violation_ratio,
        "inference_avg_p95": s.inference_avg_p95,
        "resource_switch_count": s.resource_switch_count,
        "training_final_progress": s.training_final_progress,
        "training_avg_slowdown": s.training_avg_slowdown,
    }


def make_predictive_scheduler(error_std: float, alpha: float = 0.3):
    """构造"预测辅助"调度器类(elastic + 预测前瞻)。

    简化实现:预测负载超过阈值 → 直接让渡 1 个 GPU(提前反应)。
    这不是完整调度器,是 Experiment E 的**最小验证**——
    证明"预测误差敏感性"这个研究问题存在且可量化。
    """
    from src.scheduler.elastic import ElasticScheduler
    from src.scheduler.base import ResourceDecision

    class PredictiveScheduler(ElasticScheduler):
        """elastic + 预测前瞻。"""

        name = "predictive"

        def __init__(self, cfg: dict):
            super().__init__(cfg)
            s = cfg.get("scheduler", {})
            self.predictor = EMAPredictor(EMAPredictorConfig(
                alpha=s.get("ema_alpha", alpha),
                error_std=s.get("pred_error_std", error_std),
            ))
            self._last_pred_switch = None

        def step(self, ctx):
            # 先喂观测:P95 反推负载压力
            load_proxy = ctx.inference_p95_ms / max(ctx.slo_p95_ms, 1.0)
            self.predictor.update(load_proxy)
            predicted = self.predictor.predict()

            # 预测压力 > 阈值 → 提前让渡 1 GPU(带保护窗口)
            s = self.cfg.get("scheduler", {})
            threshold = s.get("predict_preempt_ratio", 0.6)
            hold = s.get("min_gpu_hold_duration_ticks", 24)
            tick = int(ctx.now_s / self.cfg.get("simulation", {}).get("tick_seconds", 5))
            if (predicted > threshold and ctx.training_gpus > ctx.training_min_gpu):
                if self._last_pred_switch is None or tick - self._last_pred_switch >= hold:
                    self._last_pred_switch = tick
                    return ResourceDecision(
                        old_training_gpu=ctx.training_gpus,
                        new_training_gpu=ctx.training_gpus - 1,
                        old_inference_gpu=ctx.inference_gpus,
                        new_inference_gpu=ctx.inference_gpus + 1,
                        reason="predict_preempt",
                        inference_p95=ctx.inference_p95_ms,
                    )
            # 否则走 elastic 反应式
            return super().step(ctx)

    return PredictiveScheduler


def run_with_predictor(cfg: dict, error_std: float, alpha: float = 0.3) -> dict:
    """跑"预测辅助"调度:注册预测调度器并复用 engine.run_experiment。"""
    from src.simulator import engine as engine_mod

    cls = make_predictive_scheduler(error_std, alpha)
    engine_mod.SCHEDULERS["predictive"] = cls  # 临时注册
    try:
        collector = engine_mod.run_experiment("predictive", cfg)
    finally:
        engine_mod.SCHEDULERS.pop("predictive", None)  # 清理,不留脏状态

    s = collector.summarize()
    return {
        "slo_violation_ratio": s.slo_violation_ratio,
        "inference_avg_p95": s.inference_avg_p95,
        "resource_switch_count": s.resource_switch_count,
        "training_final_progress": s.training_final_progress,
        "training_avg_slowdown": s.training_avg_slowdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment E: 预测误差敏感性")
    parser.add_argument("--error", type=float, default=None)
    parser.add_argument("--config", default="config/experiment_a_gpu_bound.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    baseline = _baseline_summary("elastic", cfg)
    errors = [args.error] if args.error is not None else [0.0, 0.1, 0.3, 0.5]

    rows = {"elastic_baseline": baseline, "predictive": {}}
    for e in errors:
        rows["predictive"][f"err={e}"] = run_with_predictor(cfg, e)

    out = pathlib.Path("results/experiment_e.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"结果: {out}")
    print(f"{'配置':<14}{'SLO违率':>10}{'P95(ms)':>10}{'切换':>6}{'训练进度':>10}")
    b = rows["elastic_baseline"]
    print(f"{'elastic基线':<14}{b['slo_violation_ratio']:>10.1%}{b['inference_avg_p95']:>10.0f}"
          f"{b['resource_switch_count']:>6}{b['training_final_progress']:>10.0f}")
    for k, v in rows["predictive"].items():
        print(f"{k:<14}{v['slo_violation_ratio']:>10.1%}{v['inference_avg_p95']:>10.0f}"
              f"{v['resource_switch_count']:>6}{v['training_final_progress']:>10.0f}")


if __name__ == "__main__":
    main()
