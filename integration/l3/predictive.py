r"""L3 版"预测辅助"调度器 —— 复用 Experiment E 的 EMA 预夺逻辑,只把步长从 1 改成 max_scale_step。

E 的 PredictiveScheduler(experiments/run_experiment_e.py)预夺时硬编码 `training_gpus - 1`;
L3 的 max_scale_step=500,若还 -1 等于没动。本类对齐:E 的预测→预夺语义 + L3 的步长语义。
面试口径:决策逻辑 100% 来自项目一(EMA 预测器 + ElasticScheduler Rules 1-5),本文件只改一个数字。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import get
from src.scheduler.base import ResourceDecision
from src.scheduler.elastic import ElasticScheduler
from experiments.predictor import EMAPredictor, EMAPredictorConfig


class L3PredictiveScheduler(ElasticScheduler):
    """elastic + 预测前瞻(L3 步长版)。预夺用 max_scale_step,其余 = E 的原逻辑。"""

    name = "predictive"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        s = cfg.get("scheduler", {})
        self.predictor = EMAPredictor(EMAPredictorConfig(
            alpha=s.get("ema_alpha", 0.3),
            error_std=s.get("pred_error_std", 0.0),
        ))
        self._last_pred_switch = None

    def step(self, ctx):
        load_proxy = ctx.inference_p95_ms / max(ctx.slo_p95_ms, 1.0)
        self.predictor.update(load_proxy)
        predicted = self.predictor.predict()

        s = self.cfg.get("scheduler", {})
        threshold = s.get("predict_preempt_ratio", 0.6)
        hold = s.get("min_gpu_hold_duration_ticks", 24)
        step = int(self.max_scale_step)
        tick = int(ctx.now_s / get(self.cfg, "simulation.tick_seconds", 5))

        if (predicted > threshold and ctx.training_gpus > ctx.training_min_gpu):
            if self._last_pred_switch is None or tick - self._last_pred_switch >= hold:
                self._last_pred_switch = tick
                new_train = max(ctx.training_min_gpu, ctx.training_gpus - step)
                return ResourceDecision(
                    old_training_gpu=ctx.training_gpus,
                    new_training_gpu=new_train,
                    old_inference_gpu=ctx.inference_gpus,
                    new_inference_gpu=ctx.inference_gpus + (ctx.training_gpus - new_train),
                    reason="predict_preempt",
                    inference_p95=ctx.inference_p95_ms,
                )
        return super().step(ctx)
