"""指数平滑(EMA)预测器 —— Experiment E 的核心组件 (spec 新增量)。

问题:elastic 调度是"反应式"(看到 P95 爆了才让渡 GPU),有滞后。
Experiment E 验证:如果对未来负载有**预测**,能不能提前准备、减少违约?
但预测不可能完美 —— 核心研究问题:
    "预测误差多大时,预测辅助调度仍优于纯反应式?"

🔍 决策点 (归来后深挖):
1. **EMA 平滑系数 α** 的含义:α 大 → 紧跟近期变化(噪声大);
   α 小 → 平滑(滞后大)。这是"预测器最关键的参数"。
2. **为什么用 EMA 不用线性回归**:单步预测,EMA 简单、可解释、
   参数就一个 α。真实调度器常用更复杂模型,但验证"预测值是否有用"
   先用最朴素的可控误差模型。
3. **误差注入方式**:在 EMA 预测值上加高斯噪声(受控 std)。
   误差=0 是"完美预测"上界;误差增大看调度退化。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class EMAPredictorConfig:
    alpha: float = 0.3
    horizon_steps: int = 1
    error_std: float = 0.0
    seed: int = 42


class EMAPredictor:
    """指数平滑预测器。可注入受控误差,模拟"预测不完美"。

    API:
        update(observed: float) -> None   喂新观测
        predict() -> float                当前预测值
    """

    def __init__(self, config: EMAPredictorConfig | None = None):
        self.config = config or EMAPredictorConfig()
        self._ema: float | None = None
        self._rng = random.Random(self.config.seed)

    def update(self, observed: float) -> None:
        if self._ema is None:
            self._ema = observed
        else:
            self._ema = (self.config.alpha * observed
                         + (1 - self.config.alpha) * self._ema)

    def predict(self) -> float:
        """返回带误差的预测值(噪声随误差 std 注入)。"""
        if self._ema is None:
            return 0.0
        noise = self._rng.gauss(0.0, self.config.error_std) if self.config.error_std else 0.0
        return max(0.0, self._ema * (1.0 + noise))
