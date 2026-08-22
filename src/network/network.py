"""Mock Network —— 训练/推理共享网络带宽的最小模拟（spec §13-§17）。

核心行为：
  - 训练带宽：train_bw(g) = scale * (a·g² + b·g)，g<2 时为 0（spec §14 表：
    2→500, 3→900, 4→1400, 5→2000, 6→2700 Mbps，即 a=50,b=150）
  - 推理带宽：infer_bw(g, qps) = per_gpu_infer_bw·g + bw_per_qps·qps（spec §15）
    —— QPS 外生驱动 + 每 GPU 的边际网络足迹（供调度器预测，spec §17）
  - 拥塞：demand = train_bw + infer_bw；util > congestion_start_util 后延迟开始放大，
    超过 100% 后对 Training 吞吐施加惩罚（网络回压，spec §16）
  - 网络感知判定（spec §17 关键公式）：
      缩 Training 释放带宽  >=  扩 Inference 新增带宽 + 剩余余量  → 允许扩容
      released = train_bw(old_t) - train_bw(new_t)
      delta_infer = per_gpu_infer_bw * (new_i - old_i)
      允许 iff (delta_infer - released) <= headroom
"""
from __future__ import annotations

from src.config import get


class MockNetwork:
    def __init__(self, cfg: dict) -> None:
        n = cfg.get("network", {})
        self.capacity_mbps = n.get("total_bandwidth_mbps", 100000)
        self.base_latency_ms = n.get("base_latency_ms", 2)
        self.latency_gain = n.get("latency_gain_per_overload", 250)
        self.congestion_start_util = n.get("congestion_start_util", 0.5)

        # 训练带宽曲线（spec §14）
        self.train_bw_scale = n.get("training_bw_scale", 1.0)
        self.train_bw_coeff_a = n.get("train_bw_coeff_a", 50)
        self.train_bw_coeff_b = n.get("train_bw_coeff_b", 150)

        # 推理带宽（spec §15）
        self.per_gpu_infer_bw = n.get("per_gpu_infer_bw", 100)
        self.bw_per_qps = n.get("bw_per_qps", 20)

        # 拥塞回压（spec §16）
        self.penalty_per_unit = n.get("training_penalty_per_unit", 0.3)
        self.max_training_penalty = n.get("max_training_penalty", 0.7)

        # 每 tick 状态（由 update() 刷新）
        self.training_gpus: int = 0
        self.inference_gpus: int = 0
        self.qps: float = 0.0
        self.demand_mbps: float = 0.0
        self.utilization: float = 0.0
        self.overload_ratio: float = 0.0
        self.congested: bool = False
        self.latency_ms: float = 0.0
        self.throughput_scale: float = 1.0

    # ---- 带宽函数 ----
    def training_bandwidth(self, gpus: int) -> float:
        """训练占用带宽（spec §14 表）：g=1 无带宽，g>=2 按二次曲线。"""
        if gpus < 2:
            return 0.0
        return self.train_bw_scale * (
            self.train_bw_coeff_a * gpus * gpus + self.train_bw_coeff_b * gpus
        )

    def inference_bandwidth(self, gpus: int, qps: float) -> float:
        """推理占用带宽（spec §15）：QPS 驱动 + 每 GPU 边际足迹。"""
        return self.per_gpu_infer_bw * gpus + self.bw_per_qps * qps

    # ---- 每 tick 刷新 ----
    def update(self, training_gpus: int, inference_gpus: int, qps: float) -> None:
        """按当前配额与流量刷新拥塞状态（引擎每 tick 调用）。"""
        self.training_gpus = training_gpus
        self.inference_gpus = inference_gpus
        self.qps = qps
        self.demand_mbps = self.training_bandwidth(training_gpus) + self.inference_bandwidth(
            inference_gpus, qps
        )
        self.utilization = self.demand_mbps / self.capacity_mbps if self.capacity_mbps > 0 else 1.0
        self.overload_ratio = max(0.0, self.utilization - 1.0)
        self.congested = self.utilization > 1.0

        # 拥塞延迟：利用率超过阈值后线性放大（回加到推理 P95）
        over = max(0.0, self.utilization - self.congestion_start_util)
        self.latency_ms = self.base_latency_ms + self.latency_gain * over

        # 网络回压：超载后训练吞吐打折（拥塞源主要是训练流量，自身受罚）
        penalty = min(self.overload_ratio * self.penalty_per_unit, self.max_training_penalty)
        self.throughput_scale = max(0.0, 1.0 - penalty)

    @property
    def headroom_mbps(self) -> float:
        """当前可用带宽余量（0 = 已饱和）。"""
        return max(0.0, self.capacity_mbps - self.demand_mbps)

    # ---- 网络感知判定（spec §17）----
    def expansion_feasible(
        self, old_training: int, new_training: int, old_inference: int, new_inference: int
    ) -> tuple[bool, dict]:
        """从 (old_t, old_i) 迁到 (new_t, new_i) 是否不把网络推过容量。

        返回 (feasible, 明细)。明细含 released / delta_infer / net / headroom，
        供决策日志解释「为什么拒绝」。
        """
        released = self.training_bandwidth(old_training) - self.training_bandwidth(new_training)
        delta_infer = self.per_gpu_infer_bw * (new_inference - old_inference)
        net = delta_infer - released  # >0 表示净新增带宽需求
        headroom = self.headroom_mbps
        detail = {
            "released_training_bw": round(released, 1),
            "delta_inference_bw": round(delta_infer, 1),
            "net_bandwidth_delta": round(net, 1),
            "headroom_mbps": round(headroom, 1),
        }
        return net <= headroom + 1e-9, detail
