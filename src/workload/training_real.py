"""真实 PyTorch 训练 workload（Phase 3.2 Step 3，spec §6-7，Gate 1）。

真实执行 forward → backward → optimizer.step()，产出真实 samples/sec。
模型：小型 MLP（合成随机数据，无需下载），CPU/GPU 均可运行。

GPU 模型（诚实标注，spec §22）：
  - 本机物理 GPU = 1（RTX 4070 8GB）。真实计算始终跑在 1 张物理 GPU 上；
  - 调度器看到的 `allocated_gpu`（虚拟配额）>1 时，上报吞吐 = 真实单卡测量 T1 ×
    缩放系数(allocated_gpu)。缩放曲线默认用幂律 n^scaling_exponent（与 Mock 同构，
    Step 9 用真实测量校准替换）。
  - 「REAL」= 单卡测量 / GPU util / mem / progress 累积；「EMULATED」= 多卡缩放。
    绝不冒充真实多卡数据。
"""
from __future__ import annotations

import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get


class RealTrainingJob:
    """真实训练任务：被 LocalRuntimeAdapter 经 duck-type 契约驱动。"""

    def __init__(self, cfg: dict) -> None:
        local = get(cfg, "local", {})
        lt = get(local, "training", {})
        t = get(cfg, "training", {})

        self.initial_gpu = int(get(t, "initial_gpu", 6))
        self.min_gpu = int(get(t, "min_gpu", 1))
        self.total_work = float(get(lt, "total_work", get(t, "total_work", 100000)))
        self.max_allowed_slowdown = float(get(t, "max_allowed_slowdown", 3.0))
        self.scaling_exponent = float(get(lt, "scaling_exponent", get(t, "scaling_exponent", 0.85)))
        self.calibration_curve: dict[int, float] = {
            int(k): float(v) for k, v in get(lt, "calibration_curve", {}).items()
        }

        self.input_dim = int(get(lt, "input_dim", 256))
        self.hidden_dim = int(get(lt, "hidden_dim", 1024))
        self.layers = int(get(lt, "layers", 4))
        self.output_dim = int(get(lt, "output_dim", 10))
        self.batch_size = int(get(lt, "batch_size", 256))
        self.max_steps_per_tick = int(get(lt, "max_steps_per_tick", 100_000))

        dev = get(local, "device", "auto")
        if dev == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev)
        self._is_gpu = self.device.type == "cuda"

        self.allocated_gpu: int = self.initial_gpu
        self.progress: float = 0.0
        self.state: str = "RUNNING"
        self._last_measured_sps: float = 0.0
        self._gpu_util: float = 0.0
        self._gpu_mem_mb: float = 0.0

        self._build_model()

    def _build_model(self) -> None:
        torch.manual_seed(42)
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(max(0, self.layers - 1)):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def scaling_at(self, gpus: int) -> float:
        """EMULATED 多卡缩放系数。校准曲线（Step 9）优先，否则幂律默认。"""
        if gpus <= 0:
            return 0.0
        if self.calibration_curve:
            return float(self.calibration_curve.get(int(gpus), gpus**self.scaling_exponent))
        return float(gpus**self.scaling_exponent)

    @property
    def throughput(self) -> float:
        """当前配额下的样本/秒。REAL T1 × EMULATED scale(allocated_gpu)。"""
        return self._last_measured_sps * self.scaling_at(self.allocated_gpu)

    @property
    def remaining_work(self) -> float:
        return max(0.0, self.total_work - self.progress)

    @property
    def estimated_completion_time_s(self) -> float:
        tp = self.throughput
        return self.remaining_work / tp if tp > 0 else float("inf")

    @property
    def slowdown_vs_initial(self) -> float:
        """相对 initial_gpu 的完成时间放大（EMULATED，来自缩放曲线比值）。"""
        base = self.scaling_at(self.initial_gpu)
        cur = self.scaling_at(self.allocated_gpu)
        if cur <= 0:
            return float("inf")
        return base / cur

    def would_allow_degradation(self, new_gpus: int) -> bool:
        if new_gpus < self.min_gpu:
            return False
        return self.slowdown_at(new_gpus) <= self.max_allowed_slowdown

    def slowdown_at(self, gpus: int) -> float:
        base = self.scaling_at(self.initial_gpu)
        cur = self.scaling_at(gpus)
        if cur <= 0:
            return float("inf")
        return base / cur

    def _read_gpu_util_pct(self) -> float:
        """读取当前 GPU 利用率（0~100）。优先 torch，回退 nvidia-smi。"""
        try:
            return float(torch.cuda.utilization())
        except Exception:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip()
                return float(out.splitlines()[0])
            except Exception:
                return 0.0

    def _sample_gpu_metrics(self, util_samples: list[float] | None = None) -> None:
        if not self._is_gpu:
            self._gpu_util = 0.0
            self._gpu_mem_mb = 0.0
            return
        try:
            self._gpu_mem_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        except Exception:
            self._gpu_mem_mb = 0.0
        if util_samples:
            self._gpu_util = sum(util_samples) / len(util_samples)
        else:
            self._gpu_util = self._read_gpu_util_pct()

    @property
    def gpu_utilization(self) -> float:
        return self._gpu_util / 100.0

    @property
    def gpu_memory_mb(self) -> float:
        return self._gpu_mem_mb

    def set_gpus(self, gpus: int) -> None:
        self.allocated_gpu = max(0, int(gpus))

    def start(self) -> None:
        self.model.train()

    def stop(self) -> None:
        self.optimizer = None

    def tick(self, seconds: float) -> float:
        """真实训练 `seconds` 秒，返回本 tick 处理的样本数（REAL）。"""
        if self.state in ("COMPLETED", "FAILED"):
            return 0.0
        if self.allocated_gpu <= 0:
            self.state = "PAUSED"
            return 0.0
        self.state = "DEGRADED" if self.allocated_gpu < self.initial_gpu else "RUNNING"

        self.model.train()
        samples = 0
        steps = 0
        start = time.perf_counter()
        deadline = start + max(0.1, seconds)
        util_samples: list[float] = []
        last_sample = start
        while time.perf_counter() < deadline and steps < self.max_steps_per_tick:
            x = torch.randn(self.batch_size, self.input_dim, device=self.device)
            y = torch.randint(0, self.output_dim, (self.batch_size,), device=self.device)
            loss = F.cross_entropy(self.model(x), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            samples += self.batch_size
            steps += 1
            if self._is_gpu and time.perf_counter() - last_sample >= 0.5:
                util_samples.append(self._read_gpu_util_pct())
                last_sample = time.perf_counter()

        elapsed = time.perf_counter() - start
        self._last_measured_sps = samples / elapsed if elapsed > 0 else 0.0
        self.progress += samples
        if util_samples:
            util_samples.append(self._read_gpu_util_pct())
        self._sample_gpu_metrics(util_samples)
        if self.progress >= self.total_work:
            self.state = "COMPLETED"
        return float(samples)
