"""真实 PyTorch 训练 workload 契约测试（Phase 3.2 Step 3，spec §6-7）。

测试在 CPU + 小模型上运行（快、确定），验证：
  - 真实 forward/backward/step → progress 累积、throughput>0
  - 上报吞吐 = 真实测量 × EMULATED 多卡缩放（诚实标注）
  - 状态机（RUNNING/DEGRADED/PAUSED/COMPLETED）、min_gpu 保护
  - 与 LocalRuntimeAdapter 需要的 duck-type 契约一致
"""
from __future__ import annotations

from src.workload.training_real import RealTrainingJob

CFG = {
    "training": {
        "initial_gpu": 6, "min_gpu": 1, "total_work": 100000,
        "max_allowed_slowdown": 3.0, "scaling_exponent": 0.85,
    },
    "local": {
        "device": "cpu",
        "training": {
            "input_dim": 16, "hidden_dim": 32, "layers": 2, "output_dim": 5,
            "batch_size": 16, "total_work": 500000, "scaling_exponent": 0.85,
        },
    },
}

TINY_CFG = {
    **CFG,
    "local": {**CFG["local"], "training": {**CFG["local"]["training"], "total_work": 1000}},
}


def build(cfg: dict = None) -> RealTrainingJob:
    return RealTrainingJob(cfg or CFG)


class TestRealForwardBackwardStep:
    def test_tick_runs_real_compute_and_accumulates_progress(self):
        job = build()
        job.start()
        n = job.tick(0.5)
        assert n > 0, "真实训练应产出样本"
        assert job.progress > 0.0 and job.progress >= n
        assert job._last_measured_sps > 0.0
        assert job.state == "RUNNING"

    def test_throughput_is_real_measurement_times_emulated_scaling(self):
        """初始配额=6（与 Mock 一致）：吞吐 = 真实测量 × EMULATED 缩放；1 GPU = REAL。"""
        job = build()
        job.start()
        job.tick(0.5)
        real = job._last_measured_sps
        # @initial 6 GPU：EMULATED 缩放
        assert abs(job.throughput - real * (6**0.85)) < real * 0.2
        # @1 GPU = REAL 测量本身
        job.set_gpus(1)
        assert abs(job.throughput - real) < 1e-6
        # @2 GPU：EMULATED ×2^0.85
        job.set_gpus(2)
        assert abs(job.throughput - real * (2**0.85)) < real * 0.2


class TestStateMachine:
    def test_degrades_when_below_initial(self):
        job = build()
        job.set_gpus(2)  # < initial 6
        job.tick(0.5)
        assert job.state == "DEGRADED"

    def test_paused_when_zero_gpus(self):
        job = build()
        job.set_gpus(0)
        n = job.tick(0.5)
        assert job.state == "PAUSED" and n == 0.0

    def test_completes_when_work_done(self):
        job = build(TINY_CFG)  # total_work=1000，一个 tick 即可完成
        job.start()
        job.tick(2.0)
        assert job.state == "COMPLETED"
        assert job.progress >= job.total_work

    def test_slowdown_vs_initial(self):
        job = build()
        assert job.slowdown_vs_initial == 1.0  # 初始配额
        job.set_gpus(2)
        assert job.slowdown_vs_initial > 1.0

    def test_would_allow_degradation_respects_min_gpu(self):
        """min_gpu=1 且 max_allowed_slowdown=3.0：slowdown(1)=4.58>3 → 拒绝。"""
        job = build()
        assert job.would_allow_degradation(3) is True  # slowdown(3)=(6/3)^0.85≈1.80 ≤3
        assert job.would_allow_degradation(0) is False  # < min_gpu
        assert job.would_allow_degradation(1) is False  # slowdown(1)=6^0.85≈4.58 >3
        assert job.would_allow_degradation(2) is True  # slowdown(2)=3^0.85≈2.54 ≤3


class TestAdapterContract:
    def test_duck_type_fields_exist(self):
        """LocalRuntimeAdapter 依赖的字段/方法齐全。"""
        job = build()
        for attr in ("initial_gpu", "min_gpu", "state", "progress", "remaining_work",
                     "throughput", "estimated_completion_time_s", "gpu_utilization",
                     "gpu_memory_mb", "slowdown_vs_initial"):
            assert hasattr(job, attr), f"missing {attr}"
        for meth in ("set_gpus", "would_allow_degradation", "tick", "start", "stop"):
            assert callable(getattr(job, meth, None)), f"missing {meth}"

    def test_gpu_metrics_reported_zero_on_cpu(self):
        job = build()
        job.tick(0.3)
        assert job.gpu_utilization == 0.0
        assert job.gpu_memory_mb == 0.0
