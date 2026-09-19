"""真实 HTTP 推理 workload 契约测试（Phase 3.2 Step 5，spec §8-9）。

CPU + 小模型上运行（快、可复现）。验证：
  - 真实 HTTP server 起停、真实负载下产出真实延迟/吞吐测量
  - 容量模型（EMULATED：capacity = capacity_per_gpu × allocated_gpu）
  - 上报给调度器的延迟 = 真实曲线校准的利用率映射 + 网络拥塞延迟
  - 防抖状态机（OVERLOADED/RUNNING）、set_gpus 影响容量
  - real_stats 单独保留真实测量（Reality Gap 用）
"""
from __future__ import annotations

from src.workload.inference_real import HTTPInferenceService

CFG = {
    "simulation": {"total_gpus": 8},
    "training": {"initial_gpu": 6},
    "inference": {
        "slo_p95_ms": 6, "overload_trigger_ticks": 2, "recovery_trigger_ticks": 2,
    },
    "local": {
        "device": "cpu",
        "inference": {
            "port": 8711, "input_dim": 16, "hidden_dim": 32, "layers": 2,
            "output_dim": 5, "generator_workers": 4, "max_batch_size": 8,
            "flush_interval_ms": 1.0,
            "capacity_per_gpu": 100.0,
            "base_p95_ms": 2.0, "overload_p95_ms": 10.0, "saturation_qps": 1000.0,
            "slo_p95_ms": 6.0,
        },
    },
}


def build() -> HTTPInferenceService:
    return HTTPInferenceService(CFG)


class TestServerLifecycle:
    def test_start_stop_and_serve_produces_real_measurements(self):
        svc = build()
        svc.start()
        svc.serve(50.0, 0.6)
        r = svc.real_stats
        assert svc._server is not None
        assert r["throughput_qps"] > 0.0, "真实负载应完成请求"
        assert r["p95_ms"] > 0.0
        svc.stop()
        assert svc._server is None

    def test_start_is_idempotent(self):
        svc = build()
        svc.start()
        svc.start()
        svc.stop()
        svc.stop()


class TestCapacityModel:
    def test_capacity_is_per_gpu_times_allocated_gpus(self):
        svc = build()
        assert svc.allocated_gpu == 2
        assert svc.capacity_qps == 200.0

    def test_set_gpus_affects_capacity(self):
        svc = build()
        svc.set_gpus(4)
        assert svc.capacity_qps == 400.0
        svc.set_gpus(0)
        assert svc.capacity_qps == 0.0


class TestEmulatedLatencyModel:
    def test_util_below_capacity_is_healthy(self):
        svc = build()
        svc.set_gpus(4)
        svc.serve(100.0, 0.5)
        assert svc.p95_ms < svc.slo_p95_ms
        assert svc.slo_violated is False

    def test_util_above_capacity_violates_slo(self):
        svc = build()
        svc.set_gpus(2)
        svc.serve(600.0, 0.6)
        assert svc.slo_violated is True
        assert svc.p95_ms >= svc.overload_p95_ms * 0.7

    def test_network_latency_folds_into_reported_p95(self):
        svc = build()
        svc.serve(50.0, 0.5, network_latency_ms=20.0)
        assert svc.p95_ms > 20.0
        assert svc.slo_violated is True

    def test_real_stats_kept_separately(self):
        svc = build()
        svc.serve(50.0, 0.5, network_latency_ms=20.0)
        r = svc.real_stats
        assert "p95_ms" in r and "throughput_qps" in r
        assert r["p95_ms"] <= svc.p95_ms


class TestDebounce:
    def test_overloaded_after_trigger_ticks(self):
        svc = build()
        svc.set_gpus(2)
        svc.serve(600.0, 0.6)
        assert svc.state != "OVERLOADED" or svc.overload_counter >= 1
        svc.serve(600.0, 0.6)
        assert svc.state == "OVERLOADED"

    def test_recovers_after_healthy_ticks(self):
        svc = build()
        svc.set_gpus(2)
        svc.serve(600.0, 0.6)
        svc.serve(600.0, 0.6)
        assert svc.state == "OVERLOADED"
        svc.set_gpus(4)
        svc.serve(50.0, 0.5)
        svc.serve(50.0, 0.5)
        assert svc.state == "RUNNING"
