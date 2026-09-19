"""MockInferenceService 测试（spec §7/§8/§9：容量、非线性延迟、防抖）。"""
from src.workload.inference import InferenceState, MockInferenceService

CFG = {
    "inference": {
        "capacity_per_gpu": 10,
        "slo_p95_ms": 300,
        "p95_base_ms": 200,
        "p95_overload_ms": 900,
        "overload_trigger_ticks": 3,
        "recovery_trigger_ticks": 5,
    }
}


def make_service(**over):
    return MockInferenceService({"inference": {**CFG["inference"], **over}})


class TestCapacity:
    def test_capacity_linear_in_gpus(self):
        svc = make_service()
        svc.gpus = 2
        assert svc.capacity_qps == 20
        svc.gpus = 5
        assert svc.capacity_qps == 50

    def test_under_capacity_p95_ok(self):
        svc = make_service()
        svc.gpus = 3
        svc.step(20)
        assert svc.slo_violated is False
        assert svc.p95_ms < svc.slo_p95_ms

    def test_over_capacity_p95_spikes(self):
        svc = make_service()
        svc.gpus = 3
        svc.step(60)
        assert svc.p95_ms > svc.slo_p95_ms
        assert svc.queue_length > 0


class TestHysteresis:
    def test_overload_needs_consecutive_ticks(self):
        """防抖：连续 3 tick 过载才置 OVERLOADED。"""
        svc = make_service()
        svc.gpus = 2
        svc.step(40)
        assert svc.state == InferenceState.RUNNING
        svc.step(40)
        assert svc.state == InferenceState.RUNNING
        svc.step(40)
        assert svc.state == InferenceState.OVERLOADED

    def test_transient_spike_does_not_confirm(self):
        """瞬时尖峰（1 tick）不应触发过载确认。"""
        svc = make_service()
        svc.gpus = 3
        svc.step(60)
        assert svc.state == InferenceState.RUNNING
        svc.step(20)
        assert svc.state == InferenceState.RUNNING

    def test_recovery_needs_consecutive_healthy_ticks(self):
        svc = make_service()
        svc.gpus = 2
        for _ in range(3):
            svc.step(40)
        assert svc.state == InferenceState.OVERLOADED
        svc.step(10)
        assert svc.state in (InferenceState.RECOVERING, InferenceState.OVERLOADED)
        for _ in range(4):
            svc.step(10)
        assert svc.state == InferenceState.RUNNING
