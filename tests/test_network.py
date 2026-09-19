"""MockNetwork 测试（spec §14-§17：训练/推理带宽、拥塞、网络感知判定）。"""
import pytest

from src.network.network import MockNetwork

CFG = {"network": {
    "total_bandwidth_mbps": 2200,
    "base_latency_ms": 2,
    "latency_gain_per_overload": 250,
    "congestion_start_util": 0.5,
    "training_bw_scale": 1.0,
    "train_bw_coeff_a": 50,
    "train_bw_coeff_b": 150,
    "per_gpu_infer_bw": 600,
    "bw_per_qps": 30,
    "training_penalty_per_unit": 0.3,
    "max_training_penalty": 0.7,
}}


def make(**over) -> MockNetwork:
    return MockNetwork({"network": {**CFG["network"], **over}})


class TestTrainingBandwidth:
    def test_matches_spec_table(self):
        """spec §14：2→500, 3→900, 4→1400, 5→2000, 6→2700 Mbps。"""
        n = make()
        for g, expected in [(2, 500), (3, 900), (4, 1400), (5, 2000), (6, 2700)]:
            assert n.training_bandwidth(g) == pytest.approx(expected)

    def test_zero_below_two(self):
        """g=1 无训练带宽（spec §14：1→0）。"""
        n = make()
        assert n.training_bandwidth(1) == 0.0
        assert n.training_bandwidth(0) == 0.0

    def test_scale_linear(self):
        n = make(training_bw_scale=0.2)
        assert n.training_bandwidth(4) == pytest.approx(280)


class TestInferenceBandwidth:
    def test_qps_and_per_gpu(self):
        """spec §15：QPS 驱动 + 每 GPU 边际带宽。"""
        n = make()
        assert n.inference_bandwidth(4, 30) == pytest.approx(3300)


class TestCongestion:
    def test_under_capacity_no_latency_penalty(self):
        n = make(total_bandwidth_mbps=30000)
        n.update(training_gpus=6, inference_gpus=2, qps=10)
        assert n.congested is False
        assert n.latency_ms == n.base_latency_ms
        assert n.throughput_scale == 1.0

    def test_over_capacity_latency_and_penalty(self):
        n = make()
        n.update(training_gpus=4, inference_gpus=4, qps=30)
        assert n.congested is True
        assert n.utilization > 1.0
        assert n.latency_ms > n.base_latency_ms
        assert n.throughput_scale < 1.0

    def test_penalty_capped(self):
        n = make(max_training_penalty=0.7, total_bandwidth_mbps=1000)
        n.update(training_gpus=1, inference_gpus=7, qps=30)
        assert n.throughput_scale == pytest.approx(0.3)


class TestExpansionFeasible:
    def test_blocked_when_released_less_than_delta(self):
        """spec §17：缩训练释放带宽 < 扩推理新增带宽 + 无余量 → 拒绝扩容。"""
        n = make()
        n.update(training_gpus=4, inference_gpus=4, qps=30)
        feasible, detail = n.expansion_feasible(4, 3, 4, 5)
        assert feasible is False
        assert detail["released_training_bw"] == pytest.approx(500)
        assert detail["delta_inference_bw"] == pytest.approx(600)

    def test_allowed_when_headroom_covers(self):
        n = make(total_bandwidth_mbps=30000)
        n.update(training_gpus=4, inference_gpus=4, qps=10)
        feasible, _ = n.expansion_feasible(4, 3, 4, 5)
        assert feasible is True

    def test_flat_curve_blocks_more(self):
        """训练带宽曲线越平，释放越少，越容易被拒（Experiment B 场景）。"""
        n = make(training_bw_scale=0.2)
        n.update(training_gpus=4, inference_gpus=4, qps=30)
        feasible, detail = n.expansion_feasible(4, 3, 4, 5)
        assert feasible is False
        assert detail["released_training_bw"] == pytest.approx(100)
