"""MockTrainingJob 测试（spec §5/§6/§12：幂律吞吐、降级不停止、slowdown 保护）。"""
import pytest

from src.workload.training import MockTrainingJob, TrainingState

CFG = {
    "training": {
        "total_work": 100000,
        "initial_gpu": 6,
        "min_gpu": 1,
        "base_throughput": 15,
        "scaling_exponent": 0.85,
        "max_allowed_slowdown": 3.0,
    }
}


def make_job(**over):
    cfg = {"training": {**CFG["training"], **over}}
    return MockTrainingJob(cfg)


class TestThroughput:
    def test_power_law_monotonic_diminishing(self):
        job = make_job()
        t1, t2, t3 = (job.throughput_at(g) for g in (1, 2, 3))
        assert t1 < t2 < t3  # 单调
        # 边际递减：2卡收益 > 从1到2 还是 从2到3 都变小
        assert (t3 - t2) < (t2 - t1)
        assert t1 == pytest.approx(15.0)

    def test_six_gpu_matches_spec_table(self):
        # spec §5: 6卡 ~71；幂律: 15*6^0.85 ≈ 68.5
        job = make_job()
        assert job.throughput_at(6) == pytest.approx(68.5, abs=0.5)

    def test_three_gpu_matches_spec_table(self):
        job = make_job()
        assert job.throughput_at(3) == pytest.approx(38.1, abs=0.5)


class TestProgress:
    def test_progress_accumulates(self):
        job = make_job()
        job.step(6)
        p1 = job.progress
        job.step(6)
        assert job.progress > p1

    def test_degraded_still_runs(self):
        """AC-06：资源减少后 Training 不停止，progress 继续增加。"""
        job = make_job()
        job.step(6)
        p_full = job.progress
        job.step(3)  # 降级到 3 卡
        assert job.state == TrainingState.DEGRADED
        assert job.progress > p_full  # 仍在推进

    def test_slowdown_grows_when_degraded(self):
        job = make_job()
        job.step(6)
        s_full = job.slowdown_vs_initial
        job.gpus = 3
        assert job.slowdown_vs_initial > s_full

    def test_completed_when_done(self):
        job = make_job(total_work=100)
        for _ in range(10):
            job.step(6)
        assert job.state == TrainingState.COMPLETED

    def test_would_allow_degradation_respects_slowdown(self):
        """§12：超过 max_allowed_slowdown 的降级应被拒绝。"""
        job = make_job(max_allowed_slowdown=2.0)
        # 6卡降1卡：68.5/52.9 ≈ 1.3x，允许
        assert job.would_allow_degradation(5) is True
        # 6卡降1卡（min_gpu=1）仍允许，但降到 1 卡：68.5/15 ≈ 4.6x > 2.0 拒绝
        assert job.would_allow_degradation(1) is False
