"""NetemNetwork 契约 + Gate 3 能力测试（Phase 3.2 Step 7，spec §10-12）。

验证：
  - 与 MockNetwork 同一核算契约：相同配置/输入下状态字段逐字节一致
    （统一 Runtime Contract，spec §3.2）—— 调度器对两者行为可比；
  - §17 扩容闸门（expansion_feasible）与 Mock 一致；
  - Gate 3：unshare+tc 探测（probe）、apply/clear shaping 命令可用
    （注意：父进程 netns 的 lo 不受 namespace 整形影响，这里只验证命令能力，
     真实延迟抬升的 REAL 测量在 R3 的 namespace 内完成）。
"""
from __future__ import annotations

from src.network import MockNetwork, NetemNetwork
from src.runtime.local import LocalRuntimeAdapter
from tests.test_local_runtime import StubInference, StubTraining

CFG = {
    "simulation": {"total_gpus": 8, "tick_seconds": 1},
    "training": {
        "total_work": 100000, "initial_gpu": 4, "min_gpu": 1,
        "base_throughput": 15, "scaling_exponent": 0.85, "max_allowed_slowdown": 3.0,
    },
    "inference": {
        "capacity_per_gpu": 10, "slo_p95_ms": 300, "p95_base_ms": 200,
        "p95_overload_ms": 900, "overload_trigger_ticks": 3, "recovery_trigger_ticks": 5,
    },
    "network": {
        "total_bandwidth_mbps": 2200,
        "base_latency_ms": 2,
        "latency_gain_per_overload": 250,
        "congestion_start_util": 0.5,
        "training_bw_scale": 0.2,
        "train_bw_coeff_a": 50,
        "train_bw_coeff_b": 150,
        "per_gpu_infer_bw": 600,
        "bw_per_qps": 30,
        "training_penalty_per_unit": 0.3,
        "max_training_penalty": 0.7,
    },
}

FIELDS = ("demand_mbps", "utilization", "congested", "latency_ms",
          "throughput_scale", "headroom_mbps")


def _states(cfg=None):
    cfg = cfg or CFG
    return MockNetwork(cfg), NetemNetwork(cfg)


def _assert_same(mock, netem, train_gpus, infer_gpus, qps):
    for f in FIELDS:
        assert abs(getattr(netem, f) - getattr(mock, f)) < 1e-9, f
    # 带宽函数（方法）与 LocalRuntimeAdapter 读取的拆分属性
    assert abs(netem.training_bandwidth(train_gpus) - mock.training_bandwidth(train_gpus)) < 1e-9
    assert abs(netem.inference_bandwidth(infer_gpus, qps) - mock.inference_bandwidth(infer_gpus, qps)) < 1e-9
    assert abs(netem.training_bandwidth_mbps - mock.training_bandwidth(train_gpus)) < 1e-9
    assert abs(netem.inference_bandwidth_mbps - mock.inference_bandwidth(infer_gpus, qps)) < 1e-9


def test_contract_identical_to_mock():
    mock, netem = _states()
    mock.update(4, 4, 30.0)
    netem.update(4, 4, 30.0)
    _assert_same(mock, netem, 4, 4, 30.0)
    assert netem.capacity_mbps == mock.capacity_mbps


def test_contract_identical_across_allocations():
    mock, netem = _states()
    for (t, i, q) in [(6, 2, 10.0), (3, 5, 900.0), (1, 7, 50.0)]:
        mock.update(t, i, q)
        netem.update(t, i, q)
        _assert_same(mock, netem, t, i, q)


def test_gate_identical_to_mock():
    mock, netem = _states()
    mock.update(4, 4, 30.0)
    netem.update(4, 4, 30.0)
    assert mock.expansion_feasible(4, 3, 4, 5) == netem.expansion_feasible(4, 3, 4, 5)
    # NET_BOUND 语义：释放训练带宽 < 新增推理带宽 → 拦截
    ok, detail = netem.expansion_feasible(4, 3, 4, 5)
    assert ok is False
    assert set(detail) >= {"released_training_bw", "delta_inference_bw",
                           "headroom_mbps"}


def test_probe_gate3_capability():
    netem = NetemNetwork(CFG)
    assert netem.probe() is True  # 本机 unshare+tc 可用（已实测）
    assert netem.shaping_available is True
    # probe 幂等
    assert netem.probe() is True


def test_apply_clear_shaping_commands_run():
    netem = NetemNetwork(CFG)
    netem.start()  # 触发 probe
    ok, err = netem.apply_shaping(20.0)
    assert ok, err  # 命令成功（作用于瞬态 namespace）
    assert netem.real_delay_ms == 20.0
    ok, err = netem.clear_shaping()
    assert ok, err
    assert netem.real_delay_ms == 0.0
    netem.stop()


def test_adapter_wiring_with_netem_network():
    """LocalRuntimeAdapter + NetemNetwork：start 探测 + tick 网络状态流入调度器视图。"""
    rm_cfg = {"simulation": {"total_gpus": 8}, "training": {"initial_gpu": 4, "min_gpu": 1}}
    from src.cluster.resource_manager import ResourceManager

    rm = ResourceManager(total_gpus=8, initial_training=4, min_training=1)
    a = LocalRuntimeAdapter(
        rm=rm,
        training=StubTraining(CFG),
        inference=StubInference(CFG),
        network=NetemNetwork(CFG),
        tick_seconds=1.0,
    )
    a.start()  # getattr(start) → NetemNetwork.probe
    a.advance(30.0)
    ns = a.get_network_state()
    assert ns.utilization > 1.0  # NET_BOUND 已拥塞
    assert ns.congested is True
    assert ns.headroom_mbps == 0.0
    assert ns.training_bandwidth_mbps == 280.0  # 0.2*(50*16+150*4)
    a.stop()
