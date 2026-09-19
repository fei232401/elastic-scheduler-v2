"""GpuPool / ResourceManager 测试（spec AC-01/AC-05 相关）。"""
import pytest

from src.cluster.gpu_pool import GpuPool
from src.cluster.resource_manager import ResourceManager


class TestGpuPool:
    def test_alloc_release(self):
        pool = GpuPool(8)
        assert pool.free_gpus == 8
        ids = pool.alloc_gpus(3)
        assert len(ids) == 3
        assert pool.free_gpus == 5
        pool.free_n(2)
        assert pool.free_gpus == 7

    def test_alloc_too_many_raises(self):
        pool = GpuPool(8)
        pool.alloc_gpus(8)
        with pytest.raises(ValueError):
            pool.alloc_gpus(1)

    def test_free_by_ids_idempotent(self):
        pool = GpuPool(8)
        pool.alloc_gpus(2)
        pool.free_gpus_by_ids([0, 0, 1])
        assert pool.free_gpus == 8


class TestResourceManager:
    def test_initial_allocation(self):
        rm = ResourceManager(total_gpus=8, initial_training=6, min_training=1)
        assert rm.state.training_gpus == 6
        assert rm.state.inference_gpus == 2

    def test_shift_balances_total(self):
        rm = ResourceManager(total_gpus=8, initial_training=6, min_training=1)
        rm.shift(-1)
        assert rm.state.training_gpus == 5
        assert rm.state.inference_gpus == 3
        rm.shift(1)
        assert rm.state.training_gpus == 6
        assert rm.state.inference_gpus == 2

    def test_shift_below_min_raises(self):
        rm = ResourceManager(total_gpus=8, initial_training=2, min_training=1)
        with pytest.raises(ValueError):
            rm.shift(-2)
        assert rm.state.training_gpus == 2
