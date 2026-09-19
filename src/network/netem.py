"""Local 网络控制器 —— Mock 带宽核算模型（EMULATED）+ 真实 tc/netem 整形（REAL）。

（Phase 3.2 Step 7，spec §10-12 / Gate 3）

与 Mock 的关系（统一 Runtime Contract，spec §3.2/§22 诚实标注）：
  - 调度器看到的带宽核算 / 拥塞延迟 / §17 闸门 = 与 MockNetwork **同一套代码**
    （继承），因此调度器对 Local 的行为与对 Mock 逐字节一致 —— 这是可比性的前提；
  - 区别在「网络现实」层：本机是单机 loopback，不存在共享真实网络，带宽占用 /
    利用率是模型量（EMULATED，无法真实测量）；但**延迟可以用 tc/netem 真实施加**：
    - `apply_shaping(delay_ms)`：在 namespace 的 lo 上施加 netem 延迟（REAL），
      真实 HTTP 请求真的被拖慢 —— R3（profile_network）在 namespace 内跑完整
      推理栈，量到真实 P95 抬升；
    - 非 root 无 CAP_NET_ADMIN，但 `unshare -Urn` 在 user+net namespace 内映射为
      root，可对 namespace 内 lo 加 netem（已实测 ADD_OK/DEL_OK + 真实 RTT 抬升）。
      父进程所在主 netns 的 lo 不受影响（没有权限），所以真实整形必须把
      实验整体搬进 namespace —— 这正是 R3 的做法，诚实标注。

本机没有「真实的共享带宽」可测量，因此 NetworkState 里的带宽数字永远标 EMULATED；
R3 只把「拥塞 → 真实延迟抬升 → P95 越过 SLO」这一机制做成 REAL 测量。
"""
from __future__ import annotations

import shutil
import subprocess

from src.network.network import MockNetwork


def _netem_delay_ms() -> float:
    """读取当前 namespace lo 上 netem 延迟（无整形返回 0.0）。失败返回 -1。"""
    try:
        out = subprocess.run(
            ["tc", "qdisc", "show", "dev", "lo"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        if "netem" not in out:
            return 0.0
        for tok in out.replace("delay", " delay ").split():
            if tok.endswith("ms") and tok[:-2].replace(".", "").isdigit():
                return float(tok[:-2])
        return 0.0
    except Exception:
        return -1.0


class NetemNetwork(MockNetwork):
    """Mock 核算 + 真实 netem 整形。继承 MockNetwork 保证调度器视角行为一致。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.real_delay_ms: float = 0.0
        self.shaping_available: bool = False
        self.tc_available: bool = shutil.which("tc") is not None
        self.training_bandwidth_mbps: float = 0.0
        self.inference_bandwidth_mbps: float = 0.0

    def update(self, training_gpus: int, inference_gpus: int, qps: float) -> None:
        """Mock 核算 + 额外填充带宽拆分（LocalRuntimeAdapter.get_network_state 直接读取）。"""
        super().update(training_gpus, inference_gpus, qps)
        self.training_bandwidth_mbps = self.training_bandwidth(training_gpus)
        self.inference_bandwidth_mbps = self.inference_bandwidth(inference_gpus, qps)

    def _unshare_cmd(self, body: str) -> tuple[int, str]:
        """在 user+net namespace 内以 root 执行 `tc`（非 root 无 CAP_NET_ADMIN 的解法）。"""
        try:
            r = subprocess.run(
                ["unshare", "-Urn", "bash", "-c", body],
                capture_output=True, text=True, timeout=15,
            )
            return r.returncode, (r.stdout + r.stderr).strip()
        except Exception as e:
            return -1, str(e)

    def probe(self) -> bool:
        """Gate 3：验证 unshare+tc 可用（namespace lo 加/删零延迟 qdisc）。"""
        if not self.tc_available:
            self.shaping_available = False
            return False
        rc, _ = self._unshare_cmd(
            "ip link set lo up && "
            "tc qdisc add dev lo root netem delay 0ms 2>/dev/null && "
            "tc qdisc del dev lo root"
        )
        self.shaping_available = rc == 0
        return self.shaping_available

    def apply_shaping(self, delay_ms: float) -> tuple[bool, str]:
        """在 namespace lo 上施加真实 netem 延迟（REAL）。

        注意：只作用于本次 `unshare` 产生的 namespace 内 lo —— 父进程主 netns 的
        loopback 不受影响。要让真实 HTTP 流量穿越整形，必须把实验跑进 namespace
        （R3 的做法）。这里返回值只是能力证明 + 供 R3 驱动复用。
        """
        self.real_delay_ms = float(delay_ms)
        rc, err = self._unshare_cmd(
            "ip link set lo up && "
            f"tc qdisc del dev lo root 2>/dev/null; "
            f"tc qdisc add dev lo root netem delay {self.real_delay_ms}ms"
        )
        return rc == 0, err

    def clear_shaping(self) -> tuple[bool, str]:
        rc, err = self._unshare_cmd(
            "ip link set lo up && tc qdisc del dev lo root 2>/dev/null; true"
        )
        self.real_delay_ms = 0.0
        return rc == 0, err

    def start(self) -> None:
        """探测 tc/netem 可用性（Gate 3）。"""
        self.probe()

    def stop(self) -> None:
        """移除整形（若本进程真有权限）。"""
        if self.real_delay_ms:
            self.clear_shaping()
