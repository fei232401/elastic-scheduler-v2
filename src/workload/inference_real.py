"""真实 HTTP 推理 workload（Phase 3.2 Step 5，spec §8-9，Gate 2）。

组件：
  - 真实 HTTP server：ThreadingHTTPServer + 小型 MLP，跑在 1 张物理 GPU；
  - 动态批处理（真实推理 server 的标准行为，如 vLLM）：请求进队列，
    batcher 攒批（达到 max_batch 或 flush 间隔）后一次批前向。
    超载 → 队列增长 → 真实等待延迟上升 —— 这是 R2 的真实 latency curve；
  - 真实 load generator：按目标 QPS 发真实 HTTP 请求，测真实 P50/P95/P99。

GPU / 容量模型（诚实标注，spec §22）：
  - server 的真实延迟曲线在 Step 6（R2 profiling）测量，作为地面真值；
  - 上报给调度器的 capacity_qps = capacity_per_gpu × allocated_gpu（虚拟配额，
    EMULATED 线性模型，与 Mock 同构；capacity_per_gpu 由 Step 6 校准为 C1/initial_gpu）；
  - 上报给调度器的 P95 = 真实曲线校准的「利用率→延迟」映射（util = qps/capacity），
    初始配额处退化为真实测量；
  - 真实测量（real p95 / throughput / GPU util / mem）单独保留（`real_stats`），
    供 Reality Gap 报告使用 —— 虚拟扩卡提高 EMULATED 容量但物理真实 P95 受 1 卡
    饱和限制，正是 spec §18 #6（GPU 增加但 capacity 增长很小）失败案例的来源。
"""
from __future__ import annotations

import http.server
import json
import threading
import time

import torch
import torch.nn as nn

from src.config import get


class _InferHandler(http.server.BaseHTTPRequestHandler):
    """POST /infer → 入批队列 → 批前向 → 200。"""

    service = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 0:
            self.rfile.read(length)
        dt_ms = self.service._submit()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')
        self.service._record(dt_ms)

    def log_message(self, *args) -> None:
        pass


class HTTPInferenceService:
    """真实推理服务：被 LocalRuntimeAdapter 经 duck-type 契约驱动。"""

    def __init__(self, cfg: dict) -> None:
        local = get(cfg, "local", {})
        li = get(local, "inference", {})
        inf = get(cfg, "inference", {})
        sim = get(cfg, "simulation", {})

        self.slo_p95_ms = float(get(li, "slo_p95_ms", get(inf, "slo_p95_ms", 300)))
        self.capacity_per_gpu = float(get(li, "capacity_per_gpu", get(inf, "capacity_per_gpu", 10)))
        self.overload_trigger_ticks = int(get(inf, "overload_trigger_ticks", 3))
        self.recovery_trigger_ticks = int(get(inf, "recovery_trigger_ticks", 5))
        self.host = str(get(li, "host", "127.0.0.1"))
        self.port = int(get(li, "port", 8701))
        self.workers = int(get(li, "generator_workers", 16))

        self.max_batch_size = int(get(li, "max_batch_size", 16))
        self.flush_interval_ms = float(get(li, "flush_interval_ms", 1.0))

        self.input_dim = int(get(li, "input_dim", 64))
        self.hidden_dim = int(get(li, "hidden_dim", 256))
        self.layers = int(get(li, "layers", 3))
        self.output_dim = int(get(li, "output_dim", 10))
        dev = get(local, "device", "auto")
        self.device = torch.device(
            "cuda" if dev == "auto" and torch.cuda.is_available() else "cpu"
        )
        self._is_gpu = self.device.type == "cuda"

        total = int(get(sim, "total_gpus", 8))
        initial_t = int(get(get(cfg, "training", {}), "initial_gpu", 6))
        self.allocated_gpu: int = total - initial_t

        self.base_p95_ms = float(get(li, "base_p95_ms", 3.0))
        self.overload_p95_ms = float(get(li, "overload_p95_ms", 10.0))
        self.saturation_qps = float(get(li, "saturation_qps", 1500.0))

        self.qps: float = 0.0
        self.throughput_qps: float = 0.0
        self.p50_ms: float = 0.0
        self.p95_ms: float = 0.0
        self.p99_ms: float = 0.0
        self.queue_length: int = 0
        self.slo_violated: bool = False
        self.state: str = "RUNNING"
        self.gpu_utilization: float = 0.0
        self.gpu_memory_mb: float = 0.0
        self._real_p50_ms: float = 0.0
        self._real_p95_ms: float = 0.0
        self._real_p99_ms: float = 0.0
        self._real_throughput_qps: float = 0.0
        self._e2e_p50_ms: float = 0.0
        self._e2e_p95_ms: float = 0.0
        self._e2e_p99_ms: float = 0.0
        self._oc: int = 0
        self._rc: int = 0
        self._latencies: list[float] = []
        self._e2e_latencies: list[float] = []
        self._lock = threading.Lock()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._batcher_thread: threading.Thread | None = None
        self._running = False
        self._batch_queue: list[tuple[threading.Event, dict]] = []

        self._build_model()

    def _build_model(self) -> None:
        torch.manual_seed(42)
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(max(0, self.layers - 1)):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        self.model.eval()

    def _submit(self) -> float:
        """入批队列并等待结果，返回本请求端到端延迟（ms，REAL）。"""
        ev = threading.Event()
        holder: dict = {"out": None}
        t0 = time.perf_counter()
        with self._lock:
            self._batch_queue.append((ev, holder))
        if not ev.wait(timeout=30.0):
            return 30_000.0
        return (time.perf_counter() - t0) * 1000.0

    def _batch_loop(self) -> None:
        while self._running:
            batch: list[tuple[threading.Event, dict]] = []
            with self._lock:
                while self._batch_queue and len(batch) < self.max_batch_size:
                    batch.append(self._batch_queue.pop(0))
            if not batch:
                time.sleep(self.flush_interval_ms / 1000.0)
                continue
            b = len(batch)
            x = torch.randn(b, self.input_dim, device=self.device)
            with torch.no_grad():
                self.model(x)
            for _, holder in batch:
                holder["out"] = True
            for ev, _ in batch:
                ev.set()

    def _record(self, dt_ms: float) -> None:
        with self._lock:
            self._latencies.append(dt_ms)

    @staticmethod
    def _percentiles(vals: list[float], ps: list[float]) -> list[float]:
        if not vals:
            return [0.0] * len(ps)
        s = sorted(vals)
        n = len(s)
        out = []
        for p in ps:
            out.append(s[max(0, min(n - 1, int(round(p * (n - 1)))))])
        return out

    @property
    def capacity_qps(self) -> float:
        """EMULATED 容量：capacity_per_gpu × allocated_gpu（与 Mock 同构）。"""
        return self.capacity_per_gpu * self.allocated_gpu

    def _emulated_latency(self, util: float) -> tuple[float, float, float]:
        """用真实曲线校准的利用率→延迟映射（Shape 同 Mock，参数来自真实测量）。"""
        base, over = self.base_p95_ms, self.overload_p95_ms
        if util <= 1.0:
            p95 = base * (0.75 + 0.25 * util)
            p50 = p95 * 0.6
            p99 = p95 * 1.02
        else:
            p95 = base + (over - base) * min(1.0, (util - 1.0) ** 1.5)
            p50 = p95 * 0.6
            p99 = p95 * (1.0 + 0.05 * min(2.0, util - 1.0))
        return p50, p95, p99

    def set_gpus(self, gpus: int) -> None:
        self.allocated_gpu = max(0, int(gpus))

    def start(self) -> None:
        if self._server is not None:
            return
        _InferHandler.service = self
        self._server = http.server.ThreadingHTTPServer((self.host, self.port), _InferHandler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        self._running = True
        self._batcher_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._batcher_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2)
            self._server_thread = None
        if self._batcher_thread is not None:
            self._batcher_thread.join(timeout=2)
            self._batcher_thread = None

    def serve(self, qps: float, seconds: float, network_latency_ms: float = 0.0) -> None:
        """驱动真实 HTTP 负载 `seconds` 秒，测真实延迟 + 计算上报给调度器的 EMULATED 延迟。

        Args:
            qps: 目标到达率（请求/秒）。
            seconds: 负载窗口时长。
            network_latency_ms: 网络拥塞延迟（Real，来自 tc/netem 控制器），
                叠加到上报给调度器的 EMULATED P95（镜像 Mock 的 network_latency_ms 机制）。
        """
        with self._lock:
            self._latencies = []
            self._e2e_latencies = []
        dur = max(0.2, seconds)
        deadline = time.perf_counter() + dur
        stop = threading.Event()
        workers = [
            threading.Thread(target=self._worker, args=(qps, deadline, stop), daemon=True)
            for _ in range(self.workers)
        ]
        for w in workers:
            w.start()
        time.sleep(dur)
        stop.set()
        for w in workers:
            w.join(timeout=3)

        with self._lock:
            lats = list(self._latencies)
            e2e = list(self._e2e_latencies)
        self.qps = float(qps)
        self._real_throughput_qps = len(lats) / dur if dur > 0 else 0.0
        self.throughput_qps = self._real_throughput_qps

        real50, real95, real99 = self._percentiles(lats, [0.5, 0.95, 0.99])
        self._real_p50_ms, self._real_p95_ms, self._real_p99_ms = real50, real95, real99
        e2e50, e2e95, e2e99 = self._percentiles(e2e, [0.5, 0.95, 0.99])
        self._e2e_p50_ms, self._e2e_p95_ms, self._e2e_p99_ms = e2e50, e2e95, e2e99

        cap = self.capacity_qps
        util = qps / cap if cap > 0 else 1.0
        e50, e95, e99 = self._emulated_latency(util)
        self.p50_ms = e50 + network_latency_ms * 0.5
        self.p95_ms = e95 + network_latency_ms
        self.p99_ms = e99 + network_latency_ms
        self.queue_length = int(round(max(0.0, qps - cap) * 2.0))
        self.slo_violated = self.p95_ms > self.slo_p95_ms

        self._sample_gpu_metrics()

        if self.slo_violated:
            self._oc += 1
            self._rc = 0
        else:
            self._rc += 1
            self._oc = 0
        if self._oc >= self.overload_trigger_ticks and self.state != "OVERLOADED":
            self.state = "OVERLOADED"
        elif self._rc >= self.recovery_trigger_ticks and self.state == "OVERLOADED":
            self.state = "RUNNING"
            self._oc = 0
            self._rc = 0

    def _worker(self, target_qps: float, deadline: float, stop: threading.Event) -> None:
        """负载生成线程：尽量逼近 target_qps，记录每个请求延迟（server 侧 + 客户端观测）。"""
        import urllib.request
        payload = json.dumps({"x": [0.0] * self.input_dim}).encode()
        interval = 0.0 if target_qps <= 0 else self.workers / target_qps
        while not stop.is_set() and time.perf_counter() < deadline:
            next_t = time.perf_counter()
            try:
                req = urllib.request.Request(
                    f"http://{self.host}:{self.port}/infer",
                    data=payload, headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                e2e_ms = (time.perf_counter() - next_t) * 1000.0
                with self._lock:
                    self._e2e_latencies.append(e2e_ms)
            except Exception:
                pass
            if interval > 0:
                remain = interval - (time.perf_counter() - next_t)
                if remain > 0:
                    time.sleep(remain)

    def _sample_gpu_metrics(self) -> None:
        if not self._is_gpu:
            self.gpu_utilization = 0.0
            self.gpu_memory_mb = 0.0
            return
        try:
            self.gpu_memory_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        except Exception:
            self.gpu_memory_mb = 0.0
        try:
            self.gpu_utilization = float(torch.cuda.utilization()) / 100.0
        except Exception:
            self.gpu_utilization = 0.0

    @property
    def overload_counter(self) -> int:
        return self._oc

    @property
    def recovery_counter(self) -> int:
        return self._rc

    @property
    def real_stats(self) -> dict:
        """真实测量（REAL），Reality Gap 报告用。

        含两套延迟：
          - p95_ms：server 侧处理延迟（请求到达 server 后才计时，不含网络 RTT）；
          - e2e_p95_ms：客户端观测的端到端延迟（含网络 RTT，用户感知 —— SLO 对应
            这个口径）。R2 的容量校准用 server 侧；R3 的拥塞抬升用 e2e 侧。
        """
        return {
            "p50_ms": self._real_p50_ms,
            "p95_ms": self._real_p95_ms,
            "p99_ms": self._real_p99_ms,
            "e2e_p50_ms": self._e2e_p50_ms,
            "e2e_p95_ms": self._e2e_p95_ms,
            "e2e_p99_ms": self._e2e_p99_ms,
            "throughput_qps": self._real_throughput_qps,
        }
