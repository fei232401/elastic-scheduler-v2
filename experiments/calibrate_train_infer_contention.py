"""Task #15(项目一真机增量):单卡「训练让渡 → 推理」争抢校准。

把项目一核心机制"训练让渡算力 = 买推理 SLO 保护"从 g^0.85 的 EMULATED 假设
(证据 D)换成单卡 regime 下的**实测数字**(证据 A)。

方法(一块 4090,真实争抢):
    - 推理:1 × vLLM OpenAI server(Qwen2.5-1.5B,util=0.5,**prefix-caching 默认关**——
      本实验测"训练 vs 推理"争抢,不是路由;关缓存避免跨格 cache 预热污染 TTFT)
    - 训练:1 × 真实 PyTorch MLP(复用 R1 的 src.workload.training_real.RealTrainingJob,
      512→2048×4→10, batch 1024,compute-bound util≈95%;步 10ms 级能实质占用 SM)
    - 矩阵:intensity ∈ {0,25,50,75,100}(训练强度 = 步循环占空比;0=完全让渡)
            × 并发度 ∈ {1,4,8,16}(推理并发)
    - 每格测:推理 decode t/s + TTFT p50/p95(stream 首 token)+ 训练 samples/s(心跳文件
      差分,窗口内真实吞吐)
    - 诚实性:
      ① 单卡物理争抢 = REAL;多卡缩放仍是 EMULATED(mock 边界不变);
      ② 训练让渡"代价" = 训练 samples/s 随 intensity 下降(线性是构造出来的,实测值是真数);
      ③ 反向争抢(训练全速时推理也在拖慢训练)也如实测量 → 指向项目一 v2 双向预测;
      ④ 每格一个样本,差异需结合量级看,不作严格统计检验。

🔍 决策点(归来后深挖):
- **为什么单 vLLM server 而非离线 API**:要 TTFT(stream 首 token)这个 serving 指标,
  让渡收益故事的核心是"训练让渡 → 推理 TTFT 改善"。HTTP 开销在格间恒定,比较中抵消。
- **为什么 intensity 用步循环占空比**:最直观的"训练让渡"操作(降步频 = 让出 GPU 时间片);
  100%=满占空比,0%=完全让渡。真实平台对应"训练 checkpoint 频率/批量降级"(Kueue 3.1 原语)。
- **为什么训练侧复用 R1 的 MLP 而非别的模型**:项目一 R1 就用的它,训练负载前后一致;
  小 MLP 是 compute-bound(util≈98%),能和 vLLM 的 kernel 真实争抢 SM。
- **为什么 util=0.5**:单实例留足余量给训练(~2GB),总占用 ~15GB < 24GB;不叠加多实例,
  本实验只回答"训练 vs 推理"的单卡争抢,多实例 contention 是项目二 §4.6 已实测的。

用法(在 AutoDL 4090 有卡模式):
    python -m experiments.calibrate_train_infer_contention \
        --model /root/autodl-tmp/kv-scheduler/models/Qwen2.5-1.5B-Instruct
输出: results/calibrate_train_infer_contention.json + 汇总表。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = ""
DEFAULT_INTENSITIES = "0,25,50,75,100"
DEFAULT_CONCURRENCIES = "1,4,8,16"
DEFAULT_WINDOW_S = 25.0
DEFAULT_RAMP_S = 3.0
DEFAULT_WARMUP_S = 12.0   # server 就绪后的预热负载:vLLM 首个负载窗口含引擎热身(模型/调度器懒初始化),
                          # 不预热会污染第一个测量格(冒烟实测:第一格 TTFT p95 155ms vs 后续 66ms)
DEFAULT_MAX_TOKENS = 128
DEFAULT_PROMPT_TOKENS = 600
DEFAULT_PORT = 8001
DEFAULT_GPU_FRAC = 0.5

# 固定 prompt:长度 ~600 token(和 decode 载荷对称)。prefill 是 GEMM/SM-bound,
# 训练争抢对 TTFT 的影响必须靠真实 prefill 才能测出来 → prompt 拉长,且**默认关
# prefix caching**(本实验测"训练 vs 推理"争抢,不测路由;关缓存消除跨格 cache 预热污染,
# 否则第一格 cold、后续 warm,TTFT 方向会被测反 —— 冒烟实测踩过)。
_BASE = (
    "请详细解释在大规模分布式系统中,KV cache 的显存管理与推理调度的关系。"
    "请分别从 prefill 阶段、decode 阶段、以及多实例并发共享 GPU 的角度展开,"
    "并说明为什么请求级调度需要感知 KV 亲缘性而不是只看负载。"
    "请务必给出完整的、结构化的、有条理的回答,覆盖调度信号的选择、"
    "让渡与归还的时机,以及 SLO 保护的成本权衡。"
)
_PROMPT = (_BASE * 20)[:DEFAULT_PROMPT_TOKENS]


# ---------------------------------------------------------------------------
# 训练 worker(subprocess 运行,占空比控制)
# ---------------------------------------------------------------------------

def _worker_main(args) -> int:
    """在独立进程里跑训练:步循环 + 占空比 sleep,心跳写文件,结束写 JSON。"""
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    import torch.nn.functional as F
    from src.workload.training_real import RealTrainingJob

    intensity = args.intensity
    duration = args.duration
    assert 0 < intensity <= 100

    # 比 R1 默认更大:单卡 4090 上步时间 ~3-8ms,强度 100(背靠背)能实质占用 SM,
    # 让"训练 vs 推理"争抢真实可测。仍是 compute-bound MLP,结构不变。
    cfg = {
        "local": {"training": {"input_dim": 512, "hidden_dim": 2048, "layers": 4,
                               "output_dim": 10, "batch_size": 1024, "total_work": 1e18}},
        "training": {"initial_gpu": 6, "min_gpu": 1},
    }
    job = RealTrainingJob(cfg)
    job.start()
    model, opt = job.model, job.optimizer
    dev = job.device
    bs = job.batch_size
    in_d = job.input_dim
    out_d = job.output_dim

    hb = open(args.heartbeat, "w")
    def beat() -> None:
        hb.write(json.dumps({"wall": time.time(), "steps": steps, "samples": samples}) + "\n")
        hb.flush()

    t0 = time.time()
    steps = 0
    samples = 0
    step_ms_sum = 0.0
    util_samples: list[float] = []
    last_util = time.time()
    while time.time() - t0 < duration:
        s = time.perf_counter()
        x = torch.randn(bs, in_d, device=dev)
        y = torch.randint(0, out_d, (bs,), device=dev)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_ms = (time.perf_counter() - s) * 1000.0
        steps += 1
        samples += bs
        step_ms_sum += step_ms
        if time.time() - last_util >= 0.5:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2).stdout.strip()
                util_samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            last_util = time.time()
        # 占空比控制:step 占 intensity% 的周期,其余 sleep(模拟让渡降强度)
        sleep_s = max(0.0, step_ms / 1000.0 * (100.0 / intensity - 1.0))
        if sleep_s:
            time.sleep(sleep_s)
        beat()

    hb.close()
    wall = time.time() - t0
    out = {
        "intensity": intensity,
        "wall_s": round(wall, 2),
        "steps": steps,
        "samples_per_s": round(samples / wall, 1),
        "avg_step_ms": round(step_ms_sum / max(1, steps), 2),
        "gpu_util_pct_avg": round(sum(util_samples) / max(1, len(util_samples)), 1),
    }
    json.dump(out, open(args.out, "w"))
    return 0


# ---------------------------------------------------------------------------
# vLLM 实例(单实例,复用项目二真机闭环的启动/清理逻辑)
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, model: str, port: int, gpu_frac: float, prefix_cache: bool = False):
        self.model = model
        self.port = port
        self.gpu_frac = gpu_frac
        self.prefix_cache = prefix_cache
        self.url = f"http://localhost:{port}"
        self.proc = None

    def launch(self) -> None:
        log = open(f"/tmp/vllm_contend_{self.port}.log", "w")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model, "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_frac),
            "--max-model-len", "4096", "--enforce-eager", "--served-model-name", "m",
        ]
        if self.prefix_cache:
            cmd.insert(-1, "--enable-prefix-caching")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
        for attempt in range(240):
            if self._health():
                print(f"  server {self.url} ready after {attempt * 2}s", flush=True)
                return
            if attempt % 15 == 0:
                print(f"  waiting {self.url}... {attempt * 2}s", flush=True)
            time.sleep(2)
        raise RuntimeError(f"server 启动超时;看 /tmp/vllm_contend_{self.port}.log")

    def _health(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=2).status_code == 200
        except Exception:
            return False

    def kill(self) -> None:
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
            self.proc = None
        subprocess.run("pkill -9 -f 'vllm.entrypoints.openai'", shell=True,
                       stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f 'VLLM::EngineCore'", shell=True,
                       stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# 推理负载(并发 × 时间窗,逐请求 TTFT/total + token 计数)
# ---------------------------------------------------------------------------

def _send_one(url: str, payload: dict) -> dict:
    t0 = time.monotonic()
    ttft = None
    tokens = 0
    try:
        with requests.post(f"{url}/v1/chat/completions", json=payload,
                           stream=True, timeout=300) as r:
            for raw in r.iter_lines(decode_unicode=True):
                line = (raw or "").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = obj["choices"][0].get("delta", {})
                c = delta.get("content")
                if c:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    tokens += 1
    except Exception:
        return {"ttft_s": float("nan"), "total_s": float("nan"), "tokens": 0}
    total = time.monotonic() - t0
    if ttft is None:
        ttft = total
    return {"ttft_s": ttft, "total_s": total, "tokens": tokens}


def measure_inference(url: str, concurrency: int, window_s: float, max_tokens: int) -> dict:
    """并发 c 线程连续发请求直到时间窗耗尽;返回聚合 decode t/s + TTFT 分位。"""
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": _PROMPT}],
        "max_tokens": max_tokens, "min_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
    }
    results: list[dict] = []
    lock = threading.Lock()
    stop_at = time.monotonic() + window_s

    def worker() -> None:
        while True:
            if time.monotonic() >= stop_at:
                return
            r = _send_one(url, payload)
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    ok = [r for r in results if r["tokens"] > 0]
    tot_tokens = sum(r["tokens"] for r in ok)
    ttfts = sorted(r["ttft_s"] for r in ok)
    totals = sorted(r["total_s"] for r in ok)
    def pct(v, p):
        return v[min(len(v) - 1, int(p / 100 * len(v)))] * 1000 if v else float("nan")
    return {
        "concurrency": concurrency,
        "window_s": round(wall, 2),
        "n_requests": len(ok),
        "decode_tok_per_s": round(tot_tokens / wall, 1),
        "ttft_p50_ms": round(pct(ttfts, 50), 1),
        "ttft_p95_ms": round(pct(ttfts, 95), 1),
        "total_p95_ms": round(pct(totals, 95), 1),
        "avg_resp_tokens": round(tot_tokens / max(1, len(ok)), 1),
    }


# ---------------------------------------------------------------------------
# 训练 worker 进程管理
# ---------------------------------------------------------------------------

def run_train_worker(intensity: int, duration: float, hb_path: Path, out_path: Path) -> subprocess.Popen:
    script = str(Path(__file__).resolve())
    cmd = [
        sys.executable, script, "--worker", "--intensity", str(intensity),
        "--duration", str(duration), "--heartbeat", str(hb_path), "--out", str(out_path),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def read_training_window(hb_path: Path, t_start: float, t_end: float) -> dict:
    """读心跳文件,算 [t_start, t_end] 窗口内训练真实 samples/s。"""
    beats = []
    try:
        for line in open(hb_path):
            line = line.strip()
            if line:
                beats.append(json.loads(line))
    except Exception:
        pass
    window = [b for b in beats if t_start <= b["wall"] <= t_end]
    if len(window) < 2:
        return {"samples_per_s": 0.0, "window_beats": len(window)}
    b0, b1 = window[0], window[-1]
    dt = b1["wall"] - b0["wall"]
    sps = (b1["samples"] - b0["samples"]) / dt if dt > 0 else 0.0
    return {"samples_per_s": round(sps, 1), "window_beats": len(window)}


def _solo_train_baseline(duration: float, hb: Path, out: Path) -> dict:
    """无推理时训练全速(强度 100)的 samples/s 基线。"""
    p = run_train_worker(100, duration, hb, out)
    t_start = time.time() + 1.0
    p.wait()
    t_end = time.time()
    w = read_training_window(hb, t_start, t_end)
    return {"samples_per_s": w["samples_per_s"]}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="项目一 4090 单卡 训练×推理 争抢校准")
    ap.add_argument("--model", default=DEFAULT_MODEL, required=False)
    ap.add_argument("--intensities", default=DEFAULT_INTENSITIES)
    ap.add_argument("--concurrencies", default=DEFAULT_CONCURRENCIES)
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--ramp", type=float, default=DEFAULT_RAMP_S)
    ap.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC)
    ap.add_argument("--prefix-cache", action="store_true", help="默认关;开则 server 加 --enable-prefix-caching")
    ap.add_argument("--out", default="results/calibrate_train_infer_contention.json")
    ap.add_argument("--worker", action="store_true", help="训练 worker 模式(内部)")
    ap.add_argument("--intensity", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--heartbeat", default="")
    args = ap.parse_args()

    if args.worker:
        sys.exit(_worker_main(args))

    model = args.model
    if not model:
        ap.error("--model 必填")

    RES = REPO_ROOT / args.out
    RES.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/contend_worker")

    intensities = [int(x) for x in args.intensities.split(",") if x.strip()]
    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    print(f"矩阵:{len(intensities)} 强度 × {len(concurrencies)} 并发;"
          f"每格推理窗 {args.window}s + 训练 ramp {args.ramp}s", flush=True)

    # 0. 训练 solo 基线(强度 100,无推理)
    print("\n=== 基线:训练全速 solo samples/s(无推理)...", flush=True)
    hb0 = tmp.with_suffix(".hb0")
    out0 = tmp.with_suffix(".w0")
    solo = _solo_train_baseline(args.ramp + 6.0, hb0, out0)
    print(f"  训练 solo = {solo['samples_per_s']} samples/s", flush=True)

    # 1. 起 vLLM server
    print("\n=== 启动 vLLM server(util=%.2f,单实例,prefix_cache=%s)..." %
          (args.gpu_frac, args.prefix_cache), flush=True)
    server = Server(model, args.port, args.gpu_frac, prefix_cache=args.prefix_cache)
    server.launch()

    # 预热:vLLM 引擎懒初始化(模型/调度器首轮热身)会污染第一个测量格 → 先跑一段丢弃
    print(f"\n=== 预热 {args.warmup}s(并发 8,结果丢弃)...", flush=True)
    measure_inference(server.url, 8, args.warmup, args.max_tokens)

    rows = []
    cells = [(i, c) for i in intensities for c in concurrencies]
    random.Random(42).shuffle(cells)   # 固定种子打乱格序,避免机器漂移与强度系统性相关
    try:
        for intensity, conv in cells:
            label = f"intensity={intensity}%({100 - intensity}%让渡)×并发{conv}"
            print(f"\n=== 格:{label}...", flush=True)
            hb = tmp.with_suffix(f".hb_{intensity}_{conv}")
            out = tmp.with_suffix(f".w_{intensity}_{conv}")
            if hb.exists():
                hb.unlink()
            train_p = None
            if intensity > 0:
                train_p = run_train_worker(intensity, args.window + 2 * args.ramp, hb, out)
                time.sleep(args.ramp)  # 等训练 ramp 到目标占空比
            t_win_start = time.time()
            inf = measure_inference(server.url, conv, args.window, args.max_tokens)
            t_win_end = time.time()
            tr = read_training_window(hb, t_win_start, t_win_end)
            if train_p is not None:
                train_p.wait(timeout=args.window + args.ramp + 10)
                try:
                    os.killpg(os.getpgid(train_p.pid), signal.SIGKILL)
                except Exception:
                    pass
            row = {"intensity": intensity, "concurrency": conv, **inf, **tr}
            rows.append(row)
            print(f"  decode={inf['decode_tok_per_s']} t/s, TTFT p50/p95="
                  f"{inf['ttft_p50_ms']}/{inf['ttft_p95_ms']}ms,"
                  f"训练={tr['samples_per_s']} sps", flush=True)
    finally:
        server.kill()

    # 2. 汇总表
    print("\n" + "=" * 88)
    print("4090 单卡 训练×推理 争抢矩阵(训练强度 vs 推理并发)")
    print("=" * 88)
    hdr = f"{'intensity':>10}{'让渡%':>7}{'并发':>5}{'decode_t/s':>12}{'TTFTp50':>9}" \
          f"{'TTFTp95':>9}{'totalP95':>10}{'train_sp/s':>12}"
    print(hdr)
    for r in rows:
        print(f"{r['intensity']:>10}{100 - r['intensity']:>7}{r['concurrency']:>5}"
              f"{r['decode_tok_per_s']:>12}{r['ttft_p50_ms']:>9}{r['ttft_p95_ms']:>9}"
              f"{r['total_p95_ms']:>10}{r['samples_per_s']:>12}")
    print("-" * 88)

    # 3. 让渡收益摘要(每并发:强度 100 vs 0)
    print("\n让渡收益(训练全速→完全让渡):")
    by_conv = {}
    for r in rows:
        by_conv.setdefault(r["concurrency"], {})[r["intensity"]] = r
    for conv, d in sorted(by_conv.items()):
        if 0 in d and 100 in d:
            a, b = d[100], d[0]
            dgain = (b["decode_tok_per_s"] / a["decode_tok_per_s"] - 1) * 100 if a["decode_tok_per_s"] else float("nan")
            tdiff = a["ttft_p95_ms"] - b["ttft_p95_ms"]
            print(f"  并发{conv}: 让渡后 decode {a['decode_tok_per_s']}→{b['decode_tok_per_s']} t/s"
                  f"(+{dgain:.0f}%),TTFT p95 {a['ttft_p95_ms']}→{b['ttft_p95_ms']}ms"
                  f"(训练全速时高 {tdiff:+.1f}ms)")

    # 4. 反向共置代价(有推理时训练吞吐 vs solo 基线)——冒烟实测:训练掉 44%
    full_conv = sorted(by_conv)[-1]
    d = by_conv[full_conv]
    if 100 in d and solo["samples_per_s"]:
        cost = (1 - d[100]["samples_per_s"] / solo["samples_per_s"]) * 100
        print(f"\n共置代价(推理并发{full_conv}时训练):solo {solo['samples_per_s']} → "
              f"{d[100]['samples_per_s']} sps(训练降 {cost:.0f}%)")

    meta = {
        "model": model, "intensities": intensities, "concurrencies": concurrencies,
        "window_s": args.window, "ramp_s": args.ramp, "max_tokens": args.max_tokens,
        "gpu_frac": args.gpu_frac, "train_solo_sps": solo["samples_per_s"],
        "date": "2026-08-22",
    }
    json.dump({"meta": meta, "rows": rows}, open(RES, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {RES}")


if __name__ == "__main__":
    main()
