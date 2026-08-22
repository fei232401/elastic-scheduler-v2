"""R3 — 真实网络 profiling 驱动（Phase 3.2 Step 8，spec §15 / Gate 3）。

父进程无 CAP_NET_ADMIN，无法在自身 netns 的 lo 上施加整形。解法：把整个测量
进程用 `unshare -Urn` 搬进一个全新的 user+net namespace（进程成为该 netns 内的
root，可对 lo 加 netem），R3 内核（_r3_core.py）在 namespace 内起真实 HTTP
推理栈，依次施加不同 netem 延迟，量真实 P95 抬升曲线。

用法: python -m experiments.profile_network [--slo 15] [--out results/profiles/network_profile.json]
输出: results/profiles/network_profile.json（由 namespace 内的内核写入，同文件系统）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser(description="R3: real network latency under netem")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--slo", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=2.5)
    ap.add_argument("--out", default="results/profiles/network_profile.json")
    args = ap.parse_args()

    cmd = [
        "unshare", "-Urn", "python", "-m", "experiments._r3_core",
        "--config", args.config,
        "--slo", str(args.slo),
        "--duration", str(args.duration),
        "--out", args.out,
    ]
    r = subprocess.run(cmd, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"R3 core failed with rc={r.returncode}")

    out = pathlib.Path(args.out)
    if out.exists():
        print(out.read_text())


if __name__ == "__main__":
    main()
