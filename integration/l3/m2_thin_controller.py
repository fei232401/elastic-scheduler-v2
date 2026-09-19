#!/usr/bin/env python3
r"""L3 M2 · 薄 controller —— 潮汐 → 决策引擎 → 真实 Kueue 弹性 Job 最小闭环。

复用 mock 组件（零改动，面试核心卖点）：
  - src.scheduler.elastic.ElasticScheduler   : 决策引擎（Rules 1-5 + min_hold 防抖 + §12 闸门）
  - src.workload.inference.MockInferenceService : 效应模型（容量/P95/OVERLOADED 防抖状态机）
  - src.workload.traffic.TrafficGenerator     : BurstGPT 潮汐（70min custom schedule，seed 确定性）
  - src.workload.training.MockTrainingJob     : §12 训练进度保护（slowdown 闸门）

平台层翻译（与 mock 的唯一差异，也是 L3 的意义）：
  - 1 pod = 1 GPU。训练 = Kueue 弹性 Job（annotation `kueue.x-k8s.io/elastic-job` +
    label `kueue.x-k8s.io/queue-name`）；推理 = 普通 Deployment（controller 直接 scale，
    不占 Kueue 配额语义，守恒由决策引擎保证）。
  - 每 tick 从【真实集群】读"实际 Running" pod 数喂回效应模型（真实反馈，非理想假设）；
    决策 changed → patch Job parallelism / Deployment replicas → Kueue 真实执行弹性机制。

用法（AutoDL，仓库根目录执行）：
  python integration/l3/m2_thin_controller.py \
      --config integration/l3/l3_smoke.yaml \          # 先冒烟
      --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml \
      --out /opt/l3-controller/run_smoke.json
  python integration/l3/m2_thin_controller.py --dump-manifests --config integration/l3/l3_default.yaml
  python integration/l3/m2_thin_controller.py \          # 全量
      --config integration/l3/l3_default.yaml \
      --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml \
      --out /opt/l3-controller/run_full.json --tick-wall 1.0

确定性：seed 42（与 mock 同源），jitter 0.05 走 TrafficGenerator 的 RNG。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import get, load_config
from src.scheduler.elastic import ElasticScheduler
from src.scheduler.base import SchedulerContext
from src.workload.inference import MockInferenceService
from src.workload.training import MockTrainingJob
from src.workload.traffic import TrafficGenerator
from integration.l3.predictive import L3PredictiveScheduler

NS = "default"
TRAIN_JOB = "l3-train-job"
INFER_DEPLOY = "l3-infer-svc"
TRAIN_LABEL = f"job-name={TRAIN_JOB}"
INFER_LABEL = "app=l3-infer-svc"
QUEUE = "lq-train-infer"
COMPLETIONS = 100000


def build_job(cfg: dict):
    from kubernetes import client
    init = get(cfg, "training.initial_gpu", 8000)
    return client.V1Job(
        metadata=client.V1ObjectMeta(
            name=TRAIN_JOB,
            namespace=NS,
            labels={"kueue.x-k8s.io/queue-name": QUEUE},
            annotations={"kueue.x-k8s.io/elastic-job": "true"},
        ),
        spec=client.V1JobSpec(
            parallelism=init,
            completions=COMPLETIONS,
            suspend=True,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"kueue.x-k8s.io/queue-name": QUEUE, "app": TRAIN_JOB},
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="worker",
                            image="registry.k8s.io/pause:3.10",
                            resources=client.V1ResourceRequirements(
                                requests={"nvidia.com/gpu": "1"},
                                limits={"nvidia.com/gpu": "1"},
                            ),
                        )
                    ],
                ),
            ),
        ),
    )


def build_deploy(cfg: dict):
    from kubernetes import client
    total = get(cfg, "simulation.total_gpus", 10000)
    init_train = get(cfg, "training.initial_gpu", 8000)
    init = total - init_train
    return client.V1Deployment(
        metadata=client.V1ObjectMeta(name=INFER_DEPLOY, namespace=NS),
        spec=client.V1DeploymentSpec(
            replicas=init,
            selector=client.V1LabelSelector(match_labels={"app": INFER_DEPLOY}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": INFER_DEPLOY}),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="worker",
                            image="registry.k8s.io/pause:3.10",
                            resources=client.V1ResourceRequirements(
                                requests={"nvidia.com/gpu": "1"},
                                limits={"nvidia.com/gpu": "1"},
                            ),
                        )
                    ],
                ),
            ),
        ),
    )


def read_actual(batch, apps):
    """读回真实集群"实际交付"的 GPU 数（读 status，不做全量 pod list——10k 集群上逐 tick
    list 太重）。训练读 Job.status.ready（Ready pod = 真正可服务的卡），推理读
    Deployment.status.availableReplicas。都有值才返回（scale 操作后有天然滞后=诚实反馈）。"""
    js = batch.read_namespaced_job(TRAIN_JOB, NS).status
    ds = apps.read_namespaced_deployment(INFER_DEPLOY, NS).status
    a_train = js.ready if (js and js.ready is not None) else (js.active if js else 0)
    a_infer = ds.available_replicas if ds else 0
    if a_infer is None:
        a_infer = ds.replicas if ds else 0
    return int(a_train or 0), int(a_infer or 0)


def ensure_objects(apps, batch, cfg: dict) -> None:
    """幂等创建：已存在则 patch 到配置初始值。"""
    from kubernetes.client.rest import ApiException

    job = build_job(cfg)
    deploy = build_deploy(cfg)
    try:
        batch.create_namespaced_job(NS, job)
        print(f"[init] created Job {TRAIN_JOB} parallelism={job.spec.parallelism}")
    except ApiException as e:
        if e.status == 409:
            batch.patch_namespaced_job(TRAIN_JOB, NS, {"spec": {"parallelism": job.spec.parallelism}})
            print(f"[init] Job exists, patched parallelism={job.spec.parallelism}")
        else:
            raise
    try:
        apps.create_namespaced_deployment(NS, deploy)
        print(f"[init] created Deployment {INFER_DEPLOY} replicas={deploy.spec.replicas}")
    except ApiException as e:
        if e.status == 409:
            apps.patch_namespaced_deployment(
                INFER_DEPLOY, NS, {"spec": {"replicas": deploy.spec.replicas}}
            )
            print(f"[init] Deployment exists, patched replicas={deploy.spec.replicas}")
        else:
            raise


def wait_initial(batch, apps, cfg: dict, timeout_s: float = 1800.0) -> tuple[int, int]:
    """等训练/推理 pod 拉到【全量初始】并稳定（真实集群需要时间）。

    为什么等满而不是 90%：bring-up 阶段 actual < initial 会被决策引擎误判为"训练被降级"，
    触发 Rule3 归还（启动瞬态，非潮汐行为）。等满 + 连续两次确认稳定后再启动决策循环。
    返回 (actual_train, actual_infer)。
    """
    total = get(cfg, "simulation.total_gpus", 10000)
    want_train = get(cfg, "training.initial_gpu", 8000)
    want_infer = total - want_train
    t0 = time.time()
    stable = False
    while time.time() - t0 < timeout_s:
        a_train, a_infer = read_actual(batch, apps)
        ok_t = a_train >= want_train
        ok_i = a_infer >= want_infer
        print(
            f"[init] waiting running pods: train {a_train}/{want_train}  infer {a_infer}/{want_infer}"
            + ("  OK" if ok_t and ok_i else ""),
            flush=True,
        )
        if ok_t and ok_i:
            if not stable:
                stable = True
                time.sleep(8)
                continue
            a2_train, a2_infer = read_actual(batch, apps)
            if a2_train >= want_train and a2_infer >= want_infer:
                return a_train, a_infer
            stable = False
        time.sleep(10)
    raise RuntimeError(f"initial bring-up timeout: train={a_train}/{want_train} infer={a_infer}/{want_infer}")


def dump_manifests(cfg: dict) -> None:
    import yaml
    from kubernetes import client

    def _clean(obj):
        data = client.ApiClient().sanitize_for_serialization(obj)
        return {k: v for k, v in data.items() if v is not None}

    print("--- 训练弹性 Job ---")
    print(yaml.safe_dump({"apiVersion": "batch/v1", "kind": "Job", "metadata": _clean(build_job(cfg).metadata), "spec": _clean(build_job(cfg).spec)}, sort_keys=False))
    print("--- 推理 Deployment ---")
    print(yaml.safe_dump({"apiVersion": "apps/v1", "kind": "Deployment", "metadata": _clean(build_deploy(cfg).metadata), "spec": _clean(build_deploy(cfg).spec)}, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "integration/l3/l3_default.yaml"))
    ap.add_argument("--kubeconfig", default="/root/.kwok/clusters/l3/kubeconfig.yaml")
    ap.add_argument("--out", default="/opt/l3-controller/run.json")
    ap.add_argument("--ticks", type=int, default=None, help="覆盖 duration_ticks")
    ap.add_argument("--tick-wall", type=float, default=1.0, help="每 tick 墙钟秒数")
    ap.add_argument("--scheduler", default="elastic", choices=["elastic", "predictive"])
    ap.add_argument("--dump-manifests", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ticks = args.ticks or get(cfg, "simulation.duration_ticks", 840)
    tick_seconds = get(cfg, "simulation.tick_seconds", 5)
    seed = get(cfg, "simulation.seed", 42)

    from kubernetes import client, config

    if args.dump_manifests:
        dump_manifests(cfg)
        return

    config.load_kube_config(config_file=args.kubeconfig)
    kc = client.ApiClient()
    apps = client.AppsV1Api(kc)
    batch = client.BatchV1Api(kc)

    scheduler = L3PredictiveScheduler(cfg) if args.scheduler == "predictive" else ElasticScheduler(cfg)
    infer = MockInferenceService(cfg)
    training = MockTrainingJob(cfg)
    traffic = TrafficGenerator(cfg, rng=random.Random(seed))
    scheduler.reset()

    total_gpus = get(cfg, "simulation.total_gpus", 10000)
    min_gpu = get(cfg, "training.min_gpu", 4000)
    init_train = get(cfg, "training.initial_gpu", 8000)
    init_infer = get(cfg, "inference", {}).get("initial_gpu", total_gpus - init_train)

    ensure_objects(apps, batch, cfg)
    actual_train, actual_infer = wait_initial(batch, apps, cfg)
    print(f"[init] initial bring-up complete: train={actual_train} infer={actual_infer}")

    records = []
    gpu_h_train = 0.0
    gpu_h_infer = 0.0
    slo_violated_ticks = 0
    moves_down = 0
    moves_up = 0

    for t in range(ticks):
        now_s = t * tick_seconds
        minute = now_s / 60.0
        qps = traffic.qps_at_second(now_s)

        actual_train, actual_infer = read_actual(batch, apps)

        infer.set_gpus(actual_infer)
        infer.step(qps)
        training.gpus = actual_train

        ctx = SchedulerContext(
            training_gpus=actual_train,
            inference_gpus=actual_infer,
            training_min_gpu=min_gpu,
            training_initial_gpu=init_train,
            inference_p95_ms=infer.p95_ms,
            slo_p95_ms=infer.slo_p95_ms,
            inference_state=infer.state.value,
            training_throughput=training.throughput,
            training_slowdown=training.slowdown_vs_initial,
            training_would_allow_degradation=training.would_allow_degradation(actual_train),
            inference_overload_counter=infer._overload_counter,
            inference_recovery_counter=infer._recovery_counter,
            now_s=now_s,
            runtime=None,
            extra={"would_allow_degradation": training.would_allow_degradation},
        )
        decision = scheduler.step(ctx)

        if decision.changed:
            if decision.new_training_gpu != actual_train:
                batch.patch_namespaced_job(
                    TRAIN_JOB, NS, {"spec": {"parallelism": int(decision.new_training_gpu)}}
                )
            if decision.new_inference_gpu != actual_infer:
                apps.patch_namespaced_deployment(
                    INFER_DEPLOY, NS, {"spec": {"replicas": int(decision.new_inference_gpu)}}
                )
            if decision.new_training_gpu < actual_train:
                moves_down += 1
            else:
                moves_up += 1
            print(
                f"[t={t:4d} min={minute:5.1f}] {decision.reason} | "
                f"qps={qps:8.0f} p95={infer.p95_ms:6.1f} {infer.state.value:10s} | "
                f"train {actual_train}->{decision.new_training_gpu} infer {actual_infer}->{decision.new_inference_gpu}",
                flush=True,
            )

        gpu_h_train += actual_train * tick_seconds / 3600.0
        gpu_h_infer += actual_infer * tick_seconds / 3600.0
        if infer.p95_ms > infer.slo_p95_ms:
            slo_violated_ticks += 1

        records.append(
            {
                "tick": t,
                "minute": round(minute, 2),
                "qps": round(qps, 1),
                "p95_ms": round(infer.p95_ms, 1),
                "state": infer.state.value,
                "overload_cnt": infer._overload_counter,
                "recovery_cnt": infer._recovery_counter,
                "intended_train": int(decision.new_training_gpu),
                "actual_train": int(actual_train),
                "intended_infer": int(decision.new_inference_gpu),
                "actual_infer": int(actual_infer),
                "changed": bool(decision.changed),
                "reason": decision.reason,
            }
        )
        time.sleep(args.tick_wall)

    summary = {
        "config": str(Path(args.config).name),
        "total_gpus": total_gpus,
        "tick_seconds": tick_seconds,
        "ticks": ticks,
        "seed": seed,
        "final_train": actual_train,
        "final_infer": actual_infer,
        "gpu_h_train": round(gpu_h_train, 1),
        "gpu_h_infer": round(gpu_h_infer, 1),
        "gpu_h_total": round(gpu_h_train + gpu_h_infer, 1),
        "slo_violation_ticks": slo_violated_ticks,
        "slo_violation_pct": round(100.0 * slo_violated_ticks / ticks, 1),
        "moves_down": moves_down,
        "moves_up": moves_up,
        "peak_p95_ms": max(r["p95_ms"] for r in records),
        "max_train_transfer": init_train - min(r["actual_train"] for r in records),
        "min_actual_train": min(r["actual_train"] for r in records),
        "max_actual_infer": max(r["actual_infer"] for r in records),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8"
    )

    print("\n=== L3 M2 运行摘要 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] 记录已写: {out_path}")


if __name__ == "__main__":
    main()
