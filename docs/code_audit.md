# Phase 3.3 · T1 — 代码与实验审计报告

> 范围：`src/` 全部 55 个 Python 文件、6 个 YAML 配置、`tests/` 15 个测试文件、
> `docs/`、`README.md`、`PROJECT_SPEC.md`。
> 方法：人工逐文件深读（全部 correctness-critical 模块）+ 机械广度扫描
> （死代码 / 重复实现 / 命名 / 文档-代码不一致 / 测试覆盖 / 配置键使用）。
>
> 处置原则（规格 §37）：**只修影响核心正确性或可复现性的问题**；纯风格 / 死代码
> **记录不重构**——不为了"漂亮"破坏稳定实验，不做大规模重构。

---

## 1. 结论摘要

| 类别 | 数量 | 处置 |
|---|---|---|
| 发现问题 | 40+ | 修复 **6** 项（全部为核心正确性 / 可复现性），其余**记录不修**（附理由） |
| 实验路径代码改动 | 0 | 未触碰任何 Scheduler / Workload / Network / Runtime 逻辑 |
| 回归验证 | 通过 | 135 单测全绿（121 + 新增 14）；Mock 3×4 矩阵双跑 byte-identical（T1 整树 66 文件；T11 干净 results/ 复核 51 文件，见 `docs/phase3_3_t11_reproducibility.md`） |

---

## 2. 已修复问题（核心正确性 / 可复现性）

| # | 位置 | 问题 | 修复 | 理由 |
|---|---|---|---|---|
| F1 | `requirements.txt` | 缺 `torch` 与 `pytest`；README 却指导 `pip install -r requirements.txt` + `pytest tests/` + 跑真实 workload 实验 | 补充 `torch>=2.0`、`pytest>=7.0` | 陌生人按 README 无法安装依赖、无法跑测试——**可复现性断裂**（§36 Code Audit 首要目标） |
| F2 | `README.md` §8.3 | "RuntimeAdapter ABC（**10 个抽象方法**）" 过时 | 改为 "**14 个抽象成员（9 方法 + 5 属性）**" | 文档与代码计数不一致 |
| F3 | `README.md` §8.4 | "metrics_*.csv 全部 **24 列**" 过时 | 改为 "全部 **25 列**"（`CSV_FIELDS` 实计 25） | 同上 |
| F4 | `src/simulator/engine.py` docstring | "统一经 MockRuntimeAdapter 驱动" 过时（Phase 3.2 已支持 Local） | 改为经 RuntimeAdapter 抽象 + `build_runtime(mode)` 分派 | 文档-代码不一致 |
| F5 | `experiments/calibrate.py` | (a) 打印引用了 `calibration.get('gpu_util')`——该键从未产出，恒打印 "(见 R1)"；(b) docstring 声称 local_calibrated.yaml "可直接加载"，但该文件缺 `local.inference/training` 服务与模型形状键 | (a) 移除死引用；(b) 修正 docstring 说明规范入口为 `config/local_experiment.yaml` | 死输出误导 + 过度声明 |
| F6 | `src/workload/traffic.py` | **`TrafficGenerator` 零直接测试**——它是每个实验唯一的输入信号驱动（全部 shipped 配置用 `custom`），此前仅被引擎间接覆盖 | 新增 `tests/test_traffic.py`（14 用例：custom 插值/边界/排序/默认潮汐 + constant/step/sine/spike + 抖动确定性/非负） | 实验输入的正确性是核心正确性；纯新增测试，零行为改动 |

> **修复纪律**：6 项修复中，F1–F5 全部是文档/依赖/打印层面的修改，F6 是纯新增测试——
> 无一触碰实验执行路径。这是"审计修复"而非"重构"。

---

## 3. 记录不修问题（附理由）

### 3.1 死代码（write-only / 未使用）

| # | 位置 | 内容 | 不修理由 |
|---|---|---|---|
| R1 | `src/network/network.py:18` | `from src.config import get` 未使用 | 纯美学，删除无证据价值 |
| R2 | `src/workload/inference.py:12` | `import math` 未使用 | 同上 |
| R3 | `src/workload/traffic.py:10` | `from typing import Callable` 未使用 | 同上 |
| R4 | `src/simulator/engine.py:22` | `load_config` 导入未使用 | 同上 |
| R5 | `src/runtime/base.py:16` | `field` 导入未使用 | 同上 |
| R6 | `src/workload/training.py` | `_was_degraded` 赋值两次从未读取 | 同上 |
| R7 | `src/cluster/resource_manager.py` | `ResourceState` 类 + `state` 属性 + `unused_gpus` 属性从未使用 | 同上（`unused_gpus` 是旧实现遗留，已被 `ClusterState.free_gpu` 取代，见 README §8.4） |
| R8 | `src/metrics/collector.py` | `rows_as_lists()` 从未调用 | 同上 |
| R9 | `src/network/netem.py` | 模块级 `_netem_delay_ms()` 从未调用 | 同上 |
| R10 | `src/simulator/clock.py` | `now_min` 属性从未使用 | 同上 |
| R11 | `src/scheduler/base.py` | `SchedulerContext` 5 个字段（training_throughput / training_slowdown / training_would_allow_degradation / inference_overload_counter / inference_recovery_counter）被 engine 填充但无调度器读取 | 调度器经 `ctx.extra` 访问降速保护；这些字段是冗余接口。删除属接口变更，风险 > 收益 |
| R12 | `src/simulator/engine.py` | `build_runtime` 的 `rng`/`seed` 参数未使用（adapter 内部硬编码 42） | 删除参数会改接口签名；且"显式种子"是确定性纪律的一部分，保留参数无害 |
| R13 | `src/workload/traffic.py` | constant/step/sine/spike 分支实际未使用（全配置走 custom） | 保留（spec §9 要求支持 5 种类型）；F6 已补测试覆盖 |
| R14 | 全部 6 个 YAML | `scheduler.interval_ticks` / `scheduler.cooldown_ticks` 从未被代码读取 | 防振荡由 `min_gpu_hold_duration_ticks` 落实；这两个键是 spec 遗留。删除会改变配置契约，记录即可 |

### 3.2 重复实现（不合并，理由：合并=重构，风险覆盖稳定实验）

| # | 位置 | 内容 | 不修理由 |
|---|---|---|---|
| R15 | `inference.py:_compute_latency` / `inference_real.py:_emulated_latency` / `test_local_runtime.py:StubInference` | 同一 P95 延迟模型三处实现 | Mock 与 Local 有意各自独立（真实路径以后可演进）；合并会改动已校准行为 |
| R16 | `training.py:throughput_at` / `training_real.py:scaling_at` / `StubTraining` | 吞吐/缩放公式三处实现 | 同上；`training_real` 的 EMULATED 缩放语义（Gap 2）与 Mock 不同，强制统一会掩盖 Reality Gap |
| R17 | `training.py`+`training_real.py`+两个 runtime | `slowdown_vs_initial` / `would_allow_degradation` 同逻辑 ~5 处 | 纯风格 |
| R18 | `training.py`/`training_real.py`/`StubTraining` | `remaining_work`/`estimated_completion_time_s` 同实现 | 纯风格 |
| R19 | `network.py` vs `test_local_runtime.py:StubNetwork` | `expansion_feasible` 符号相反但数学等价（`net=delta−released` 判定 `net≤headroom` vs `net=released−added` 判定 `net≥−headroom`） | 等价且仅存在于测试桩；改符号会重写测试，无证据价值。**已记录为风险**：未来若两处漂移会产生隐性 bug |
| R20 | `experiments/run.py` vs `run_local.py` | `SUMMARY_FIELDS`/`_run_one`/`_print_table` 近重复 | 合并需改实验入口，属重构；记录 |
| R21 | `engine.py` / `test_runtime.py` / `test_local_runtime.py` | SchedulerContext 组装 ~20 行三处重复 | 纯风格 |

### 3.3 命名不一致（记录，不统一：统一=大规模改动）

| # | 不一致 |
|---|---|
| R22 | P95 延迟 5 个名字：`InferenceState.p95` / workload `p95_ms` / `SchedulerContext.inference_p95_ms` / `ResourceDecision.inference_p95` / `TickMetrics.inference_p95` |
| R23 | `p95_base_ms`（Mock 侧）vs `base_p95_ms`（Local 侧）；`p95_overload_ms` vs `overload_p95_ms` |
| R24 | `training_gpu`（CSV）vs `training_gpus`（ctx/network）vs `allocated_training_gpu`（ClusterState） |
| R25 | 拥塞延迟 3 名字：`latency_ms` / `congestion_latency_ms` / `network_latency_ms`；`congested` vs `network_congested` |
| R26 | 带宽容量 3 名字：`capacity_mbps` / `total_bandwidth_mbps` / `network_capacity`；headroom 3 名字 |
| R27 | 防抖计数器：`_overload_counter` / `_oc` / `inference_overload_counter` |
| R28 | min_hold：attr `min_hold_ticks` / config `min_gpu_hold_duration_ticks` / reason "min_gpu_hold" |
| R29 | 训练带宽：`training_bw` / `training_bandwidth` / `training_bandwidth_mbps`；`penalty_per_unit` vs `training_penalty_per_unit`；`preempt_floor` vs `preempt_min_training` |

### 3.4 文档-代码不一致（已修 F2/F4/F5，其余记录）

| # | 内容 | 处置 |
|---|---|---|
| R30 | `metrics/exporter.py` 只出 **5/7** 张图（缺 GPU utilization 图与调度决策时间线图），未达 PROJECT_SPEC §22 | 记录。README 已如实写"5 张核心图"；补 2 张图属新功能，违反"close, don't expand"。证据价值已被 metrics/decisions CSV 覆盖 |
| R31 | `gpu_utilization` 指标 = `(training+inference)/total`，由守恒律**恒为 1.0**（分配率恒等式，非利用率度量） | 记录。README §3.1 已如实写"1.0"；改名/改语义会破坏 metrics CSV 兼容与 byte-identical，**不修**，仅在此与最终报告如实标注 |
| R32 | PROJECT_SPEC §24 目录树列出的 `src/network/congestion.py`、`run_static.py` 等 5 个文件不存在；§8/§13 参数（带宽 10000、latency 1ms 等）与实际配置不符 | spec 是历史规格存档（docx 转档），不改历史文档；实际以代码 + 本报告为准 |
| R33 | `config/local_calibrated.yaml` 的 `local` 段缺服务/模型形状键（host/port/input_dim/hidden_dim/layers 等），加载时静默回退代码默认值 | 已修 docstring 过度声明（F5b）；管线上限不扩。规范入口 `local_experiment.yaml` 的 local 段完整（host/port/dims 齐全），本地实验均由此加载 |
| R34 | 配置访问 3 种风格并存（dotted `get()` / `cfg.get("x",{}).get("y")` / 混合） | 风格不统一，记录。这正是 R14 死键未被发现的原因；统一访问层属重构 |

### 3.5 测试覆盖缺口（记录 + 已补关键缺口 F6）

| # | 缺口 | 处置 |
|---|---|---|
| R35 | `src/config.py`（load_config/get）无直接测试 | 记录；被引擎间接覆盖 |
| R36 | `src/metrics/collector.py` / `exporter.py` 无直接测试（仅引擎副作用覆盖） | 记录；collector 逻辑经 135 测试 + Mock 双跑 byte-identical 双重验证 |
| R37 | `run_experiment` mode="local" 仅 2-tick 冒烟（test_unified_engine） | 记录；完整 Local 4 策略循环由 T2 全潮汐实验覆盖（见后续报告） |
| R38 | `network_aware._gated_noop` 的 utilization 传播未断言 | 记录；已由 cross-runtime 决策一致性（reality_gap §3）间接验证 |

---

## 4. 实验审计（数据诚实性核查）

| 检查项 | 结果 |
|---|---|
| 是否修改原始数据 / 隐藏失败实验 / 调参迎合 Mock？ | **否**。`results/` gitignored（.gitignore 第 3 行）；本阶段对任何实验结果文件零改动 |
| 是否修改 Scheduler 让结果更漂亮（§3.1 最高优先级）？ | **否**。6 项修复无一触碰 scheduler/workload/network/runtime 逻辑（§2） |
| Mock 确定性 | **验证通过**：`python -m experiments.run --matrix` 连续两次运行，结果文件 md5 **全部一致**（本轮实测含早期输出整树 `baseline=66 rerun=66 differing=0`；T11 基线态从干净 results/ 重跑为 51 文件全同，口径说明见 `docs/phase3_3_t11_reproducibility.md` §2） |
| 测试回归 | **通过**：审计前后 121 → 135（新增 14 个 traffic 测试，纯新增） |
| EMULATED/REAL 标注 | 保持：`src/calibration/calibrator.py` 的 REAL/EMULATED/DESIGN 出处链未动；`docs/reality_gap.md` Gap 2（训练代价量纲）继续如实标注"EMULATED 多卡缩放是假设非测量" |

---

## 5. 后续报告引用方式

最终技术评审（§36 T9）的 **Code Audit** 一节将直接引用本文件：已修 6 项（含理由）、
记录不修 30+ 项（含"不修理由 = §37 close-don't-expand / 不破坏稳定实验 / 纯美学"分类）。
任何读者可据此区分：**什么是 bug 已修、什么是有意保留的工程取舍**。
