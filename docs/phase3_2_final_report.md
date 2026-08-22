# Phase 3.2 最终报告 — 本地运行时 × 真实 Workload × 网络现实

> **目标**：在真实 workload（PyTorch 训练 + HTTP 推理 + 网络受限）上验证 Mock 阶段的三项关键发现
> （Finding A / B / C），量化 Mock 假设与真实观测之间的 **Reality Gap**。
> **完整 Gap 分析** → [`docs/reality_gap.md`](reality_gap.md)

---

## 1. 本阶段完成的工作（Steps 1-13）

| Step | 交付 | 产物 |
|---|---|---|
| 1-2 | Runtime Contract 复用 + LocalRuntimeAdapter 骨架 | `src/runtime/local.py` |
| 3-4 | 真实 PyTorch 训练 + R1 单卡校准 | `src/workload/training_real.py`、`results/profiles/training_profile.json` |
| 5-6 | 真实 HTTP 推理 + R2 校准（含 e2e 延迟语义修正） | `src/workload/inference_real.py`、`inference_profile.json` |
| 7-8 | tc/netem 网络控制器 + R3 真实整形测量 | `src/network/netem.py`、`network_profile.json` |
| 9 | 校准数据管线（T1 REAL × g^0.85 EMULATED 显式数据工件） | `src/calibration/`、`config/local_calibrated.yaml` |
| 10 | Mock vs Local 统一实验（4 策略，同一 tick 循环） | `experiments/run_local.py --compare`、`results/local/mock_vs_local.json` |
| 11-12 | Reality Gap 报告 | `docs/reality_gap.md` |
| 13 | 本报告 + README 更新 | `docs/phase3_2_final_report.md` |

## 2. 如何复现

```bash
# 全量单测（121 个，含 Step 10 统一引擎）
python -m pytest tests/ -v

# Mock vs Local 统一对比（Local 4 策略真实运行 ~12min + Mock 4 策略）
python -m experiments.run_local --compare

# Mock 3×4 矩阵回归（证明阶段一/二结果未被破坏）
python -m experiments.run --matrix

# 校准管线（R1/R2/R3 → local_calibrated.yaml + profiles）
python -m experiments.profile_training && python -m experiments.profile_inference \
  && python -m experiments.profile_network && python -m experiments.calibrate
```

## 3. 结果总表（Mock vs Local，网络受限潮汐，种子 42）

| 策略 | runtime | SLO违率 | 平均P95(ms) | 切换 | 训练进度 | 降速 | 网利用率均 |
|---|---|---|---|---|---|---|---|
| static | Mock | 100% | 506 | 0 | 35104 | 1.00 | 1.48 |
| static | Local | 100% | 336 | 0 | 14383616 | 1.00 | 1.82 |
| network_aware | Mock | 100% | 506 | 0 | 35104 | 1.00 | 1.48 |
| network_aware | Local | 100% | 336 | 0 | 13785856 | 1.00 | 1.82 |
| elastic | Mock | 100% | 525 | 2 | 16633 | 1.79 | 1.93 |
| elastic | Local | 100% | 433 | 2 | 13987584 | 1.67 | 2.20 |
| hard_preemption | Mock | 100% | 581 | 1 | 8262 | 3.24 | 2.16 |
| hard_preemption | Local | 100% | 497 | 1 | 14599936 | 3.17 | 2.46 |

**决策逐条一致**：elastic 两套都是 4→3→2（Rule1 连抢）、hard 都是 4→1 一刀切、network_aware/static 都零切换。

## 4. 三项发现验证结论

- **Finding A（硬抢占过度反应）** ✅：hard_preemption 一次性让渡到 floor，P95 反而最差（581/497ms），SLO 100% 违约，训练代价最重（EMULATED 降速 3.24/3.17）。"快"无收益。
- **Finding B（GPU 空闲 ≠ 系统可扩展）** ✅：GPU util 0.75（推理容量 2253 vs 峰值 1700 qps）但网络已饱和；elastic/hard 扩推理进网络 → 利用率与 P95 双双上升，network_aware 拦截 → P95 最优。
- **Finding C（网络扩容闸门边界）** ✅：紧网络 + 平训练曲线（释放 100 Mbps ≪ 增 600 Mbps）→ 闸门全拦 → network_aware == static；松网络 + 陡曲线 → 闸门放行 → network_aware == elastic。两极端行为都正确。

## 5. Reality Gap 摘要（详见 `reality_gap.md`）

| # | Gap | 量级 | 性质 |
|---|---|---|---|
| 1 | 推理延迟基准假设 | 200ms(Mock 假设) vs 5.27ms(R2 实测) = **38×** | 假设被真实测量推翻 |
| 2 | **训练代价量纲** | Mock 进度损失 53% vs Local 3% | **最重要**：单物理 GPU 上虚拟配额不改变真实工作量；EMULATED 多卡缩放是假设非测量 |
| 3 | GPU 触发信号 | Local 的 GPU 延迟带(5-15ms) 带太窄，GPU 永不触发 SLO | Local 只覆盖 network-bound 触发 |
| 4 | 网络整形真实性 | 实验循环用 EMULATED 延迟模型；真实 netem 仅 R3 交叉验证(10ms→RTT 20.8ms) | LOCAL-EMULATED-LATENCY 标注 |
| 5 | 利用率绝对级差 | 1.48 vs 1.82 | config 参数不同，趋势一致 |

## 6. Failure cases（诚实口径，spec §29.7）

1. **Elastic 不如 Hard Preemption？** — 未构造出。A/B/C 全场景 elastic 均占优或持平；hard 仅理论优势（单步恢复 SLO）在网络受限时无收益，被 strictly dominated。
2. **Network-aware 不如 GPU-only？** — A/B 两极端均未更差。潜在风险：陡曲线+中拥塞时闸门可能过度保守（未验证）。
3. **何时资源震荡？** — 本实验无病态震荡（网络持续拥塞 → 信号单向 → 单调抽取）；风险在 SLO 边界摆动场景，靠防抖 + min_hold 兜底。
4. **何时 SLO 仍违约？** — **所有网络受限运行、所有策略，SLO 100% 违约**。结构性拥塞无法靠移动 GPU 修复——瓶颈是网络不是 GPU。GPU-only 调度器此时唯一正确的动作是 network_aware 那样**别动**。

## 7. 诚实边界与未做事项（红线）

- 未触碰 K8s / 多节点 / RDMA / RoCE / PFC / InfiniBand / 真实多机 distributed / GPU hot migration / 生产级 controller（spec §20 红线）。
- 未解决 Dynamic GPU Hot Resize（§21）。
- 本机只有 1 张物理 GPU：**所有多 GPU 结果（g>1）为 EMULATED 缩放**，T1 与延迟为 REAL 实测——不冒充真实多卡数据（§22）。
- 为保 Mock 可比性未修改任何 Scheduler；一切 Gap 只记录、分析，不"修漂亮"（§3.1 最高优先级）。

## 8. 给成品（阶段二正式 Network-aware 实现）的启示

网络受限时"正确调度"首先是**不动作**：停止 GPU-only 的盲目扩入。network_aware 闸门在网络受限场景的价值已被 Mock 与 Local 两套 runtime 共同证明；下一步（真实多卡/生产级）应在此基础上叠加 Scenario C（双瓶颈）与参数自适应扫描。
