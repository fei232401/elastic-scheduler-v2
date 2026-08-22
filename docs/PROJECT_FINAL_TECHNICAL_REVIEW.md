# 最终技术评审（PROJECT FINAL TECHNICAL REVIEW）

> 潮汐流量训推过渡调度器（Elastic Training-Inference Scheduler）
> 全项目（Phase 1 → 3.3）最终技术评审。面向"陌生工程师"：独立可读，无需读过其他文档。
> 证据等级约定见 [`docs/evidence_matrix.md`](evidence_matrix.md)（A 实测 / B runtime / C Mock / D 未验证）。

---

## 1. Problem（问题）

在线服务面临**潮汐流量**：夜间训练任务占满 GPU、白天推理流量上升、高峰过去又回落。
固定配额的两难：
- 训练/推理各占一半 → 高峰推理排队（SLO 违约）、低谷训练吃不满；
- 给推理留足 → 训练永远少一块。

需要一个调度器在**同一 GPU 池**里按流量动态过渡配额：高峰把训练配额让给推理（保障推理 SLO），
流量回落后再还回来（不饿死训练）。难点：让渡必须**逐步、可逆、且感知真正的瓶颈**
（GPU 不够 vs 网络不够是两种完全不同的病）。

## 2. Hypothesis（假设）

1. **Rule-based 弹性调度**（防抖 + 逐级 ±1 + min_hold）能在潮汐高峰保护推理 SLO，并在低谷完整归还
   训练配额——代价可控、不振荡。
2. **硬抢占（一次性让渡到底）是坏的弹性策略**（过度反应，代价大、换不回 SLO）。
3. **GPU 有空闲 ≠ 系统可扩展**：网络受限时，GPU-only 调度器会把卡挪进网络、加剧拥塞；
   正确的动作是"不动作"。需要一个**网络感知闸门**（扩容可行 iff 净增带宽 ≤ 余量）。

## 3. Scope（范围）

- **已实现**：Mock 集群（GPU Pool / 训练幂律吞吐 / 推理非线性延迟 / 潮汐流量 / 确定性时钟）、
  4 个调度策略（static / elastic / hard_preemption / network_aware）、统一 Runtime 抽象
  （Mock 数学模型 ↔ Local 真实 workload 同一契约）、真实 PyTorch 训练 + 真实 HTTP 推理 +
  tc/netem 网络整形、校准管线（REAL/EMULATED/DESIGN 出处）、指标采集 + 5 图、
  121→135 测试、3 实验 × 4 策略矩阵、全潮汐/网络严重度/参数扫描三套实验、全套诚实文档。
- **环境**：WSL2 + 单张 RTX 4070 8GB；Python 3.14；numpy/matplotlib/PyYAML/torch/pytest。

## 4. Non-goals（非目标，规格红线 §20/§21）

明确**不做**（本阶段不可达 / 明确排除）：Kubernetes / 多节点 / RDMA / RoCE / PFC / InfiniBand /
真实多机 distributed training / GPU 热迁移 / 生产级 controller / Dynamic GPU Hot Resize。
**多 GPU 物理验证不可达**（本机 1 张 GPU）——见 §12/§13。

## 5. Architecture（架构）

见 [`docs/architecture.md`](architecture.md)（mermaid + ASCII）。一句话：
**输入（潮汐）→ 调度（4 策略）→ RuntimeAdapter 统一契约 →（Mock | Local）资源 → 指标/证据**。
核心分层：Scheduler 只通过 `SchedulerContext` + `ResourceDecision` 与 runtime 交互，
不直接持有任何资源对象——这让同一套调度代码在 Mock 与真实 workload 上逐条产生相同决策。

## 6. Scheduler（调度器）

| 策略 | 规则 | 行为特征 |
|---|---|---|
| static | 恒 6/2（或配置固定） | 基线；高峰不保护 SLO |
| elastic | Rule1 SLO 违约（防抖确认）→ 让 1 卡；Rule2 接近 SLO 预警；Rule3 健康 → 还 1 卡；Rule4 min_hold 防抖；Rule5 每次 ±1 | 逐级、可逆、有 min_hold |
| hard_preemption | 过载确认 → 一次让渡到底（floor）；健康 → 全量归还 | "快"但过冲 |
| network_aware | = elastic + §17 闸门：扩容 iff `新增带宽−释放带宽 ≤ headroom` | 网络受限时**不动作** |

决策经 `SchedulerContext`（决策前快照）→ `ResourceDecision`（含 reason/timestamp，可回放）。

## 7. Runtime（运行时抽象）

- `RuntimeAdapter`（ABC）：`Cluster/Training/Inference/Network` 四状态视图 +
  `apply_resource_decision` + `network_expansion_feasible`；14 抽象成员。
- `MockRuntimeAdapter`：包装数学组件（毫秒级，确定性 seed 42）。
- `LocalRuntimeAdapter`：真实 wall-clock——真实 HTTP 推理、真实 PyTorch 训练、netem 网络；
  多卡为虚拟配额 × EMULATED 缩放。
- `build_runtime(mode)` 按配置分派；engine 的 tick 循环对两种 runtime 完全相同。

## 8. Workload（工作负载）

| 组件 | Mock（C） | Local（真实） |
|---|---|---|
| 训练 | `throughput = base×g^0.85`；progress += 幂律吞吐 | 真实 PyTorch MLP；progress += 真实样本（A）；上报吞吐 = T1×scale(g^0.85)（A/D） |
| 推理 | base 200ms / SLO 300ms 非线性延迟 + 防抖 | 真实 HTTP（A：base 5.27ms / capacity 563 qps / SLO 15ms） |
| 流量 | custom 分段线性潮汐（10→60→10） | 压缩潮汐（75s） |
| 网络 | util = demand/capacity；拥塞延迟 EMULATED（C） | 循环 EMULATED（C）+ R3 真实 netem 交叉验证（A） |

## 9. Experiments（实验谱系）

| 阶段 | 实验 | 入口 | 产出 |
|---|---|---|---|
| Phase 1 | 首版 Static vs Elastic（GPU-only） | `experiments/run.py` | SLO 49.4%→1.3% 核心价值 |
| Phase 2 | 3 实验 × 4 策略矩阵（A GPU-bound / B 网络受限 / C 双瓶颈） | `--matrix` | 三项 Finding A/B/C |
| Phase 3.1 | Runtime 抽象重构回归 | 全测试 + 矩阵 | 12 组 summary byte-identical |
| Phase 3.2 | Mock vs Local 统一对比（真实 workload） | `run_local --compare` | 决策跨 runtime 一致 + Reality Gap |
| Phase 3.3 T2 | 全潮汐周期（70min，4 策略，Mock + Local） | `run.py --all` | 让渡-归还闭环（表 + 决策时间线） |
| Phase 3.3 T3 | 网络严重度轴（Mild/Moderate/Severe） | `--matrix --config` | 过渡区自我破坏发现 |
| Phase 3.3 T4 | 参数扫描（6 参数 → 振荡/稳定/过度保守） | `scan_parameters.py` | 区域图 |
| Phase 3.3 T1 | 代码审计 + 修复 + 回归 | pytest + 双跑 diff | 135 测试 + 51 文件 byte-identical |
| Phase 3.3 T11 | 基线态回归 + 复现性复核 | pytest + 干净 results/ 双跑 diff + Local 双跑 | 135 测试绿 + 51 文件全同 + Local 噪声 0.2% |

**确定性**：Mock 全部 seed 42；干净 results/ 双跑矩阵 51 文件 md5 全同（A，T11 复核；
T1 审计对含早期输出的整树 diff 为 66，口径说明见 `docs/phase3_3_t11_reproducibility.md` §2）。
**Local 噪声**：真实测量有 <5% 噪声（T11 双跑实测进度差 0.203%），已文档化（MTG2）。

## 10. Results（关键结果）

### 全潮汐（GPU-bound，70min，Mock）：弹性闭环成立

| 策略 | SLO违率 | 平均P95 | 切换 | 训练进度 | 降速 |
|---|---|---|---|---|---|
| static | 49.8% | 491ms | 0 | 57783 | 1.00 |
| **elastic** | **1.4%** | **207ms** | **6** | 46285 | 1.33 |
| hard_preemption | 10.2% | 240ms | 8 | 39871 | 2.42 |
| network_aware | 1.4% | 207ms | 6 | 46285 | 1.33 |

- elastic 让渡 6→5→4→3（峰前 3 次）、归还 3→4→5→6（峰后 3 次），**完整闭环、零振荡**；
  SLO 违约 49.8%→1.4%，峰值 P95 902→336ms；**代价 = 训练进度 -20%**（显式、无免费午餐）。
- hard_preemption 4 次 6↔1 ping-pong，SLO 10.2%（elastic 的 7×），训练更贵——"快"无收益。

### 网络严重度轴（exp B workload，只变带宽）：过渡区新发现

| 严重度 | 拥塞比例 | 最优策略 | 关键数字 |
|---|---|---|---|
| Mild（30000） | 0% | elastic（= network_aware） | SLO 0.6%，无回归 |
| **Moderate（3200）** | 48% | **network_aware（不动作）** | elastic/hard 自我破坏：**SLO 91% vs static 60.7%**，拥塞 405→755 |
| Severe（2200） | 100% | network_aware（不动作） | 全策略 SLO 100%违约；elastic/hard 白牺牲训练 |

**Moderate 是最重要的新发现**：局部拥塞时 GPU-only elastic 把网络延迟误判为"缺 GPU"，扩容推理
→ +1200 Mbps 需求 → 拥塞扩散 → 比"不动"更差 30pp。network_aware 闸门正确拦截。

### 参数扫描（6 参数）：稳定区/振荡区/过度保守区

| 参数 | 振荡区 | 稳定区 | 过度保守区 |
|---|---|---|---|
| SLO 阈值 | ≥400（滞后→thrash，10 切换） | **250–300** | ≤200 |
| min_hold | — | 0–48（本场景不敏感） | ≥96 |
| scale_step | ≥4（=温和硬抢占） | 1–3 | — |
| congestion_start | — | 全范围（network_aware 恒 0 切换） | — |
| max_slowdown | — | 2–3 | ≤1.2（网络受限下反而最优） |
| ramp | — | 全斜率 | — |

**根因**：振荡 = 反应滞后 + 过冲（与 hard_preemption 同源），不是防抖不够。

### 跨 runtime 验证（最重要）
elastic 4→3→2 / hard 4→1 / network_aware 0 切换，Mock 与 Local **逐条一致**（B）。
Scheduler 行为与 runtime 无关——Finding A/B/C 不是 Mock 模型的偶然产物。

## 11. Failure Cases（失败案例）

14 例完整目录见 [`docs/failure_cases.md`](failure_cases.md)。要点：
- 网络结构性拥塞下 SLO 全策略违约（物理边界，不可修）；
- 硬抢占过冲 / 过渡区自我破坏 / 参数振荡（调度边界，已定位）；
- Mock 延迟假设证伪 / 训练代价不可观测 / GPU 触发不可构造（Gap，如实标 D）；
- 工程事故 4 例已修（含 run.py 静默跑错 Local 配置）。

## 12. Reality Gap（现实差距）

五类完整分析见 [`docs/reality_gap.md`](reality_gap.md)。最深的三条：
1. **RG1 训练代价不可观测（D）**：单物理 GPU 上虚拟配额不改变真实工作量——Local 上报降速是
   EMULATED 比率，真实训练从未变慢。
2. **MG1 延迟假设被证伪**：Mock 200ms vs R2 实测 5.27ms（38×）。
3. **NG1 网络延迟 EMULATED**：循环用模型值，真实 netem 仅 R3 交叉验证。
**A:D = 4:6**——项目核心的"真实多卡能力"声明全部是 D。

## 13. Limitations（局限性，诚实口径）

- **单 GPU 环境**：无法验证多 GPU 下真实 GPU 资源让渡对分布式训练吞吐的物理影响（D）。
  **所有 g>1 结果 = EMULATED 缩放**，绝不伪装成真实多卡数据（§7 最高纪律）。
- **Local 实验为压缩潮汐**（75s，非 70min）；GPU 触发路径 Local 不可构造。
- **过渡区发现（Moderate）仅 Mock 验证**，未跨 runtime。
- **网络拥塞延迟为模型值**（真实 netem 仅探针）。
- **指标局限**：`gpu_utilization` 恒 1.0（分配率恒等式，非利用率）；5/7 张图（spec §22 缺 2 张，
  证据已被 CSV 覆盖）。

## 14. Conclusions（结论：证明了什么 / 没证明什么）

**已证明（证据等级明确）**：
1. Rule-based 弹性调度在 GPU-bound 潮汐下**保护 SLO 且可逆闭环**（C 数值 + A 确定性），
   显式代价训练 -20%。
2. 硬抢占被**严格占优**（全场景未找到反例）。
3. GPU 空闲 ≠ 系统可扩展；网络感知闸门在网络受限时是必要的（B 决策 + C 数值）。
4. **Scheduler 行为 runtime 无关**——同一决策逻辑在纯模型与真实 workload 上逐条一致（B）。
5. 在**过渡区（部分拥塞）GPU-only 会自我破坏**，闸门价值在此最大（C）。

**未证明（明确标注 D）**：
1. **多 GPU 下真实资源让渡对训练吞吐的物理影响**——单卡环境物理不可达；
2. **训练抢占的真实代价**——单卡上不可观测；
3. **闸门在真实多机/生产网络的边界行为**。

**诚实判断**：本项目证明了**调度决策逻辑的正确性**（可复现、跨 runtime、完整闭环），
但**没有证明**多 GPU 物理性能——两者必须分开表述，任何将 EMULATED 写作 REAL 的说法都是违规。

## 15. Future Work（未来工作，仅描述不实现）

**Phase 4 方向**（本轮红线不触碰，仅规划）：
1. **K8s RuntimeAdapter**：把 ResourceDecision 映射到真实 K8s 调度（训练 Job 缩容/推理扩 Pod），
   跨节点（需多机 + 共享存储 + NCCL/RDMA 场景）。
2. **多 GPU 物理验证**：≥2 卡真机测量 g^0.85 缩放假设 → 把 MG2 从 D 升级为 A/B。
3. **参数自适应**：把 T4 的扫描区域做成在线自适应（SLO 阈值/降速上限随流量学习）。
4. **Scenario C 双瓶颈细化** + 闸门过度保守风险的定向触发实验。
5. **指标补全**：GPU 利用率（真实 `torch.cuda` 采样，非分配率）、调度决策时间线图。
6. **生产化**：控制器/冲突解决/多租户/可观测性（Prometheus 语义）。

> **停止条件**：本项目按规格 §35 于 12 项交付完成即停止，不自行开发 Phase 4。
