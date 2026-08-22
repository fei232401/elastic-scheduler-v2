# Reality Gap 报告（Phase 3.3 T7 重构版）

> 潮汐流量训推过渡调度器 —— 在真实 workload（PyTorch 训练 + HTTP 推理 + 网络受限）上验证
> Mock 阶段三项关键发现（Finding A / B / C），并把 Mock 假设与真实观测之间的差距按
> **五类 Reality Gap** 系统化：**Model / Resource / Runtime / Network / Measurement**。
>
> 实验入口：`python -m experiments.run_local --compare` → `results/local/mock_vs_local.json`
> 数据标签：**REAL**（本机真实测量）/ **EMULATED**（单卡实测 × 虚拟多卡缩放假设）/
> **MOCK**（纯数学模型）/ **DESIGN**（规格设计参数）。
> 每条 Gap 标注证据等级（A 实测 / B runtime / C Mock / D 未验证，见
> [`docs/evidence_matrix.md`](evidence_matrix.md)）。

---

## 1. 实验口径

同一 Runtime Contract、同一 tick 循环、同一 4 策略（static / elastic / hard_preemption /
network_aware）、同一网络受限潮汐形态（镜像 Mock Experiment B），分别在两种 runtime 上跑：

| 维度 | Mock runtime | Local runtime |
|---|---|---|
| 推进方式 | 纯数学模型，毫秒级 | 真实 wall-clock tick（tick=2.5s，30 tick） |
| 推理 | Mock 容量/延迟模型 | 真实 HTTP server + 真实请求（REAL） |
| 训练 | Mock 幂律吞吐 | 真实 PyTorch MLP（1 张物理 GPU，REAL） |
| 多 GPU | 数学模型 | 虚拟配额 × EMULATED 缩放（g^0.85） |
| 网络延迟 | EMULATED 拥塞公式 | EMULATED 拥塞公式（R3 已用真实 netem 交叉验证） |
| tick 数 | 840（70min 虚拟） | 30（75s 真实，压缩潮汐） |

**诚实标注**：Local 实验循环的拥塞延迟是 EMULATED 模型值（与 Mock 同一公式），
真实 tc/netem 整形仅在 R3 探针中施加并单独验证（netem 10ms → 真实 RTT 20.8ms）。

---

## 2. 结果总表

（字段：SLO 违约率 / 平均 P95(ms) / 资源切换次数 / 训练进度 / 训练平均降速 / 网络平均利用率 / 拥塞 tick）

**Mock（exp B，840 tick）**

| 策略 | SLO违率 | P95均 | 切换 | 进度 | 降速 | 网利用率均 | 拥塞tick |
|---|---|---|---|---|---|---|---|
| static | 100% | 506 | 0 | 35104 | 1.00 | 1.48 | 840 |
| network_aware | 100% | 506 | 0 | 35104 | 1.00 | 1.48 | 840 |
| elastic | 100% | 525 | 2 | 16633 | 1.79 | 1.93 | 840 |
| hard_preemption | 100% | 581 | 1 | 8262 | 3.24 | 2.16 | 840 |

**Local（30 tick，真实 workload）**

| 策略 | SLO违率 | P95均 | 切换 | 进度 | 降速 | 网利用率均 | 拥塞tick |
|---|---|---|---|---|---|---|---|
| static | 100% | 336 | 0 | 13982976 | 1.00 | 1.82 | 30 |
| network_aware | 100% | 336 | 0 | 14115840 | 1.00 | 1.82 | 30 |
| elastic | 100% | 433 | 2 | 14754560 | 1.67 | 2.20 | 30 |
| hard_preemption | 100% | 497 | 1 | 14065152 | 3.17 | 2.46 | 30 |

> 训练进度量纲不同（Mock 虚拟工作量 10^4 / Local 真实样本 10^8），不直接比绝对值。
> 硬抢占 Local 进度（14.07M）与 static（13.98M）的 ~0.6% 差异是真实测量噪声（见 Gap 5）。

---

## 3. 决策一致性（最硬的验证）

两套 runtime 的调度决策**逐条一致**：

| 策略 | Mock | Local |
|---|---|---|
| elastic | t=10s 训练 4→3（Rule1 过载）→ t=130s 3→2（Rule1，DEGRADED） | t=2.5s 4→3 → t=17.5s 3→2 |
| hard_preemption | t=10s 训练 4→1 一刀切（推理 4→7） | t=2.5s 4→1 一刀切 |
| network_aware | 0 切换（闸门拦截，headroom 0） | 0 切换（闸门拦截） |
| static | 0 切换 | 0 切换 |

→ **Scheduler 行为与 runtime 无关**（证据等级 B）。Runtime Contract 兑现，三项 Finding 的
结论是 runtime 无关的，而非 Mock 模型的偶然产物。**但**（见 Gap 2/3/6）决策一致≠代价一致：
Local 的决策在真实训练上不产生可测量的代价。

---

## 4. 五类 Reality Gap

### 4.1 Model Gaps（模型假设 vs 真实）

**MG1 — 推理延迟基准假设被证伪（38×）** | 等级：D（假设被 A 证伪）
- Mock 假设 base 200ms（spec §8，DESIGN）→ R2 实测 **5.27ms**（REAL）→ **38×** 差距。
- SLO 也从 300ms（DESIGN）校准为 15ms（20×）。
- 影响：Mock 与 Local 的 P95 **绝对量级不可跨 config 比**（506 vs 336ms）；只有趋势/排序/拐点可比。
- 归属：Model——spec 的延迟参数是设计假设，不是测量。

**MG2 — 多卡吞吐缩放 g^0.85 是假设非测量** | 等级：D
- Mock 与 Local 都假定 `throughput ∝ g^0.85`（spec §5 修正公式）；单卡点 T1 为 REAL（A），
  **g>1 的形状无任何物理测量支撑**（本机 1 张物理 GPU）。
- 影响：所有"多卡性能数字"（全潮汐 -20%、严重度表、参数扫描）都是 EMULATED 缩放下的 Mock 输出。
- 归属：Model——缩放形状是模型假设。

**MG3 — 拥塞延迟公式是 EMULATED** | 等级：C（R3 交叉验证方向，A 仅在探针）
- `latency = base + gain×max(0, util − congestion_start)` 在实验循环里是模型值；
- 真实 netem 效应仅在 R3 探针中单独验证（方向一致），未进入实验循环。

### 4.2 Resource Gaps（资源层）

**RG1 — 训练代价在单物理 GPU 上不可观测（最重要）** | 等级：D
- 根因（代码层）：Local `progress += 真实样本数`（REAL，单卡照跑满速）；虚拟配额只改变
  **上报给调度器的 EMULATED 吞吐**（`T1 × scale(g^0.85)`），不改变真实工作量。
- 量级：Mock 认为 elastic 进度损失 53%、hard 76%；Local 真实进度差 <5%（噪声）。**趋势不一致**。
- 含义：**单卡上"抢占的真实代价"不可观测**——没有第二张卡可以被抢走，真实训练从未变慢。
- 归属：Resource——物理资源只有 1 张 GPU 的结构性限制（spec §7 要求的诚实边界）。

**RG2 — GPU 触发路径在 Local 不可构造** | 等级：C（仅 Mock exp A 覆盖）
- Local GPU 延迟带 base 5.27ms / SLO 15ms 带太窄，GPU 单独永不触发 SLO；过载信号 100% 来自网络。
- 影响：GPU-bound 决策路径只在 Mock（C）验证，Local 不覆盖（E20）。

### 4.3 Runtime Gaps（runtime 层）

**RTG1 — tick 语义不同：虚拟 vs wall-clock** | 等级：A/B
- Mock 840 tick 是虚拟 70min（毫秒级跑完）；Local 30 tick 是真实 75s（每个 tick 真实 HTTP + 训练跑
  tick_seconds）。
- 影响：Local 无法跑完整 70min 潮汐（需 ~30min/策略），只能压缩形态；全潮汐闭环数字（T2）只在 Mock。

**RTG2 — 决策一致但信号 EMULATED** | 等级：B
- Scheduler 代码路径真实执行（决策一致，B），但**驱动决策的推理延迟/训练吞吐是 EMULATED 值**——
  Local 的"真实"体现在执行链路，不体现在量化信号上。

### 4.4 Network Gaps（网络层）

**NG1 — 实验循环网络延迟为 EMULATED，真实 netem 仅 R3 交叉验证** | 等级：C（探针 A）
- 施加真实 tc/netem 到循环会导致实验慢 10× 且不可重复，故循环内用模型值。
- R3 独立验证真实链路：netem 10ms → 真实 e2e RTT 20.8ms；40ms netem → 吞吐 1188→150 qps 崩塌；
  Server P95 平、e2e P95 抬升（测量接口语义差异）。
- 标注：LOCAL-EMULATED-LATENCY。

**NG2 — 网络严重度轴（Mild/Moderate/Severe）只在 Mock** | 等级：C
- T3 的过渡区发现（Moderate 下 GPU-only 自我破坏）仅在 Mock 验证；Local 无法构造该 GPU/网络
  混合触发形态（RG2），未跨 runtime 验证。

### 4.5 Measurement Gaps（测量层）

**MTG1 — P95 口径：server vs e2e** | 等级：A
- `real_stats`（server 处理延迟）与 `e2e_p95`（client 总往返）测的不是同一东西；
  netem 40ms 下 Server P95 平、e2e P95 显著抬升。混用口径会误判拥塞影响。
- 已修正语义并文档化（`inference_real.py`）。

**MTG2 — 训练进度含真实噪声** | 等级：A
- Local 各策略进度差 <5%（硬抢占 14.07M vs static 13.98M 的 +0.6%）是测量噪声，不是调度效果；
  报告必须把"噪声级差异"与"系统差异"分开。

**MTG3 — GPU 利用率指标 = 分配率（恒 1.0）** | 等级：A（但语义非利用率）
- `gpu_utilization = (training+inference)/total` 由守恒律恒为 1.0，是分配率恒等式，不是利用率度量。
- 影响：不能用它论证"GPU 利用情况"；Finding B 用的"GPU 有空闲"是**推理容量 vs QPS** 的论证，
  不是这个指标。

**MTG4 — 训练降速只能报 EMULATED 比率** | 等级：D
- Local 的 elastic 1.67 / hard 3.17 是 EMULATED 上报比率（RG1），不是真实训练降速测量。

---

## 5. 三项 Mock Finding 的跨 runtime 验证

### Finding A — 硬抢占一次性大幅让渡 → 过度反应，代价大且不换回 SLO ✅（等级：C 数值 + B 决策）

| | Mock | Local |
|---|---|---|
| 决策 | 训练 4→1 一刀切 | 训练 4→1 一刀切（同一决策） |
| 训练代价 | 进度 8262（static 的 24%），降速 3.24 | EMULATED 降速 3.17（最差），真实进度不受影响 |
| P95 | 581ms（最差） | 497ms（最差） |
| SLO | 100% 违约 | 100% 违约 |

让渡 GPU 既没恢复 SLO（网络才是瓶颈），又付出最重训练代价。硬抢占的"快"在此无收益。
全潮汐版本见 `docs/phase3_3_t2_full_tide.md` 表 3（4 次 6↔1 ping-pong）。

### Finding B — GPU-only 调度在 Network-bound → 拥塞更糟，P95 更差 ✅（方向 B + C 数值）

- elastic / hard 扩容推理卡 → 网络需求上升（Local 利用率 1.82→2.20 / 2.46），EMULATED
  拥塞延迟上升 → **P95 变差**（336→433 / 497ms）。
- **GPU 有空闲**：推理容量 4×563.3≈2253 qps vs 峰值 1700 qps（GPU util 0.75），但系统
  **不可扩展**——瓶颈在网络。GPU 空闲 ≠ 系统可扩展。
- network_aware 拦截扩容 → P95 最优（336ms）。

### Finding C — 网络扩容闸门边界 ✅（C；两极端方向已 B 验证）

| 场景 | 训练曲线 | 释放带宽(4→3) | 扩推理+1卡 | 净增 | 闸门 | network_aware 行为 |
|---|---|---|---|---|---|---|
| Local exp B（紧网络，平曲线 scale 0.2） | 平 | 100 Mbps | +600 Mbps | **+500 > headroom 0** | 拦截 | == static（0 切换，P95 相同） |
| Mock exp A（松网络 30000 Mbps） | 陡 | 大 | 100 Mbps | 富余 | 放行 | == elastic（6 切换，数字相同） |

- **紧 + 平** → 拦截（正确：扩容只增拥塞）；**松 + 陡** → 放行（正确：释放带宽够推理用）。
- **T3 延伸（C）**：在过渡区（Moderate，48% 拥塞），闸门价值**最大**——它拦住了 elastic 会做的
  两次自我破坏性扩容（SLO 91% vs 60.7%）。过度保守风险（高估释放/低估增量）未观测到。
- network_aware 从未比 elastic 更差。

---

## 6. Failure Cases

完整目录（14 例，统一格式）见 [`docs/failure_cases.md`](failure_cases.md)。本报告要点：

1. **网络受限下所有策略 SLO 100% 违约**（Mock 840/840、Local 30/30 tick）——结构性拥塞无法靠
   移动 GPU 修复，瓶颈不是 GPU。GPU-only 调度器此时唯一正确的动作是 network_aware 那样**别动**。
2. **Elastic 全面占优 hard**（全场景未构造出反例，见 failure_cases #14）。
3. **过渡区自我破坏**（failure_cases #3）是 T3 新发现，Mock-only。
4. 工程事故（run.py 静默跑错 / netem 无效 / P95 口径 / 分支名）均已修复并保留记录。

---

## 7. 结论

1. **三项 Mock Finding 全部跨 runtime 验证**（决策逐条一致、相对排序与趋势一致、拐点一致）——
   等级 B（方向/行为）+ C（数值）。
2. **五类 Gap 中最深的是 Resource（RG1）**：单物理 GPU 上 EMULATED 多卡缩放是**假设而非测量**
   （D）——本地实验能复现调度决策，但无法让"抢占代价"落在真实训练上。
3. **模型假设的诚实披露**：Mock 延迟基准 200ms 已被 R2 证伪（38×，MG1）；多卡缩放 g^0.85 是
   假设（MG2）；网络拥塞延迟是 EMULATED（NG1）。
4. **对成品的启示**：网络受限时"正确的调度"首先是**不动作**（停止 GPU-only 的盲目扩入）；
   过渡区（部分拥塞）是 GPU-only 最危险、也是网络闸门价值最大的地方。
5. **所有 Gap 只记录、分析，不"修漂亮"**（§3.1 最高优先级）：不改 scheduler、不改 progress
   语义去迎合 Mock；改不了的（单卡物理边界）如实标 D。
