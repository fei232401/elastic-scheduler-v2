# Phase 3.3 · T3 — 网络严重度实验报告

> 目标：沿**网络严重度**单一轴（拥塞比例 0% → 100%）观察四策略行为，找出
> 振荡 / 稳定 / 过度保守区域。
> 数据标签：**MOCK**（纯数学模型，seed 42，840 tick）。复现：
> ```bash
> for s in mild:experiment_mild_network moderate:experiment_moderate_network severe:experiment_b_network_bound; do
>   python -m experiments.run --matrix --config config/${s#*:}.yaml --results-dir results/severity/${s%%:*}
> done
> ```

---

## 1. 实验口径：单一变量 = 网络容量

三档配置**共享同一 workload**（exp B 的潮汐：QPS 10→30→10、训练 4 卡、推理 4 卡、
`per_gpu_infer_bw 600`、`bw_per_qps 30`、`training_bw_scale 0.2`），**仅改变网络容量**：

| 严重度 | 配置 | 网络容量 | 需求 vs 容量 | 拥塞定义（demand > capacity） |
|---|---|---|---|---|
| **Mild（轻度）** | `experiment_mild_network.yaml` | 30000 Mbps | 离峰 2980 / 峰值 3580 ≪ 30000 | 永不拥塞（0/840 tick） |
| **Moderate（中度）** | `experiment_moderate_network.yaml` | 3200 Mbps | 离峰 2980 < 3200，峰值 3580 > 3200 | **局部拥塞**（qps ≳ 17，约 48% tick） |
| **Severe（重度）** | `experiment_b_network_bound.yaml`（原 B） | 2200 Mbps | 离峰 2980 > 2200 | 结构性拥塞（840/840 tick） |

拥塞布尔 = `utilization > 1.0`（`src/network/network.py:77`）。

---

## 2. 完整结果表（3 严重度 × 4 策略）

| 严重度 | 策略 | SLO违率 | 平均P95(ms) | 最大P95(ms) | 切换 | 训练进度 | 训练降速 | 网利用率均 | 拥塞tick |
|---|---|---|---|---|---|---|---|---|---|
| **Mild** | static | 31.3% | 263 | 505 | 0 | 40938 | 1.00 | 0.11 | 0 |
| | elastic | **0.6%** | **183** | **340** | **2** | 40684 | 1.01 | 0.11 | 0 |
| | hard_preemption | **0.6%** | **183** | **340** | **2** | 40128 | 1.06 | 0.11 | 0 |
| | network_aware | **0.6%** | **183** | **340** | **2** | 40684 | 1.01 | 0.11 | 0 |
| **Moderate** | static | 60.7% | 391 | 663 | 0 | **40438** | 1.00 | 1.01 | 405 |
| | elastic | **91.0%** | **369** | **416** | 2 | 22739 | 1.71 | 1.30 | **755** |
| | hard_preemption | **91.0%** | 403 | 451 | 1 | 13747 | 3.02 | 1.44 | **755** |
| | network_aware | **60.7%** | **391** | **663** | 0 | **40438** | 1.00 | 1.01 | 405 |
| **Severe** | static | 100% | 506 | 792 | 0 | 35104 | 1.00 | 1.48 | 840 |
| | elastic | 100% | 525 | 581 | 2 | 16633 | 1.79 | 1.93 | 840 |
| | hard_preemption | 100% | 581 | 634 | 1 | 8262 | 3.24 | 2.16 | 840 |
| | network_aware | 100% | 506 | 792 | 0 | 35104 | 1.00 | 1.48 | 840 |

---

## 3. 分档解读

### 3.1 Mild（拥塞 0%）：网络不约束 → network_aware ≡ elastic，无回归

- 网络 util ≤ 0.12，闸门从不拦截（`net ≤ headroom` 恒成立）→ `network_aware` 与 `elastic`
  数值逐位相同（0.6% / 183ms / 2 切换 / 40684 进度）。
- elastic 做了一次快速的"让渡-归还"（t=1080s 4→3 → t=1200s 3→4），SLO 违约从 static 的
  31.3% 降到 0.6%，训练几乎无代价（降速 1.01）。
- **这就是 Finding C 的"松网络 → 放行"端**，且证明网络感知模块在无约束时不引入回归。

### 3.2 Moderate（拥塞 48%）：**过渡区 —— GPU-only 自我破坏，闸门价值最大**

这是本实验最重要的、**非显然**的发现：

| 指标 | static / network_aware | elastic | hard_preemption |
|---|---|---|---|
| SLO 违率 | **60.7%** | **91.0%**（比不动差 +30pp） | **91.0%**（差 +30pp） |
| 拥塞 tick | 405 | **755**（多了 350） | **755** |
| 训练进度 | 40438 | 22739（损失 44%） | 13747（损失 66%） |

**机制（决策日志）**：
```
elastic   t=420s (min 7)  4->3  p95=301ms  Rule1   ← 把网络延迟误判为缺 GPU
elastic   t=540s (min 9)  3->2  p95=324ms  Rule1   ← 再让 1 卡进推理
```
- 在 Moderate 下，即使低谷期 util ≈ 0.95 已经把 EMULATED P95 抬过 SLO（300ms）——这是
  **网络拥塞延迟**，不是 GPU 容量不足。elastic 按 Rule1 把它当成"推理缺卡"→ 让渡 GPU 给推理
  → 推理卡 4→6 → `per_gpu_infer_bw 600×2 = +1200 Mbps` 网络需求 → 拥塞窗口从 405 tick 扩散到
  755 tick → **后续 SLO 违约反而更持久**。
- 而 static 从不动作 → 拥塞不扩散 → SLO 违约 60.7%。
- **network_aware 的 §17 闸门在 t=420/540 拦截了这两次扩容**（`delta_infer − released > headroom`）
  → 与 static 完全一致（60.7% / 405 tick）。**闸门在这里的价值最大**：不是挡住"必然失败的动作"
  （Severe 下任何动作都救不了），而是挡住**会把局势搞得更糟的动作**。

> **这是对 Finding C 边界条件的量化延伸**：网络受限时 GPU-only 调度的危害不是"无效"
> （Severe：反正全违约），而是"自我破坏"（Moderate：主动把 60.7% 的违约率恶化到 91%）。
> 越接近过渡区，闸门越值钱。

### 3.3 Severe（拥塞 100%）：结构性瓶颈 → 任何 GPU 移动都无效

- 拥塞 840/840 tick，需求恒 > 容量：**移动 GPU 无法修复非 GPU 的瓶颈**。
- 所有策略 SLO 100% 违约；elastic/hard 白白损失训练（进度 16633/8262，降速 1.79/3.24）
  且 P95 更差（525/581 vs 506）。
- network_aware 正确**不动作**（== static，零训练损失）。Severe 是 Finding B 的原始场景。

---

## 4. 沿严重度轴的策略行为总结（稳定区 / 过渡区 / 结构性区）

| 区间 | 拥塞比例 | 最优动作 | 观察到的行为 |
|---|---|---|---|
| **稳定区（Mild）** | 0% | elastic 正常让渡-归还 | SLO 0.6%；network_aware == elastic（无回归） |
| **过渡区（Moderate）** | ~48% | **不动作**（network_aware） | elastic/hard 自我破坏：SLO 91% vs 60.7%，拥塞 405→755 |
| **结构性区（Severe）** | 100% | 不动作（network_aware） | 全策略 100% 违约；elastic/hard 白白牺牲训练 |

- **没有观察到病态振荡**（弹性规则防抖 + min_hold 生效；本轴信号单向）。
- **过度保守风险未触发**：network_aware 在 Moderate 的拦截是"正确保守"——被拦的扩容确实
  净增拥塞（released 100 Mbps ≪ 扩 600 Mbps）。未发现闸门误拦"本可安全完成的扩容"的场景
  （该风险需陡训练曲线 + 高估释放带宽的组合，本轮无此案例）。

---

## 5. 诚实边界

- 全部为 **MOCK** 数值（数学模型 + seed 42），跨 runtime 有效性：Mild/Severe 两极端已由
  Phase 3.2 cross-runtime 验证（decision 逐条一致）；**Moderate 是新场景，尚未跨 runtime 验证**
  （Local 的 GPU 延迟带太窄，无法构造该 GPU/网络混合触发形态，见 reality_gap Gap 3）。
- 网络拥塞延迟为 EMULATED 模型值；真实 tc/netem 仅 R3 交叉验证（`docs/reality_gap.md`）。
- Moderate 的 3200 Mbps 是设计值（介于 2980 离峰 / 3580 峰值之间），实测拥塞 405/840 tick
  （48%）如实报告，未做任何"调参找好看数字"。
