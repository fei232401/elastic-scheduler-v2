# 证据矩阵（Evidence Matrix）

> 本项目所有结论的证据等级分类，**防止把模拟写成实测**（spec §7/§22/§28 最高优先级诚实约束）。
>
> ## 分类定义
> - **A — Physically Validated**：本机真实测量，数字即物理事实（RTX 4070 8GB 单卡实测）。
> - **B — Runtime Validated**：在真实 runtime（Local：真实 PyTorch 训练 + 真实 HTTP 推理 +
>   tc/netem 网络）上跑通过；**但**驱动它的多卡缩放 / 网络拥塞延迟是 EMULATED 模型值。
> - **C — Mock Simulated**：仅在 Mock 数学模型（seed 42）上验证；数字是模型输出，非测量。
> - **D — Not Validated**：未验证（假设、外推、红线外能力、被证伪的假设）。**D 绝不写 A。**
>
> 标注规则（写进任何报告前必须自查）：**"真实/实测"只允许 A**；**"行为一致/方向正确"最多 B**；
> **"数值/量化"来自 Mock 就是 C**；**任何多 GPU / 训练代价 / 生产级声明如无实测即 D**。

---

## 1. 完整声明清单

| # | 声明（Claim） | 等级 | 证据位置 |
|---|---|---|---|
| E1 | 单卡 PyTorch 训练吞吐（R1，T1=REAL） | **A** | `results/profiles/training_profile.json`、`calibration.json` |
| E2 | HTTP 推理延迟/容量（R2：base 5.27ms、capacity 563.3 qps、saturation 1126.7） | **A** | `results/profiles/inference_profile.json` |
| E3 | tc/netem 真实网络整形（R3：netem 10ms→真实 RTT 20.8ms；40ms→吞吐 1188→150 qps 崩塌） | **A** | `results/profiles/network_profile.json`、`reality_gap.md` Gap 4 |
| E4 | Mock 实验确定性（同 seed 双跑 byte-identical，干净 results/ 重跑 51 文件 md5 全同） | **A** | T1 审计 §4 + T11 复核（`docs/phase3_3_t11_reproducibility.md`） |
| E5 | Scheduler 决策在 Mock 与 Local 逐条一致（elastic 4→3→2、hard 4→1、network_aware/static 0 切换） | **B** | `reality_gap.md` §3、T2 表 4 |
| E6 | 扩容推理→网络需求↑→EMULATED P95↑ 的方向在真实 workload 上成立 | **B** | `reality_gap.md` §5 Finding B |
| E7 | network_aware 闸门在 Local runtime 拦截扩容 | **B** | `reality_gap.md` §3（决策逐条一致） |
| E8 | 网络结构性拥塞下 SLO 全策略违约（Local 30/30 tick） | **B** | `reality_gap.md` §6.4 |
| E9 | 全潮汐闭环：elastic 6→5→4→3→4→5→6（70min，GPU-bound） | **C** | T2 表 1/表 2 |
| E10 | 全潮汐 SLO 违约 49.8%→1.4%、峰值 P95 902→336ms、代价训练 -20% | **C** | T2 表 1 |
| E11 | Finding A：硬抢占一次性让渡=过度反应（SLO 10.2% vs 1.4%、训练 -31%） | **C** | T2 表 1、T3 |
| E12 | Finding B：GPU 空闲 ≠ 系统可扩展（网络瓶颈） | **C**（方向 B） | T3 3.3、`reality_gap.md` §5 |
| E13 | Finding C：网络闸门边界（紧+平→拦，松+陡→放） | **C** | T3 3.1/3.2、`reality_gap.md` §5 |
| E14 | 网络严重度过渡区：Moderate 下 GPU-only 自我破坏（SLO 91% vs 60.7%、拥塞 405→755） | **C** | T3 3.2 |
| E15 | 参数扫描区域（振荡/稳定/过度保守） | **C** | T4、`results/scan/scan_all.json` |
| E16 | 本地推理基准假设 200ms（Mock 默认） | **D（已被 R2 证伪）** | `reality_gap.md` Gap 1（38×） |
| E17 | 多卡吞吐缩放 g^0.85（g>1） | **D（EMULATED 假设非测量）** | `reality_gap.md` Gap 2、§7 |
| E18 | 真实分布式训练上的 GPU 资源让渡代价（Multi-GPU handoff） | **D（单卡无法验证）** | `reality_gap.md` Gap 2 |
| E19 | Local 训练降速比率（elastic 1.67 / hard 3.17） | **D（EMULATED 上报比率，真实训练未变慢）** | `reality_gap.md` Gap 2 |
| E20 | GPU-bound 触发路径（Local 上不可构造） | **C**（仅 Mock exp A；Local 不覆盖） | `reality_gap.md` Gap 3 |
| E21 | GPU 利用率指标 = 分配率（恒 1.0） | **A**（守恒恒等式，实测为 1.0；语义非利用率） | `code_audit.md` R31 |
| E22 | K8s / 多节点 / RDMA / 生产级控制器 / GPU 热迁移能力 | **D（红线外，未验证）** | `phase3_2_final_report.md` §7 |

---

## 2. 高风险声明（最容易误标为 A，必须逐条诚实）

| 声明 | 常见错误写法 | 正确写法 |
|---|---|---|
| E17 多卡缩放 | "弹性缩卡后吞吐按 g^0.85 变化" | "单卡吞吐为 REAL 实测；**多卡缩放为 EMULATED 假设**（g^0.85，非物理测量）" |
| E18 抢占代价 | "弹性让渡使训练变慢 20%"（当指真实多卡时） | "GPU-bound 场景下 Mock 模型显示训练进度 -20%；**真实多卡上的物理代价未验证（D）**" |
| E19 训练降速 | "Local 训练降速 1.67" | "Local 上报的降速为 **EMULATED 比率**；真实训练因单卡配额不变从未变慢" |
| E16 延迟假设 | "推理 P95 基准 200ms" | "Mock 假设 200ms；**R2 实测 5.27ms（38×），假设已被证伪**" |
| E20 GPU 触发 | "GPU 容量触发 SLO 扩容" | "仅在 Mock exp A（C）验证；**Local 无法构造 GPU 触发形态**（Gap 3）" |
| E13 网络闸门 | "闸门在网络受限时拦截"（当指全部情况） | "闸门拦截仅在**训练曲线平 + 带宽紧**时约束力强；松+陡时正确放行（Finding C 边界）" |

---

## 3. 反查表：把"结论"翻译成"证据等级"

写任何文档时，从结论反查它站在哪条证据上：

| 你要写的结论 | 必须同时写 | 等级 |
|---|---|---|
| "弹性调度保护 SLO" | 数值来自 Mock 全潮汐（C）；决策在 Local 复现（B） | C（数值）+ B（行为） |
| "网络感知闸门有效" | 方向在 Local 验证（B）；量化在 Mock（C）；Moderate 过渡区未跨 runtime（C） | C/B |
| "硬抢占更差" | 决策在 Local 复现（B）；代价数值是 Mock（C）；真实训练未变慢（D） | C |
| "本机延迟 5.27ms" | 这是 R2 真实测量 | **A** |
| "单卡训练吞吐 T1" | 这是 R1 真实测量 | **A** |
| "多卡性能" | 单卡 A，多卡缩放 D | A/D |
| "本地真实 workload 跑通" | 真实 HTTP + 真实 PyTorch 确实执行 | A（机制）+ B（EMULATED 量化） |

---

## 4. 本项目的证据等级统计（诚实口径）

| 等级 | 声明数 | 说明 |
|---|---|---|
| A（物理实测） | 4 | R1/R2/R3/确定性 |
| B（runtime 验证） | 4 | 决策一致 / 方向 / 闸门拦截 / 结构性违约 |
| C（Mock 模拟） | 7 | 全潮汐 / Finding A/B/C / 严重度过渡 / 参数扫描 / GPU 触发 |
| D（未验证） | 6 | 多卡缩放 / 真实抢占代价 / EMULATED 降速 / Mock 延迟假设（证伪）/ K8s 等红线 |

**A:D = 4:6 —— 项目最核心的"真实多卡能力"声明全部是 D**。这不是缺陷，是诚实边界：
单张物理 GPU 的环境决定了分布式验证不在本阶段可达范围（spec §7 明确要求如实标注）。
任何对外表述若把 D 说成 A，即违背本项目如实标注的最高优先级纪律。
