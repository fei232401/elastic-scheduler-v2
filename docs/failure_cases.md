# 失败案例目录（Failure Cases）

> 目标：**主动记录失败与边界**，不只看成功案例（spec §29.7/§36）。
> 统一格式：**Scenario / Expected / Actual / Root cause / Impact / Mitigation / Current status**。
> 证据等级见 [`docs/evidence_matrix.md`](evidence_matrix.md)（C=Mock、B=runtime、A=实测、D=未验证）。

---

## 1. 网络受限下弹性让渡换不回 SLO（结构性拥塞）

- **Scenario**：网络深度拥塞（exp B，2200 Mbps，拥塞 840/840 tick；Local 30/30 tick），elastic 逐级让渡 4→3→2。
- **Expected**：让渡 GPU 给推理 → SLO 恢复。
- **Actual**：SLO **100% 违约**（Mock 与 Local 全部策略）；elastic 比 static P95 更差（Mock 525 vs 506、Local 433 vs 336）。
- **Root cause**：瓶颈是网络不是 GPU——移动 GPU 不改变"需求恒 > 容量"；GPU-only 扩容反而净增网络需求（`delta_infer − released > 0`）。
- **Impact**：所有网络受限运行 SLO 全违约；elastic/hard 白白牺牲训练（进度 -53%/-76% Mock）。
- **Mitigation**：network_aware 闸门拒绝扩容（== static）；这是"正确的调度 = 不动作"。
- **Current status**：**已文档化**（reality_gap §6.4、T3 3.3）。结构性拥塞无法靠 GPU 调度修复，属物理边界。

## 2. 硬抢占全量 slam → P95 短暂恢复又复发（ping-pong）

- **Scenario**：全潮汐（default.yaml）硬抢占在过载确认后一次让渡 6→1。
- **Expected**：一刀切让渡 → SLO 一步恢复。
- **Actual**：P95 恢复仅 ~120s（24 tick）后复发；4 次 6→1 slam + 4 次 1→6 归还（8 切换）；SLO 违约 10.2%（elastic 的 7×）；训练降速 2.42。
- **Root cause**：全量让渡过冲 → 推理卡 4→9 超量 → 峰前负载再次击穿；无中间档，只能在"满配/地板"间跳。
- **Impact**：训练配额大幅震荡，进度比 elastic 再低 14%；"快"无收益（Finding A）。
- **Mitigation**：elastic 的逐级 ±1 让渡 + min_hold 防抖。
- **Current status**：**已文档化**（T2 表 3、T3）。弹性调度器对硬抢占严格占优。

## 3. Moderate 网络下 elastic 比 static 更差（过渡区自我破坏）【T3 新发现】

- **Scenario**：局部拥塞（3200 Mbps，48% tick 拥塞），elastic 在网络延迟触发 SLO 违约时按 Rule1 让渡。
- **Expected**：让渡保护 SLO（同 GPU-bound 场景）。
- **Actual**：elastic/hard SLO **91.0% vs static 60.7%**（+30pp 更差）；拥塞 tick 405→755；训练进度 -44%/-66%。
- **Root cause**：elastic 把**网络拥塞延迟**误判为"推理缺卡"→ 让渡推理 4→6 → +1200 Mbps 网络需求 → 拥塞扩散 → 违约更持久。**网络没有给调度器"这是网络问题"的信号**。
- **Impact**：GPU-only 调度在过渡区主动恶化局势——比 Severe（反正全违约）更危险。
- **Mitigation**：network_aware 闸门在 t=420/540 拦截这两次扩容（正确保守）。闸门价值在过渡区**最大**。
- **Current status**：**已文档化**（T3 3.2）。仅 Mock 验证（C），Moderate 未跨 runtime 验证（Local 无法构造该触发形态）。

## 4. slo_threshold 过松 → 3↔4 thrash 振荡【T4 参数扫描】

- **Scenario**：SLO 阈值设 400/500ms（默认 300），elastic 全潮汐。
- **Expected**：阈值松 → 更少动作、更稳。
- **Actual**：切换 **10**（默认 6），SLO 违约 11.3%（默认 1.4% 的 8×）。
- **Root cause**：反应滞后——Rule1 等 p95 深越过 SLO 才动作，一让渡 P95 掉回 healthy 带 → Rule3 立即归还 → 再越线 → 再让渡：3↔4 在 t=2075/2195/2315 thrash。
- **Impact**：振荡 + 违约率恶化；与"防抖不够"无关，是**阈值带过宽导致滞后-过冲环**。
- **Mitigation**：SLO 阈值保持 250–300（稳定甜区）；不做"越松越稳"的错误推断。
- **Current status**：**已文档化**（T4 §1）。

## 5. scale_step=4 → 6→2 slam 振荡【T4 参数扫描】

- **Scenario**：max_scale_step=4，elastic 全潮汐。
- **Expected**：大步让渡 → 更快恢复 SLO。
- **Actual**：切换 6、SLO 6.1%、训练 -10% 且更贵（降速 1.71）；决策呈 `6→2 → 2→6 → 6→2…` slam 环。
- **Root cause**：步长=4 等于"温和版硬抢占"——过冲后 Rule3 全量归还，再 slam。
- **Impact**：与 Finding A 同源（过冲）；步长增大无任何收益。
- **Mitigation**：step 保持 1–3；默认 1 已够。
- **Current status**：**已文档化**（T4 §3）。

## 6. min_hold=96 tick → 过度粘滞，SLO 违约 5×【T4 参数扫描】

- **Scenario**：min_gpu_hold_duration_ticks=96（8 分钟），elastic 全潮汐。
- **Expected**：更长的 hold → 更稳。
- **Actual**：SLO 7.7%（默认 1.4%），训练配额归还过慢。
- **Root cause**：hold 窗口掩盖了"已经可以归还"的时机——回落期训练配额迟迟回不到 6。
- **Impact**：过度保守：不振荡，但牺牲 SLO。
- **Mitigation**：min_hold 保持 24 以内（0–48 在本场景稳定区）。
- **Current status**：**已文档化**（T4 §2）。

## 7. Mock 推理延迟假设 200ms 被 R2 实测 5.27ms 证伪（38×）

- **Scenario**：spec §8 假设推理 P95 基准 200ms（SLO 300ms）；Phase 3.2 R2 真实测量本机推理。
- **Expected**：假设成立，Mock 延迟量级接近真实。
- **Actual**：R2 实测 **5.27ms**（38× 差距）；SLO 也需按 R2 校准为 15ms（20×）。
- **Root cause**：spec 的 200ms 是**设计假设**（无测量依据），真实单卡 HTTP 推理远快。
- **Impact**：Mock 的 P95 绝对量级（506ms）与 Local（336ms）不可直接比；只有趋势/排序/拐点可跨 runtime。
- **Mitigation**：保留两套基准并如实标注（Mock 200 / Local 5.27）；绝对值不跨 config 比较。
- **Current status**：**已文档化**（reality_gap Gap 1）。假设已被测量推翻（E16：D）。

## 8. Local 训练代价不可观测（EMULATED 降速比率，真实训练未变慢）

- **Scenario**：Local runtime 上让渡训练 GPU，期望看到真实训练变慢。
- **Expected**：抢占在真实训练上产生可测量的代价。
- **Actual**：真实 PyTorch 进度几乎不受配额影响（elastic vs static 进度差 <5%，噪声级）；上报降速 1.67/3.17 是 EMULATED 比率。
- **Root cause**：单张物理 GPU——虚拟配额不改变真实工作量；没有第二张卡可以被抢走（`progress += 真实样本数`）。
- **Impact**：**"抢占的真实代价"在单卡环境不可观测**；训练代价维度只能报 EMULATED（spec §7 必须如实标注）。
- **Mitigation**：E19 标 D；报告里明写"训练代价为 EMULATED 上报比率，非真实测量"。
- **Current status**：**已文档化**（reality_gap Gap 2）。不可修复于单卡，属物理边界。

## 9. Local GPU 触发路径不可构造

- **Scenario**：想验证"GPU 容量不足 → SLO → 扩容"在 Local 上的决策路径。
- **Expected**：Local 复现 Mock exp A 的 GPU 触发。
- **Actual**：Local GPU 延迟带 base 5.27ms / SLO 15ms——带太窄，GPU 单独永不触发 SLO；过载信号 100% 来自网络。
- **Root cause**：校准后延迟带过窄；除非把 SLO 收到 6ms 以下或改带宽假设。
- **Impact**：GPU 触发路径只在 Mock exp A（C）覆盖，Local 不覆盖（E20）。
- **Mitigation**：如实记录 Gap 3；不伪造一个"能触发"的配置。
- **Current status**：**已文档化**（reality_gap Gap 3）。诚实限制，未修。

## 10. run.py 喂 Local 配置静默跑 Mock【T2 现场踩坑，已修】

- **Scenario**：`python -m experiments.run --config config/local_experiment.yaml`（想跑 Local 全潮汐）。
- **Expected**：真实 workload 结果（Phase 3.2 Local 表：static P95 336ms、进度 14M）。
- **Actual**：静默产出 **Mock** 结果（static P95 601ms、进度 1103）——与 Phase 3.2 Local 数字完全不符。
- **Root cause**：`run_experiment(mode="mock")` 是默认；run.py 不传 mode；`local_experiment.yaml` 的 `local:` 段被忽略，跑的是"Mock 模型 + Local 配置的网络参数"。
- **Impact**：结果被误标为 Local（若当时直接采用就是数据造假级别的错误）；误导性输出。
- **Mitigation**：**已修**——run.py 检测到 `local:` 段即向 stderr 警告"请用 run_local"；规范 Local 入口 = `run_local.py`。
- **Current status**：**已修复**（T2 提交 d8adf4b）；T2/T3 报告使用正确入口重跑的数据。

## 11. R3 真实 netem 整形最初失败（网卡 lo DOWN）

- **Scenario**：Phase 3.2 R3 想在回环接口施加 tc/netem 延迟/整形。
- **Expected**：netem 生效，测得真实 RTT 抬升。
- **Actual**：最初失败——`unshare -Urn` 命名空间下回环接口 DOWN，tc 命令静默无效。
- **Root cause**：命名空间隔离下 lo 接口未启用（WSL2 环境细节）。
- **Impact**：若不发现会得到"整形无效"的错误结论（或整形根本没施加）。
- **Mitigation**：显式 `ip link set lo up`；R3 最终有效（netem 10ms → 真实 RTT 20.8ms）。
- **Current status**：**已修复并验证**（reality_gap Gap 4；`src/network/netem.py`）。

## 12. 服务器端 P95 与客户端 e2e P95 语义差异（测量接口陷阱）

- **Scenario**：R2 同时测 server 端处理延迟与 client 端总往返。
- **Expected**：两者近似（RTT ≈ 处理延迟）。
- **Actual**：netem 40ms 下 Server P95 平、e2e P95 显著抬升——两个"P95"测的不是同一个东西。
- **Root cause**：e2e 含排队/调度/网络往返；server 只测处理。用错口径会误判拥塞影响。
- **Impact**：指标口径混用 → 结论错位（Local 实验的 P95 语义曾因此修正）。
- **Mitigation**：明确区分 real_stats（server）与 e2e_p95（client）；报告注明用的是哪个。
- **Current status**：**已修正语义并文档化**（`inference_real.py`；reality_gap Gap 4）。

## 13. git push 失败："src refspec main does not match"

- **Scenario**：初始化仓库后用 `git push origin main`。
- **Expected**：正常推送。
- **Actual**：拒绝——`src refspec main does not match any`。
- **Root cause**：默认分支是 `master` 不是 `main`（`git init` 默认）。
- **Impact**：若未发现会用错误分支名反复重试、误以为推送失败。
- **Mitigation**：`git push origin master`（本项目全用 master）；README 注明。
- **Current status**：**已解决**（操作层面）。

## 14. 未构造出的失败：Elastic 何时不如 Hard Preemption？

- **Scenario**：主动寻找 elastic 劣于 hard_preemption 的场景（A/B/C + 全潮汐 + 严重度 + 参数扫描）。
- **Expected**：可能找到一个（如硬 deadline 单步恢复场景）。
- **Actual**：**全场景未构造出**——elastic 均占优或持平；hard 唯一理论优势（1 tick 恢复 SLO）在网络受限时因结构性拥塞无效。
- **Root cause**：spec 第一版不把 SLO 当硬 deadline；在软 SLO 语义下一次性让渡永远劣于逐级。
- **Impact**：无负面影响；这是"失败的失败案例"——诚实记录未找到的反例。
- **Mitigation**：不硬造一个；如实写"hard 被 strictly dominated，除非未来引入硬 deadline 语义"。
- **Current status**：**已文档化**（reality_gap §6.1、T3）。保留为理论风险区。

---

## 汇总

| # | 类别 | 状态 |
|---|---|---|
| 1-3 | 调度策略边界（结构性拥塞 / 硬抢占过冲 / 过渡区自我破坏） | 已文档化 |
| 4-6 | 参数扫描边界（阈值滞后振荡 / 步长过冲 / hold 粘滞） | 已文档化 |
| 7-9 | Reality Gap（延迟假设证伪 / 训练代价不可观测 / GPU 触发不可构造） | 已文档化（物理边界） |
| 10-13 | 工程事故（CLI 静默跑错 / netem 无效 / 指标口径 / 分支名） | **已修复** |
| 14 | 未构造出的失败（elastic 全面占优） | 诚实记录 |

> **原则**：没有为了让结果漂亮而隐藏或删除任何失败案例；第 10 例若未被发现会成为
> 数据误标，现已修复并保留完整记录（§28 不隐藏失败）。
