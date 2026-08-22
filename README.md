# Elastic Training-Inference Resource Scheduler(算法层)

> **定位**:集群层 Training↔Inference 弹性资源调度 —— 潮汐流量下训练逐级让渡 GPU 给推理、
> 高峰保护推理 SLO、流量回落后完整归还。
> **来源**:源于早期原型 elastic-training-inference(commit e07c46c),其 `src/`/`tests/`/`experiments/` 已逐字节复用进本仓库;+ **Experiment E**(预测误差敏感性, v2 新增)。
> **状态**:135 tests 全绿 ✅ + 与早期实现逐字节一致验证 ✅ + Experiment E 已交付 ✅(e4638e3) + **4090 真机争抢校准 ✅**(§5.6) + **L3 万卡模拟验证设计 ✅**(§9)
> **规模证据链**:Mock 决策(mock 135 tests)→ 单卡真实争抢(L2, §5.6)→ 万卡模拟集群(L3 设计, §9)。边界诚实声明见 §7。

---

## 1. 一句话定位

一个 rule-based 弹性调度器,回答:**在共享 GPU 集群上,训练任务能否作为"弹性 workload"
在潮汐流量下逐级让渡 GPU 给推理、保护推理 SLO、再归还?** 全部用确定性 Mock 数学建模,
核心结论由跨 runtime(Mock / 单机真实 workload)双重验证。

**一句话结论**:弹性调度在 GPU-bound 潮汐下完成"让渡-归还"闭环,SLO 违约 49.8%→1.4%
(显式代价:训练慢 20%);硬抢占被严格占优;网络受限时 GPU-only 调度的正确动作是"不动作";
预测辅助(v2 Experiment E)能消除 SLO 违约,但暴露了"单向预夺牺牲训练"的 v2 优化方向。

## 2. 与早期原型的关系

- **v2 = 早期原型 + Experiment E**。早期原型(弹性训练推理调度器,30 节 spec 的完整实现,22 commits)
  **保留作参考**,不再开发。
- **复用方式 = 字面复用**:v2 基线 commit `20b345c`,`src/`、`tests/`、`experiments/` 与早期实现
  逐字节一致(diff 验证),135 tests 全绿——早期实现的所有结论在 v2 中原样成立,不重写不漂移。
- **v2 增量只有一个:Experiment E**(见 §5.5)——在早期实现的 elastic 之上加预测前瞻,扫预测误差敏感性。
- 为什么这么做(设计说明):基线已验证成熟(证据矩阵 + 基线态回归全绿),与其重写,
  不如**逐行复用进新 repo + 加一个直指 v2 方向的新实验**,把"从早期实现学到什么、v2 往哪走"说明白。

## 3. 架构

```
config/          default.yaml(全潮汐)+ experiment_{a,b,c}* + 网络严重度/校准配置
src/
  cluster/       GpuPool(8 GPU token)+ ResourceManager(配额流转)
  workload/      Mock 训练/推理/潮汐 + 真实 PyTorch/HTTP(跨 runtime)
  network/       MockNetwork(带宽/拥塞/延迟/闸门)+ NetemNetwork(tc/netem)
  runtime/       RuntimeAdapter(ABC)+ Mock/Local 两个实现(同一契约)
  scheduler/     base / static / elastic / preemptive / network_aware
  calibration/   R1/R2/R3 测量 → local_calibrated.yaml(REAL/EMULATED/DESIGN 出处)
  metrics/       collector(CSV/JSON/决策日志)+ exporter(5 图)
  simulator/     clock + engine(经 build_runtime(mode) 分派 Mock/Local)
experiments/     run.py(矩阵)/ run_local.py(真实对比)/ run_experiment_e.py(预测)/ ...
tests/           135 tests
results/         可重跑输出(已 gitignore)
```

## 4. 核心问题

**"潮汐来了,训练让不让?让多少?何时还?"**

- **static**:固定 6:2,高峰 SLO 崩(49.8% 违约)——基线,证明"不动"不行。
- **elastic**:峰前 `6→5→4→3` 逐级让渡、峰后 `3→4→5→6` 归还,零振荡。
- **hard_preemption**:一次性 `6↔1` ping-pong——"快"无收益,还更贵。
- **network_aware**:elastic + 网络闸门,拥塞时"不动作"。

## 5. 实验证据(全部确定性 seed 42;证据等级见 docs/evidence_matrix.md)

### 5.1 全潮汐闭环(GPU-bound,70min,840 tick,Mock)

| 策略 | SLO违率 | 平均P95 | 切换 | 训练进度 | 训练降速 |
|---|---|---|---|---|---|
| static | 49.8% | 491ms | 0 | 57783 | 1.00 |
| **elastic** | **1.4%** | **207ms** | **6** | 46285 | 1.33 |
| hard_preemption | 10.2% | 240ms | 8 | 39871 | 2.42 |
| network_aware | 1.4% | 207ms | 6 | 46285 | 1.33 |

- elastic 峰前 `6→5→4→3`(3 次 Rule1)、峰后 `3→4→5→6`(3 次 Rule3),零振荡。
- **无免费午餐**:保护 SLO 的代价 = 训练进度 −20%。
- hard_preemption 4 次 `6↔1` ping-pong:SLO 是 elastic 的 7×、训练更贵。

### 5.2 网络严重度轴(exp B workload,只变带宽)

| 严重度 | 拥塞比例 | 最优策略 | 关键数字 |
|---|---|---|---|
| Mild(30000 Mbps) | 0% | elastic(= network_aware) | SLO 0.6%,闸门无回归 |
| **Moderate(3200)** | 48% | **network_aware(不动作)** | **GPU-only 自我破坏:SLO 91% vs 不动 60.7%** |
| Severe(2200) | 100% | network_aware(不动作) | 全策略 100% 违约(结构性瓶颈,移动 GPU 无效) |

**过渡区发现(最重要)**:局部拥塞时 GPU-only elastic 把"网络拥塞延迟"误判为"缺 GPU",
扩容推理 → 净增带宽需求 → 拥塞扩散 → **比不动更差 30pp**。网络感知闸门在此价值最大。

### 5.3 跨 runtime 验证(最重要)

elastic `4→3→2` / hard `4→1` / network_aware 0 切换,**Mock 与 Local 真实 workload 逐条一致**——
Scheduler 行为与 runtime 无关,三项核心 Finding 不是 Mock 模型的偶然产物。

### 5.4 参数扫描(振荡/稳定/过度保守区域)

| 参数 | 振荡区 | 稳定区 | 过度保守区 |
|---|---|---|---|
| SLO 阈值 | ≥400(滞后→thrash) | **250–300** | ≤200 |
| min_hold | — | 0–48 | ≥96(粘滞) |
| scale_step | ≥4(=温和硬抢占) | 1–3 | — |
| max_slowdown | — | 2–3 | ≤1.2(网络受限下反而最优) |

### 5.5 Experiment E —— 预测误差敏感性(v2 新增,核心交付)

**问题**:调度器依赖对未来负载的预测。预测误差多大时,"预测辅助调度"仍优于纯反应式 elastic?

**方法**:predictive = 反应式 elastic + EMA 预测器**提前分配** GPU(误差受控高斯噪声),
sweep 误差 std ∈ {0, 0.1, 0.3, 0.5}(0.5 = ±50% 预测误差),同 seed 同潮汐同配置公平对比。
复现:`python -m experiments.run_experiment_e`。

| 配置 | SLO违率 | P95(ms) | 切换次数 | 训练进度 | 训练慢化 |
|---|---|---|---|---|---|
| elastic 基线(纯反应式) | 1.4% | 207 | 6 | 46285 | 1.33× |
| predictive err=0.0(完美) | **0.0%** | 184 | 38 | 34557 | 2.42× |
| predictive err=0.1 | 0.0% | 180 | 45 | 27508 | 2.89× |
| predictive err=0.3 | 0.0% | 180 | 44 | 25968 | 3.11× |
| predictive err=0.5 | 0.0% | 180 | 44 | 25846 | 3.14× |

**三个诚实发现(v2 核心,必须说明)**:
1. **预测一定赢 SLO**:0% vs 1.4%,P95 180 vs 207ms——提前让渡确实消除违约;
2. **误差不敏感(出乎意料)**:err=0.0→0.5 结果几乎不变。机理:本实现只"预测预夺",
   预夺到训练下限后停在稳态,±50% 噪声翻不动稳态决策 → **敏感度不在预夺路径**;
3. **代价是训练被单向牺牲**:进度 −25%~−44%、切换 6→44 次。因为预夺单向,P95 被压到
   180ms(远低于 SLO)后,反应式恢复逻辑几乎不触发,GPU 迟迟不还训练(慢化 1.33→3.14×)。

**这直接指向 v2**:预测必须**双向**(既预夺也预测恢复),或引入**成本目标函数**
(权衡 SLO 违约代价 vs 训练进度代价)。v1 的 E 是"最小验证",证明"预测有用但副作用明显"。

### 5.6 4090 真机争抢校准(单卡 训练让渡→推理 收益曲线,Task #15)

**问题**:mock 里"训练让渡给推理"的物理收益/代价是 EMULATED 比率(§7 诚实边界 #1)。
真机单卡上,训练 job(占空比 intensity)与推理 vLLM(并发 concurrency)共置时,双方各受多大影响?

**方法**(`experiments/calibrate_train_infer_contention.py`,AutoDL RTX 4090 24GB):
- 训练侧:真实 `RealTrainingJob` MLP(512→2048×4, batch 1024)子进程,按 `intensity`(占空比)睡眠让步;
- 推理侧:Qwen2.5-1.5B-Instruct vLLM OpenAI server,**禁用 prefix caching**(本实验测争抢,不测路由)、600-token prompt 强制全 decode;
- 矩阵:5 intensity × 4 concurrency,每格 25s 窗口 + 3s 斜坡,**12s warmup 后正式计**;
  确定性 `seed 42` 打乱格子顺序(修复冒烟期 vLLM 引擎预热污染);
- 结果 JSON:`experiments/results/calibrate_train_infer_contention.json`(solo 训练 345,694 sps 基线)。

**结果(aggregate,每格 25s 稳态)**:

| 训练 intensity | 推理 decode(t/s)@conv16 | 训练 sps@conv16 | 训练 sps@conv1 |
|---|---|---|---|
| 0%(全让渡) | 1229.1 | — | — |
| 25% | 1206.9 | 116,371 | 122,171 |
| 50% | 1137.7 | 197,588 | 199,556 |
| 75% | 1210.2 | 191,225 | 201,501 |
| 100%(全占) | 1130.8 | 200,215 | 202,441 |
| solo(无推理) | — | 345,694 | 345,694 |

**三个诚实发现(核心,必须说明)**:
1. **让渡收益小(1.5B decode 场景)**:训练 100%→0% 全让渡,推理 decode 仅 +8.7%
   (1130.8→1229.1 t/s),TTFT p95 71.4→75.6ms 为噪声级。物理原因:1.5B decode
   **非 SM-bound**(decode 聚合随并发近线性 80→1229 t/s),推理不吃 SM,训练让出的算力
   无处兑现 → 让渡机制对"compute-bound 推理"(大模型/prefill 大 batch)才有显著收益;
2. **共置代价大(给训练)**:任何推理并发共存时,训练 sps 从 solo 345.7k 跌至 ~190-200k
   (**-42%~-45%**,各并发稳定)。即"推理对训练的干扰"远大于"训练对推理的干扰";
3. **intensity 占空比不是线性让渡**:25% 占空比时训练仅 ~117k sps(约 1/3 solo),而 50/75/100%
   都落在 ~190-200k —— 主动睡眠让步的真实让渡曲线非线性,量化"让多少训练工作量才能换
   多少推理收益"需要真机曲线而非 EMULATED 比率。

**对本项目的意义**:真机校准补上了 mock 最大边界(§7 #1 的 EMULATED g^0.85 / 让渡比率);
同时诚实标注:让渡收益的**量级依赖推理负载形态**(1.5B decode 收益小,大模型 prefill 才显著)。
这为 L3(§9)提供了"分配→SLO/降本"的效果模型输入,也避免把 1.5B 场景的弱收益夸成普适结论。

## 6. 关键参数(已校准)

| 参数 | 值 | 来源 |
|---|---|---|
| SLO 阈值 | 300ms | 参数扫描稳定区(§5.4) |
| scale_step / min_hold | 1 / 12 | 参数扫描稳定区(§5.4) |
| 集群规模 | 8 GPU token | spec 定义 |
| 潮汐 | 70min / 840 tick | seed 42 确定性 |
| g^0.85 缩放 | EMULATED | **非物理测量**(见 §7.1) |
| EMA α / horizon | 见 predictor.py | E 设计决策 |
| 真机争抢:让渡收益/共置代价 | §5.6 曲线 | **4090 实测**(solo 345.7k sps,共置 -42~45%) |

## 7. 诚实边界(主动说明)

1. **多 GPU 缩放(g^0.85)仍是 EMULATED**:§5.6 只实测了**单卡**训练↔推理共置的真实争抢,
   多卡/分布式训练的弹性物理影响仍未实测(证据等级 D)。这是项目最大剩余边界,
   也是 L3(§9)要解决的规模问题。
2. **Mock 推理延迟假设 200ms 被真实测量证伪**(R2 实测 5.27ms,38×):Mock 与 Local 的 P95
   绝对量级不可比;可比的是趋势/排序/决策。
3. **Local 训练代价不可观测**:单卡上虚拟配额不改变真实工作量,降速是 EMULATED 比率。
4. **E 的发现是单向预夺的产物**:只证明"预测有用但副作用明显",不能推广到任意预测器结构;
   双向预测 + 成本目标是 v2 待验证方向(已写好问题,没做)。
5. **参数扫描 / 过渡区发现为 Mock 数值**,未跨 runtime 验证。
6. **红线外不做**:K8s / 多节点 / RDMA / 生产级 controller / GPU 热迁移(注:L3 §9 为设计
   ——K8s/Kueue/KWOK 验证属**下一步**已规划,非当前已交付)。
7. **§5.6 校准只覆盖一个负载形态**:Qwen2.5-1.5B decode、单卡、GPU-frac 0.5。
   "训练让渡收益小"只在**该形态**成立;大模型 prefill / 更大 batch 的让渡收益**未测**,
   不能反向推广为"让渡没用"。对本项目的影响已诚实标注在 §5.6。

## 8. 复现

```bash
pip install -r requirements.txt

python -m pytest tests/ -q                 # 135 tests
python -m experiments.run --scheduler elastic --config config/default.yaml
python -m experiments.run --all --config config/default.yaml   # 4 策略全潮汐对比
python -m experiments.run --matrix                             # 3 实验 × 4 策略矩阵
python -m experiments.run --matrix --config config/experiment_moderate_network.yaml  # 严重度轴
python -m experiments.scan_parameters                          # 参数扫描
python -m experiments.run_local --compare                      # Mock vs Local(真实 workload)
python -m experiments.run_experiment_e                         # Experiment E(预测 sweep)
```

**4090 真机争抢校准(§5.6,需 AutoDL 4090 环境,见脚本 docstring 环境要求)**:
```bash
ssh -p <port> root@<autodl-host>                       # 4090 实例
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
python /root/autodl-tmp/elastic-scheduler-v2/experiments/calibrate_train_infer_contention.py \
  --intensities 0,25,50,75,100 --concurrencies 1,4,8,16 \
  --window 25 --warmup 12 --out results/calibrate_train_infer_contention.json
```

所有结果 `results/` 可一键重跑;**报告数字全部来自上述命令,未修改任何实验输出**。

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/evidence_matrix.md` | 全部声明的证据等级 A/B/C/D + 反查表 |
| `docs/reality_gap.md` | 五类 Reality Gap |
| `docs/failure_cases.md` | 14 个失败案例 |
| `docs/PROJECT_FINAL_TECHNICAL_REVIEW.md` | 最终技术评审(15 节) |
| `docs/code_audit.md` | 代码审计 |
| `docs/PROJECT_SPEC.md` | 原始规格(30 节) |
| `docs/architecture.md` | 架构 |
| `docs/phase3_3_t11_reproducibility.md` | 基线态回归 + 复现性(51 文件 md5 全同) |
| `integration/KUEUE_KOORDINATOR_MAPPING.md` | 平台层设计:v1 模块 → Kueue/Koordinator/自研三层映射 + 三个执行原语取舍 + 薄 controller 职责 + 阶段 B 落地路径 |
| `integration/L3_SIMULATOR_VALIDATION.md` | L3 万卡模拟验证设计:KWOK(伪造 10k GPU 节点)+ 真实 K8s 控制面 + Kueue 弹性配额 + 薄 controller 复用决策引擎。含工具选型决策、万卡声明诚实边界、分阶段落地路径 |

## 10. 设计决策记录

每个文件 docstring 里有 `🔍 决策点`。核心几个:
- 让渡粒度:整卡 token(1/2/…)vs 百分比(见 scheduler/elastic.py)
- 网络闸门如何判定"该动还是不动"(见 scheduler/network_aware.py)
- runtime 抽象的分界(Mock/Local 同一契约,见 src/runtime/base.py)
- E 里"预测只做预夺不做恢复"为什么是 v1 的正确最小验证(见 experiments/run_experiment_e.py)
