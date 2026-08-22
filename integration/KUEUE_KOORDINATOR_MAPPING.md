# Kueue + Koordinator 集成映射(算法层平台层 · Phase A)

> **定位**:把 v1 已交付的"确定性 mock 算法内核"(决策层)映射到真实 K8s 平台——
> **Kueue**(准入/配额/队列)+ **Koordinator**(节点弹性/colocation/设备共享)
> + **自研薄 controller**(把策略决策翻译成平台原语)。
> 本文档是**设计文档,不是代码**:每条映射都标注成立条件与诚实边界;阶段 B(真实 demo)见 §9。
>
> **一句话**:调度器的"大脑"是我写的(策略引擎 + 预测),"手脚"是行业标准
> (Kueue 管准入、Koordinator 管弹性),我用一个薄 controller 把两者接起来。

---

## 0. 为什么是这三层(不是"用 Kueue 就完事")

真实 K8s 缺两块,正好 Kueue/Koordinator 各补一块:

- **缺"按配额排队 + 优先级抢占"** → Kueue 补。但它只做 **准入/排队/Workload 级抢占**,
  **不做运行中任务的动态扩缩容**(没有"给运行中 pod 减卡"的原语)。
- **缺"节点级弹性复用"** → Koordinator 补(colocation 借闲置算力、GPU 共享、
  ElasticQuota 借贷)。但它不做"跨任务、按策略、有时序"的让渡决策。

于是"训推弹性让渡"(峰前让/峰后还、按 SLO 与拥塞闸门决策)这个**跨任务 × 按策略 × 有时序**
的决策,两个平台都不负责——这就是**自研策略引擎 + 薄 controller** 的生态位。
分工是需求逼出来的,不是硬套框架。

## 1. 三层架构总览

```
┌─ 决策层(自研 · v1 mock 已交付 = 这里)──────────────────────────┐
│  信号:潮汐负载 + 拥塞 + 训练进度 + 预测(Experiment E)            │
│  决策:elastic 让渡规则(6→5→4→3)/ network_aware 闸门/预测前瞻     │
│  "何时让 / 让多少 / 何时还"                                    │
└───────────────┬──────────────────────────────────────────────┘
                │  emit: ResourceDecision{train_gpu, infer_gpu, action}
┌─ 准入层:Kueue ─┴─────────────────────────────────────────────┐
│  ClusterQueue(配额池) → LocalQueue(命名空间队列) → Workload    │
│  职责:按配额准入推理/训练任务;高优先级 Workload 抢占低优先级     │
└──────────────────┬────────────────────────────────────────────┘
┌─ 执行层:Koordinator + 自研薄 controller ──────────────────────┐
│  controller:把 ResourceDecision 翻译成 §4 的平台原语            │
│  Koordinator:colocation 借算力 / GPU 共享 / ElasticQuota 借贷  │
│  koordlet:节点代理执行 QoS 调整 / 回收                          │
└───────────────────────────────────────────────────────────────┘
```

**谁写的**:决策层 = 用户(核心贡献,135 tests + E);controller = 用户(薄,见 §7);
准入/执行底座 = Kueue/Koordinator(行业标准,基于/复用)。

## 2. 模块映射表(v1 mock → 平台实体)

| v1 mock 模块 | 映射到 | 实现者 | 成立条件 / 诚实边界 |
|---|---|---|---|
| `GpuPool`(8 GPU token) | K8s 节点 GPU + Koordinator `Device`/GPU share | 平台 | token 粒度是抽象;真实 GPU 粒度看设备共享模式,需 probe |
| `ResourceManager`(配额流转) | Kueue `ClusterQueue`/`LocalQueue` | 平台 | 配额语义一致;**Kueue 不 resize 运行中 pod** → 让渡走 §4 原语 |
| `scheduler/elastic`(让渡-归还规则) | **自研策略引擎**(决策层) | **用户** | 规则逻辑可直接移植(纯函数,无平台依赖) |
| `scheduler/network_aware`(拥塞闸门) | 自研 + Koordinator 节点网络信号 | 用户+平台 | 信号来源需 probe(节点 exporter / koordlet 指标) |
| `scheduler/preemptive`(hard) | Kueue 优先级抢占 / 训练可抢占化 | 用户+平台 | v1 已证明 hard 是坏策略 → 真实平台默认不做 |
| `runtime/Mock` | 沙箱执行(现状) | 用户 | 保留:确定性回归 + 算法开发 |
| `runtime/Local` | 单机真实 workload | 用户 | 保留:算法在真实负载上的交叉验证 |
| **(缺失)KubernetesRuntime** | Kueue+Koordinator 真集群 | 用户(§7) | 阶段 B;需 k3d/kind + 双框架 CRD 版本配对 |
| `simulator/engine`(时钟驱动) | controller 的 reconcile 循环(事件驱动) | 用户 | 语义近似:mock 是确定性 tick,真集群是异步 reconcile |

## 3. 三个执行原语 + 取舍(核心设计决策)

"给运行中训练减 GPU、给推理加"没有原生原语。真实落地只能选(或组合)下面三者。
**v1 的 `6→5→4→3` 是 token 抽象,真实粒度取决于选哪个原语**:

### 3.1 可抢占化训练(checkpoint/分段)
让训练任务本身可被抢占、暂停、重排队(存 checkpoint)。
- **优点**:最诚实;对应 v1 里 hard_preemption 的教训(一次性让渡是坏的)——
  真实落地应是"训练 checkpoint 频率 × 抢占粒度"受控的弹性;
- **代价**:训练需要 checkpoint 支持;抢占粒度 = checkpoint 间隔,比 token 粗。

### 3.2 Koordinator colocation / ElasticQuota
批任务借 LSR(延迟敏感)pod 的闲置算力;ElasticQuota 允许配额间借还。
- **优点**:利用"推理高峰时训练闲置资源"之外的反向复用;
- **边界**:colocation 主要面向 CPU/内存回收,**GPU 弹性靠设备共享而非抢占**——
  "借 GPU"比"借 CPU"弱,需 probe 清楚。

### 3.3 Kueue 优先级抢占 / Workload 重排队
推理 Workload 高优先级,可抢占排队中的训练 Workload(或训练降级让配额)。
- **优点**:纯 Kueue 原生,零自研;
- **边界**:Kueue 抢占的是**排队/已准入的 Workload**,运行中任务的"抢占"实际是
  **驱逐 + 重排队** → 必然伴随重启,这就是"可抢占化"需求的来源。

### 3.4 组合映射(6→5→4→3 怎么落地)

| 决策(mock) | 推荐真实落地 | 粒度 |
|---|---|---|
| 训练 6→4(让 2 给推理) | Kueue:训练 Workload 降优先级 + 推理 Workload 按配额准入(3.3) | Workload 级 |
| 推理并发不够需更多算力 | Koordinator:GPU share 提升 + colocation 借闲置(3.2) | 节点级 |
| 训练被抢 → 暂停 | checkpoint 化训练,controller 触发 checkpoint + 驱逐(3.1) | 任务级 |

**推荐路径**:主用 3.1(训练可抢占化)+ 3.3(Kueue 配额/优先级),3.2 作为节点级补充。
理由:v1 已经证明"一次性硬抢占"是坏的,真实落地必须让"让渡成本"可控且可预测。

## 4. 归还路径(峰后"怎么还")

- 推理降并发(Kueue 收 Workload 配额)→ 释放的配额/节点资源:
  - Koordinator 借出的算力归还给 LSR;或
  - controller 把训练 Workload 的请求资源调回(若训练支持动态)或重新准入更大配额。
- **诚实点**:真实归还同样受 checkpoint/重启粒度约束,不是 token 级平滑递增。
  这是 mock 与平台最大的语义差距,主动说明。

## 5. 与 Experiment E(预测)的关系

- E 证明"提前让渡能消 SLO 违约"→ 真实对应:预测器读负载信号 → controller **提前**调 Kueue
  配额/优先级(推理 Workload 先准入),而不是事后反应。
- E 还证明**预测误差不敏感**(单向预夺锁定稳态)→ 落到平台是好消息:controller 不需要高精度
  预测,±50% 误差不改变结果 → 降低了对监控/预测基建的要求。
- E 的代价教训(单向预夺牺牲训练)→ 真实落地必须加**归还路径预测**(§4),否则训练被长期饿死。

## 6. 自研薄 controller 职责清单(MVP)

1. 监听负载/拥塞信号(潮汐、推理延迟、网络);
2. 调用决策层(纯函数,直接复用 v1 scheduler 代码)→ `ResourceDecision`;
3. 翻译:决策 →  Kueue Workload 优先级/配额调整, Koordinator colocation/GPU share 调整,
    训练 checkpoint+驱逐(若走 3.1);
4. 回读集群状态 → 防抖(对应 v1 的 min_hold/scale_step 参数);
5. 审计:每次决策写日志(复用 v1 `decisions_*.csv` 的语义)。

**判定标准**:controller 保持"薄"——不做任何调度决策,只做翻译与落地。
决策质量全部由 v1 已测试的策略引擎保证 → 135 tests 的可信度顺延到平台层。

## 7. 诚实边界与未验证项(主动说明)

1. **token 级让渡是抽象**:真实平台粒度是 Workload/节点级,`6→5→4→3` 平滑性不保真;
2. **Kueue 不做运行中 resize**:一切"让渡"最终是驱逐/重排队/配额调整,带重启成本;
3. **GPU 的"借用"比 CPU 弱**:Koordinator colocation 的 GPU 弹性靠设备共享,不是抢占,
   需 probe 真实行为;
4. **本机仍单 GPU**:阶段 B 即使上了 k3d,Kueue/Koordinator 的"多卡弹性"仍是单机上的
   模拟(集群层行为 EMULATED),与 v1 的边界一致;
5. **双框架版本配对**:Kueue + Koordinator 的 CRD/控制器版本兼容需实测(kind 单集群即可),
   是阶段 B 的第一个前置条件。

## 8. 落地路径(阶段 B 提案)

```
阶段 B1(环境):kind/k3d 起 Kueue + Koordinator,跑通最小 CRD 生命周期
阶段 B2(控制器):薄 controller 接一个决策切片(如"潮汐 → Kueue 优先级抢占落地")
阶段 B3(验证):同 v1 的 mock 潮汐,对比"决策序列"在真集群的执行结果
红线:B 不改调度算法,只验证翻译层正确;算法回归仍用 mock 135 tests
```

**前置条件**:在有卡/无卡 AutoDL 或本地 k3d 上进行;每步时间盒化。

## 9. 结论

> 算法层以确定性 mock 验证 135 个测试(含预测敏感性实验 E);平台层不重复造轮子,
> 用 Kueue 管配额准入、Koordinator 管节点弹性,只写一个薄 controller 把决策翻译成
> 真实原语(训练可抢占化 + 优先级抢占 + colocation)。真实 K8s 没有"运行时给 pod 减卡"
> 的原语,所以让渡粒度是 Workload 级、带 checkpoint 成本——这是执行层设计,也是
> mock 与平台之间标清的第一个诚实边界。
