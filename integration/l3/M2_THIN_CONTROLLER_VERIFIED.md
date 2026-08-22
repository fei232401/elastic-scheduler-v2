# L3 M2 · 薄 Controller 最小闭环验证结果(决策引擎 → 真实 Kueue 弹性)

> 时间:2026-08-22 · 环境:AutoDL(KWOK 10k 集群 + Kueue v0.19.2 + kubeconfig `/root/.kwok/clusters/l3/kubeconfig.yaml`)
> 结论速览:**机制成立,但"真实反馈滞后"在万卡尺度暴露——同一引擎同配置,理想反馈 SLO 3.5% vs 真实反馈 36.9%,资源效率几乎不变(GPU-hour -0.8%)。代价全在 SLO。**

## 1. 运行矩阵(两种反馈回路)

| | 冒烟(smoke) | 全量(live, 真实反馈) | 离线(ideal, 对照) |
|---|---|---|---|
| 配置 | `l3_smoke.yaml`(池 200) | `l3_default.yaml`(池 10000) | 同 l3_default |
| 反馈 | 真实集群(KWOK) | 真实集群(KWOK) | 决策同 tick 即时生效 |
| ticks | 240 | 840 | 840 |
| SLO 违约 | 通过(机制验证) | **36.9%** | **3.5%** |
| 峰值 P95 | — | **753.1ms** | 337.3ms |
| 下移/上移 | — | 13 / 6 | 6 / 6 |
| 训练最低 | — | 5500(→8000 全归还) | 5000(→8000 全归还) |
| GPU-hour 总量 | — | 11568.1 | 11666.7(−0.8%) |

产物:`results/run_full.json`(live 记录)、`results/offline_replay.json`(理想反馈)、`results/full.log`。

## 2. 机制验证(最小闭环成立,讲这条链)

```
BurstGPT 潮汐(qps_at_second, seed42) 
  → 薄 controller 每 tick:读真实集群 actual(Job.status.ready / Deployment.availableReplicas)
  → 复用 MockInferenceService(容量/P95/防抖状态机) + MockTrainingJob(§12 slowdown 闸门)
  → ElasticScheduler.step(ctx) → ResourceDecision(零改动复用)
  → patch Job parallelism / Deployment replicas → Kueue ElasticJobsViaWorkloadSlices 真实执行
  → 下一 tick 读回 actual(真实反馈,非理想假设)
```

- **完整周期实证**:训练 8000 → 5500(Rule1 逐级让渡 13 次)→ **8000 全额归还**(Rule3,6 次),首让渡 t=182(min 15.2),末让渡 t=470(min 39.2),首归还 t=589(min 49.1)。让渡→归还闭环在 10k 池上真实跑通。
- **Kueue 弹性原语被真实驱动**:scale-up 走 workload slice replacement,scale-down 走 in-place pod 数更新——M1 验证过的机制,这次由决策引擎的 ResourceDecision 触发(不是手改)。
- **§12 闸门生效**:`would_allow_degradation` 检查 max_allowed_slowdown=2.0,训练降级始终在保护内。

## 3. 核心诚实发现(为什么 3.5% → 36.9%)

**唯一变量 = 反馈回路的时间常数。** 对比离线/现场:
- 离线(理想):决策 new_* 同 tick 生效 → 容量跟得上需求爬升,只有防抖+hold 窗口内的瞬时超 SLO(29 tick)。
- 现场(真实):每笔 ±500 被 `min_gpu_hold_duration_ticks=24`(2 分钟)序列化,叠加 3-tick 防抖(15s),
  → **容量以"500 GPU/2min"的锯齿追"1.7k qps/min"的爬升** → 峰值段持续欠容量。

具体数字(live 记录):
- 峰值需求-实际缺口最大 **2693 GPU**(min 34.8:需 6193,实际 3500)→ 过载比 ~1.77 → P95 指数区 753ms。
- 现场 max_infer 只到 4500(离线到 5000);训练最低只到 5500 而非 floor 4000——因为**追到一半潮汐已回落 + recovery debounce**,容量尚未给足需求已退。
- 违约窗口 = 全爬升+峰段(min 15.0 ~ min 41.2,310/840 tick)。
- 但 **GPU-hour 几乎不变**(−0.8%)、训练最终全额归还:滞后没有烧钱,代价集中在 SLO。

**一句话**:mock/离线的"即时生效"假设,掩盖了真实集群"决策→执行→生效"的时间常数;万卡尺度下 reactive 弹性永远追不上潮汐爬升斜率。这正是 Experiment E(预测前移)的价值点——且必须**双向预测**(上升段提前扩容,而非 E 目前主要防的下降段振荡),并配**更细步长/自适应 hold**。

## 4. 诚实边界

- 规模声明只到"决策逻辑 + 平台翻译在 10k 模拟集群成立",不冒充生产可靠性(KWOK 伪造节点/零网络)。
- 效果模型仍是 mock 值(`capacity_per_gpu=10`);L2 校准曲线替换 = M3。
- live 的 intended≠actual 仅 1 tick(KWOK Ready 近即时)——本结论是**决策粒度/防抖**造成的滞后,**不是**真实生产里 pod 拉起几十分钟那种滞后;生产只会更差。这是保守下限。

## 5. 复现命令(AutoDL,仓库根)

```bash
# 冒烟(池 200,先验证链路)
python integration/l3/m2_thin_controller.py --config integration/l3/l3_smoke.yaml \
    --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml --out /opt/l3-controller/run_smoke.json

# 全量(live, 10k 池, ~15 分钟)
python integration/l3/m2_thin_controller.py --config integration/l3/l3_default.yaml \
    --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml --out /opt/l3-controller/run_full.json --tick-wall 1.0

# 离线理想反馈对照(本地即可,不需要集群)
python integration/l3/offline_replay.py --config integration/l3/l3_default.yaml \
    --out integration/l3/results/offline_replay.json
```

## 6. 下一步

- **M3 效果模型**:把 mock `capacity_per_gpu=10` 换成 L2 实测校准曲线(4090 并发→TTFT/P95),看 SLO 结论是否被量化改变。
- **M4 三列矩阵**:static / elastic / elastic+预测(同 seed 公平对比)——E 的 EMAPredictor 需按本节发现改**双向预测**。
