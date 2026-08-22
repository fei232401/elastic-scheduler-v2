# Elastic Training-Inference Scheduler — 训推弹性资源调度 Sandbox（规格第一版）

> 来源：`第一版.docx`（2026-08-15 完整版）转档。本项目 = 潮汐流量训推过渡调度器，独立于 CyberRouter+HeteroServe 合并体。
> 当前开发范围：**MVP Core（Step 1-12，GPU-only，Static + Elastic）**。

---

# Elastic Training-Inference Scheduler
## 训推弹性资源调度与网络感知调度 Sandbox
**Project Type**: AI Infra / Resource Scheduling / LLM Infrastructure
**Target Environment**: Personal PC / Mock Cluster
**Implementation Strategy**: Mock-first, real workload later
**Primary Goal**: Reproduce the engineering behavior of a shared Training + Online Inference cluster and validate elastic resource scheduling.
---
> **[MVP 实施策略声明]**
> 本项目定位为**最小可用体（MVP）**，目标是用最短工程路径验证"逐级弹性让渡 GPU 是否可行"这一核心工程直觉。
> 因此，虽然下文完整列出了四种 Scheduler 和 Network 模块，但**实际编码实施时，强烈建议按两阶段推进**：
> - **阶段一（MVP Core）**：仅实现 **Static Scheduler** 和 **Elastic Scheduler**，完成 Step 1～Step 12，跑通完整训推潮汐周期，生成 GPU / QPS / P95 / Training throughput 四张核心图。
> - **阶段二（迭代增强）**：在阶段一稳定后，再加入 **Hard Preemption**、**Network 模块** 和 **Network-aware Scheduler**。
>
> 这样能保证在 2~3 个晚上内看到第一个可信的工程结果，而不会陷入四种策略并行开发的框架复杂度中。
---
## 1. 项目目标
构建一个可以在个人电脑上运行的 Training-Inference Elastic Scheduler Sandbox。
模拟真实 AI 集群中的以下场景：
```
Mock GPU Cluster
│
┌──────┴──────┐
↓             ↓
Training      Inference
Workload      Workload
│             │
└──────┬──────┘
↓
Elastic Scheduler
│
┌──────┴──────┐
↓             ↓
GPU           Network
Allocation    Allocation
│             │
└──────┬──────┘
↓
Metrics System
```
系统必须能够模拟：
- Training 长时间运行。
- Online Inference 流量具有潮汐变化。
- Inference 高峰时需要更多 GPU。
- Scheduler 从 Training 回收部分 GPU。
- Training 不直接停止，而是逐级降级。
- Inference 高峰结束后，Training 恢复资源。
- Training 和 Inference 存在网络通信需求。
- GPU 资源调整可能导致网络通信压力变化。
- 网络可能成为新的瓶颈。
- Scheduler 可以感知 GPU + Network 状态。
- 最终通过实验数据比较不同调度策略。
---
## 2. 核心问题
项目不是为了证明一个全新的学术理论。
核心工程问题是：
> 在 Inference 流量存在潮汐变化的共享 GPU 集群中，是否可以让 Training 作为弹性 workload，在不违反 Inference SLO 的情况下逐级让渡 GPU 资源，从而减少资源浪费和 Training 的不必要中断？
第二阶段进一步研究：
> 如果 GPU 和网络资源同时受到限制，仅考虑 GPU 的调度是否会导致错误决策？Network-aware Scheduler 是否能够改善训推资源转换过程中的性能？
---
## 3. 必须实现的三种 Scheduler
系统必须至少实现三个策略。
### 3.1 Static Scheduler
固定资源：
```
Total GPU = 8
Training = 6
Inference = 2
```
整个实验过程中不改变。
作用：
- Baseline。
### 3.2 Hard Preemption Scheduler
Inference 高峰时直接大量回收 Training GPU。
例如：
```
正常： Training = 6  Inference = 2
Inference 高峰： Training = 0  Inference = 8
高峰结束： Training = 6  Inference = 2
```
作用：
- 模拟传统粗粒度抢占/隔离式策略。
### 3.3 Elastic Scheduler
我们的主要方案。
资源逐步调整：
```
正常： Training = 6  Inference = 2
Inference 上升： Training = 5  Inference = 3
继续上升： Training = 4  Inference = 4
继续上升： Training = 3  Inference = 5
```
流量下降：
```
Training = 4  Inference = 4
Training = 5  Inference = 3
Training = 6  Inference = 2
```
禁止直接从 6 GPU → 0 GPU，除非进入紧急保护状态。
> **[MVP 实施建议]** 
> 在阶段一（MVP Core）中，优先实现 **Static** 和 **Elastic** 两种即可。Hard Preemption 可在阶段二作为"暴力对比"加入。代码设计时请预留 Scheduler 基类（`base.py`），方便后续插拔。
---
## 4. Mock Cluster
第一版统一使用：
```
TOTAL_GPUS = 8
```
GPU 不需要真实存在。
将 GPU 建模成：
```
GPU Resource Token
```
例如：
```
GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  GPU 6  GPU 7
```
Scheduler 管理这些逻辑资源。
---
## 5. Training Workload
Training 不需要第一阶段真正训练 LLM。
第一版实现成：
```
MockTrainingJob
```
具有以下状态：
```
RUNNING
DEGRADED
PAUSED
COMPLETED
FAILED
```
核心参数：
```
training:
total_work: 100000
initial_gpu: 6
min_gpu: 1
checkpoint_interval: 300
```
Training 每个 simulation tick 根据 GPU 数量推进进度。
例如：
```
1 GPU → throughput = 15
2 GPU → throughput = 29
3 GPU → throughput = 41
4 GPU → throughput = 52
5 GPU → throughput = 62
6 GPU → throughput = 71
7 GPU → throughput = 79
8 GPU → throughput = 85
```
注意：
- 这些只是第一版 Mock 参数。
- 代码必须把它们放进配置文件，不能硬编码。
> **[MVP 工程修正：亚线性吞吐缩放]**
>
> 上述查表法可用，但为了更贴近真实分布式训练"边际收益递减"特性，建议在配置中改用幂律公式替代硬编码表，只需一行改动：
>
> ```yaml
> training:
>   base_throughput: 15   # 单卡基准吞吐
>   scaling_exponent: 0.85  # 亚线性缩放指数
```
计算公式：throughput = base_throughput * (gpu_count ** scaling_exponent)
· 6卡：15 * (6^0.85) ≈ 15 * 4.57 ≈ 68.5（接近原表 71）
· 3卡：15 * (3^0.85) ≈ 15 * 2.54 ≈ 38.1（接近原表 41）
好处是：从 6 卡降到 3 卡，吞吐降为原来的 ~55% 而非 50%，单卡效率随卡数减少而提升，这会让 Scheduler 在降级时"负罪感"更小，更愿意让出 GPU；反之恢复时收益更明显。这更贴近真实工程感知，且避免硬编码表带来的维护负担。
---
6. Training 需要模拟的真实行为
Training 不能简单地：
```
GPU * constant = throughput
```
需要体现分布式训练的基本特征：
· GPU ↑ → throughput ↑ 但边际收益下降
同时：
· GPU ↓ → throughput ↓ → training completion time ↑
因此 Scheduler 减少 Training GPU 后：
· Training 不停止
· Training progress 继续增加
· 但 throughput 下降
这是整个项目最重要的行为之一。
---
7. Inference Workload
实现：
```
MockInferenceService
```
状态：
```
RUNNING
OVERLOADED
RECOVERING
```
输入：
· QPS
输出：
· throughput
· queue_length
· latency P95
· P99
· SLO status
---
8. Inference Capacity
第一版使用配置：
```
inference:
  capacity_per_gpu: 10
  slo:
    p95_ms: 300
```
例如：
```
1 GPU → 10 QPS
2 GPU → 20 QPS
3 GPU → 30 QPS
...
```
但是需要加入简单的非线性延迟模型。
当：
```
incoming_qps <= capacity
```
则：
```
P95 ≈ 150~250ms
```
当：
```
incoming_qps > capacity
```
则：
```
queue_length ↑
P95 ↑↑
P99 ↑↑
```
例如：
```
Demand = 20  Capacity = 30  P95 = 180ms  SLO = PASS
```
而：
```
Demand = 40  Capacity = 30  P95 = 450ms  SLO = FAIL
```
---
9. Inference Traffic Generator
必须实现真实的潮汐流量周期。
不要只做随机 QPS。
默认 scenario：
```
0 - 10 min   LOW
10 - 20 min  RISING
20 - 40 min  PEAK
40 - 50 min  FALLING
50 - 70 min  LOW
```
例如：
```
Time    QPS
0       10
5       12
10      18
15      25
20      35
25      45
30      55
35      60
40      55
45      40
50      25
60      15
70      10
```
必须支持：
· constant
· step
· sine
· spike
· custom
[MVP 工程修正：过载防抖（Hysteresis）]
真实系统中，QPS 在 capacity 边界来回跳动会导致 P95 剧烈震荡，从而引发 Scheduler 频繁无效决策。
建议在 Inference 模型中增加两个简单计数器，实现滞后恢复：
```python
overload_counter: int = 0      # 连续过载 tick 数
recovery_counter: int = 0      # 连续健康 tick 数
```
触发扩容条件：连续 3 个 tick（15秒）P95 > SLO，才认定为真正过载。
触发缩容条件：连续 5 个 tick（25秒）P95 < SLO，才认定为真正恢复。
这两个阈值配置在 config/default.yaml 中：
```yaml
inference:
  overload_trigger_ticks: 3
  recovery_trigger_ticks: 5
```
这是工程上最朴素的防抖手段，比滑动窗口更简单（只需两个整数），能有效避免"一触即发、一放就缩"的震荡。
---
10. Scheduler
Scheduler 每个 tick 执行一次。
默认：
```
SCHEDULER_INTERVAL = 5 seconds
```
流程：
1. Collect metrics
2. Check inference SLO
3. Check inference demand
4. Check training state
5. Determine whether Training can be degraded
6. Calculate resource change
7. Apply resource allocation
8. Record decision
9. Repeat
---
11. Elastic Scheduler 的核心规则
第一版不要做机器学习，不要做复杂优化算法。
使用 Rule-based Controller。
Rule 1：Inference SLO violation
如果：
```
Inference P95 > SLO
```
则：
```
if Training GPU > Training minimum GPU:
    reclaim 1 GPU
```
Rule 2：Inference approaching SLO
如果：
```
P95 > 80% * SLO
```
则：
· prepare resource reclaim
· 但不要立即大规模抢占。
Rule 3：Inference healthy
如果：
```
P95 < 60% * SLO
```
并且 Training 当前处于 degraded：
```
return 1 GPU to Training
```
Rule 4：防止震荡
禁止：
```
+1 GPU  -1 GPU  +1 GPU  -1 GPU
```
每次资源变化后必须保持：
```
COOLDOWN = 30 seconds
```
[MVP 工程修正：加强防抖 —— 使用最小持有时长替代冷却期]
原 COOLDOWN = 30s 在 5 秒一个 tick 下仅相当于 6 个 tick。如果流量持续微涨，6 个 tick 后 Scheduler 又会再抢 1 卡，可能导致 10 分钟内抢 10 次，Training 一直在降级，根本没机会稳定运行。
建议将冷却逻辑改为 MIN_GPU_HOLD_DURATION：
```yaml
scheduler:
  min_gpu_hold_duration: 120   # 单位：秒
```
即：一旦 Training 被降级（减少了 GPU），在 120 秒（24 个 tick）内禁止再次降级，即使 P95 仍然超限也不动。
对应代码逻辑：
```python
if last_degraded_at and (now - last_degraded_at) < MIN_GPU_HOLD_DURATION:
    # 仍在保护期内，不允许再次降级
    return
```
这模拟了真实工程中"让系统先稳定下来"的常识。验证完"逐级让渡是否可行"之后，再逐步缩短这个窗口来测试极限。
Rule 5：逐级调整
默认：
```
MAX_SCALE_STEP = 1 GPU
```
一次最多：
```
Training -1
Inference +1
```
---
12. Training 的"可降级"机制
第一版不要把"deadline"作为绝对数学约束。
使用：
```
training utility / feasibility
```
来判断。
Training 每次向 Scheduler 提供：
```
progress
remaining_work
current_throughput
estimated_completion_time
current_gpu
```
例如：
```
remaining_work = 50000
GPU = 4
throughput = 52
estimated_remaining_time = 50000 / 52
```
Scheduler 可以知道：
· 再拿走一个 GPU 后，Training 会慢多少。
第一版可以配置：
```
training:
  min_gpu: 1
  max_allowed_slowdown: 3.0
```
后续再把它升级成更复杂的 utility model。
---
13. Network 模块
这是项目第二个核心部分。
不要真的要求个人电脑拥有 RDMA / InfiniBand。
第一阶段：
· Mock Network。
建立：
```
NetworkResource
```
参数：
```
network:
  total_bandwidth_mbps: 10000
  latency_ms: 1
```
两个 workload 都消耗网络：
```
Training
  └─ communication bandwidth
Inference
  └─ KV / request / response traffic
```
---
14. Training Network Demand
Training GPU 越多：
```
GPU ↑ → distributed communication ↑ → network demand ↑
```
例如：
```
1 GPU → 0 Mbps
2 GPU → 500 Mbps
3 GPU → 900 Mbps
4 GPU → 1400 Mbps
5 GPU → 2000 Mbps
6 GPU → 2700 Mbps
```
同样只是 Mock 参数。
---
15. Inference Network Demand
Inference：
```
QPS ↑ → network traffic ↑
```
例如：
```
10 QPS → 200 Mbps
20 QPS → 400 Mbps
40 QPS → 800 Mbps
60 QPS → 1200 Mbps
```
---
16. Network Congestion
计算：
```
network_demand = training_network_demand + inference_network_demand
```
如果：
```
demand <= capacity
```
则：
```
network = healthy
```
如果：
```
demand > capacity
```
则：
```
queue ↑
latency ↑
packet delay ↑
```
最终影响：
· Inference latency
· Training throughput
这样就建立：
```
GPU resource ↓
workload scale ↓
network demand ↓
network congestion ↓
latency / throughput
```
---
17. Network-aware Scheduler
第一版：
· GPU-aware Scheduler
第二版：
· GPU + Network-aware Scheduler
Network-aware Scheduler 在做资源调整前必须检查：
· 如果增加 Inference GPU：
  · Inference network demand 是否增加？
  · Training 网络需求是否因为 GPU 变化而改变？
  · 总 network demand 是否超过 capacity？
如果：
```
GPU 有资源
但 Network 已经饱和
```
则：
· 不能继续增加 Inference GPU。
这是第二阶段最重要的实验。
[MVP 工程修正：必须计算"缩 Training 释放的带宽"]
如果忽略这一点，Network-aware Scheduler 在 Network Bound 场景下会永远觉得网络不够，永远不敢扩容，实验直接失效。
正确逻辑：缩 Training 本身会释放大量带宽（因为 GPU 减少 → 分布式通信减少），这部分释放出来的带宽恰好可以被 Inference 增量使用。
决策公式应为：
```
# 假设从 6 卡缩到 5 卡，Training 带宽从 2700 Mbps 降到 2000 Mbps
released_bandwidth = old_training_bandwidth - new_training_bandwidth
inferred_delta_bandwidth = new_inference_gpu * per_gpu_infer_bw - old_inference_gpu * per_gpu_infer_bw
if (inferred_delta_bandwidth - released_bandwidth) <= available_headroom:
    # 允许扩容
else:
    # 网络会超载，禁止扩容
```
这一行公式是 Network-aware 策略有效性的基石，务必实现。
---
18. Network 实验场景
至少设计以下场景：
Scenario A：GPU bound
· GPU 紧张
· Network 富余
预期：
· GPU-aware 和 Network-aware 表现接近。
Scenario B：Network bound
· GPU 有剩余
· Network 饱和
预期：
· GPU-only Scheduler 可能继续扩张 Inference，导致网络拥塞。
· Network-aware Scheduler 应该停止扩张。
Scenario C：GPU + Network 双瓶颈
· GPU 紧张
· Network 也紧张
观察：
· Scheduler 如何在两个资源之间做 trade-off。
---
19. 完整模拟周期
系统启动：
```
Total GPU = 8
Training = 6
Inference = 2
```
Phase 1：低峰
```
Inference QPS = 10
状态： Training = 6  Inference = 2
```
Phase 2：流量上升
```
QPS: 10 → 15 → 20 → 25 → 30 → 35
```
Scheduler 开始观察：
```
P95 ↑
```
然后：
```
Training 6 → 5  Inference 2 → 3
```
继续：
```
Training 5 → 4  Inference 3 → 4
```
Phase 3：Inference Peak
例如：
```
QPS = 60
```
Scheduler：
```
Training = 3  Inference = 5
```
Training：
· 仍然 RUNNING
· throughput ↓
· progress ↑
Inference：
```
P95 <= SLO
```
Phase 4：Network Bottleneck
人为增加 Training communication：
```
Training network demand ↑
```
最终：
```
GPU capacity: OK
Network: SATURATED
```
此时测试：
· GPU-only Scheduler 是否会错误地继续增加 Inference GPU，导致 Network congestion + P95 ↑
· Network-aware Scheduler 是否能够停止扩容
Phase 5：Inference 下降
```
60 → 50 → 40 → 30 → 20 → 10 QPS
```
Scheduler：
```
Inference 5 → 4 → 3 → 2
Training 3 → 4 → 5 → 6
```
Training 恢复。
---
20. 最终实验必须比较什么？
必须统一 workload、统一初始状态。
比较：
```
Static Scheduler
vs
Hard Preemption Scheduler
vs
GPU-aware Elastic Scheduler
vs
GPU+Network-aware Elastic Scheduler
```
[MVP 实施建议]
阶段一只比较 Static 和 Elastic（GPU-aware） 两种。Hard Preemption 和 Network-aware 在阶段二加入。这能让你的 MVP 代码量缩减约 40%，但核心结论（"逐级弹性调度优于固定分配"）已经可以完整证明。
---
21. 必须采集的 Metrics
Inference
· QPS
· throughput
· P50
· P95
· P99
· queue_length
· SLO violation count
· SLO violation duration
Training
· GPU allocation
· throughput
· progress
· remaining_work
· estimated_completion_time
· degradation_duration
· pause_count
· resume_count
Cluster
· GPU utilization
· Training GPU
· Inference GPU
· unused GPU
· resource_switch_count
· resource_switch_frequency
Network
· total_bandwidth
· training_bandwidth
· inference_bandwidth
· network_utilization
· network_queue
· network_latency
· congestion_duration
---
22. 最终必须生成的图
至少生成以下图表。
图 1：Traffic
· QPS vs Time
图 2：GPU allocation
· Training GPU
· Inference GPU vs Time
图 3：Inference P95
· P95 vs Time
· 并画 SLO threshold
图 4：Training throughput
· Training throughput vs Time
图 5：GPU utilization
· GPU utilization vs Time
图 6：Network utilization
· Network utilization vs Time
图 7：Scheduler decision
记录：
```
timestamp
old_training_gpu
new_training_gpu
old_inference_gpu
new_inference_gpu
reason
inference_p95
network_utilization
```
最终生成：
· Scheduler decision timeline
---
23. 最终需要回答的工程问题
实验结束后，README 必须能够回答：
Q1 Elastic Scheduler 是否减少了 GPU 空闲？
Q2 Inference SLO 是否得到满足？
Q3 Training 是否可以在不停止的情况下承受资源下降？
Q4 相比 Hard Preemption，Elastic Scheduler 是否减少 Training disruption？
Q5 Inference 流量下降后，Training 是否能够自动恢复资源？
Q6 GPU-only Scheduler 遇到 Network Bottleneck 会发生什么？
Q7 Network-aware Scheduler 是否能够识别：GPU available ≠ system actually capable of scaling？
---
24. 工程目录
Agent 最终应该构建：
```
elastic-scheduler-v2/
│
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
│
├── config/
│   ├── default.yaml
│   ├── gpu_bound.yaml
│   ├── network_bound.yaml
│   └── mixed_bound.yaml
│
├── src/
│   ├── cluster/
│   │   ├── gpu_pool.py
│   │   └── resource_manager.py
│   │
│   ├── workload/
│   │   ├── training.py
│   │   ├── inference.py
│   │   └── traffic.py
│   │
│   ├── network/
│   │   ├── network.py
│   │   └── congestion.py
│   │
│   ├── scheduler/
│   │   ├── base.py
│   │   ├── static.py
│   │   ├── preemptive.py
│   │   ├── elastic.py
│   │   └── network_aware.py
│   │
│   ├── metrics/
│   │   ├── collector.py
│   │   └── exporter.py
│   │
│   └── simulator/
│       ├── engine.py
│       └── clock.py
│
├── experiments/
│   ├── run_static.py
│   ├── run_preemptive.py
│   ├── run_elastic.py
│   └── run_network_aware.py
│
├── tests/
│   ├── test_gpu_pool.py
│   ├── test_training.py
│   ├── test_inference.py
│   ├── test_scheduler.py
│   └── test_network.py
│
├── results/
│   └── plots/
```
---
25. 验收标准
Agent 不能只做到"代码能运行"。
必须满足以下 Acceptance Criteria。
AC-01 系统能够启动 Mock Cluster：8 GPU
AC-02 Training 能够持续运行并产生 progress。
AC-03 Inference QPS 能够按照预设周期变化。
AC-04 Inference 高峰能够触发资源调整。
AC-05 Elastic Scheduler 一次最多调整 1 GPU。
AC-06 Training 在资源减少后仍然继续运行。
AC-07 Inference 流量下降后 Training 能够恢复资源。
AC-08 Static / Preemptive / Elastic 三种策略都能够运行同一个 scenario。
AC-09 所有策略能够输出统一 metrics。
AC-10 Network congestion 能够被 Mock。
AC-11 GPU-only Scheduler 与 Network-aware Scheduler 能够运行相同 Network-bound scenario。
AC-12 最终自动生成：metrics.csv / summary.json / *.png
---
26. 第一阶段不要做的事情
非常重要。
Agent 禁止一开始就引入：
· Kubernetes
· vLLM
· NCCL
· RDMA
· InfiniBand
· Slurm
· Prometheus
· Grafana
· 真实 LLM
第一阶段全部 Mock。
原因很简单：
我们现在验证的是调度逻辑和工程行为，不是搭集群。
否则很容易最后变成："花三天解决 Docker/K8s 网络问题，但是还不知道 Scheduler 到底有没有效果。"
[MVP 补充]
阶段一（MVP Core）甚至可以先不做 Network 模块（Step 13 及以后），先把 GPU-only 的训推潮汐跑通。Network 是阶段二的增强项。
---
27. 第二阶段再逐渐真实化
完成 Mock 闭环后：
```
Mock Training → PyTorch Training
Mock Inference → 真实 inference workload
Mock Network → Linux network namespace → tc/netem → iperf3
Mock Metrics → Prometheus
Mock Cluster → Docker → Kubernetes
```
如果个人电脑 GPU/显存允许，再考虑：
· PyTorch + small model
而不是强行跑 LLM。
---
28. Agent 执行顺序
Agent 必须严格按照以下顺序工作：
Step 1 建立项目目录
Step 2 实现 Mock GPU Pool
Step 3 实现 Mock Training
Step 4 实现 Mock Inference
Step 5 实现 Traffic Generator
Step 6 实现 Simulation Clock
Step 7 实现 Static Scheduler
Step 8 实现 Hard Preemption Scheduler
Step 9 实现 Elastic Scheduler
[MVP 插入 Step 9.5：确定性随机种子]
在开始第一次实验之前，必须在模拟器入口处固定随机种子：
```python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```
这保证 Static / Elastic / Preemptive 跑的是完全一样的 QPS 时间序列（包括微小抖动），否则最终对比图会因为随机性差异而失去说服力。这一步零成本，但极其重要。
Step 10 实现 Metrics Collector
[MVP 执行建议：首次停止点]
完成 Step 10 后，先不要继续往下做 Step 11～Step 12。实际上，Step 11 和 Step 12 就是第一次完整实验。所以顺序是：
· Step 11 完成第一次完整训推潮汐实验（仅 GPU，无 Network）
· Step 12 生成 GPU / QPS / P95 / Training throughput 图
此时你应该停下来，观察 Static vs Elastic 的对比图。如果 Elastic 没有明显优于 Static，说明核心假设不成立，需要调整规则，而不是继续堆砌 Network 模块。
Step 13 加入 Mock Network
Step 14 制造 Network Bottleneck
Step 15 实现 GPU-only Scheduler
Step 16 实现 Network-aware Scheduler
Step 17 完成 GPU-bound / Network-bound / Mixed-bound 三组实验
Step 18 生成最终实验报告
---
29. Agent 最终必须给我的东西
当整个项目完成后，不要只告诉我：
"项目运行成功。"
必须输出：
1. 项目结构
2. 如何启动
例如：
```bash
python -m experiments.run_elastic --config config/default.yaml
```
3. 完整实验结果
```
Static  |  Preemptive  |  Elastic  |  Network-aware
```
4. Metrics 对比表
5. 所有关键曲线
6. 每个 Scheduler 为什么做出某次决策
例如：
```
t=120s  Inference P95 = 341ms  SLO = 300ms  Training GPU = 6
Decision: Training 6 → 5  Inference 2 → 3
Reason: Inference SLO violation
Training can be degraded
Network utilization = 42%
```
7. Failure cases
必须主动找：
· 什么时候 Elastic Scheduler 不如 Hard Preemption？
· 什么时候 Network-aware 不如 GPU-only？
· 什么时候资源震荡？
· 什么时候 SLO 仍然违约？
不要只寻找成功案例。
---
30. 项目的最终形态
最后我们希望看到的是这样一条完整链路：
```
┌────────────────────┐
│  Traffic Generator │
│  Low → Peak → Low  │
└─────────┬──────────┘
          │
          ↓
┌────────────────┐
│   Scheduler    │
└───────┬────────┘
        │
┌───────┴───────┐
↓               ↓
Training        Inference
Resource        Resource
│               │
↓               ↓
Training        Inference
Workload        Workload
│               │
└───────┬───────┘
        ↓
     Network
        │
┌───────┴───────┐
↓               ↓
Healthy         Congested
│               │
└───────┬───────┘
        ↓
     Metrics
        │
        ↓
┌──────────────────┐
│  Experiment /    │
│  Comparison      │
└──────────────────┘
```
最终我们要用一整个真实周期证明：
```
低峰
  ↓
Training 占更多资源
  ↓
Inference 流量上升
  ↓
Scheduler 感知
  ↓
Training 渐进降级
  ↓
Inference 获得 GPU
  ↓
Inference Peak
  ↓
Network 可能成为瓶颈
  ↓
Network-aware Scheduler 介入
  ↓
Inference 流量下降
  ↓
Training 恢复
  ↓
整个周期结束
```
这就是我们真正要落地的第一版完整工程项目。
特别强调：第一版的目标不是"模拟 H100 集群"，而是把真实集群中的因果关系完整跑通。 只要 Agent 最终能够回答"什么时候让 GPU、让多少、让了以后 Training 怎么变化、Inference 是否满足 SLO、网络是否成为新的瓶颈"，这个 Sandbox 就已经具有非常明确的 AI Infra 工程价值。
[MVP 最终建议]
如果你把这份文档交给 Agent，强烈建议在 Prompt 中明确说：
"先只实现 Step 1～Step 12，跑出第一轮完整的 GPU-only 训推潮汐实验；不要一次性把 Network / 四种策略全做完。"
这样你能在最短时间内看到第一个真正的工程结果，然后再决定是否值得投入 Network 和 Preemptive 的增强。这正是 MVP 的核心思想。
---
```
---

---

## 附录：MVP Core 开发范围声明

本规格为全量 30 节（含 Network/Preemptive/Network-aware，属阶段二）。第一版仅实现：
- Step 1-12：Mock GPU Pool / Training / Inference / Traffic / Clock / Static + Elastic Scheduler / Metrics
- 确定性种子（Step 9.5）；四张核心图（Step 12）；metrics.csv + summary.json + decisions.csv
- **不做**：K8s / vLLM / NCCL / RDMA / Prometheus / 真实 LLM（spec §26）；Network 模块与其余 Scheduler（阶段二）
