# L3 · M3/M4 校准效果模型 + 三列矩阵验证(2026-08-22)

> **一句话**:把 L2 校准的真实 4090 常数喂进 L3 万卡模型,离线对比 static/elastic/predictive 三列;再上 KWOK+Kueue 现场跑 elastic/predictive,**发现"预测的优势在真实反馈滞后下反噬"**——离线 predictive 最优(14.2%),现场 predictive 反而差于 elastic(31.4% vs 27.6%)。

---

## 1. 校准配置(l3_calibrated.yaml,vs l3_default 只改 4 个常数)

| 参数 | default(mock) | calibrated(L2 锚点) | 来源 |
|---|---|---|---|
| `capacity_per_gpu` | 10 | **10(不变)** | L2 验证:1229.1 decode t/s ÷ 123.5 avg tokens ≈ 9.95 req/s |
| `p95_base_ms` | 200 | **10** | 4090 实测低载 TTFT(峰值保持下真实 ~10ms,mock 悲观 20×) |
| `p95_overload_ms` | 800 | **3000** | 4090 实测饱和 TTFT(保守下界,真机 up to 19.7s) |
| `slo_p95_ms` | 300 | **1000** | TTFT ≤ 1s,对齐链式闭环口径(非 mock 的 300ms) |

其余与 l3_default 一致:total_gpus=10000、train 初始 8000/min 4000、峰值 60k qps、840 ticks、seed 42、max_scale_step=500、min_hold=24、approach 0.8 / healthy 0.6。

---

## 2. 三列矩阵 · 离线(理想反馈,offline_replay.py)

同一决策引擎 + 同一校准配置,仅分配"同 tick 即时生效"(理想假设):

| 策略 | SLO 违约 | peak P95 | 切换 | GPU-hour |
|---|---|---|---|---|
| static | **44.8%** | 3000ms | 0 | 11666.7 |
| elastic | 18.2% | 2362ms | 12 | 11666.7 |
| predictive | **14.2%** | 2084ms | 30 | 11666.7 |

→ 理想反馈下:elastic 大幅改善(44.8→18.2),预测再补 4pt(→14.2),代价切换 2.5×。

## 3. 三列矩阵 · 现场(真实反馈,m2_thin_controller.py + KWOK/Kueue)

| 策略 | SLO 违约 | peak P95 | 切换 | GPU-hour |
|---|---|---|---|---|
| elastic | **27.6%** | 2408ms | 17 | 11708.9 |
| predictive | **31.4%** | 3000ms | 42 | 11023.2 |

→ 真实反馈下:两者都劣于理想值(反馈滞后结构性,见 M2),**且 predictive 反超 elastic 变差**(27.6→31.4)。

---

## 4. 核心发现:预测优势在真实反馈下反噬

**现象**:离线 predictive 比 elastic 好 4.0pt;现场反转为差 3.8pt。峰值 P95 现场 predictive 顶到饱和 3000ms。

**机理**(两层,都来自反馈回路):
1. **反馈滞后吃掉提前量**:预测预夺本应"提前让渡→峰值到达时已就位"。但现场决策要经过 patch → Kueue → status 更新才有真实时间常数,让渡生效慢于预测窗口 → 预夺的"提前"被滞后抵消,峰值还是追不上。
2. **切换成本在真实系统里放大**:predictive 42 次切换(elastic 17 次,2.5×)。理想反馈下切换是免费的(同 tick 生效);现场每次切换都有执行时延 + min_hold=24tick 序列化,42 次切换把"500 GPU 锯齿"切得更碎,决策抖动直接变成 SLO 违约。
3. **误差也被滞后放大**:predictive 依赖 p95→load_proxy 预测,但喂给预测器的 p95 本身已滞后 → 预测的输入就是过时信号,预测误差叠加反馈滞后,双重失真。

**对比 M2 的发现**:M2 证明"反馈滞后让 elastic 从 3.5%→36.9%";M3/M4 进一步证明"滞后同样吃掉预测的收益,甚至反噬"。**结论:真实系统里,预测器不能只在理想反馈假设下验证;反馈回路的时间常数是预测收益的第一前提**。这为 v2 指向:预测必须建模执行时延,或改为"预测 + 预执行确认"。

**诚实边界**:
- 只有 elastic/predictive 上了现场,static 现场未跑(离线即可,现场跑无收益,决策引擎不动作)。
- GPU-hour 差异小(live 11023 vs 11667),弹性主要是 SLO 收益,降本在万卡 scale 上被调度开销抵消。
- 单次运行、seed 42、KWOK 不跑真实容器——延迟仍是 L2 校准模型推的,不是现场真实 vLLM 延迟。

---

## 5. 复现命令(AutoDL,仓库根目录)

```bash
# 离线三列矩阵(理想反馈)
python integration/l3/offline_replay.py --config integration/l3/l3_calibrated.yaml --scheduler static  --out integration/l3/results/calibrated_static.json
python integration/l3/offline_replay.py --config integration/l3/l3_calibrated.yaml --scheduler elastic   --out integration/l3/results/calibrated_elastic.json
python integration/l3/offline_replay.py --config integration/l3/l3_calibrated.yaml --scheduler predictive --out integration/l3/results/calibrated_predictive.json

# 现场 elastic / predictive(真实反馈,KWOK 10k + Kueue)
python integration/l3/m2_thin_controller.py --config integration/l3/l3_calibrated.yaml \
    --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml --out /opt/l3-controller/run_cal_elastic.json --tick-wall 1.0
python integration/l3/m2_thin_controller.py --config integration/l3/l3_calibrated.yaml \
    --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml --out /opt/l3-controller/run_cal_predictive.json --tick-wall 1.0 --scheduler predictive
```

现场文件已 scp 回 WSL:`integration/l3/results/run_cal_elastic.json` / `run_cal_predictive.json`。

---

## 6. 设计说明

> "我把 L2 校准的真实常数喂进万卡模型,三列矩阵在理想反馈下:static 44.8%→elastic 18.2%→predictive 14.2%,预测有效;但上 KWOK 真反馈后反了——elastic 27.6%、predictive 31.4%。原因:预测的提前量被反馈滞后吃掉,42 次切换在真实系统里被放大成抖动。这证明反馈回路时间常数是预测收益的第一前提,v2 的预测必须建模执行时延。"
