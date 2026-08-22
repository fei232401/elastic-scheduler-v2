# 回归与复现性报告（Phase 3.3 T11）

> 基线态最终回归：**测试全绿 + Mock 字节级确定性 + Local 真实测量噪声容差**。
> 三个证据各自独立，均在本报告日期于最终提交上现场重跑（非引用旧存档）。
> 证据等级见 [`docs/evidence_matrix.md`](evidence_matrix.md)。

---

## 1. 回归：全量测试

```bash
python -m pytest tests/ -q
# 135 passed in 30.77s（1 warning：pynvml 弃用提示，非失败）
```

135 个测试全部通过（121 既有 + 14 个 traffic 测试，纯新增、零改动既有断言）。

## 2. Mock 确定性：字节级双跑

**方法（基线态，干净 `results/`）**：

```bash
rm -rf results
python -m experiments.run --matrix   # 基准
cp -r results /tmp/baseline
python -m experiments.run --matrix   # 重跑
md5sum 比对两棵树全部文件
```

**结果**：`total=51 differing=0` —— 51 个文件 md5 **全部一致**（含 PNG 二进制）。

51 个文件构成 = 3 实验（A/B/C）× 4 策略 ×（`metrics.csv` + `decisions.csv` + `summary.json`）+ 每实验 5 张图：

| 类别 | 每实验 | ×3 实验 |
|---|---|---|
| metrics / decisions / summary（数据） | 4 策略 × 3 = 12 | 36 |
| plots（PNG，二进制） | 5 | 15 |
| **合计** | 17 | **51** |

确定性来源：全 Mock 组件 seed 42（`engine.py` 入口固定 `random.seed(42); np.random.seed(42)`）；
纯数学模型无 wall-clock 依赖。决策日志逐条一致 → 下游任何数字（SLO/P95/切换/进度）可逐文件复现。

**与 T1 审计"66 文件"的口径说明**：T1 审计（`docs/code_audit.md` §4）对**当时已包含早期实验遗留输出的
results/ 整树**做 diff，得 66 文件全同；本报告从**干净 results/** 按 README 文档命令重跑，`--matrix`
自身产出 51 文件全同。两条测量都成立；**基线态可复现口径以本报告 51 为准**（E4 证据复核）。

## 3. Local 真实测量噪声容差

Local runtime 含真实 wall-clock（真实 HTTP + 真实 PyTorch + 真实测量），**不可能也不应该字节级一致**。
T11 要验证的是：噪声在文档声明的容差内，且**所有确定性信号（决策/上报比率）跨次一致**。

**方法**：`python -m experiments.run_local --scheduler elastic --config config/local_experiment.yaml`
连续跑两遍，比对 `summary_elastic.json`。

**结果**：

| 字段 | Run 1 | Run 2 | 差异 |
|---|---|---|---|
| `training_final_progress`（真实样本） | 14397440 | 14368256 | **0.203%** |
| `training_avg_slowdown`（EMULATED 比率） | 1.671 | 1.671 | 0 |
| `resource_switch_count` | 2 | 2 | 0 |
| `slo_violation_ratio` | 1.0 | 1.0 | 0 |
| `inference_avg_p95`（ms） | 432.54 | 432.54 | 0（2 位小数） |

- 训练进度差 **0.203%**，落在已文档化的 <5% 噪声带内（reality_gap **MTG2**：真实测量噪声，非调度效果）。
- **决策（2 次让渡）、EMULATED 降速比率、SLO 违约、P95 全部跨次一致** —— 调度行为在真实 runtime 上同样可复现。
- 选择的策略：elastic（主策略，2 次切换最富信息量）；static/network_aware 为 0 切换平凡路径，hard 已在
  Phase 3.2 双跑中验证（decision 一致）。

## 4. 结论与证据等级

| 声明 | 证据 |
|---|---|
| Mock 实验确定性（同 seed 双跑 51 文件 byte-identical） | **A**（本报告现场复测；E4 复核） |
| 回归：135 测试全绿 | **A**（本报告现场重跑） |
| Local 决策跨次一致（切换/SLO/P95/上报比率） | **A**（本报告现场复测） |
| Local 进度含 <5% 真实噪声（progress 差 0.2%，MTG2） | **A**（本报告现场复测） |

> 全部数字来自本文所列命令，未对任何实验输出做修改（§28）。复现材料：README §六 + 本文命令；
> 任一读者在 `pip install -r requirements.txt` 后可直接复现。
