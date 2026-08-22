# L3 M1 · 万卡骨架 + Kueue 弹性机制验证结果

> **日期**:2026-08-22 ｜ **环境**:AutoDL(128 vCPU / 1007 GB)· RTX 4090 无卡模式(纯 CPU)
> **结果**:✅ 1250 假节点 × 8 GPU = **10,000 卡**模拟集群 + Kueue v0.19.2 控制器 + 弹性机制双向验证通过

---

## 1. 最终状态快照

| 组件 | 版本/形态 | 状态 |
|---|---|---|
| 集群 | KWOK v0.8.0 `--runtime binary` | ✅ 1250 节点全 Ready,GPU 容量 10,000 |
| 控制面 | 真实 etcd 3.6.10 / kube-apiserver / kube-controller-manager / kube-scheduler v1.36.1 | ✅ 真进程 |
| Kueue | v0.19.2 从源码编译(Go 1.24.4→自动切 1.26.7 toolchain) | ✅ 18 controllers running |
| 弹性机制 | `kueue.x-k8s.io/elastic-job` + `ElasticJobsViaWorkloadSlices`(v0.18 起 beta 默认开) | ✅ 双向验证 |
| 配额对象 | ResourceFlavor `gpu-flavor` + ClusterQueue `cq-train-infer`(nominal 10k GPU)+ LocalQueue | ✅ Active |

## 2. M1 关键决策与踩坑(深入记录)

### 2.1 下载绕墙
- dl.k8s.io body 被限速(~38KB/s 单连接),release 二进制(github)被墙 → **ghfast.top 代理**下载 kwok/kwokctl;
- k8s 控制面二进制用 **aria2 -x16 多连接**破单连接限速;kubectl 损坏残档(6.2MB)→ 段错误,换回完整版;
- Kueue 无独立 release 二进制 → **Go 编译**(`go install sigs.k8s.io/kueue/cmd/kueue@v0.19.2`,GOPROXY=goproxy.cn)。

### 2.2 Kueue 独立二进制跑通(免 Docker/cert-manager)——3 个 TLS 证书点
Kueue 控制器作为裸进程跑(无容器、无 secret 挂载)时,`internalCertManagement` 的 bootstrap
会永久阻塞(它等 certDir 里出现证书文件,而 rotator 不写本地文件)。**绕开方案**:
1. `internalCertManagement.enable: false` → 跳过 bootstrap/rotator;
2. 自签证书补 3 个路径:`/tmp/k8s-webhook-server/serving-certs/`(webhook)、
   `/etc/kueue/metrics/certs/`(metrics)、`/visibility/`(visibility server);
3. **删 2 个 webhook configuration**(否则 failurePolicy=Fail 且 service 无 endpoint → 建对象被拒);
4. **5 个 CRD 的 conversion 策略 Webhook→None**(同因,否则 Workload 读写触发转换调用失败)。
> 诚实:`manifests.yaml` 里默认带 cert-manager + Deployment(容器形态),本方案是为"裸进程+模拟集群"裁剪,
> 生产仍是标准 Helm 安装;讲清这是测试基座裁剪,不是 Kueue 能力缺陷。

### 2.3 配额字段修正
- v0.19.2 ClusterQueue 的 ResourceQuota **没有 `guaranteedQuota`**(我早期记忆过时),只有
  `nominalQuota` + `lendingLimit`(cohort 借用)。弹性由 annotation 触发,非配额字段。
- 大 CRD(workloads)用 `kubectl apply` 因 `last-applied-configuration` 注解超 256KB 失败 →
  必须 `kubectl apply --server-side=true`。
- 扩展资源 nvidia.com/gpu 必须同时写 request + limit(不可超卖)。

## 3. 弹性机制验证证据(Kueue 原生,非自研)

弹性 Job:`elastic-long-job`,注解 `kueue.x-k8s.io/elastic-job: "true"`,
label `kueue.x-k8s.io/queue-name: lq-train-infer`,每 pod 1 GPU。

** 提交 → 被 admit → pod 调度到 GPU 假节点**
```
NAME                         QUEUE            RESERVED IN      ADMITTED
job-elastic-long-job-97196   lq-train-infer   cq-train-infer   True
# pod: elastic-long-job-* 调度到 kwok-gpu-0000/0285/0500/0750...
```

** 并行度下调 20→10(原地改,不新建切片)**
```
job-elastic-long-job-97196   True   10     ← pod 计数 20→10
```

** 并行度上调 10→30(新建替换切片,旧切片链上标注)**
```
NAME                         ADMITTED   PODS   REPLACES
job-elastic-long-job-97196   True       10     <none>                                  ← 旧
job-elastic-long-job-cb238   True       30     default/job-elastic-long-job-97196       ← 新切片,链上
# 新切片注解: workload-slice-name=job-elastic-long-job-97196 (链根)
#            workload-slice-replacement-for=default/job-elastic-long-job-97196
```

**这正是 L3 要的平台翻译**:决策引擎的"训练 6→4"→ controller 把训练 Job parallelism 下调 →
Kueue 弹性机制真实执行配额释放 → 空出 GPU 给推理。机制验证通过,进入 M2 薄 controller。

## 4. 诚实边界

- 节点是假的(KWOK 只维护状态,不跑容器);GPU 容量是声明式 capacity,不是真实卡;
- Kueue 是裁剪形态跑(webhook/conversion 关掉),验证的是**核心配额+弹性逻辑**,不是完整部署;
- 12 万卡声明 = "决策逻辑 + 平台原语在模拟集群上成立",生产可靠性不在 L3 范围(设计文档 §3)。

## 5. 复现命令(全部在 AutoDL)

```bash
# 集群(本地二进制,绕墙)
kwokctl create cluster --name l3 --runtime binary \
  --kube-apiserver-binary /opt/k8s-bin/kube-apiserver \
  --kube-controller-manager-binary /opt/k8s-bin/kube-controller-manager \
  --kube-scheduler-binary /opt/k8s-bin/kube-scheduler \
  --etcd-binary /opt/k8s-bin/etcd --kwok-controller-binary /usr/local/bin/kwok
kubectl --kubeconfig /root/.kwok/clusters/l3/kubeconfig.yaml apply -f /opt/k8s-bin/nodes/   # 1250 假节点
# Kueue 控制器(免容器)
/root/go/bin/kueue --kubeconfig $K --config /opt/k8s-bin/kueue/l3-config.yaml &
# 配额 + 弹性 Job(本目录 clusterqueue.yaml / elastic-job-demo.yaml)
kubectl apply -f integration/l3/clusterqueue.yaml
kubectl apply -f integration/l3/elastic-job-demo.yaml
# 弹性验证
kubectl patch job elastic-long-job --type merge -p '{"spec":{"parallelism":30}}'
kubectl get workloads.kueue.x-k8s.io -o wide
```
