# LAION100M HashPrune 策略参数扫描 Runbook

本 runbook 用于比较 HashSlot、DistanceFirst、SoftTier 与 HardTier 在各自调优参数下的构图与统一图搜索表现。实验固定 LAION100M 数据、seed 42、fanout 8/3/1、`R=64`、`Lbuild=100`、12-bit HashPrune、8 个线程和 16 GiB GPU workspace。所有图均使用 `rabitq1-adaptive-exact` 导航，并以原始 `f32` 向量完成最终精确排序。

## 冻结矩阵

正式执行包含 15 个不重复的构图配置：

| 顺序 | 配置 | 模式 | Q | SoftTier alpha |
|---:|---|---|---:|---:|
| 1 | HashSlot-Q1 | `rhalf-hash-slot-1` | 1 | 1.00 |
| 2 | DistanceFirst-Q2 | `rhalf-distance-first-q3` | 2 | 1.00 |
| 3 | DistanceFirst-Q3 | `rhalf-distance-first-q3` | 3 | 1.00 |
| 4 | HardTier-Q2 | `rhalf-hard-tier-q3` | 2 | strict |
| 5 | HardTier-Q3 | `rhalf-hard-tier-q3` | 3 | strict |
| 6--10 | SoftTier-Q2 | `rhalf-soft-tier-q3` | 2 | 1.05/1.10/1.20/1.40/2.00 |
| 11--15 | SoftTier-Q3 | `rhalf-soft-tier-q3` | 3 | 1.05/1.10/1.20/1.40/2.00 |

Q1 下 DistanceFirst、SoftTier 和 HardTier 退化为相同的单层选择，因此只构建一次 HashSlot-Q1。SoftTier alpha=1.0 与相同 Q 下的 DistanceFirst 等价，不重复构建；HardTier 保留为不可跨层的严格端点。

## 冻结输入

```text
base    /data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin
query   /data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin
GT      /data_nvme_kioxia/tyj/data/laion-512-100M/gt100.bin
sidecar /data_staging/wbh/unified_graph_rabitq1/sidecars/laion100m.rabitq1
```

RaBitQ1 sidecar 的冻结 SHA-256 为：

```text
46a7758e2008313f1d31893966db2afe10c46d12103780b4ee06b796c975fae0
```

输出根目录为 `/data_staging/wbh/hashprune_policy_sweep/laion100m/<RUN_ID>/`。runner 同时持有该扫描的 batch lock 与全局 SHEAR 实验锁，GPU 非空闲、sidecar 不匹配、无法无密码清空 page cache 或剩余空间不足 300 GiB 时均拒绝开始。

## 搜索预热与计量

每张已验证图只进行一次图级预热：

1. 清空 page cache 一次；
2. 顺序读入 base、graph、RaBitQ1 sidecar、query 和 GT，并记录每个文件的字节数与耗时；
3. 对完整 10,000-query 集执行一次不计时的 `L=400` 搜索；
4. 第一轮按 `L=25,50,100,200,400` 计量；
5. 第二轮按 `L=400,200,100,50,25` 计量。

不再为每个 L 单独预热。每个 L 的两次结果文件 SHA-256 和 Recall@10 必须一致；QPS 保留两个原始值并取算术平均。反向的第二轮用于平衡热状态或后台漂移。

## 执行

先运行契约测试与 release build：

```bash
python3 -m unittest script/test_laion100m_hashprune_policy_sweep.py -v
bash script/run_laion100m_hashprune_policy_sweep.sh prepare
```

执行不接触正式数据的 15 事务 smoke：

```bash
RUN_ID=contract-smoke \
OUTPUT_BASE=/data_staging/wbh/hashprune_policy_sweep/smoke \
bash script/run_laion100m_hashprune_policy_sweep.sh smoke
```

正式运行前检查：

```bash
bash script/run_laion100m_hashprune_policy_sweep.sh preflight
```

确认 GPU 和存储未被其他实验占用后，启动已批准的顺序事务：

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)" \
bash script/run_laion100m_hashprune_policy_sweep.sh formal
```

禁止对失败的 100M 事务自动重试或跳过。失败时 batch 和当前事务的 `STATUS` 变为 `failed`，完整 index/OOM/SQ payload、命令、日志与监控证据均保留，由人工检查后决定是否创建新的 RUN_ID。

已完成批次可以显式重算报告：

```bash
BATCH_ROOT=/data_staging/wbh/hashprune_policy_sweep/laion100m/<RUN_ID> \
bash script/run_laion100m_hashprune_policy_sweep.sh summarize
```

## 事务接受与清理

每个事务依次完成：构图、严格图格式校验、图质量测量、profile 参数核对、一次预热和十次计量搜索、成对结果一致性、证据 fsync。只有 `STATUS=accepted` 后，runner 才移除该事务自身的 `index/`、`oom/` 和 `sq_sidecar/`。命令、profile、driver timing、`time -v`、GPU 监控、图 SHA/字节数、图质量、搜索结果与指标均保留。

profile 必须记录并匹配 `requested_hash_quota`、`effective_hash_quota`、`requested_priority_alpha` 和 `effective_priority_alpha`；Q2 不允许出现第三层输出，所有命名策略的派生 R-half 容量均为 32，且不启用距离 veto。

## 报告与判定

最终报告位于 `<RUN_ID>/report/`：

```text
all_configurations.tsv   15 × 5 × 2 = 150 个原始搜索测量
recall_qps_pareto.tsv    每个 L 的 Recall--QPS Pareto 点
matched_budget.tsv       图预算匹配的逐对比较
per_method_winners.tsv   每种方法、每个 L 的最优参数
summary.json
summary.md
```

图预算匹配要求平均度数差不超过 0.25 且图字节数相对差不超过 1%。推荐要求两次 QPS 的相对 spread 不超过 5%，并满足以下之一：Recall 优势至少 0.001 且 QPS 不低；或 Recall 差小于 0.001 时 QPS 优势至少 2%。若没有唯一候选通过这些门槛，报告必须写为 `unresolved`，不能根据单个最高 Recall 或单次最高 QPS 宣称方法胜出。
