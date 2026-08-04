# Wiki10M HashPrune SoftTier row-cap 优化实验

本 runbook 用于完成冻结的 18 图实验，判断 SoftTier 在图资源匹配时能否改善
低 \(L\) 的 Recall@10--QPS 权衡。第一阶段只改变每行最多保留的 backfill
数量，不修改 SoftTier 分数、搜索后端或精确重排方式。

## 固定输入与输出

```text
base=/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
query=/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
gt=/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
sidecar=/data_staging/wbh/unified_graph_rabitq1/sidecars/wiki10m.rabitq1
output=/data_staging/wbh/hashprune_softtier_row_cap/wiki10m
```

所有构图固定 seed 42、\(R=64\)、Lbuild=100、12-bit HashPrune、8 个线程、
16 GiB GPU workspace 和 fanout 8/3/1。现有 Wiki10M measurement runbook
使用 6/3/1，因此其结果只能作为动机证据，不能作为本实验的匹配基线。

搜索固定使用 `rabitq1-adaptive-exact` 导航和原始 f32 向量精确 rerank。

## 冻结的 18 项矩阵

矩阵按 cap 交错执行；每个 cap 内的顺序固定：

| 序号 | 配置 |
|---:|---|
| 1 | HashSlot-Q1-C2 |
| 2 | DistanceFirst-Q3-C2 |
| 3 | HardTier-Q3-C2 |
| 4 | SoftTier-Q3-A1.02-C2 |
| 5 | SoftTier-Q3-A1.05-C2 |
| 6 | SoftTier-Q3-A1.10-C2 |
| 7 | HashSlot-Q1-C3 |
| 8 | DistanceFirst-Q3-C3 |
| 9 | HardTier-Q3-C3 |
| 10 | SoftTier-Q3-A1.02-C3 |
| 11 | SoftTier-Q3-A1.05-C3 |
| 12 | SoftTier-Q3-A1.10-C3 |
| 13 | HashSlot-Q1-C4 |
| 14 | DistanceFirst-Q3-C4 |
| 15 | HardTier-Q3-C4 |
| 16 | SoftTier-Q3-A1.02-C4 |
| 17 | SoftTier-Q3-A1.05-C4 |
| 18 | SoftTier-Q3-A1.10-C4 |

HashSlot 使用默认 quota 1；其他三种方法使用默认 quota 3。不得显式传入
quota override。仅三项 A1.10 SoftTier 构图启用抽样反事实诊断；A1.10 使用
默认 alpha，A1.02 与 A1.05 才显式传入 alpha。

## 搜索顺序

每张图只执行以下搜索序列：

1. 全部 10,000 条 query 在 L=200 做一次 graph-level warmup；
2. 只做一轮测量，顺序为 `10,16,25,32,50,100,200`；
3. 不对每个 \(L\) 单独预热，也不执行反向或第二轮测量。

报告必须把 QPS 标为 single-run estimate，不得据此声称统计显著性。

## 执行

先进入仓库：

```bash
cd /path/to/SHEAR
```

### 1. prepare

```bash
bash script/run_wiki10m_softtier_row_cap_sweep.sh prepare
```

### 2. preflight

preflight 检查 base/query/GT、RaBitQ1 sidecar 及其 JSON manifest，调用
sidecar builder 的 validate-only，记录 SHA-256，并检查 CUDA binary、GPU
空闲、实验锁和剩余空间。

```bash
bash script/run_wiki10m_softtier_row_cap_sweep.sh preflight
```

任何检查失败时不得开始 formal。

### 3. smoke

```bash
SMOKE_ROOT=$(mktemp -d)
OUTPUT_BASE="$SMOKE_ROOT" RUN_ID=smoke \
  bash script/run_wiki10m_softtier_row_cap_sweep.sh smoke
```

要求得到 18 个 accepted transaction、18 次 warmup 和 126 次 measured
调用，共 144 次。失败注入必须保留当前事务、尝试次数 1，并且不创建后续
事务。

### 4. formal

先生成并固定一次 run ID：

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
printf '%s\n' "$RUN_ID"
```

把打印值原样代入；不要在后续命令中重新计算：

```bash
RUN_ID=<literal-run-id> \
  bash script/run_wiki10m_softtier_row_cap_sweep.sh formal
```

runner 顺序执行且不重试。一个事务只有在 graph/profile/diagnostics/search
均验证、证据 fsync 且状态原子更新后才成为 accepted。

### 5. summarize

formal 完成后可确定性重建报告：

```bash
BATCH_ROOT=/data_staging/wbh/hashprune_softtier_row_cap/wiki10m/<literal-run-id> \
  bash script/run_wiki10m_softtier_row_cap_sweep.sh summarize
```

## monitor、stop 与恢复边界

monitor 时只读检查固定 run root：

```bash
BATCH_ROOT=/data_staging/wbh/hashprune_softtier_row_cap/wiki10m/<literal-run-id>
cat "$BATCH_ROOT/STATUS"
find "$BATCH_ROOT/transactions" -mindepth 2 -maxdepth 2 -name STATUS \
  -print -exec cat {} \;
nvidia-smi
df -h /data_staging/wbh
```

如果用户要求 stop，向正在运行的 runner 前台任务发送一次 SIGINT，等待其
退出并确认 batch/current transaction 变为 failed；不要用模糊模式批量
终止进程。失败事务的 index、OOM、SQ、日志与搜索证据必须保留。这个
run root 不得续跑、跳过或复用；修复原因后使用新 RUN_ID 从头执行冻结
矩阵。

## cleanup 规则

accepted transaction 的 cleanup 仅删除该事务自己的：

```text
index/
oom/
sq_sidecar/
```

必须保留 `profile.json`、构图与 GPU 日志、graph SHA/资源记录、diagnostics、
search outputs、passport、STATUS 和报告。failed/running transaction 不得
cleanup。不要手工删除 `/data_staging/wbh` 下的共享数据或 RaBitQ1 sidecar。

## 资源匹配与成功门槛

一张固定 SoftTier 图只能与满足以下两项的对照图比较：

- 平均度绝对差不超过 0.25；
- index bytes 相对差不超过 1%。

低预算点固定为 \(L=\{10,16,25,32,50\}\)。同一张 SoftTier 图必须分别
对 HashSlot、DistanceFirst 和 HardTier 的资源匹配调优包络在至少 3/5 点
满足下列之一：

- Recall 下降不超过 0.001，同时 QPS 至少提高 2%；或
- QPS 下降不超过 2%，同时 Recall 至少提高 0.003。

其余低 \(L\) 点不得同时损失超过 0.001 Recall 和 2% QPS。不得插值、
外推，或把不同 SoftTier 图的胜点拼成一张部署图的结果。

## 第一阶段边界

完成 18 图和冻结报告后即结束第一阶段。若门槛失败，必须根据 capacity
binding、SoftTier--endpoint divergence、Recall/QPS 与搜索 work counters
定位原因；不进行事后扩展参数网格，也不选择事后阈值。distance-utility
gate 只能在另一个经过确认的设计与任务中实现。
