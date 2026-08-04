# LAION35M HashPrune cap=6、fanout 4/2/1 消融 Runbook

> 为兼容已配置的调用入口，本次直接复用名称中仍含 `cap3_fanout1041` 的
> runner、汇总器和测试文件；冻结矩阵、manifest schema 与输出目录才是本次
> cap=6、fanout 4/2/1 实验的权威配置。

## 1. 目的与结论边界

本实验在 LAION35M 上固定构图参数和新版统一搜索协议，比较 Plain、
HashSlot-1、DistanceFirst-Q3、SoftTier-Q3 与 HardTier-Q3。四种补边方法均
使用 `cap=6`；SoftTier 使用 `priority alpha=1.02`。

构图采用 fanout 4/2/1，将名义重叠子树机会限制为
\(4\times2\times1=8\)；作为参照，8/3/1 和已中止的 10/4/1 分别为 24 与
40。去重、分区大小、primary 选边和图饱和都会改变实际候选数，因此
不得把 8/24/40 当作实测候选数增幅。

该设置用于降低 primary 候选供给，观察补边方法相对 Plain 能恢复多少
Recall@10--QPS，以及四种选边策略之间的差异。Plain 不作为门控：无论其平均
出度或 Recall 高低，runner 都继续执行其余四种方法。只有命令失败、图结构
损坏、参数 passport 不符、证据缺失或用户中止才停止批次。

本实验重点观察低搜索预算 \(L=10,16,25,32,50\) 的 Recall@10--QPS 权衡，
同时报告大 \(L\) 的 \(L=100,200\) 结果。每个搜索点只测一次，QPS 必须标为
单次运行估计，不得声称统计显著性。

## 2. 冻结输入

```text
base=/data_staging/wbh/laion-512-35M/base.fbin
query=/data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin
gt=/data_staging/wbh/laion-512-35M/gt100.bin
rows=35000000
dim=512
sidecar=/data_staging/wbh/unified_graph_rabitq1/sidecars/laion35m.rabitq1
sidecar_manifest=/data_staging/wbh/unified_graph_rabitq1/sidecars/laion35m.rabitq1.json
output=/data_staging/wbh/hashprune_softtier_cap6/laion35m_f421/<RUN_ID>
```

LAION35M base 是规范 LAION100M base 的前 3500 万行；ground truth 必须针对
该子集生成。RaBitQ1 sidecar 必须从同一 base 构建并通过 `validate-only`。

## 3. 冻结方法顺序

| 顺序 | 标签 | 策略 | 行 cap | priority alpha |
|---:|---|---|---:|---:|
| 1 | `plain` | `off` | N/A | N/A |
| 2 | `c6_hashslot_q1` | `rhalf-hash-slot-1` | 6 | N/A |
| 3 | `c6_distance_first_q3` | `rhalf-distance-first-q3` | 6 | N/A |
| 4 | `c6_soft_tier_q3_a1p02` | `rhalf-soft-tier-q3` | 6 | 1.02 |
| 5 | `c6_hard_tier_q3` | `rhalf-hard-tier-q3` | 6 | N/A |

runner 必须按此顺序执行；失败后立即停止，不自动重试，也不跳过失败方法。
不得传入 hash quota override。legacy backfill alpha 固定为 1.0。

## 4. 冻结构图与搜索口径

构图：

```text
R=64
Lbuild=100
threads=8
m_hash_bits=12
seed=42
fanout=4/2/1
root_risk_candidate_rs=1,2,4
leaf_size_min/max=256/1280
leaf_knn=2
psamp_fraction=0.05
GPU workspace=16 GiB
D0 capacity ratio=0.08/0.08/0.08
partitioner=gpu-subtree
OOM I/O=buffered
```

搜索：

```text
backend=rabitq1-adaptive-exact
storage=mmap
threads=8
Recall@10
L=10,16,25,32,50,100,200
```

每张图只在全部搜索之前执行一次 \(L=200\) 预热，然后按升序执行七个正式
\(L\)。不得对每个 \(L\) 单独预热，也不得增加反向或重复轮次。

## 5. 静态验证与准备

在仓库根目录执行：

```bash
python3 script/test_laion35m_hashprune_cap3_fanout1041.py
bash -n script/run_laion35m_hashprune_cap3_fanout1041.sh
python3 -m py_compile \
  script/summarize_laion35m_hashprune_cap3_fanout1041.py \
  script/test_laion35m_hashprune_cap3_fanout1041.py
```

构建正式二进制：

```bash
bash script/run_laion35m_hashprune_cap3_fanout1041.sh prepare
```

## 6. 构建或复用 LAION35M RaBitQ1 sidecar

```bash
bash script/run_laion35m_hashprune_cap3_fanout1041.sh build-sidecar
```

若 sidecar 与 manifest 已存在，命令只验证并复用。出现残缺或不匹配文件时，
runner 会拒绝覆盖。只有人工检查后才能显式设置：

```bash
REBUILD_SIDECAR=1 \
  bash script/run_laion35m_hashprune_cap3_fanout1041.sh build-sidecar
```

构建记录保存在：

```text
/data_staging/wbh/hashprune_softtier_cap6/laion35m_f421/sidecar_build/
```

其中应包含命令、time、stdout/stderr、validate-only 输出、SHA-256 与状态。

## 7. Smoke

使用独立、不可复用的 smoke ID：

```bash
RUN_ID=smoke-YYYYMMDD-HHMMSS \
  bash script/run_laion35m_hashprune_cap3_fanout1041.sh smoke
```

验收：

- 五个 synthetic transaction 顺序正确；
- cap 参数只出现在四种补边方法且都等于 6；
- SoftTier 独有 `--hashprune-soft-tier-priority-alpha 1.02`；
- 共五次 warmup 和 35 次 measured call；
- accepted transaction 只清理自身的 `index/`、`oom/`、`sq_sidecar/`；
- profile、命令、搜索表和确定性汇总报告仍然存在。

smoke 与 formal 必须使用不同的 `RUN_ID`，因为 batch root 一经创建便不可
覆盖。

## 8. Formal preflight

先确定一个尚未创建的 formal ID：

```bash
RUN_ID=laion35m-c6-f421-YYYYMMDD-HHMMSS \
  bash script/run_laion35m_hashprune_cap3_fanout1041.sh preflight
```

preflight 检查数据头与长度、GT、sidecar/manifest、release binaries、CUDA
构建、GPU 空闲、两个实验锁、免密 page-cache drop 和 staging 剩余空间。它
不会创建 formal batch root。

此前的 LAION35M 五策略结果使用旧搜索协议、隐式 \(R/2\) cap 和 SoftTier
alpha 1.1，且索引已清理，因此不能替代本实验中的任何一行。

## 9. Formal 命令确认门

preflight 通过后，必须展示唯一的 formal 命令并再次确认：

```bash
RUN_ID=laion35m-c6-f421-YYYYMMDD-HHMMSS \
  bash script/run_laion35m_hashprune_cap3_fanout1041.sh formal
```

未获得该精确命令的确认前不得启动长实验。启动后不得修改参数、重排方法或
自动重试。

## 10. 监控与故障处理

formal 运行期间约 30 秒检查一次：

- 当前 transaction 与构图阶段；
- 进程是否仍存活；
- GPU utilization、HBM 峰值和异常回落；
- `time.log`、`profile.json`、GPU monitor 与输出文件是否持续更新；
- stderr 是否出现 OOM、CUDA、I/O 或结构验证错误。

除用户设定的硬超时外，异常只报告，不自动终止。若命令失败或被中断，batch
和当前 transaction 标为 `failed`，保留当前 payload、日志和命令，且不启动
后续方法。

## 11. 报告再生成

```bash
BATCH_ROOT=/data_staging/wbh/hashprune_softtier_cap6/laion35m_f421/<FORMAL_RUN_ID> \
  bash script/run_laion35m_hashprune_cap3_fanout1041.sh summarize
```

核心输出：

```text
report/all_configurations.tsv
report/per_graph_curves.tsv
report/candidate_pressure.tsv
report/plain_uplift.tsv
report/same_cap_comparisons.tsv
report/resource_matched_comparisons.tsv
report/summary.json
report/summary.md
```

## 12. 结果判定

先用 `plain_uplift.tsv` 比较四种补边策略在每个相同 \(L\) 下相对 Plain 的
Recall、QPS 和 Pareto 关系，再比较四种补边策略彼此的差异。四种策略的
requested/effective cap 应均为 6。

`shortlist_replacements` 或 `shortlist_capacity_rejections` 非零只说明 cap
产生了真实选择压力；两者均为零也是应报告的实验结果，不作为停止或验收门槛。

SoftTier 与其他补边策略的资源匹配要求：

- 平均出度差不超过 0.25；
- 图字节差不超过 1%。

Plain 是无补边参考，不受该门槛约束。只有固定 SoftTier 图在低 \(L\) 范围
对三个资源匹配替代策略均满足预先规定的 Pareto 判据，才能支持 SoftTier
优越性；否则应报告混合证据、等价或回退。

## 13. 清理边界

一个 transaction 只有在图结构、profile passport、搜索结果、hash、资源
指标和汇总输入全部持久化后才能标为 accepted。随后只允许删除该 transaction
目录下的：

```text
index/
oom/
sq_sidecar/
```

共享 LAION35M RaBitQ1 sidecar、profile、命令、日志、search outputs、
manifest 和报告必须保留。
