# Wiki10M 单 shard GPU 路径剖析

## 目的与证据边界

本实验在同一个 Wiki10M ACE local shard 上比较 CAGRA、Tagore-CAGRA 和
Tagore-Vamana，回答两个问题：

1. dense 距离计算、irregular 图状态处理、mixed 融合 kernel 和 other GPU
   工作分别占用多少关键路径时间；
2. 各类 kernel 的 SM/Tensor、DRAM/L2、occupancy、long-scoreboard、barrier
   和 atomic 指标有什么差异。

单个 Wiki10M shard 只能作为机制证据，不能外推到所有维度和图分布。
固定参数结果是 fixed-work 比较；若讨论方法性能，必须同时展示 Recall@10，
并明确它不是自动调参后的 matched-quality 比较。

## 冻结输入

规范源文件为：

```text
/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
```

Wiki10M 文件为：

```text
/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
```

runner 会检查 Wiki10M 的 header 为 `10000000 x 1024`、文件长度为
`40960000008` 字节，并比较它与规范源前 10M 行的 payload。正式质量验证固定复用
`/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`（10000×1024），再针对
local shard 重新生成 top-10 ground truth；只有 8192×128 smoke 使用
shard-derived query。所有新产物写到：

```text
/data_staging/wbh/wiki10m_local_shard_gpu_profile
```

local shard 来自冻结的 32-way ACE partition plan。选择规则是：在
`[core][augmented]` 合并行数不超过 1,100,000 的 partition 中，选择行数最大的
一个；并列时选择最小 partition ID。输出顺序固定为 core 后接 augmented，
禁止截断。如果 32 路计划没有可用 partition，准备阶段失败；此时另行批准后
才能建立 40 路计划，不能由 runner 静默改变合同。

## 固定构建合同

| 项目 | 值 |
|---|---|
| metric | squared L2 |
| intermediate_degree | `intermediate_degree=128` |
| final_degree | `final_degree=64` |
| nominal iterations | `iterations=16` |
| device | GPU 0 |
| CPU threads | OMP 8，BLAS 1 |
| Tagore pruning | `TOPM=64`, `threshold=1.2` |
| seed | 20260713 |

CAGRA 使用 cuVS NN-Descent。Tagore-CAGRA 和 Tagore-Vamana 都先调用相同的
`GNN_descent`，再分别执行 CAGRA pruning 或 Vamana pruning。三种输出统一成
DiskANN `_mem.index`，使用同一个“距 shard 质心最近向量”入口点。

### 冻结 Tagore 模块

默认模块为：

```text
/data_staging/wbh/wiki10m_local_shard_gpu_profile/toolchain/tagore_k128_current/Tagore.cpython-312-x86_64-linux-gnu.so
```

其冻结 SHA-256 为：

```text
426070b8085cfbbe046d8e4395ced552ef00141e584aff3697377d5dfc7a8283
```

runner 在任何构建前检查该哈希。旧的
`tagore_cagra_wiki35m_measurement/tagore_build_k128` 模块不能用于本实验：
8192×128 smoke 两次分别产生 3,007 和 3,035 个含重复边的 CAGRA 行。用当前
Tagore 源码重建的冻结模块通过同一 shard 的逐行边界、自环和重复边校验。
模块由以下命令构建；源码或编译器变化后必须使用新目录和新哈希，不能覆盖后
继续与旧结果混用：

```bash
env CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
  cmake -S /path/to/Tagore \
    -B /data_staging/wbh/wiki10m_local_shard_gpu_profile/toolchain/tagore_k128_current \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -Dpybind11_DIR=/data_staging/wbh/tagore_wiki10m_measurement/venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11 \
    -DPYTHON_EXECUTABLE=/data_staging/wbh/cuvs_tagore_ace_venv/bin/python \
    -DBUILD_TESTING=OFF
cmake --build \
  /data_staging/wbh/wiki10m_local_shard_gpu_profile/toolchain/tagore_k128_current \
  --target Tagore --parallel 8
```

## 三层测量必须分离

- `timing`：一次 warm-up，随后三次不带 profiler 的轮转重复；这是规范 wall
  time。
- `Nsight Systems`：每种方法单独采集一次，用于关键路径和 kernel overlap；
  不能替代 timing。
- `Nsight Compute`：按 dense、irregular、mixed 类别选择早/中/晚 launch，
  但保留每个实际 kernel 名称各自的 counter signature；资源百分比彼此不互补，
  不能堆叠成 100%。

Nsight Systems 的排他优先级为 dense、irregular、mixed、other、transfer、
CPU graph、coordination、residual。排他项之和与 local-build wall 的误差不能
超过 `max(2 ms, 0.5%)`。未匹配 kernel 全部保留；other 中未分类的 GPU active
time 超过 `5%` 时，正式比较失败。

报告还在 GPU kernel active union 内按 dense、irregular、mixed、other 的同一
优先级生成 `gpu_path_exclusive_ms` 和 `gpu_path_exclusive_share`。这组份额闭合
到 100%，用于回答“GPU 活跃时间里计算/图状态路径各占多少”；local-build
critical-path share 则用于回答它们占端到端 wall 的多少。两种分母不能混用。

### CFS 指标不能拆成逻辑三阶段

Python `Tagore.Pruning()` 的 NVTX wall 同时包含 `cal_ep_gpu`、两个 CFS
kernel、D2H 和主机 `StoreVamana/StoreCAGRA`，所以它只记为
`prune_api_wall_ms`。Nsys 另行输出：

```text
prune_setup_gpu_ms
cfs_primary_gpu_ms
cfs_reverse_finalize_gpu_ms
cfs_gpu_ms
cfs_time_basis=summed_nsys_kernel_elapsed_time
cfs_share_of_gpu_kernel_time
cfs_share_of_local_build_wall
prune_transfer_ms
prune_host_residual_ms
cfs_logical_stage_split_supported=false
```

`cfs_primary_gpu_ms` 对应 `select_path` 或 `select_1hop_cagra`；
`cfs_reverse_finalize_gpu_ms` 对应 `filter_reverse` 或
`filter_reverse_1hop`。这只是两段融合实现时间，不能改写成纯 Collect、
Filter、Store 三列。

Vamana 的 `select_path`/`filter_reverse` 同时含高维距离计算与图更新，语义标签
为 `mixed_cfs`，并计入 mixed 关键路径。CAGRA 的
`select_1hop_cagra`/`filter_reverse_1hop` 标签为 `irregular_cfs`，并计入
irregular 关键路径。

### NCU 覆盖率与 memory-bound 边界

NCU 的 compute、DRAM/L2 和 stall 百分比不能相加成“GPU 内访存时间”。报告先
对每个实际 kernel 单独生成 NCU 签名，再映射回同名 Nsys kernel 的累计启动
耗时。class 级平均只用于描述，不能把一个 class 签名套给其中所有 kernel。
输出：

```text
gpu_kernel_elapsed_sum_ms
ncu_classified_nsys_time_ms
ncu_coverage_denominator_kind=summed_nsys_kernel_elapsed_time
ncu_coverage_denominator_ms
ncu_coverage_of_gpu_kernel_elapsed_time
ncu_coverage_of_gpu_active_time
memory_bound_nsys_time_ms
memory_bound_execution_share
memory_bound_share_status
```

兼容字段 `ncu_coverage_of_gpu_active_time` 与
`ncu_coverage_of_gpu_kernel_elapsed_time` 数值相同，但其真实分母由
`ncu_coverage_denominator_kind` 明确为 Nsys 所有 kernel 的累计启动耗时，不是
GPU active interval union。这样在并发 kernel 存在时，分子和分母仍保持同一
可加口径。

只有三种方法各自的 kernel-time coverage 都达到预注册的 `80%`，且覆盖 kernel
被逐一明确分类为 compute-bound、memory/latency-bound 或 mixed，才允许
`memory_bound_share_status=SUPPORTED` 并报告 memory/latency-bound 时间占比。
`insufficient_or_balanced` 不计入覆盖率；`mixed_compute_memory_latency` 计入已分类
覆盖率，但不计入“纯 memory/latency-bound”分子。
否则状态必须是 `INCOMPLETE`，论文只能报告 irregular/mixed graph-state
kernel share。

正式运行前固定以下逐 kernel 计数器签名阈值：

- `SM throughput >= 60%` 或 `Tensor-pipe activity >= 30%`：compute signal；
- `DRAM throughput >= 60%` 或 `L2 throughput >= 60%`：memory-throughput
  signal，且不因 SM 较高而取消；
- `long-scoreboard stalls >= 25%`：latency signal；
- compute 与 memory/latency 信号同时成立：`mixed_compute_memory_latency`；
- atomic 指令数只作诊断，不参与 bound 分类。

这些阈值只生成 counter signature，不能解释为可相加的“计算时间百分比”和
“访存时间百分比”。每个实际 kernel 独立判断，不能用捕获 class 的平均值代替。

## 执行

进入仓库：

```bash
cd /path/to/SHEAR
```

先准备或复用冻结 shard。此步骤可能创建完整 10M ACE plan，耗时和空间都较大，
但不计入 local-build timing：

```bash
bash script/run_wiki10m_local_shard_gpu_profile.sh prepare
```

先运行 8192×128 小规模 GPU smoke。它会运行三种 builder、Nsys、NCU 权限
探测（有权限时继续采样）、严格图校验和小型本地搜索，但不会产生 Wiki10M
正式结论：

```bash
RUN_ID=wiki10m-local-shard-smoke-$(date +%Y%m%d-%H%M%S) \
  bash script/run_wiki10m_local_shard_gpu_profile.sh smoke
```

确认 smoke 后，正式运行：

```bash
RUN_ID=wiki10m-local-shard-formal-$(date +%Y%m%d-%H%M%S) \
  bash script/run_wiki10m_local_shard_gpu_profile.sh formal
```

也可用同一个 `RUN_ID` 分阶段运行 `timing`、`nsys`、`ncu`、`quality`。阶段失败后
先检查该阶段日志；runner 不会静默重试或覆盖已有目录。

从已有产物重建报告：

```bash
RUN_ID=<existing-run-id> \
  bash script/run_wiki10m_local_shard_gpu_profile.sh summarize
```

删除清单中登记的 graph/result payload：

```bash
RUN_ID=<existing-run-id> \
  bash script/run_wiki10m_local_shard_gpu_profile.sh cleanup
```

cleanup 只处理 `generated_artifacts.txt` 中、且解析后仍位于当前 run root 下的
显式文件；不会删除 prepared shard、输入、日志、`.nsys-rep`、`.ncu-rep` 或
`report/`。被移除的生成文件可通过重跑对应阶段恢复。

## 输出与验收

每个正式 run 位于：

```text
/data_staging/wbh/wiki10m_local_shard_gpu_profile/runs/<run-id>/
```

主要证据包括：

- `provenance/sha256.txt`，其中固定 driver、两类 analyzer、summarizer、
  runner、plan、spec 和本 runbook 的源码哈希；
- `timing/<method>/repeat_{1,2,3}/profile.json` 和 `time.log`；
- `nsys/<method>/timeline.json`、trace 和未分类 kernel TSV；
- `ncu/<method>/<kernel-class>/metrics.json` 中的 per-kernel signature 及原始
  CSV；
- `quality/common/quality_contract.json` 中绑定的 shard/query/GT 哈希；
- `quality/<method>/graph_validation.json`、`graph_quality.tsv`、完整的
  `L×repeat` 搜索命令/exit/metrics 和 `search_summary.tsv`；
- `report/timing.tsv`、`critical_path.tsv`、`ncu_metrics.tsv`、`hbm.tsv`、
  `summary.json` 和 `report.md`。

正式报告只有在以下条件全部满足时才接受：shard hash 与参数一致；三次 timing
的 command、exit code、`time.log` 都完整且不带 profiler；三种图结构合法且
`graph_quality.tsv` 可交叉核对；query/GT 哈希与构图 shard 绑定；
`L={25,50,100,200,400}×repeat={1,2,3}` 完整；Nsys capture/export/profile/trace
完整，local-build 关键路径和 GPU-active 路径都闭合；未分类 GPU time 不超过
5%；profiled 与 unprofiled 时间分开。

summarizer 不信任这些派生 JSON 字段本身：它会重新分析每个 `trace.sqlite`，并
要求结果与 `timeline.json` 完全一致；也会从每组原始 NCU CSV 重新生成 class
及 per-kernel metrics/signature，要求与 `metrics.json` 完全一致；每个捕获
kernel 还必须满足 method-aware 语义 class，并匹配同次 Nsys aggregate。未分类
占比、图可达率及 analytical HBM 下界分别从底层毫秒数、整数节点数和固定构图
参数重算；图可达率只允许 TSV 六位小数格式带来的舍入误差。任一不一致都会拒绝
报告。

分阶段执行时不刷新初始 provenance。summarizer 会把 `sha256.txt` 与当前八个
协议/实现文件逐一复核；任一阶段之间发生源码或计划漂移，报告直接拒绝，必须用
新 run ID 重跑。

`gpu-monitor.csv` 每 100 ms 记录 device index 0 的整卡 `memory.used`。有效记录
必须至少包含一个同步的运行前基线和一个后续样本，且所有样本都属于 index 0。
报告同时给出绝对峰值和相对首个基线样本的 peak delta；它们用于方法间实测 HBM
比较。驱动中的 analytical bytes 只是共同的结构下界，不能当作三种实现的实测
内存差异。

若 NCU 因 `ERR_NVGPUCTRPERM` 无法访问性能计数器，runner 写入
`NCU_INCOMPLETE`，保留 timing 和 Nsys 结果并停止后续 NCU capture。此时报告
只能标记 INCOMPLETE，不能用 `nvidia-smi` utilization 代替缺失 counter。
其他 NCU 错误仍会使阶段失败。

## 论文表述

当 Tagore 路径具有可复现的 irregular/mixed GPU-active 份额、GPU-active
dense share 相对 CAGRA 降低，并且 NCU 显示 dense 与 graph-state kernel 的
资源特征差异时，允许表述：

> 图状态处理与稠密距离计算共享设备预算，并挤占稠密计算占空比，使 GPU 难以
> 持续保持单一密集计算模式。

“共享设备预算”不等于“同时竞争设备资源”。只有当 trace 显示 dense 与
graph-state kernel 实际 overlap，并且同一个归一化 dense kernel 在 overlap
期间相对 standalone replay 明确变慢时，才能使用“同时竞争设备资源”这一更强
结论。当前 runner 尚未实现与 method/kernel/trace/构建合同绑定的 standalone
与 overlap replay，因此 `contention_claim_supported` 固定为 false；手写
`dense_overlap_slowdown.json` 不能开启该结论。只有顺序执行或只有整体利用率
下降时，禁止使用该表述。
