# LAION35M ACE non-Disk 与 Disk Motivation 实验

## 目标

在同一份 LAION35M、同一分区数和同一局部 CAGRA 参数下，对比 ACE 的
原生内存路径与强制磁盘路径，并将局部 GNND 构建拆分为 GPU kernel、CPU
图更新、CPU--GPU 协调及其重叠时间。该实验不设置 cgroup、不执行搜索。

## 固定设置

```text
base=/data_staging/wbh/laion-512-35M/base.fbin
rows=35000000
dim=512
metric=squared-L2
graph_degree=64
intermediate_graph_degree=128
ef_construction=176
npartitions=40
local_builder=cagra
cagra_local_solver=nn_descent
execution_mode=pipelined
OMP_NUM_THREADS=8
BLAS_threads=1
```

一次 28-partition 预运行表明，forced Disk 会把总 GPU 需求估为
23.40 GiB，并自动将分区数提高到 35。35 是刚好满足 80\% HBM 门限的
最低值，缺少运行时余量；因此正式比较共同固定为 40 partitions，估算约
16.4 GiB。第一组设置 `use_disk=false`，必须由日志确认进入 in-memory
mode；第二组设置 `use_disk=true`，必须确认进入 forced disk mode。两组的
core/augmented partition 标签摘要和逐分区行数必须一致，否则整组比较无效。

为避免两次独立 K-Means 分区造成任务形状差异，non-Disk 组以
`assignment_create` 保存逐点 partition 标签，Disk 组以
`assignment_replay` 重放同一标签。两组均按原始 ID 顺序生成确定性的
partition mapping。报告中的归一化 E2E 会从 non-Disk 组剔除 assignment
写出时间、从 Disk 组剔除 assignment 读入时间，并向 Disk 组补入 non-Disk
实测的 partition 计算时间；原始 E2E 同时保留。

## 执行

```bash
cd /path/to/SHEAR

bash script/run_laion35m_ace_disk_motivation.sh preflight

RUN_ID=laion35m-ace-disk-motivation-$(date +%Y%m%d-%H%M%S) \
  bash script/run_laion35m_ace_disk_motivation.sh formal
```

`preflight` 会验证：

- swap 已关闭且 GPU 上没有其他计算进程；
- 已有 LAION35M 的 header 和长度正确；
- 其 71,680,000,000 字节 payload 与 canonical LAION100M 的前 35M 行逐字节一致；
- Python 扩展解析到本次编译的 cuVS 动态库；
- HBM 和 `/data_staging/wbh` 空间满足正式运行要求。

正式运行前也可执行完整 runner 冒烟：

```bash
RUN_ID=laion35m-ace-disk-smoke-$(date +%Y%m%d-%H%M%S) \
  bash script/run_laion35m_ace_disk_motivation.sh smoke
```

## 测量边界

每组先顺序预热原始 fbin；预热时间单独写入 `prewarm_time.log`，不计入
ACE build E2E。`driver_profile.json` 中的 `ace_build_wall_ms` 从调用
`cagra.build` 前开始，到 ACE index 返回后结束。Nsight Systems 使用
`cuda,nvtx` trace 且关闭 CPU sampling；两组均承担相同 profiling 开销。

局部构建同时报告两种视图：

- active time：GPU kernel、GPU memcpy、CPU 图更新和协调区间分别求并集，
  允许彼此重叠；
- additive critical path：按 GPU compute、GPU transfer、CPU graph、
  exposed coordination、other 的优先级切成互斥区间，其和必须闭合到
  local-build wall。

不能把 active time 直接相加。

## 输出

每次正式运行写入：

```text
/data_staging/wbh/laion35m_ace_disk_motivation/<run-id>/
  ace_non_disk/
  ace_disk/
  report/summary.json
  report/partition_timeline.tsv
  report/report.md
```

每组保留命令、stderr、`/usr/bin/time -v`、GPU/CPU/内存/I/O 监控、
Nsight report、SQLite 和时间线 JSON。报告写入并同步后，runner 只删除
`generated_artifacts.txt` 中登记且位于本次 run root 内的图、重排数据和
mapping；不会修改已有 LAION35M。

若需要对已有 run 重新汇总或再次执行安全清理：

```bash
RUN_ID=<run-id> bash script/run_laion35m_ace_disk_motivation.sh summarize
RUN_ID=<run-id> bash script/run_laion35m_ace_disk_motivation.sh cleanup
```

## 已接受的正式记录

```text
/data_staging/wbh/laion35m_ace_disk_motivation/
  laion35m-ace-disk-motivation-frozen-p40-20260727/
```

该记录的分区摘要为 `1fc137dcdb4358ab00f0f9a3809ccf16`，40 个
partition 的 core/augmented 行数逐项一致。non-Disk 与 Disk 的归一化
E2E 分别为 444.281 s 和 559.738 s；详细的重排、pipeline I/O、GPU
计算、CPU 图操作及协调时间见该目录下的 `report/report.md`。
