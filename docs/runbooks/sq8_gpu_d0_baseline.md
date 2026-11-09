# 普通 SQ8 构图基线与计时协议

协议版本：`sq8-laion10m-gpu-d0-20261107`。适用于普通 SQ8 与 SQ4/SQ4+ 的构图速度比较，以及 SQ8 等价优化的 A/B 验证。默认跳过检索；改变分区/量化/选边语义或用户要求质量时，按 AGENTS.md 另做质量验收。

## 基线身份与边界

普通 SQ8 的标准入口是 **GPU 编码 + GPU D0 分配**：`--gpu-subtree-d0-partition-mode gpu-kmeans-coarse --gpu-subtree-sq-staging-mode gpu-d0 --gpu-subtree-sq-bits 8`。CPU 预生成全局 sidecar、CPU FP32 Tagore D0 是独立实验模式，不能静默替代普通 SQ8 性能基线。

比较同一优化时，A 用上次验证版本，B 只加入该项改动并固定 D0；D0 本身的优化固定输入与种子。比较 SQ8 与现有 SQ4+ 正常入口时，两者 D0 不同，必须报告各自分区/任务哈希，称为系统对比，不能把总差值归于某个内核。不同轮次、冷/热缓存、CPU/GPU D0 的历史记录不能混成严格 A/B。

## LAION10M 固定配置

| 项目 | 配置 |
|---|---|
| 底库 | 10,000,000 × 512 FP32，20,480,000,008 字节（含头） |
| SHA256（完整 fbin） | `89632b353305bf33fb3733ef4b7f2d2c15a2485e49fe86c3b20d851ffc0ed0c6` |
| 机器 | 已验证机器为 RTX 4090 / CUDA 12.8；换机重新建立基线 |
| 距离、种子 | L2、42 |
| D0 | 64 个中心、top-6、训练比例 0.002，即 20,000 样本 |
| D0 GPU 上传分片 | 262,144 行 |
| 线程 | 请求线程及 Rayon 池 8，OpenBLAS 1；HashPrune 沿用相同 adaptive 策略（initial 8 / max 64），并记录实际线程数 |
| 内存预算 | 主机显式 256 GiB，GPU 工作区 16 GiB |
| I/O | `--oom-enable --oom-io-mode buffered`，输出使用相同存储设备 |
| 构图 | R=64、Lbuild=100、m_hash_bits=12、fanout=6/3、psamp=0.05 |
| 叶子 | min/max=256/1280、K=2、tile_rows=128、ticket_batch=8 |
| 执行器 | `fused-persistent`，`SHEAR_PIPNN_PERSISTENT_ASYNC_D2H=0` |
| 普通 SQ8 开关 | `SHEAR_PIPNN_SQ8_WMMA_OPT_MASK=0`，关闭 SQ4+、SQ8+、旋转残差及 raw-D1 诊断 |
| 观测 | `SHEAR_PIPNN_PROFILE=1`，`SHEAR_PIPNN_PRUNE_BACKEND=hashprune` |

SQ4+ 对照保留已经验证的一点一码、原生 INT4、tile 交叉投影复用和 8×32 分配，mask=1536，`SHEAR_PIPNN_SQ4_PRIMARY_TRIANGULAR=0`。其 D0 为 FP32 Tagore。修改这些选项须建立有明确名称的新变体，不能覆盖基线记录。

## 每轮执行顺序

1. 检查没有竞争构图进程，记录机器、源码差异、新文件、编译参数及二进制 SHA256。分别预热各方法的 GPU 路径；共享相同的 CUDA 缓存策略。
2. 校验底库形状和字节数。只对底库文件执行 `POSIX_FADV_DONTNEED`，随后完整顺序读取并校验 SHA256；用 `mincore` 记录驱逐前后及预热后的页面数，计时前要求全部 5,000,001 页驻留。预热不计入构图。
3. 每轮从原始向量重新生成 D0、编码及图。使用 `/usr/bin/time -f '%e %M %I %O %F %R'` 包住正常构图进程；包含进程内 CUDA/NVRTC 准备、编码、分区、剪枝和最终图输出。采样 `/proc/<pid>/io`、RSS、VmSwap，保留日志和 profile。
4. 两版用 A1/B1/B2/A2；三版用 A1/B1/C1/C2/B2/A2。报告逐轮值、中位数和范围。冷缓存、Direct I/O、编码复用等另开实验，不得与这一热缓存协议混报。
5. 在计时外全量审查图节点数、度数上限、邻居边界、自环及重复边。纯构图提速不追加 Recall/QPS。

如果更大底库无法满足全驻留条件，明确建立单独的缓存/I/O 协议；不能把它标作此 10M 热缓存协议。100M 仍需该规模实测，不能外推 10M 收益。

## 必须核对的运行证据

- `gpu_subtree.sq_gpu_staging`：`enabled=true`、`mode=gpu-d0`、`bits=8`、`d0_assign_kernel_mode=cuda-wmma-batch`，量化/分配计时非零，分配行数为 10M。
- 此固定预算与 buffered 配置必须复用本轮刚生成的内存编码：`sidecar_source=gpu-d0-resident`，日志出现 `resident sidecar reused from just-created runtime cache`，不得重新 preload。缓存复用仅限本次进程，不能复用上轮预生成 sidecar。
- 预算不足、部分驻留、自动预算或 Direct I/O 可合法走磁盘路径，但必须另列；不能用这些回退运行冒充本基线。主工作区与实验 worktree 的源码可能不同，不能仅凭目录或分支名认定修复已经生效。
- 10M 标准 D0 分区/任务哈希为 `17602056864155019802 / 5716008263347199385`；GPU 大节点 WMMA 计数非零，CPU 回退、大节点 DP4A 和队列溢出为零。
- 分开报告读取、GPU 量化、GPU D0 分配、sidecar 重载、D1+、GPU 内核、准备、上传、等待、输出、墙钟、RSS 和实际 I/O。SQ8 的 sketch 融入 D0，SQ4+ 的 sketch 单列。嵌套、流水重叠及多线程累计值不可直接相加。

## 已验证材料与只读核验

本机实验目录为 `/data/profwang/rswork/SHEAR-sq4/runs/sq8_gpu_d0_restore_20261107`。其中 **B1/B2 是恢复后的普通 SQ8 基线**；A1/A2 是原落盘诊断对照，C1/C2 是 SQ4+。不要把 A 当作后续默认 SQ8 基线。

修复版 `bin/after` 的 SHA256 为 `d46d7e33a57642b635d646fe5385582eb7ce2c3abd8656b73c8e9f9b2735e5ff`。新版本必须保存新的源码/二进制身份、配置与验证证据，不能仅沿用旧版名称。完整命令见 B1/B2 的 `cmd.json`、`env.json`；运行流程见该目录 `run.py`。重放例子（每个轮次名必须新建）：

```bash
cd /data/profwang/rswork/SHEAR-sq4/runs/sq8_gpu_d0_restore_20261107
source env.sh
python3 run.py B-recheck1 C-recheck1 C-recheck2 B-recheck2
```

后续标准 SQ8 运行完成后，从仓库根目录检查保存的证据：

```bash
python3 script/check_sq8_laion10m_baseline.py \
  /data/profwang/rswork/SHEAR-sq4/runs/sq8_gpu_d0_restore_20261107/B1
```

核验只读现有文件，不运行构图、不修改缓存。缺材料、CPU D0、冷缓存、错误 I/O、未复用编码、输入/参数/二进制或 D0 哈希漂移都会返回非零状态。它不替代对实验顺序、竞争任务和因果归因的审查。核验新验收的二进制时显式传 `--expected-binary-sha256`，同时更新实验协议和来源；不能为通过检查而覆写历史证据。

参考中位数为 SQ8 B=66.605 s、SQ4+ C=47.700 s（两轮）。它们是该机器与版本的实测值，不是后续测试的通过阈值。详细记录见实验目录的 `measured.md`、`summary.json`、`verified.json`、`tests.log`。

## 上三角的适用边界

分区分配矩阵为 `N×L`（point×leader），叶子候选矩阵为同一点集的 `N×N`。距离函数对称不等于 `N×L` 矩形任务本身对称：`d(point, leader)` 的反向通常不在任务集合中，不能按行列序号丢弃“下三角”。分区优先评估 8×32 矩形分配和数据复用。

若 L 个 leader 是点集内的不同样本，只有 `leader×leader` 子块可复用对称点积，最多少算 `L(L-1)/2` 个非对角点对，占原 `NL` 工作量的 `(L-1)/(2N)`。例如 N=4096、L=64，仅约 0.77%；映射、同步和双向结果更新可能抵消收益。若 leader 是不属于点集的训练质心，则没有这个共享子块。

叶子中同一编码空间的点积可用上三角复用，但两个端点各自的评分、并列处理和 top-k 更新仍需完成。上三角适用性不由 CTA 数决定；跨 CTA 时还需衡量临时矩阵、同步及合并开销。8×32 是 tile 形状，在数学上也能覆盖叶子矩阵，只是当前实测用途和收益在分区大节点，不能据此宣称叶子上绝对不可用。
