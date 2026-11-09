# SQ4+ 按规模固定参数与实际执行审计

SQ4+ 后续构建使用 `script/run_sq4_plus.py`。它按底库实际行数选择
`laion10m` 或 `laion100m`，在运行前保存完整命令、环境、二进制哈希及配置来源，
完成后检查实际 profile；不依赖调用终端里遗留的 `SHEAR_*` 实验变量。
被替换的环境变量也会写入 manifest。

10M 增量基准与100M论文配置是两个协议，不能互称。
直接运行通用 `build_shear_index` 时，裸 CLI 的 fanout=10、psamp=0.001、
叶子上限2048等通用默认值不构成论文配置；需要使用下列入口或等价的完整参数。

一点一码保留为当前构图速度优化的编码基线；显存预算、D0路由leader数量和CPU lazy split仍需分别选择，不能把编码方案的采用当作这些参数已最优。固定64中心的100M历史比较使用fanout6、采样0.002及60%/50%/15%阈值，见[逐分区与一点一码比较](../../runs/sq4_partition_centers_100m_20261108/compare-primary64.md)；它与本页100M论文公共参数独立。增加路由leader时保持主编码中心64及fanout不变，先检查分区尾部、实际字节准入和拆分工作量；改变分区的方案需另外验证检索质量。

一点一码的预取路径已默认逐子树 gather、逐子树交付。仍先完成当前父分区的全部 lazy split，随后准备一个子树就交给消费线程；split 算法、成员顺序、seed、单预取 worker 和逐子树 GPU 上传方式不变。直接调用构建二进制时，`SHEAR_PIPNN_SQ4_STREAM_GATHER=0` 可恢复原整批 gather/交付，`1` 显式开启；本页配置入口会清理继承的 `SHEAR_*` 变量。

该改动的 [LAION10M A/B 报告](../../runs/sq4_stream_gather_20261108/report.md) 使用历史固定64中心、GPU k-means及60%/50%/15%配置，另列同样固定64的8%触发组，每版三轮。触发组输入等待降低9.7%、完整构建中位数降低2.1%、峰值RSS降低约144 MiB；原配置没有split，未见明确速度收益。此项验收不替代下述预算规划配方，也未据此推算100M收益。

| 参数 | `laion10m` 增量配置 | `laion100m` 当前配置（已验收SQ4优化） |
|---|---:|---:|
| fanout D0/D1/更深层 | 6/3/1 | 6/3/1 |
| D0中心训练采样率 | 0.002 | 0.002 |
| D0 k-means 迭代 | 4 | 4 |
| 递归 leader 采样率 psamp | 0.05 | 0.05 |
| D0规划预算 target / small-child low | 0.08/0.08 | 0.08/0.08 |
| 子树上传准入阈值 high | 0.50 | 0.50 |
| D0 leader规划 | 共用容量规划器 | 共用容量规划器 |
| leader质量下限/上限 | 64/10000 | 64/10000 |
| R / Lbuild / hash bits | 64/100/12 | 64/100/12 |
| 叶子 min/max / K | 256/1280/2 | 256/1280/2 |
| seed / 构建线程 / BLAS线程 | 42/8/1 | 42/8/1 |
| GPU工作区 | 16 GiB | 16 GiB |
| 主机预算 | 显式256 GiB | 显式256 GiB |
| I/O / 执行器 | buffered / fused-persistent | buffered / fused-persistent |
| ticket batch / leaf tile | 8/128 | 8/128 |
| scheduler / 高维leaf执行 | auto→baseline / auto→staged | auto→baseline / auto→staged |
| HashPrune | SoftTier Q3，priority alpha=1.02，每点最多3条补边 | 相同 |

2026-11-09用户要求100M默认入口采用最新已验收优化，并统一fanout6/3/1、采样0.002、
主机256GiB、target/low=0.08、high=0.50。旧论文配方的fanout8/3/1、采样0.01和自动主机预算
属于历史重放配置；当前入口不再默认采用它们。其余公共参数参考 [论文协议](p1_p2_shear_four_dataset.md)。
10M依据 [已验收10M增量协议](sq8_gpu_d0_baseline.md)，再叠加用户已要求的预算规划、
8%规划预算及SQ4+优化。用户于2026-11-09明确将上传准入阈值与规划预算分开，
high改为50%，target/low保持8%；50%是本轮SQ4+配置，不是论文原8%上传阈值。
原始固定Tagore/64上限及8%/8%/8%记录仅用于历史重放，未改写原始逐轮日志。

## SQ4+ 相对论文普通SQ8仍有的方法和开关差异

预算/阈值扫描见 [SQ4+容量扫描](sq4_capacity_sweep.md)：fanout6/3/1、
采样0.002、13组target/high。100M默认已与该扫描的8%/50%基线对齐；入口支持显式
--fanout-top、--d0-sample-fraction、--planning-budget-ratio和
--admission-threshold-ratio，实际覆盖写入manifest并核对执行profile。

不能将“公共参数对齐”说成“算法完全相同”。以下是明确保留的差异：

| 项目 | 论文普通SQ8 | 当前SQ4+ |
|---|---|---|
| 编码/距离 | SQ8 INT8 | FHT主中心残差packed INT4，一点一码，64个编码中心 |
| D0中心训练 | GPU SQ8分配、CPU更新中心 | CPU FP32训练中心；不是论文GPU训练器 |
| D0全量分配 | SQ8 GPU WMMA batch | 原空间FP32 GPU-exact，保持CPU参考浮点表达式 |
| D0原向量批次 | 262144行分片 | 4096行，与CPU融合sketch共用 |
| D1/叶子 | INT8 WMMA | 原生INT4 WMMA，保留原融合叶子 |
| CPU超标准入距离 | SQ8 | SQ4主中心残差，仍可能重聚类 |
| 优化mask | 普通SQ8=0 | 525824：投影复用512、8×32分配1024、PUBLISH复制524288 |
| persistent async D2H | 论文未覆盖，库默认开启 | 保留已验收SQ4实验的显式关闭 |

CPU sketch融合、GPU-exact、逐子树gather/交付和mask525824为默认启用配置；GPU sketch、按ID均分、
上三角、micro16、CUTLASS和其他实验mask未默认启用。
100M入口按论文显式开启通用leaf-WMMA开关；10M保持其原已验收设置。
这不把普通SQ8的D0/叶子实现替换进SQ4+路径。

## 使用

```bash
python3 script/run_sq4_plus.py \
  --binary /path/to/build_shear_index \
  --data-path /path/to/laion/base.fbin \
  --run-dir /path/to/new-run
```

`--dry-run` 只输出可审查的命令和配置，不构建。
`--profile laion100m --validation-prefix` 可显式用较小前缀核对100M配方的入口；
记录会标为前缀验证，不能写成完成100M实验。
`--memory-budget-gb 256` 与当前默认值相同，显式指定时仍写入覆盖记录。
`--routing-leaders 528` 可显式固定路由leader数量，编码质心仍为64；
它同时固定质量下限和数量上限，覆盖信息会写入manifest并审计。未指定时沿用预算公式。

两套命令均不包含固定 `tagore-kmeans-topk`、固定分区数或 `root-leader-cap=64`。
运行后必须证明预算模式实际为 `budget-kmeans-topk`、规划预算为8%、上传阈值为50%、leader数符合公式，
并检查真实fanout、训练样本数、GPU-exact/CPU复核、编码上传量、重载/回退、调度器、
HashPrune策略和已启用优化的日志。审计不通过则返回错误。

当前SQ4+路径按实际字节预算反推两种容量，保留完整容量模型中的固定开销。
比例分母为配置的GPU工作区16 GiB，8%规划目标为1,374,389,534 bytes（约1.28 GiB），
50%准入阈值为8,589,934,592 bytes（8 GiB）。profile中`threshold_policy=byte-admission`，
须分别核对`target_budget_bytes`和`high_budget_bytes`，不能只检查全量行数乘比例。
当前容量模型给出 `C_target=1,515,875`、`C_high=9,475,114`：
10M配方的 `K_capacity=40`、实际64；当前100M默认fanout=6给出 `K_capacity=396`；历史fanout=8才是528。
提高high不改变上述leader规划。
fanout从6变8也会使分区成员数从6N变8N，编码上传量增加1/3，不能归因于预算规划本身。

`C_target < subtree_rows <= C_high`的子树允许上传；超过high或未通过实际字节preflight时，
才由原lazy split算法按`ceil(rows/C_target)`选择拆分leader，递归修复直到子树满足high准入。
50%是当前模型下的保守阈值，不等于把整个进程的nvidia-smi占用限制为8 GiB；其他CUDA分配仍占显存。
标准入口保持lazy split开启，并逐子树审计上传所需字节不超过high，避免关闭准入后静默使用全工作区。

本次分离预算/阈值的证据见 [10M报告](../../runs/sq4_budget_threshold_20261108/report.md)。
10M主对照ABBA完整墙钟中位数39.130→36.285秒，RSS增加约236 MiB、GPU进程采样峰值增加640 MiB；
Recall最大下降0.0465个百分点，98%同召回插值QPS变化−0.21%且范围重叠。超阈值触发组也已验证，
保留50%准入配置并记录上述代价。用户明确要求暂不启动100M，不能把此前8%实验当成新阈值的100M验证。
历史配置审计见 [原8%报告](../../runs/sq4_config_audit_20261108/report.md)。

100M单轮固定配置测试记录见 [失败报告](../../runs/sq4_laion100m_paper8_20261108/report.md)。
CPU FP32中心训练按用户要求作为新增优化固定保留。此前行数阈值/字节准入边界不一致
已修复，并复现原失败行数、验证最大安全行数边界和完整100M路径；该历史修复没有放宽8%。
后续固定528路由的100M对照见 [100M报告](../../runs/sq4_random_leaders_100m_20261108/report.md)。
该对照采用原8%上传阈值及显式256GiB主机预算，用户要求跳过额外10M全量验证；
不能当成新50%阈值的100M结果，也不能静默替代本配方的自动主机预算。

默认配方更新说明（2026-11-09）：此次只改入口默认参数及文档；用户明确要求不运行测试，因为已有实验在跑。本次未重启现有任务。完整操作交接见主工作区 /data/profwang/rswork/SHEAR/HANDOFF_SQ4_EXPERIMENTS.md。
