# Handoff：SQ4+ 一点一码实验（规划8%，准入50%）

更新日期：2026-11-09。目标为启用当前已验收优化，以 LAION100M 执行完整构建和 Recall@10–QPS 评测。本文件为操作说明，生成文件本身不会启动或修改正在运行的实验。

## 1. 仓库与版本

- GitHub：https://github.com/wbh1765091327/SHEAR
- 主仓库：/data/profwang/rswork/SHEAR
- **本实验实际工作树：/data/profwang/rswork/SHEAR-sq4**。
- 入口：script/run_sq4_plus.py；完整扫描入口：script/run_sq4_capacity_sweep.py。
- 当前入口及部分SQ4实现包含本地未提交文件。不能假设从GitHub重新clone就已具备本文功能；交接时同时交付该工作树的源码差异、未跟踪源码、实验脚本和冻结二进制，不能只给仓库URL或HEAD。
- 优先使用已验收冻结构建程序：runs/sq4_budget_threshold_20261108/bin/B。
- 构建程序 SHA256：ff51a77b0b88283fbd9badce1c6d79493e71fde24de19cb9784d3895bae4b652。
- 搜索程序：runs/sq4_budget_threshold_20261108/bin/search_shear_index。
- 图审查程序：runs/sq4_budget_threshold_20261108/bin/audit。

本文的“最新优化”指已验收并保留的端到端配置，不代表把所有实验开关叠加。上三角、micro16、CUTLASS、GPU sketch和按ID均分等实验路径不加入本次配置。

## 2. 共同参数与两种容量

| 项目 | 本次配置 |
|---|---|
| 编码 | SQ4+，一点一码，原生packed INT4，64个主编码中心 |
| fanout D0 / D1 / 更深层 | 6 / 3 / 1 |
| root risk candidates | 1,2,4,6，由入口跟随fanout派生 |
| D0训练采样 | 0.002，即100M上20万；同时用于路由及编码中心训练 |
| k-means迭代 / seed | 4 / 42 |
| CPU / BLAS线程 | 8 / 1 |
| 主机 / GPU工作区 | 显式256 GiB / 16 GiB |
| 规划target / low | 0.08 / 0.08 |
| 上传high | 0.50 |
| D0路由leader | 自动预算规划，100M约396；**不要固定为64** |
| 最大度 / Lbuild / hash bits | 64 / 100 / 12 |
| leaf min/max / leaf K | 256 / 1280 / 2 |
| 递归psamp | 0.05，与D0训练采样0.002不同 |
| 执行器 / I/O | fused-persistent / buffered |
| ticket batch / leaf tile rows | 8 / 128 |
| HashPrune | rhalf-soft-tier-q3，alpha=1.02，每点最多3条补边 |

比例分母是**配置的16 GiB工作区**，不是24 GiB物理显存：

- target=8%：1,374,389,534 bytes，约1.28 GiB，用于计算D0路由leader数量，也用于超限后CPU split的目标大小。
- high=50%：8,589,934,592 bytes，即8 GiB，用于子树上传前的字节准入检查。
- 当前512D模型对应 C_target=1,515,875，C_high=9,475,114；实际以profile中的字节模型为准。
- 路由数量按 ceil(N×fanout/C_target) 加质量下限64、数量上限10000确定；64个**编码中心**与路由leader数量相互独立。
- 超过8%目标但满足50%准入的子树可以直接上传；超过high或字节preflight失败才做CPU lazy split。
- 保持lazy split开启。50%不是GPU进程总显存上限，也不是绝不会OOM的保证，须记录实际峰值和剩余显存。

## 3. 必须开启的优化与对应开关

正常入口自动清理继承的 SHEAR_* 环境变量并设置下表。**在shell中另行export实验mask，不会覆盖这个入口。** 要改变实验配置，修改明确的入口选项并审计manifest；不要误以为旧shell变量已生效。

| 功能 | 入口设置 / 原始二进制开关 |
|---|---|
| SQ4+一点一码 | --gpu-subtree-sq4-plus、--gpu-subtree-sq4-primary-center |
| 原生INT4 | --gpu-subtree-sq-bits 4、--gpu-subtree-sq-native-int4 |
| GPU全量D0分配及编码 | SHEAR_PIPNN_SQ4_D0_BACKEND=gpu-exact |
| CPU FP32中心训练 | SHEAR_PIPNN_SQ4_D0_TRAIN=cpu |
| 原向量读取融合CPU sketch | SHEAR_PIPNN_SQ4_D0_FUSED_SKETCH=1，SHEAR_PIPNN_SQ4_D0_SKETCH_BACKEND=cpu |
| 已验收GPU优化组合 | SHEAR_PIPNN_SQ8_WMMA_OPT_MASK=525824 |
| 投影复用 | mask中的512 |
| 8×32 point×leader分配 | mask中的1024 |
| PUBLISH并行成员复制 | mask中的524288 |
| 逐子树gather及交付 | SHEAR_PIPNN_SQ4_STREAM_GATHER=1 |
| 异步结果scatter | SHEAR_PIPNN_ASYNC_RESULT_SCATTER=1 |
| 复现实验的D2H模式 | SHEAR_PIPNN_PERSISTENT_ASYNC_D2H=0 |
| 记录profile | SHEAR_PIPNN_PROFILE=1 |

GPU负责D0全量距离分配和残差编码，中心训练仍为CPU FP32；不要把它写成GPU k-means训练。本版先完成当前父分区全部lazy split，再逐个gather/交付子树；可与前一个GPU任务重叠。

## 4. 单组8%/50%：检查和启动

先确认没有其他竞争CPU/GPU/磁盘的实验。以下仅在准备新任务时手动执行，不要用来重启现有扫描。

```bash
cd /data/profwang/rswork/SHEAR-sq4
source runs/sq4_capacity_sweep_implementation_20261109/env.sh

# 路径必须新建，不复用已有结果目录。
SQ4_HANDOFF_RUN="$PWD/runs/sq4_handoff_8_50_$(date +%Y%m%d_%H%M%S)"

# 先输出有效命令及环境，不构建、不创建该结果目录。
python3 script/run_sq4_plus.py   --profile laion100m   --binary "$PWD/runs/sq4_budget_threshold_20261108/bin/B"   --data-path /data/tyj/data/laion-512-100M/base.fbin   --run-dir "$SQ4_HANDOFF_RUN"   --memory-budget-gb 256   --fanout-top 6   --d0-sample-fraction 0.002   --planning-budget-ratio 0.08   --admission-threshold-ratio 0.50   --dry-run

# 确认上述清单后启动构建；去掉dry-run，其余参数相同。
python3 script/run_sq4_plus.py   --profile laion100m   --binary "$PWD/runs/sq4_budget_threshold_20261108/bin/B"   --data-path /data/tyj/data/laion-512-100M/base.fbin   --run-dir "$SQ4_HANDOFF_RUN"   --memory-budget-gb 256   --fanout-top 6   --d0-sample-fraction 0.002   --planning-budget-ratio 0.08   --admission-threshold-ratio 0.50
```

当前100M入口默认已改为fanout6/3/1、采样0.2%、主机256 GiB、target/low=8%、high=50%，并启用本文列出的已验收优化。命令仍显式写出6、0.002、256，便于审查和重放。不要加入 --routing-leaders 64；也不要恢复high/target/low全8%的历史配置。

本机env.sh加载工具链、CUDA运行库及隔离绘图依赖。迁移机器时须替换这些绝对路径。新编译程序应另存二进制及SHA256，不覆盖冻结B；本地历史编译命令见 runs/sq4_budget_threshold_20261108/compile-check.sh，不能忽略其中Cargo配置和离线依赖路径。

## 5. 评测、计时和清理

单组构建入口**不自动跑检索，也不自动删图**。构建完成后，先用审查程序检查节点、度数、非法ID、重复边和自环：

```bash
runs/sq4_budget_threshold_20261108/bin/audit   "$SQ4_HANDOFF_RUN/index/_mem.index" 100000000   > "$SQ4_HANDOFF_RUN/graph-audit.txt"
```

核验底库、查询和真值完整SHA256；确认真值针对100M而非10M。预热完整底库、当前图、查询和真值，用实际逐页触读及smaps RSS核验初始驻留，记录运行中实际读盘。先做一次L64预热检索，再测四档：

```bash
for SQ4_HANDOFF_L in 64 128 256 512; do
  runs/sq4_budget_threshold_20261108/bin/search_shear_index     --index-path-prefix "$SQ4_HANDOFF_RUN/index"     --data-path /data/tyj/data/laion-512-100M/base.fbin     --query-file /data/tyj/data/laion-512-100M/query.fbin     --gt-file /data/tyj/data/laion-512-100M/gt100.bin     --recall-at 10 --search-list-size "$SQ4_HANDOFF_L"     --num-threads 8 --storage mmap --search-backend exact     --result-path "$SQ4_HANDOFF_RUN/search-$SQ4_HANDOFF_L.fbin"     --metrics-path "$SQ4_HANDOFF_RUN/search-$SQ4_HANDOFF_L.tsv"     > "$SQ4_HANDOFF_RUN/search-$SQ4_HANDOFF_L.log" 2>&1
done
```

四档TSV均须为10000查询，核对storage=mmap、search_backend=exact、recall_at=10及实际L。
95%/97%同召回QPS只在实测点覆盖时插值。严格计时及自动预热可复用 script/sq4_sweep_runtime.py 的 prewarm、run_measured、quality；单组入口本身的墙钟还包含Python启动和配置审计，不能直接冒充扫描器的纯构建进程GNU time口径。

每组至少保留：命令和环境、源码差异/未跟踪源码、二进制及SHA256、输入身份、build.log、profile.json、完整构建墙钟、峰值RSS、100ms GPU显存采样、图审查、图SHA256和四档原始检索指标。报告CPU split次数/时间、gather、接收等待、GPU事件时间、D0各阶段和峰值显存；嵌套/重叠时间不可相加。

**先完成结构审查和四档Recall–QPS并保存证据，再清理本轮图、分区临时文件及检索邻居数组。** 保留TSV、日志、profile、哈希和可重放二进制。只操作自己新建的结果/临时目录；不删除数据集、历史结果、冻结二进制或其他正在运行任务的目录。评测失败时保留现场，不执行清理。

## 6. 全参数扫描与当前任务的边界

script/run_sq4_capacity_sweep.py 默认只生成13组清单；它不是单独启动8%/50%的命令。
--validate-10m执行新入口10M门禁；--execute显式启动完整100M矩阵，随后复测及普通SQ8参考；
--validation-report指定匹配门禁，--skip-10m-gate-reason只能用于记录用户明确授权的跳过理由，不能伪造通过记录。
已完成轮次的--resume受源码版本、二进制、输入及运行库检查约束；中断未形成结果的现场不能覆盖。

现有100M扫描目录：/data/profwang/rswork/SHEAR-sq4/runs/sq4_capacity_sweep_100m_20261109_execute。
接手时先只读检查 status.json、controller-pid.json、controller.log、各组result.json和cleanup.json。
该任务发生过“单点分区正常跳过GPU，审计却要求全部成员上传”的审计修正，恢复入口为该目录下 recover_and_continue.py；
不要盲目从原始命令重启或改动正在运行的脚本。单点分区的明确skip日志应纳入成员守恒，不能把它当作数据丢失或CPU回退。
扫描器会在图审查和四档检索通过后自动清理每组图及临时数据，输出 report.md、summary.json、CSV和热力图。
当前性能结论以实际完成/复测记录为准，不把单次初筛或“暂未OOM”当作全局最优、安全边界或最终质量结论。

## 7. 输入指纹与验收重点

| 输入 | SHA256 |
|---|---|
| base | 4cb8c8d2586728ae81f3a01db19050547c88a2d7801f76133198ca2465d23dac |
| query | 671720739208ba56731c679d2acf68458b50ed391290f1f6ebce6d7ec122444c |
| gt100m | 7cddc8b966119e38bf16f6b23259bad1a65458da158cccb440fba47d20952930 |

profile中必须验证：budget-kmeans-topk、byte-admission、target_budget_bytes=1374389534、high_budget_bytes=8589934592、lazy split开启、实际路由数量约396、primary centers=64、int4-wmma、零意外CPU fallback/queue overflow、零sidecar重载、D0全量GPU-exact且CPU复核为0。

日志中应见投影复用、primary large INT4 point×leader 8x32、PUBLISH cooperative member copies及per_subtree_gather=true。上传逐条满足 admission_required_bytes ≤ admission_budget_bytes = 8589934592。

除正常单点skip外，D1成员上传为N×6，每成员272字节；正常单点skip须由明确日志逐条解释。gpu_wait_for_sidecar是提交侧接收等待，不是CUDA事件测得的GPU空闲时间。D0 host_read含拷贝，不等于磁盘等待；码与成员回传目前仅有合计计时，不伪拆两个时间。

相关源码说明：/data/profwang/rswork/SHEAR-sq4/docs/runbooks/sq4_plus_profiles.md；完整扫描说明：/data/profwang/rswork/SHEAR-sq4/docs/runbooks/sq4_capacity_sweep.md。

2026-11-09默认值更新：仅修改新启动时读取的入口配方及文档。按用户要求，本次未运行测试、构建或新实验；现有实验不重启。入口源码哈希已变化，后续恢复扫描须先核对已有版本记录，不绕过版本检查。
