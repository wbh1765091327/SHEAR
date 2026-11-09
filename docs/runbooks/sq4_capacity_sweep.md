# SQ4+ LAION100M 预算与上传阈值扫描

本任务固定 fanout **6/3/1**、候选集合 **1,2,4,6**、D0 采样 **0.002**、
k-means 4 轮、seed42、64 个主编码中心、8 个 CPU 线程/1 个 BLAS 线程、
256 GiB 主机预算和 16 GiB GPU 工作区。
沿用一点一码、CPU FP32 训练、融合 CPU sketch、GPU-exact D0、mask525824、
逐子树 gather/交付和原 split 算法。没有修改 GPU 内核。

2026-11-09起，100M默认入口也采用本页fanout6/3/1、采样0.002、主机256GiB及8%/50%基线；
其他target/high组合仍需显式覆盖。
0.2% 在 100M 上为 20 万样本，同时影响路由和编码中心训练。
它不是与 1% 质量等价的声明；历史不同配置只能作系统参考。

## 入口与任务清单

script/run_sq4_plus.py 新增：

~~~text
--fanout-top 6
--d0-sample-fraction 0.002
--planning-budget-ratio 0.08
--admission-threshold-ratio 0.50
~~~

fanout 覆盖同时派生候选集合；low 跟随 target。
约束为 0 < target ≤ high ≤ 1，本扫描额外只使用不超过 0.50 的 high。
覆盖写入 manifest，实际样本数、D0 leader、上传量及逐子树字节准入由 profile/log 审计。

默认命令**只生成清单，不启动构建、不读取完整 204.8 GB 底库、不初始化 GPU**：

~~~bash
python3 script/run_sq4_capacity_sweep.py \
  --run-dir runs/sq4_capacity_sweep_100m
~~~

| target | high |
|---|---|
| 4% | 8%、16%、32%、50% |
| 8% | 8%、16%、32%、50% |
| 16% | 16%、32%、50% |
| 32% | 32%、50% |

共13组。8%/50%先跑，其余按seed42固定打乱。预计路由K约792/396/198/99，
运行时使用原字节模型计算并校验，不固定K。比例分母为16 GiB工作区，不是物理显存。
high=50%对应8 GiB模型准入，进程还存在模型外的CUDA分配。

## 显式执行与10M门禁

Python需要matplotlib及其依赖才能执行/绘图；只生成清单和核心测试使用标准库。
主机需Linux /proc、GNU time、NVML、CUDA运行环境及既有数据和冻结二进制。
默认路径引用当前仓库历史验收产物；迁移机器时通过CLI提供二进制、CUDA缓存、
旧入口、SQ8参考目录和输入manifest。

输入manifest包含base、query以及gt100m或gt10m的路径与已知SHA256。
执行前检查实际文件形状、完整SHA256、真值ID范围；每轮检查文件身份不变。
默认100M/10M使用各自历史验收的输入身份，不混用真值。

~~~bash
# 仅执行10M门禁；不会继续自动启动100M。
python3 script/run_sq4_capacity_sweep.py \
  --run-dir runs/sq4_capacity_sweep_gate10m \
  --validate-10m

# 只有收到新的100M启动指令后执行。
python3 script/run_sq4_capacity_sweep.py \
  --run-dir runs/sq4_capacity_sweep_100m \
  --execute \
  --validation-report runs/sq4_capacity_sweep_gate10m/validation.json
~~~

10M门禁用同一冻结构建二进制，对旧入口A和显式覆盖入口B做8%/50%的A/B/B/A，
另做固定6路由的A/B：top6强制每个D0父分区包含全部10M行，覆盖真实超阈值split。
共6次10M完整构建及四档检索。该诊断固定K不进入100M矩阵。
门禁检查D0/主编码中心一致、正常GPU路径、图结构和清理完成；
并绑定构建、搜索二进制及扫描脚本哈希。修改脚本或二进制后旧门禁失效。

100M初筛每组一次，再选有效非基线中的两组B/C，按A/B/C/C/B/A复测，
共19次参数构建。优先选择不同实际任务，未split且D0成员/任务及主中心哈希一致时
才认证等效；相同子树行数不足以证明成员相同。等效组取较低high代表。
显存观测余量不足1 GiB不进入稳健候选。候选不足两组会报告不足，不伪造完整验收。

完整墙钟差小于两组最大轮间范围或基线2%时，每个接近的候选额外执行一次ABBA，
最多再加8次参数构建。该规则是实际差异判定，不是统计显著性检验。
普通SQ8另重放一次已验收冻结版本并检索，不进入参数排名：
默认参考为 sq4_sq8_laion100m_incremental_20261107/S2，
二进制哈希必须匹配，保留其已验收参数，输出路径全部重定向到本任务。
这1次参考构建及10M门禁不计入19次参数构建。

--resume复用已完成或已记录失败的轮次，不覆盖任何结果。
发生中断且未形成结果的目录会保留并报错，需先检查现场，不能静默重做或覆盖。
GPU锁防止两个扫描器同时运行；NVML在构建前及运行中检查竞争GPU进程。
OOM和准入拒绝分别记录，失败不降低阈值重试，也不重置GPU。

## 计时、质量和清理

完整墙钟为构建进程GNU time，包含图写出，输出及临时文件统一使用独立tmpfs目录。
预热、校验、图审查、检索、归档与清理均在计时外。每轮复制相同冻结CUDA缓存，
重新训练、编码和D0分配。初始缓存以实际逐页触读、smaps RSS和最后一遍零新增物理读取/
major faults核验，不使用可能受权限保护的mincore；运行中实际读写另报。

每100 ms直接调用NVML采样GPU进程及整卡显存，保存实际时间戳；可能遗漏更短峰值。
每个子树保存行数、模型需求/准入字节、split深度和完成计时；根分区分布及split距离对数
从已有profile和完整日志恢复。距离对数为每次split的rows×split_count之和。
既有D0码/成员回传只有合计时间，字节数可分开，不把合计时间伪拆成两项。
原向量host_read是累计读取/拷贝时间，不是纯磁盘等待。
split、gather、等待及GPU时间存在流水重叠，不能相加为墙钟。
GPU每百万成员耗时与子树分桶吞吐是工作负载相关指标，不是SM硬件效率的直接测量。

每轮检索前预热完整底库、当前图及查询/真值，先做一次L64预热，再测
L64/128/256/512；固定10,000查询、8线程、mmap/exact和Recall@10。
95%/97%同召回QPS只在相邻实测点覆盖时插值，不外推；标明可覆盖目标的轮数。
图审查要求节点数正确、度数≤64、无非法邻居、重复边和自环。

仅在构建审计、图审查、四档检索结果和图SHA256全部检查成功后，
删除本任务拥有的图及临时目录，包括基线/优选图；清理检索输出的邻居数组，
保留原始TSV、日志、profile、资源采样、命令、源码差异、可重放二进制及哈希。
清理校验所有权标记、直接子目录关系及图身份；历史文件不在清理范围。
失败现场保留，磁盘或主机资源不足时停止，不通过删除历史现场恢复运行。

## 报告与验证状态

每轮有manifest/env/cmd/profile/time/resources、metrics、subtrees、splits、root-sizes、
图审查和quality记录。任务级有plan、execution、status、inputs、
finalists、noise-confirmations、summary、runs.csv、quality.csv、report.md及PNG/PDF热力图。

~~~bash
python3 -m unittest discover -s script -p 'test_sq4*.py' -v
python3 script/run_sq4_capacity_sweep.py \
  --run-dir runs/sq4_capacity_sweep_100m --report-only
~~~

本次实现已做针对性测试及8193点GPU固定输入核对。
**尚未执行新入口的10M全量门禁或100M扫描，性能/质量结论均待验证。**
脚本不会修改生产默认配置。速度更快而质量下降的情况会原样报告；
50%配置未OOM也不代表已经实际覆盖并验证50%的容量边界。
