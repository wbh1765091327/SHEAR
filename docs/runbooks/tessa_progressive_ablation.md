# TESSA 渐进式消融实验运行手册

本手册是 Wiki10M 上 TESSA A0--A3 正式消融的唯一执行协议。可执行参数的单一事实来源是
`script/run_tessa_progressive_ablation.sh`；使用 `print-contract` 生成完整命令，禁止另行手写或临时修改
变体参数。

## 1. 实验边界

本实验只比较 D1+ 子树 assignment 与 leaf scoring 的表示和执行器。四种模式共同固定：

- D0：`gpu-kmeans-coarse`、GPU-D0 SQ staging、D0 `wmma-batch`；
- fanout：top/second/deeper = `6/3/1`，root risk candidates 为 `1,2,4,6`；
- seed 42，`R=64`、`Lbuild=100`、`leaf-knn=2`、leaf size `256..1280`；
- 八个 CPU 线程，16 GiB CUDA workspace，buffered OOM I/O；
- 12-bit hash 与 SoftTier-Q3：`rhalf-soft-tier-q3 / none / 0 / 1.0`；
- 相同 release binary、相同数据与查询、相同缓存协议。

四个主变体为：

| 模式 | D1+ 表示 | Assignment | Leaf scoring | large-ticket batch |
|---|---|---|---|---:|
| A0 | 原始 FP32 | tiled FP32 SIMT | tiled FP32 SIMT | 1（不归因） |
| A1 | 原始 FP32 | TF32 WMMA | TF32 WMMA | 1（不归因） |
| A2 | SQ8 | warp DP4A | warp DP4A | 1 |
| A3 | SQ8 | signed-int8 WMMA | signed-int8 WMMA | 8 |

A0/A1 通过 raw-D1 diagnostic 仅让 D1+ 消费原始 FP32，D0 仍保持 GPU-D0 SQ staging。A2--A3 同时改变
执行器和 batch，因此只能解释为完整 TESSA 包的增量。DP4A/WMMA × batch-1/batch-8 的交互仅由同一批
冻结 assignment 任务的 2×2 replay 解释；batch-8 不作用于 leaf scoring。

## 2. 规范输入与输出

输入：

```text
base:  /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
       header = 10000000 × 1024
query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
       header = 10000 × 1024
GT:    /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
       header = 10000 × 100
```

输出根必须是：

```text
/data_staging/wbh/tessa_ablation/wiki10m/<YYYYmmdd_HHMMSS>/
```

输出树：

```text
RUN_ROOT/
├── material_passport.md
├── commands.sh.txt
├── prior_evidence.md
├── capture/
│   ├── fragments/
│   ├── provenance.json
│   └── final/manifest.json
├── operator_replay/
│   ├── samples.tsv
│   ├── task_metrics.tsv
│   ├── summary.json
│   └── summary.md
├── full_builds/A{0,1,2,3}/run{1,2,3}/
├── search/A{0,1,2,3}/L{25,50,100,200,400}/
├── results.tsv
├── summary.json
└── summary.md
```

## 3. 构建与实现验证

仓库使用 nightly 工具链。若 `cargo` 不在默认 PATH：

```bash
cd /path/to/SHEAR
export PATH=/path/to/rust/bin:$PATH
```

构建四个正式二进制：

```bash
cargo build --release \
  -p build_shear_index \
  -p search_shear_index \
  -p tessa_operator_replay \
  --features build_shear_index/cuda,tessa_operator_replay/cuda
cargo build --release -p search_shear_index --bin graph_quality
```

在生成任何正式命令前必须完成：

```bash
cargo fmt --all -- --check
cargo test -p shear_core tessa_ --lib -- --nocapture
cargo test -p build_shear_index tessa_ -- --nocapture
cargo test -p tessa_operator_replay -- --nocapture
cargo test -p shear_core \
  cuda_persistent_subtree_kernel_nvrtc_compiles_with_sq_int8_wmma \
  --features pipnn_cuda --lib -- --nocapture
cargo test -p shear_core \
  cuda_persistent_sq8_large_wmma_matches_dp4a_on_device \
  --features pipnn_cuda --lib -- --nocapture
cargo test -p shear_core \
  cuda_persistent_sq8_leaf_wmma_matches_dp4a_on_device \
  --features pipnn_cuda --lib -- --nocapture
cargo test -p shear_core \
  cuda_tessa_replay_sq8_dp4a_and_wmma_are_bit_exact \
  --features pipnn_cuda --lib -- --nocapture
python3 -m unittest -v script/test_tessa_progressive_ablation.py
```

CUDA 测试若因设备不可用而跳过或失败，不得进入正式实验。

## 4. 命令展示与二次批准

生成一个尚不存在的 run root：

```bash
cd /path/to/SHEAR
SHEAR_TESSA_RUN_ID="$(date +%Y%m%d_%H%M%S)"
SHEAR_TESSA_RUN_ROOT="/data_staging/wbh/tessa_ablation/wiki10m/${SHEAR_TESSA_RUN_ID}"
printf '%s\n' "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh print-contract \
  "${SHEAR_TESSA_RUN_ROOT}"
```

`print-contract` 必须只打印、不创建 `RUN_ROOT`。将它输出的完整 preflight、capture、replay、A0--A3
build、search、summarize 和 cleanup 命令展示给用户，取得第二次明确批准后才能执行。实现方案的批准不等于
正式长跑批准。

批准后按顺序逐阶段执行；每阶段结束先检查产物，不能一次性串联：

```bash
bash script/run_tessa_progressive_ablation.sh preflight "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh capture "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh replay "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh screen "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh repeat "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh search "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh summarize "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh cleanup "${SHEAR_TESSA_RUN_ROOT}"
```

## 5. Phase 合同

### preflight

检查三个输入头、四个 release 二进制、GPU、磁盘空间和 run-root 边界。记录 Git commit、dirty diff hash、
二进制 SHA-256、环境、GPU 和完整命令矩阵。历史 2026-07-12/15/18 结果写入 `prior_evidence.md`，但因
二进制、配置或变体矩阵不匹配而明确标为不可复用。

### capture

使用与 A3 相同的 binary 和合同单独执行一次非计时归因的生产任务捕获。捕获 I/O 不计入任何性能结果。
assignment 分桶为行数 `256..1024 / 1025..8192 / >8192` × leader 数
`<=16 / 17..32 / 33..64 / 65..128`，每个非空桶最多 16 个任务；leaf 分桶为
`256..512 / 513..896 / 897..1280`，每桶最多 64 个任务，seed 42。finalize 必须验证 profile 与
provenance 中的 `root_leader_hash`、`d0_partition_hash`、`d0_task_list_hash` 一致。

### replay

单进程加载一次冻结任务，完成 5 个 warmup sweep 和 20 个 measured sweep。按区组轮换 A0--A3 和两个
辅助 assignment 模式，输出 GPU event time、pairs/s、logical payload、Overlap@r_h、Overlap@k 与 safety
counters。A2/A3 的候选 ID 和 f32 距离位必须逐项完全一致；任何差异立即失败。

### screen

from-scratch 顺序固定为 A0、A1、A2、A3。每个 build 由
`timeout --signal=TERM --kill-after=30s 60m` 包裹并用 `monitor_gpu_build.sh` 采样；每 30 秒记录 heartbeat，
连续 90 秒无输出只告警。退出非零、60m 超时、CUDA 错误或门禁失败时停止，不重跑、不替换样本。

每个索引必须由 `graph_quality` 验证 10,000,000 nodes、`self_loop_edges == 0`、邻居不越界，并记录出度、
互反性和可达性。profile 必须符合 requested/actual backend、DP4A/WMMA tile、ticket claims/physical tiles，
且 fallback、queue overflow、watchdog trip 全为 0。四模式与 capture 的三个 D0 hash 必须全部一致。

### repeat

筛选样本为 repeat 1。其余顺序固定：

```text
repeat 2: A3, A2, A1, A0
repeat 3: A1, A3, A0, A2
```

失败样本保留且不自动补跑。repeat 2/3 的索引只在 profile、time、graph quality、index bytes 和
`build_record.json` 落盘后删除；每种模式的 repeat 1 索引保留给搜索。

### search

固定 `L=25,50,100,200,400`。每个 L 先对 A0--A3 各预热一次，再运行三次 measured search；模式顺序为
repeat 1 的 A0/A1/A2/A3、repeat 2 的 A3/A2/A1/A0、repeat 3 的 A1/A3/A0/A2。每个单元记录
Recall@10、QPS、wall 和 result bytes。

### summarize

主要 build 指标使用三次的中位数与 `[min,max]`；search 的 Recall@10/QPS 使用均值与范围。A2--A3 对每个
相同 repeat ordinal 计算：

```text
gain_r = (A2_r - A3_r) / A2_r
```

external wall 和 persistent event 分别报告三项 gain 的中位数与范围。只有中位数至少 3% 且最小 paired
gain 大于 0，才称该测量层收益稳定；否则使用“未观察到稳定系统收益”。

每个固定 L 用三次 Recall@10 均值与 A0 比较；任一损失 `> 0.005`，不得声称质量保持。matched-recall
QPS 的目标为 Recall@10 = 0.95，只能在相邻实测点之间线性插值；未包围 0.95 时报告 `NA`，禁止外推。

### cleanup

只有以下条件全部满足才允许清理：12 个 build records 完整、四个 retained indexes 均有结构检查和
`index_bytes`、60 个 measured search records 完整、`results.tsv/summary.json/summary.md` 已生成。只删除
run root 内明确列入 `cleanup_manifest.txt` 的 `_mem.index`、SQ sidecar 和 OOM 临时文件；保留 capture ID、
manifest、日志、profile、GPU samples、搜索结果和汇总。

## 6. 结果解释边界

- A0--A1 同时包含执行硬件路径与 TF32 舍入，不能只归因于 Tensor Core。
- A1--A2 同时改变原始 FP32/TF32 与 SQ8/DP4A，不能只归因于压缩或 DP4A。
- A2--A3 同时改变 DP4A→WMMA 和 batch-1→batch-8，只能称为完整 TESSA 包。
- 仅冻结 assignment 的 2×2 replay 可讨论 executor 主效应、batch 主效应及交互；不能外推到 leaf。
- 若 GPU event 改善而 external wall 不改善，只报告算子/内核局部收益。
- task Overlap 不能替代完整索引的 Recall@10--QPS 质量验证。

论文中的 TESSA 表格和结论只能在全部门禁通过后从 `results.tsv`、`summary.json` 和 material passport 引用；
旧实验不能作为本次三重复中的任何一个样本。

## 7. Post-hoc SQ8 full-build 2×2 扩展

本扩展补齐原正式实验缺失的两个系统级单元格。它复用已验收父运行中的 A2/A3，但不修改父运行的
`results.tsv`、`summary.json`、`summary.md` 或 A0--A3 定义。

固定父运行：

```text
/data_staging/wbh/tessa_ablation/wiki10m/20260721_011618
```

完整矩阵为：

| Executor | batch 1 | batch 8 |
|---|---|---|
| warp-DP4A | D1 = A2（已有） | D8（新增） |
| signed-int8 WMMA | W1（新增） | W8 = A3（已有） |

W1 只将 A3 的 large-ticket batch 从 8 改为 1；D8 只将 A2 的 large-ticket batch 从 1 改为 8。
assignment 与 leaf executor 在每个单元格内保持一致，batch 仍只作用于 large assignment。

### 7.1 合同展示与批准

首先只打印命令，不创建扩展目录：

```bash
bash script/run_tessa_progressive_ablation.sh print-2x2-contract \
  /data_staging/wbh/tessa_ablation/wiki10m/20260721_011618
```

必须向用户展示完整输出并取得正式长跑批准，之后才能执行 `preflight-2x2`。preflight 会验证父运行
材料、A2/A3 三次构建、A0/A2/A3 搜索记录、三个 D0 hash，以及当前 release 二进制与父运行记录的
SHA-256 完全一致；任何不匹配都停止，不重建二进制或替换基线。

### 7.2 分阶段执行

批准后逐阶段运行并检查每阶段产物：

```bash
SHEAR_TESSA_PARENT=/data_staging/wbh/tessa_ablation/wiki10m/20260721_011618
bash script/run_tessa_progressive_ablation.sh preflight-2x2 "${SHEAR_TESSA_PARENT}"
bash script/run_tessa_progressive_ablation.sh screen-2x2 "${SHEAR_TESSA_PARENT}"
bash script/run_tessa_progressive_ablation.sh repeat-2x2 "${SHEAR_TESSA_PARENT}"
bash script/run_tessa_progressive_ablation.sh search-2x2 "${SHEAR_TESSA_PARENT}"
bash script/run_tessa_progressive_ablation.sh summarize-2x2 "${SHEAR_TESSA_PARENT}"
bash script/run_tessa_progressive_ablation.sh cleanup-2x2 "${SHEAR_TESSA_PARENT}"
```

新增产物只写入：

```text
PARENT_RUN/full_build_2x2/
```

W1/D8 各执行三次 from-scratch 构建。顺序为 repeat 1 的 W1/D8、repeat 2 的 D8/W1、repeat 3 的
W1/D8。每个模式的 repeat-1 索引用于 `L=25,50,100,200,400` 搜索；每个 L 先预热，再按 W1/D8、
D8/W1、W1/D8 完成三次正式搜索。

每个构建必须满足 requested/actual executor 与 batch 合同、相关 tile 计数大于零且另一执行器计数为
零、fallback/overflow/watchdog 为零、图结构有效，并与 A2/A3 共享三个 D0 hash。每个搜索记录必须
包含 10,000 个 query、非空结果以及可解析 Recall@10/QPS。

### 7.3 汇总与解释

四个单元格分别报告三次 external wall 与 persistent event 的中位数和 `[min,max]`。A2/A3 早于
W1/D8 完成，因此禁止按 repeat ordinal 配对；只报告以下 cell-median 比值：

```text
WMMA@batch1       = median(D1) / median(W1)
WMMA@batch8       = median(D8) / median(W8)
batch8 under DP4A = median(D1) / median(D8)
batch8 under WMMA = median(W1) / median(W8)
interaction       = WMMA@batch8 / WMMA@batch1
```

速度比大于 1 表示分母单元格更快；interaction 大于 1 表示 batch-8 增强了 WMMA 相对 DP4A 的优势。
只有 wall 与 persistent event 同时支持时，才声明 WMMA 在相应 batch 下更快。W1/D8 的每个固定 L
Recall@10 相对 A0 的最大允许损失仍为 0.005，并继续报告 matched Recall@10=0.95 QPS。

本扩展完成后，原手册中“只有冻结 assignment replay 可拆分 executor、batch 与交互”的限制只适用于
原 A0--A3 主实验；新的 `full_build_2x2/` 可提供系统级拆分证据。它仍是 post-hoc、非同时交错基线，
不得隐藏这一时间复用限制。cleanup 只能删除扩展目录内的新 `_mem.index`、SQ sidecar 和 OOM 临时文件。

## 8. SQ8 leaf DP4A 修复后的 fresh full-build 2×2

若 `warp-dp4a` leaf executor 的实现发生变化，不得将新 DP4A 单元与第 7 节的旧 WMMA 单元组合为正式
比较。应使用同一 release binary 重新执行四个 SQ8 单元：

| Executor | batch 1 | batch 8 |
|---|---|---|
| packed DP4A | D1 = A2 | D8 |
| signed-int8 WMMA | T1 = W1 | T8 = A3 |

每个单元执行三次 from-scratch 构建。固定顺序为 run 1 的 A2/W1/D8/A3、run 2 的
A3/D8/W1/A2，以及 run 3 的 W1/A3/A2/D8。batch 只改变 large assignment 的领取粒度；assignment
与 leaf executor 在每个单元内保持一致。

### 8.1 合同展示与批准

生成尚不存在的 run root，并只打印合同：

```bash
cd /path/to/SHEAR
SHEAR_TESSA_RUN_ID="$(date +%Y%m%d_%H%M%S)_leaf_dp4a_fix"
SHEAR_TESSA_RUN_ROOT="/data_staging/wbh/tessa_ablation/wiki10m/${SHEAR_TESSA_RUN_ID}"
bash script/run_tessa_progressive_ablation.sh print-fresh-2x2-contract \
  "${SHEAR_TESSA_RUN_ROOT}"
```

`print-fresh-2x2-contract` 只能打印，不能创建 run root。必须向用户展示完整输出并取得正式长跑批准，
然后才能执行 preflight；实现修复的批准不等于正式实验的执行批准。

### 8.2 分阶段执行

批准后逐阶段执行并检查产物：

```bash
bash script/run_tessa_progressive_ablation.sh preflight-fresh-2x2 "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh screen-fresh-2x2 "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh repeat-fresh-2x2 "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh search-fresh-2x2 "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh summarize-fresh-2x2 "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh cleanup-fresh-2x2 "${SHEAR_TESSA_RUN_ROOT}"
```

preflight 检查 Wiki10M base、query 与 ground truth 的 header，验证 ground-truth ID 范围、GPU 与剩余
空间，并记录二进制、脚本、git diff 和环境哈希。每次构建必须满足 executor/batch 合同，DP4A 或
WMMA 的 assignment 与 leaf tile 计数必须同时非零，另一执行器的计数必须为零；三个 D0 hash 在 12 次
构建中必须一致，且 fallback、overflow 与 watchdog 均为零。

每个单元保留 run-1 索引，并在 \(L\in\{25,50,100,200,400\}\) 上先预热、再执行三次正式搜索。
汇总报告 external wall、persistent event、固定 \(L\) 的 Recall@10/QPS 以及 matched
Recall@10 \(=0.95\) QPS。四个单元由同一二进制交错执行，因此可以报告 cell-median executor、batch
及交互比值；不得再将第 7 节旧 D1/D8 的 scalar-int8 结果称为 DP4A 结果。

cleanup 仅删除当前 run root 中已列入 manifest 的 `_mem.index`、SQ sidecar 与 OOM 临时文件，并保留
profile、计时、GPU samples、图检查、搜索记录、命令合同和最终汇总。

### 8.3 仅重测修复影响的 DP4A 单元

若目的仅是验证 leaf DP4A 修复及 batch-1/batch-8 差异，可将范围收缩为 D1=A2 与 D8 两个单元，但不得
据此重新估计 WMMA 相对 DP4A 的速度比。使用以下独立合同：

```bash
bash script/run_tessa_progressive_ablation.sh print-fresh-dp4a-contract \
  "${SHEAR_TESSA_RUN_ROOT}"
```

合同获批后依次执行：

```bash
bash script/run_tessa_progressive_ablation.sh preflight-fresh-dp4a "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh screen-fresh-dp4a "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh repeat-fresh-dp4a "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh search-fresh-dp4a "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh summarize-fresh-dp4a "${SHEAR_TESSA_RUN_ROOT}"
bash script/run_tessa_progressive_ablation.sh cleanup-fresh-dp4a "${SHEAR_TESSA_RUN_ROOT}"
```

D1/D8 各执行三次 from-scratch 构建，顺序为 A2/D8、D8/A2、A2/D8。两个 run-1 索引均执行相同的
Recall@10--QPS sweep；汇总只报告修复后的 DP4A 计时、D1/D8 的 batch 差异和二者之间的质量一致性。
