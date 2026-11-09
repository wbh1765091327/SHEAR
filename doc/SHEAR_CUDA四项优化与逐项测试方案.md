# SHEAR CUDA 四项优化与逐项测试方案

日期：2026-11-03。

**状态：首轮 O1、O2-A、O3-A/B、O4-A 已按独立开关实现并测试；推荐组合为 mask 26。** 实际命令、原始样本、资源属性、构建与搜索结果见[本轮实验记录](/home/wbh/rswork/SHEAR/runs/cuda_four_optimizations/b0_20261103/report.md)。O2-B、O4-B/C 仍为后续探索项；文中的步骤和预期描述保留作为复现实验的设计依据。

完整优化清单见配套文档：[SHEAR CUDA CTA 线程利用与调度优化方案](./SHEAR_CUDA_CTA线程利用与调度优化方案.md)。本文继续负责 O1～O4 的详细实验设计；配套文档保留 P1～P6、C1～C4、G1～G5、S1 的完整编号，并展开新增的成员分发、任务粒度、发布与调度方案。

| 配套文档编号 | 与本文的关系 |
|---|---|
| G1：C tile 布局 | 对应 O1，详细方案见第 4 节 |
| C1：候选状态 | 对应 O2，详细方案见第 5 节 |
| G2/G3：共享 B、较宽维度搬运 | 分别对应 O3-A/O3-B，详细方案见第 6 节 |
| G4/G5：距离恢复、候选维护线程分工 | 分别对应 O4-A、O4-B/C，详细方案见第 7 节 |
| P1～P6、C2～C4、S1 | 新增优化环节，详细方案见配套文档；分别建立独立对照实验 |

O3 的向量维度搬运与 P1 的分区成员 ID 搬运属于不同步骤。O2 对 staged 输出的验证也不等于 C4 的行锁/归并协议优化。两份文档共用基线和记录规范，但不把多个编号的同时改动当成单项收益。

## 1. 实验目标与范围

针对当前 SQ8 WMMA 路径，依次尝试以下四项优化：

| 编号 | 优化项 | 解决的问题 | 主要作用阶段 |
|---|---|---|---|
| O1 | 调整 C tile 的存储与读取布局 | 后处理跨行读取 C，可能产生 shared memory bank conflict | 分区距离计算、叶内候选计算 |
| O2 | 精简候选状态与改善寄存器使用 | 通用数组、中转状态及动态索引可能增加指令、寄存器或 local memory 开销 | 主要是叶内候选计算 |
| O3 | CTA 内共享 B，并单独尝试更宽的维度搬运 | 不同 warp 重复加载同一目标块，维度搬运粒度较小 | 分区距离计算、叶内候选计算 |
| O4 | 重新分配后处理线程 | WMMA 完成后，通常只有前 16 个 lane 恢复距离、维护候选 | 分区距离计算、叶内候选计算 |

本方案首先分析 persistent 大分区 assignment 与通用 SQ8 WMMA leaf 路径。D0、DP4A、TF32、SQ4、RabitQ、D96/K2 专用路径分别记录，不混入同一组性能比较。

本方案的“分区 CTA”和“候选计算 CTA”指承担对应工作的 CTA。persistent kernel 可以承载多个工作阶段，实际需要记录执行的函数和分支，不能只根据 kernel 名判断。

### 1.1 四个优化在计算流程中的位置

```text
全局内存中的 SQ8 量化码
        ↓
按 ID 找到源向量、leader 或候选向量
        ↓
加载到 shared memory，组织 A/B tile    ← O3
        ↓
WMMA 沿维度累加内积
        ↓
将 accumulator 写成 C tile            ← O1
        ↓
读取 C，恢复距离                       ← O1、O4
        ↓
维护 leader 结果或候选 TopK             ← O2、O4
        ↓
写出分区归属或候选结果
        ↓
计数、scatter、发布任务或跨任务归并
```

每项都先与同一个原始基线比较。通过验证后，再单独建立累计组合，评价优化之间的交互。

### 1.2 必须区分的数量

| 名称 | 含义 | 本方案中的处理 |
|---|---|---|
| `D` | 原始向量维度 | 覆盖低维与高维 |
| `padded_dim` | 实际参与量化内积的补齐维度 | 记录实际值，确保补齐位置正确 |
| WMMA `M/N/K` | 一次矩阵计算的行数、列数、归约长度 | 当前 SQ8 子块主要为 16×16×16 |
| `local_fanout` | 分区阶段给一条向量保留多少个 leader 对应的子分区 | 与候选输出数量分别控制 |
| `k` / `leaf_knn` | 叶任务为一条向量保留多少个输出邻居 | 当前源码默认 `leaf_knn=2`；每次记录实际配置 |
| `leaf_dco_survivors` | 精排前的候选保留数量 | 精排场景可能大于输出 `k` |
| 数组容量 32 | 通用实现可容纳的最大状态数量 | 不代表实际一定保留或输出 32 个 |

不精排、`k=2` 时，量化筛选可以直接维护 Top2；启用精排且 survivor 数为 32 时，才可能出现“先保留 32 个，再精排输出 2 个”的流程。

## 2. 源码定位与已有测试

主要源码：[compute_backend.rs](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs)。以下行号是编写本文时的位置，后续优先按符号定位。

| 符号或代码 | 当前定位 | 用途 |
|---|---|---|
| `GpuPersistentSq8WmmaScratch` | [4977 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:4977) | 每个 warp 的 A/B/C 及元数据缓冲 |
| `gpu_persistent_large_task_sq8_wmma_range_slots4` | [14812 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:14812) | 大分区 SQ8 WMMA 距离与 leader 选择 |
| `gpu_persistent_compute_topk_leaf_tile_cta_sq_int8_wmma_filter` | [10485 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:10485) | 通用 SQ8 WMMA 候选计算 |
| `gpu_persistent_sq_wmma_consume_sq_filter_c_tile` | [10142 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:10142) | 从 C 恢复距离，维护 survivor |
| `best_*`、`survivor_*` 数组 | [10667 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:10667) | O2 的主要检查对象 |
| `gpu_persistent_leaf_insert_survivor_fp32` | [8156 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:8156) | 候选插入与距离容差规则 |
| `gpu_persistent_merge_topk_row_locked` | [7349 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:7349) | staged 输出的跨任务归并 |
| `persistent_sq8_leaf_mode_code` | [582 行附近](/home/wbh/rswork/SHEAR/shear-core/src/algorithm/pipnn/compute_backend.rs:582) | leaf 模式选择条件 |

已有设备正确性测试可作为起点，不能替代为新方案增加的边界用例：

```bash
cd /home/wbh/rswork/SHEAR
source script/local_env.sh
export PATH="/usr/local/cuda/bin:$PATH"

# 大分区 WMMA 与 DP4A 比较。
cargo test -p shear_core --features pipnn_cuda --lib \
  cuda_persistent_sq8_large_wmma_matches_dp4a_on_device \
  -- --nocapture --test-threads=1

# 通用 leaf WMMA 与 DP4A 比较。
cargo test -p shear_core --features pipnn_cuda --lib \
  cuda_persistent_sq8_leaf_wmma_matches_dp4a_on_device \
  -- --nocapture --test-threads=1

# 如果改动影响 D96/K2 共享的代码或 scratch，补跑这一项。
cargo test -p shear_core --features pipnn_cuda --lib \
  cuda_persistent_sq8_d96_k2_leaf_matches_scalar_on_device \
  -- --nocapture --test-threads=1
```

上述命令已用于本轮设备验证，并增加 D112 宽搬运尾部、组合变体、staged 归并和 Compute Sanitizer 检查。检查输出中的实际执行测试数量；编译成功或过滤后执行 0 个测试，不算设备验证通过。设备测试日志见[实验目录](/home/wbh/rswork/SHEAR/runs/cuda_four_optimizations/b0_20261103/)。

## 3. 先建立基线 B0

### 3.1 固定实验条件

开始修改前保存：源码提交号、工作区补丁、CUDA 源码哈希、构建命令、编译选项、驱动与 GPU 信息、完整运行参数、输入与任务清单哈希。

当前工作区已有其他修改，基线必须覆盖工作区实际状态，不能只记录 Git 提交号。保存补丁或使用独立实验目录，避免覆盖现有改动。

首次环境检查得到：

- GPU：NVIDIA GeForce RTX 4090，约 24 GiB 显存，compute capability 8.9。
- CUDA 工具位于 `/usr/local/cuda/bin`，目录当前对应 CUDA 12.9。
- Nsight Compute 查询返回 `ERR_NVGPUCTRPERM`，当前用户无法读取 GPU 性能计数器。

因此，后续可以先用 CUDA event 计时、编译/加载后的资源属性和设备正确性测试推进。bank conflict、tensor 活跃度等硬件指标在权限可用前标记为“未采集”，不根据耗时反推为已证实。本方案不要求修改驱动权限或系统配置。

### 3.2 固定真正执行的路径

先测试 dense SQ8、WMMA、无精排、无 audit 的通用路径。记录 assignment 与 leaf 的实际执行器、leaf 模式、任务行数、large-ticket batch、列任务切分方式。

低维数据可能由 Auto 选择 row-owned；D96/K2 还有显式专用实现。需要证明被测任务实际进入目标函数，避免“改了通用函数，却测到其他分支”。必要时使用现有显式配置或测试 harness 直接调用目标函数。

先分别测 assignment 和 leaf，再测完整 persistent 阶段。完整阶段包含计数、scatter、队列和结果归并，优化距离计算不保证同比例改善总时间。

已有 TF32 shape sweep 和 `SHEAR_PIPNN_WMMA_CTA_SHARED_B` 不能直接作为本次 SQ8 实验的性能证据：需要核对它们控制的函数是否就是被修改的 SQ8 路径。

### 3.3 固定输入，避免任务变化干扰

算子实验优先冻结成员 ID、leader ID、leaf row ID、norm、scale、量化码和任务边界。基线与变体读取相同输入，使用独立且初始化一致的输出与计数器。

多次重新运行完整分区，即使随机种子相同，也可能因调度和输出排列不同产生不同任务；不能直接把这些差异归因于 tile 优化。

固定任务 replay 能回答“同一批计算是否更快”；完整构建能回答“实际构建是否更快”。两层都需要，但先完成前者。

### 3.4 建议覆盖的数据形状

| 维度或数量 | 首轮测试 | 边界扩展 |
|---|---|---|
| 原始维度 D | 96、128、256、512、1024 | 158 等需要补齐的维度，记录实际 padded_dim |
| 分区源行数 | 128 | 19、123，以及不满一个 warp 行块的任务 |
| leader 数量 | 16、32、128 | 17，验证最后一个目标 tile |
| 分区 fanout | 1、2、4 | 实际生产配置 |
| leaf 行数 | 128、256、512 | 17、19、127、129、513，验证尾块与 staged 切分边界 |
| leaf 输出 k | 2、4 | 通用回退路径补测 8、32 |
| 任务数量 | 足够让 GPU 有充足工作 | 单 CTA 正确性与大量 CTA 性能分别测试 |

leaf 行数与向量维度是两个独立变量。数据集总体很大，也可能产生很小的 leaf，不能忽略任务内部尾块与无效 warp。

先做代表性组合，再对每项优化增加有针对性的边界组合，不必立即运行所有变量的笛卡尔积。

### 3.5 通用验证与计时规则

**正确性：** 先比较完整 int32 内积，再比较距离、邻居/leader ID、分区归属、输出数量与错误标志。纯数据搬运与布局改动应保持内积完全一致；默认要求相同输入顺序下结果 ID 与基线一致。

距离比较保留原来的计算顺序、显式舍入和编译选项。若不能逐位一致，先定位原因，再约定绝对/相对误差阈值；不能仅因距离差很小就接受 TopK ID 改变。

增加零向量、重复向量、相等距离、接近距离容差的候选、自身候选、无效填充、源/目标尾块、空任务上层处理等用例。无效填充是正常尾块，不应误报成越界 ID；真正错误的 ID 继续遵循原有错误处理。

**计时：** 先完成编译、模块加载、数据上传和至少 5 次预热，再记录至少 20 个样本。小 kernel 用多次连续执行组成计时批次，基线与变体的批次长度相同；使用同一 CUDA stream 的 event，并检查同步和执行错误。

纯算子时间与包含上传/下载的时间分开记录。性能 replay 与 sanitizer/profiler 运行分开，避免工具开销污染计时。记录 median、P25/P75、min/max 与原始样本，交替运行 B0 和变体，确认提升超过基线自身波动。

**资源：** 记录最终加载 kernel 的寄存器数、静态/动态 shared memory、local memory，并在可用时检查 occupancy、spill/local load-store、shared bank conflict、global sectors、缓存命中与等待原因。辅助函数的数组声明不是实际资源分配结果；NVRTC 路径应检查实际 PTX/JIT 加载后的 kernel，不能只拿另一个 nvcc 编译结果代替。

## 4. O1：C tile 的存储与读取布局

### 4.1 要解决的问题

当前结果按行存储，前 16 个 lane 各负责一条源向量：

```cpp
// 当前：线程 r 负责源向量 r。
int dot = c_tile[lane * 16 + col];
```

同一条读取指令中，相邻 lane 相隔 16 个 int。按常见的 32 个 bank、每个 32 位字依次映射 bank 的规则，16 个 lane 聚集在两个 bank 的不同地址，存在 8 路冲突的风险。最终要核验编译后指令与硬件指标。

### 4.2 单项改动

```cpp
// 概要：直接按列写 C，让相同列的不同行位于连续地址。
nvcuda::wmma::store_matrix_sync(
    c_tile, acc_frag, 16, nvcuda::wmma::mem_col_major
);
__syncwarp();

// 细节：逻辑 C[row][col] 不变，只改变其物理下标。
int dot = c_tile[col * 16 + lane];
```

先生成 O1-P，只修改分区路径；再生成 O1-L，只修改 leaf 路径。二者分别对照 B0，最后才测试同时启用。

同步修改该 C tile 的所有读取者，包括共用 helper、调试读取和直接检查 scratch.c 的测试。不要全文件替换所有 C 下标，以免误改 TF32、SQ4 或其他仍使用行存的路径。

### 4.3 验证顺序

1. 用已知矩阵检查 C 的 256 个逻辑元素，确认列存地址公式正确。
2. 对完整维度累加结果逐元素比较 scalar/DP4A 内积。
3. 比较分区 leader 选择、leaf 候选 ID 与距离。
4. 补测 17 个 leader、19 条源行等不完整 tile。
5. 运行对应现有设备测试。原测试若直接按行读 scratch.c，应改为正确读取逻辑元素，保留独立 scalar oracle。
6. 先测结果处理片段，再测包含 WMMA store 的整个函数与 persistent 阶段。

### 4.4 判断收益

预期减少 C 读取冲突，不改变点对数量、输入量化码布局或 TopK 算法。需要同时观察 WMMA store、C load 与总耗时；store 变慢可能抵消读取收益。

硬件计数器暂不可用时，只能报告“列存版本更快/更慢及幅度”，不能报告“实测消除了 8 路冲突”。

## 5. O2：候选状态与寄存器使用

### 5.1 要解决的问题

通用 leaf 路径声明多组容量为 32 的 `best_*`、`survivor_*` 数组。无精排时，survivor 中的结果随后复制到 best 再输出，可能存在中转状态与动态索引开销。

固定容量可能增加初始化、寄存器或 local memory 开销，但编译器可能消除部分状态或复用存储；必须先检查实际 kernel，不能直接按数组元素数计算“浪费的寄存器”。

分区路径已主要使用容量为 4 的 leader 状态，本项优先实验 leaf，不能直接套用同样的收益预期。

### 5.2 首轮方案：保持容量语义，消除中转

先不决定是否把容量 32 改为 2 或 4。对 dense SQ8、无精排、无 audit 的专用分支，尝试直接从 survivor 输出或执行相同的 staged 合并，避免 best 的额外中转。

```cpp
// 当前逻辑示意：
// survivor_* -> best_* -> 最终输出/归并。

// 拟议逻辑示意：仅限无精排、无 audit 的分支。
// survivor_* -> 最终输出/归并。
// k、候选顺序、距离计算、重复处理与归并规则全部保持一致。
```

此变体编号 O2-A。包含 staged 输出时，传入归并函数的数量、距离、ID 及顺序必须与原路径相同。

如果基线编译器已经消除了中转，可能没有性能或资源变化；这也是有效实验结论，不继续宣称这部分是瓶颈。

### 5.3 可选后续方案：小 k 专用化

只有在后续决定继续研究容量时，再增加 O2-B：针对实际 k 生成编译期专用状态，或为 k≤4 提供有边界检查的专用实现。通用路径仍保留其他 k 的支持。

最终结果容量和 survivor 容量分别设计。输出 2 个、精排前保留 32 个时，不能把 survivor 一并缩到 2；分区 fanout≤4 也不能推出 leaf k≤4。

### 5.4 验证与判断收益

1. 比较 B0 与 O2-A 实际 kernel 的寄存器、local memory 与相关指令。
2. 测 k=2、4，检查所有 ID 和距离，覆盖重复向量与容差附近的候选。
3. 分别测 row-owned 直接输出和 staged 合并；检查行锁、输出数量与错误标志。
4. 原有精排、audit 与较大 k 的配置继续走原路径，确认路由正确；若改动共用 helper，则补测这些配置。
5. 记录是否确实减少中转、初始化或 local load-store，以及总耗时是否改善。

资源减少但耗时不变，说明当前工作负载可能不受该资源限制；不能仅凭寄存器数下降判定优化成功。

## 6. O3：共享 B 与扩大搬运粒度

### 6.1 要解决的问题

当前多个 warp 各自处理 16 条源向量，扫描相同 leader/候选块，但每个 warp 维护私有 B 缓冲。相同目标数据被多个 warp 请求、写入 shared，再分别加载 fragment。

另外，当前每次准备 16 维，搬运、同步与 WMMA 逐块进行。

把本项拆成两个独立实验，不能一起修改后将收益全部归因给“共享 B”。

### 6.2 O3-A：保持 16 维切片，只共享 B

```cpp
// 概要：CTA 协作加载一份 B，各 warp 保持自己的 A 与 accumulator。
for (每个目标块) {
    清空每个 warp 的 accumulator;
    for (每个16维切片) {
        CTA协作加载公共B;
        各warp加载私有A;
        __syncthreads();

        各warp从私有A和公共B加载fragment并执行WMMA;

        // 所有warp使用完B后，才能覆盖公共缓冲。
        __syncthreads();
    }
    各warp处理自己的C;
}
```

目标是减少 B 的重复全局加载、地址计算和 shared 写入。每个 warp 的 B fragment 仍各自存在，不会因为 shared B 公共化而自动合并寄存器 fragment。

需要修改的配套行为：

- 无有效源行的 warp 仍需参与 CTA 同步，不能继续提前返回。
- 目标块与维度循环次数在 CTA 内一致；尾部统一补零。
- B 的 ID、有效列数等元数据对参与 warp 一致。
- debug abort、错误处理等提前退出必须满足 CTA 同步约束。
- 第二次同步保证没有 warp 在其他 warp 读完前覆盖公共 B。

主要代价是将部分 warp 同步升级为 CTA 同步。缓存已能服务重复读取时，节省显存流量的幅度可能有限；逻辑请求减少不等于显存流量同比减少。

### 6.3 O3-B：独立尝试更宽搬运

先保留原有每 warp B，单独把搬运宽度从 16 扩大到 32 或 64 维。一次准备较宽缓冲，再用多个 K=16 的 WMMA 子块消费：

```text
搬运64维 → 消费0..15 → 消费16..31 → 消费32..47 → 消费48..63
```

调整缓冲大小、leading dimension、对齐与所有加载下标。D=96 等不能整除宽度的情况，最后一次搬运必须补零，不能越界读取。

宽搬运目标是改善连续访问、减少部分细碎操作与同步，为向量化加载创造条件；单纯把循环步长改大不保证加载指令减少，需检查实际生成代码。

O3-A、O3-B 各自完成后，再建立 O3-AB，评价共享 B 与宽搬运是否相互促进。

这两项都不会自动消除每次切换目标块时重新加载 A 的行为。完整缓存 A 或改变外层循环顺序属于额外实验，不混入首轮归因。

### 6.4 验证与判断收益

1. 分别验证各 K 切片与最终完整内积。
2. 覆盖 D=96、需要补齐的维度、高维以及搬运宽度的尾部。
3. 覆盖单个有效 warp、部分有效 warp、8 个有效 warp；检查死锁、越界和错误退出。
4. 在独立正确性运行中使用 Compute Sanitizer 检查越界与 shared 竞争；工具运行不计入性能结果。
5. 分区与 leaf 分别测，比较 16、32、64、128 个目标以及小 leaf。
6. 同时记录缓冲大小、驻留能力、同步成本和总时间。

保留标准是相同点对工作量下总时间稳定下降。若只改善大任务，应记录适用形状，考虑按形状选择，不直接全局替换。

## 7. O4：使用后处理阶段闲置的半个 warp

### 7.1 要解决的问题

当前一个 warp 协作生成 16×16 个内积，之后前 16 个 lane 各处理一条源向量。后 16 个 lane 在距离恢复、候选插入阶段通常没有有效工作。

这一现象只描述后处理阶段，不表示整个 kernel 利用率为 50%。优化要评价节省的后处理时间能否覆盖新增通信和资源成本。

### 7.2 O4-A：双线程恢复距离，保留串行 TopK 顺序

两个线程负责同一条源向量：前半 warp 处理偶数列，后半处理奇数列。计算完成后，由原来的 owner 按原列顺序更新结果。

```cpp
unsigned int row = lane & 15u;    // 两个lane对应同一源行。
unsigned int group = lane >> 4;  // 0处理偶数列，1处理奇数列。

for (unsigned int col = group; col < active_cols; col += 2u) {
    读取该源行与该目标的内积;
    恢复距离并存入临时结果;
}
__syncwarp();

if (lane < 16u && 源行有效) {
    // 与基线保持相同候选顺序，先不改变TopK算法。
    for (unsigned int col = 0; col < active_cols; ++col) {
        使用临时距离执行原来的候选插入;
    }
}
```

先对 B0 独立测 O4-A，此时沿用基线 C 布局；随后再与 O1 组合。独立版本若仍受 bank conflict 影响，应如实记录。

临时距离可以使用独立 shared 缓冲；若复用 C 缓冲，需要证明所有旧内积都在覆盖前被正确读取，并保持正确类型访问与同步。

该方案让距离恢复使用 32 个 lane，TopK 插入仍由 16 个 lane 执行。优点是保留候选处理顺序，适合作为低风险起点；代价是额外搬运和同步，可能得不偿失。

### 7.3 O4-B：双线程分别维护局部 TopK，再归并

两个线程分别处理偶数、奇数候选并维护局部 TopK，结束后归并到一份输出。可以跨多个目标块积累局部结果，减少归并频率。

这个方案能分摊更多候选比较，但新增距离与 ID 交换、局部状态及归并。小 k 值得尝试，大 k 需要单独测。

**先解决结果一致性，再讨论加速。** 当前候选比较使用距离容差与 ID：距离接近时按 ID 比较。这种近似相等关系不保证传递性，改变插入顺序可能改变结果；不能直接假定分组 TopK 的归并等价于原串行插入。

增加容差链等对抗用例：某三条距离依次接近，相邻距离差在 epsilon 内，而两端距离差超过 epsilon，并使用不同 ID 排列。若结果改变，先停留为算法实验，不把它作为保持语义的执行优化接入。

改变比较规则以构成稳定全序属于额外算法变更，需要单独评估，不混在本次线程优化中。

### 7.4 O4-C：一个 warp 处理 32 条源向量

使用两组 16 行的 A 与 accumulator，共用同一目标块 B。最终前 16 个 lane 处理第一组源向量，后 16 个 lane 处理第二组，每条向量仍由一个线程维护结果。

预期好处是分摊后处理且有机会复用 B fragment；代价是更多 accumulator、源状态和结果缓冲。

必须连同任务粒度测试：

| 组织 | 每 CTA 覆盖源行数 | 配套问题 |
|---|---:|---|
| 基线：8 warp × 16 行 | 128 | 当前默认任务行数与此匹配 |
| 4 warp × 32 行 | 128 | 检查 blockDim、scratch 与分支假设 |
| 8 warp × 32 行 | 256 | 调整任务行数与调度；小 leaf 可能浪费更多 |

若保持 128 行任务，却简单改成 8 warp × 32 行，只会把部分 lane 闲置转变为部分 warp 无源行可处理。

不要顺带把 large-ticket batch 从 8 改成其他数量。若分区需要改变 ticket 与行块的映射，单独记录为该变体的必要配套修改，并保持总成员集合与实际工作量一致。

### 7.5 验证与判断收益

分区和 leaf 分别检查元数据广播、源/目标映射、无效行、自身候选、TopK ID、tie-breaking、输出数量及尾部处理。O4-B 特别检查重排插入顺序；O4-C 特别检查任务分解和每条源行恰好被处理一次。

记录后处理片段、完整距离函数与阶段总耗时。32 个 lane 都忙不等于执行更快；比较新增 shared 操作、shuffle、归并、寄存器占用与驻留能力。

## 8. 实验执行顺序与版本编号

| 顺序 | 变体 | 唯一目的 | 对照 |
|---:|---|---|---|
| 0 | B0 | 保存实际源码基线，确认设备正确性与计时稳定性 | 无 |
| 1 | O1-P / O1-L | 分别调整分区/leaf 的 C 布局 | B0 |
| 2 | O2-A | 无精排、无 audit 路径去除候选中转状态 | B0 |
| 3 | O3-A-P / O3-A-L | 保持 K 切片宽度，只共享 B | B0 |
| 4 | O3-B-P / O3-B-L | 保持私有 B，只扩大搬运宽度 | B0 |
| 5 | O4-A-P / O4-A-L | 双线程恢复距离，保持插入顺序 | B0 |
| 6 | O4-B / O4-C | 在前述正确性基础上探索完整后处理分摊 | B0 |
| 7 | 组合变体 | 仅组合已经通过验证且有收益的修改 | B0 与各单项 |

O2-B 的容量专用化暂列可选，不阻塞其他三项测试。O3-AB、O1+O4 等组合单独命名，不用于替代独立实验。

每个变体按以下次序推进：

```text
明确改动范围与适用输入
        ↓
编译与小输入设备正确性
        ↓
边界/重复/近似相等距离测试
        ↓
必要的同步、越界检查
        ↓
固定任务算子性能与资源记录
        ↓
真实任务分布 replay
        ↓
完整构建时间与图质量检查
```

任何一步发现语义变化、错误标志或死锁，先修复，不继续以该结果计算加速比。

完整构建时分别比较 assignment、leaf、scatter/merge 与总时间，并检查 recall/QPS、输出规模等质量与工作量指标。分区输出若只因并发产生排列差异，比较以父任务和 leader 标识规范化后的成员多重集，保留 fanout 复制次数，不能仅比较原始输出字节哈希。

本轮使用 TESSA 固定任务 replay 和独立的十进制位掩码 `SHEAR_PIPNN_SQ8_WMMA_OPT_MASK`，默认值 0，具体位含义和命令见[实验记录](/home/wbh/rswork/SHEAR/runs/cuda_four_optimizations/b0_20261103/report.md)。回放和完整建图分别测量，继续遵守[TESSA 运行手册](/home/wbh/rswork/SHEAR/docs/runbooks/tessa_progressive_ablation.md)的固定输入与正式对照要求。

## 9. 每项实验的记录模板

```text
实验编号：
状态：待实现 / 正确性失败 / 无稳定收益 / 有收益 / 仅特定形状有收益
源码提交、工作区补丁和 CUDA 源码哈希：
实际执行路径、变体开关与编译选项：
GPU、驱动、CUDA、运行时间：
输入与任务清单哈希：
D / padded_dim / source_rows / target_rows：
local_fanout / leaf k / survivors / rerank / audit：
CTA线程数 / 有效warp数 / leaf_tile_rows / ticket batch / staged列切分：

正确性：内积、距离、ID、输出数量、错误标志、边界用例
同步和越界检查：
寄存器 / shared / local memory：
硬件计数器：已采集指标，或未采集及原因
预热次数 / 测量样本数 / 每样本执行次数：
B0时间：median、P25/P75、min/max
变体时间：median、P25/P75、min/max
speedup = B0 median / variant median
时间变化百分比 = (variant median - B0 median) / B0 median × 100%
实际点对数与候选处理数量：
原始样本、构建日志、完整命令与输出路径：
是否保留、适用形状与原因：
```

本轮结果汇总。数值为 LAION50k、D512、RTX 4090 上固定任务 replay 的中位耗时，单位 ms；B0 assignment/leaf/合计约 4.451/20.850/25.303。各项完整数据、P25/P75 和内核资源见[实验记录](/home/wbh/rswork/SHEAR/runs/cuda_four_optimizations/b0_20261103/report.md)与 `replay/results.tsv`。

| 变体 | mask | assignment | leaf | 合计 | 结论 |
|---|---:|---:|---:|---:|---|
| O1-P / O1-L | 1 / 2 | 4.448 / 4.449 | 20.796 / 18.577 | 25.237 / 23.027 | 仅 leaf 列存有明显收益 |
| O2-A | 4 | 4.439 | 20.866 | 25.309 | 正确但无稳定收益 |
| O3-A-P / O3-A-L | 8 / 16 | 3.208 / 4.448 | 20.787 / 15.160 | 23.998 / 19.628 | 两侧均有明显收益 |
| O3-B-P / O3-B-L | 32 / 64 | 4.581 / 4.455 | 20.863 / 27.028 | 25.449 / 31.477 | 32 维宽搬运在此形状变慢 |
| O4-A-P / O4-A-L | 128 / 256 | 4.409 / 4.458 | 20.789 / 20.972 | 25.201 / 25.436 | 分区收益很小，leaf 无收益 |
| O1-L + O3-A-P/L | **26** | **3.215** | **13.705** | **16.924** | 固定算子约 1.50×；推荐本轮组合 |
| O3-A+B-P / O3-A+B-L | 40 / 80 | 3.171 / 4.449 | 20.869 / 18.399 | 24.044 / 22.853 | 分区小幅获益，叶侧劣于 O3-A-L |
| O1-L + O3-A-P/L + O3-B-P | 58 | 3.163 | 13.661 | 16.826 | 固定任务小幅快于相邻 mask 26；额外五次整图构建无收益 |
| O2-B、O4-B/C | — | — | — | — | 后续探索，尚未实现，未计入加速结论 |

完整建图各 5 次交替运行：B0 与 mask 26 的墙钟中位数为 1.60 s 和 1.57 s，persistent kernel event 中位数为 970 ms 和 934 ms。两组均无 fallback/队列溢出。两组独立图在 L=256/512 上的 Recall@10 与 QPS 没有显示稳定差异；固定任务 1.50× 不能直接外推为端到端 1.50×。

## 10. 完成标准

- 每项都有明确适用范围、独立变体和同输入基线。
- 结果一致性、边界和同步检查通过，或明确记录为何失败。
- 性能结论有原始样本，提升超过基线波动；无收益同样保留记录。
- 区分源码推导、编译资源证据、硬件指标与实际耗时，不相互替代。
- 明确是算子收益、阶段收益还是完整构建收益，并记录未验证部分。
- 组合方案另做验证，不把多项同时变化的结果归因给某一项。

本文作为四项 CUDA 优化的执行清单；量化 CTA 线程数适配继续保留在原有 [SHEAR CUDA 优化记录](/home/wbh/rswork/SHEAR/doc/SHEAR_CUDA优化记录.md) 中，属于另一项实验。
