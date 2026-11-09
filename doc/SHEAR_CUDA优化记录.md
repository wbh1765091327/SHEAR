# SHEAR CUDA 优化记录

记录日期：2026-11-03。

本文记录源码阅读中发现的优化候选，区分当前实现、拟议方案与验证要求。条目顺序表示讨论顺序，不代表已经测得的瓶颈或优化优先级。

## 1. SQ8 量化 CTA 线程数按向量维度适配

**状态：待实现、待验证。** 本条仅记录方案，尚未修改 CUDA 实现或获得新的性能测量结果。

### 1.1 概要

当前 GPU-D0 的 SQ8 量化使用一个 CTA（即一个 CUDA block）处理一条向量，每个 CTA 固定使用 256 个线程。数据批次包含 N 条向量时，量化 grid 包含 N 个 CTA，由 GPU 按资源情况调度执行。

目标场景是大规模数据集，向量维度主要在 96～1024 之间。本项聚焦低维输入时 CTA 内的有效工作分配，以及归约、同步和资源占用；以整批量化时间和 D0 阶段时间判断优化效果。

拟议方向：根据维度选择量化 CTA 的线程数，先对 64、128、256 线程进行对照实验。候选线程数不需要整除向量维度，也不需要整除整张 GPU 的总线程容量。

### 1.2 当前实现与关键代码

源码位于 `shear-core/src/algorithm/pipnn/compute_backend.rs`。

- 设备函数：`sq8_quantize_rows_kernel`，核对时起始行为 18110。
- 主机配置：`quantize_sq8_and_assign_global_d0` 内的 `cfg_quantize`，核对时位于 24784 附近。
- [打开源码](../shear-core/src/algorithm/pipnn/compute_backend.rs)。行号会随文件变化，优先按符号名定位。

下面摘录原始语句并补充中文注释，省略无关配置和函数上下文。

```rust
// 概要：当前固定一个 CTA 处理一条向量，每 CTA 256 线程。
let block_threads = 256u32;

// 两个 f32 共享数组：每线程的最大绝对值、每线程的平方和。
// 256 线程时，动态共享内存为 2 × 256 × 4 = 2048 字节。
let quantize_shared = (2usize)
    .saturating_mul(block_threads as usize)
    .saturating_mul(std::mem::size_of::<f32>())
    as u32;

// ……省略其他执行阶段的共享内存配置。

let cfg_quantize = LaunchConfig {
    grid_dim: (rows_u32, 1, 1),        // 本批 N 条向量对应 N 个 CTA。
    block_dim: (block_threads, 1, 1), // 每个 CTA 的线程数。
    shared_mem_bytes: quantize_shared,
};
```

```cpp
// 概要：CTA 内的线程按固定步长覆盖一条向量的所有维度。
unsigned long long row = (unsigned long long)blockIdx.x;
// ……省略 row 越界检查。
unsigned int tid = threadIdx.x;

// ……省略共享内存指针和局部变量初始化。

for (unsigned int axis = tid; axis < dim; axis += blockDim.x) {
    // 同一轮相邻线程读取相邻维度；下一轮跨 blockDim.x 个维度。
    float value = raw_vectors[row * (unsigned long long)dim + axis];
    local_max_abs = fmaxf(local_max_abs, fabsf(value));
    local_norm += value * value;
}

max_abs_shared[tid] = local_max_abs;
norm_shared[tid] = local_norm;
__syncthreads();

// 概要：不断折半，得到整条向量的最大绝对值和平方范数。
// 这份实现要求 blockDim.x 为 2 的幂，才能完整合并所有线程结果。
for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
        max_abs_shared[tid] = fmaxf(max_abs_shared[tid], max_abs_shared[tid + stride]);
        norm_shared[tid] += norm_shared[tid + stride];
    }
    // 每轮的下一步依赖本轮结果；CTA 内线程共同到达屏障。
    __syncthreads();
}
```

量化输出还有第二个跨步循环，遍历 `padded_dim` 并生成 INT8 codes。超过真实 `dim` 的位置填零。调整线程数时，需要同时考虑原始维度读取与编码输出两个循环。

### 1.3 潜在开销

固定使用 256 线程时，原始维度读取阶段的分工如下：

| 向量维度 | 读取循环的分工 | 观察 |
|---:|---|---|
| 96 | 前 96 个线程各一次，其余 160 个线程零次 | 有较多线程没有原始维度可读 |
| 128 | 前 128 个线程各一次，其余 128 个线程零次 | 可以测试较小 CTA |
| 158 | 前 158 个线程各一次，其余 98 个线程零次 | 可以测试 128 线程跨步读取 |
| 256 | 全部线程各一次 | 维度读取分工均衡 |
| 512 | 全部线程各两次 | 维度读取分工均衡 |
| 1024 | 全部线程各四次 | 维度读取分工均衡 |

“读取循环零次”不表示该线程在整个 kernel 中完全空闲：这些线程仍写入局部统计值，参与归约、同步，也可能处理输出的填充位置。因此不能把读取阶段的有效线程比例直接当作整个 kernel 的性能比例。

缩小 CTA 可能减少归约与同步工作、降低每 CTA 共享内存开销，但也改变驻留 CTA 数、warp 数和每线程循环量。是否更快，需要整体测量。

### 1.4 线程数选择的约束

1. **硬件组织：优先使用 32 的倍数。** 一个 warp 包含 32 个线程。78 或 79 线程的 block 在硬件上均需要三个 warp，最后一个 warp 有未使用的 lane。
2. **当前算法：使用 2 的幂。** 当前折半归约直接用于 96、78、79 等线程数会遗漏部分统计量，影响最大绝对值、范数和量化结果。保持归约代码时，先使用 32、64、128、256 等值。
3. **无需整除维度。** `axis += blockDim.x` 已支持跨步覆盖所有维度。维度不能整除线程数时，部分线程多执行一轮。
4. **无需整除所有 SM 的总线程容量。** CTA 整体驻留在单个 SM；每个 SM 根据线程、寄存器、共享内存及其他限制容纳 CTA。整张 GPU 的线程容量不是选择 block 大小的整除依据。

例如，158 维使用 128 线程时：

```text
thread 0～29：各处理两维，例如 thread 0 处理 0、128。
thread 30～127：各处理一维，例如 thread 30 处理 30。

覆盖维度数 = 30 × 2 + 98 × 1 = 158。
```

158 的一半是 79，但采用 79 线程不会自动带来更好的硬件效率，并且与当前归约实现不兼容。

### 1.5 第一阶段方案：保留归约，调整量化启动配置

保持“一个 CTA 处理一条向量”的结构，在量化启动前按维度选择 64、128 或 256 线程。先建立以下测量矩阵，再决定维度分桶策略：

| 维度 | 候选线程数 | 对照基线 |
|---:|---|---|
| 96、128 | 64、128、256 | 原始 256 线程 |
| 158、160、256 | 64、128、256 | 原始 256 线程 |
| 512、768、1024 | 128、256 | 原始 256 线程 |

实施时需要联动检查：

- `cfg_quantize.block_dim` 与动态共享内存 `2 × block_threads × sizeof(float)`。
- 普通数据行与 leader 量化所使用的启动配置及共享内存参数。
- 当前 `block_threads`、`assign_shared` 等变量是否被 scalar assignment 路径复用。量化线程数适配应使用清晰的独立配置，避免无意改变其他阶段。
- 原始 `dim`、`padded_dim` 与 codes 行跨度保持一致。
- 没有原始维度可读的线程仍应提供正确的归约初值并参与必要屏障，不应在屏障前随意提前退出。

本方案不预先认定某个维度应使用唯一线程数，也不要求让每个线程的循环次数完全相同。

### 1.6 后续候选：改造归约或一个 CTA 处理多条向量

若第一阶段收益不足，可单独评估：

- 用 warp 内归约加跨 warp 合并，支持 96、160 等为 32 倍数、但不是 2 的幂的 CTA 大小。
- 对低维向量采用 warp-per-vector，一个 CTA 包含多个 warp，各 warp 分别处理不同向量。

这些方案会改变归约方式、索引布局、同步与任务映射，需独立验证。不能仅修改 `block_dim` 就认为已经完成这些优化。

### 1.7 验证方法与通过条件

**正确性：**

- 覆盖 96、128、158、160、256、512、768、1024 维，以及零向量、重复向量和不同数值范围。
- 检查 INT8 codes、scale、补零位置及输出边界；在相同有限输入和相同量化公式下，核对编码与 scale 是否保持一致。
- 修改线程数会改变浮点求和的分组顺序，平方范数可能出现低位差异。记录范数绝对/相对误差，并检查后续 D0 归属变化，尤其是距离接近或相同的输入。
- 对完整构图检查候选与图结构，并在固定搜索配置下比较召回率。局部量化测试不能替代端到端质量验证。

**性能：**

- 使用足够大的相同数据批次、相同 GPU、相同输入与缓存条件进行对照；另以小批次检查边界正确性。
- 预热 NVRTC 编译、模块加载和必要初始化，单独使用 CUDA event 测量量化 kernel 时间。
- 分开记录 H2D、量化、D0 assignment、D2H，以及整个 D0 阶段耗时，避免混淆提交时间和 GPU 执行时间。
- 比较向量吞吐量 `本批向量数 / 量化 kernel 时间`，使用多次测量判断收益是否稳定。
- 必要时检查寄存器、共享内存、驻留 warp、有效 lane、访存和屏障等待，解释耗时变化。

**通过条件：** 在正确性和构图质量符合要求的前提下，目标维度的量化吞吐有稳定提升，并确认整体 D0 或构图时间是否受益。不能仅凭线程参与比例提高或 occupancy 增加宣称性能优化成功。

### 1.8 参考

- [CUDA Programming Model：Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)
- [CUDA Best Practices Guide：Thread and Block Heuristics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics)
