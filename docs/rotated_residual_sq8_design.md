# 旋转后聚类残差 SQ8 设计与实现

## 目标与范围

为 float L2 的 GPU subtree 建图增加可切换的 SQ8 实验模式：使用一套共享的正交变换旋转数据和粗聚类中心；每条向量相对其唯一归属的粗中心取残差，再用现有逐向量 signed int8 标量量化。默认 SQ8 路径保持现状，以相同构图参数比较邻居选择质量、端到端耗时与峰值内存。

这里的“聚类中心”指编码前已经训练好的粗中心。当前 `gpu-kmeans-coarse` 会在 D0 SQ8 编码前训练中心；每条数据的单一编码中心 ID 仍须在编码前确定，不能直接使用随后产生的 top-k D0 分配结果。D0 top-k 可以包含多个中心，但每条向量只存一份相对其编码中心的 code。

## 编码与距离

令原向量为 `x_i`，归属中心为 `c_{a_i}`，共享正交变换为 `R`：

```text
z_i = R x_i
m_k = R c_k
r_i = z_i - m_{a_i} = R (x_i - c_{a_i})
q_i = SQ8(r_i)                 // scale_i * signed-int8 code_i
x̂_i (在旋转空间) = m_{a_i} + q_i
```

旋转只用于编码和近似距离；精确复核仍可读原始向量。`R` 必须对所有向量及中心一致，保存其类型、维度和 seed。首版宜使用已有的可复现快速正交变换实现，并限制其支持的维度；任意稠密随机矩阵会带来 `O(N d²)` 的预处理成本。

两条量化向量的**重建距离**为：

```text
δ = m_{a_i} - m_{a_j}
d̂(i,j) = ||δ||² + ||q_i-q_j||² + 2 δ·(q_i-q_j)
```

同一中心时 `δ=0`，只用残差范数与整数 code 点积即可。不同中心时必须加入中心差和交叉项；只比较残差 code 会得到错误距离。也可保留原向量的精确范数，使用当前内核风格的点积估计：

```text
d̃(i,j) = ||x_i||² + ||x_j||²
          - 2 [m_{a_i}·m_{a_j} + m_{a_i}·q_j
               + m_{a_j}·q_i + q_i·q_j]
```

这两个估计器的误差特性不同，不可混用范数定义。首版应固定一种估计器，并在 CPU 参考实现与每条 GPU 距离路径中使用相同公式。对跨中心项可先直接计算建立正确性基线，再评估预计算 `中心×中心` 和 `向量×中心` 点积表；后者需额外 `4 N C` 字节（FP32），例如 `N=10M, C=16` 约 640 MB，不宜默认常驻。

## 现有代码接入点

1. **中心和归属。** `rbc_partition.rs` 的 `gpu-kmeans-coarse` 已在 SQ8 D0 编码前训练中心。新增最近中心归属步骤，明确编码中心 ID 与 D0 top-k 结果的区别；先支持该模式，其他 D0 模式按条件拒绝或单独提供中心训练。
2. **编码。** 旋转全体数据与中心在数学上等价于先计算 `x_i-c_{a_i}` 再旋转；归属中心也可在原坐标求得。当前实现用主机 FHT-Kac 变换与 SQ8 量化，一条原始向量只编码一次。输出为对齐的 signed-int8 code、逐向量 scale 和中心 ID。全 GPU 编码仍是后续性能优化方向。
3. **距离。** D0 使用 GPU 计算重建向量与粗中心的距离。D1+ 从 sidecar 收集所需 code、scale 与中心 ID，在主机重建 `m_{a_i}+q_i`，随后交给现有 GPU FP32 分区与叶块距离内核。这样得到与上式相同的跨中心距离，但当前 D1+ 不使用 INT8 WMMA 点积；直接在 GPU kernel 中叠加中心项仍是后续优化方向。
4. **sidecar。** 新格式有独立的 `PGSRR2` magic 与版本 2，保存 `.codes`、`.meta`、`.centers`；每行 meta 为中心 ID、scale、残差范数，共 12 字节。头部记录行数、维度、seed 与中心表哈希，并在读取时校验。旧 SQ8 sidecar 保持原格式，两种格式不会互读。
5. **参数。** 新模式需显式开关，默认关闭；校验维度、float L2、`gpu-kmeans-coarse`、中心数及与现有 RabitQ / TurboQuant 模式的互斥关系。

## 验证顺序

先在小数据上用 CPU FP32 重建距离校验 GPU D0、D1+ 和叶块逐对距离及 top-k；覆盖同中心、跨中心、零残差和旋转维度边界。然后做四组消融：原 SQ8、仅旋转、仅残差、旋转加残差。比较量化距离误差、pairwise 排序翻转率、最终 recall、建图总时长、各 kernel 时长、sidecar 大小及 GPU 峰值内存。任何加速结论都以端到端建图时间为准。

## 预期与主要风险

残差范围更小或旋转削弱坐标极值时，SQ8 步长可能降低，邻居排序可能改善；收益依赖数据分布。中心归属、旋转和跨中心修正会增加工作，不能预设建图提速。特别要检查同一叶块中的向量是否使用不同编码中心，以及跨中心修正的数据读取是否抵消 WMMA 整数点积的优势。

## 当前实现阶段

实验入口是 `--gpu-subtree-sq8-rotated-residual-d0`，配合 `--gpu-subtree-sq-staging-mode gpu-d0 --gpu-subtree-d0-partition-mode gpu-kmeans-coarse` 使用；`--gpu-subtree-sq8-rotated-residual-seed` 控制共享 FHT-Kac 旋转。默认关闭，原 SQ8 不变。D0 使用最近粗中心编码残差，GPU 按重建向量到各粗中心的 L2 距离求 top-k。

D1+ 有两个互斥选项：`--gpu-subtree-sq-staging-raw-d1-diagnostic` 继续使用原始向量；或使用 `--gpu-subtree-sq8-rotated-residual-d1 --gpu-subtree-sq-wmma-mode build`，在 D0 生成一份供所有子分区复用的全局旋转残差 SQ8 sidecar。即使同一向量进入多个 D0 子分区，也只编码和存储一次。D1+ 不再读取原向量或按子分区重新量化。编码、中心归属和旋转仍在主机执行；D1+ 距离排序在 GPU 上执行。

sidecar 中的 `norm` 是量化残差平方范数。D0 比较重建向量与粗中心时会加编码中心；D1+ 在主机重建每个点的 `m_{a_i}+q_i`，重算其范数后，GPU 按重建向量的 FP32 L2 距离比较。子分区内可以混有不同编码中心，因此不能省去中心项。当前实现选择重建距离估计器，不使用原始向量的精确范数估计器。

完整实验模式的 CLI 组合：

```text
--gpu-subtree-sq-staging-mode gpu-d0
--gpu-subtree-d0-partition-mode gpu-kmeans-coarse
--gpu-subtree-sq-wmma-mode build
--gpu-subtree-sq8-rotated-residual-d0
--gpu-subtree-sq8-rotated-residual-d1
--gpu-subtree-sq8-rotated-residual-seed 42
```

当前限制：float L2、OOM GPU subtree、SQ8、2 的幂维度；CUDA D0 最多 256 个粗中心、top fanout 最多 32。sidecar 默认写在索引目录下的 `rotated_residual_sq8.{codes,meta,centers}`，每次建图重新生成；当前没有跨运行复用入口。原模式默认关闭这些开关，仍走原 sidecar 与距离核。

## D1+ GPU INT8 中心感知模式

在上述完整实验模式的参数上增加 `--gpu-subtree-sq8-rotated-residual-d1-int8`，即可选择独立的 GPU INT8 距离路径。只使用 `--gpu-subtree-sq8-rotated-residual-d1` 时仍走原有的主机 FP32 重建路径；两者共用 v2 sidecar，原 SQ8 模式也保持不变。

新模式的 D1+ 只收集每点的残差 INT8 code、scale、残差范数和中心 ID。GPU 上传这些数据及旋转中心后，按子树预计算 `P[i,k] = (scale_i × code_i)·m_k`、`G[a,b] = m_a·m_b`，并由残差范数得到重建向量范数 `N_i`。分区和叶块继续用 INT8 点积得到 `code_i·code_j`，最终比较

```text
max(0, N_i + N_j - 2 × [scale_i scale_j (code_i·code_j)
                      + P[i,a_j] + P[j,a_i] + G[a_i,a_j]])
```

因此中心不同的点也按完整重建距离比较，不上传逐点 FP32 重建向量。当前直接模式固定使用 warp DP4A 分区与叶块执行器，以及 fused persistent / staged leaf 路径；其它 SQ8 执行器仍供原模式使用。投影表为每个 D1 子树临时分配，额外 GPU 空间约 `4 × 子树行数 × 粗中心数` 字节；中心多或 D0 fanout 高时，投影预计算会重复，需结合端到端测量评估收益。D0 主机旋转量化保持现状。
