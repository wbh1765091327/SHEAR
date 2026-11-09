# FP32 upper-triangular leaf tile experiment on V100

This standalone experiment checks whether upper-triangular **tile scheduling**
preserves all non-self neighbors and reduces the cost of leaf-local top-k. It
does not run the full SHEAR builder or use Tensor Core instructions. The V100
(SM70) computes FP32 distances on CUDA cores; each tile is 16 × 16.

Run from the repository root with a CUDA toolkit on `PATH`. On the reviewed
host, `nvcc` is at
`/data00/home/zhaoweichen.db/repo/Gpu-Index/build-cuda-toolkit/cuda/bin/nvcc`:

```bash
nvcc -O3 -std=c++17 -arch=sm_70 script/bench_upper_triangle_fp32.cu -o /tmp/bench_upper_triangle_fp32
/tmp/bench_upper_triangle_fp32 128 64 96 15
```

Arguments are `rows_per_leaf leaves dimension iterations`. Input vectors are
deterministic random FP32 values (seed 42), with top-k = 8. The benchmark warms
each path three times, alternates the execution order, and reports median CUDA
event times. Allocation, input transfer, and CPU validation are outside the
timed region.

The full path evaluates every tile and selects candidates for its 16 source
rows. The upper path evaluates each off-diagonal tile once and selects
candidates for both its rows and columns. Both evaluate every diagonal tile in
full, mask self-pairs, and select candidates for its rows only. In a complete
16 × 16 diagonal tile, all 240 non-self entries remain available. The direct
top-k variant writes tile-local top-k lists and merges them by source row; it
does not materialize the full distance matrix. A second variant materializes a
dense matrix to verify elementwise symmetry and provide a simpler comparison.

All cases passed: the first leaf's dense distance matrices matched bitwise, full and upper
top-k IDs matched across every leaf, the direct and dense top-k IDs matched,
and the first leaf's top-k matched a double-precision CPU reference. No
self-neighbor appeared. The 17- and 130-row cases also tested partial tiles.

| Dimension | Rows/leaf | Leaves | Full direct top-k (ms) | Upper direct top-k (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 16 | 128 | 0.0430 | 0.0430 | 1.00× |
| 96 | 17 | 8 | 0.0317 | 0.0328 | 0.97× |
| 96 | 32 | 128 | 0.0973 | 0.0768 | 1.27× |
| 96 | 48 | 128 | 0.1853 | 0.1382 | 1.34× |
| 96 | 64 | 128 | 0.3113 | 0.2202 | 1.41× |
| 96 | 128 | 64 | 0.6646 | 0.4198 | 1.58× |
| 96 | 130 | 32 | 0.3768 | 0.2458 | 1.53× |
| 96 | 256 | 32 | 1.2892 | 0.7598 | 1.70× |
| 96 | 512 | 16 | 2.1371 | 1.2175 | 1.76× |
| 96 | 1280 | 4 | 4.0172 | 2.2753 | 1.77× |
| 1024 | 128 | 32 | 2.7587 | 1.5985 | 1.73× |
| 1024 | 512 | 4 | 5.2296 | 2.8150 | 1.86× |

These are GPU microbenchmark timings, not end-to-end construction gains. The
current production leaf operator uses different scheduling and its persistent
TF32 path requires SM80+. Integrating the same tile traversal there requires
its own implementation, top-k correctness test, and build-level measurement.

## 760-dimensional comparison

The same FP32 kernel was run at exactly 760 dimensions, with no K padding.
The table reports the direct tile-to-top-k path, including its row-wise merge.
For example: `/tmp/bench_upper_triangle_fp32 128 64 760 15`.

| Rows/leaf | Leaves | Full (ms) | Upper (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | 0.1311 | 0.1311 | 1.00× |
| 64 | 128 | 1.5421 | 0.9912 | 1.56× |
| 128 | 64 | 2.5252 | 1.4930 | 1.69× |
| 256 | 32 | 4.6367 | 2.5917 | 1.79× |
| 512 | 16 | 9.1515 | 4.8589 | 1.88× |
| 1280 | 4 | 14.2797 | 7.4721 | 1.91× |

Full and upper GPU top-k IDs matched exactly across all leaves in every case.
The first leaf's dense distance matrices also matched bitwise. Against the
FP64 CPU audit, one rank at 512 rows and three ranks at 1280 rows differed
because of near-tied distances; the largest true-distance difference was
0.000401. This is a floating-point ordering effect shared by both GPU paths.
The timings do not model `m16n8k64`: that shape would need to handle a final
56-element K slice (or pad eight values to 768), and V100 cannot execute its
native INT4 Tensor Core operation.
