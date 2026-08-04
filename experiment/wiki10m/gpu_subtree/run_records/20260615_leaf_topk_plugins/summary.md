# Wiki10M Leaf Top-K Backend Five-Operator Measurement

- Generated: 2026-06-15T17:25:25+08:00
- Suite root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins`
- Git commit recorded in new run metadata: `59686f9693a0837d7906879ab60ab7822592ff61`
- Code state: working tree contains the persistent top-k backend implementation under test; run metadata records full `git status --short`.
- GPU recorded in run metadata: NVIDIA GeForce RTX 4090, driver `580.105.08`, CUDA `13.0`.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin` (`10000000 x 1024`)
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin` (`10000 x 1024`)
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin` (`10000 x 100`, max id `9999985`, ids in range)
- Common build parameters: `--partitioner gpu-subtree`, `--pipnn-backend cuda`, `--pipnn-gpu-workspace-budget-gb 16`, `--oom-memory-budget-gb 200`, `--seed 42`, `--leaf-size-min 140`, `--leaf-size-max 640`, `--leaf-knn 2`, `--max-leaders 64`, `--fanout-top 4`, `--fanout-second 2`, `--root-risk-candidate-rs 1,2,4`.
- Variable under test: `--gpu-subtree-leaf-topk-backend`.

## Backend Status

| Backend | Status | Resident persistent? | Library path? | Evidence |
|---|---:|---:|---:|---|
| `persistent-custom` | measured | yes | no | build/search exit `0`; queue overflow `0` |
| `persistent-tiled-gemm-fp32` | measured | yes | no | build/search exit `0`; queue overflow `0` |
| `cuvs-bruteforce` | measured | yes | no | build/search exit `0`; queue overflow `0` |
| `persistent-wmma-tf32` | measured | yes | no | build/search exit `0`; queue overflow `0` |
| `cublas-gemm` | measured | partition-only persistent; top-k library path | yes, cuBLAS GEMM + host scatter | build/search exit `0`; queue overflow `0` |

## Build And Search Results

| Backend | Build wall | GPU total s | Top-k/partition equiv s | Max RSS KiB | `_mem.index` bytes | Search s | QPS | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistent-custom` | 11:37.01 | 573.566 | 447.465 | 15088688 | 902117084 | 4.10 | 2440.2 | 0.8387 |
| `persistent-tiled-gemm-fp32` | 9:22.26 | 438.286 | 312.228 | 15088528 | 902050020 | 4.01 | 2492.7 | 0.8381 |
| `cuvs-bruteforce` | 10:47.91 | 523.927 | 396.867 | 15108036 | 902132400 | 3.79 | 2640.4 | 0.8414 |
| `persistent-wmma-tf32` | 9:27.91 | 441.489 | 314.409 | 15082940 | 902086488 | 4.04 | 2475.6 | 0.8399 |
| `cublas-gemm` | 10:06.06 | 478.010 | 353.222 | 15194504 | 902085648 | 3.84 | 2601.2 | 0.8397 |

## GPU-Subtree Profile

All times are seconds, converted from `profile.json` milliseconds.

| Backend | `persistent_total` | `partition_only` | `library_topk` | `lib_gemm` | `lib_select` | `lib_h2d` | `lib_d2h` | `lib_cpu_scatter` | resident top-k event | leaves | top-k rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistent-custom` | 447.465 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 447.462 | 999092 | 80000000 |
| `persistent-tiled-gemm-fp32` | 312.228 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 312.228 | 997084 | 80000000 |
| `cuvs-bruteforce` | 396.867 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 396.866 | 998622 | 80000000 |
| `persistent-wmma-tf32` | 314.409 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 314.408 | 998906 | 80000000 |
| `cublas-gemm` | 191.359 | 191.359 | 161.863 | 6.462 | 40.811 | 31.944 | 9.962 | 155.400 | 0.000 | 999356 | 80000000 |

## Build-Time Breakdown

| Component | `persistent-custom` | `persistent-tiled-gemm-fp32` | `cuvs-bruteforce` | `persistent-wmma-tf32` | `cublas-gemm` |
|---|---:|---:|---:|---:|---:|
| Build wall | 697.010 | 562.260 | 647.910 | 567.910 | 606.060 |
| GPU-subtree total | 573.566 | 438.286 | 523.927 | 441.489 | 478.010 |
| Persistent kernel / partition stage | 447.465 | 312.228 | 396.867 | 314.409 | 191.359 |
| Partition + top-k equivalent | 447.465 | 312.228 | 396.867 | 314.409 | 353.222 |
| Host load vectors | 92.286 | 91.832 | 92.896 | 93.822 | 95.192 |
| Host norms | 22.541 | 22.755 | 22.641 | 22.776 | 22.590 |
| Subtree vector H2D | 6.618 | 6.620 | 6.612 | 6.612 | 6.623 |
| Vector upload | 6.633 | 6.635 | 6.627 | 6.627 | 6.638 |
| CPU compact flush | 42.072 | 42.464 | 42.304 | 42.049 | 42.784 |
| CPU result scatter | 3.328 | 3.450 | 3.376 | 3.444 | 155.400 |
| Host control total | 118.185 | 118.060 | 118.938 | 120.066 | 273.207 |
| Library top-k total | 0.000 | 0.000 | 0.000 | 0.000 | 161.863 |
| Library CPU scatter | 0.000 | 0.000 | 0.000 | 0.000 | 155.400 |

## Rankings

### Build wall

1. `persistent-tiled-gemm-fp32`: 9:22.26
2. `persistent-wmma-tf32`: 9:27.91
3. `cublas-gemm`: 10:06.06
4. `cuvs-bruteforce`: 10:47.91
5. `persistent-custom`: 11:37.01

### GPU-subtree total

1. `persistent-tiled-gemm-fp32`: 438.286s
2. `persistent-wmma-tf32`: 441.489s
3. `cublas-gemm`: 478.010s
4. `cuvs-bruteforce`: 523.927s
5. `persistent-custom`: 573.566s

### Recall@10

1. `cuvs-bruteforce`: 0.8414
2. `persistent-wmma-tf32`: 0.8399
3. `cublas-gemm`: 0.8397
4. `persistent-custom`: 0.8387
5. `persistent-tiled-gemm-fp32`: 0.8381

### Search QPS

1. `cuvs-bruteforce`: 2640.4
2. `cublas-gemm`: 2601.2
3. `persistent-tiled-gemm-fp32`: 2492.7
4. `persistent-wmma-tf32`: 2475.6
5. `persistent-custom`: 2440.2

## Relative Comparison

| Backend | Wall vs custom | GPU total vs custom | Recall delta vs custom | QPS delta vs custom | Wall vs tiled | GPU total vs tiled | Recall delta vs tiled |
|---|---:|---:|---:|---:|---:|---:|---:|
| `persistent-custom` | +0.00s (+0.00%) | +0.000s (+0.00%) | +0.0000 | +0.0 | +134.75s (+23.97%) | +135.280s (+30.87%) | +0.0006 |
| `persistent-tiled-gemm-fp32` | -134.75s (-19.33%) | -135.280s (-23.59%) | -0.0006 | +52.5 | +0.00s (+0.00%) | +0.000s (+0.00%) | +0.0000 |
| `cuvs-bruteforce` | -49.10s (-7.04%) | -49.639s (-8.65%) | +0.0027 | +200.2 | +85.65s (+15.23%) | +85.641s (+19.54%) | +0.0033 |
| `persistent-wmma-tf32` | -129.10s (-18.52%) | -132.077s (-23.03%) | +0.0012 | +35.4 | +5.65s (+1.00%) | +3.203s (+0.73%) | +0.0018 |
| `cublas-gemm` | -90.95s (-13.05%) | -95.556s (-16.66%) | +0.0010 | +161.0 | +43.80s (+7.79%) | +39.724s (+9.06%) | +0.0016 |

## Operator Analysis

- `persistent-tiled-gemm-fp32` is the fastest build in this five-operator suite: `9:22.26`, GPU-subtree total `438.286s`. It keeps the resident producer-consumer semantics and avoids all library/host scatter work. It is the best current build-time choice.
- `persistent-wmma-tf32` is close to tiled GEMM: `9:27.91`, only `+5.65s` wall and `+3.203s` GPU total versus tiled. Current implementation is a TF32-rounded persistent worker rather than full fragment-level WMMA; its result shows the backend wiring is viable but not yet faster than FP32 tiled GEMM.
- `cuvs-bruteforce` is persistent and library-free, but its direct L2 brute-force scan is compute-heavy: `10:47.91`, slower than tiled by `85.65s`. It has the best Recall@10 in this run (`0.8414`) and best search QPS (`2640.4`), so it may be useful as a quality/reference operator rather than the fastest builder.
- `cublas-gemm` remains faster than `persistent-custom` but slower than tiled/WMMA in end-to-end build. Its raw library GEMM time is small (`6.462s`), but `library_topk_cpu_scatter_ms=155.400s` dominates its host-side overhead and breaks the fully resident top-k worker model.
- `persistent-custom` is now the slowest build (`11:37.01`) and lowest search QPS (`2440.2`) in this suite. It remains the baseline because it is simple and fully resident.

## Pass Checks

- Input headers verified: base `(10000000, 1024)`, query `(10000, 1024)`, GT `(10000, 100)`.
- GT range verified: max id `9999985`, file size `8000008`.
- `persistent-custom`: pass; build/search exit `0`, `_mem.index` bytes `902117084`, OOM budget `214748364800`, queue overflow `0`, partition claim/pop match, top-k claim/pop match, max leaf size `640`, search results bytes `400008`.
- `persistent-tiled-gemm-fp32`: pass; build/search exit `0`, `_mem.index` bytes `902050020`, OOM budget `214748364800`, queue overflow `0`, partition claim/pop match, top-k claim/pop match, max leaf size `640`, search results bytes `400008`.
- `cuvs-bruteforce`: pass; build/search exit `0`, `_mem.index` bytes `902132400`, OOM budget `214748364800`, queue overflow `0`, partition claim/pop match, top-k claim/pop match, max leaf size `640`, search results bytes `400008`.
- `persistent-wmma-tf32`: pass; build/search exit `0`, `_mem.index` bytes `902086488`, OOM budget `214748364800`, queue overflow `0`, partition claim/pop match, top-k claim/pop match, max leaf size `640`, search results bytes `400008`.
- `cublas-gemm`: pass; build/search exit `0`, `_mem.index` bytes `902085648`, OOM budget `214748364800`, queue overflow `0`, partition claim/pop match, top-k claim/pop match, max leaf size `640`, search results bytes `400008`.

## Source Logs

- `persistent-custom`: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_custom`
  - build: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_custom/stdout.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_custom/stderr.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_custom/time.log`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_custom/profile.json`
  - search: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_custom/search.log`
- `persistent-tiled-gemm-fp32`: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_tiled_gemm_fp32`
  - build: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_tiled_gemm_fp32/stdout.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_tiled_gemm_fp32/stderr.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_tiled_gemm_fp32/time.log`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_tiled_gemm_fp32/profile.json`
  - search: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_tiled_gemm_fp32/search.log`
- `cuvs-bruteforce`: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cuvs_bruteforce`
  - build: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cuvs_bruteforce/stdout.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cuvs_bruteforce/stderr.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cuvs_bruteforce/time.log`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cuvs_bruteforce/profile.json`
  - search: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cuvs_bruteforce/search.log`
- `persistent-wmma-tf32`: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_wmma_tf32`
  - build: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_wmma_tf32/stdout.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_wmma_tf32/stderr.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_wmma_tf32/time.log`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_wmma_tf32/profile.json`
  - search: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/persistent_wmma_tf32/search.log`
- `cublas-gemm`: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cublas_gemm`
  - build: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cublas_gemm/stdout.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cublas_gemm/stderr.log` / `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cublas_gemm/time.log`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cublas_gemm/profile.json`
  - search: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_leaf_topk_plugins/cublas_gemm/search.log`
