# Wiki10M WMMA CTA-Shared B Tile Measurement

- Suite root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_201957_wmma_cta_shared_b`
- Code under test: `persistent-wmma-tf32` refactored to CTA-shared B tile, per-warp A row-band pack, cached B metadata, and 64-row fallback to `persistent-tiled-gemm-fp32`.
- Data: Wiki10M base `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`; query `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`; GT `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`.
- Common params: gpu-subtree, CUDA backend, 16 GiB GPU workspace, 200 GiB OOM budget, seed 42, leaf size 140..640, leaf-knn 2, `SEARCH_L=100`, Recall@10.
- Tooling note: `ncu` / `nv-nsight-cu-cli` not found on PATH; `nsys` is present but does not provide the requested `tensor_pipe_active`, shared bank conflict, or register-count counters. No Nsight Compute counter claims are made here.

## Results

| Backend | Build wall | GPU-subtree total | persistent total | resident event | host control | CPU flush | Max RSS KiB | Search s | QPS | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistent-tiled-gemm-fp32` | 9:23.47 | 438.728s | 313.461s | 313.461s | 118.246s | 41.784s | 15081992 | 3.98 | 2515.2 | 0.8386 |
| `persistent-wmma-tf32` | 9:35.82 | 452.137s | 325.663s | 325.661s | 118.091s | 42.128s | 15089172 | 3.89 | 2572.9 | 0.8410 |

## Delta vs Fresh Baseline

- GPU-subtree total: +13.409s (+3.06%)
- persistent_total: +12.202s (+3.89%)
- resident event: +12.200s (+3.89%)
- host_control: -0.155s (-0.13%)
- CPU compact flush: +0.344s (+0.82%)
- Build wall: candidate `9:35.82` vs baseline `9:23.47`.
- Recall@10 delta: +0.0024; QPS delta: +57.7.

## Pass Checks

- `persistent-tiled-gemm-fp32`: build/search exit 0; queue overflow `0`; partition claims/pops `32524/32524`; top-k claims/pops `1287603/1287603`; leaf pushes `1287603`; leaf size max `640`; `_mem.index` bytes `902218148`; search result bytes `400008`.
- `persistent-wmma-tf32`: build/search exit 0; queue overflow `0`; partition claims/pops `32526/32526`; top-k claims/pops `1287352/1287352`; leaf pushes `1287352`; leaf size max `640`; `_mem.index` bytes `902203428`; search result bytes `400008`.

## Interpretation

The CTA-shared B tile refactor improves over the previous true-WMMA candidate but still trails the fresh tiled-FP32 baseline. The gap is almost entirely inside the persistent resident kernel: host control is slightly lower for WMMA, while resident event time is +12.200s. That means the remaining loss is not D2H, search, HashPrune flush, or host-side scatter; it is the device-side leaf worker.

The current WMMA loop now removes the largest obvious redundant B packing by sharing one B tile across the CTA. The remaining costs are still structural: every 16x16 column tile requires CTA barriers for B metadata/B tile reuse, every warp still packs its own A tile for each 8-axis slice, and the accumulator still goes through shared memory before scalar top-k insertion. The 64-row fallback avoids the smallest leaves, but Wiki10M still has many partially filled row/column tiles, so WMMA pays padding/barrier/store overhead on leaf shapes that are not dense GEMM-like enough to amortize Tensor Core throughput.

Compared with the previous true-WMMA run (`20260615_192942_true_wmma_postopt`), resident event improved from 328.466s to 325.661s, so CTA-shared B helped by about 2.805s. The fresh tiled-FP32 resident event is 313.461s, leaving about 12.200s to close.
