# Wiki10M GPU-Subtree Measurement

- Run ID: 20260616_225913_d0_cuda_root_wmma_gemmfix
- Date: 2026-06-16
- Git commit: 3ff96972031a1396535e6174ae180eadf2b9e0c0
- Code changes under test: D0/root assignment backend routed through CUDA cuBLAS assignment backend; D1 RBC assignment and leaf top-k remain persistent-wmma-tf32; D1 child error diagnostics added.
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: yes, removed unsupported legacy `--gpu-subtree-output compact-topk` flag. The current `build_pipnn_index` CLI rejects it and gpu-subtree uses compact top-k implicitly.
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 4:44.30 (284.30s)
- Max RSS: 15310984 KB
- `_mem.index` bytes: 901823264
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Root assignment backend: cuda
- D1 RBC assignment backend: persistent-wmma-tf32 / subtree profile persistent-wmma-tf32
- D1 leaf build/top-k backend: persistent-wmma-tf32
- D0 wall: 33.612s (33612 ms)
- D0 child runs: 64, points with root fanout replication: 40000000, max/p95/p50: 1378182 / 1221231 / 543772
- D1+ backend wall: 197.931s (197931 ms)
- D1+ producer wall: 203.529s (203529 ms)
- Async flush wall: 45.851s (45851 ms)
- Persistent total/kernel: 55.714s (55714 ms) / 55.713s (55713 ms)
- Persistent success count: 64
- Persistent queue overflow flags: 0
- Persistent partition claims/pops: 32437 / 32437
- Persistent top-k claims/pops: 1284201 / 1284201
- Persistent large-node splits/max rows: 32437 / 1378182
- Persistent leaf size max: 640

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search wall time: 3.82s
- QPS: 2620.7
- Recall: 0.8367
- Result file: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_225913_d0_cuda_root_wmma_gemmfix/search_results.fbin (400008 bytes)

## Comparison Against 20260616_185933_staged_async_wmma

| Metric | Current | Baseline | Delta |
| --- | ---: | ---: | ---: |
| Build wall | 284.30s | 271.25s | +13.05s (+4.8%) |
| D0 wall | 33612 ms | 20884 ms | +12728 ms (+60.9%) |
| D1+ backend | 197931 ms | 197699 ms | +232 ms (+0.1%) |
| D1+ producer | 203529 ms | 203273 ms | +256 ms (+0.1%) |
| Async flush | 45851 ms | 46114 ms | -263 ms (-0.6%) |
| Persistent total | 55714 ms | 55831 ms | -117 ms (-0.2%) |
| Top-k kernel | 55713 ms | 55830 ms | -117 ms (-0.2%) |
| Search QPS | 2620.7 | 2548.5 | +72.2 (+2.8%) |
| Recall@10 | 0.8367 | 0.8406 | -0.0039 |

## D0 GPU Assignment Analysis

This run suggests that we should pause further optimization of the current D0 CUDA/cuBLAS root-assignment path. The root-assignment backend changed from CPU to CUDA, but D0 wall time increased from 20.884s to 33.612s (+12.728s, +60.9%), and the end-to-end build wall time increased from 271.25s to 284.30s (+13.05s, +4.8%). In contrast, D1+ backend time, D1+ producer time, persistent-WMMA total time, top-k kernel time, and async flush time are effectively unchanged or slightly faster. The D0 child-run distribution is also nearly identical to the CPU-baseline run: 64 child runs, 40,000,000 root-fanout assignments, and max/p95/p50 child sizes of 1,378,182 / 1,221,231 / 543,772 versus 1,378,181 / 1,221,235 / 543,772 in the baseline. Therefore, the regression is not explained by a different partition shape; it is isolated to the current D0 assignment execution path.

从 D2T, 即 data-to-transfer/device-transfer ratio 的角度看，当前 D0 GPU 化并不理想。Wiki10M 的 base vector payload 约为 `10M * 1024 * sizeof(f32) = 40.96 GB`，D0 CUDA 路径除了 dense distance/GEMM 计算外，还需要支付 point-store 读取、host 侧 packing/layout conversion、H2D transfer、GPU kernel execution、D2H transfer、stream synchronization，以及后续 OOM child-run materialization 的成本。GPU 擅长大批量、规则、高算术强度并且能够融合输出处理的 workload；但是当前 D0 assignment 的输出很小且格式敏感，每个点最终只是 fanout=4 的 root leader 记录，后续还必须转换成外部 spool/run-store 消费的 assignment records。也就是说，GPU 加速覆盖的是距离计算片段，而不是完整的 D0 数据通路。

The current implementation makes this boundary explicit. D0 calls `assign_clusters_l2(...)`, receives host-side `Vec<Vec<u32>>` clusters, validates identity-ordered root point IDs, builds `per_point_leaders`, writes assignment records, and flushes them into the D0 spool/child-run representation. This preserves correctness and lets the existing OOM pipeline consume the result, but it leaves host memory traffic, allocation/reordering, and external-run materialization on the critical path. Consequently, CUDA/cuBLAS can execute the dense scoring step efficiently while the full D0 stage still loses time to transfer and format conversion. The measured result is consistent with this mechanism: GPU arithmetic throughput is not enough when the operator does not directly emit the format required by the downstream pipeline.

This differs from the D1 persistent-WMMA pipeline. D1 RBC assignment and leaf build/top-k both use `persistent-wmma-tf32`, keeping subtree partitioning and top-k generation resident inside the device pipeline and returning compact top-k batches to the CPU flush stage. That path amortizes data movement over more GPU-side work and avoids repeatedly converting dense GPU outputs into OOM partition records. In this run, D1+ backend time changes by only +232 ms (+0.1%), persistent total time improves by 117 ms (-0.2%), top-k kernel time improves by 117 ms (-0.2%), and async flush improves by 263 ms (-0.6%). These numbers support keeping the D1+ persistent-WMMA path as the GPU optimization focus while treating the current D0 CUDA/cuBLAS assignment as a measured regression.

The practical conclusion is to stop optimizing this D0 GPU implementation for now. If D0 is revisited, the target should not be incremental tuning around the current cuBLAS-return-clusters path. A better design would use a fused or persistent root-assignment operator that performs scoring, top-k leader selection, and assignment-record or child-run emission in one chunked pipeline, with overlap among H2D transfer, device assignment, D2H output, and CPU materialization. Until that design exists, CPU D0 remains the better default boundary for the staged async GPU-subtree builder, while D1 RBC and leaf build should remain on `persistent-wmma-tf32`.

## Verdict

- Pass/fail: pass for build, profile sanity checks, and mem-search execution; performance regression for D0/root assignment wall time.
- Compared baseline: 20260616_185933_staged_async_wmma, same Wiki10M base/query/GT, `SEARCH_L=100`, `RECALL_AT=10`, 16 GiB GPU workspace.
- Regression or improvement: D0 CUDA cuBLAS root assignment is slower than CPU root assignment on this run (+12.728s, +60.9%), and dominates the build wall regression (+13.05s, +4.8%). D1 persistent-WMMA and async flush are effectively unchanged/slightly faster. Search QPS is higher (+2.8%), but Recall@10 is lower by 0.0039 and should be treated as small stochastic/index-quality regression until repeated.
- Decision: pause further optimization of the current D0 CUDA/cuBLAS assignment path. The next D0 attempt should be a fused/persistent assignment-record or child-run emission design rather than incremental tuning of the present host cluster conversion path.
- Generated index cleanup: cleaned after this summary and search result were written.
- Notes: The failed diagnostic run `20260616_225731_d0_cuda_root_wmma_gemmfix_diag` exited at CLI parsing because the current driver rejects legacy `--gpu-subtree-output`; no index artifact was produced there. This successful run kept D1 RBC assignment and leaf top-k on `persistent-wmma-tf32` as requested.
