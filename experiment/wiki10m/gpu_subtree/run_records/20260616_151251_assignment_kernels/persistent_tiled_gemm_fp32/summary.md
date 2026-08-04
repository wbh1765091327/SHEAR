# Wiki10M GPU-Subtree Assignment Backend Measurement

- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_151251_assignment_kernels/persistent_tiled_gemm_fp32`
- Assignment backend: `persistent-tiled-gemm-fp32`
- Leaf top-k backend: `persistent-custom`
- Build command changed from runbook: yes, omitted stale `--gpu-subtree-output compact-topk`; added fixed assignment backend; pinned leaf top-k backend
- Search command changed from runbook: no
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`

## Build
- Exit code: `0`
- Wall time: `6:42.26`
- Max RSS kB: `15112572`
- `_mem.index` bytes: `901809540`
- Profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_151251_assignment_kernels/persistent_tiled_gemm_fp32/profile.json`

### GPU Subtree Fields
- `assignment_backend`: `persistent-tiled-gemm-fp32`
- `subtree_assignment_backend`: `persistent-tiled-gemm-fp32`
- `leaf_topk_backend`: `persistent-custom`
- `total_ms`: `278830`
- `persistent_total_ms`: `149627`
- `resident_topk_kernel_event_ms`: `149626`
- `persistent_attempt_count`: `64`
- `persistent_success_count`: `64`
- `persistent_fallback_count`: `0`
- `persistent_queue_overflow_flags`: `0`
- `persistent_partition_worker_blocks`: `128`
- `persistent_topk_worker_blocks`: `128`
- `persistent_partition_pops`: `32481`
- `persistent_partition_claims`: `32481`
- `persistent_topk_pops`: `1284155`
- `persistent_topk_claims`: `1284155`
- `persistent_leaf_pushes`: `1284155`
- `persistent_large_node_splits`: `32481`
- `persistent_large_node_max_rows`: `1378181`
- `persistent_leaf_size_max`: `640`
- `assignment_passes`: `32481`
- `leaf_count`: `996016`
- `compact_topk_rows`: `80000000`

## Search
- Exit code: `0`
- Search L: `100`
- Recall@: `10`
- Search line: `Search: 10000 queries in 3.50s (2854.4 QPS)`
- Recall line: `Recall@10: 0.8402`
- Result bytes: `400008`

## Verdict
- Pass/fail: `pass`
- Compared baseline: assignment-kernel suite in sibling directories
- Generated index cleanup: pending at summary write time
