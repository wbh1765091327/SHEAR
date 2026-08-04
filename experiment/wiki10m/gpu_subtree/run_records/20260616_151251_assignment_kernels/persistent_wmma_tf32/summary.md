# Wiki10M GPU-Subtree Assignment Backend Measurement

- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_151251_assignment_kernels/persistent_wmma_tf32`
- Assignment backend: `persistent-wmma-tf32`
- Leaf top-k backend: `persistent-custom`
- Build command changed from runbook: yes, omitted stale `--gpu-subtree-output compact-topk`; added fixed assignment backend; pinned leaf top-k backend
- Search command changed from runbook: no
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`

## Build
- Exit code: `0`
- Wall time: `5:57.22`
- Max RSS kB: `15093268`
- `_mem.index` bytes: `901653108`
- Profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_151251_assignment_kernels/persistent_wmma_tf32/profile.json`

### GPU Subtree Fields
- `assignment_backend`: `persistent-wmma-tf32`
- `subtree_assignment_backend`: `persistent-wmma-tf32`
- `leaf_topk_backend`: `persistent-custom`
- `total_ms`: `229877`
- `persistent_total_ms`: `100493`
- `resident_topk_kernel_event_ms`: `100491`
- `persistent_attempt_count`: `64`
- `persistent_success_count`: `64`
- `persistent_fallback_count`: `0`
- `persistent_queue_overflow_flags`: `0`
- `persistent_partition_worker_blocks`: `128`
- `persistent_topk_worker_blocks`: `128`
- `persistent_partition_pops`: `32541`
- `persistent_partition_claims`: `32541`
- `persistent_topk_pops`: `1285844`
- `persistent_topk_claims`: `1285844`
- `persistent_leaf_pushes`: `1285844`
- `persistent_large_node_splits`: `32541`
- `persistent_large_node_max_rows`: `1378181`
- `persistent_leaf_size_max`: `640`
- `assignment_passes`: `32541`
- `leaf_count`: `998352`
- `compact_topk_rows`: `80000000`

## Search
- Exit code: `0`
- Search L: `100`
- Recall@: `10`
- Search line: `Search: 10000 queries in 3.77s (2655.4 QPS)`
- Recall line: `Recall@10: 0.8371`
- Result bytes: `400008`

## Verdict
- Pass/fail: `pass`
- Compared baseline: assignment-kernel suite in sibling directories
- Generated index cleanup: pending at summary write time
