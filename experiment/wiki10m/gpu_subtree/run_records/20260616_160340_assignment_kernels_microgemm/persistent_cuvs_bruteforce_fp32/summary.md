# Wiki10M Assignment Backend Measurement

- Backend: `persistent-cuvs-bruteforce-fp32`
- Leaf top-k backend: `persistent-custom`
- Build command changed from runbook: yes, omitted stale `--gpu-subtree-output compact-topk`, added fixed assignment backend, pinned leaf top-k backend.
- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_160340_assignment_kernels_microgemm/persistent_cuvs_bruteforce_fp32`

## Build

- Exit code: 0
- Wall time: `10:28.64`
- Max RSS kB: `15087760`
- `_mem.index` bytes before cleanup: `901792860`
- OOM memory budget bytes: `214748364800`
- Partitioner: `gpu-subtree`

## GPU Subtree

- assignment_backend: `persistent-cuvs-bruteforce-fp32`
- subtree_assignment_backend: `persistent-cuvs-bruteforce-fp32`
- leaf_topk_backend: `persistent-custom`
- total_ms: `497639`
- persistent_total_ms: `364396`
- resident_topk_kernel_event_ms: `364396`
- persistent_attempt/success/fallback: `64/64/0`
- queue_overflow_flags: `0`
- partition claims/pops: `32565/32565`
- topk claims/pops: `1286179/1286179`
- leaf_count: `998496`
- compact_topk_rows: `80000000`
- fanout_assignment_slots: `191578530`
- replicated_rows: `191578530`
- unique_resident_rows: `40000000`
- large_node_splits/max_rows: `32565/1378181`
- leaf_size_max: `640`

## Search

- Exit code: 0
- Search L: 100
- Recall@: 10
- Search wall: `Search: 10000 queries in 4.14s (2413.0 QPS)`
- QPS: `Search: 10000 queries in 4.14s (2413.0 QPS)`
- Recall: `Recall@10: 0.8396`
- search_results bytes: `400008`

## Verdict

- Pass/fail: pass
- Generated index cleanup: pending in metadata
