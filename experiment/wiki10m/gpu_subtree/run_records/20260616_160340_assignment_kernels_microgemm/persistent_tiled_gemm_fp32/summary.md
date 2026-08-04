# Wiki10M Assignment Backend Measurement

- Backend: `persistent-tiled-gemm-fp32`
- Leaf top-k backend: `persistent-custom`
- Build command changed from runbook: yes, omitted stale `--gpu-subtree-output compact-topk`, added fixed assignment backend, pinned leaf top-k backend.
- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_160340_assignment_kernels_microgemm/persistent_tiled_gemm_fp32`

## Build

- Exit code: 0
- Wall time: `6:45.27`
- Max RSS kB: `15088252`
- `_mem.index` bytes before cleanup: `901515056`
- OOM memory budget bytes: `214748364800`
- Partitioner: `gpu-subtree`

## GPU Subtree

- assignment_backend: `persistent-tiled-gemm-fp32`
- subtree_assignment_backend: `persistent-tiled-gemm-fp32`
- leaf_topk_backend: `persistent-custom`
- total_ms: `273697`
- persistent_total_ms: `141106`
- resident_topk_kernel_event_ms: `141104`
- persistent_attempt/success/fallback: `64/64/0`
- queue_overflow_flags: `0`
- partition claims/pops: `32626/32626`
- topk claims/pops: `1286811/1286811`
- leaf_count: `999509`
- compact_topk_rows: `80000000`
- fanout_assignment_slots: `191611166`
- replicated_rows: `191611166`
- unique_resident_rows: `40000000`
- large_node_splits/max_rows: `32626/1378181`
- leaf_size_max: `640`

## Search

- Exit code: 0
- Search L: 100
- Recall@: 10
- Search wall: `Search: 10000 queries in 3.77s (2654.5 QPS)`
- QPS: `Search: 10000 queries in 3.77s (2654.5 QPS)`
- Recall: `Recall@10: 0.8437`
- search_results bytes: `400008`

## Verdict

- Pass/fail: pass
- Generated index cleanup: pending in metadata
