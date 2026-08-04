# Wiki10M Assignment Backend Measurement

- Backend: `persistent-wmma-tf32`
- Leaf top-k backend: `persistent-custom`
- Build command changed from runbook: yes, omitted stale `--gpu-subtree-output compact-topk`, added fixed assignment backend, pinned leaf top-k backend.
- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_160340_assignment_kernels_microgemm/persistent_wmma_tf32`

## Build

- Exit code: 0
- Wall time: `6:03.78`
- Max RSS kB: `15098072`
- `_mem.index` bytes before cleanup: `901859948`
- OOM memory budget bytes: `214748364800`
- Partitioner: `gpu-subtree`

## GPU Subtree

- assignment_backend: `persistent-wmma-tf32`
- subtree_assignment_backend: `persistent-wmma-tf32`
- leaf_topk_backend: `persistent-custom`
- total_ms: `233526`
- persistent_total_ms: `100400`
- resident_topk_kernel_event_ms: `100399`
- persistent_attempt/success/fallback: `64/64/0`
- queue_overflow_flags: `0`
- partition claims/pops: `32595/32595`
- topk claims/pops: `1287059/1287059`
- leaf_count: `999990`
- compact_topk_rows: `80000000`
- fanout_assignment_slots: `191607976`
- replicated_rows: `191607976`
- unique_resident_rows: `40000000`
- large_node_splits/max_rows: `32595/1378181`
- leaf_size_max: `640`

## Search

- Exit code: 0
- Search L: 100
- Recall@: 10
- Search wall: `Search: 10000 queries in 3.57s (2800.0 QPS)`
- QPS: `Search: 10000 queries in 3.57s (2800.0 QPS)`
- Recall: `Recall@10: 0.8416`
- search_results bytes: `400008`

## Verdict

- Pass/fail: pass
- Generated index cleanup: pending in metadata
