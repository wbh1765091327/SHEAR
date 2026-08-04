# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_100146_bounded_row_arena_v3
- Date: 2026-06-14
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: bounded reusable persistent partition row arena, multiplier=4
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 9:26.02
- Max RSS: 15089956 KB
- `_mem.index` bytes: 921354544
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Notable `profile.json` fields:
  - persistent_attempt_count: 64
  - persistent_success_count: 64
  - persistent_queue_overflow_flags: 0
  - persistent_partition_claims: 107015
  - persistent_partition_pops: 107015
  - persistent_topk_claims: 1665695
  - persistent_topk_pops: 1665695
  - persistent_leaf_pushes: 1665695
  - persistent_large_node_splits: 107015
  - persistent_large_node_max_rows: 1378181
  - persistent_kernel_event_ms: 320900
  - persistent_total_ms: 320900
  - persistent_topk_rows: 80000000
  - persistent_leaf_count: 1665695
  - persistent_leaf_size_min: 1
  - persistent_leaf_size_max: 256
  - persistent_leaf_row_amplification_x1000: 2000
  - workspace_peak_bytes: 254363344
  - workspace_budget_bytes: 17179869184
  - persistent_partition_worker_blocks: 128
  - persistent_topk_worker_blocks: 128

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search exit code: 0
- Search: 10000 queries in 4.20s (2380.8 QPS)
- Recall@10: 0.8203
- Result file bytes: 400008

## Verdict

- Pass/fail: pass
- Compared baseline: 20260614_081629_persistent_profile_facts
- Regression or improvement: build time roughly neutral; recall remains in the same lower persistent family around 0.820x
- Notes: persistent producer-consumer counters are self-consistent and overflow-free with bounded reusable row arena multiplier=4.
