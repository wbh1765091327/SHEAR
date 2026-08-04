# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_103722_medium_node_local_split
- Date: 2026-06-14
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: medium nodes above leaf_max but within small_node_limit route through local persistent splitter
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 16:26.58
- Max RSS: 15088320 KB
- `_mem.index` bytes: 919501548
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Notable `profile.json` fields:
  - persistent_attempt_count: 64
  - persistent_success_count: 64
  - persistent_queue_overflow_flags: 0
  - persistent_partition_claims: 107587
  - persistent_partition_pops: 107587
  - persistent_topk_claims: 823436
  - persistent_topk_pops: 823436
  - persistent_leaf_pushes: 823436
  - persistent_large_node_splits: 4153
  - persistent_large_node_max_rows: 1378181
  - persistent_kernel_event_ms: 741000
  - persistent_total_ms: 741009
  - persistent_topk_rows: 80000000
  - persistent_leaf_count: 823436
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
- Search: 10000 queries in 4.07s (2455.7 QPS)
- Recall@10: 0.8257
- Result file bytes: 400008

## Verdict

- Pass/fail: pass
- Compared baseline: 20260614_100146_bounded_row_arena_v3
- Regression or improvement: Recall improved from 0.8203 to 0.8257, but build wall time regressed from 9:26.02 to 16:26.58 and persistent kernel time increased from ~321s to ~741s.
- Notes: route fix removes overflow on Wiki10M but is too conservative for performance; Wiki35M is required to decide whether it solves the prior 0x400 failure.
