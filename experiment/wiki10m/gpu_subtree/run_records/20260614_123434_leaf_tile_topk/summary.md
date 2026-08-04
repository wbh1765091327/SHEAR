# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_123434_leaf_tile_topk
- Date: 2026-06-14
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: semantic leaf top-k row tiling; persistent leaf queue uses tile descriptors
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 18:31.99
- Max RSS: 15096660 KB
- `_mem.index` bytes: 881366276
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Partitioner kind: gpu-subtree
- Notable `profile.json` fields:
  - persistent_attempt_count: 64
  - persistent_success_count: 64
  - persistent_queue_overflow_flags: 0
  - persistent_partition_worker_blocks: 128
  - persistent_topk_worker_blocks: 128
  - persistent_partition_claims: 10574
  - persistent_partition_pops: 10574
  - persistent_topk_claims: 803654
  - persistent_topk_pops: 803654
  - persistent_leaf_pushes: 803654
  - persistent_leaf_count: 388171
  - persistent_leaf_size_min: 1
  - persistent_leaf_size_max: 1280
  - persistent_topk_rows: 80000000
  - persistent_kernel_event_ms: 866292
  - persistent_total_ms: 866296
  - persistent_large_node_splits: 4174
  - persistent_large_node_max_rows: 1378181
  - workspace_peak_bytes: 257119760
  - workspace_budget_bytes: 17179869184
  - topk_tasks_per_semantic_leaf: 2.070
  - avg_topk_rows_per_task: 99.55
  - avg_rows_per_semantic_leaf: 206.09

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search exit code: 0
- Search: 10000 queries in 3.63s (2753.8 QPS)
- Recall@10: 0.8554
- Result file bytes: 400008

## Verdict

- Pass/fail: partial pass
- Compared baseline: 20260614_103722_medium_node_local_split
- Regression or improvement: Recall improved from 0.8257 to 0.8554 and search QPS improved from 2455.7 to 2753.8, but build wall time regressed from 16:26.58 to 18:31.99; persistent kernel time regressed from 741000 ms to 866292 ms.
- Notes: Persistent producer-consumer counters are balanced and overflow-free. The new semantic leaf tiling behavior is visible: `persistent_leaf_count` dropped to 388171 while `persistent_topk_claims` stayed high at 803654 tile tasks. The original runbook check `persistent_leaf_size_max <= 1024` fails because this run records `persistent_leaf_size_max=1280`; code and logs indicate Wiki10M uses `adaptive_c_max=1280` (`leaf-size-max=640` doubled for >=10M), so this is a policy/criteria mismatch or intentional semantic-leaf expansion, not a queue overflow.
