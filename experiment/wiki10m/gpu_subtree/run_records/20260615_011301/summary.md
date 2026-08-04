# Wiki10M GPU-Subtree Measurement

- Run ID: 20260615_011301
- Date: 2026-06-15T01:13:01+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: persistent CUDA subtree top-k now uses the planner semantic leaf cap instead of widening `leaf_max` to `adaptive_c_max`; added a CUDA regression test for `adaptive_c_max > c_max`.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 11:42.40
- Max RSS: 15086492 KB
- `_mem.index` bytes: 901943736
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Profile JSON: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_011301/profile.json`
- Notable `profile.json` fields:
  - `partitioner.kind`: `gpu-subtree`
  - `persistent_attempt_count`: 64
  - `persistent_success_count`: 64
  - `persistent_fallback_count`: 0
  - `persistent_queue_overflow_flags`: 0
  - `persistent_partition_worker_blocks`: 128
  - `persistent_topk_worker_blocks`: 128
  - `persistent_partition_claims`: 32479
  - `persistent_partition_pops`: 32479
  - `persistent_leaf_pushes`: 1285031
  - `persistent_topk_claims`: 1285031
  - `persistent_topk_pops`: 1285031
  - `persistent_large_node_splits`: 32479
  - `persistent_large_node_max_rows`: 1378181
  - `persistent_leaf_size_max`: 640
  - `persistent_leaf_max`: 640
  - `persistent_leaf_tile_rows`: 128
  - `persistent_partition_helped_topk_tiles`: 46
  - `persistent_topk_helped_large_tiles`: 612799073
  - `persistent_elastic_role_switches`: 612799119

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search wall time: 4.10s
- QPS: 2437.0
- Recall: 0.8419
- Result file: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_011301/search_results.fbin`
- Result file bytes: 400008

## Verdict

- Pass/fail: pass
- Compared baseline: no accepted numeric baseline is recorded yet for this exact Wiki10M base/query/GT/search-parameter run family.
- Regression or improvement: the previous run `20260615_004924` built successfully but failed the mandatory persistent profile gate with `persistent_leaf_size_max=1280`; this run passes that gate with `persistent_leaf_size_max=640` and completed search validation.
- Notes: build stdout contains `PiPNN: index build completed successfully`; build stderr is empty; search exited with status 0.
