# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_055026_persistent_leaf_output_bound
- Date: 2026-06-14T06:00:41+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace; persistent leaf/top-k output capacity bounded by root fanout instead of append-only queue rows.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Elapsed (wall clock) time (h:mm:ss or m:ss): 9:23.79
- Maximum resident set size (kbytes): 15104664
- `_mem.index` bytes: 921370760
- `profile.json` bytes: 8930
- GPU workspace GiB: 16
- Root partition line: `2026-06-13T21:51:30.242120Z  INFO deloann::algorithm::pipnn::rbc_partition: [rbc/par-spool] point_reads=64 range_reads=10073 range_rows=10000000 hit_ratio=1.00 workers=8 build_x_ms=35148 gemm_ms=19460 topk_ms=4460 merge_ms=0 spool_write_ms=73 flush_ms=0 total_ms=19820`

## Persistent Profile

- persistent_attempt_count: 64
- persistent_success_count: 64
- persistent_queue_overflow_flags: 0
- persistent_partition_claims/pops: 107192 / 107192
- persistent_topk_claims/pops: 1665384 / 1665384
- persistent_leaf_pushes: 1665384
- persistent_large_node_splits: 107192
- persistent_large_node_max_rows: 1378181
- persistent_kernel_event_ms: 319686
- persistent_total_ms: 319688
- persistent_topk_rows: 80000000
- persistent_leaf_count: 1665384
- persistent_leaf_size_min/max: 1 / 256
- persistent_leaf_row_amplification_x1000: 2000
- workspace_peak_bytes: 298206716
- workspace_budget_bytes: 17179869184

## Search

- Exit code: 0
- Search L: 100
- Recall@: 10
- Search: 10000 queries in 4.16s (2401.9 QPS)
- Recall@10: 0.8183
- Result bytes: 400008

## Verdict

- Pass/fail: pending manual interpretation from profile/search fields.
