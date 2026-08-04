# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_050505_persistent_generation_fix
- Date: 2026-06-14T05:05:05+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace; persistent large-task CTA-uniform phase snapshot plus generation-tagged single-slot task/tile cursors.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: `9:26.80`
- Max RSS: `15114128` kbytes
- `_mem.index` bytes: `921384380`
- OOM memory budget bytes: `214748364800`
- GPU workspace GiB: 16
- Root partition line: `[rbc/par-spool] range_rows=10000000 total_ms=19853`
- `profile.json`: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_050505_persistent_generation_fix/profile.json`

## Persistent Profile

- `persistent_attempt_count`: 64
- `persistent_success_count`: 64
- `persistent_queue_overflow_flags`: 0
- `persistent_partition_claims/pops`: 107112 / 107112
- `persistent_topk_claims/pops`: 1665113 / 1665113
- `persistent_leaf_pushes`: 1665113
- `persistent_large_node_splits`: 107112
- `persistent_large_node_max_rows`: 1378181
- `persistent_large_node_rows`: 181166115
- `persistent_kernel_event_ms`: 319884
- `persistent_total_ms`: 319889
- `persistent_topk_rows`: 80000000
- `persistent_leaf_count`: 1665113
- `persistent_leaf_size_min/max`: 1 / 256
- `persistent_leaf_row_amplification_x1000`: 2000
- `resident_topk_kernel_event_ms`: 319884
- `host_control_ms`: 115200

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search: `Search: 10000 queries in 4.09s (2443.7 QPS)`
- Recall: `0.8210`
- Result file: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_050505_persistent_generation_fix/search_results.fbin` (400008 bytes)

## Verdict

- Pass/fail: pass for Wiki10M runbook gates.
- Compared baseline: previous complete fallback-style run `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028` built and searched in `16:54.18`, Recall@10 `0.8559`, QPS `2791.6`; this run built in `9:26.80`, Recall@10 `0.8210`, QPS `2443.7`.
- Regression or improvement: build time improved materially; Recall@10 and QPS are lower than the cited fallback-style baseline and should be treated as quality/perf regressions unless this persistent path has a separate accepted baseline.
- Notes: persistent device-side producer-consumer path completed with no queue overflow and consistent claim/pop counters. The dominant time is persistent kernel/top-k (`319884 ms`) plus host control (`115200 ms`).
