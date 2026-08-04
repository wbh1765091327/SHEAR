# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_101425_bounded_row_arena_v3
- Date: 2026-06-14
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: bounded reusable persistent partition row arena, multiplier=4
- Base: /data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin
- Build command changed from Wiki10M runbook shape: dataset/output root only
- Search command changed from runbook shape: not run because build failed

## Build

- Exit code: 1
- Wall time: 12:50.72
- Max RSS: 40398716 KB
- _mem.index bytes: 24
- profile.json: missing because build failed before profile emission
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- RBC par-spool: point_reads=64 range_reads=35255 range_rows=35000000 hit_ratio=1.00 workers=8 build_x_ms=129057 gemm_ms=68607 topk_ms=15697 total_ms=72330

## Persistent Failure Snapshot

- Error: CUDA persistent subtree overflow: resident_rows=2056127 overflow_flags=0x400 leaf_rows=4111718 leaf_count=94177 partition_pops=3881 leaf_pushes=94177 topk_pops=94176 assignment_passes=0 max_active_points=0 persistent_kernel_event_ms=17757.521 progress_epoch=494762 watchdog_last_epoch=494740 watchdog_stall_polls=58917 watchdog_trips=0 active_work_ticks=6749456 active_work_trips=0 snapshot_partition_head=3881 snapshot_partition_tail=3881 snapshot_leaf_head=94174 snapshot_leaf_pushes=94174 snapshot_active_partition=1 snapshot_active_topk=9 snapshot_last_node_rows=267 snapshot_last_node_depth=6 snapshot_last_leaf_rows=8 snapshot_large_lock=4 snapshot_last_node_index=3880 snapshot_last_leaf_index=94176
- overflow_flags=0x400 means GPU_PERSISTENT_OVERSIZED_TERMINAL_NODE (1 << 10), not row-arena overflow.
- resident_rows: 2056127
- leaf_rows: 4111718
- leaf_count: 94177
- partition_pops: 3881
- leaf_pushes: 94177
- topk_pops: 94176
- persistent_kernel_event_ms: 17757.521
- snapshot_last_node_rows: 267
- snapshot_last_node_depth: 6
- snapshot_last_leaf_rows: 8
- snapshot_active_partition: 1
- snapshot_active_topk: 9

## Search

- Not run. The build failed and _mem.index is only 24 bytes.

## Verdict

- Pass/fail: fail
- Compared previous Wiki35M failure: progressed past CUDA allocation/admission OOM into real persistent kernel execution.
- Regression or improvement: improves scale admission versus the prior allocation failure, but still does not complete Wiki35M.
- Root-cause direction: remaining failure is algorithmic termination/leaf sizing in the persistent recursive splitter, where a terminal node with 267 rows exceeds persistent leaf_max=256.
