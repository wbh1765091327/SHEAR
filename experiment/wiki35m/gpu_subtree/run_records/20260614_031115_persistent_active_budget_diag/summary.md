# Wiki35M GPU-Subtree Diagnostic Measurement

- Run ID: 20260614_031115_persistent_active_budget_diag
- Date: 2026-06-14 Asia/Shanghai
- Build command: derived from Wiki10M GPU-subtree runbook, using Wiki35M canonical base/query/GT and the same GPU-subtree compact-topk parameters.
- Build command changed from full measurement intent: yes, added `timeout --signal=INT --kill-after=60s 600s` for diagnostic containment.
- Code changes under test: persistent active large-node work budget diagnostic.

## Build

- Exit code: 1
- `_mem.index` bytes: 24
- `profile.json`: absent
- Last completed stage: root GPU-subtree partition completed; `partition_d00_runs.bin` size 560,273,672 bytes.
- Strict error: `CUDA persistent subtree overflow`.
- `overflow_flags`: `0x1000`, active large-node work budget diagnostic.
- `resident_rows`: 546,143
- `persistent_kernel_event_ms`: 232.763
- `partition_pops`: 1
- `leaf_pushes`: 0
- `topk_pops`: 0
- `assignment_passes`: 0
- `progress_epoch`: 4
- `watchdog_trips`: 0
- `active_work_ticks`: 546,143
- `active_work_trips`: 132
- `snapshot_partition_head`: 1
- `snapshot_partition_tail`: 1
- `snapshot_active_partition`: 1
- `snapshot_active_topk`: 0
- `snapshot_last_node_rows`: 546,143
- `snapshot_last_node_depth`: 1
- `snapshot_large_lock`: 1
- `snapshot_last_node_index`: 0

## Verdict

Fail. Wiki35M reaches the same failure mode as Wiki10M, with a larger first root-child persistent node. The persistent producer-consumer path does not reach leaf production; the top-k consumer has no work. The immediate performance optimization target is large-node recursive split/assignment inside the persistent kernel, especially the single-CTA, single-lock large-node path.
