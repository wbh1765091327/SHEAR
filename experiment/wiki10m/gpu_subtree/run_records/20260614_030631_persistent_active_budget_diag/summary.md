# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_030631_persistent_active_budget_diag
- Date: 2026-06-14 Asia/Shanghai
- Build command changed from runbook: yes, added `timeout --signal=INT --kill-after=60s 600s` for diagnostic containment.
- Code changes under test: persistent active large-node work budget diagnostic.

## Build

- Exit code: 1
- `_mem.index` bytes: 24
- `profile.json`: absent
- Last completed CPU/root stage: root GPU-subtree partition completed with `[rbc/par-spool] total_ms=19926`.
- Strict error: `CUDA persistent subtree overflow`.
- `overflow_flags`: `0x1000`, active large-node work budget diagnostic.
- `resident_rows`: 148,098
- `persistent_kernel_event_ms`: 228.717
- `partition_pops`: 1
- `leaf_pushes`: 0
- `topk_pops`: 0
- `assignment_passes`: 0
- `progress_epoch`: 4
- `watchdog_trips`: 0
- `active_work_ticks`: 148,098
- `active_work_trips`: 35
- `snapshot_partition_head`: 1
- `snapshot_partition_tail`: 1
- `snapshot_active_partition`: 1
- `snapshot_active_topk`: 0
- `snapshot_last_node_rows`: 148,098
- `snapshot_last_node_depth`: 1
- `snapshot_large_lock`: 1
- `snapshot_last_node_index`: 0

## Verdict

Fail. This is now an interpretable failure: the first persistent subtree kernel reaches the root-child large-node CTA split and spends all useful work inside active partition processing while holding `large_scratch_lock`. No leaf is produced and no top-k worker consumes anything before the active-work budget trips. Therefore the current fully device-side producer-consumer path is not yet performance-measurable end-to-end on Wiki10M; the next optimization target is the large-node split implementation, not leaf top-k consumer overlap.
