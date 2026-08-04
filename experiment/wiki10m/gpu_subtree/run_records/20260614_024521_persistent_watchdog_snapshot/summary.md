# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_024521_persistent_watchdog_snapshot
- Date: 2026-06-14 Asia/Shanghai
- Build command changed from runbook: yes, added `timeout --signal=INT --kill-after=60s 2700s` for diagnostic containment.
- Code changes under test: persistent watchdog snapshot instrumentation before init-watchdog fix.

## Build

- Exit code: 1
- Wall time: 1:06.45
- Max RSS: 9,616,372 KB
- `_mem.index` bytes: 24
- `profile.json`: absent
- Last completed stage: root GPU-subtree partition `[rbc/par-spool] total_ms=19926`.
- Failure: `CUDA persistent subtree overflow` on `resident_rows=148098` with `overflow_flags=0x800`.
- Diagnostic counters: `progress_epoch=0`, `partition_pops=0`, `leaf_pushes=0`, `topk_pops=0`, `snapshot_partition_head=0`, `snapshot_partition_tail=0`, `snapshot_active_partition=0`, `snapshot_active_topk=0`.

## Verdict

Fail. This was not a valid performance/profile run. It showed a watchdog false positive during kernel init/root publication, before any partition or top-k worker made progress.
