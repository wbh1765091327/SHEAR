# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_082839_persistent_profile_facts
- Date: 2026-06-14T08:49:49+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: current dirty workspace, persistent producer-consumer path
- Base: /data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin
- Build command changed from Wiki10M runbook shape: dataset/GT/output roots only
- Search command changed from runbook: not run because build failed

## Build

- Exit code: 1
- Wall time: Elapsed (wall clock) time (h:mm:ss or m:ss): 20:04.91
- Max RSS: Maximum resident set size (kbytes): 42621012
- _mem.index bytes: 24
- Profile: missing/not emitted because build failed before successful profile write

## Failure

```text
PiPNN: error: Index { err: "Failed to allocate CUDA GPU persistent subtree partition rows: IndexError: Failed to allocate CUDA GPU persistent subtree partition rows: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"); persistent_plan={max_queue_nodes=5305648 max_queue_rows=57909698 max_leaf_count=658072 max_leaf_rows=5264518 large_scratch_rows=5264518 required_bytes=569557860 budget_bytes=17179869184 resident_device_bytes=10802790936 device_admission_required_bytes=11372348796 device_admission_budget_bytes=11045339672 fallback_reason_flags=0x0 overflow_flags=0x0}" }
```

## Verdict

- Pass/fail: fail for Wiki35M build.
- Search: not run.
- Main fact: fully device-side persistent path reaches the same large resident subtree pressure point and fails allocating partition rows after resident upload; no CPU fallback occurs.
