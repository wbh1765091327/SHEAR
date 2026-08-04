# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_090616_resize_mempool_trim
- Date: 2026-06-14T09:26:12+08:00
- Code changes under test: current dirty workspace, persistent producer-consumer path with resize-time CUDA mempool trim
- Base: /data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin
- Build command changed from Wiki10M runbook shape: dataset path only

## Build

- Exit code: 1
- Wall time: Elapsed (wall clock) time (h:mm:ss or m:ss): 19:55.66
- Max RSS: Maximum resident set size (kbytes): 42367388
- _mem.index bytes: 24
- Profile: not emitted because build failed
- Search: not run because build failed

## Root Partition

```text
GPU-subtree partition starting: 35,000,000 points, adaptive_c_max=1,280, min_recurse=2,560
[rbc/par-spool] point_reads=64 range_reads=35255 range_rows=35000000 hit_ratio=1.00 workers=8 build_x_ms=127911 gemm_ms=68448 topk_ms=15648 merge_ms=0 spool_write_ms=241 flush_ms=0 total_ms=72617
```

## Failure

```text
Failed to allocate CUDA GPU persistent subtree partition rows:
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")

persistent_plan={
  max_queue_nodes=5305648
  max_queue_rows=57909698
  max_leaf_count=658072
  max_leaf_rows=5264518
  large_scratch_rows=5264518
  required_bytes=569557860
  budget_bytes=17179869184
  resident_device_bytes=10802790936
  device_admission_required_bytes=11372348796
  device_admission_budget_bytes=11045339672
  fallback_reason_flags=0x0
  overflow_flags=0x0
}
```

## Verdict

- Pass/fail: fail for Wiki35M measurement.
- The resize-time CUDA mempool trim did not remove the scale bottleneck.
- Failure is still before persistent kernel completion/search, at persistent partition-row allocation.
- Next optimization should address persistent workspace shape/admission, especially append-only `max_queue_rows`, rather than another fallback path.
