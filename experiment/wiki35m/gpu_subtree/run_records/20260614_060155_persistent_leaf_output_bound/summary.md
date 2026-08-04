# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_060155_persistent_leaf_output_bound
- Date: 2026-06-14T06:22:05+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace; persistent leaf/top-k output capacity bounded by root fanout instead of append-only queue rows.
- Base: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin`
- Build command changed from Wiki10M runbook shape: yes, full Wiki35M base/GT/output root.
- Search command changed from runbook: yes, full Wiki35M base/GT.

## Build

- Exit code: 1
- Elapsed (wall clock) time (h:mm:ss or m:ss): 20:09.27
- Maximum resident set size (kbytes): 42733248
- `_mem.index` bytes: 24
- `profile.json` bytes: 0
- GPU workspace GiB: 16
- Root partition line: `2026-06-13T22:05:59.348695Z  INFO deloann::algorithm::pipnn::rbc_partition: [rbc/par-spool] point_reads=64 range_reads=35255 range_rows=35000000 hit_ratio=1.00 workers=8 build_x_ms=125912 gemm_ms=67858 topk_ms=15607 merge_ms=0 spool_write_ms=254 flush_ms=0 total_ms=71843`
- Error: `Error: Index { err: "Failed to allocate CUDA GPU persistent subtree partition rows: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")" }`

## Search

- Exit code: not_run
- Search L: 100
- Recall@: 10
- Search: not available
- Recall: not available
- Result bytes: 0

## Verdict

- Pass/fail: pending manual interpretation from profile/search fields.
