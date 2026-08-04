# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_051708_persistent_generation_fix
- Date: 2026-06-14T05:17:08+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace; persistent large-task CTA-uniform phase snapshot plus generation-tagged single-slot task/tile cursors.
- Base: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin`
- Build command changed from Wiki10M runbook shape: yes, full Wiki35M base/GT and output root.
- Search command changed from runbook: not run because build failed.

## Input Checks

- Wiki35M base header: `35000000 1024`
- Query header: `10000 1024`
- GT header: `10000 100`
- GT id range: checked before build, all ids `< 35000000`

## Build

- Exit code: 1
- Wall time: `16:43.92`
- Max RSS: `41190584` kbytes
- `_mem.index` bytes: `24`
- `profile.json`: absent
- GPU workspace GiB: 16
- Root partition line: `[rbc/par-spool] point_reads=64 range_reads=35255 range_rows=35000000 hit_ratio=1.00 workers=8 build_x_ms=127447 gemm_ms=68358 topk_ms=15829 merge_ms=0 spool_write_ms=244 flush_ms=0 total_ms=72190`
- Error: `Error: Index { err: "Failed to allocate CUDA GPU persistent subtree top-k distances: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")" }`

## Search

- Not run. The build did not produce a valid `_mem.index` or `profile.json`.

## Verdict

- Pass/fail: fail for Wiki35M scale validation.
- Compared baseline: no successful persistent device-side producer-consumer Wiki35M baseline in this run family.
- Regression or improvement: improved over the earlier symptom because the run progressed past root into the persistent path and failed with a concrete allocation error instead of requiring manual termination, but it still does not complete Wiki35M.
- Notes: The failure is an actual CUDA allocation failure for `GPU persistent subtree top-k distances` after ~16m44s, with the process having reached ~39 GiB RSS and GPU memory previously sampled near the 24GB device limit. This points to persistent planner/device workspace sizing for full 35M rather than the earlier large-task queue liveness bug.
