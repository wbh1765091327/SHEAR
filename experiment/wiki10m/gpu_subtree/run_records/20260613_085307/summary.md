# Wiki10M GPU-Subtree Measurement

- Run ID: 20260613_085307
- Date: 2026-06-13T08:56:49+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: device-side persistent producer-consumer multi-CTA prototype as current working tree
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 1
- Wall time: 1:25.89
- Max RSS: 9564076 kbytes
- `_mem.index` bytes: 24
- OOM memory budget bytes: 214748364800 requested by command; profile.json was not produced because build failed before final profile write
- GPU workspace GiB: 16
- Notable observed fields:
  - GPU-subtree partition started for 10,000,000 points with adaptive_c_max=1,280 and min_recurse=2,560.
  - D0 par-spool line: point_reads=64 range_reads=10073 range_rows=10000000 hit_ratio=1.00 workers=8 build_x_ms=35355 gemm_ms=19412 topk_ms=4382 total_ms=20163.
  - Failure: CUDA persistent strict error on resident_rows=148098, leaf_knn=2, fanout_top=4, fanout_second=2, max_leaders=64, fallback_reason_flags=0x2, overflow_flags=0x200, required_bytes=4876571200, budget_bytes=17179869184.

## Search

- Search L: 100 planned
- Recall@: 10 planned
- Query count: 10000 expected from query header
- Search wall time: not run
- QPS: not run
- Recall: not run
- Result file: not produced

## Verdict

- Pass/fail: fail
- Compared baseline: none, build did not complete
- Regression or improvement: not measurable
- Notes: This run was blocked by the current strict persistent prototype's large-subtree admission rule. The first resident subtree needing GPU persistent processing had 148,098 unique resident rows, while `plan_gpu_persistent_subtree` rejects resident_rows above the 4,096 local node limit with fallback_reason_flags=0x2 / overflow_flags=0x200. This confirms the next required optimization is large resident subtree support in the persistent device-side path, not search tuning.
