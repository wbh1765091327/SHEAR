# Wiki10M GPU-Subtree Persistent Measurement

- Run ID: `20260613_161255_persistent_full_pty`
- Date: `2026-06-13T16:12:55+08:00`
- Runbook: `/home/wbh/cppwork/PiPGANN/docs/runbooks/wiki10m_gpu_subtree_measurement.md`
- Launch mode: foreground PTY session, after two background launches exited before `time.log`/`build.exit` could be written.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: not run because build did not complete

## Input Checks

- Wiki10M base header: `10000000 1024`
- Query header: `10000 1024`
- Wiki10M GT header: `10000 100`
- Wiki10M GT max id: `9999985`, all ids `< 10000000`

## Build

- Build exit code recorded by wrapper: `130`
- Termination: manual SIGINT after no host-side progress beyond the first D0 spool log.
- `/usr/bin/time` wall time: `18:16.65`
- `/usr/bin/time` max RSS: `10403556 kbytes`
- `profile.json`: not generated
- `_mem.index`: exists but is `0` bytes
- `stderr.log`: `0` bytes
- GPU observation before termination: RTX 4090 at `100%` SM utilization, about `14.5 GiB` FB memory used by `build_pipnn_index`, power about `85 W`.

Last stdout progress:

```text
2026-06-13T08:13:21.326911Z  INFO deloann::algorithm::pipnn: PiPNN OOM thread model: requested_threads=8 pool_threads=8 leaf_budget_threads=8 memory_budget_bytes=214748364800
2026-06-13T08:13:39.273937Z  INFO deloann::algorithm::pipnn: PiPNN OOM resident-reservoir mode enabled: num_points=10000000 reservoir_bytes_estimate=8720000000 alloc_elapsed=2.921696216s
2026-06-13T08:13:39.286634Z  INFO deloann::algorithm::pipnn::rbc_partition: GPU-subtree partition starting: 10,000,000 points, adaptive_c_max=1,280, min_recurse=2,560
2026-06-13T08:13:59.009265Z  INFO deloann::algorithm::pipnn::rbc_partition: [rbc/par-spool] point_reads=64 range_reads=10073 range_rows=10000000 hit_ratio=1.00 workers=8 build_x_ms=34716 gemm_ms=19229 topk_ms=4455 merge_ms=0 spool_write_ms=64 flush_ms=0 total_ms=19719
```

## Search

Not run. The build did not produce a non-empty `_mem.index`.

## Verdict

- Pass/fail: fail for runbook completion.
- Full profile facts: unavailable, because `profile.json` is only written after build completion and this run did not complete.
- Observed fact: after D0 `par-spool`, the current fully device-side persistent path entered a long GPU-resident segment that kept SM utilization at `100%` but did not return to the host or write any further log/profile/index output within `18:16.65`.
- Wiki35M was not launched after this failure because it would exercise the same persistent path at larger scale without first obtaining a successful Wiki10M profile.

