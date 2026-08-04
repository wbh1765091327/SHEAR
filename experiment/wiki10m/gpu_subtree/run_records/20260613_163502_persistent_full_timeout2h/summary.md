# Wiki10M GPU-Subtree Persistent Measurement

- Run ID: `20260613_163502_persistent_full_timeout2h`
- Date: `2026-06-13T16:35:02+08:00`
- Runbook: `/home/wbh/cppwork/PiPGANN/docs/runbooks/wiki10m_gpu_subtree_measurement.md`
- Launch mode: foreground PTY session with `timeout --signal=INT --kill-after=60s 7200s`
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: yes, only an outer 2h timeout wrapper was added.
- Search command changed from runbook: not run because build did not complete.

## Input Checks

- Wiki10M base header: `10000000 1024`
- Query header: `10000 1024`
- Wiki10M GT header: `10000 100`
- Wiki10M GT max id: `9999985`, all ids `< 10000000`

## Build

- Exit code: `124`
- Timeout: `7200s`
- Wall time from `time.log`: `2:00:00`
- Max RSS from `time.log`: `10710952 kbytes`
- `profile.json`: not generated
- `_mem.index`: exists but is `0` bytes
- `stderr.log`: `0` bytes
- GPU observation during run: RTX 4090 stayed at `100%` SM utilization for the long resident segment, with about `14.1-14.8 GiB` FB memory used by `build_pipnn_index`.
- Host thread observation during run: one `build_pipnn_index` thread stayed CPU-running at about one full core; the remaining worker/CUDA event threads were mostly in futex/poll waits.

Last stdout progress:

```text
2026-06-13T08:35:28.183841Z  INFO deloann::algorithm::pipnn: PiPNN OOM thread model: requested_threads=8 pool_threads=8 leaf_budget_threads=8 memory_budget_bytes=214748364800
2026-06-13T08:35:46.052606Z  INFO deloann::algorithm::pipnn: PiPNN OOM resident-reservoir mode enabled: num_points=10000000 reservoir_bytes_estimate=8720000000 alloc_elapsed=2.885396758s
2026-06-13T08:35:46.065906Z  INFO deloann::algorithm::pipnn::rbc_partition: GPU-subtree partition starting: 10,000,000 points, adaptive_c_max=1,280, min_recurse=2,560
2026-06-13T08:36:05.857187Z  INFO deloann::algorithm::pipnn::rbc_partition: [rbc/par-spool] point_reads=64 range_reads=10073 range_rows=10000000 hit_ratio=1.00 workers=8 build_x_ms=35077 gemm_ms=19338 topk_ms=4433 merge_ms=0 spool_write_ms=62 flush_ms=0 total_ms=19788
```

## Search

Not run. The build did not produce a non-empty `_mem.index`.

## Comparison Notes

- Previous successful Wiki10M run `20260611_214028` completed build in `16:54.18`, but did not exercise the intended queue success path.
- Previous queue experiment runs `20260612_025458_two_tier_queue_admission_fix` and `20260612_0904_level_dispatch_queue` completed in about `32:51.76` and `33:12.44`.
- This run did not return to host after the first D0 `par-spool` log even after `2:00:00`.

## Verdict

- Pass/fail: fail for runbook completion.
- Full `profile.json` facts: unavailable because the build never completed and the profile is only written at completion.
- Scale-validation conclusion: current default fully device-side persistent producer-consumer path is not ready for Wiki10M/Wiki35M scale validation. On Wiki10M it enters a long GPU-resident segment after D0 spool and fails to complete within 2 hours.
- Wiki35M was not launched because the same path failed on Wiki10M, and launching 35M would not produce meaningful profile facts before fixing the 10M completion/progress issue.

