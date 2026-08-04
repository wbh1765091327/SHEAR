# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_035546_persistent_profile_facts
- Date: 2026-06-14T03:55:46+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace; persistent device-side producer-consumer path, including active-work telemetry-only behavior and CTA large-node base-leader start from current tree.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: yes, manual early stop after >10 minutes without host-visible profile/index output instead of waiting for the 3600s wrapper timeout.
- Search command changed from runbook: not run because build did not complete.

## Input Checks

- Wiki10M base header: `10000000 1024`
- Query header: `10000 1024`
- GT header: `10000 100`
- GT id range: ok, `max_id=9999985`, all ids `< 10000000`
- GT bytes: `8000008`

## Build

- Exit code: manually terminated after persistent GPU stage did not return to host.
- Observed elapsed before stop: >10 minutes from process start; root partition completed at ~38.6s after process start.
- Max RSS: not available from `/usr/bin/time` because process was manually terminated before `time.log` was written.
- `_mem.index` bytes: `0`
- `profile.json`: absent
- GPU workspace GiB: 16
- Root partition profile line: `[rbc/par-spool] ... range_rows=10000000 ... total_ms=20279`
- Last stdout line remained the root `[rbc/par-spool]` line throughout the persistent stage.
- GPU state during persistent stage: repeatedly sampled at `100%` utilization with ~14.1-14.9 GiB device memory.
- Host-visible persistent counters: unavailable because the persistent kernel did not return and no `profile.json` was written.

## Search

- Not run. The build did not produce a valid `_mem.index`.

## Verdict

- Pass/fail: fail for the runbook.
- Compared baseline: previous complete fallback-style run `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028` built and searched successfully in `16:54.18`, but did not exercise the intended successful persistent queue path.
- Regression or improvement: current persistent path did not produce runbook profile facts or an index on Wiki10M.
- Notes: The important profile fact is negative: after the root partition completed (`total_ms=20279`), the fully device-side persistent subtree stage occupied the GPU but did not return to host within the observation window, so no persistent counters/profile could be collected. This matches a large-node producer bottleneck / long-running persistent-kernel symptom rather than a completed producer-consumer pipeline.

## Artifacts

- Metadata: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_035546_persistent_profile_facts/metadata.txt`
- Build stdout: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_035546_persistent_profile_facts/stdout.log`
- Build stderr: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_035546_persistent_profile_facts/stderr.log`
- Time log: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_035546_persistent_profile_facts/time.log` (0 bytes)
- Profile: absent
- Index: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_035546_persistent_profile_facts/index/_mem.index` (0 bytes)
