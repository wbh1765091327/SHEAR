# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_040729_persistent_profile_facts
- Date: 2026-06-14T04:07:29+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace; persistent device-side producer-consumer path, including active-work telemetry-only behavior and CTA large-node base-leader start from current tree.
- Base: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin`
- Build command changed from Wiki10M runbook shape: same graph-build parameters, but full Wiki35M base/GT and output root; manual early stop after ~10 minutes process elapsed without host-visible profile/index output.
- Search command changed from runbook: not run because build did not complete.

## Input Checks

- Wiki35M base header: `35000000 1024`
- Query header: `10000 1024`
- GT header: `10000 100`
- GT id range: ok, `max_id=34999997`, all ids `< 35000000`
- GT bytes: `8000008`

## Build

- Exit code: manually terminated after persistent GPU stage did not return to host.
- Observed elapsed before stop: ~10 minutes from process start; root partition completed at ~3m48s after process start.
- Max RSS: not available from `/usr/bin/time` because process was manually terminated before `time.log` was written.
- `_mem.index` bytes: `0`
- `profile.json`: absent
- GPU workspace GiB: 16
- Root partition profile line: `[rbc/par-spool] ... range_rows=35000000 ... total_ms=73263`
- Last stdout line remained the root `[rbc/par-spool]` line throughout the persistent stage.
- GPU state during persistent stage: repeatedly sampled at `100%` utilization with ~15.8-16.6 GiB device memory.
- Host-visible persistent counters: unavailable because the persistent kernel did not return and no `profile.json` was written.

## Search

- Not run. The build did not produce a valid `_mem.index`.

## Verdict

- Pass/fail: fail for the runbook-style scale validation.
- Compared baseline: no successful persistent device-side producer-consumer Wiki35M baseline in this run directory family.
- Regression or improvement: current persistent path did not produce runbook profile facts or an index on Wiki35M.
- Notes: This reproduces the Wiki10M symptom at larger scale. After root partition completed (`total_ms=73263`), the persistent subtree stage occupied the GPU and did not return to host within the observation window. No persistent counters/profile could be collected.

## Artifacts

- Metadata: `/data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_040729_persistent_profile_facts/metadata.txt`
- Build stdout: `/data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_040729_persistent_profile_facts/stdout.log`
- Build stderr: `/data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_040729_persistent_profile_facts/stderr.log`
- Time log: `/data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_040729_persistent_profile_facts/time.log` (0 bytes)
- Profile: absent
- Index: `/data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_040729_persistent_profile_facts/index/_mem.index` (0 bytes)
