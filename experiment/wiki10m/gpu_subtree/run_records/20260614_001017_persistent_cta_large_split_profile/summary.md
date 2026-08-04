# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_001017_persistent_cta_large_split_profile
- Date: 2026-06-14T00:10:17+08:00
- Command: runbook GPU-subtree compact-topk build, CUDA backend, 16 GiB GPU workspace
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin

## Build

- Exit code: 143 after manual termination
- Observation window: about 15 minutes wall time
- Last stage reached: GPU-subtree partition started, then `[rbc/par-spool]` completed in stdout
- Last stdout line: `[rbc/par-spool] ... range_rows=10000000 ... total_ms=19855`
- GPU state during observation: RTX 4090 mostly 100% utilization, memory about 14.5-15.1 GiB
- `_mem.index` bytes: 0
- `profile.json`: not produced
- `time.log`: 0 bytes because the process was killed before `/usr/bin/time` could finish cleanly
- OOM temp dir: about 611 MiB

## Search

- Not run. Build did not produce a non-empty `_mem.index`.

## Verdict

- Fail for runbook pass criteria.
- The current fully device-side persistent producer-consumer path does not produce Wiki10M `profile.json` or `_mem.index` within this observation window.
- It reaches the persistent GPU-subtree phase and keeps the GPU busy, but no host-visible counters/profile are emitted before kernel return.
