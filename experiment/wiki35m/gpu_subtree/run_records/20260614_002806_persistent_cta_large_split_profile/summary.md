# Wiki35M GPU-Subtree Profile Attempt

- Run ID: 20260614_002806_persistent_cta_large_split_profile
- Date: 2026-06-14T00:28:06+08:00
- Command: same GPU-subtree compact-topk command shape as the Wiki10M runbook, CUDA backend, 16 GiB GPU workspace
- Base: /data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin

## Build

- Exit code: 143 after manual termination
- Observation window: about 8.5 minutes wall time; persistent GPU phase observed for about 4 minutes after root par-spool completion
- Last stage reached: GPU-subtree partition started, then `[rbc/par-spool]` completed in stdout
- Last stdout line: `[rbc/par-spool] ... range_rows=35000000 ... total_ms=81754`
- GPU state during persistent phase: RTX 4090 100% utilization, memory about 15.8-16.6 GiB
- Host RSS during persistent phase: about 32.4-34.9 GiB
- `_mem.index` bytes: 0
- `profile.json`: not produced
- `time.log`: 0 bytes because the process was killed before `/usr/bin/time` could finish cleanly
- OOM temp dir: about 2.1 GiB

## Search

- Not run. Build did not produce a non-empty `_mem.index`.

## Verdict

- Fail for profile-producing validation.
- Wiki35M reaches the same post-`[rbc/par-spool]` persistent GPU-subtree phase as Wiki10M and does not emit `profile.json` or `_mem.index` during the observation window.
- This confirms the current blocker is in the persistent GPU-subtree path after root spool, not in Wiki10M data preparation.
