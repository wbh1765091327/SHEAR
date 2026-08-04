# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_015601_persistent_mid_cta_split_leafdesc_x8
- Date: 2026-06-14
- Git commit: see metadata.txt
- Code changes under test: persistent mid-size nodes routed to CTA split; persistent leaf descriptor capacity increased x8
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: yes, wrapped with timeout/INT control and stopped manually at 30 minutes because the persistent kernel did not return
- Search command changed from runbook: not run; build did not produce a valid index

## Build

- Exit status: interrupted by signal 2 after 30:02.54 wall time; wrapper observed exit code 130
- Wall time: 30:02.54
- Max RSS: 11,428,344 KB
- `_mem.index` bytes: 0
- `profile.json`: not produced
- OOM memory budget bytes: unavailable in profile because profile was not produced; command used `--oom-memory-budget-gb 200`
- GPU workspace GiB: 16
- Last stdout line: `[rbc/par-spool] ... total_ms=19984`
- Observed GPU state while stuck: SM utilization 99-100%, memory utilization 0%, GPU memory roughly 14.5-15.5 GiB
- Host-side boundary: `run_persistent_subtree_topk` launches `gpu_persistent_subtree_topk` and waits in `stream.synchronize()`; persistent counters are downloaded only after the kernel returns

## Search

- Not run. The build did not produce a non-empty `_mem.index` or `profile.json`.

## Verdict

- Pass/fail: fail
- Compared baseline: previous failing persistent runs
- Regression or improvement: improvement over the immediate descriptor-overflow failure only in the narrow sense that it no longer failed fast with `overflow_flags=0x4`; it still does not complete the Wiki10M runbook
- Notes: This run does not provide valid profile facts for performance tuning. The next required work is device-side observability/watchdog or a completion-condition fix for the persistent producer-consumer kernel before Wiki35M can be measured meaningfully.
