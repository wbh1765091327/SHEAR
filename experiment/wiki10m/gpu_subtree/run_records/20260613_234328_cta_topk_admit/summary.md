# Wiki10M GPU-Subtree Measurement

- Run ID: 20260613_234328_cta_topk_admit
- Date: 2026-06-13
- Code changes under test: planner admits Wiki10M observed persistent subtree; persistent top-k worker uses CTA row-parallel leaf processing
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: wrapped with timeout and manually interrupted after persistent kernel showed no progress
- Search command: not run because build did not complete

## Build

- Exit code wrapper: 130
- Time log: command terminated by signal 2
- Wall time: 11:30.29
- Max RSS: 9,616,336 kbytes
- `_mem.index` bytes: 0
- `profile.json`: missing
- GPU observation while running: RTX 4090 at 100% SM, about 14.1 GiB used
- Last stdout before interrupt:
  - root GPU-subtree partition started for 10,000,000 points
  - `[rbc/par-spool] ... total_ms=19815`

## Persistent Path Fact

- Unlike the earlier `20260613_231224_profile_facts` run, this run did not fail with `CUDA persistent strict error`.
- The planner admitted the observed Wiki10M resident subtree and the process entered the persistent CUDA kernel.
- The persistent kernel did not return within 11 minutes, so no counters/profile were available.
- This isolates the remaining blocker to in-kernel large-node processing, not host fallback/admission.

## Verdict

- Fail as an end-to-end graph build.
- Progress over previous run: strict admission gate no longer blocks Wiki10M.
- Remaining blocker: large-node recursive split is still effectively scalar inside the persistent kernel; CTA row-parallel top-k improved focused tests but does not make the full Wiki10M resident subtree complete.
