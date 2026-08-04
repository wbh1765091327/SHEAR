# Wiki10M GPU-Subtree Measurement

- Run ID: 20260613_231224_profile_facts
- Date: 2026-06-13
- Git commit: recorded in metadata.txt
- Code changes under test: current dirty workspace, persistent device-side producer-consumer strict default with scalar large-node gate
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: no, except wrapped with timeout for safety
- Search command changed from runbook: not run because build failed

## Build

- Exit code: 1
- Wall time: 1:05.81
- Max RSS: 9,567,076 kbytes
- `_mem.index` bytes: 24
- `profile.json`: missing
- OOM memory budget bytes: requested 214,748,364,800
- GPU workspace GiB: 16
- Last diagnostic: `CUDA persistent strict error: unsupported subtree`
- Persistent admission values:
  - `resident_rows=148098`
  - `leaf_knn=2`
  - `fanout_top=4`
  - `fanout_second=2`
  - `max_leaders=64`
  - `leaf_min=128`
  - `leaf_target=512`
  - `leaf_max=1024`
  - `small_node_limit=4096`
  - `scalar_large_node_limit=16384`
  - `max_queue_nodes=297354`
  - `max_queue_rows=3258156`
  - `max_leaf_count=4632`
  - `max_leaf_rows=296196`
  - `fallback_reason_flags=0x2`
  - `overflow_flags=0x200`
  - `required_bytes=30982964`
  - `budget_bytes=17179869184`

## Search

- Not run. The build did not produce a valid `_mem.index` or `profile.json`.

## Verdict

- Fail by current pass criteria.
- This run did not reproduce the earlier 2h GPU-resident hang. It failed fast due to the strict scalar large-node admission gate.
- Profile fact: current fully device-side persistent path cannot be treated as Wiki10M-ready while its large-node split/top-k work remains scalar-limited at 16,384 resident rows.
