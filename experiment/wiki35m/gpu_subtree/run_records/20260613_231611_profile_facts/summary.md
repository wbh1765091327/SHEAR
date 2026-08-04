# Wiki35M GPU-Subtree Scale Check

- Run ID: 20260613_231611_profile_facts
- Date: 2026-06-13
- Git commit: recorded in metadata.txt
- Code changes under test: current dirty workspace, persistent device-side producer-consumer strict default with scalar large-node gate
- Base: /data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin
- Build command: Wiki10M runbook command family, with base/GT changed to Wiki35M
- Search command: not run because build failed

## Build

- Exit code: 1
- Wall time: 7:32.19
- Max RSS: 32,548,736 kbytes
- `_mem.index` bytes: 24
- `profile.json`: missing
- OOM memory budget bytes: requested 214,748,364,800
- GPU workspace GiB: 16
- Resident reservoir estimate: 30,520,000,000 bytes
- Root GPU-subtree partition started for 35,000,000 points
- Last diagnostic: `CUDA persistent strict error: unsupported subtree`
- Persistent admission values:
  - `resident_rows=546143`
  - `leaf_knn=2`
  - `fanout_top=4`
  - `fanout_second=2`
  - `max_leaders=64`
  - `leaf_min=128`
  - `leaf_target=512`
  - `leaf_max=1024`
  - `small_node_limit=4096`
  - `scalar_large_node_limit=16384`
  - `max_queue_nodes=1096553`
  - `max_queue_rows=12015146`
  - `max_leaf_count=17068`
  - `max_leaf_rows=1092286`
  - `fallback_reason_flags=0x2`
  - `overflow_flags=0x200`
  - `required_bytes=114248892`
  - `budget_bytes=17179869184`

## Search

- Not run. The build did not produce a valid `_mem.index` or `profile.json`.

## Verdict

- Fail as an end-to-end graph build.
- This scale check confirms that the current fully device-side persistent path would need to handle resident subtrees at least in the 546k-row range for Wiki35M under this run family.
- The current scalar large-node admission limit is 16,384 rows, so the default strict path correctly aborts instead of entering the previously observed long-running scalar persistent kernel.
