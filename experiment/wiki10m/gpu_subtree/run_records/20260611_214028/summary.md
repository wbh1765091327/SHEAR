# Wiki10M GPU-Subtree Measurement

- Run ID: 20260611_214028
- Date: 2026-06-11T21:40:28+08:00
- Host: ccyilong
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: dirty workspace on branch gpu-next-tasks-2026-06-09; see `metadata.txt` for full `git status --short`.
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: no

## Input Checks

- Wiki10M base header: `10000000 1024`
- Query header: `10000 1024`
- GT header: `10000 100`
- Wiki10M base bytes: `40960000008`
- Query bytes: `40960008`
- GT bytes: `8000008`
- GT id range: ok, `max_id=9999985`, all ids are `< 10000000`

## Build

- Exit code: 0
- Wall time: `16:54.18`
- Max RSS: `15158808 kbytes`
- `_mem.index` bytes: `880296036`
- `profile.json` bytes: `6933`
- OOM memory budget bytes: `214748364800`
- GPU workspace GiB: 16
- GPU workspace budget bytes: `17179869184`
- GPU workspace peak bytes: `55271192`
- Partitioner kind: `gpu-subtree`
- Root assignment backend: `cpu`
- Subtree assignment backend: `cuda`
- Leaf top-k backend: `cuda`
- Child count: `64`
- Child points: `40000000`
- Leaf count: `263401`
- Compact top-k batches: `263401`
- Compact top-k rows: `80000000`
- Assignment passes: `10469`
- H2D bytes: `164484214416`
- D2H bytes: `1280000000`
- Resident H2D bytes: `164160000000`
- Resident top-k metadata H2D bytes: `324214416`
- Resident top-k metadata H2D ms: `14`
- Assignment kernel ms: `139828`
- Top-k kernel ms: `619027`
- Top-k D2H ms: `619087`
- CPU cluster merge ms: `535`
- CPU leaf global-to-local ms: `2190`
- CPU compact flush ms: `40711`
- CPU fallbacks: `0`
- Compact unsupported count: `0`
- Queue passes: `0`
- Queue kernel ms: `0`
- Queue overflow count: `64`
- Queue overflow flags: `2` (`GPU_QUEUE_OVERFLOW_QUEUE_ROW`)

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search wall time: `3.58s`
- QPS: `2791.6`
- Recall@10: `0.8559`
- Result file: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/search_results.fbin`
- Result bytes: `400008`

## Verdict

- Pass/fail: pass for full build + mem-search execution under the runbook command.
- Compared baseline: no accepted canonical baseline with the same base, query, GT, `SEARCH_L=100`, `RECALL_AT=10`, build parameters, and 16 GiB GPU workspace was found under `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs`.
- Regression or improvement: not classified; no same-contract accepted baseline found.
- Important note: this run did not exercise the intended successful level-based GPU resident queue path. `profile.json` records `queue_passes=0`, `queue_overflow_count=64`, `queue_overflow_flags=2`, and `resident_topk_metadata_h2d_bytes=324214416`. Flag `2` is `GPU_QUEUE_OVERFLOW_QUEUE_ROW`, so all 64 child runs overflowed queue row capacity and fell back to the existing resident compact-topk path. The build and search succeeded, but this is not evidence that the full 10M workload is using the new queue path.

## Artifacts

- Metadata: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/metadata.txt`
- Build stdout: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/stdout.log`
- Build stderr: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/stderr.log`
- Time log: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/time.log`
- Profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/profile.json`
- Index: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/index/_mem.index`
- Search log: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/search.log`
- Search results: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260611_214028/search_results.fbin`
