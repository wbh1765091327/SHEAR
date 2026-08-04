# Wiki10M GPU-Subtree Measurement

- Run ID: 20260617_120701_cpu_d0_wmma_leaf_tiledfp32
- Date: 2026-06-17
- Git commit: 59686f9693a0837d7906879ab60ab7822592ff61
- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_120701_cpu_d0_wmma_leaf_tiledfp32`
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 11:42.23
- Max RSS: 10830284 KB (10.33 GiB)
- `_mem.index` bytes: 1933790824
- OOM memory budget bytes: 214748364800
- GPU workspace bytes: 17179869184
- Workspace peak bytes: 89538208
- Leaf size min/max: 256 / 1280
- Max leaders: 1500 root D0 leaders; D1+ device leader stride is 128
- Fanout schedule: depth0=8, depth1=3, depth>=2=1
- Root risk candidate rs: 1,2,4,8
- Root assignment backend: cpu
- D1 assignment backend: persistent-wmma-tf32
- D1 subtree assignment backend: persistent-wmma-tf32
- Leaf top-k selected backend: persistent-wmma-tf32
- Leaf top-k execution note: `persistent-wmma-tf32` is selected, but leaf WMMA top-k is disabled by default; the persistent worker uses the tiled FP32 top-k fallback unless `DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1` is explicitly set.

## Profile

- D0 wall: 52307 ms
- D1+ producer wall: 602607 ms
- D1+ backend wall: 593850 ms
- Persistent total: 220168 ms
- Persistent kernel event: 220136 ms
- Host load vectors: 213865 ms
- Host norms: 45323 ms
- Vector upload: 13209 ms
- Async flush: 134508 ms, batches=5820334, rows=240000000
- CPU result scatter: 8960 ms
- Persistent attempts/successes: 1500 / 1500
- Persistent queue overflow flags: 0
- Persistent worker blocks: partition=128, topk=128
- Persistent effective fanout: root=3, deeper=1
- Persistent leaves: count=5820334, max=1280, rows=240000000
- Persistent large node splits: 69130
- H2D bytes: 328320000000
- D2H bytes: 4893125344

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search wall time: 5.97 s
- QPS: 1673.8
- Recall@10: 0.9449
- Result file: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_120701_cpu_d0_wmma_leaf_tiledfp32/search_results.fbin`
- Result bytes: 400008

## Comparison

- Current default-path result: 11:42.23 build wall, Recall@10 0.9449, QPS 1673.8.
- Closest previous successful CPU-D0 + `persistent-wmma-tf32` run in the staging tree, `20260616_185933_staged_async_wmma`, used only 64 persistent successes/root leaders and therefore is not parameter-equivalent to this default 1500-leader run. It built in 4:31.25 with Recall@10 0.8406 and QPS 2548.5.
- The higher default leader count and fanout-quality contract materially improve recall here: 0.9449 vs 0.8406, at the cost of a longer build and lower search QPS from the generated graph.
- The earlier attempt in this session, `20260617_115655_cpu_d0_wmma_guard640`, failed at D1+ child 9 with `CUDA_ERROR_ILLEGAL_ADDRESS` while leaf WMMA top-k was still enabled for smaller leaves. That is why the accepted default keeps leaf `persistent-wmma-tf32` selected but disables the leaf WMMA top-k subpath internally.

## Verdict

- Pass/fail: pass
- Required profile checks: pass
- Search validation: pass
- Generated index cleanup: pass
- Notes: D0 uses CPU root assignment. D1 assignment uses persistent WMMA TF32. Leaf backend records `persistent-wmma-tf32`, but current stable default executes leaf top-k through the persistent tiled FP32 fallback.
