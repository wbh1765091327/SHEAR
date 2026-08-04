# Wiki10M GPU-Subtree Measurement

- Run ID: 20260617_125414_cpu_d0_wmma_perwarp_default_sweep
- Date: 2026-06-17
- Git commit: 59686f9693a0837d7906879ab60ab7822592ff61
- Code changes under test: canonical default now does not set `DELOANN_PIPNN_WMMA_CTA_SHARED_B`; WMMA B staging is per-warp default.
- Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep`
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Build command changed from runbook: no
- Search command changed from runbook: no
- Default path: CPU D0/root assignment + resident GPU D1+/leaf with selected `persistent-wmma-tf32`; leaf WMMA top-k disabled by default; per-warp-B WMMA staging.

## Build

- Exit code: 0
- Wall time: 11:39.35
- Max RSS: 10764088 KB (10.27 GiB)
- `_mem.index` bytes: 1933826632
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
- WMMA B staging note: canonical run did not set `DELOANN_PIPNN_WMMA_CTA_SHARED_B`; per-warp-B is the accepted default. CTA-shared-B is diagnostic only.

## Profile

- D0 wall: 52435 ms
- D1+ producer wall: 597884 ms
- D1+ backend wall: 589384 ms
- Persistent total: 220641 ms
- Persistent kernel event: 220609 ms
- Host load vectors: 213038 ms
- Host norms: 45512 ms
- Vector upload: 13200 ms
- Async flush: 142328 ms, batches=5820344, rows=240000000
- CPU result scatter: 8834 ms
- Persistent attempts/successes: 1500 / 1500
- Persistent queue overflow flags: 0
- Persistent worker blocks: partition=128, topk=128
- Persistent effective fanout: root=3, deeper=1
- Persistent leaves: count=5820344, max=1280, rows=240000000
- Persistent large node splits: 69130
- H2D bytes: 328320000000
- D2H bytes: 4893125504

## Search

- Search sweep file: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep/search_sweep.tsv`
- Search L values: 25,50,100,200,400
- Recall@: 10
- Query count: 10000
- Recall/QPS table:
  | L | Recall@10 | QPS | Wall time | Result file |
  |---|-----------|-----|-----------|-------------|
  | 25 | 0.8296 | 3686.5 | 2.71 s | `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep/search/search_l25_results.fbin` |
  | 50 | 0.8986 | 2541.8 | 3.93 s | `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep/search/search_l50_results.fbin` |
  | 100 | 0.9439 | 1656.0 | 6.04 s | `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep/search/search_l100_results.fbin` |
  | 200 | 0.9709 | 1007.7 | 9.92 s | `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep/search/search_l200_results.fbin` |
  | 400 | 0.9852 | 601.3 | 16.63 s | `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_125414_cpu_d0_wmma_perwarp_default_sweep/search/search_l400_results.fbin` |

- Best QPS point: L=25, QPS=3686.5, Recall@10=0.8296
- Best recall point: L=400, Recall@10=0.9852, QPS=601.3
- Default comparison point (`L=100`): Recall@10=0.9439, QPS=1656.0, wall=6.04 s

## Comparison

- Previous accepted CPU-D0 current-default run `20260617_120701_cpu_d0_wmma_leaf_tiledfp32` used the same 1500 root-leader quality contract and L=100 reported Recall@10 0.9449, QPS 1673.8, build wall 11:42.23.
- This per-warp-B canonical run reports L=100 Recall@10 0.9439, QPS 1656.0, build wall 11:39.35. The L=100 quality is effectively unchanged, with slightly lower QPS and a slightly shorter build wall in this single run.
- Do not compare this sweep against earlier 64-leader runs as code-only performance changes; those runs used materially different quality parameters.

## Verdict

- Pass/fail: pass
- Required profile checks: pass
- Search validation: pass
- Generated index cleanup: pass
- Notes: D0 uses CPU root assignment. D1 assignment uses persistent WMMA TF32. Leaf backend records `persistent-wmma-tf32`, but current stable default executes leaf top-k through the persistent tiled FP32 fallback. Canonical run does not set `DELOANN_PIPNN_WMMA_CTA_SHARED_B`.
