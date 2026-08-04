# Wiki10M Fanout6 Leaf WMMA B Sweep
Run root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep`
Build parameters: `fanout_top=6`, `fanout_second=3`, `max_leaders=1500`, `leaf_size_min=256`, `leaf_size_max=1280`, `leaf_knn=2`, `DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1`.

## Build Summary
| Variant | CTA shared B | Exit | Wall | Max RSS KB | `_mem.index` bytes | persistent successes | overflow flags | leaf max observed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per-warp-B | 0 | 0 | 9:57.20 | 10636900 | 1540517636 | 1500 | 0 | 1280 |
| CTA-shared-B | 1 | 0 | 9:53.24 | 10633056 | 1541215720 | 1500 | 0 | 1280 |

## Recall Sweep
| Variant | L | Recall@10 | QPS | Search wall s |
|---|---:|---:|---:|---:|
| per-warp-B | 25 | 0.7960 | 4504.4 | 2.22 |
| per-warp-B | 50 | 0.8761 | 3137.8 | 3.19 |
| per-warp-B | 100 | 0.9311 | 1991.0 | 5.02 |
| per-warp-B | 200 | 0.9638 | 1190.6 | 8.40 |
| per-warp-B | 400 | 0.9813 | 704.0 | 14.20 |
| CTA-shared-B | 25 | 0.7983 | 4436.2 | 2.25 |
| CTA-shared-B | 50 | 0.8777 | 3112.9 | 3.21 |
| CTA-shared-B | 100 | 0.9319 | 2034.0 | 4.92 |
| CTA-shared-B | 200 | 0.9648 | 1227.4 | 8.15 |
| CTA-shared-B | 400 | 0.9829 | 696.1 | 14.37 |

## Artifacts
- per-warp-B: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep/per_warp_b`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep/per_warp_b/profile.json`
  - search sweep: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep/per_warp_b/search_sweep.tsv`
- CTA-shared-B: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep/cta_shared_b`
  - profile: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep/cta_shared_b/profile.json`
  - search sweep: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_192834_fanout6_leaf_wmma_b_sweep/cta_shared_b/search_sweep.tsv`

## Notes
- Both variants completed the full build without `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Search used Wiki35M query file and Wiki10M GT generated against `wiki_10m_1024.fbin`.
- `_mem.index` byte size is recorded above before cleanup.
