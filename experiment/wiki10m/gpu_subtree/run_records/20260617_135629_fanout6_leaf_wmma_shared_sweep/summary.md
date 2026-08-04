# Wiki10M Fanout6 Forced Leaf WMMA Top-k Shared-B Diagnostic

- Suite: `20260617_135629_fanout6_leaf_wmma_shared_sweep`
- Purpose: compare forced leaf WMMA top-k with WMMA B staging disabled/enabled.
- Build mode: build-only diagnostic; no search run.
- Dataset: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Parameters: `--max-leaders 1500`, `--fanout-top 6`, `--fanout-second 3`, depth>=2 fanout effectively 1, `--root-risk-candidate-rs 1,2,4,6`.
- Forced leaf WMMA top-k: `DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1`.
- D1+ assignment backend: `persistent-wmma-tf32`.
- Leaf top-k backend: `persistent-wmma-tf32`.

## Results

| Variant | `DELOANN_PIPNN_WMMA_CTA_SHARED_B` | Exit | Wall time | Max RSS KB | D0 total ms | Failure |
|---|---:|---:|---:|---:|---:|---|
| per-warp-B | 0 | 1 | 1:47.24 | 9656616 | 52033 | D1+ child 8/1500, run_len=10515, extents=3, `CUDA_ERROR_ILLEGAL_ADDRESS` |
| CTA-shared-B | 1 | 1 | 1:43.27 | 9646564 | 50891 | D1+ child 16/1500, run_len=11664, extents=3, `CUDA_ERROR_ILLEGAL_ADDRESS` |

## Interpretation

Both forced leaf WMMA top-k variants failed before completing the full graph build. Therefore this suite does not produce comparable completed build times for shared-B on/off. The only valid timing comparison is time-to-failure, and that is not a throughput result because each variant failed on a different D1+ child.

The diagnostic confirms the earlier reason for keeping `DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK` off in the default path: full Wiki10M with forced leaf WMMA top-k still hits CUDA illegal memory access, even with root fanout reduced to 6 and second-level fanout 3.

## Artifacts

- per-warp-B logs: `per_warp_b/stdout.log`, `per_warp_b/stderr.log`, `per_warp_b/time.log`, `per_warp_b/metadata.txt`
- CTA-shared-B logs: `cta_shared_b/stdout.log`, `cta_shared_b/stderr.log`, `cta_shared_b/time.log`, `cta_shared_b/metadata.txt`
- No valid `_mem.index` was produced; only 24-byte partial headers were left by the failed runs.
