# RTX 4090 handoff: upper-triangular tiles for native SQ4 leaf top-k

This branch contains a runnable FP32 scheduling experiment and its V100
measurements. **The production native INT4 path has not yet been changed.**
The next implementation and measurement belong on the RTX 4090 host.

## Verified scheduling invariant

For a leaf with `T = ceil(rows / 16)` logical row and column tiles, evaluate
tile pairs `(i, j)` only when `i <= j`. For `i < j`, compute one 16 × 16 dot
tile and use each result for both directed candidates: row `i` to column `j`
and row `j` to column `i`. For `i == j`, execute the full dot tile once, then
consume its rows only. Exclude self-pairs by comparing the **resident local
IDs**, as the existing kernel does; do not drop the diagonal tile. A full
diagonal tile still contains 240 useful directed off-diagonal entries.

The standalone [FP32 benchmark](../../script/bench_upper_triangle_fp32.cu)
implements both dense-matrix and direct tile-local top-k variants. On V100,
760-dimensional direct top-k gave 1.69× at 128 rows/leaf, 1.88× at 512, and
1.91× at 1280, with identical GPU top-k IDs across all tested leaves. See the
[measurement record](upper_triangle_fp32_v100.md) for all sizes, near-tie
details, and limits. These are scheduling measurements, not an INT4 speedup
estimate or an end-to-end SHEAR result.

## Current native INT4 implementation

- `shear-core/src/algorithm/pipnn/compute_backend.rs` contains
  `gpu_persistent_compute_topk_leaf_tile_cta_sq_int4_wmma_filter`. Its loop over
  `col_base` currently visits every column tile for each 16-row band, so a
  symmetric off-diagonal tile is computed in both orientations.
- The present packed-S4 WMMA shape is **8 × 8 × 32** with four MMA accumulators
  (`acc00`, `acc01`, `acc10`, `acc11`) combined into a logical 16 × 16 tile. It
  is not currently `m16n8k64`. Keep the four MMA operations for a scheduled
  tile and eliminate its *mirrored tile invocation*.
- `gpu_persistent_sq_wmma_consume_sq_filter_c_tile` converts the integer dot
  to distance using the source and target scales/norms and inserts candidates
  using `gpu_persistent_leaf_insert_survivor_fp32`. The reverse candidate must
  use the transposed `c_tile` index and swap source/target metadata; reusing
  the forward row's metadata would corrupt the distance and IDs.
- The native SQ4 mode currently keeps full-column leaf descriptors by default.
  The regression test `gpu_persistent_sq_native_int4_keeps_full_column_leaf_tasks`
  records a prior staged column-task slowdown (17:55.02 to 19:12.34 on
  Wiki35M, with 1.82 billion row-lock spins). Do not implement symmetry by
  merely enabling the existing staged column-task mode.
- For 760 input dimensions, the present 16-element sidecar alignment pads to
  768; the native kernel's K=32 loop therefore executes 24 slices and each
  packed-S4 code row uses 384 bytes. If a different RTX 4090 kernel uses
  `m16n8k64`, it needs 12 K=64 slices and two 16 × 8 MMA calls per logical
  16 × 16 tile. The same logical tile schedule applies.

## Implementation route

1. Add an opt-in native-SQ4 upper-triangular leaf mode, leaving the current
   path as the comparison baseline. Gate it on packed-S4, quantized-only
   output, no exact rerank/audit, and a workspace budget that covers its
   bounded scratch. Preserve fallback reporting so a run cannot silently
   compare different kernels.
2. Schedule logical tile pairs `(i, j)` with `i <= j` at the leaf-task level.
   Keep each diagonal tile intact. For an off-diagonal tile, produce both
   row-side and column-side candidate lists from one `c_tile`; map `c_tile[r,
   c]` to the reverse candidate for source column `c` and target row `r`.
   Each list needs its own source local ID, scale, norm, target global ID, and
   self-filter.
3. Merge candidate lists per source row through bounded scratch and a final
   owner for that row, as the FP32 direct benchmark does. Reuse the existing
   admission/workspace accounting. Avoid a contended per-row lock for every
   tile pair. The existing `2e-5` tie window in the survivor insertion helper
   can make results depend on merge order; compare against the baseline with
   near-tie cases and either preserve canonical candidate order or define and
   test a deterministic ordering rule.
4. First compare isolated native-SQ4 leaf tasks on RTX 4090, then run the same
   full build/search configuration with the feature disabled and enabled.
   Check exact top-k IDs/distances where deterministic, self-exclusion,
   duplicate resident IDs, partial tiles (15/16/17/31/32 rows), 760 dimensions,
   and representative leaves up to 1280 rows. Report actual INT4 WMMA tile
   counts, fallback count, merge time, scratch peak, build wall time, and
   recall. Keep input, `k`, threads, GPU workspace, and output mode identical.

The useful performance comparison is the full native-INT4 leaf pipeline,
including both directional candidate updates and merge. Skipped MMA tile
count alone is insufficient to establish a construction speedup.
