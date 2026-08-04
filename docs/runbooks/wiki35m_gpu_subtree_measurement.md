# Wiki35M GPU-Subtree Measurement Runbook

This runbook is the required measurement procedure for full Wiki35M
GPU-subtree/resident graph-construction validation. Use it after changes that
can affect full 35M build time, GPU-subtree behavior, persistent resident
top-k behavior, emitted `_mem.index` quality, or post-build search recall.

The primary measurement is a full Wiki35M graph build with:

```text
build_shear_index --partitioner gpu-subtree
```

The current default path for this runbook is:

```text
D0/root assignment: GPU k-means coarse D0 with GPU-D0 SQ sidecar staging,
  capacity-aware root planning, and lazy D1+ admission split
D1+ subtree execution: fused-persistent resident GPU kernel
D1/subtree assignment backend: persistent-wmma-tf32
Leaf/top-k backend: persistent-wmma-tf32
D0 SQ staging: GPU-D0 SQ sidecar rows generated during coarse assignment
HashPrune sketch source: computed from the same SQ GPU-D0 raw staging batches;
  standalone sketch persist is skipped
D1+ distance source: SQ-int8 WMMA sidecar in the persistent kernel
Leaf WMMA top-k subpath: forced enabled for canonical Wiki35M validation
WMMA B tile staging: per-warp
OOM I/O mode: buffered/page-cache
Compact lune witness metadata: disabled
Compact lune pre-HashPrune pruning: disabled
Compact spill digest: disabled
Pruning backend: HashPrune
LRPF pre-spill compression: disabled
```

The current `build_shear_index` CLI is persistent-GPU-subtree only. Do not use
the stale `--gpu-subtree-output compact-topk` flag from older notes; current
compact top-k behavior is selected through the GPU-subtree leaf top-k backend,
for example `--gpu-subtree-leaf-topk-backend persistent-wmma-tf32`.

## Deprecated and Diagnostic Command Guard

This runbook is for the current Wiki35M validation path. Do not execute a build
from this runbook if the command contains any retired or diagnostic knob below:

- `--gpu-subtree-output compact-topk`
- `--oom-strict-io` or `--oom-io-mode direct`
- `--oom-raw-vector-cache-gb`
- any `--gpu-subtree-d0-capacity-*` override other than the default
  `--gpu-subtree-d0-capacity-enable`
- `--gpu-subtree-d0-partition-mode tagore-kmeans-top2`
- `--gpu-subtree-root-assignment-backend auto` or
  `--gpu-subtree-root-assignment-backend cuda`
- `--gpu-subtree-assignment-backend persistent-cuvs-bruteforce-fp32`
- `--gpu-subtree-leaf-topk-backend cuvs-bruteforce` or
  `--gpu-subtree-leaf-topk-backend cublas-gemm`
- `--gpu-subtree-assignment-dco-*` or `--gpu-subtree-leaf-dco-*`
- `--gpu-subtree-vector-staging-mode profile-only`,
  `--gpu-subtree-vector-staging-mode fanout-slab`, or
  `--gpu-subtree-vector-staging-mode rolling-cache`
- `SHEAR_PIPNN_WMMA_CTA_SHARED_B=1`
- `SHEAR_PIPNN_COMPACT_LUNE_WITNESS=1`,
  `SHEAR_PIPNN_COMPACT_LUNE_PRUNE=1`, or
  `SHEAR_PIPNN_COMPACT_SPILL_DIGEST=1`
- `SHEAR_PIPNN_PRUNE_BACKEND=lrpf`
- `SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=1`

Any run with one of these settings is a separate diagnostic experiment. It must
not be reported as passing this runbook.

## Diagnostic OOM I/O Allocation Rule

The canonical Wiki35M validation command omits `--oom-memory-budget-gb` and uses
the host/cgroup-aware automatic budget with `--oom-io-mode buffered`. Explicit
budgets remain diagnostic overrides and must be recorded as such.

When memory is not sufficient to keep the SQ+WMMA sidecar fully resident and a
diagnostic run intentionally uses `--oom-io-mode direct`, prioritize reclaimable
OOM memory for the HashPrune reservoir instead of partial SQ sidecar residency.
In this regime, partial SQ sidecar residency still reduces SQ gather time, but
the more important lever is reducing candidate spill and external reduce work.
For a strict direct-I/O sidecar-zero test, use
`--oom-sidecar-resident-budget-cap-gb 0` together with a high reservoir share.

Do not extrapolate this rule to the buffered/page-cache path or to cases where
memory is sufficient to keep the SQ sidecar fully resident. Those cases must be
measured separately because page cache, resident sidecar, and reservoir compete
under different system-memory behavior.

Rule source: the Wiki35M 32 GiB direct sidecar-zero reservoir-share experiment
at
`/data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260702_172251_wiki35m_direct_sidecar0_reservoir_runbook_d0cap`.
The run used runbook D0-capacity semantics with no explicit `--max-leaders`,
`--oom-memory-budget-gb 32`, `--oom-io-mode direct`, SQ+WMMA sidecar rebuild,
and compact witness/lune/spill-digest disabled. `R70a/R70b` drift was small
(`wall_s` drift 0.103%, `stage_total_ms` drift 0.153%), and
`R100_sidecar0a/R100_sidecar0b` drift was also small (`wall_s` drift 0.691%,
`stage_total_ms` drift 0.043%).

Summary of that source experiment:

```text
R70 avg:
  wall_s=2003.425, stage_total=1541.039s
  SQ gather ~=253.7s, spill ~=35.90 GiB, external reduce ~=208.5s

R100 sidecar0 avg:
  wall_s=1975.460, stage_total=1526.039s
  SQ gather ~=365.4s, spill ~=13.68 GiB, external reduce ~=76.5s

Interpretation:
  sidecar0 increased SQ gather by about 111.7s, but reduced external reduce by
  about 132.0s and spill by about 22.2 GiB, for a small net wall-time win and a
  much lower disk-spill footprint.
```

This Wiki35M runbook fixes the build parameter contract below:

```text
OOM memory budget: 200 GiB
leaf_size_min:    256
leaf_size_max:    1280
fanout schedule:  depth 0 = 8, depth 1 = 3, depth >= 2 = 1
D0 capacity:      enabled with default ratios and no explicit root-leader cap override
D0 partition:     gpu-kmeans-coarse
D0 SQ staging:    gpu-d0, upload shard rows = 262144
HashPrune sketch: SQ staging fused sketch enabled; profile must show
                  sq_staging_fused_sketch_enabled=true and sketch_persist_ms=0
D0 DCO/SQ filter: disabled; GPU k-means coarse D0 owns D0 assignment
D1+ SQ+WMMA:      persistent-kernel build with per-build regenerated SQ sidecar
```

In the current parameter model, the requested fanout schedule
`8, 3, 1, 1, 1, ...` is represented by:

```text
--fanout-top 8
--fanout-second 3
```

`PiPNNParams::fanout_for_depth` maps depth `0` to `fanout_top`, depth `1` to
`fanout_second`, and every deeper depth to `1`.

The GPU-D0 path must use `--gpu-subtree-d0-partition-mode gpu-kmeans-coarse`
and `--gpu-subtree-sq-staging-mode gpu-d0`. This means the D0 coarse
assignment and SQ sidecar creation are performed in the GPU-D0 preflight route
before the resident D1+ child loop starts. The legacy
`gpu_subtree.root_assignment_backend` profile field remains `cpu` in this
runbook command; use `gpu_subtree.d0_partition_mode` and
`gpu_subtree.sq_gpu_staging` to verify the actual D0 route.

The default HashPrune sketch path is fused into SQ GPU-D0 staging. There is no
separate CLI flag for this route: the expected profile evidence is
`gpu_subtree.sq_staging_fused_sketch_enabled == true`,
`gpu_subtree.sq_staging_fused_sketch_compute_ms > 0`, and
`serial_timing.rows.sketch_persist_ms == 0`. This means the sketch is still
computed from raw float vectors, but from the same raw host batches already read
for SQ quantization and D0 assignment.

The persistent D1+ GPU planner uses a separate per-node device queue leader slot
limit of `GPU_QUEUE_LEADER_STRIDE = 128`. That `128` cap is not the root D0
leader count. It only bounds the fixed-size D1+ worker scratch layout after
the GPU-D0 k-means coarse assignment has selected the Wiki35M root leaders.

The D0 capacity path must remain enabled in the default command. Do not add an
explicit root-leader cap; let the capacity planner choose the root leader count
from the resident admission budget, then let lazy D1+ admission split repair
oversized children before they reach the resident backend.

After the build succeeds, validate the generated `_mem.index` with
`search_shear_index` against the canonical Wiki35M query and full-35M ground
truth. After search validation and result summarization are complete, remove
the generated index artifact files for the run. Keep logs, profiles, command
records, summaries, search outputs, and the recorded `_mem.index` byte size.

## Required Inputs

Repository root:

```bash
cd /path/to/SHEAR
```

Canonical data root:

```text
/data_nvme_kioxia/tyj/data
```

Measurement output root:

```text
/data_staging/wbh
```

Full Wiki35M base, query, and ground truth:

```text
/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
/data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin
```

Canonical Wiki35M measurement files:

```bash
export DATA_ROOT=/data_nvme_kioxia/tyj/data
export STAGING_ROOT=/data_staging/wbh
export WIKI35M_DIR="${DATA_ROOT}/wiki-1024-35M"
export WIKI35M_BASE="${WIKI35M_DIR}/en_base.fbin"
export WIKI35M_QUERY_PATH="${WIKI35M_DIR}/query.fbin"
export WIKI35M_GT_PATH="${WIKI35M_DIR}/gt100.bin"
```

The Wiki35M base must be an `.fbin` file containing 35,000,000 `f32` vectors of
dimension 1024. The query file must contain 10,000 `f32` vectors of dimension
1024. The GT file must be generated against the full Wiki35M base, not against
a subset.

## Input Sanity Checks

Fail fast before starting the expensive graph build:

```bash
test -s "${WIKI35M_BASE}"
test -s "${WIKI35M_QUERY_PATH}"
test -s "${WIKI35M_GT_PATH}"
od -An -tu4 -N 8 "${WIKI35M_BASE}"
od -An -tu4 -N 8 "${WIKI35M_QUERY_PATH}"
od -An -tu4 -N 8 "${WIKI35M_GT_PATH}"
stat -c "%n %s" "${WIKI35M_BASE}" "${WIKI35M_QUERY_PATH}" "${WIKI35M_GT_PATH}"
```

Expected headers and file sizes:

```text
WIKI35M_BASE:       35000000 1024    size 143360000008
WIKI35M_QUERY_PATH:    10000 1024    size 40960008
WIKI35M_GT_PATH:       10000 100     size 4000008 or 8000008
```

The known Wiki35M GT on this host is expected to be DiskANN ids+distances
format with size `8000008` bytes.

Verify the maximum GT id is in range:

```bash
python3 - <<'PY'
import os
import struct
import array

path = os.environ["WIKI35M_GT_PATH"]
size = os.path.getsize(path)
with open(path, "rb") as f:
    n, d = struct.unpack("<II", f.read(8))
    assert (n, d) == (10_000, 100), (n, d)
    ids = array.array("I")
    ids.fromfile(f, n * d)
    if ids.itemsize != 4:
        raise RuntimeError("unexpected u32 item size")
    max_id = max(ids)
    assert max_id < 35_000_000, max_id
    expected_ids_only = 8 + n * d * 4
    expected_with_dists = 8 + n * d * 8
    assert size in (expected_ids_only, expected_with_dists), (size, expected_ids_only, expected_with_dists)
    print(f"gt ids ok: n={n} d={d} max_id={max_id} size={size}")
PY
```

On non-little-endian machines, replace the quick `array.array("I")` check with
an explicit little-endian `struct.iter_unpack("<I", ...)` parser. The benchmark
hosts used for this project are little-endian Linux machines.

## Output Layout

Use one fresh run directory per measurement:

```bash
export RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="${STAGING_ROOT}/wiki35m_gpu_subtree_measurement/runs/${RUN_ID}"
export INDEX_DIR="${RUN_ROOT}/index"
export OOM_DIR="${RUN_ROOT}/oom"
export PROFILE_JSON="${RUN_ROOT}/profile.json"
export BUILD_STDOUT="${RUN_ROOT}/stdout.log"
export BUILD_STDERR="${RUN_ROOT}/stderr.log"
export TIME_LOG="${RUN_ROOT}/time.log"
export DRIVER_TIMING_JSON="${RUN_ROOT}/driver_timing.json"
export GPU_MONITOR_DIR="${RUN_ROOT}/gpu_monitor"
export SQ_WMMA_SIDECAR_DIR="${RUN_ROOT}/sq_sidecar"
export SQ_WMMA_SIDECAR_PREFIX="${SQ_WMMA_SIDECAR_DIR}/en_base.sq8"
export SEARCH_DIR="${RUN_ROOT}/search"
export SEARCH_SWEEP_TSV="${RUN_ROOT}/search_sweep.tsv"
mkdir -p "${INDEX_DIR}" "${OOM_DIR}" "${GPU_MONITOR_DIR}" "${SEARCH_DIR}" "${SQ_WMMA_SIDECAR_DIR}"
```

Expected files after a complete run:

```text
${RUN_ROOT}/metadata.txt
${RUN_ROOT}/stdout.log
${RUN_ROOT}/stderr.log
${RUN_ROOT}/time.log
${RUN_ROOT}/driver_timing.json
${RUN_ROOT}/gpu_monitor/gpu_samples.csv
${RUN_ROOT}/gpu_monitor/gpu_processes.csv
${RUN_ROOT}/gpu_monitor/summary.txt
${RUN_ROOT}/sq_sidecar/
${RUN_ROOT}/profile.json
${RUN_ROOT}/index/_mem.index
${RUN_ROOT}/oom/
${RUN_ROOT}/search/
${RUN_ROOT}/search/search_l*.log
${RUN_ROOT}/search/search_l*_results.fbin
${RUN_ROOT}/search_sweep.tsv
${RUN_ROOT}/summary.md
```

`${RUN_ROOT}/index/_mem.index` is a temporary validation artifact. It must exist
through post-build checks and mem-search validation, but it should not be kept
after the summary records its byte size and the search result has been written.

Write basic provenance before running:

```bash
{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_short_begin"
  git status --short
  echo "git_status_short_end"
  echo "base=${WIKI35M_BASE}"
  echo "query=${WIKI35M_QUERY_PATH}"
  echo "gt=${WIKI35M_GT_PATH}"
  echo "data_root=${DATA_ROOT}"
  echo "staging_root=${STAGING_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "gpu_monitor_dir=${GPU_MONITOR_DIR}"
  echo "gpu_monitor_script=script/monitor_gpu_build.sh"
  echo "driver_timing_json=${DRIVER_TIMING_JSON}"
  echo "sq_wmma_sidecar_prefix=${SQ_WMMA_SIDECAR_PREFIX}"
  echo "default_path=gpu_kmeans_coarse_d0+gpu_d0_sq_sidecar+sq_staging_fused_sketch+fused_persistent_sq_wmma+d1_assignment_persistent_wmma_tf32+leaf_persistent_wmma_tf32"
  echo "oom_io_mode=buffered"
  echo "oom_memory_budget=auto"
  echo "leaf_size_min=256"
  echo "leaf_size_max=1280"
  echo "fanout_schedule_depth0_depth1_depth2plus=8,3,1"
  echo "d0_capacity_enable=true"
  echo "d0_partition_mode=gpu-kmeans-coarse"
  echo "sq_staging_mode=gpu-d0"
  echo "sq_staging_upload_shard_rows=262144"
  echo "sq_staging_fused_sketch_expected=true"
  echo "d0_dco_mode=none"
  echo "d0_dco_survivors=0"
  echo "gpu_subtree_execution_mode=fused-persistent"
  echo "sq_wmma_mode=build"
  echo "sq_wmma_sidecar_mode=rebuild"
  echo "root_risk_candidate_rs=1,2,4,8"
  echo "leaf_wmma_topk_forced=SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1"
  echo "wmma_cta_shared_b=SHEAR_PIPNN_WMMA_CTA_SHARED_B=0"
  echo "compact_lune_witness=SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0"
  echo "compact_lune_prune=SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0"
  echo "compact_spill_digest=SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0"
  echo "prune_backend=SHEAR_PIPNN_PRUNE_BACKEND=hashprune"
  echo "lrpf_pre_spill_compress=SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0"
  nvidia-smi || true
} > "${RUN_ROOT}/metadata.txt" 2>&1
```

## Build Binaries

Build the full graph-construction driver with CUDA support and the in-memory
search validator:

```bash
cargo build --release \
  -p build_shear_index \
  -p search_shear_index \
  --features build_shear_index/cuda
```

If CUDA support is not available, do not use this runbook to claim
GPU-subtree/resident optimization results.

## SQ+WMMA Sidecar Lifecycle

Do not prepare or reuse a shared SQ+WMMA sidecar before the graph build. The
timed build must run with `--gpu-subtree-sq-staging-mode gpu-d0` and
`--gpu-subtree-sq-wmma-sidecar-mode rebuild`, generate the SQ sidecar inside
the GPU-D0 staging path, consume it for D0 assignment and D1+ SQ+WMMA work in the same
process, and delete `${SQ_WMMA_SIDECAR_PREFIX}.codes` and
`${SQ_WMMA_SIDECAR_PREFIX}.meta` before the driver exits.
On the default HashPrune path, the same SQ GPU-D0 staging pass should also
compute the resident HashPrune sketch from the raw host buffers, so a separate
standalone sketch persist stage is not expected.

## Full 35M Graph Build

Run the full GPU-subtree persistent compact-topk build. The build command is
wrapped with `script/monitor_gpu_build.sh` so GPU utilization, memory, power,
temperature, and compute-process samples are recorded under
`${GPU_MONITOR_DIR}` while preserving the wrapped command exit code:

```bash
bash script/monitor_gpu_build.sh \
  --gpu 0 \
  --interval 1 \
  --out-dir "${GPU_MONITOR_DIR}" \
  --tag wiki35m_gpu_subtree_build \
  -- \
  bash -lc '
    set -euo pipefail
    SHEAR_PIPNN_PROFILE=1 \
    SHEAR_PIPNN_PHASE_MONITOR=1 \
    SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 \
    SHEAR_PIPNN_WMMA_CTA_SHARED_B=0 \
    SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0 \
    SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0 \
    SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0 \
    SHEAR_PIPNN_PRUNE_BACKEND=hashprune \
    SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIME_LOG}" \
    target/release/build_shear_index \
      --data-type float \
      --dist-fn l2 \
      --data-path "${WIKI35M_BASE}" \
      --index-path-prefix "${INDEX_DIR}" \
      --overwrite \
      --oom-enable \
      --oom-io-mode buffered \
      --oom-temp-dir "${OOM_DIR}" \
      --oom-profile-json "${PROFILE_JSON}" \
      --driver-timing-json "${DRIVER_TIMING_JSON}" \
      --max-degree 64 \
      --lbuild 100 \
      --num-threads 8 \
      --leaf-size-min 256 \
      --leaf-size-max 1280 \
      --leaf-knn 2 \
      --psamp-fraction 0.05 \
      --fanout-top 8 \
      --fanout-second 3 \
      --root-risk-candidate-rs 1,2,4,8 \
      --seed 42 \
      --partitioner gpu-subtree \
      --gpu-subtree-root-assignment-backend cpu \
      --gpu-subtree-assignment-backend persistent-wmma-tf32 \
      --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 \
      --gpu-subtree-execution-mode fused-persistent \
      --gpu-subtree-d0-capacity-enable \
      --gpu-subtree-d0-partition-mode gpu-kmeans-coarse \
      --gpu-subtree-sq-staging-mode gpu-d0 \
      --gpu-subtree-sq-staging-upload-shard-rows 262144 \
      --gpu-subtree-sq-wmma-mode build \
      --gpu-subtree-sq-wmma-sidecar-prefix "${SQ_WMMA_SIDECAR_PREFIX}" \
      --gpu-subtree-sq-wmma-sidecar-mode rebuild \
      --pipnn-backend cuda \
      --pipnn-gpu-device 0 \
      --pipnn-gpu-workspace-budget-gb 16 \
      > "${BUILD_STDOUT}" \
      2> "${BUILD_STDERR}"
  '
```

Important parameter notes:

- Do not add `--auto-params`; it overrides the manual leaf and fanout settings.
- Keep `--oom-io-mode buffered`. Do not use `--oom-strict-io` or
  `--oom-io-mode direct` in this runbook; those settings are retired as
  canonical validation defaults.
- Keep `--gpu-subtree-root-assignment-backend cpu`. This is the current default
  path for D0/root assignment; do not switch it to CUDA unless the experiment is
  explicitly measuring D0 GPU assignment.
- Keep `--gpu-subtree-d0-capacity-enable` and do not set an explicit
  root-leader cap. Capacity-aware D0 chooses the root leader count from
  resident admission pressure; lazy D1+ admission split is the safety repair for
  oversized children.
- Keep `--gpu-subtree-d0-partition-mode gpu-kmeans-coarse`,
  `--gpu-subtree-sq-staging-mode gpu-d0`, and
  `--gpu-subtree-sq-staging-upload-shard-rows 262144`. This is the canonical
  GPU-D0 sidecar route: D0 coarse assignment and SQ sidecar creation are done in
  the same GPU-D0 preflight path before D1+ resident processing.
- Keep the default SQ-staging fused HashPrune sketch path. It is selected by the
  HashPrune + GPU-subtree + `gpu-d0` SQ staging combination, not by an explicit
  command-line flag. Treat the run as mismatched if
  `gpu_subtree.sq_staging_fused_sketch_enabled` is not `true`, if
  `gpu_subtree.sq_staging_fused_sketch_compute_ms` is `0`, or if
  `serial_timing.rows.sketch_persist_ms` is nonzero.
- Do not enable D0 DCO/SQ filter in this runbook. Current CLI validation
  supports D0 DCO only for `rbc-fanout`, while this default uses
  `gpu-kmeans-coarse`; the GPU-D0 SQ sidecar is for GPU-D0 assignment staging
  and D1+ SQ+WMMA reuse, not the old D0 DCO filter.
- Keep both `--gpu-subtree-assignment-backend persistent-wmma-tf32` and
  `--gpu-subtree-leaf-topk-backend persistent-wmma-tf32`. This records the
  current fastest resident GPU-subtree path measured for the D1 assignment and
  leaf/top-k workers. The selected leaf backend is still
  `persistent-wmma-tf32`; this Wiki35M validation forces the leaf WMMA top-k
  subpath with `SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1` while keeping per-warp-B
  staging fixed. If it fails with a CUDA illegal address, record the failed
  child/stage and report the run as a failed validation rather than a completed
  build.
- Keep `SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1` and
  `SHEAR_PIPNN_WMMA_CTA_SHARED_B=0` for canonical measurement runs. The
  accepted default WMMA B-tile staging is the per-warp-B path. Set
  `SHEAR_PIPNN_WMMA_CTA_SHARED_B=1` only for an explicit CTA-shared-B
  diagnostic or sensitivity run, and report it as a different operator variant.
- Keep `--gpu-subtree-execution-mode fused-persistent`,
  `--gpu-subtree-sq-wmma-mode build`, and
  `--gpu-subtree-sq-wmma-sidecar-mode rebuild`. D1+ should stay on the
  persistent kernel. The SQ-int8 WMMA sidecar must be generated inside the
  timed GPU-D0 route, consumed by id for that build only, and deleted by the
  driver before exit; do not point this prefix at a shared reusable sidecar.
- Do not add raw-vector cache, assignment/leaf DCO, vector-staging flags, D0
  DCO/SQ filter flags, D0 partition modes other than `gpu-kmeans-coarse`, SQ
  staging modes other than `gpu-d0`, or custom
  `--gpu-subtree-d0-capacity-*` thresholds to this canonical command. Those
  remain separate one-variable A/B runs.
- Keep `--root-risk-candidate-rs 1,2,4,8` with `--fanout-top 8`. The current CLI
  parses root-risk candidates even when the fixed root fanout policy is used,
  and the default candidate list contains `10`, which is invalid when
  `fanout_top` is `8`.
- `--leaf-size-max 1280` is the semantic persistent leaf cap. For 35M,
  `adaptive_c_max` may be logged as `2560` because large datasets double the
  partitioning cap, but the persistent GPU profile should still report
  `persistent_leaf_max = 1280` and `persistent_leaf_size_max <= 1280`.
- The default workspace budget here is 16 GiB to match the recent GPU-subtree
  run family. If an experiment intentionally changes this value,
  record the reason in `metadata.txt` and do not compare it as a pure code
  optimization against runs with a different workspace budget.

## Post-Build Sanity Checks

The build must satisfy all checks below before running search validation:

```bash
test -s "${INDEX_DIR}/_mem.index"
test -s "${PROFILE_JSON}"
test -s "${DRIVER_TIMING_JSON}"
grep -q "PiPNN: index build completed successfully" "${BUILD_STDOUT}"
test ! -e "${SQ_WMMA_SIDECAR_PREFIX}.codes"
test ! -e "${SQ_WMMA_SIDECAR_PREFIX}.meta"
grep -q "Maximum resident set size" "${TIME_LOG}"
```

Record these values in the result summary:

- build exit code
- wall-clock time from `time.log`
- maximum resident set size from `time.log`
- `_mem.index` byte size
- `profile.json` path
- `driver_timing.json` path, including `total_wall_ms`,
  `build_call_wall_ms`, `profile_stage_total_ms`,
  `total_minus_profile_stage_ms`, `build_call_minus_profile_stage_ms`, and
  phase rows such as `startup`, `load_data`, `build_graph`, and `save_graph`
- SQ+WMMA sidecar cleanup status; `${SQ_WMMA_SIDECAR_PREFIX}.codes` and
  `${SQ_WMMA_SIDECAR_PREFIX}.meta` must not exist after the build command exits
- resolved `memory_budget_bytes` from `profile.json`
- `oom_io_mode` from `profile.json`; it should be `buffered`
- `partitioner.kind` from `profile.json`; it should be `gpu-subtree`
- `root_fanout.fixed_fanout` from `profile.json`; it should be `8`
- `gpu_subtree.root_assignment_backend` from `profile.json`; it should be `cpu`
- `gpu_subtree.d0_partition_mode` from `profile.json`; it should be
  `gpu-kmeans-coarse`
- `gpu_subtree.sq_staging_mode` and `gpu_subtree.sq_gpu_staging` from
  `profile.json`; they should confirm `gpu-d0` staging with
  `upload_shard_rows == 262144`
- `gpu_subtree.sq_staging_fused_sketch_enabled`,
  `gpu_subtree.sq_staging_fused_sketch_compute_ms`, and
  `serial_timing.rows.sketch_persist_ms` from `profile.json`; they should
  confirm fused raw-buffer sketch computation with standalone sketch persist
  skipped
- `gpu_subtree.assignment_backend` from `profile.json`; it should be
  `persistent-wmma-tf32`
- `gpu_subtree.subtree_assignment_backend` from `profile.json`; it should be
  `persistent-wmma-tf32`
- `gpu_subtree.leaf_topk_backend` from `profile.json`; it should be
  `persistent-wmma-tf32`
- `gpu_subtree.persistent_effective_root_fanout` from `profile.json`; it should
  be `3` for resident subtrees
- `gpu_subtree.persistent_effective_deeper_fanout` from `profile.json`; it
  should be `1`
- `gpu_subtree.persistent_leaf_max` and `gpu_subtree.persistent_leaf_size_max`
- `memory_plan.raw_vector_cache_budget_bytes` and
  `raw_vector_cache.enabled` from `profile.json`; they should confirm that the
  retired raw-vector cache is disabled
- `gpu_subtree.assignment_dco_mode`, `gpu_subtree.leaf_dco_mode`,
  `gpu_subtree.d0_dco_mode`, `gpu_subtree.d0_dco_survivors`, and
  `gpu_subtree.d0_capacity.enabled`; they should confirm that assignment/leaf
  DCO and D0 DCO are disabled, and D0 capacity is enabled
- `gpu_subtree.sq_wmma_mode`, `gpu_subtree.sq_wmma_sidecar_mode`,
  `gpu_subtree.sq_wmma_kernel_event_ms`, and
  `gpu_subtree.persistent_kernel_event_ms`; they should confirm that the D1+
  SQ+WMMA persistent-kernel path ran
- `serial_timing` from `profile.json`; `schema` should be
  `gpu_subtree_serial_nonoverlap_v1` and `accounted_sum_ms` should match
  `stage_total_ms`
- resident-subtree, GPU, pipeline, and leaf-flow timing fields if present

Recommended extraction commands:

```bash
stat -c "mem_index_bytes=%s" "${INDEX_DIR}/_mem.index"
grep -E "Elapsed \\(wall clock\\)|Maximum resident set size" "${TIME_LOG}"
rg -n '"oom_io_mode"|"memory_budget_bytes"|"memory_plan"|"raw_vector_cache"|"partitioner"|"root_fanout"|"root_assignment_backend"|"d0_partition_mode"|"sq_staging_mode"|"sq_gpu_staging"|"sq_staging_fused_sketch_enabled"|"sq_staging_fused_sketch_compute_ms"|"subtree_assignment_backend"|"assignment_backend"|"leaf_topk_backend"|"assignment_dco_mode"|"leaf_dco_mode"|"d0_dco_mode"|"d0_dco_rerank_mode"|"d0_dco_survivors"|"d0_sq_cpu_kernel"|"d0_dco_sidecar_gather_ms"|"d0_dco_approx_ms"|"d0_capacity"|"sq_wmma_mode"|"sq_wmma_sidecar_mode"|"sq_wmma_kernel_event_ms"|"persistent_kernel_event_ms"|"persistent_partition_frontier_end"|"persistent_partition_level_rounds"|"persistent_effective_root_fanout"|"persistent_effective_deeper_fanout"|"persistent_leaf_max"|"persistent_leaf_size_max"|"sketch_persist_ms"|"serial_timing"|"resident_subtree"|"gpu_subtree"|"pipeline"|"leaf_flow"' "${PROFILE_JSON}"
```

Validate the persistent device-side producer-consumer profile before reporting
the build as a Wiki35M GPU-subtree/resident run:

```bash
python3 - <<'PY'
import json
import os

profile_path = os.environ["PROFILE_JSON"]
with open(profile_path, "r", encoding="utf-8") as f:
    profile = json.load(f)

gpu = profile["gpu_subtree"]
root = profile.get("root_fanout", {})
serial = profile["serial_timing"]

assert profile["memory_budget_bytes"] == 214_748_364_800, profile["memory_budget_bytes"]
assert profile["oom_io_mode"] == "buffered", profile["oom_io_mode"]
assert profile["memory_plan"]["raw_vector_cache_budget_bytes"] == 0, profile["memory_plan"]
assert profile["raw_vector_cache"]["enabled"] is False, profile["raw_vector_cache"]
assert profile["partitioner"]["kind"] == "gpu-subtree", profile["partitioner"]
assert root.get("fixed_fanout") == 8, root
assert gpu["enabled"] is True, gpu
assert gpu["root_assignment_backend"] == "cpu", gpu["root_assignment_backend"]
assert gpu["d0_partition_mode"] == "gpu-kmeans-coarse", gpu["d0_partition_mode"]
assert gpu["sq_staging_mode"] == "gpu-d0", gpu["sq_staging_mode"]
sq_gpu = gpu["sq_gpu_staging"]
assert sq_gpu["enabled"] is True, sq_gpu
assert sq_gpu["mode"] == "gpu-d0", sq_gpu
assert sq_gpu["upload_shard_rows"] == 262_144, sq_gpu
assert gpu["sq_staging_fused_sketch_enabled"] is True, gpu["sq_staging_fused_sketch_enabled"]
assert gpu["sq_staging_fused_sketch_compute_ms"] > 0, gpu["sq_staging_fused_sketch_compute_ms"]
assert serial["rows"]["sketch_persist_ms"] == 0, serial["rows"]
assert gpu["assignment_backend"] == "persistent-wmma-tf32", gpu["assignment_backend"]
assert gpu["subtree_assignment_backend"] == "persistent-wmma-tf32", gpu["subtree_assignment_backend"]
assert gpu["leaf_topk_backend"] == "persistent-wmma-tf32", gpu["leaf_topk_backend"]
assert gpu["assignment_dco_mode"] == "none", gpu["assignment_dco_mode"]
assert gpu["leaf_dco_mode"] == "none", gpu["leaf_dco_mode"]
assert gpu["d0_capacity"]["enabled"] is True, gpu["d0_capacity"]
assert gpu["d0_dco_mode"] == "none", gpu["d0_dco_mode"]
assert gpu["d0_dco_survivors"] == 0, gpu["d0_dco_survivors"]
assert gpu["d0_dco_sidecar_gather_ms"] == 0, gpu["d0_dco_sidecar_gather_ms"]
assert gpu["d0_dco_approx_ms"] == 0, gpu["d0_dco_approx_ms"]
assert gpu["sq_wmma_mode"] == "build", gpu["sq_wmma_mode"]
assert gpu["sq_wmma_sidecar_mode"] == "rebuild", gpu["sq_wmma_sidecar_mode"]
assert gpu["sq_wmma_sidecar_gather_ms"] > 0, gpu["sq_wmma_sidecar_gather_ms"]
assert gpu["sq_wmma_kernel_event_ms"] > 0, gpu["sq_wmma_kernel_event_ms"]
assert gpu["persistent_kernel_event_ms"] > 0, gpu["persistent_kernel_event_ms"]
assert gpu["persistent_success_count"] > 0, gpu
assert gpu["persistent_queue_overflow_flags"] == 0, gpu["persistent_queue_overflow_flags"]
assert gpu["persistent_partition_claims"] == gpu["persistent_partition_pops"], gpu
assert gpu["persistent_topk_claims"] == gpu["persistent_topk_pops"], gpu
assert gpu["persistent_leaf_pushes"] == gpu["persistent_topk_claims"], gpu
assert gpu["persistent_large_node_splits"] > 0, gpu
assert gpu["persistent_effective_root_fanout"] == 3, gpu["persistent_effective_root_fanout"]
assert gpu["persistent_effective_deeper_fanout"] == 1, gpu["persistent_effective_deeper_fanout"]
assert gpu["persistent_leaf_max"] == 1280, gpu["persistent_leaf_max"]
assert gpu["persistent_leaf_size_max"] <= 1280, gpu["persistent_leaf_size_max"]
assert gpu["persistent_partition_worker_blocks"] > 1 or gpu["persistent_topk_worker_blocks"] > 1, gpu
assert serial["schema"] == "gpu_subtree_serial_nonoverlap_v1", serial
assert serial["accounted_sum_ms"] == serial["stage_total_ms"], serial
print("Wiki35M persistent GPU-subtree profile checks passed")
PY
```

If these assertions fail, keep all logs and profile files, mark the run as
failed, and investigate before running mem-search validation.

## Serial Non-Overlap Timing Metric

Use `profile.json.serial_timing` for bottleneck ranking. The
`serial_timing.rows` values are the default serial, non-overlapping timing
addends for this runbook, and `serial_timing.accounted_sum_ms` must equal
`serial_timing.stage_total_ms` before reporting a bottleneck table. Repeated
D1+ child-loop work, including visible sidecar wait, payload H2D, persistent
kernel, and result scatter, is already accumulated into the corresponding D1+
rows.

Use `driver_timing.json` for whole-build wall reconciliation. Its
`profile_stage_total_ms` is copied from `profile.json.serial_timing`, while
`total_minus_profile_stage_ms` and `build_call_minus_profile_stage_ms` show the
outer wall that is not inside the serial stage. Do not rank bottlenecks from
driver phases alone; use them to explain the gap between `/usr/bin/time`,
driver wall, and profile stage.

Interpret the rows as:

- `d0_root_assignment_ms` is the unsplit D0 route only when no GPU-D0 subrows
  are available. GPU-D0 paths expand it into
  `d0_kmeans_train_ms`, `d0_kmeans_assign_ms`, `d0_sq_raw_read_ms`,
  `d0_sq_raw_h2d_ms`, `d0_sq_quantize_ms`, `d0_sq_d0_assign_ms`,
  `d0_sq_qcode_write_ms`, and `d0_residual_ms`. These subrows are
  serial-visible slices clipped to `gpu_subtree.d0_wall_ms`; they are not the
  sum of every nested SQ staging activity when the staging pipeline overlaps.
  Use `serial_timing.diagnostics.d0_root_assignment_total_ms`,
  `serial_timing.diagnostics.d0_nested_requested_total_ms`,
  `serial_timing.diagnostics.d0_nested_overlap_hidden_ms`, and
  `gpu_subtree.sq_gpu_staging.*` for nested D0 diagnostics.
- `d1plus_sidecar_wait_ms`, `d1plus_payload_h2d_ms`,
  `d1plus_persistent_kernel_ms`, `d1plus_cpu_result_scatter_ms`,
  `d1plus_child_replay_ms`, `d1plus_lazy_split_ms`, and
  `d1plus_push_wait_ms`: accumulated D1+ repeated child-loop work. Full
  sidecar gather time is reported as
  `serial_timing.diagnostics.d1plus_sidecar_gather_total_ms`; hidden preload
  overlap must not be added to the serial rows.
- `d1plus_backend_residual_ms`, `d1plus_producer_residual_ms`,
  `streaming_residual_ms`, and `stage_outer_residual_ms`: residual boundaries
  that make the serial rows sum back to `stage_total_ms`. In particular,
  `d1plus_producer_residual_ms` is the remaining producer time after visible
  sidecar wait, backend wall, child replay, lazy split, and queue push wait have
  been subtracted.
- `serial_timing.diagnostics`: nested or overlap-only counters such as
  `cpu_compact_flush_ms`, `edge_sink_ms`, and sidecar hidden/preload overlap.
  They are useful for tuning, but must not be added to the serial rows.

## Mem Search Validation

Search the generated `_mem.index` with the mmap search validator across a sweep
of search list sizes. The search phase must report a recall/QPS pair for every
`SEARCH_L` so we can compare quality-throughput curves, not just one operating
point. Keep `L=100` in the sweep because it is the historical comparability
point, and use the same default sweep as the Wiki10M runbook.

```bash
export RECALL_AT=10
export SEARCH_THREADS=8
export SEARCH_L_VALUES="${SEARCH_L_VALUES:-25 50 100 200 400}"

printf "search_l\trecall_at\tqueries\twall_s\tqps\trecall\tresult_bytes\tlog\tresult\n" \
  > "${SEARCH_SWEEP_TSV}"

for SEARCH_L in ${SEARCH_L_VALUES}; do
  export SEARCH_L
  export SEARCH_LOG="${SEARCH_DIR}/search_l${SEARCH_L}.log"
  export SEARCH_RESULTS="${SEARCH_DIR}/search_l${SEARCH_L}_results.fbin"

  target/release/search_shear_index \
    --index-path-prefix "${INDEX_DIR}" \
    --data-path "${WIKI35M_BASE}" \
    --query-file "${WIKI35M_QUERY_PATH}" \
    --gt-file "${WIKI35M_GT_PATH}" \
    --result-path "${SEARCH_RESULTS}" \
    --recall-at "${RECALL_AT}" \
    --search-list-size "${SEARCH_L}" \
    --num-threads "${SEARCH_THREADS}" \
    --storage mmap \
    > "${SEARCH_LOG}" \
    2>&1

  export SEARCH_RESULT_BYTES
  SEARCH_RESULT_BYTES="$(stat -c "%s" "${SEARCH_RESULTS}")"
  python3 - <<'PY' >> "${SEARCH_SWEEP_TSV}"
import os
import re

log_path = os.environ["SEARCH_LOG"]
with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

search = re.search(r"Search:\s+(\d+)\s+queries in\s+([0-9.]+)s\s+\(([0-9.]+)\s+QPS\)", text)
recall = re.search(r"Recall@(\d+):\s+([0-9.]+)", text)
if search is None or recall is None:
    raise SystemExit(f"failed to parse search metrics from {log_path}")

print(
    "\t".join(
        [
            os.environ["SEARCH_L"],
            recall.group(1),
            search.group(1),
            search.group(2),
            search.group(3),
            recall.group(2),
            os.environ["SEARCH_RESULT_BYTES"],
            log_path,
            os.environ["SEARCH_RESULTS"],
        ]
    )
)
PY
done
```

Record these search fields:

- query count and dimension
- `SEARCH_L_VALUES`
- `RECALL_AT`
- `SEARCH_THREADS`
- for each `SEARCH_L`: search wall time, QPS, Recall@10, result-file byte size
- path to `${SEARCH_SWEEP_TSV}` and all per-L logs/results
- the `L=100` row, which remains the default comparison point
- the best recall point, best QPS point, and any recall/QPS knee observed in the
  sweep

Useful extraction commands:

```bash
cat "${SEARCH_SWEEP_TSV}"
for log in "${SEARCH_DIR}"/search_l*.log; do
  echo "== ${log} =="
  grep -E "Base data:|Graph:|Queries:|Search:|Recall@" "${log}"
done
```

## Required Pass Criteria

A measurement passes only if all items below are true:

- The build command exits with status 0.
- `${INDEX_DIR}/_mem.index` exists and is non-empty.
- `${PROFILE_JSON}` exists and records the expected 200 GiB OOM memory budget.
- `${PROFILE_JSON}` records `oom_io_mode == "buffered"`.
- `${PROFILE_JSON}` records `partitioner.kind == "gpu-subtree"`.
- `${PROFILE_JSON}` records `root_fanout.fixed_fanout == 8`.
- `${PROFILE_JSON}` records `gpu_subtree.root_assignment_backend == "cpu"`.
- `${PROFILE_JSON}` records
  `gpu_subtree.d0_partition_mode == "gpu-kmeans-coarse"`.
- `${PROFILE_JSON}` records `gpu_subtree.sq_staging_mode == "gpu-d0"` and
  `gpu_subtree.sq_gpu_staging.upload_shard_rows == 262144`.
- `${PROFILE_JSON}` records
  `gpu_subtree.sq_staging_fused_sketch_enabled == true`,
  `gpu_subtree.sq_staging_fused_sketch_compute_ms > 0`, and
  `serial_timing.rows.sketch_persist_ms == 0`.
- `${PROFILE_JSON}` records
  `gpu_subtree.assignment_backend == "persistent-wmma-tf32"`.
- `${PROFILE_JSON}` records
  `gpu_subtree.subtree_assignment_backend == "persistent-wmma-tf32"`.
- `${PROFILE_JSON}` records
  `gpu_subtree.leaf_topk_backend == "persistent-wmma-tf32"`.
- `${PROFILE_JSON}` records
  `memory_plan.raw_vector_cache_budget_bytes == 0` and
  `raw_vector_cache.enabled == false`.
- `${PROFILE_JSON}` records `gpu_subtree.assignment_dco_mode == "none"` and
  `gpu_subtree.leaf_dco_mode == "none"`.
- `${PROFILE_JSON}` records `gpu_subtree.d0_capacity.enabled == true`,
  `gpu_subtree.d0_dco_mode == "none"`,
  `gpu_subtree.d0_dco_survivors == 0`,
  `gpu_subtree.d0_dco_sidecar_gather_ms == 0`, and
  `gpu_subtree.d0_dco_approx_ms == 0`.
- `${PROFILE_JSON}` records `gpu_subtree.sq_wmma_mode == "build"`,
  `gpu_subtree.sq_wmma_sidecar_mode == "rebuild"`, and
  `gpu_subtree.sq_wmma_sidecar_gather_ms > 0`.
- `${PROFILE_JSON}` records `gpu_subtree.sq_wmma_kernel_event_ms > 0`
  and `gpu_subtree.persistent_kernel_event_ms > 0`.
- `${PROFILE_JSON}` records `gpu_subtree.persistent_effective_root_fanout == 3`.
- `${PROFILE_JSON}` records `gpu_subtree.persistent_effective_deeper_fanout == 1`.
- `${PROFILE_JSON}` records `gpu_subtree.persistent_leaf_max == 1280`.
- `${PROFILE_JSON}` records `gpu_subtree.persistent_leaf_size_max <= 1280`.
- `${PROFILE_JSON}` records `serial_timing.schema ==
  "gpu_subtree_serial_nonoverlap_v1"` and
  `serial_timing.accounted_sum_ms == serial_timing.stage_total_ms`.
- `${PROFILE_JSON}` records `persistent_partition_worker_blocks > 1` or
  `persistent_topk_worker_blocks > 1`.
- `${PROFILE_JSON}` records
  `persistent_partition_claims == persistent_partition_pops`.
- `${PROFILE_JSON}` records `persistent_topk_claims == persistent_topk_pops`.
- `${PROFILE_JSON}` records `persistent_queue_overflow_flags == 0`.
- `${PROFILE_JSON}` records `persistent_large_node_splits > 0`.
- The build stdout contains `PiPNN: index build completed successfully`.
- Every search command in `SEARCH_L_VALUES` exits with status 0.
- Search uses query vectors and GT generated for the full Wiki35M base file.
- `${SEARCH_SWEEP_TSV}` exists and records one row per `SEARCH_L`, plus a
  header.
- The search sweep includes `L=100` for historical comparability and should
  include at least one lower-throughput/faster point such as `L=50` plus one
  higher-recall/slower point such as `L=200` or above.
- For each `SEARCH_L`, `Recall@10` is at least the current accepted baseline for
  that same `SEARCH_L`, or any drop is explicitly called out as a regression
  requiring investigation.
- QPS and Recall@10 are recorded together for every row; do not report one
  without the other.
- The generated index artifact files are cleaned after search validation and
  summary generation, or the summary explicitly states why they were retained.

Until numeric Wiki35M baselines are established for this exact query/GT pair,
use the previous accepted run with the same base, query, GT, `SEARCH_L`,
`RECALL_AT`, build parameters, GPU workspace budget, and GPU-subtree backend
settings as the comparison baseline for each matching sweep row. Do not compare
`L=50` QPS/recall against an `L=100` baseline as if they were the same operating
point.

## Result Summary Template

Create or append a short summary after every run. Store it near the run output,
for example `${RUN_ROOT}/summary.md`:

```md
# Wiki35M GPU-Subtree Measurement

- Run ID:
- Date:
- Git commit:
- Code changes under test:
- Base:
- Query:
- GT:
- Build command changed from runbook: yes/no
- Search command changed from runbook: yes/no
- Default path: GPU k-means coarse D0 + GPU-D0 SQ sidecar + SQ-staging fused
  HashPrune sketch + resident GPU D1+/leaf with persistent-wmma-tf32

## Build

- Exit code:
- Wall time:
- Max RSS:
- `_mem.index` bytes:
- OOM memory budget bytes:
- GPU workspace GiB:
- Leaf size min/max:
- D0 partition/staging:
- SQ-staging fused sketch:
- Fanout schedule:
- Root risk candidate rs:
- Notable `profile.json` fields:
- Serial non-overlap timing:
  | Step | ms |
  |------|----|

## Search

- Search sweep file:
- Search L values:
- Recall@:
- Query count:
- Recall/QPS table:
  | L | Recall@10 | QPS | Wall time | Result file |
  |---|-----------|-----|-----------|-------------|
- Best QPS point:
- Best recall point:
- Default comparison point (`L=100`):

## Verdict

- Pass/fail:
- Compared baseline:
- Regression or improvement:
- Generated index cleanup: cleaned/retained, with reason if retained
- Notes:
```

## Required Post-Run Index Cleanup

After mem-search validation and `${RUN_ROOT}/summary.md` are complete, remove
the generated index artifact files from the run. Do not remove `profile.json`,
`metadata.txt`, build/search logs, command files, `${SEARCH_SWEEP_TSV}`,
per-L `search_l*_results.fbin`, `summary.md`, or diagnostic reports. Record
`_mem.index` byte size in the summary before deleting the file.

Default cleanup for the standard Wiki35M run layout:

```bash
test -s "${RUN_ROOT}/summary.md"
test -s "${SEARCH_SWEEP_TSV}"
test "$(find "${SEARCH_DIR}" -maxdepth 1 -name 'search_l*_results.fbin' -size +0 | wc -l)" -gt 0
grep -q "Recall@" "${SEARCH_DIR}"/search_l*.log
stat -c "cleanup_mem_index_bytes=%s" "${INDEX_DIR}/_mem.index" >> "${RUN_ROOT}/metadata.txt"
rm -f "${INDEX_DIR}/_mem.index"
rmdir "${INDEX_DIR}" 2>/dev/null || true
```

Verify cleanup:

```bash
test ! -e "${INDEX_DIR}/_mem.index"
```

If a follow-up comparison needs the generated index, document the reason in
`${RUN_ROOT}/summary.md` and clean it as soon as that comparison finishes.

## Optional Diagnostics

### LRPF witness backend diagnostics

LRPF runs are non-canonical until build quality and search recall are reported
against a matched HashPrune baseline. For a plain LRPF candidate-pool run, keep
pre-spill compression disabled:

```bash
SHEAR_PIPNN_PRUNE_BACKEND=lrpf
SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0
```

For disk-pressure experiments only, enable LRPF pre-spill compression and record
the candidate fragment bytes, profile `lrpf` counters, and search recall:

```bash
SHEAR_PIPNN_PRUNE_BACKEND=lrpf
SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=1
SHEAR_PIPNN_LRPF_PRE_SPILL_WIDTH_MULTIPLIER=2.0
```

Nsight Compute profiling and resident replay microbenchmarks are optional
diagnostic data. They do not replace the full Wiki35M graph build and mem-search
validation above, and timings collected under profilers must not be reported as
the canonical build time.

Store diagnostic outputs under the staging tree, not in the repository:

```bash
export DIAG_ROOT="${STAGING_ROOT}/wiki35m_gpu_subtree_measurement/diagnostics"
mkdir -p "${DIAG_ROOT}"
```

After profiling or replay diagnostics, run the normal full Wiki35M measurement
again before claiming a performance improvement. The accepted performance
numbers remain `time.log`, `profile.json`, and `search_sweep.tsv` plus the
per-L search logs from the full runbook path.
