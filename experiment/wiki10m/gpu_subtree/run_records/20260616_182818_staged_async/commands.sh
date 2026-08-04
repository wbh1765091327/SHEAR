# Commands are appended by the measurement driver during this session.
# Fail-fast input checks
test -s "${WIKI10M_BASE}"
test -s "${WIKI10M_QUERY_PATH}"
test -s "${WIKI10M_GT_PATH}"
od -An -tu4 -N 8 "${WIKI10M_BASE}"
od -An -tu4 -N 8 "${WIKI10M_QUERY_PATH}"
od -An -tu4 -N 8 "${WIKI10M_GT_PATH}"
python3 - <<'PY'
import os
import struct
import array

path = os.environ["WIKI10M_GT_PATH"]
size = os.path.getsize(path)
with open(path, "rb") as f:
    n, d = struct.unpack("<II", f.read(8))
    assert (n, d) == (10_000, 100), (n, d)
    ids = array.array("I")
    ids.fromfile(f, n * d)
    if ids.itemsize != 4:
        raise RuntimeError("unexpected u32 item size")
    max_id = max(ids)
    assert max_id < 10_000_000, max_id
    expected_ids_only = 8 + n * d * 4
    expected_with_dists = 8 + n * d * 8
    assert size in (expected_ids_only, expected_with_dists), (size, expected_ids_only, expected_with_dists)
    print(f"gt ids ok: n={n} d={d} max_id={max_id} size={size}")
PY
# Build CUDA graph-construction and search binaries
cargo build --release \
  -p build_pipnn_index \
  -p search_mem_index \
  --features build_pipnn_index/pipnn_cuda
# Full Wiki10M graph build
DELOANN_PIPNN_PROFILE=1 \
DELOANN_PIPNN_PHASE_MONITOR=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/usr/bin/time -v -o "${TIME_LOG}" \
target/release/build_pipnn_index \
  --data-type float \
  --dist-fn l2 \
  --data-path "${WIKI10M_BASE}" \
  --index-path-prefix "${INDEX_DIR}" \
  --overwrite \
  --oom-enable \
  --oom-memory-budget-gb 200 \
  --oom-temp-dir "${OOM_DIR}" \
  --oom-profile-json "${PROFILE_JSON}" \
  --max-degree 64 \
  --lbuild 100 \
  --num-threads 8 \
  --leaf-size-min 140 \
  --leaf-size-max 640 \
  --leaf-knn 2 \
  --max-leaders 64 \
  --psamp-fraction 0.05 \
  --fanout-top 4 \
  --fanout-second 2 \
  --root-risk-candidate-rs 1,2,4 \
  --seed 42 \
  --partitioner gpu-subtree \
  --gpu-subtree-output compact-topk \
  --pipnn-backend cuda \
  --pipnn-gpu-device 0 \
  --pipnn-gpu-workspace-budget-gb 16 \
  > "${BUILD_STDOUT}" \
  2> "${BUILD_STDERR}"
# Full Wiki10M graph build retry with current CLI contract.
# Difference from runbook: remove obsolete --gpu-subtree-output compact-topk.
# Current gpu-subtree OOM path uses compact top-k directly; leaf top-k backend remains explicit/default.
DELOANN_PIPNN_PROFILE=1 \
DELOANN_PIPNN_PHASE_MONITOR=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/usr/bin/time -v -o "${TIME_LOG}" \
target/release/build_pipnn_index \
  --data-type float \
  --dist-fn l2 \
  --data-path "${WIKI10M_BASE}" \
  --index-path-prefix "${INDEX_DIR}" \
  --overwrite \
  --oom-enable \
  --oom-memory-budget-gb 200 \
  --oom-temp-dir "${OOM_DIR}" \
  --oom-profile-json "${PROFILE_JSON}" \
  --max-degree 64 \
  --lbuild 100 \
  --num-threads 8 \
  --leaf-size-min 140 \
  --leaf-size-max 640 \
  --leaf-knn 2 \
  --max-leaders 64 \
  --psamp-fraction 0.05 \
  --fanout-top 4 \
  --fanout-second 2 \
  --root-risk-candidate-rs 1,2,4 \
  --seed 42 \
  --partitioner gpu-subtree \
  --pipnn-backend cuda \
  --pipnn-gpu-device 0 \
  --pipnn-gpu-workspace-budget-gb 16 \
  > "${BUILD_STDOUT}" \
  2> "${BUILD_STDERR}"
# Post-build sanity checks
test -s "${INDEX_DIR}/_mem.index"
test -s "${PROFILE_JSON}"
grep -q "PiPNN: index build completed successfully" "${BUILD_STDOUT}"
grep -q "Maximum resident set size" "${TIME_LOG}"
stat -c "mem_index_bytes=%s" "${INDEX_DIR}/_mem.index"
grep -E "Elapsed \\(wall clock\\)|Maximum resident set size" "${TIME_LOG}"
rg -n '"memory_budget_bytes"|"partitioner"|"resident_subtree"|"gpu"|"pipeline"|"leaf_flow"' "${PROFILE_JSON}"
python3 - <<'PY'
import json
import os

profile_path = os.environ["PROFILE_JSON"]
with open(profile_path, "r", encoding="utf-8") as f:
    profile = json.load(f)

gpu = profile["gpu_subtree"]
assert gpu["persistent_success_count"] > 0, gpu
assert gpu["persistent_queue_overflow_flags"] == 0, gpu["persistent_queue_overflow_flags"]
assert gpu["persistent_partition_claims"] == gpu["persistent_partition_pops"], gpu
assert gpu["persistent_topk_claims"] == gpu["persistent_topk_pops"], gpu
assert gpu["persistent_leaf_pushes"] == gpu["persistent_topk_claims"], gpu
assert gpu["persistent_large_node_splits"] > 0, gpu
assert gpu["persistent_large_node_max_rows"] >= 148098, gpu["persistent_large_node_max_rows"]
assert gpu["persistent_leaf_size_max"] <= 1024, gpu["persistent_leaf_size_max"]
print("persistent large-node profile checks passed")
PY
# Mem search validation
export RECALL_AT=10
export SEARCH_L=100
export SEARCH_THREADS=8
target/release/search_mem_index \
  --index-path-prefix "${INDEX_DIR}" \
  --data-path "${WIKI10M_BASE}" \
  --query-file "${WIKI10M_QUERY_PATH}" \
  --gt-file "${WIKI10M_GT_PATH}" \
  --result-path "${SEARCH_RESULTS}" \
  --recall-at "${RECALL_AT}" \
  --search-list-size "${SEARCH_L}" \
  --num-threads "${SEARCH_THREADS}" \
  > "${SEARCH_LOG}" \
  2>&1
# Required post-run index cleanup
test -s "${RUN_ROOT}/summary.md"
test -s "${SEARCH_RESULTS}"
grep -q "Recall@" "${SEARCH_LOG}"
stat -c "cleanup_mem_index_bytes=%s" "${INDEX_DIR}/_mem.index" >> "${RUN_ROOT}/metadata.txt"
rm -f "${INDEX_DIR}/_mem.index"
rmdir "${INDEX_DIR}" 2>/dev/null || true
test ! -e "${INDEX_DIR}/_mem.index"
