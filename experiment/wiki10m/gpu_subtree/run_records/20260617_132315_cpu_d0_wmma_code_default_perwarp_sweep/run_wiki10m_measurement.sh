#!/usr/bin/env bash
set -Eeuo pipefail
trap 'code=$?; echo "FAILED line=${LINENO} code=${code} cmd=${BASH_COMMAND}" >> "${RUN_ROOT:-/tmp}/run_driver.log"; exit $code' ERR

cd /home/wbh/cppwork/PiPGANN

export DATA_ROOT=/data_nvme_kioxia/tyj/data
export STAGING_ROOT=/data_staging/wbh
export WIKI35M_DIR="${DATA_ROOT}/wiki-1024-35M"
export WIKI35M_BASE="${WIKI35M_DIR}/en_base.fbin"
export WIKI10M_DATA_DIR="${STAGING_ROOT}/wiki10m_gpu_subtree_measurement/data"
export WIKI10M_BASE="${WIKI10M_DATA_DIR}/wiki_10m_1024.fbin"
export WIKI10M_QUERY_PATH="${WIKI35M_DIR}/query.fbin"
export WIKI10M_GT_PATH="${WIKI10M_DATA_DIR}/wiki_10m_gt100.bin"

export RUN_ROOT="/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_132315_cpu_d0_wmma_code_default_perwarp_sweep"
export RUN_ID="20260617_132315_cpu_d0_wmma_code_default_perwarp_sweep"
export INDEX_DIR="${RUN_ROOT}/index"
export OOM_DIR="${RUN_ROOT}/oom"
export PROFILE_JSON="${RUN_ROOT}/profile.json"
export BUILD_STDOUT="${RUN_ROOT}/stdout.log"
export BUILD_STDERR="${RUN_ROOT}/stderr.log"
export TIME_LOG="${RUN_ROOT}/time.log"
export SEARCH_DIR="${RUN_ROOT}/search"
export SEARCH_SWEEP_TSV="${RUN_ROOT}/search_sweep.tsv"
export RECALL_AT=10
export SEARCH_THREADS=8
export SEARCH_L_VALUES="${SEARCH_L_VALUES:-25 50 100 200 400}"
mkdir -p "${INDEX_DIR}" "${OOM_DIR}" "${SEARCH_DIR}"

unset DELOANN_PIPNN_WMMA_CTA_SHARED_B
unset DELOANN_PIPNN_WMMA_PER_WARP_B
unset DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "run_id=${RUN_ID}"
  echo "run_root=${RUN_ROOT}"
  echo "base=${WIKI10M_BASE}"
  echo "query=${WIKI10M_QUERY_PATH}"
  echo "gt=${WIKI10M_GT_PATH}"
  echo "data_root=${DATA_ROOT}"
  echo "staging_root=${STAGING_ROOT}"
  echo "default_path=cpu_d0+resident_gpu+d1_assignment_persistent_wmma_tf32+leaf_persistent_wmma_tf32"
  echo "wmma_b_staging=code-default-per-warp"
  echo "DELOANN_PIPNN_WMMA_CTA_SHARED_B=${DELOANN_PIPNN_WMMA_CTA_SHARED_B-UNSET}"
  echo "DELOANN_PIPNN_WMMA_PER_WARP_B=${DELOANN_PIPNN_WMMA_PER_WARP_B-UNSET}"
  echo "DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=${DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK-UNSET}"
  echo "search_l_values=${SEARCH_L_VALUES}"
  echo "oom_memory_budget_gib=200"
  echo "leaf_size_min=256"
  echo "leaf_size_max=1280"
  echo "max_leaders=1500"
  echo "fanout_schedule_depth0_depth1_depth2plus=8,3,1"
  echo "root_risk_candidate_rs=1,2,4,8"
  echo "git_status_short_begin"
  git status --short
  echo "git_status_short_end"
  nvidia-smi || true
} > "${RUN_ROOT}/metadata.txt" 2>&1

echo "after_metadata" >> "${RUN_ROOT}/run_driver.log"

test -s "${WIKI10M_BASE}"
test -s "${WIKI10M_QUERY_PATH}"
test -s "${WIKI10M_GT_PATH}"
od -An -tu4 -N 8 "${WIKI10M_BASE}" > "${RUN_ROOT}/base_header.txt"
od -An -tu4 -N 8 "${WIKI10M_QUERY_PATH}" > "${RUN_ROOT}/query_header.txt"
od -An -tu4 -N 8 "${WIKI10M_GT_PATH}" > "${RUN_ROOT}/gt_header.txt"
python3 - <<'PY' > "${RUN_ROOT}/gt_check.txt"
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
    max_id = max(ids)
    assert max_id < 10_000_000, max_id
    expected_ids_only = 8 + n * d * 4
    expected_with_dists = 8 + n * d * 8
    assert size in (expected_ids_only, expected_with_dists), (size, expected_ids_only, expected_with_dists)
    print(f"gt ids ok: n={n} d={d} max_id={max_id} size={size}")
PY

echo "after_input_checks" >> "${RUN_ROOT}/run_driver.log"

cat > "${RUN_ROOT}/build_command.sh" <<'CMD'
#!/usr/bin/env bash
set -euo pipefail
cd /home/wbh/cppwork/PiPGANN
unset DELOANN_PIPNN_WMMA_CTA_SHARED_B
unset DELOANN_PIPNN_WMMA_PER_WARP_B
unset DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK
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
  --leaf-size-min 256 \
  --leaf-size-max 1280 \
  --leaf-knn 2 \
  --max-leaders 1500 \
  --psamp-fraction 0.05 \
  --fanout-top 8 \
  --fanout-second 3 \
  --root-risk-candidate-rs 1,2,4,8 \
  --seed 42 \
  --partitioner gpu-subtree \
  --gpu-subtree-root-assignment-backend cpu \
  --gpu-subtree-assignment-backend persistent-wmma-tf32 \
  --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 \
  --pipnn-backend cuda \
  --pipnn-gpu-device 0 \
  --pipnn-gpu-workspace-budget-gb 16 \
  > "${BUILD_STDOUT}" \
  2> "${BUILD_STDERR}"
CMD
chmod +x "${RUN_ROOT}/build_command.sh"

echo "before_build" >> "${RUN_ROOT}/run_driver.log"
"${RUN_ROOT}/build_command.sh"
BUILD_EXIT=$?
echo "build_exit_code=${BUILD_EXIT}" >> "${RUN_ROOT}/metadata.txt"
stat -c "mem_index_bytes=%s" "${INDEX_DIR}/_mem.index" >> "${RUN_ROOT}/metadata.txt"

test -s "${INDEX_DIR}/_mem.index"
test -s "${PROFILE_JSON}"
grep -q "PiPNN: index build completed successfully" "${BUILD_STDOUT}"
grep -q "Maximum resident set size" "${TIME_LOG}"
python3 - <<'PY' > "${RUN_ROOT}/profile_check.txt"
import json
import os
profile_path = os.environ["PROFILE_JSON"]
with open(profile_path, "r", encoding="utf-8") as f:
    profile = json.load(f)
gpu = profile["gpu_subtree"]
root = profile.get("root_fanout", {})
assert profile["memory_budget_bytes"] == 214_748_364_800, profile["memory_budget_bytes"]
assert profile["partitioner"]["kind"] == "gpu-subtree", profile["partitioner"]
assert root.get("fixed_fanout") == 8, root
assert gpu["enabled"] is True, gpu
assert gpu["root_assignment_backend"] == "cpu", gpu["root_assignment_backend"]
assert gpu["assignment_backend"] == "persistent-wmma-tf32", gpu["assignment_backend"]
assert gpu["subtree_assignment_backend"] == "persistent-wmma-tf32", gpu["subtree_assignment_backend"]
assert gpu["leaf_topk_backend"] == "persistent-wmma-tf32", gpu["leaf_topk_backend"]
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
print("Wiki10M persistent GPU-subtree profile checks passed")
PY

printf "search_l\trecall_at\tqueries\twall_s\tqps\trecall\tresult_bytes\tlog\tresult\n" > "${SEARCH_SWEEP_TSV}"
for SEARCH_L in ${SEARCH_L_VALUES}; do
  export SEARCH_L
  export SEARCH_LOG="${SEARCH_DIR}/search_l${SEARCH_L}.log"
  export SEARCH_RESULTS="${SEARCH_DIR}/search_l${SEARCH_L}_results.fbin"
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
print("\t".join([
    os.environ["SEARCH_L"],
    recall.group(1),
    search.group(1),
    search.group(2),
    search.group(3),
    recall.group(2),
    os.environ["SEARCH_RESULT_BYTES"],
    log_path,
    os.environ["SEARCH_RESULTS"],
]))
PY
done

echo "completed" >> "${RUN_ROOT}/run_driver.log"
