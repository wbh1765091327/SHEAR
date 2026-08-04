#!/usr/bin/env bash
set -u
variant="$1"
cta_shared="$2"
export DATA_ROOT=/data_nvme_kioxia/tyj/data
export STAGING_ROOT=/data_staging/wbh
export WIKI35M_DIR="${DATA_ROOT}/wiki-1024-35M"
export WIKI10M_DATA_DIR="${STAGING_ROOT}/wiki10m_gpu_subtree_measurement/data"
export WIKI10M_BASE="${WIKI10M_DATA_DIR}/wiki_10m_1024.fbin"
export WIKI10M_QUERY_PATH="${WIKI35M_DIR}/query.fbin"
export WIKI10M_GT_PATH="${WIKI10M_DATA_DIR}/wiki_10m_gt100.bin"
export SUITE_ROOT="$(dirname "$0")"
export RUN_ROOT="${SUITE_ROOT}/${variant}"
export INDEX_DIR="${RUN_ROOT}/index"
export OOM_DIR="${RUN_ROOT}/oom"
export PROFILE_JSON="${RUN_ROOT}/profile.json"
export BUILD_STDOUT="${RUN_ROOT}/stdout.log"
export BUILD_STDERR="${RUN_ROOT}/stderr.log"
export TIME_LOG="${RUN_ROOT}/time.log"
export SEARCH_DIR="${RUN_ROOT}/search"
export SEARCH_SWEEP_TSV="${RUN_ROOT}/search_sweep.tsv"
mkdir -p "${INDEX_DIR}" "${OOM_DIR}" "${SEARCH_DIR}"
{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_short_begin"
  git status --short
  echo "git_status_short_end"
  echo "base=${WIKI10M_BASE}"
  echo "query=${WIKI10M_QUERY_PATH}"
  echo "gt=${WIKI10M_GT_PATH}"
  echo "suite_root=${SUITE_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "variant=${variant}"
  echo "DELOANN_PIPNN_WMMA_CTA_SHARED_B=${cta_shared}"
  echo "default_path=cpu_d0+resident_gpu+d1_assignment_persistent_wmma_tf32+leaf_persistent_wmma_tf32"
  echo "oom_memory_budget_gib=200"
  echo "leaf_size_min=256"
  echo "leaf_size_max=1280"
  echo "max_leaders=1500"
  echo "fanout_schedule_depth0_depth1_depth2plus=6,3,1"
  echo "root_risk_candidate_rs=1,2,4,6"
  echo "leaf_wmma_topk_forced=DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1"
  nvidia-smi || true
} > "${RUN_ROOT}/metadata.txt" 2>&1
(
  set -o pipefail
  DELOANN_PIPNN_PROFILE=1 \
  DELOANN_PIPNN_PHASE_MONITOR=1 \
  DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 \
  DELOANN_PIPNN_WMMA_CTA_SHARED_B="${cta_shared}" \
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
    --fanout-top 6 \
    --fanout-second 3 \
    --root-risk-candidate-rs 1,2,4,6 \
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
)
status=$?
echo "${status}" > "${RUN_ROOT}/build_exit_code.txt"
exit "${status}"
