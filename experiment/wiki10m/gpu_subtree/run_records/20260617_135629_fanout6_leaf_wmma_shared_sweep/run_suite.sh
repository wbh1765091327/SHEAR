#!/usr/bin/env bash
set -u

cd /home/wbh/cppwork/PiPGANN || exit 1

export DATA_ROOT=/data_nvme_kioxia/tyj/data
export STAGING_ROOT=/data_staging/wbh
export WIKI35M_DIR="${DATA_ROOT}/wiki-1024-35M"
export WIKI10M_DATA_DIR="${STAGING_ROOT}/wiki10m_gpu_subtree_measurement/data"
export WIKI10M_BASE="${WIKI10M_DATA_DIR}/wiki_10m_1024.fbin"
export WIKI10M_QUERY_PATH="${WIKI35M_DIR}/query.fbin"
export WIKI10M_GT_PATH="${WIKI10M_DATA_DIR}/wiki_10m_gt100.bin"
export SUITE_ROOT="/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_135629_fanout6_leaf_wmma_shared_sweep"

run_one() {
  local variant="$1"
  local shared_value="$2"
  local run_root="${SUITE_ROOT}/${variant}"
  local index_dir="${run_root}/index"
  local oom_dir="${run_root}/oom"
  local profile_json="${run_root}/profile.json"
  local stdout_log="${run_root}/stdout.log"
  local stderr_log="${run_root}/stderr.log"
  local time_log="${run_root}/time.log"
  mkdir -p "${index_dir}" "${oom_dir}"

  {
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "cwd=$(pwd)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "suite_root=${SUITE_ROOT}"
    echo "variant=${variant}"
    echo "run_root=${run_root}"
    echo "base=${WIKI10M_BASE}"
    echo "query=${WIKI10M_QUERY_PATH}"
    echo "gt=${WIKI10M_GT_PATH}"
    echo "root_assignment_backend=cpu"
    echo "assignment_backend=persistent-wmma-tf32"
    echo "leaf_topk_backend=persistent-wmma-tf32"
    echo "leaf_wmma_topk_forced=1"
    echo "DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1"
    echo "DELOANN_PIPNN_WMMA_CTA_SHARED_B=${shared_value}"
    echo "fanout_schedule_depth0_depth1_depth2plus=6,3,1"
    echo "root_risk_candidate_rs=1,2,4,6"
    echo "max_leaders=1500"
    echo "leaf_size_min=256"
    echo "leaf_size_max=1280"
    echo "oom_memory_budget_gib=200"
    echo "gpu_workspace_budget_gib=16"
    echo "build_only=true"
    echo "git_status_short_begin"
    git status --short
    echo "git_status_short_end"
    nvidia-smi || true
  } > "${run_root}/metadata.txt" 2>&1

  cat > "${run_root}/build_command.sh" <<CMD
#!/usr/bin/env bash
set -u
cd /home/wbh/cppwork/PiPGANN
DELOANN_PIPNN_PROFILE=1 \\
DELOANN_PIPNN_PHASE_MONITOR=1 \\
DELOANN_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 \\
DELOANN_PIPNN_WMMA_CTA_SHARED_B=${shared_value} \\
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
/usr/bin/time -v -o "${time_log}" \\
target/release/build_pipnn_index \\
  --data-type float \\
  --dist-fn l2 \\
  --data-path "${WIKI10M_BASE}" \\
  --index-path-prefix "${index_dir}" \\
  --overwrite \\
  --oom-enable \\
  --oom-memory-budget-gb 200 \\
  --oom-temp-dir "${oom_dir}" \\
  --oom-profile-json "${profile_json}" \\
  --max-degree 64 \\
  --lbuild 100 \\
  --num-threads 8 \\
  --leaf-size-min 256 \\
  --leaf-size-max 1280 \\
  --leaf-knn 2 \\
  --max-leaders 1500 \\
  --psamp-fraction 0.05 \\
  --fanout-top 6 \\
  --fanout-second 3 \\
  --root-risk-candidate-rs 1,2,4,6 \\
  --seed 42 \\
  --partitioner gpu-subtree \\
  --gpu-subtree-root-assignment-backend cpu \\
  --gpu-subtree-assignment-backend persistent-wmma-tf32 \\
  --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 \\
  --pipnn-backend cuda \\
  --pipnn-gpu-device 0 \\
  --pipnn-gpu-workspace-budget-gb 16 \\
  > "${stdout_log}" \\
  2> "${stderr_log}"
CMD
  chmod +x "${run_root}/build_command.sh"

  echo "${variant}: start $(date --iso-8601=seconds)" | tee -a "${SUITE_ROOT}/suite.log"
  "${run_root}/build_command.sh"
  local code=$?
  echo "build_exit_code=${code}" >> "${run_root}/metadata.txt"
  if [ -e "${index_dir}/_mem.index" ]; then
    stat -c "mem_index_bytes=%s" "${index_dir}/_mem.index" >> "${run_root}/metadata.txt" || true
  else
    echo "mem_index_bytes=missing" >> "${run_root}/metadata.txt"
  fi
  if [ -s "${time_log}" ]; then
    grep -E "Elapsed \\(wall clock\\)|Maximum resident set size|Exit status" "${time_log}" >> "${run_root}/metadata.txt" || true
  fi
  echo "${variant}: exit=${code} end $(date --iso-8601=seconds)" | tee -a "${SUITE_ROOT}/suite.log"
  return 0
}

run_one per_warp_b 0
run_one cta_shared_b 1
