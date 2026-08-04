#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/env.sh"
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
