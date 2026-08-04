#!/usr/bin/env bash
set -euo pipefail
DELOANN_PIPNN_PROFILE=1 \
DELOANN_PIPNN_PHASE_MONITOR=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=INT --kill-after=60s 5400s \
/usr/bin/time -v -o "/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_014545_persistent_mid_cta_split/time.log" \
target/release/build_pipnn_index \
  --data-type float \
  --dist-fn l2 \
  --data-path "/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin" \
  --index-path-prefix "/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_014545_persistent_mid_cta_split/index" \
  --overwrite \
  --oom-enable \
  --oom-memory-budget-gb 200 \
  --oom-temp-dir "/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_014545_persistent_mid_cta_split/oom" \
  --oom-profile-json "/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_014545_persistent_mid_cta_split/profile.json" \
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
  > "/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_014545_persistent_mid_cta_split/stdout.log" \
  2> "/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_014545_persistent_mid_cta_split/stderr.log"
