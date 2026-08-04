#!/usr/bin/env bash
set -euo pipefail
DELOANN_PIPNN_D1_CHILD_DIAGNOSTICS=1 \
DELOANN_PIPNN_WMMA_PER_WARP_B=1 \
/usr/bin/time -v target/release/build_pipnn_index \
  --data-type float \
  --dist-fn l2 \
  --data-path /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin \
  --index-path-prefix /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_114254_diag_per_warp_leaf_wmma_current_default/index \
  --overwrite \
  --oom-enable \
  --oom-memory-budget-gb 200 \
  --oom-temp-dir /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_114254_diag_per_warp_leaf_wmma_current_default/oom \
  --oom-profile-json /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_114254_diag_per_warp_leaf_wmma_current_default/profile.json \
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
  >/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_114254_diag_per_warp_leaf_wmma_current_default/stdout.log \
  2>/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260617_114254_diag_per_warp_leaf_wmma_current_default/stderr.log
