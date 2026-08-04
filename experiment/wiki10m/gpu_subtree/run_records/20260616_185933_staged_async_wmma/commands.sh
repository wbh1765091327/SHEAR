# Commands are appended by the measurement driver during this session.
# Fail-fast input checks
test -s "${WIKI10M_BASE}"
test -s "${WIKI10M_QUERY_PATH}"
test -s "${WIKI10M_GT_PATH}"
od -An -tu4 -N 8 "${WIKI10M_BASE}"
od -An -tu4 -N 8 "${WIKI10M_QUERY_PATH}"
od -An -tu4 -N 8 "${WIKI10M_GT_PATH}"
python3 - <<'PY'
import os, struct, array
path=os.environ['WIKI10M_GT_PATH']
size=os.path.getsize(path)
with open(path,'rb') as f:
    n,d=struct.unpack('<II', f.read(8))
    assert (n,d)==(10000,100),(n,d)
    ids=array.array('I'); ids.fromfile(f,n*d)
    max_id=max(ids)
    assert max_id < 10000000, max_id
    assert size in (8+n*d*4, 8+n*d*8), size
    print(f'gt ids ok: n={n} d={d} max_id={max_id} size={size}')
PY
# Build CUDA graph-construction and search binaries
cargo build --release -p build_pipnn_index -p search_mem_index --features build_pipnn_index/pipnn_cuda
# Full Wiki10M graph build with WMMA assignment and leaf top-k backends
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
  --gpu-subtree-assignment-backend persistent-wmma-tf32 \
  --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 \
  --pipnn-backend cuda \
  --pipnn-gpu-device 0 \
  --pipnn-gpu-workspace-budget-gb 16 \
  > "${BUILD_STDOUT}" \
  2> "${BUILD_STDERR}"
# Search validation for WMMA run
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
