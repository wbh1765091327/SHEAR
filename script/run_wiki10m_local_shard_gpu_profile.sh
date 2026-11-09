#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_env.sh"
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TAGORE_ROOT="${TAGORE_ROOT:-$REPO_ROOT/../Tagore}"
CUVS_SOURCE_ROOT="${CUVS_SOURCE_ROOT:-$REPO_ROOT/../cuvs_tagore_ace}"
DISKANN_ROOT="${DISKANN_ROOT:-$REPO_ROOT/../DiskANN-OOM}"
DRIVER="$REPO_ROOT/script/wiki10m_local_shard_profile.py"
NSYS_ANALYZER="$REPO_ROOT/script/analyze_wiki10m_local_shard_nsys.py"
NCU_ANALYZER="$REPO_ROOT/script/analyze_wiki10m_local_shard_ncu.py"
SUMMARIZER="$REPO_ROOT/script/summarize_wiki10m_local_shard_profile.py"

CANONICAL_BASE="${SHEAR_DATA_ROOT}/wiki-1024-35M/en_base.fbin"
CANONICAL_QUERY="${SHEAR_DATA_ROOT}/wiki-1024-35M/query.fbin"
WIKI10M_BASE="${SHEAR_DATA_ROOT}/wiki10m/wiki_10m_1024.fbin"
EXPERIMENT_ROOT="${SHEAR_OUTPUT_ROOT}/wiki10m_local_shard_gpu_profile"
PREPARED_ROOT="$EXPERIMENT_ROOT/prepared"
PREPARED_SHARD_DIR="$PREPARED_ROOT/shard"
PREPARED_MANIFEST="$PREPARED_SHARD_DIR/shard_manifest.json"
PLAN_DIR="$PREPARED_ROOT/ace_plan_32"
RUNS_ROOT="$EXPERIMENT_ROOT/runs"

CUVS_PYTHON="${CUVS_PYTHON:-${SHEAR_OUTPUT_ROOT}/cuvs_tagore_ace_venv/bin/python}"
CUVS_INSTALL_ROOT="${CUVS_INSTALL_ROOT:-${SHEAR_OUTPUT_ROOT}/shear_paper_evaluation/build/cuvs_release_shared_ace/install}"
CUVS_SITE_PACKAGES="${CUVS_SITE_PACKAGES:-${SHEAR_OUTPUT_ROOT}/cuvs_tagore_ace_venv/lib/python3.12/site-packages}"
CUVS_RUNTIME_LIBRARY_PATH="${CUVS_RUNTIME_LIBRARY_PATH:-$CUVS_INSTALL_ROOT/lib:$CUVS_SITE_PACKAGES/librmm/lib64:$CUVS_SITE_PACKAGES/libraft/lib64:$CUVS_SITE_PACKAGES/rapids_logger/lib64:/usr/local/cuda-13.0/lib64}"
TAGORE_MODULE_DIR="${TAGORE_MODULE_DIR:-${SHEAR_OUTPUT_ROOT}/wiki10m_local_shard_gpu_profile/toolchain/tagore_k128_current}"
TAGORE_MODULE_SHA256="${TAGORE_MODULE_SHA256:-426070b8085cfbbe046d8e4395ced552ef00141e584aff3697377d5dfc7a8283}"
TAGORE_PLAN_TOOL="${TAGORE_PLAN_TOOL:-$TAGORE_ROOT/tools/tagore_cagra_native_ace_build.py}"
SEARCH_BIN="${SEARCH_BIN:-$REPO_ROOT/target/release/search_shear_index}"
GRAPH_QUALITY_BIN="${GRAPH_QUALITY_BIN:-$REPO_ROOT/target/release/graph_quality}"
GROUNDTRUTH_BIN="${GROUNDTRUTH_BIN:-$DISKANN_ROOT/build/apps/utils/compute_groundtruth}"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
NCU_BIN="${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}"
NVCC_BIN="${NVCC_BIN:-/usr/local/cuda-13.0/bin/nvcc}"
# Canonical timeline command: nsys profile --trace=cuda,nvtx

METHODS=(cagra tagore_cagra tagore_vamana)
REPEATS=(repeat_1 repeat_2 repeat_3)
ORDERS=(
  "cagra tagore_cagra tagore_vamana"
  "tagore_cagra tagore_vamana cagra"
  "tagore_vamana cagra tagore_cagra"
)

MODE="${1:-}"
RUN_ID="${RUN_ID:-wiki10m-local-shard-${MODE:-run}-$(date +%Y%m%d-%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'invalid RUN_ID: %s\n' "$RUN_ID" >&2
  exit 2
}
RUN_ROOT="$RUNS_ROOT/$RUN_ID"

usage() {
  printf 'Usage: %s {prepare|smoke|timing|nsys|ncu|quality|summarize|formal|cleanup}\n' "$0"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "missing executable: $1"
}

run_python() {
  env \
    LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    PYTHONPATH="$REPO_ROOT" \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
    "$CUVS_PYTHON" "$@"
}

check_idle_machine() {
  local swap_rows gpu_pids
  swap_rows="$(swapon --show --noheadings 2>/dev/null || true)"
  [[ -z "$swap_rows" ]] || die "active swap is not allowed:\n$swap_rows"
  gpu_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      sed '/^[[:space:]]*$/d' || true
  )"
  [[ -z "$gpu_pids" ]] || die "GPU has active compute processes: $gpu_pids"
}

preflight_tools() {
  require_file "$DRIVER"
  require_file "$NSYS_ANALYZER"
  require_file "$NCU_ANALYZER"
  require_file "$SUMMARIZER"
  require_file "$TAGORE_PLAN_TOOL"
  require_executable "$CUVS_PYTHON"
  require_executable "$SEARCH_BIN"
  require_executable "$GRAPH_QUALITY_BIN"
  require_executable "$GROUNDTRUTH_BIN"
  require_executable "$NCU_BIN"
  require_executable "$NVCC_BIN"
  [[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || die "nsys is unavailable"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v ldd >/dev/null || die "ldd is unavailable"
  command -v cmp >/dev/null || die "cmp is unavailable"
  command -v sha256sum >/dev/null || die "sha256sum is unavailable"
  command -v /usr/bin/time >/dev/null || die "/usr/bin/time is unavailable"

  run_python -c 'import sqlite3; from cupy.cuda import nvtx; assert nvtx.available; nvtx.RangePush("preflight"); nvtx.RangePop()'
  local tagore_modules=()
  mapfile -t tagore_modules < <(
    find "$TAGORE_MODULE_DIR" -maxdepth 1 -type f -name 'Tagore*.so' -print
  )
  [[ "${#tagore_modules[@]}" -eq 1 ]] ||
    die "expected exactly one Tagore Python module in $TAGORE_MODULE_DIR"
  local tagore_module_hash
  tagore_module_hash="$(sha256sum "${tagore_modules[0]}" | awk '{print $1}')"
  [[ "$tagore_module_hash" == "$TAGORE_MODULE_SHA256" ]] ||
    die "Tagore module SHA-256 mismatch: $tagore_module_hash"
  run_python -c 'import sys; sys.path.insert(0, sys.argv[1]); import Tagore; assert hasattr(Tagore, "GNN_descent") and hasattr(Tagore, "Pruning")' \
    "$TAGORE_MODULE_DIR"
  local extension resolved expected
  extension="$(
    find "$CUVS_SITE_PACKAGES/cuvs/neighbors/cagra" -maxdepth 1 \
      -type f -name 'cagra*.so' -print -quit
  )"
  [[ -n "$extension" ]] || die "cannot locate the cuVS CAGRA Python extension"
  resolved="$(
    env LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" ldd "$extension" |
      awk '$1 == "libcuvs.so" {print $3; exit}'
  )"
  expected="$(readlink -f "$CUVS_INSTALL_ROOT/lib/libcuvs.so")"
  [[ -n "$resolved" && "$(readlink -f "$resolved")" == "$expected" ]] ||
    die "CAGRA extension does not resolve to expected libcuvs.so"
  run_python -c 'import sys, types, cuvs; stub=types.ModuleType("cuvs.neighbors.mg"); stub.__path__=[]; sys.modules.setdefault("cuvs.neighbors.mg", stub); from cuvs.neighbors import cagra; cagra.IndexParams(metric="sqeuclidean", intermediate_graph_degree=128, graph_degree=64, build_algo="nn_descent", nn_descent_niter=16)'
}

validate_canonical_query() {
  require_file "$CANONICAL_QUERY"
  run_python - "$CANONICAL_QUERY" <<'PY'
import struct
import sys
from pathlib import Path

query = Path(sys.argv[1])
with query.open("rb") as stream:
    query_header = stream.read(8)
if query_header != struct.pack("<ii", 10_000, 1024):
    raise SystemExit(f"unexpected Wiki35M query header: {query_header!r}")
if query.stat().st_size != 40_960_008:
    raise SystemExit(f"unexpected Wiki35M query byte length: {query.stat().st_size}")
PY
}

validate_formal_input() {
  require_file "$CANONICAL_BASE"
  require_file "$WIKI10M_BASE"
  validate_canonical_query
  run_python - "$WIKI10M_BASE" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as stream:
    header = stream.read(8)
if header != struct.pack("<ii", 10_000_000, 1024):
    raise SystemExit(f"unexpected Wiki10M fbin header: {header!r}")
expected = 40_960_000_008
if path.stat().st_size != expected:
    raise SystemExit(f"unexpected Wiki10M byte length: {path.stat().st_size} != {expected}")
PY
  cmp --ignore-initial=8:8 --bytes=40960000000 "$CANONICAL_BASE" "$WIKI10M_BASE"
}

init_run_root() {
  mkdir -p "$RUN_ROOT"
  mkdir -p "$RUN_ROOT/provenance"
  touch "$RUN_ROOT/generated_artifacts.txt"
}

write_provenance() {
  init_run_root
  local tagore_module
  tagore_module="$(
    find "$TAGORE_MODULE_DIR" -maxdepth 1 -type f -name 'Tagore*.so' -print -quit
  )"
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'cuVS_python=%s\n' "$CUVS_PYTHON"
    printf 'cuVS_install_root=%s\n' "$CUVS_INSTALL_ROOT"
    printf 'tagore_module_dir=%s\n' "$TAGORE_MODULE_DIR"
    printf 'tagore_module_sha256_expected=%s\n' "$TAGORE_MODULE_SHA256"
    printf 'tagore_module_sha256_actual=%s\n' \
      "$(sha256sum "$tagore_module" | awk '{print $1}')"
  } >"$RUN_ROOT/provenance/run.env"
  git -C "$REPO_ROOT" rev-parse HEAD >"$RUN_ROOT/provenance/shear.commit"
  git -C "$REPO_ROOT" status --short >"$RUN_ROOT/provenance/shear.status"
  git -C "$TAGORE_ROOT" rev-parse HEAD >"$RUN_ROOT/provenance/tagore.commit"
  git -C "$TAGORE_ROOT" status --short >"$RUN_ROOT/provenance/tagore.status"
  git -C "$CUVS_SOURCE_ROOT" rev-parse HEAD >"$RUN_ROOT/provenance/cuvs.commit"
  git -C "$CUVS_SOURCE_ROOT" status --short >"$RUN_ROOT/provenance/cuvs.status"
  sha256sum \
    "$DRIVER" "$NSYS_ANALYZER" "$NCU_ANALYZER" "$SUMMARIZER" \
    "$REPO_ROOT/script/run_wiki10m_local_shard_gpu_profile.sh" \
    "$REPO_ROOT/docs/runbooks/wiki10m_local_shard_gpu_path_profile.md" \
    "$CUVS_INSTALL_ROOT/lib/libcuvs.so" \
    "$TAGORE_MODULE_DIR"/Tagore*.so \
    "$SEARCH_BIN" "$GRAPH_QUALITY_BIN" "$GROUNDTRUTH_BIN" \
    >"$RUN_ROOT/provenance/sha256.txt"
  nvidia-smi -q >"$RUN_ROOT/provenance/nvidia-smi-q.txt"
  "$NSYS_BIN" --version >"$RUN_ROOT/provenance/nsys-version.txt"
  "$NCU_BIN" --version >"$RUN_ROOT/provenance/ncu-version.txt"
  run_python --version >"$RUN_ROOT/provenance/python-version.txt" 2>&1
}

register_artifact() {
  local artifact
  artifact="$(readlink -m "$1")"
  case "$artifact" in
    "$RUN_ROOT"/*) ;;
    *) die "refusing to register artifact outside run root: $artifact" ;;
  esac
  printf '%s\n' "$artifact" >>"$RUN_ROOT/generated_artifacts.txt"
}

verify_shard() {
  run_python -c \
    'import sys; from pathlib import Path; from script.wiki10m_local_shard_profile import verify_shard_manifest; verify_shard_manifest(Path(sys.argv[1]))' \
    "$1"
}

prepare_phase() {
  preflight_tools
  check_idle_machine
  validate_formal_input
  mkdir -p "$PREPARED_ROOT"
  if [[ -f "$PREPARED_MANIFEST" ]]; then
    verify_shard "$PREPARED_MANIFEST"
    printf 'Reusing verified frozen shard: %s\n' "$PREPARED_MANIFEST"
    return
  fi
  [[ ! -e "$PREPARED_SHARD_DIR" ]] ||
    die "prepared shard directory exists without a verified manifest"
  [[ ! -e "$PLAN_DIR" ]] ||
    die "ACE plan directory already exists but no reusable shard is available"

  local work_dir
  work_dir="$(mktemp -d "$PREPARED_ROOT/plan-build.XXXXXX")"
  mkdir -p "$work_dir/ace_build"
  env \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
    LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    PYTHONPATH="$REPO_ROOT" \
    "$CUVS_PYTHON" "$TAGORE_PLAN_TOOL" \
      --input-fbin "$WIKI10M_BASE" \
      --output-dir "$work_dir/ace_build" \
      --output-graph-bin "$work_dir/full.graph.bin" \
      --output-mapping-bin "$work_dir/full.mapping.bin" \
      --profile-json "$work_dir/plan-build-profile.json" \
      --graph-degree 64 \
      --intermediate-graph-degree 128 \
      --device 0 \
      --npartitions 32 \
      --ef-construction 120 \
      --use-disk \
      --local-builder cagra \
      --cagra-local-solver nn-descent \
      --execution-mode pipelined \
      --tagore-iterations 16 \
      --tagore-seed 20260713 \
      --partition-plan-mode create \
      --partition-plan-dir "$PLAN_DIR" \
      --skip-input-hash \
      >"$work_dir/stdout.log" 2>"$work_dir/stderr.log"

  run_python "$DRIVER" extract \
    --source-fbin "$WIKI10M_BASE" \
    --plan "$PLAN_DIR/partition_plan.bin" \
    --reordered "$PLAN_DIR/reordered_dataset.npy" \
    --augmented "$PLAN_DIR/augmented_dataset.npy" \
    --output-dir "$PREPARED_SHARD_DIR" \
    --max-rows 1100000
  verify_shard "$PREPARED_MANIFEST"

  local artifact
  for artifact in \
    "$work_dir/full.graph.bin" \
    "$work_dir/full.mapping.bin" \
    "$work_dir/ace_build/cagra_graph.npy" \
    "$work_dir/ace_build/dataset_mapping.npy" \
    "$work_dir/ace_build/reordered_dataset.npy" \
    "$work_dir/ace_build/augmented_dataset.npy" \
    "$PLAN_DIR/dataset_mapping.npy" \
    "$PLAN_DIR/reordered_dataset.npy" \
    "$PLAN_DIR/augmented_dataset.npy"; do
    if [[ -f "$artifact" ]]; then
      printf 'Removing registered preparation artifact: %s (%s bytes)\n' \
        "$artifact" "$(stat -c %s "$artifact")"
      rm -f -- "$artifact"
    fi
  done
  printf 'Prepared and verified frozen shard: %s\n' "$PREPARED_MANIFEST"
}

start_gpu_monitor() {
  local output="$1"
  printf 'timestamp,index,memory.used [MiB]\n' >"$output"
  nvidia-smi \
    --id=0 \
    --query-gpu=timestamp,index,memory.used \
    --format=csv,noheader,nounits >>"$output"
  nvidia-smi \
    --id=0 \
    --query-gpu=timestamp,index,memory.used \
    --format=csv,noheader,nounits -lms 100 >>"$output" 2>&1 &
  printf '%s\n' "$!"
}

stop_gpu_monitor() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

driver_command() {
  local method="$1" manifest="$2" case_root="$3" profiled="$4"
  DRIVER_COMMAND=(
    "$CUVS_PYTHON" "$DRIVER" build
    --method "$method"
    --manifest "$manifest"
    --output-dir "$case_root/index"
    --profile-json "$case_root/profile.json"
    --device 0
    --tagore-module-dir "$TAGORE_MODULE_DIR"
    --cuvs-disable-mg-import
  )
  if [[ "$profiled" == "true" ]]; then
    DRIVER_COMMAND+=(--profiled-timing)
  fi
}

record_command() {
  local output="$1"
  shift
  printf '%q ' "$@" >"$output"
  printf '\n' >>"$output"
}

run_build_case() {
  local method="$1" manifest="$2" case_root="$3" profiled="$4"
  [[ ! -e "$case_root" ]] || die "case already exists: $case_root"
  mkdir -p "$case_root"
  driver_command "$method" "$manifest" "$case_root" "$profiled"
  record_command "$case_root/command.sh" "${DRIVER_COMMAND[@]}"
  local monitor_pid status
  monitor_pid="$(start_gpu_monitor "$case_root/gpu-monitor.csv")"
  set +e
  env \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
    LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    PYTHONPATH="$REPO_ROOT" \
    /usr/bin/time -v -o "$case_root/time.log" \
    "${DRIVER_COMMAND[@]}" >"$case_root/stdout.log" 2>"$case_root/stderr.log"
  status=$?
  set -e
  stop_gpu_monitor "$monitor_pid"
  printf '%s\n' "$status" >"$case_root/exit_code"
  [[ "$status" -eq 0 ]] || die "$method build failed in $case_root"
  register_artifact "$case_root/index/_mem.index"
  if [[ -f "$case_root/index/$method.raw.index" ]]; then
    register_artifact "$case_root/index/$method.raw.index"
  fi
}

timing_phase() {
  local manifest="${1:-$PREPARED_MANIFEST}"
  verify_shard "$manifest"
  init_run_root
  [[ ! -e "$RUN_ROOT/timing" ]] || die "timing phase already exists"
  mkdir -p "$RUN_ROOT/timing"
  local method repeat order
  for method in "${METHODS[@]}"; do
    run_build_case "$method" "$manifest" "$RUN_ROOT/timing/$method/warmup" false
  done
  for index in "${!REPEATS[@]}"; do
    repeat="${REPEATS[$index]}"
    order="${ORDERS[$index]}"
    for method in $order; do
      run_build_case "$method" "$manifest" "$RUN_ROOT/timing/$method/$repeat" false
    done
  done
}

run_nsys_case() {
  local method="$1" manifest="$2" case_root="$3"
  [[ ! -e "$case_root" ]] || die "Nsys case already exists: $case_root"
  mkdir -p "$case_root"
  driver_command "$method" "$manifest" "$case_root" true
  local nsys_command=(
    "$NSYS_BIN" profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --force-overwrite=false
    --output="$case_root/trace"
    "${DRIVER_COMMAND[@]}"
  )
  record_command "$case_root/command.sh" "${nsys_command[@]}"
  local status
  set +e
  env \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
    LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    PYTHONPATH="$REPO_ROOT" \
    "${nsys_command[@]}" \
      >"$case_root/stdout.log" 2>"$case_root/stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$case_root/exit_code"
  [[ "$status" -eq 0 ]] || die "Nsys $method capture failed"
  local export_command=(
    "$NSYS_BIN" export
    --type=sqlite
    --force-overwrite=false
    --output="$case_root/trace.sqlite"
    "$case_root/trace.nsys-rep"
  )
  record_command "$case_root/export.command.sh" "${export_command[@]}"
  set +e
  "${export_command[@]}" \
    >"$case_root/export.stdout.log" 2>"$case_root/export.stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$case_root/export.exit_code"
  [[ "$status" -eq 0 ]] || die "Nsys $method export failed"
  local analyze_command=(
    python3 "$NSYS_ANALYZER"
    --sqlite "$case_root/trace.sqlite"
    --method "$method"
    --output "$case_root/timeline.json"
    --unclassified-tsv "$case_root/unclassified_kernels.tsv"
  )
  record_command "$case_root/analyze.command.sh" "${analyze_command[@]}"
  "${analyze_command[@]}"
  register_artifact "$case_root/index/_mem.index"
  if [[ -f "$case_root/index/$method.raw.index" ]]; then
    register_artifact "$case_root/index/$method.raw.index"
  fi
}

nsys_phase() {
  local manifest="${1:-$PREPARED_MANIFEST}"
  verify_shard "$manifest"
  init_run_root
  [[ ! -e "$RUN_ROOT/nsys" ]] || die "Nsys phase already exists"
  mkdir -p "$RUN_ROOT/nsys"
  local method
  for method in "${METHODS[@]}"; do
    run_nsys_case "$method" "$manifest" "$RUN_ROOT/nsys/$method"
  done
}

probe_ncu_permission() {
  local probe_root="$RUN_ROOT/ncu/probe"
  mkdir -p "$probe_root"
  printf '%s\n' \
    '#include <cuda_runtime.h>' \
    '__global__ void ncu_permission_probe(int *value) { if (threadIdx.x == 0) *value = 1; }' \
    'int main() { int *value = nullptr; cudaMalloc(&value, sizeof(int)); ncu_permission_probe<<<1, 32>>>(value); int status = cudaDeviceSynchronize(); cudaFree(value); return status == cudaSuccess ? 0 : 1; }' \
    >"$probe_root/probe.cu"
  "$NVCC_BIN" -O2 "$probe_root/probe.cu" -o "$probe_root/probe"
  local ncu_probe_command=(
    "$NCU_BIN"
    --section SpeedOfLight
    --kernel-name regex:ncu_permission_probe
    --launch-count 1
    --export "$probe_root/probe-report"
    "$probe_root/probe"
  )
  record_command "$probe_root/command.sh" "${ncu_probe_command[@]}"
  local status
  set +e
  "${ncu_probe_command[@]}" \
    >"$probe_root/stdout.log" 2>"$probe_root/stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$probe_root/exit_code"
  if [[ "$status" -eq 0 ]]; then
    printf 'NCU_READY\n' >"$RUN_ROOT/ncu/status.txt"
    return 0
  fi
  if grep -q 'ERR_NVGPUCTRPERM' "$probe_root/stdout.log" "$probe_root/stderr.log"; then
    printf 'NCU_INCOMPLETE\n' >"$RUN_ROOT/ncu/status.txt"
    return 1
  fi
  die "NCU permission probe failed for a reason other than ERR_NVGPUCTRPERM"
}

run_ncu_capture() {
  local method="$1" kernel_class="$2" regex="$3" skip="$4" manifest="$5"
  local class_root="$RUN_ROOT/ncu/$method/$kernel_class"
  local capture_root="$class_root/captures/skip_$skip"
  mkdir -p "$capture_root"
  driver_command "$method" "$manifest" "$capture_root" true
  local ncu_command=(
    "$NCU_BIN"
    --target-processes all
    --kernel-name "regex:$regex"
    --launch-skip "$skip"
    --launch-count 1
    --section SpeedOfLight
    --section ComputeWorkloadAnalysis
    --section MemoryWorkloadAnalysis
    --section WarpStateStats
    --section Occupancy
    --section LaunchStats
    --section InstructionStats
    --export "$capture_root/profile"
    "${DRIVER_COMMAND[@]}"
  )
  record_command "$capture_root/command.sh" "${ncu_command[@]}"
  local status
  set +e
  env \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
    LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    PYTHONPATH="$REPO_ROOT" \
    "${ncu_command[@]}" \
      >"$capture_root/stdout.log" 2>"$capture_root/stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$capture_root/exit_code"
  if [[ "$status" -ne 0 ]]; then
    if grep -q 'ERR_NVGPUCTRPERM' "$capture_root/stdout.log" "$capture_root/stderr.log"; then
      printf 'NCU_INCOMPLETE\n' >"$RUN_ROOT/ncu/status.txt"
      return 42
    fi
    die "NCU capture failed: $method/$kernel_class/skip_$skip"
  fi
  "$NCU_BIN" \
    --import "$capture_root/profile.ncu-rep" \
    --csv \
    --page raw \
    --log-file "$capture_root/raw.csv"
  python3 "$NCU_ANALYZER" \
    --csv "$capture_root/raw.csv" \
    --method "$method" \
    --kernel-class "$kernel_class" \
    --output "$capture_root/metrics.json" \
    --output-tsv "$capture_root/metrics.tsv"
  register_artifact "$capture_root/index/_mem.index"
  if [[ -f "$capture_root/index/$method.raw.index" ]]; then
    register_artifact "$capture_root/index/$method.raw.index"
  fi
}

capture_ncu_class() {
  local method="$1" kernel_class="$2" regex="$3" skips="$4" manifest="$5"
  local skip status class_root
  class_root="$RUN_ROOT/ncu/$method/$kernel_class"
  for skip in $skips; do
    set +e
    run_ncu_capture "$method" "$kernel_class" "$regex" "$skip" "$manifest"
    status=$?
    set -e
    if [[ "$status" -eq 42 ]]; then
      return 42
    fi
    [[ "$status" -eq 0 ]] || return "$status"
  done
  local csv_args=()
  while IFS= read -r raw_csv; do
    csv_args+=(--csv "$raw_csv")
  done < <(find "$class_root/captures" -type f -name raw.csv | sort)
  [[ "${#csv_args[@]}" -gt 0 ]] || die "NCU class has no raw CSV: $method/$kernel_class"
  python3 "$NCU_ANALYZER" \
    "${csv_args[@]}" \
    --method "$method" \
    --kernel-class "$kernel_class" \
    --output "$class_root/metrics.json" \
    --output-tsv "$class_root/metrics.tsv"
}

ncu_phase() {
  local manifest="${1:-$PREPARED_MANIFEST}"
  verify_shard "$manifest"
  init_run_root
  [[ ! -e "$RUN_ROOT/ncu" ]] || die "NCU phase already exists"
  mkdir -p "$RUN_ROOT/ncu"
  if ! probe_ncu_permission; then
    printf 'NCU performance counters are unavailable; retaining NCU_INCOMPLETE.\n'
    return
  fi
  local status
  local captures=(
    "cagra|dense|local_join.*wmma|0 7 15"
    "cagra|irregular|add_rev_edges_kernel|kern_fused_prune|kern_make_rev_graph_k|kern_merge_graph|0 32 33 97"
    "cagra|mixed|kern_sort|0"
    "tagore_cagra|dense|nn_descent_opt_cal|0 3 7"
    "tagore_cagra|mixed|sample_kernel6|0 3 7"
    "tagore_cagra|irregular_cfs|select_1hop_cagra|filter_reverse_1hop|0 1"
    "tagore_vamana|dense|nn_descent_opt_cal|0 3 7"
    "tagore_vamana|mixed|sample_kernel6|0 3 7"
    "tagore_vamana|mixed_cfs|select_path|filter_reverse|0 1"
  )
  local specification capture_method capture_class capture_regex capture_skips
  for specification in "${captures[@]}"; do
    capture_method="${specification%%|*}"
    specification="${specification#*|}"
    capture_class="${specification%%|*}"
    specification="${specification#*|}"
    capture_skips="${specification##*|}"
    capture_regex="${specification%|*}"
    set +e
    capture_ncu_class \
      "$capture_method" "$capture_class" "$capture_regex" "$capture_skips" "$manifest"
    status=$?
    set -e
    if [[ "$status" -eq 42 ]]; then
      printf 'NCU permission became unavailable; retaining successful captures.\n'
      return
    fi
    [[ "$status" -eq 0 ]] || die "NCU phase failed"
  done
  printf 'NCU_READY\n' >"$RUN_ROOT/ncu/status.txt"
}

create_quality_common() {
  local manifest="$1" query_count="$2" query_source="${3:-}"
  local common="$RUN_ROOT/quality/common"
  mkdir -p "$common"
  local shard_fbin
  shard_fbin="$(
    run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_fbin"])' "$manifest"
  )"
  if [[ -n "$query_source" ]]; then
    run_python "$DRIVER" copy-query \
      --source-fbin "$query_source" \
      --output-fbin "$common/query.fbin"
  else
    run_python "$DRIVER" make-query-subset \
      --source-fbin "$shard_fbin" \
      --output-fbin "$common/query.fbin" \
      --count "$query_count" \
      --seed 20260713
  fi
  local groundtruth_command=(
    "$GROUNDTRUTH_BIN"
    --data_type float \
    --dist_fn l2 \
    --base_file "$shard_fbin" \
    --query_file "$common/query.fbin" \
    --gt_file "$common/groundtruth.bin" \
    --K 10
  )
  record_command "$common/groundtruth.command.sh" "${groundtruth_command[@]}"
  local status
  set +e
  "${groundtruth_command[@]}" \
    >"$common/groundtruth.stdout.log" 2>"$common/groundtruth.stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$common/groundtruth.exit_code"
  [[ "$status" -eq 0 ]] || die "ground-truth generation failed"
  require_file "$common/groundtruth.bin"
  run_python "$DRIVER" make-quality-contract \
    --manifest "$manifest" \
    --query "$common/query.fbin" \
    --groundtruth "$common/groundtruth.bin" \
    --output "$common/quality_contract.json" \
    --expected-k 10
}

quality_phase() {
  local manifest="${1:-$PREPARED_MANIFEST}"
  local repeat_name="${2:-repeat_2}"
  local query_count="${3:-10000}"
  local search_repetitions="${4:-3}"
  local query_source="${5:-}"
  verify_shard "$manifest"
  init_run_root
  [[ ! -e "$RUN_ROOT/quality" ]] || die "quality phase already exists"
  mkdir -p "$RUN_ROOT/quality"
  create_quality_common "$manifest" "$query_count" "$query_source"
  local shard_fbin rows method index_dir method_root l repetition metrics result status
  shard_fbin="$(
    run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_fbin"])' "$manifest"
  )"
  rows="$(
    run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$manifest"
  )"
  for method in "${METHODS[@]}"; do
    index_dir="$RUN_ROOT/timing/$method/$repeat_name/index"
    require_file "$index_dir/_mem.index"
    method_root="$RUN_ROOT/quality/$method"
    mkdir -p "$method_root"
    local graph_validation_args=(
      "$DRIVER" validate-graph
      --graph "$index_dir/_mem.index" \
      --rows "$rows" \
      --expected-max-degree 64 \
      --output "$method_root/graph_validation.json"
    )
    record_command \
      "$method_root/graph_validation.command.sh" \
      "$CUVS_PYTHON" "${graph_validation_args[@]}"
    set +e
    run_python "${graph_validation_args[@]}"
    status=$?
    set -e
    printf '%s\n' "$status" >"$method_root/graph_validation.exit_code"
    [[ "$status" -eq 0 ]] || die "graph validation failed for $method"
    local graph_quality_command=(
      "$GRAPH_QUALITY_BIN"
      --index-path-prefix "$index_dir" \
      --data-path "$shard_fbin" \
      --output "$method_root/graph_quality.tsv" \
      --max-edge-samples 1000000
    )
    record_command "$method_root/graph_quality.command.sh" "${graph_quality_command[@]}"
    set +e
    "${graph_quality_command[@]}" \
      >"$method_root/graph_quality.stdout.log" \
      2>"$method_root/graph_quality.stderr.log"
    status=$?
    set -e
    printf '%s\n' "$status" >"$method_root/graph_quality.exit_code"
    [[ "$status" -eq 0 ]] || die "graph quality failed for $method"
    printf 'L\trepetition\trecall_at_10\tqps\n' >"$method_root/search_summary.tsv"
    for l in 25 50 100 200 400; do
      local warmup_command=(
        "$SEARCH_BIN"
        --index-path-prefix "$index_dir" \
        --data-path "$shard_fbin" \
        --query-file "$RUN_ROOT/quality/common/query.fbin" \
        --gt-file "$RUN_ROOT/quality/common/groundtruth.bin" \
        --result-path "$method_root/L${l}.warmup.fbin" \
        --recall-at 10 \
        --search-list-size "$l" \
        --num-threads 8 \
        --storage mmap \
        --metrics-path "$method_root/L${l}.warmup.metrics.tsv"
      )
      record_command "$method_root/L${l}.warmup.command.sh" "${warmup_command[@]}"
      set +e
      "${warmup_command[@]}" \
        >"$method_root/L${l}.warmup.stdout.log" \
        2>"$method_root/L${l}.warmup.stderr.log"
      status=$?
      set -e
      printf '%s\n' "$status" >"$method_root/L${l}.warmup.exit_code"
      [[ "$status" -eq 0 ]] || die "quality warmup failed for $method L=$l"
      register_artifact "$method_root/L${l}.warmup.fbin"
      for repetition in $(seq 1 "$search_repetitions"); do
        metrics="$method_root/L${l}.repeat_${repetition}.metrics.tsv"
        result="$method_root/L${l}.repeat_${repetition}.fbin"
        local search_command=(
          "$SEARCH_BIN"
          --index-path-prefix "$index_dir" \
          --data-path "$shard_fbin" \
          --query-file "$RUN_ROOT/quality/common/query.fbin" \
          --gt-file "$RUN_ROOT/quality/common/groundtruth.bin" \
          --result-path "$result" \
          --recall-at 10 \
          --search-list-size "$l" \
          --num-threads 8 \
          --storage mmap \
          --metrics-path "$metrics"
        )
        record_command \
          "$method_root/L${l}.repeat_${repetition}.command.sh" \
          "${search_command[@]}"
        set +e
        "${search_command[@]}" \
          >"$method_root/L${l}.repeat_${repetition}.stdout.log" \
          2>"$method_root/L${l}.repeat_${repetition}.stderr.log"
        status=$?
        set -e
        printf '%s\n' "$status" \
          >"$method_root/L${l}.repeat_${repetition}.exit_code"
        [[ "$status" -eq 0 ]] || \
          die "quality search failed for $method L=$l repeat=$repetition"
        register_artifact "$result"
        awk -F '\t' -v work="$l" -v rep="$repetition" \
          'NR == 2 {printf "%s\t%s\t%s\t%s\n", work, rep, $5, $7}' \
          "$metrics" >>"$method_root/search_summary.tsv"
      done
    done
  done
}

summarize_phase() {
  require_file "$RUN_ROOT/ncu/status.txt"
  local arguments=(
    "$SUMMARIZER"
    --run-root "$RUN_ROOT"
    --allow-ncu-incomplete
  )
  if [[ ! -f "$RUN_ROOT/timing/cagra/repeat_2/profile.json" ]]; then
    arguments+=(--allow-single-repeat)
  fi
  python3 "${arguments[@]}"
}

smoke_phase() {
  preflight_tools
  check_idle_machine
  [[ ! -e "$RUN_ROOT" ]] || die "smoke run root already exists: $RUN_ROOT"
  init_run_root
  write_provenance
  run_python "$DRIVER" make-smoke-shard \
    --output-dir "$RUN_ROOT/smoke_input" \
    --rows 8192 \
    --dim 128 \
    --seed 20260713
  local manifest="$RUN_ROOT/smoke_input/shard_manifest.json"
  mkdir -p "$RUN_ROOT/timing"
  local method
  for method in "${METHODS[@]}"; do
    run_build_case "$method" "$manifest" "$RUN_ROOT/timing/$method/repeat_1" false
  done
  nsys_phase "$manifest"
  ncu_phase "$manifest"
  quality_phase "$manifest" repeat_1 128 1
  summarize_phase
}

formal_phase() {
  preflight_tools
  check_idle_machine
  prepare_phase
  [[ ! -e "$RUN_ROOT" ]] || die "formal run root already exists: $RUN_ROOT"
  init_run_root
  write_provenance
  timing_phase "$PREPARED_MANIFEST"
  nsys_phase "$PREPARED_MANIFEST"
  ncu_phase "$PREPARED_MANIFEST"
  quality_phase "$PREPARED_MANIFEST" repeat_2 10000 3 "$CANONICAL_QUERY"
  summarize_phase
}

cleanup_phase() {
  local manifest="$RUN_ROOT/generated_artifacts.txt"
  require_file "$manifest"
  local cleanup_log="$RUN_ROOT/cleanup_manifest.tsv"
  printf 'path\tbytes\tstatus\n' >"$cleanup_log"
  local artifact resolved bytes
  while IFS= read -r artifact; do
    [[ -n "$artifact" ]] || continue
    resolved="$(readlink -m "$artifact")"
    case "$resolved" in
      "$RUN_ROOT"/*) ;;
      *) die "cleanup manifest escapes run root: $resolved" ;;
    esac
    if [[ -f "$resolved" ]]; then
      bytes="$(stat -c %s "$resolved")"
      rm -f -- "$resolved"
      printf '%s\t%s\tremoved\n' "$resolved" "$bytes" >>"$cleanup_log"
    else
      printf '%s\t0\talready_absent\n' "$resolved" >>"$cleanup_log"
    fi
  done <"$manifest"
  printf 'Cleanup removed only files registered under %s; retained logs and reports.\n' "$RUN_ROOT"
}

case "$MODE" in
  prepare)
    prepare_phase
    ;;
  smoke)
    smoke_phase
    ;;
  timing)
    preflight_tools
    check_idle_machine
    validate_formal_input
    [[ -f "$PREPARED_MANIFEST" ]] || die "run prepare first"
    [[ ! -e "$RUN_ROOT" ]] || die "timing run root already exists: $RUN_ROOT"
    init_run_root
    write_provenance
    timing_phase "$PREPARED_MANIFEST"
    ;;
  nsys)
    preflight_tools
    check_idle_machine
    nsys_phase "$PREPARED_MANIFEST"
    ;;
  ncu)
    preflight_tools
    check_idle_machine
    ncu_phase "$PREPARED_MANIFEST"
    ;;
  quality)
    preflight_tools
    validate_canonical_query
    quality_phase "$PREPARED_MANIFEST" repeat_2 10000 3 "$CANONICAL_QUERY"
    ;;
  summarize)
    summarize_phase
    ;;
  formal)
    formal_phase
    ;;
  cleanup)
    cleanup_phase
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
