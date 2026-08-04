# P1/P2 Three-Method Hot-Only Warm-Cache Baseline Runbook

## Material Passport

```text
artifact_type=executable experiment runbook
primary_dataset=DEEP100M
primary_methods=cagra_nn_descent,tagore_cagra,tagore_vamana
system_baseline=cagra_auto
controlled_wall_scope=original_vectors_to_persistence
system_wall_scope=original_vectors_to_persistence
host_memory_limit=none
version=p1_p2_three_method_runbook_v13_deep100m_hot_only_ace_auto_nn_descent
```

## What This Runbook Answers

This runbook uses one end-to-end P1 boundary and one common P2 search protocol.
Controlled CAGRA-NN-Descent and Tagore-CAGRA each independently executes the common ACE partition
stage from the original vectors before local graph construction and persistence.
Tagore-Vamana and the separately invoked CAGRA-auto use the same external P1 boundary.
Every row produces a validated `_mem.index`, completes search, and executes one
controlled warm-cache path.

External `/usr/bin/time -v` measures every P1 row; controlled and system-level
comparison classes must retain distinct labels. Method-local timers are diagnostics and are never added when
they overlap. Each measured build/search first drops page cache and prewarms its
declared inputs; prewarm completes before the corresponding timer starts.

## Scope and Non-Claims

`Uncapped` means that no external process-level host-memory limit is applied.
This runbook does not use cgroup controls, `memory.max`, `memory.high`, swap
controls, a cap sweep, or an OOM sweep. Tagore-Vamana fixes
`buffer_size=16`.

The controlled CAGRA/Tagore-CAGRA rows independently run the same ACE partition
algorithm with `npartitions=0`; partitioning and automatic count selection are
inside each measured wall. For the hot path, the pair is accepted only
when both methods report the same ACE auto-selected partition count. CAGRA
explicitly requests NN-Descent for every local build; any observed IVF-PQ local
build rejects the run. The unmodified CAGRA-auto path remains a separately
labeled system-level result.

Common external conditions are:

| Item | Fixed contract |
|---|---|
| Distance | squared L2 |
| Final maximum degree | 64 |
| GPU | device 0 only |
| Build CPU policy | All three methods: OMP 8 with BLAS 1 |
| ACE library | CAGRA and Tagore-CAGRA load the same validated Release cuVS library pair |
| Controlled ACE partitioning | independently executed with `npartitions=0`; actual count matched on the hot path |
| Controlled local build | CAGRA: forced NN-Descent; Tagore-CAGRA: Tagore; both use 16 iterations and degree 128 to 64 |
| Search threads | 8 |
| Search storage | mmap |
| Search entry policy | one native start node from the final graph |
| Cache policy | every measured build/search: record, drop, prewarm declared inputs, record; prewarm stays outside the timer |
| Controlled wall boundary | original base vectors through final persistence |
| System P1 boundary | original base vectors through final persistence |
| Cleanup | automatic after graph validation and required search; logs and metrics retained |

The DEEP100M formal order executes one hot-only warm-cache result per method:

```text
cagra_nn_descent hot -> tagore_cagra hot -> tagore_vamana hot
```

## Shared Paths and Toolchain

Compile binaries before measured builds. The executable program below verifies
all paths and refuses to start a measured method when a prerequisite is absent.
The modified cuVS Python environment and Tagore Python environment are kept
separate because they contain different extension dependencies.

CAGRA and Tagore-CAGRA must load the same installed `libcuvs.so` and
`libcuvs_c.so` pair from one isolated Release build. The gate below rejects a
stale install, a non-Release cache, disabled OpenMP, a different GPU
architecture, or runtime resolution to another library. Do not reuse the
historical `/data_staging/wbh/cuvs_tagore_ace_install` directory.

Recommended one-time build commands:

```bash
CUVS_SOURCE_ROOT=/path/to/cuvs_tagore_ace
CUVS_RELEASE_ROOT=/data_staging/wbh/shear_paper_evaluation/build/cuvs_release_shared_ace
test ! -e "$CUVS_RELEASE_ROOT" || {
  echo "refusing existing cuVS Release root: $CUVS_RELEASE_ROOT" >&2
  exit 2
}
mkdir -p "$CUVS_RELEASE_ROOT"
LIBCUVS_BUILD_DIR="$CUVS_RELEASE_ROOT/build" \
INSTALL_PREFIX="$CUVS_RELEASE_ROOT/install" \
PARALLEL_LEVEL=8 \
PATH="/path/to/cmake/bin:/usr/local/cuda-13.0/bin:$PATH" \
CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
CUDA_HOME=/usr/local/cuda-13.0 \
  "$CUVS_SOURCE_ROOT/build.sh" libcuvs --no-mg --gpu-arch=89-real \
    --cmake-args="-DBUILD_TESTS=OFF"

cd /path/to/SHEAR
cargo build --release -p search_shear_index

export CAGRA_DRIVER_ROOT="${CAGRA_DRIVER_ROOT:-/path/to/cagra-driver-checkout}"
cd "$CAGRA_DRIVER_ROOT"
cargo build --release -p build_cagra_ace_index

TAGORE_PYTHON=/data_staging/wbh/tagore_wiki10m_measurement/venv/bin/python
TAGORE_BUILD_ROOT=/data_staging/wbh/shear_paper_evaluation/build/tagore
PYBIND11_DIR="$($TAGORE_PYTHON -m pybind11 --cmakedir)"
cmake -S /path/to/Tagore -B "$TAGORE_BUILD_ROOT" -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DTAGORE_BUILD_PYTHON=ON -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPython_EXECUTABLE="$TAGORE_PYTHON" \
  -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build "$TAGORE_BUILD_ROOT" -j8
ctest --test-dir "$TAGORE_BUILD_ROOT" --output-on-failure
```

## Dataset Readiness Gate

The canonical DEEP100M candidates are:

```text
/data_nvme_kioxia/tyj/data/deep-96-100M/base.100M.fbin
/data_nvme_kioxia/tyj/data/deep-96-100M/query.public.10K.fbin
/data_nvme_kioxia/tyj/data/deep-96-100M/groundtruth.public.10K.ibin
```

Their currently observed header/length pairs are `100000000 x 96 /
38400000008`, `10000 x 96 / 3840008`, and `10000 x 100 / 8000008`.
These observations do not replace the run-local header, byte-length, GT-bound,
and SHA-256 checks. Stale `.part` and `.tmp` siblings are not selected.

## Build-Time Resource Harness

Every measured build runs in its own process group. The process-tree RSS sampler
sums live group-member RSS once per second. GNU-time Max RSS remains a secondary
field. GPU HBM comes from `script/monitor_gpu_build.sh`; `pidstat`, `vmstat`,
`iostat`, `/proc/meminfo`, and temporary-directory `du` samples are retained.

Process-tree RSS does not include attributable page cache. `MemAvailable` and
cache changes are system-wide diagnostics, so these fields cannot establish
behavior under an external memory limit.

Extract the fenced Bash payload between the markers, excluding the Markdown
fences. For example, the following command runs only the three-method smoke gate:

```bash
awk '$0=="<!-- BEGIN RUNBOOK_SCRIPT -->"{inside=1;next} $0=="<!-- END RUNBOOK_SCRIPT -->"{inside=0} inside && $0 !~ /^```/{print}' \
  docs/runbooks/p1_p2_shear_cagra_tagore.md | bash -s -- smoke
```

<!-- BEGIN RUNBOOK_SCRIPT -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-smoke}"
export LC_ALL=C
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
# Safe default; each baseline build overrides OpenMP to eight threads below.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export SHEAR_ROOT="${SHEAR_ROOT:-/path/to/SHEAR}"
export TAGORE_ROOT="${TAGORE_ROOT:-/path/to/Tagore}"
export CAGRA_DRIVER_ROOT="${CAGRA_DRIVER_ROOT:-/path/to/cagra-driver-checkout}"
export DISKANN_ROOT="${DISKANN_ROOT:-/path/to/DiskANN-OOM}"
export EVAL_ROOT="${EVAL_ROOT:-/data_staging/wbh/shear_paper_evaluation}"
export CUVS_SOURCE_ROOT="${CUVS_SOURCE_ROOT:-/path/to/cuvs_tagore_ace}"
export CUVS_RELEASE_ROOT="${CUVS_RELEASE_ROOT:-$EVAL_ROOT/build/cuvs_release_shared_ace}"
export CUVS_BUILD_ROOT="${CUVS_BUILD_ROOT:-$CUVS_RELEASE_ROOT/build}"
export CUVS_INSTALL_ROOT="${CUVS_INSTALL_ROOT:-$CUVS_RELEASE_ROOT/install}"
export CUVS_CMAKE_CACHE="${CUVS_CMAKE_CACHE:-$CUVS_BUILD_ROOT/CMakeCache.txt}"
export CUVS_PYTHON="${CUVS_PYTHON:-/data_staging/wbh/cuvs_tagore_ace_venv/bin/python}"
export CUVS_SITE_PACKAGES="${CUVS_SITE_PACKAGES:-$("$CUVS_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}"
export CUVS_RUNTIME_LIBRARY_PATH="$CUVS_INSTALL_ROOT/lib:$CUVS_SITE_PACKAGES/librmm/lib64:$CUVS_SITE_PACKAGES/libraft/lib64:$CUVS_SITE_PACKAGES/rapids_logger/lib64:/usr/local/cuda-13.0/lib64"
export TAGORE_PYTHON="${TAGORE_PYTHON:-/data_staging/wbh/tagore_wiki10m_measurement/venv/bin/python}"
export TAGORE_BUILD_ROOT="${TAGORE_BUILD_ROOT:-/data_staging/wbh/shear_paper_evaluation/build/tagore}"

readonly ACE_CONTROLLED_EF_CONSTRUCTION=176
readonly ACE_CONTROLLED_ITERATIONS=16
readonly ACE_CONTROLLED_TAGORE_SEED=20260713
readonly ACE_CAGRA_NN_DESCENT_MAX_ROWS=999999
readonly ACE_CONTROLLED_SMOKE_PARTITIONS=8
readonly ACE_CONTROLLED_FORMAL_PARTITIONS=0

export DEEP_BASE="${DEEP_BASE:-/data_nvme_kioxia/tyj/data/deep-96-100M/base.100M.fbin}"
export DEEP_QUERY="${DEEP_QUERY:-/data_nvme_kioxia/tyj/data/deep-96-100M/query.public.10K.fbin}"
export DEEP_GT="${DEEP_GT:-/data_nvme_kioxia/tyj/data/deep-96-100M/groundtruth.public.10K.ibin}"

ACTIVE_MONITOR_PIDS=()

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

die() {
  log "error: $*"
  exit 2
}

cleanup_monitors() {
  local pid
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  ACTIVE_MONITOR_PIDS=()
}
trap cleanup_monitors EXIT INT TERM

require_commands() {
  local command_name
  for command_name in bash python3 sha256sum setsid vmstat pidstat iostat \
    nvidia-smi cmake ninja g++ ps awk du stat find ldd readelf readlink; do
    command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
  done
  [[ -x /usr/bin/time ]] || die "missing /usr/bin/time"
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

cmake_cache_value() {
  local key="$1"
  awk -v key="$key" 'index($0, key ":") == 1 {sub(/^[^=]*=/, ""); print; exit}' \
    "$CUVS_CMAKE_CACHE"
}

require_cuvs_cache_value() {
  local key="$1" expected="$2" actual
  actual="$(cmake_cache_value "$key")"
  [[ "$actual" == "$expected" ]] || \
    die "cuVS cache mismatch: $key expected='$expected' actual='$actual'"
}

elf_build_id() {
  readelf -n "$1" | awk '$1 == "Build" && $2 == "ID:" {print $3; exit}'
}

verify_cuvs_runtime_resolution() {
  local extension ldd_output resolved_cpp resolved_c expected_cpp expected_c
  extension="$(find "$CUVS_SITE_PACKAGES/cuvs/neighbors/cagra" -maxdepth 1 \
    -type f -name 'cagra*.so' -print -quit)"
  [[ -n "$extension" ]] || die "missing cuVS CAGRA Python extension"
  require_file "$extension"
  ldd_output="$(env LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" ldd "$extension")" || \
    die "ldd failed for cuVS CAGRA extension: $extension"
  resolved_cpp="$(awk '$1 == "libcuvs.so" {print $3; exit}' <<<"$ldd_output")"
  resolved_c="$(awk '$1 == "libcuvs_c.so" {print $3; exit}' <<<"$ldd_output")"
  [[ -n "$resolved_cpp" && -n "$resolved_c" ]] || \
    die "cuVS CAGRA extension did not resolve both cuVS libraries"
  expected_cpp="$(readlink -f "$CUVS_INSTALL_ROOT/lib/libcuvs.so")"
  expected_c="$(readlink -f "$CUVS_INSTALL_ROOT/lib/libcuvs_c.so")"
  [[ "$(readlink -f "$resolved_cpp")" == "$expected_cpp" ]] || \
    die "runtime libcuvs.so resolved outside the shared Release install: $resolved_cpp"
  [[ "$(readlink -f "$resolved_c")" == "$expected_c" ]] || \
    die "runtime libcuvs_c.so resolved outside the shared Release install: $resolved_c"
}

verify_cuvs_release_contract() {
  local build_cpp="$CUVS_BUILD_ROOT/libcuvs.so"
  local build_c="$CUVS_BUILD_ROOT/c/libcuvs_c.so"
  local install_cpp="$CUVS_INSTALL_ROOT/lib/libcuvs.so"
  local install_c="$CUVS_INSTALL_ROOT/lib/libcuvs_c.so"
  local manifest="$CUVS_BUILD_ROOT/install_manifest.txt"
  local build_cpp_id build_c_id install_cpp_id install_c_id
  local path
  for path in "$CUVS_CMAKE_CACHE" "$manifest" "$build_cpp" "$build_c" \
    "$install_cpp" "$install_c"; do
    require_file "$path"
  done
  require_cuvs_cache_value CMAKE_BUILD_TYPE Release
  require_cuvs_cache_value CMAKE_CUDA_ARCHITECTURES 89-real
  require_cuvs_cache_value CMAKE_CUDA_FLAGS_RELEASE '-O3 -DNDEBUG'
  require_cuvs_cache_value CMAKE_CXX_FLAGS_RELEASE '-O3 -DNDEBUG'
  require_cuvs_cache_value BUILD_C_LIBRARY ON
  require_cuvs_cache_value BUILD_MG_ALGOS OFF
  require_cuvs_cache_value DISABLE_OPENMP OFF
  awk -v expected="$install_cpp" '$0 == expected {found=1} END {exit !found}' "$manifest" || \
    die "shared Release install manifest does not contain libcuvs.so"
  awk -v expected="$install_c" '$0 == expected {found=1} END {exit !found}' "$manifest" || \
    die "shared Release install manifest does not contain libcuvs_c.so"
  build_cpp_id="$(elf_build_id "$build_cpp")"
  build_c_id="$(elf_build_id "$build_c")"
  install_cpp_id="$(elf_build_id "$install_cpp")"
  install_c_id="$(elf_build_id "$install_c")"
  [[ -n "$build_cpp_id" && "$build_cpp_id" == "$install_cpp_id" ]] || \
    die "installed libcuvs.so does not match the validated Release build"
  [[ -n "$build_c_id" && "$build_c_id" == "$install_c_id" ]] || \
    die "installed libcuvs_c.so does not match the validated Release build"
  [[ "${CUVS_RUNTIME_LIBRARY_PATH%%:*}" == "$CUVS_INSTALL_ROOT/lib" ]] || \
    die "shared Release install is not first in CUVS_RUNTIME_LIBRARY_PATH"
  export CUVS_LIBCUVS_SHA256="$(sha256sum "$install_cpp" | awk '{print $1}')"
  export CUVS_LIBCUVS_C_SHA256="$(sha256sum "$install_c" | awk '{print $1}')"
  export CUVS_LIBCUVS_BUILD_ID="$install_cpp_id"
  export CUVS_LIBCUVS_C_BUILD_ID="$install_c_id"
  verify_cuvs_runtime_resolution
}

verify_controlled_cagra_source_contract() {
  python3 - "$CUVS_SOURCE_ROOT/cpp/include/cuvs/neighbors/cagra.hpp" \
    "$CUVS_SOURCE_ROOT/cpp/src/neighbors/detail/cagra/ace_local_cagra_params.hpp" \
    "$CUVS_SOURCE_ROOT/cpp/src/neighbors/detail/nn_descent.cuh" <<'PY'
import re, sys
from pathlib import Path

cagra_header, ace_local, nn_descent = (Path(value).read_text() for value in sys.argv[1:])
if "enum class ace_local_cagra_solver" not in cagra_header:
    raise RuntimeError("pinned cuVS explicit ACE local solver contract is missing")
if "ace_local_cagra_solver::nn_descent" not in ace_local:
    raise RuntimeError("pinned cuVS forced NN-Descent branch is missing")
if not re.search(r"max_iterations\s*=\s*5\s*\+\s*ef_construction\s*/\s*16", ace_local):
    raise RuntimeError("pinned ACE ef_construction-to-iteration mapping changed")
if "std::default_random_engine{seg_idx}" not in nn_descent:
    raise RuntimeError("pinned cuVS NN-Descent deterministic initialization changed")
PY
}

record_cuvs_contract() {
  local run_root="$1" key
  {
    printf 'contract=shared_release_cuvs_v1\n'
    printf 'source_root=%s\n' "$CUVS_SOURCE_ROOT"
    printf 'build_root=%s\n' "$CUVS_BUILD_ROOT"
    printf 'install_root=%s\n' "$CUVS_INSTALL_ROOT"
    printf 'runtime_library_path=%s\n' "$CUVS_RUNTIME_LIBRARY_PATH"
    printf 'libcuvs_sha256=%s\n' "$CUVS_LIBCUVS_SHA256"
    printf 'libcuvs_c_sha256=%s\n' "$CUVS_LIBCUVS_C_SHA256"
    printf 'libcuvs_build_id=%s\n' "$CUVS_LIBCUVS_BUILD_ID"
    printf 'libcuvs_c_build_id=%s\n' "$CUVS_LIBCUVS_C_BUILD_ID"
    for key in CMAKE_BUILD_TYPE CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_FLAGS_RELEASE \
      CMAKE_CXX_FLAGS_RELEASE BUILD_C_LIBRARY BUILD_MG_ALGOS DISABLE_OPENMP; do
      printf 'cmake.%s=%s\n' "$key" "$(cmake_cache_value "$key")"
    done
  } >"$run_root/cuvs_release_contract.txt"
}

fresh_dir() {
  [[ ! -e "$1" ]] || die "refusing existing run root: $1"
  mkdir -p "$1"
}

write_status() {
  local path="$1" phase="$2" exit_code="$3" role="$4" paper_eligible="$5"
  local failed_stage="${6:-}"
  python3 - "$path" "$phase" "$exit_code" "$role" "$paper_eligible" "$failed_stage" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
phase, exit_code, role, eligible, failed_stage = sys.argv[2:]
data = json.loads(path.read_text()) if path.exists() else {"history": []}
row = {"phase": phase, "timestamp": datetime.now(timezone.utc).isoformat(), "exit_code": int(exit_code)}
data["history"].append(row)
data.update(row)
data["experiment_role"] = role
data["paper_eligible"] = eligible.lower() == "true"
data["cache_path"] = os.environ.get("EXPERIMENT_CACHE_PATH", "none")
if failed_stage:
    data["failed_stage"] = failed_stage
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

write_dataset_validator() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY'
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

def header(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II", raw)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--query", type=Path, required=True)
parser.add_argument("--gt", type=Path, required=True)
parser.add_argument("--expected-rows", type=int, required=True)
parser.add_argument("--expected-dim", type=int, required=True)
parser.add_argument("--manifest", type=Path, required=True)
args = parser.parse_args()
for path in (args.base, args.query, args.gt):
    if not path.is_file():
        raise FileNotFoundError(path)
if ".part" in args.base.name or ".tmp" in args.base.name:
    raise RuntimeError(f"incomplete base candidate selected: {args.base}")
base_header = header(args.base)
query_header = header(args.query)
gt_header = header(args.gt)
if base_header != (args.expected_rows, args.expected_dim):
    raise RuntimeError(f"base header {base_header} != {(args.expected_rows, args.expected_dim)}")
if query_header[1] != args.expected_dim:
    raise RuntimeError(f"query dim {query_header[1]} != {args.expected_dim}")
if args.base.stat().st_size != 8 + args.expected_rows * args.expected_dim * 4:
    raise RuntimeError("base payload length mismatch")
if args.query.stat().st_size != 8 + query_header[0] * query_header[1] * 4:
    raise RuntimeError("query payload length mismatch")
if gt_header[0] != query_header[0]:
    raise RuntimeError(f"GT queries {gt_header[0]} != query rows {query_header[0]}")
id_bytes = gt_header[0] * gt_header[1] * 4
valid_gt_sizes = {8 + id_bytes, 8 + 2 * id_bytes}
if args.gt.stat().st_size not in valid_gt_sizes:
    raise RuntimeError(f"GT payload length {args.gt.stat().st_size} not in {sorted(valid_gt_sizes)}")
remaining = gt_header[0] * gt_header[1]
max_id = -1
with args.gt.open("rb") as stream:
    stream.seek(8)
    while remaining:
        count = min(remaining, 4 * 1024 * 1024)
        raw = stream.read(count * 4)
        if len(raw) != count * 4:
            raise RuntimeError("short GT ID payload")
        for (value,) in struct.iter_unpack("<I", raw):
            if value >= args.expected_rows:
                raise RuntimeError(f"GT id {value} is outside {args.expected_rows}")
            max_id = max(max_id, value)
        remaining -= count

def record(path, shape, kind):
    stat = path.stat()
    return {"path": str(path.resolve()), "kind": kind, "shape": list(shape),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}

manifest = {
    "schema": "p1-p2-dataset-manifest-v1",
    "base": record(args.base, base_header, "float32-fbin"),
    "query": record(args.query, query_header, "float32-fbin"),
    "ground_truth": record(args.gt, gt_header, "uint32-ids-optional-float32-distances"),
    "ground_truth_max_id": max_id,
}
args.manifest.parent.mkdir(parents=True, exist_ok=True)
tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.replace(tmp, args.manifest)
PY
  chmod +x "$output"
}

validate_dataset() {
  local run_root="$1" base="$2" query="$3" gt="$4" rows="$5" dim="$6"
  local validator="$run_root/tools/validate_dataset.py"
  write_dataset_validator "$validator"
  python3 "$validator" --base "$base" --query "$query" --gt "$gt" \
    --expected-rows "$rows" --expected-dim "$dim" \
    --manifest "$run_root/dataset_manifest.json"
}

record_environment() {
  local run_root="$1"
  mkdir -p "$run_root"
  {
    date --iso-8601=seconds
    uname -a
    lscpu
    free -h
    cat /proc/meminfo
    nvidia-smi
    mount
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
  } >"$run_root/environment.txt" 2>&1
  {
    for repo in "$SHEAR_ROOT" "$TAGORE_ROOT" "$CUVS_SOURCE_ROOT" "$CAGRA_DRIVER_ROOT"; do
      printf 'repo=%s\n' "$repo"
      git -C "$repo" rev-parse HEAD 2>/dev/null || true
      git -C "$repo" status --short 2>/dev/null || true
    done
  } >"$run_root/git_state.txt"
  {
    shopt -s nullglob
    local path
    for path in "$SHEAR_ROOT/target/release/search_shear_index" \
      "$SHEAR_ROOT/target/release/graph_quality" \
      "$CAGRA_DRIVER_ROOT/target/release/build_cagra_ace_index" \
      "$TAGORE_BUILD_ROOT"/libtagore_cagra_plugin.so "$TAGORE_BUILD_ROOT"/Tagore*.so \
      "$CUVS_BUILD_ROOT/libcuvs.so" "$CUVS_BUILD_ROOT/c/libcuvs_c.so" \
      "$CUVS_INSTALL_ROOT/lib/libcuvs.so" "$CUVS_INSTALL_ROOT/lib/libcuvs_c.so"; do
      [[ -f "$path" ]] && sha256sum "$path"
    done
    :
  } >"$run_root/binary_hashes.txt"
}

sample_meminfo() {
  local stop_file="$1" output="$2"
  printf 'sample_epoch\tMemTotal_kB\tMemFree_kB\tMemAvailable_kB\tBuffers_kB\tCached_kB\tSwapCached_kB\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    awk -v epoch="$(date +%s.%N)" '
      /^MemTotal:/ {total=$2} /^MemFree:/ {free=$2} /^MemAvailable:/ {available=$2}
      /^Buffers:/ {buffers=$2} /^Cached:/ {cached=$2} /^SwapCached:/ {swapcached=$2}
      END {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",epoch,total,free,available,buffers,cached,swapcached}
    ' /proc/meminfo >>"$output"
    sleep 1
  done
}

sample_temp() {
  local stop_file="$1" temp_root="$2" output="$3"
  printf 'sample_epoch\tallocated_bytes\tapparent_bytes\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    local allocated apparent
    allocated="$(du -s -B1 "$temp_root" 2>/dev/null | awk '{print $1}' || true)"
    apparent="$(du -sb "$temp_root" 2>/dev/null | awk '{print $1}' || true)"
    printf '%s\t%s\t%s\n' "$(date +%s.%N)" "${allocated:-0}" "${apparent:-0}" >>"$output"
    sleep 1
  done
}

sample_pgid() {
  local pgid="$1" output="$2"
  printf 'sample_epoch\tsample_iso\tpgid\tmembers\trss_kib\tpid_rss_comm\n' >"$output"
  while kill -0 -- "-$pgid" 2>/dev/null; do
    local snapshot members rss details
    snapshot="$(ps -e -o pid=,ppid=,pgid=,rss=,comm= | awk -v target="$pgid" '
      $3 == target {count += 1; sum += $4; details = details (details ? ";" : "") $1 ":" $4 ":" $5}
      END {printf "%d\t%d\t%s", count, sum, details}')"
    IFS=$'\t' read -r members rss details <<<"$snapshot"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s.%N)" \
      "$(date -u '+%Y-%m-%dT%H:%M:%S.%NZ')" "$pgid" \
      "${members:-0}" "${rss:-0}" "${details:-}" >>"$output"
    sleep 1
  done
}

summarize_resources() {
  local run_root="$1"
  python3 - "$run_root" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
valid, empty, epochs = [], 0, []
with (root / "process_tree_rss.tsv").open() as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        epochs.append(float(row["sample_epoch"]))
        if int(row["members"]): valid.append(row)
        else: empty += 1
gaps = sum((b - a) > 2.5 for a, b in zip(epochs, epochs[1:]))
peak = max(valid, key=lambda row: int(row["rss_kib"])) if valid else None
summary = {"valid_samples": len(valid), "empty_samples": empty, "missed_intervals": gaps,
           "peak_rss_kib": int(peak["rss_kib"]) if peak else 0,
           "peak_epoch": float(peak["sample_epoch"]) if peak else None,
           "peak_members": peak["pid_rss_comm"] if peak else ""}
(root / "process_tree_rss_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (root / "temp_usage.tsv").open() as stream:
    temp_rows = list(csv.DictReader(stream, delimiter="\t"))
temp_summary = {
    "peak_allocated_bytes": max((int(row["allocated_bytes"]) for row in temp_rows), default=0),
    "peak_apparent_bytes": max((int(row["apparent_bytes"]) for row in temp_rows), default=0),
}
(root / "temp_usage_summary.json").write_text(json.dumps(temp_summary, indent=2, sort_keys=True) + "\n")
if not valid:
    raise SystemExit("no valid process-tree RSS sample")
PY
}

run_measured_build() {
  local run_root="$1" temp_root="$2"
  shift 2
  mkdir -p "$run_root/gpu_monitor" "$temp_root"
  local command_file="$run_root/command.sh"
  {
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\nexec'
    printf ' %q' "$@"
    printf '\n'
  } >"$command_file"
  chmod +x "$command_file"
  local stop_file="$run_root/.monitor_stop" pgid_file="$run_root/build.pgid"
  rm -f "$stop_file" "$pgid_file"
  write_status "$run_root/status.json" building -1 "${EXPERIMENT_ROLE:-UNSET}" false
  vmstat -t 1 >"$run_root/vmstat.log" 2>&1 & local vmstat_pid=$!
  pidstat -r -d -u -p ALL 1 >"$run_root/pidstat.log" 2>&1 & local pidstat_pid=$!
  iostat -dxz -t 1 >"$run_root/iostat.log" 2>&1 & local iostat_pid=$!
  sample_meminfo "$stop_file" "$run_root/meminfo.tsv" & local meminfo_pid=$!
  sample_temp "$stop_file" "$temp_root" "$run_root/temp_usage.tsv" & local temp_pid=$!
  ACTIVE_MONITOR_PIDS=("$vmstat_pid" "$pidstat_pid" "$iostat_pid" "$meminfo_pid" "$temp_pid")
  bash "$SHEAR_ROOT/script/monitor_gpu_build.sh" \
    --out-dir "$run_root/gpu_monitor" --gpu 0 --interval 1 -- \
    setsid bash -c '
      pgid_file=$1; time_log=$2; stdout_log=$3; stderr_log=$4; payload=$5
      printf "%s\n" "$$" >"$pgid_file"
      exec /usr/bin/time -v -o "$time_log" "$payload" >"$stdout_log" 2>"$stderr_log"
    ' _ "$pgid_file" "$run_root/time.log" "$run_root/stdout.log" \
      "$run_root/stderr.log" "$command_file" &
  local gpu_wrapper_pid=$!
  local attempt
  for attempt in $(seq 1 200); do
    [[ -s "$pgid_file" ]] && break
    kill -0 "$gpu_wrapper_pid" 2>/dev/null || break
    sleep 0.05
  done
  if [[ ! -s "$pgid_file" ]]; then
    set +e; wait "$gpu_wrapper_pid"; local early_rc=$?; set -e
    touch "$stop_file"; cleanup_monitors
    write_status "$run_root/status.json" failed "$early_rc" "${EXPERIMENT_ROLE:-UNSET}" false launcher
    return "$early_rc"
  fi
  local pgid
  pgid="$(<"$pgid_file")"
  sample_pgid "$pgid" "$run_root/process_tree_rss.tsv" & local rss_pid=$!
  ACTIVE_MONITOR_PIDS+=("$rss_pid")
  set +e; wait "$gpu_wrapper_pid"; local build_rc=$?; set -e
  wait "$rss_pid" 2>/dev/null || true
  touch "$stop_file"
  cleanup_monitors
  summarize_resources "$run_root"
  if [[ "$build_rc" -eq 0 ]]; then
    write_status "$run_root/status.json" built 0 "${EXPERIMENT_ROLE:-UNSET}" false
  else
    write_status "$run_root/status.json" failed "$build_rc" "${EXPERIMENT_ROLE:-UNSET}" false build
  fi
  return "$build_rc"
}

ensure_graph_validator() {
  local tool_root="$EVAL_ROOT/build/tools"
  local source="$tool_root/strict_mem_graph_validator.cpp"
  local binary="$tool_root/strict_mem_graph_validator"
  mkdir -p "$tool_root"
  cat >"$source" <<'CPP'
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
template <class T> bool read_exact(std::ifstream& in, T& v) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof(v)));
}
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  const std::string path = argv[1], output = argv[4];
  const uint64_t expected = std::stoull(argv[2]);
  const uint32_t allowed = static_cast<uint32_t>(std::stoul(argv[3]));
  std::ifstream in(path, std::ios::binary); if (!in) return 3;
  uint64_t header_size = 0, frozen = 0; uint32_t header_degree = 0, start = 0;
  if (!read_exact(in, header_size) || !read_exact(in, header_degree) ||
      !read_exact(in, start) || !read_exact(in, frozen)) return 4;
  const uint64_t actual_size = std::filesystem::file_size(path);
  uint64_t invalid_degree=0, invalid_edge=0, self_edge=0, duplicate=0, edges=0;
  uint32_t observed_max=0; std::array<uint32_t,64> neighbors{};
  for (uint64_t node=0; node<expected; ++node) {
    uint32_t degree=0; if (!read_exact(in, degree)) return 5;
    if (degree==0 || degree>allowed || degree>neighbors.size()) {
      ++invalid_degree; if (degree>neighbors.size()) return 6;
    }
    observed_max=std::max(observed_max,degree); edges+=degree;
    if (!in.read(reinterpret_cast<char*>(neighbors.data()), degree*sizeof(uint32_t))) return 7;
    for (uint32_t i=0;i<degree;++i) { invalid_edge+=neighbors[i]>=expected; self_edge+=neighbors[i]==node; }
    std::sort(neighbors.begin(),neighbors.begin()+degree);
    for (uint32_t i=1;i<degree;++i) duplicate+=neighbors[i]==neighbors[i-1];
  }
  const auto consumed=static_cast<uint64_t>(in.tellg());
  const bool valid=header_size==actual_size && consumed==actual_size &&
    header_degree<=allowed && observed_max==header_degree && start<expected &&
    invalid_degree==0 && invalid_edge==0 && self_edge==0 && duplicate==0;
  std::ofstream out(output);
  out << "{\n  \"valid\": " << (valid?"true":"false")
      << ",\n  \"nodes\": " << expected << ",\n  \"edges\": " << edges
      << ",\n  \"header_size\": " << header_size << ",\n  \"actual_size\": " << actual_size
      << ",\n  \"header_max_degree\": " << header_degree
      << ",\n  \"observed_max_degree\": " << observed_max << ",\n  \"start\": " << start
      << ",\n  \"frozen_points\": " << frozen
      << ",\n  \"invalid_degree_nodes\": " << invalid_degree
      << ",\n  \"invalid_edges\": " << invalid_edge << ",\n  \"self_edges\": " << self_edge
      << ",\n  \"duplicate_edges\": " << duplicate << "\n}\n";
  return valid?0:1;
}
CPP
  if [[ ! -x "$binary" || "$source" -nt "$binary" ]]; then
    g++ -O3 -std=c++17 -Wall -Wextra "$source" -o "$binary"
  fi
  printf '%s\n' "$binary"
}

validate_graph() {
  local run_root="$1" index_dir="$2" base="$3" rows="$4"
  local graph="$index_dir/_mem.index" validator
  require_file "$graph"
  validator="$(ensure_graph_validator)"
  "$validator" "$graph" "$rows" 64 "$run_root/graph_validation.json"
  "$SHEAR_ROOT/target/release/graph_quality" --index-path-prefix "$index_dir" \
    --data-path "$base" --output "$run_root/graph_quality.tsv" --max-edge-samples 1000000
  stat "$graph" >"$run_root/index_size_before_cleanup.txt"
  write_status "$run_root/status.json" validated 0 "${EXPERIMENT_ROLE:-UNSET}" false
}

summarize_search() {
  local run_root="$1"
  python3 - "$run_root/search_sweep.tsv" "$run_root/search_summary.tsv" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
source, output = map(Path, sys.argv[1:])
groups = defaultdict(list)
with source.open() as stream:
    for row in csv.DictReader(stream, delimiter="\t"): groups[int(row["search_list_size"])].append(row)
rows=[]; last=-1.0
for search_list in sorted(groups):
    group=groups[search_list]; recalls={row["recall"] for row in group}
    if len(recalls)!=1: raise RuntimeError(f"recall changed at L={search_list}: {sorted(recalls)}")
    recall=float(next(iter(recalls)))
    if recall<last: raise RuntimeError(f"recall is not monotonic at L={search_list}")
    last=recall; qps=[float(row["qps"]) for row in group]
    rows.append((search_list,recall,statistics.mean(qps),statistics.stdev(qps) if len(qps)>1 else 0.0,min(qps),max(qps),len(qps)))
with output.open("w") as stream:
    stream.write("search_list_size\trecall\tqps_mean\tqps_stdev\tqps_min\tqps_max\trepeats\n")
    for row in rows: stream.write("\t".join(map(str,row))+"\n")
PY
}

run_search_sweep() {
  local run_root="$1" index_dir="$2" base="$3" query="$4" gt="$5" lists="$6" repeats="$7" role="$8" cache_path="$9"
  local search_root="$run_root/search" output_tsv="$run_root/search_sweep.tsv"
  mkdir -p "$search_root"
  cache_for_run "$role" "$cache_path" "$run_root" search "$base" "$index_dir/_mem.index" "$query" "$gt"
  local header_written=0 search_list repeat metrics result
  for search_list in $lists; do
    "$SHEAR_ROOT/target/release/search_shear_index" --index-path-prefix "$index_dir" \
      --data-path "$base" --query-file "$query" --gt-file "$gt" \
      --result-path "$search_root/warmup_L${search_list}.fbin" --recall-at 10 \
      --search-list-size "$search_list" --num-threads 8 --storage mmap \
      --metrics-path "$search_root/warmup_L${search_list}.tsv" \
      >"$search_root/warmup_L${search_list}.stdout.log" 2>"$search_root/warmup_L${search_list}.stderr.log"
    for repeat in $(seq 1 "$repeats"); do
      metrics="$search_root/L${search_list}_r${repeat}.tsv"
      result="$search_root/L${search_list}_r${repeat}.fbin"
      "$SHEAR_ROOT/target/release/search_shear_index" --index-path-prefix "$index_dir" \
        --data-path "$base" --query-file "$query" --gt-file "$gt" --result-path "$result" \
        --recall-at 10 --search-list-size "$search_list" --num-threads 8 --storage mmap \
        --metrics-path "$metrics" >"$search_root/L${search_list}_r${repeat}.stdout.log" \
        2>"$search_root/L${search_list}_r${repeat}.stderr.log"
      if [[ "$header_written" -eq 0 ]]; then
        printf 'replicate\t' >"$output_tsv"; head -n1 "$metrics" >>"$output_tsv"; header_written=1
      fi
      printf '%s\t' "$repeat" >>"$output_tsv"; tail -n1 "$metrics" >>"$output_tsv"
    done
  done
  summarize_search "$run_root"
  write_status "$run_root/status.json" searched 0 "${EXPERIMENT_ROLE:-UNSET}" false
}

finish_method() {
  local run_root="$1" role="$2" eligible=false
  [[ "$role" == P1_P2 ]] && eligible=true
  write_status "$run_root/status.json" accepted 0 "$role" "$eligible"
  log "accepted run: $run_root"
}

cleanup_method_artifacts() {
  local method="$1" run_root="$2" target bytes freed_bytes=0
  local manifest="$run_root/cleanup_manifest.tsv"
  local -a targets=("$run_root/index")
  case "$method" in
    cagra_nn_descent|cagra_auto|tagore_cagra)
      targets+=("$run_root/ace_export")
      ;;
    tagore_vamana)
      targets+=(
        "$run_root/diskann_part"
        "$run_root/tagore_input/clusters"
        "$run_root/tagore_input/centers.bin"
        "$run_root/tagore_run/local_index"
        "$run_root/tagore_run/index"
      )
      ;;
    *) die "unknown cleanup method: $method" ;;
  esac

  [[ -d "$run_root/index" ]] || die "cleanup requires validated index directory: $run_root/index"
  printf 'method\tbytes\tpath\n' >"$manifest"
  for target in "${targets[@]}"; do
    bytes=0
    if [[ -e "$target" || -L "$target" ]]; then
      bytes="$(du -sb -- "$target" | awk '{print $1}')"
    fi
    printf '%s\t%s\t%s\n' "$method" "$bytes" "$target" >>"$manifest"
    freed_bytes=$((freed_bytes + bytes))
  done
  for target in "${targets[@]}"; do
    if [[ -e "$target" || -L "$target" ]]; then rm -rf -- "$target"; fi
  done
  for target in "${targets[@]}"; do
    [[ ! -e "$target" && ! -L "$target" ]] || die "cleanup target remains: $target"
  done
  {
    printf 'method=%s\n' "$method"
    printf 'recorded_bytes=%s\n' "$freed_bytes"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'verified=false\n'
  } >"$run_root/cleanup_summary.txt"
}

verify_method_cleanup() {
  local method="$1" run_root="$2" manifest="$2/cleanup_manifest.tsv"
  local summary="$2/cleanup_summary.txt" row_method bytes path rows=0 evidence
  require_file "$manifest"
  require_file "$summary"
  while IFS=$'\t' read -r row_method bytes path; do
    [[ "$row_method" == "method" ]] && continue
    [[ "$row_method" == "$method" ]] || die "cleanup manifest method mismatch: $row_method"
    [[ "$bytes" =~ ^[0-9]+$ ]] || die "invalid cleanup byte count: $bytes"
    [[ "$path" == "$run_root"/* ]] || die "cleanup path escapes run root: $path"
    [[ ! -e "$path" && ! -L "$path" ]] || die "cleanup target remains: $path"
    rows=$((rows + 1))
  done <"$manifest"
  (( rows > 0 )) || die "cleanup manifest has no targets: $manifest"
  for evidence in "$run_root/time.log" "$run_root/graph_validation.json" \
    "$run_root/search_sweep.tsv" "$run_root/search_summary.tsv" \
    "$run_root/status.json" "$run_root/command.sh"; do
    require_file "$evidence"
  done
  {
    printf 'verified=true\n'
    printf 'verified_at=%s\n' "$(date --iso-8601=seconds)"
  } >>"$summary"
}

finalize_successful_method() {
  local method="$1" run_root="$2" role="$3" cleanup_rc
  if (set -Eeuo pipefail; cleanup_method_artifacts "$method" "$run_root"); then
    :
  else
    cleanup_rc=$?
    write_status "$run_root/status.json" failed "$cleanup_rc" "$role" false cleanup
    return "$cleanup_rc"
  fi
  if (set -Eeuo pipefail; verify_method_cleanup "$method" "$run_root"); then
    :
  else
    cleanup_rc=$?
    write_status "$run_root/status.json" failed "$cleanup_rc" "$role" false cleanup
    return "$cleanup_rc"
  fi
  finish_method "$run_root" "$role"
}

validate_controlled_ace_plan() {
  local plan_root="$1" base="$2" rows="$3" dim="$4" requested_partitions="$5"
  local plan_dir="$plan_root/artifacts"
  python3 - "$plan_dir" "$base" "$rows" "$dim" "$requested_partitions" \
    "$ACE_CAGRA_NN_DESCENT_MAX_ROWS" "$plan_root/controlled_plan_contract.json" <<'PY'
import hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np

plan_dir, base, contract_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[7])
rows, dim, requested, max_rows = map(int, sys.argv[3:7])
artifacts = (
    "partition_plan.bin",
    "reordered_dataset.npy",
    "augmented_dataset.npy",
    "dataset_mapping.npy",
)
manifest_path = plan_dir / "partition_plan_manifest.json"
for path in (base, manifest_path, *(plan_dir / name for name in artifacts)):
    if not path.is_file():
        raise RuntimeError(f"controlled plan artifact is missing: {path}")

header_struct = struct.Struct("<8s7I4Q")
with (plan_dir / "partition_plan.bin").open("rb") as stream:
    header = header_struct.unpack(stream.read(header_struct.size))
magic, version, header_size, endian, data_type, index_type, metric, _, plan_rows, plan_dim, npartitions, payload_bytes = header
if (magic, version, header_size, endian) != (b"ACEPLAN\0", 1, 68, 0x01020304):
    raise RuntimeError("controlled partition plan header contract mismatch")
if (plan_rows, plan_dim) != (rows, dim) or index_type != 1:
    raise RuntimeError("controlled partition plan shape or index type mismatch")

histogram = np.fromfile(
    plan_dir / "partition_plan.bin", dtype="<u4", count=2 * npartitions, offset=header_size
).reshape(npartitions, 2)
combined = histogram.astype(np.uint64).sum(axis=1)
if int(histogram[:, 0].astype(np.uint64).sum()) != rows:
    raise RuntimeError("controlled partition plan core histogram does not cover the dataset")
if combined.size == 0 or int(combined.min()) <= 0:
    raise RuntimeError("controlled partition plan contains an empty partition")
if int(combined.max()) > max_rows:
    raise RuntimeError(
        f"controlled partition exceeds NN-Descent threshold: {int(combined.max())} > {max_rows}"
    )

manifest = json.loads(manifest_path.read_text())
if manifest.get("format") != "cuvs-ace-partition-plan-v1":
    raise RuntimeError("controlled partition plan manifest format mismatch")
if manifest.get("input_rows") != rows or manifest.get("input_dim") != dim:
    raise RuntimeError("controlled partition plan manifest shape mismatch")
if manifest.get("npartitions_requested") != requested:
    raise RuntimeError("controlled partition plan requested partition count mismatch")
if manifest.get("npartitions_actual") != npartitions:
    raise RuntimeError("controlled partition plan actual partition count mismatch")

def file_state(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

state = {
    "schema": "shear-controlled-ace-plan-v1",
    "plan_dir": str(plan_dir.resolve()),
    "input": file_state(base),
    "rows": rows,
    "dim": dim,
    "npartitions_requested": requested,
    "npartitions_actual": int(npartitions),
    "core_rows": int(histogram[:, 0].astype(np.uint64).sum()),
    "augmented_rows": int(histogram[:, 1].astype(np.uint64).sum()),
    "combined_rows_min": int(combined.min()),
    "combined_rows_max": int(combined.max()),
    "nn_descent_row_limit": max_rows,
    "artifacts": {name: file_state(plan_dir / name) for name in artifacts},
    "manifest": file_state(manifest_path),
}
if contract_path.exists():
    expected = json.loads(contract_path.read_text())
    expected_without_hash = dict(expected)
    expected_without_hash.pop("partition_plan_sha256", None)
    if expected_without_hash != state:
        raise RuntimeError("controlled partition plan changed after creation")
else:
    digest = hashlib.sha256()
    with (plan_dir / "partition_plan.bin").open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    state["partition_plan_sha256"] = digest.hexdigest()
    temporary = contract_path.with_suffix(".json.incomplete")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, contract_path)
PY
}

audit_ace_partition_count() {
  local run_root="$1" requested_npartitions="$2"
  python3 - "$run_root/stderr.log" "$run_root/partition_count_audit.json" \
    "$requested_npartitions" <<'PY'
import json, re, sys
from pathlib import Path

stderr_path, output_path = map(Path, sys.argv[1:3])
requested = int(sys.argv[3])
stderr = stderr_path.read_text(errors="replace")
pipeline_counts = [
    int(value) for value in re.findall(r"ACE pipeline: (\d+) jobs", stderr)
]
if len(pipeline_counts) != 1:
    raise RuntimeError(
        f"expected one ACE pipeline partition record, found {len(pipeline_counts)}"
    )
actual = pipeline_counts[0]
if actual <= 0:
    raise RuntimeError("ACE reported a non-positive partition count")
start_counts = [
    int(value)
    for value in re.findall(
        r"ACE: Starting partitioned CAGRA build with (\d+) partitions", stderr
    )
]
if requested > 0 and actual != requested:
    raise RuntimeError(
        f"ACE actual partition count {actual} does not match requested partition count {requested}"
    )
record = {
    "schema": "shear-ace-partition-count-audit-v1",
    "npartitions_requested": requested,
    "npartitions_initial": start_counts[0] if len(start_counts) == 1 else None,
    "npartitions_actual": actual,
    "partition_policy": "ace_auto" if requested == 0 else "fixed",
}
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
}

audit_cagra_solver() {
  local run_root="$1" variant="$2" expected_npartitions="$3"
  python3 - "$run_root/stderr.log" "$run_root/ace_profile.json" \
    "$run_root/partition_count_audit.json" "$run_root/local_solver_audit.json" \
    "$variant" "$expected_npartitions" "$ACE_CONTROLLED_ITERATIONS" <<'PY'
import json, re, sys
from pathlib import Path

stderr_path, profile_path, partition_path, output_path = map(Path, sys.argv[1:5])
variant = sys.argv[5]
expected_npartitions, expected_iterations = int(sys.argv[6]), int(sys.argv[7])
profile = json.loads(profile_path.read_text())
partition = json.loads(partition_path.read_text())
stderr = stderr_path.read_text(errors="replace")
if partition["npartitions_requested"] != expected_npartitions:
    raise RuntimeError("partition audit request does not match the solver audit request")
actual_npartitions = int(partition["npartitions_actual"])
matches = [
    (solver, int(rows), requested)
    for solver, rows, requested in re.findall(
        r"ACE local CAGRA params: solver=(nn_descent|ivf_pq), rows=(\d+), .*?requested_solver=(auto|nn_descent)",
        stderr,
    )
]
if not matches:
    raise RuntimeError("no ACE local CAGRA solver records were emitted")
if len(matches) != actual_npartitions:
    raise RuntimeError("CAGRA solver count does not match the audited partition count")
if variant == "controlled":
    if any(solver == "ivf_pq" for solver, _, _ in matches):
        raise RuntimeError("controlled CAGRA emitted solver=ivf_pq")
    if any(requested != "nn_descent" for _, _, requested in matches):
        raise RuntimeError("controlled CAGRA did not request forced NN-Descent")
    if "requested_solver=nn_descent" not in stderr:
        raise RuntimeError("controlled CAGRA forced-solver log is missing")
    derived_iterations = 5 + profile["ef_construction"] // 16
    if derived_iterations != expected_iterations:
        raise RuntimeError("controlled CAGRA iteration mapping mismatch")
else:
    derived_iterations = None
record = {
    "schema": "shear-ace-local-solver-audit-v1",
    "variant": variant,
    "npartitions_requested": expected_npartitions,
    "npartitions_actual": actual_npartitions,
    "solver_counts": {
        "nn_descent": sum(solver == "nn_descent" for solver, _, _ in matches),
        "ivf_pq": sum(solver == "ivf_pq" for solver, _, _ in matches),
    },
    "requested_solver_counts": {
        "auto": sum(requested == "auto" for _, _, requested in matches),
        "nn_descent": sum(requested == "nn_descent" for _, _, requested in matches),
    },
    "local_builds": len(matches),
    "combined_rows_max": max(rows for _, rows, _ in matches),
    "cagra_nn_descent_iterations": derived_iterations,
}
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
}

# Secondary manual frozen-plan helper; no smoke or primary formal mode calls it.
prepare_controlled_ace_plan() {
  local plan_root="$1" base="$2" rows="$3" dim="$4" requested_partitions="$5"
  local plan_dir="$plan_root/artifacts" prep_root="$plan_root/preparation"
  fresh_dir "$plan_root"
  mkdir -p "$plan_dir" "$prep_root"
  local -a command=(
    "$CUVS_PYTHON" "$TAGORE_ROOT/tools/tagore_cagra_native_ace_build.py"
    --input-fbin "$base" --output-dir "$prep_root/output"
    --output-graph-bin /dev/null --output-mapping-bin /dev/null
    --profile-json "$prep_root/ace_profile.json"
    --graph-degree 64 --intermediate-graph-degree 128 --device 0
    --npartitions "$requested_partitions"
    --ef-construction "$ACE_CONTROLLED_EF_CONSTRUCTION"
    --use-disk --max-host-memory-gb 0 --max-gpu-memory-gb 0
    --local-builder cagra --cagra-local-solver nn-descent --execution-mode pipelined
    --tagore-iterations "$ACE_CONTROLLED_ITERATIONS"
    --tagore-seed "$ACE_CONTROLLED_TAGORE_SEED"
    --partition-plan-mode create --partition-plan-dir "$plan_dir"
    --skip-input-hash
  )
  {
    printf 'comparison_class=preparation_only\n'
    printf 'measured_p1=false\n'
    printf 'command='
    printf '%q ' env TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
      "LD_LIBRARY_PATH=$CUVS_RUNTIME_LIBRARY_PATH" "${command[@]}"
    printf '\n'
  } >"$prep_root/command.txt"
  /usr/bin/time -v -o "$prep_root/time.log" env \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    "${command[@]}" >"$prep_root/stdout.log" 2>"$prep_root/stderr.log"
  require_file "$prep_root/ace_profile.json"
  if [[ -f "$plan_dir/cagra_graph.npy" ]]; then
    stat -Lc '%s\t%n' "$plan_dir/cagra_graph.npy" >"$prep_root/discarded_graph.tsv"
    rm -f -- "$plan_dir/cagra_graph.npy"
  fi
  validate_controlled_ace_plan "$plan_root" "$base" "$rows" "$dim" "$requested_partitions"
  audit_ace_partition_count "$prep_root" "$requested_partitions"
  audit_cagra_solver "$prep_root" controlled "$requested_partitions"
  chmod a-w "$plan_dir"/partition_plan.bin "$plan_dir"/reordered_dataset.npy \
    "$plan_dir"/augmented_dataset.npy "$plan_dir"/dataset_mapping.npy \
    "$plan_dir"/partition_plan_manifest.json
}

write_ace_comparison_contract() {
  local run_root="$1" variant="$2" builder="$3" controlled_npartitions="$4"
  python3 - "$run_root/ace_comparison_contract.json" "$variant" "$builder" \
    "$controlled_npartitions" "$ACE_CONTROLLED_ITERATIONS" "$ACE_CONTROLLED_TAGORE_SEED" <<'PY'
import json, sys

from pathlib import Path

output, variant, builder = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
npartitions = int(sys.argv[4])
iterations, tagore_seed = int(sys.argv[5]), int(sys.argv[6])
record = {
    "schema": "shear-ace-comparison-contract-v1",
    "comparison_class": "matched_end_to_end" if variant == "controlled" else "system_level_auto",
    "wall_scope": "original_vectors_to_persistence",
    "local_builder": builder,
    "graph_degree": 64,
    "intermediate_graph_degree": 128,
    "partition_plan_mode": "disabled",
    "npartitions_requested": npartitions,
}
if variant == "controlled":
    record.update({
        "iterations": iterations,
        "cagra_local_solver": "nn-descent" if builder == "cagra" else "auto",
        "cagra_seed_contract": "pinned deterministic segment-index initialization; no external seed parameter",
        "tagore_seed": tagore_seed,
    })
else:
    record.update({"iterations": "cuvs_auto", "cagra_local_solver": "auto"})
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
}

# Paired secondary cleanup helper; unreachable from smoke and primary formal modes.
cleanup_controlled_ace_plan() {
  local plan_root="$1" plan_dir="$1/artifacts" bytes
  require_file "$plan_root/controlled_plan_contract.json"
  require_file "$plan_dir/partition_plan_manifest.json"
  cp -f -- "$plan_dir/partition_plan_manifest.json" \
    "$plan_root/retained_partition_plan_manifest.json"
  find "$plan_dir" -maxdepth 1 -type f -printf '%s\t%p\n' \
    >"$plan_root/cleanup_manifest.tsv"
  bytes="$(du -sb -- "$plan_dir" | awk '{print $1}')"
  chmod -R u+w "$plan_dir"
  rm -rf -- "$plan_dir"
  [[ ! -e "$plan_dir" ]] || die "controlled plan cleanup failed: $plan_dir"
  printf 'recorded_bytes=%s\ncompleted_at=%s\n' "$bytes" "$(date --iso-8601=seconds)" \
    >"$plan_root/cleanup_summary.txt"
}

validate_ace_profile() {
  python3 - "$1/ace_profile.json" "$1/stderr.log" "$2" "$3" "$4" <<'PY'
import json, sys

profile=json.load(open(sys.argv[1])); stderr=open(sys.argv[2], errors="replace").read()
builder, variant, expected_npartitions=sys.argv[3], sys.argv[4], int(sys.argv[5])
assert profile["local_builder"] == builder
assert profile["graph_degree"] == 64 and profile["intermediate_graph_degree"] == 128
if variant == "controlled":
    assert profile["partition_plan_mode"] == "disabled"
    assert profile["timing_scope"] == "from_scratch"
    assert profile["ef_construction"] == 176
    assert profile["npartitions_requested"] == expected_npartitions
    assert profile["partition_plan_dir"] is None
    assert profile["partition_plan_manifest"] is None
    if builder == "cagra":
        assert profile["cagra_local_solver"] == "nn-descent"
    else:
        assert profile["cagra_local_solver"] == "auto"
    if expected_npartitions > 0:
        assert profile["npartitions_actual"] == expected_npartitions
        if "Automatically increasing number of partitions" in stderr:
            raise RuntimeError("cuVS automatically increased the fixed partition count")
        if "Reduced number of partitions" in stderr:
            raise RuntimeError("cuVS reduced the fixed partition count")
    else:
        assert profile["npartitions_actual"] == 0
else:
    assert builder == "cagra"
    assert profile["partition_plan_mode"] == "disabled"
    assert profile["timing_scope"] == "from_scratch"
    assert profile["cagra_local_solver"] == "auto"
    assert profile["ef_construction"] == 120 and profile["npartitions_requested"] == 0
if builder == "tagore":
    assert profile["tagore_iterations"] == 16 and profile["tagore_seed"] == 20260713
PY
}

run_native_ace() {
  local variant="$1" builder="$2" run_root="$3" base="$4" query="$5" gt="$6" rows="$7" dim="$8" lists="$9"
  local repeats="${10}" role="${11}" cache_path="${12}" controlled_npartitions="${13}"
  local method_name=cagra_auto
  local -a ace_env=() tagore_env=()
  local -a ace_param_args=(--npartitions "$controlled_npartitions")
  local -a build_cache_inputs=("$base")
  if [[ "$variant" == controlled ]]; then
    (( controlled_npartitions >= 0 )) || die "controlled ACE requires a non-negative partition count"
    ace_env+=(TAGORE_ACE_PLAN_MODE=disabled)
    ace_param_args+=(--ef-construction "$ACE_CONTROLLED_EF_CONSTRUCTION")
    method_name=cagra_nn_descent
    if [[ "$builder" == cagra ]]; then
      ace_env+=(TAGORE_ACE_CAGRA_LOCAL_SOLVER=nn-descent)
    elif [[ "$builder" == tagore ]]; then
      ace_env+=(TAGORE_ACE_CAGRA_LOCAL_SOLVER=auto)
      method_name=tagore_cagra
    else
      die "invalid controlled ACE builder: $builder"
    fi
  elif [[ "$variant" == auto && "$builder" == cagra && "$controlled_npartitions" == 0 ]]; then
    ace_env+=(TAGORE_ACE_PLAN_MODE=disabled TAGORE_ACE_CAGRA_LOCAL_SOLVER=auto)
    ace_param_args=(--npartitions 0 --ef-construction 120)
  else
    die "invalid ACE variant/builder/partition combination: $variant/$builder/$controlled_npartitions"
  fi
  export EXPERIMENT_ROLE="$role"
  verify_cuvs_release_contract || return 1
  fresh_dir "$run_root" || return 1
  local index_dir="$run_root/index"
  mkdir -p "$index_dir" || return 1
  record_cache_path "$run_root" "$cache_path" || return 1
  record_environment "$run_root" || return 1
  record_cuvs_contract "$run_root" || return 1
  validate_dataset "$run_root" "$base" "$query" "$gt" "$rows" "$dim" || return 1
  write_ace_comparison_contract "$run_root" "$variant" "$builder" "$controlled_npartitions" || return 1
  cache_for_run "$role" "$cache_path" "$run_root" build "${build_cache_inputs[@]}" || return 1
  if [[ "$builder" == tagore ]]; then
    tagore_env=(TAGORE_CAGRA_PLUGIN_PATH="$TAGORE_BUILD_ROOT/libtagore_cagra_plugin.so" TAGORE_CAGRA_ITERATIONS="$ACE_CONTROLLED_ITERATIONS" TAGORE_CAGRA_SEED="$ACE_CONTROLLED_TAGORE_SEED")
  fi
  if ! run_measured_build "$run_root" "$run_root" env \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 \
    TAGORE_CUVS_PYTHON="$CUVS_PYTHON" TAGORE_CUVS_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    TAGORE_CUVS_DISABLE_MG_IMPORT=1 TAGORE_ACE_EXECUTION_MODE=pipelined \
    TAGORE_ACE_SKIP_INPUT_HASH=1 "${ace_env[@]}" \
    TAGORE_ACE_LOCAL_BUILDER="$builder" "${tagore_env[@]}" \
    "$CAGRA_DRIVER_ROOT/target/release/build_cagra_ace_index" --data-path "$base" \
      --index-path-prefix "$index_dir" --run-root "$run_root" \
      --cagra-command "$TAGORE_ROOT/tools/tagore_cagra_native_ace_build.sh" \
      --device 0 --graph-degree 64 --intermediate-graph-degree 128 \
      "${ace_param_args[@]}" --max-host-memory-gb 0 --max-gpu-memory-gb 0 \
      --use-disk --keep-intermediates; then return 1; fi
  validate_ace_profile "$run_root" "$builder" "$variant" "$controlled_npartitions" || return 1
  audit_ace_partition_count "$run_root" "$controlled_npartitions" || return 1
  if [[ "$builder" == cagra ]]; then
    audit_cagra_solver "$run_root" "$variant" "$controlled_npartitions" || return 1
  fi
  validate_graph "$run_root" "$index_dir" "$base" "$rows" || return 1
  run_search_sweep "$run_root" "$index_dir" "$base" "$query" "$gt" "$lists" "$repeats" "$role" "$cache_path" || return 1
  finalize_successful_method "$method_name" "$run_root" "$role" || return 1
}

write_converter() {
  local output="$1"; mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY'
#!/usr/bin/env python3
import argparse, json, os, shutil, struct, time
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument("--diskann-prefix",required=True); parser.add_argument("--tagore-out",type=Path,required=True)
parser.add_argument("--num-points",type=int,required=True); parser.add_argument("--dim",type=int,required=True)
parser.add_argument("--clusters",type=int,required=True); parser.add_argument("--max-points",type=int,required=True)
parser.add_argument("--consume-source",action="store_true")
args=parser.parse_args(); started=time.perf_counter(); out=args.tagore_out
cluster_dir=out/"clusters"; cluster_dir.mkdir(parents=True,exist_ok=True); centers_path=out/"centers.bin"
sentinel=np.uint32(0xFFFFFFFF); centers=np.memmap(centers_path,mode="w+",dtype="<u4",shape=(args.num_points,2)); centers[:]=sentinel
counts=[]; input_bytes=0; output_bytes=0; consumed_source_files=0
def read_header(path):
    with path.open("rb") as stream: raw=stream.read(8)
    if len(raw)!=8: raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II",raw)
for cid in range(args.clusters):
    data_path=Path(f"{args.diskann_prefix}_subshard-{cid}.bin"); ids_path=Path(f"{args.diskann_prefix}_subshard-{cid}_ids_uint32.bin")
    n_data,dim=read_header(data_path); n_ids,id_dim=read_header(ids_path)
    if dim!=args.dim or id_dim!=1 or n_data!=n_ids: raise RuntimeError(f"invalid shard headers for cluster {cid}")
    if data_path.stat().st_size!=8+n_data*dim*4 or ids_path.stat().st_size!=8+n_ids*4: raise RuntimeError(f"payload mismatch for cluster {cid}")
    ids=np.memmap(ids_path,mode="r",dtype="<u4",offset=8,shape=(n_ids,))
    if n_ids and int(ids.max())>=args.num_points: raise RuntimeError(f"out-of-range id in cluster {cid}")
    if np.unique(ids).size!=n_ids: raise RuntimeError(f"duplicate id within cluster {cid}")
    first_empty=centers[ids,0]==sentinel
    if np.any(centers[ids[~first_empty],1]!=sentinel): raise RuntimeError(f"third assignment in cluster {cid}")
    centers[ids[first_empty],0]=cid; centers[ids[~first_empty],1]=cid
    output_path=cluster_dir/f"cluster_{cid}"
    with data_path.open("rb") as source, output_path.open("wb") as target:
        source.seek(8); target.write(struct.pack("<I",n_data)); shutil.copyfileobj(source,target,length=64*1024*1024)
    if output_path.stat().st_size!=4+n_data*dim*4: raise RuntimeError(f"converted payload mismatch: {output_path}")
    counts.append(n_data); input_bytes+=data_path.stat().st_size+ids_path.stat().st_size; output_bytes+=output_path.stat().st_size
    del ids
    if args.consume_source:
        data_path.unlink(); ids_path.unlink(); consumed_source_files+=2
centers.flush(); del centers
if sum(counts)!=2*args.num_points: raise RuntimeError(f"assignment count {sum(counts)} != {2*args.num_points}")
if max(counts)>args.max_points: raise RuntimeError(f"max cluster {max(counts)} exceeds {args.max_points}")
centers=np.memmap(centers_path,mode="r",dtype="<u4",shape=(args.num_points,2))
for start in range(0,args.num_points,4*1024*1024):
    block=centers[start:start+4*1024*1024]
    if np.any(block==sentinel) or np.any(block>=args.clusters): raise RuntimeError(f"invalid centers block {start}")
del centers
if centers_path.stat().st_size!=args.num_points*8: raise RuntimeError("centers.bin length mismatch")
metadata={"schema":"diskann-to-tagore-v1","num_points":args.num_points,"dim":args.dim,"clusters":args.clusters,
          "max_points":args.max_points,"counts":counts,"min_cluster_count":min(counts),"max_cluster_count":max(counts),
          "total_cluster_points":sum(counts),"input_bytes":input_bytes,
          "output_bytes":output_bytes+centers_path.stat().st_size,"source_consumed":args.consume_source,
          "consumed_source_files":consumed_source_files,"elapsed_seconds":time.perf_counter()-started}
tmp=out/"convert_metadata.json.tmp"; tmp.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n"); os.replace(tmp,out/"convert_metadata.json")
PY
  chmod +x "$output"
}

write_tagore_launcher() {
  local output="$1"; mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY'
#!/usr/bin/env python3
import argparse, sys, time
from pathlib import Path
parser=argparse.ArgumentParser(); parser.add_argument("--build-root",required=True); parser.add_argument("--run-root",type=Path,required=True)
parser.add_argument("--dim",type=int,required=True); parser.add_argument("--num",type=int,required=True)
parser.add_argument("--clusters",type=int,required=True); parser.add_argument("--max-points",type=int,required=True)
parser.add_argument("--buffer-size",type=int,required=True); args=parser.parse_args(); sys.path.insert(0,args.build_root)
import Tagore
input_root=args.run_root/"tagore_input"; tagore_root=args.run_root/"tagore_run"
params={"devicelist":[0],"dim":args.dim,"k":96,"cluster_num":args.clusters,"max_points":args.max_points,
        "max_degree":64,"buffer_size":args.buffer_size,"num":args.num,"threshold":1.2,"final_degree":64,"topm":64}
print("TAGORE_PARAMS_START",flush=True)
for key,value in params.items(): print(f"{key}={value}",flush=True)
print("TAGORE_PARAMS_END",flush=True); started=time.perf_counter()
Tagore.largeIndex([0],args.dim,96,args.clusters,args.max_points,str(input_root/"clusters/cluster_"),
                  str(tagore_root/"local_index/local_"),str(tagore_root/"index/final.tagore.vamana"),
                  64,args.buffer_size,args.num,str(input_root/"centers.bin").encode(),1.2,64,64)
print(f"TAGORE_LARGEINDEX_SECONDS={time.perf_counter()-started:.6f}",flush=True)
PY
  chmod +x "$output"
}

run_tagore_vamana() {
  local run_root="$1" base="$2" query="$3" gt="$4" rows="$5" dim="$6" clusters="$7" max_points="$8" buffer_size="$9"
  local lists="${10}" repeats="${11}" role="${12}" cache_path="${13}"
  export EXPERIMENT_ROLE="$role"; fresh_dir "$run_root"; local index_dir="$run_root/index"
  mkdir -p "$index_dir" "$run_root/diskann_part" "$run_root/tagore_input" "$run_root/tagore_run/local_index" "$run_root/tagore_run/index" "$run_root/tools"
  record_cache_path "$run_root" "$cache_path"
  record_environment "$run_root"; validate_dataset "$run_root" "$base" "$query" "$gt" "$rows" "$dim"
  cache_for_run "$role" "$cache_path" "$run_root" build "$base"
  local converter="$run_root/tools/convert_diskann_partition_to_tagore.py" launcher="$run_root/tools/run_tagore_largeindex.py"
  local pipeline="$run_root/tools/tagore_vamana_pipeline.sh"
  write_converter "$converter"; write_tagore_launcher "$launcher"
  cat >"$pipeline" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
/usr/bin/time -v -o "$run_root/partition.time.log" "$DISKANN_ROOT/build/apps/utils/partition_data" \
  float "$base" "$run_root/diskann_part/dataset" 0.01 "$clusters" 2 \
  >"$run_root/partition.stdout.log" 2>"$run_root/partition.stderr.log"
/usr/bin/time -v -o "$run_root/conversion.time.log" "$CUVS_PYTHON" "$converter" \
  --diskann-prefix "$run_root/diskann_part/dataset" --tagore-out "$run_root/tagore_input" \
  --num-points "$rows" --dim "$dim" --clusters "$clusters" --max-points "$max_points" --consume-source \
  >"$run_root/conversion.stdout.log" 2>"$run_root/conversion.stderr.log"
/usr/bin/time -v -o "$run_root/largeindex.time.log" env CUDA_HOME=/usr/local/cuda-13.0 \
  PATH=/usr/local/cuda-13.0/bin:\$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\${LD_LIBRARY_PATH:-} \
  PYTHONPATH="$TAGORE_BUILD_ROOT" "$TAGORE_PYTHON" "$launcher" --build-root "$TAGORE_BUILD_ROOT" \
  --run-root "$run_root" --dim "$dim" --num "$rows" --clusters "$clusters" \
  --max-points "$max_points" --buffer-size "$buffer_size" \
  >"$run_root/largeindex.stdout.log" 2>"$run_root/largeindex.stderr.log"
test -s "$run_root/tagore_run/index/final.tagore.vamana"
ln -s "$run_root/tagore_run/index/final.tagore.vamana" "$index_dir/_mem.index"
EOF
  chmod +x "$pipeline"
  if ! run_measured_build "$run_root" "$run_root" env \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 \
    "$pipeline"; then return 1; fi
  find "$run_root/tagore_run/local_index" -type f -printf '%s\t%p\n' >"$run_root/local_index_inventory.tsv"
  validate_graph "$run_root" "$index_dir" "$base" "$rows"
  run_search_sweep "$run_root" "$index_dir" "$base" "$query" "$gt" "$lists" "$repeats" "$role" "$cache_path"
  finalize_successful_method tagore_vamana "$run_root" "$role"
}

generate_smoke_data() {
  local data_root="$1"; mkdir -p "$data_root"
  "$CUVS_PYTHON" - "$data_root" <<'PY'
import struct, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); rows,dim,queries,k=65536,32,100,10; rng=np.random.default_rng(20260713)
base=rng.normal(size=(rows,dim)).astype("<f4"); query_ids=rng.choice(rows,size=queries,replace=False)
query=base[query_ids].copy(); query+=rng.normal(0.0,0.0001,size=query.shape).astype("<f4")
base_norm=np.sum(base*base,axis=1); query_norm=np.sum(query*query,axis=1)
distance=query_norm[:,None]+base_norm[None,:]-2.0*query@base.T
candidate=np.argpartition(distance,k-1,axis=1)[:,:k]; candidate_distance=np.take_along_axis(distance,candidate,axis=1)
gt=np.take_along_axis(candidate,np.argsort(candidate_distance,axis=1),axis=1).astype("<u4")
for path,data in ((root/"base.fbin",base),(root/"query.fbin",query)):
    with path.open("wb") as stream: stream.write(struct.pack("<II",data.shape[0],data.shape[1])); data.tofile(stream)
with (root/"gt10.ibin").open("wb") as stream: stream.write(struct.pack("<II",queries,k)); gt.tofile(stream)
PY
}

verify_prerequisites() {
  require_commands
  verify_cuvs_release_contract
  verify_controlled_cagra_source_contract
  for path in "$SHEAR_ROOT/script/monitor_gpu_build.sh" \
    "$SHEAR_ROOT/target/release/search_shear_index" "$SHEAR_ROOT/target/release/graph_quality" \
    "$CAGRA_DRIVER_ROOT/target/release/build_cagra_ace_index" "$TAGORE_ROOT/tools/tagore_cagra_native_ace_build.sh" \
    "$TAGORE_BUILD_ROOT/libtagore_cagra_plugin.so" "$CUVS_INSTALL_ROOT/lib/libcuvs.so" \
    "$CUVS_INSTALL_ROOT/lib/libcuvs_c.so" "$CUVS_SITE_PACKAGES/librmm/lib64/librmm.so" \
    "$CUVS_SITE_PACKAGES/libraft/lib64/libraft.so" "$CUVS_SITE_PACKAGES/rapids_logger/lib64/librapids_logger.so" \
    "$DISKANN_ROOT/build/apps/utils/partition_data"; do require_file "$path"; done
  env TAGORE_CUVS_DISABLE_MG_IMPORT=1 LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    "$CUVS_PYTHON" "$TAGORE_ROOT/tools/tagore_cagra_native_ace_build.py" --help >/dev/null
  PYTHONPATH="$TAGORE_BUILD_ROOT" "$TAGORE_PYTHON" -c 'import Tagore'
  ensure_graph_validator >/dev/null
}

verify_tagore_vamana_prerequisites() {
  local -a tagore_extensions=("$TAGORE_BUILD_ROOT"/Tagore*.so)
  local path
  require_commands
  for path in "$SHEAR_ROOT/script/monitor_gpu_build.sh" \
    "$SHEAR_ROOT/target/release/search_shear_index" \
    "$SHEAR_ROOT/target/release/graph_quality" \
    "$DISKANN_ROOT/build/apps/utils/partition_data" \
    "$CUVS_PYTHON" "$TAGORE_PYTHON"; do
    require_file "$path"
  done
  [[ -f "${tagore_extensions[0]:-}" ]] || \
    die "missing Tagore Python extension under $TAGORE_BUILD_ROOT"
  "$CUVS_PYTHON" -c 'import numpy'
  PYTHONPATH="$TAGORE_BUILD_ROOT" "$TAGORE_PYTHON" -c 'import Tagore'
  ensure_graph_validator >/dev/null
}

run_method() {
  local method="$1" run_root="$2" base="$3" query="$4" gt="$5" rows="$6" dim="$7" lists="$8" repeats="$9"
  local role="${10}" clusters="${11}" max_points="${12}" buffer_size="${13}" cache_path="${14}" controlled_npartitions="${15}"
  case "$method" in
    cagra_nn_descent) run_native_ace controlled cagra "$run_root" "$base" "$query" "$gt" "$rows" "$dim" "$lists" "$repeats" "$role" "$cache_path" "$controlled_npartitions" ;;
    tagore_cagra) run_native_ace controlled tagore "$run_root" "$base" "$query" "$gt" "$rows" "$dim" "$lists" "$repeats" "$role" "$cache_path" "$controlled_npartitions" ;;
    cagra_auto) run_native_ace auto cagra "$run_root" "$base" "$query" "$gt" "$rows" "$dim" "$lists" "$repeats" "$role" "$cache_path" 0 ;;
    tagore_vamana) run_tagore_vamana "$run_root" "$base" "$query" "$gt" "$rows" "$dim" "$clusters" "$max_points" "$buffer_size" "$lists" "$repeats" "$role" "$cache_path" ;;
    *) die "unknown method: $method" ;;
  esac
}

run_smoke() {
  local timestamp root data_root method failed=0
  timestamp="$(date +%Y%m%d_%H%M%S)"; root="${SMOKE_RUN_ROOT:-$EVAL_ROOT/smoke/${timestamp}_p1_p2_three_method}"
  fresh_dir "$root"; data_root="$root/data"; generate_smoke_data "$data_root"
  printf '%s\n' "$root" >"$EVAL_ROOT/latest_smoke_root.txt"
  for method in cagra_nn_descent tagore_cagra tagore_vamana; do
    log "smoke method=$method root=$root/$method"
    if ! run_method "$method" "$root/$method" "$data_root/base.fbin" "$data_root/query.fbin" \
      "$data_root/gt10.ibin" 65536 32 "100" 1 SMOKE_ONLY 8 32768 4 none \
      "$ACE_CONTROLLED_SMOKE_PARTITIONS"; then
      failed=1; log "smoke failed: $method"
    fi
  done
  log "smoke root: $root"; return "$failed"
}

run_smoke_tagore_vamana() {
  local timestamp root data_root
  timestamp="$(date +%Y%m%d_%H%M%S)"
  root="${SMOKE_RUN_ROOT:-$EVAL_ROOT/smoke/${timestamp}_tagore_vamana}"
  fresh_dir "$root"; data_root="$root/data"; generate_smoke_data "$data_root"
  printf '%s\n' "$root" >"$EVAL_ROOT/latest_smoke_tagore_vamana_root.txt"
  log "smoke method=tagore_vamana root=$root/tagore_vamana"
  run_method tagore_vamana "$root/tagore_vamana" "$data_root/base.fbin" \
    "$data_root/query.fbin" "$data_root/gt10.ibin" 65536 32 "100" 1 \
    SMOKE_ONLY 8 32768 4 none none
  log "Tagore-Vamana smoke root: $root"
}

exclusive_host_gate() {
  local gpu_pids
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$gpu_pids" ]] || die "GPU 0 has active compute PIDs: $gpu_pids"
  sudo -n true >/dev/null 2>&1 || die "passwordless sudo is required for controlled cache protocol"
}

record_cache_path() {
  local run_root="$1" cache_path="$2"
  case "$cache_path" in
    hot|none) ;;
    *) die "unknown cache path: $cache_path" ;;
  esac
  printf '%s\n' "$cache_path" >"$run_root/cache_path.txt"
  export EXPERIMENT_CACHE_PATH="$cache_path"
}

record_cache_state() {
  local run_root="$1" phase="$2" point="$3" output="$run_root/cache_protocol.tsv"
  if [[ ! -e "$output" ]]; then
    printf 'sample_epoch\tsample_iso\tphase\tpoint\tMemFree_kB\tMemAvailable_kB\tBuffers_kB\tCached_kB\tSReclaimable_kB\tSwapCached_kB\n' >"$output"
  fi
  awk -v epoch="$(date +%s.%N)" -v iso="$(date --iso-8601=ns)" \
    -v phase="$phase" -v point="$point" '
      /^MemFree:/ {free=$2} /^MemAvailable:/ {available=$2} /^Buffers:/ {buffers=$2}
      /^Cached:/ {cached=$2} /^SReclaimable:/ {reclaimable=$2} /^SwapCached:/ {swapcached=$2}
      END {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", epoch,iso,phase,point,free,available,buffers,cached,reclaimable,swapcached}
    ' /proc/meminfo >>"$output"
}

cold_cache() {
  local run_root="$1" phase="$2"
  record_cache_state "$run_root" "$phase" before_drop
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  record_cache_state "$run_root" "$phase" after_drop
}

warm_cache_paths() {
  local run_root="$1" phase="$2"
  shift 2
  local manifest="$run_root/${phase}_prewarm_manifest.tsv"
  local timing="$run_root/${phase}_prewarm.time.log"
  local path bytes
  : >"$timing"
  printf 'phase\tbytes\tpath\n' >"$manifest"
  for path in "$@"; do
    require_file "$path"
    bytes="$(stat -Lc %s "$path")"
    printf '%s\t%s\t%s\n' "$phase" "$bytes" "$path" >>"$manifest"
    /usr/bin/time -v -a -o "$timing" dd if="$path" of=/dev/null bs=64M status=none
  done
}

hot_cache() {
  local run_root="$1" phase="$2"
  shift 2
  cold_cache "$run_root" "$phase"
  warm_cache_paths "$run_root" "$phase" "$@"
  record_cache_state "$run_root" "$phase" after_warm
}

cache_for_run() {
  local role="$1" cache_path="$2" run_root="$3" phase="$4"
  shift 4
  case "$role:$cache_path" in
    P1_P2:hot) hot_cache "$run_root" "$phase" "$@" ;;
    SMOKE_ONLY:none) ;;
    *) die "invalid role/cache path: $role/$cache_path" ;;
  esac
}

run_hot_method() {
  local root="$1" method="$2" controlled_npartitions="$3"
  shift 3
  run_method "$method" "$root/hot_path/$method/run_1" "$@" hot "$controlled_npartitions"
}

verify_controlled_partition_count_match() {
  local root="$1"
  python3 - "$root" "$root/controlled_partition_count_match.json" <<'PY'
import json, sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])
records = {}
errors = []
for method in ("cagra_nn_descent", "tagore_cagra"):
    audit_path = root / f"hot_path/{method}/run_1/partition_count_audit.json"
    records[method] = json.loads(audit_path.read_text())
cagra = records["cagra_nn_descent"]
tagore = records["tagore_cagra"]
actual_match = cagra["npartitions_actual"] == tagore["npartitions_actual"]
auto_contract = all(
    record["npartitions_requested"] == 0
    and record["partition_policy"] == "ace_auto"
    for record in records.values()
)
hot = {
    "cagra_npartitions_actual": cagra["npartitions_actual"],
    "tagore_npartitions_actual": tagore["npartitions_actual"],
    "matched": actual_match and auto_contract,
}
if not auto_contract:
    errors.append("controlled hot path did not use ACE auto partitioning")
if not actual_match:
    errors.append("partition counts differ for hot path")
record = {
    "schema": "shear-controlled-ace-partition-match-v1",
    "matched": not errors,
    "paths": {"hot": hot},
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if errors:
    raise RuntimeError("; ".join(errors))
PY
}


run_formal_deep100m() {
  exclusive_host_gate
  local timestamp root method
  timestamp="$(date +%Y%m%d_%H%M%S)"
  root="${FORMAL_DEEP100M_RUN_ROOT:-$EVAL_ROOT/p1_p2/deep100m/formal_runs/$timestamp}"
  fresh_dir "$root"
  printf '%s\n' "$root" >"$EVAL_ROOT/latest_deep100m_formal_root.txt"
  cat >"$root/formal_schedule.txt" <<'EOF'
cagra_nn_descent: hot_path/run_1
tagore_cagra: hot_path/run_1
tagore_vamana: hot_path/run_1
EOF
  printf 'ordinal\tmethod\tcache_path\trun_root\n' >"$root/formal_schedule.tsv"
  local ordinal=0
  for method in cagra_nn_descent tagore_cagra tagore_vamana; do
    ordinal=$((ordinal + 1))
    printf '%s\t%s\thot\t%s\n' "$ordinal" "$method" \
      "$root/hot_path/$method/run_1" >>"$root/formal_schedule.tsv"
  done
  for method in cagra_nn_descent tagore_cagra; do
    run_hot_method "$root" "$method" "$ACE_CONTROLLED_FORMAL_PARTITIONS" \
      "$DEEP_BASE" "$DEEP_QUERY" "$DEEP_GT" \
      100000000 96 "25 50 100 200 400" 3 P1_P2 128 5000000 16
  done
  verify_controlled_partition_count_match "$root"
  run_hot_method "$root" tagore_vamana "$ACE_CONTROLLED_FORMAL_PARTITIONS" \
    "$DEEP_BASE" "$DEEP_QUERY" "$DEEP_GT" \
    100000000 96 "25 50 100 200 400" 3 P1_P2 128 5000000 16
  log "Deep100M formal root: $root"
}

run_formal_deep100m_ace_pair() {
  exclusive_host_gate
  local timestamp root method
  timestamp="$(date +%Y%m%d_%H%M%S)"
  root="${ACE_PAIR_RUN_ROOT:-$EVAL_ROOT/p1_p2/deep100m_ace_pair/$timestamp}"
  fresh_dir "$root"
  printf '%s\n' "$root" >"$EVAL_ROOT/latest_deep100m_ace_pair_root.txt"
  cat >"$root/formal_schedule.txt" <<'EOF'
cagra_nn_descent: hot_path/run_1
tagore_cagra: hot_path/run_1
EOF
  printf 'ordinal\tmethod\tcache_path\trun_root\n' >"$root/formal_schedule.tsv"
  local ordinal=0
  for method in cagra_nn_descent tagore_cagra; do
    ordinal=$((ordinal + 1))
    printf '%s\t%s\thot\t%s\n' "$ordinal" "$method" \
      "$root/hot_path/$method/run_1" >>"$root/formal_schedule.tsv"
  done
  for method in cagra_nn_descent tagore_cagra; do
    log "Deep100M ACE hot-only method=$method root=$root"
    run_hot_method "$root" "$method" "$ACE_CONTROLLED_FORMAL_PARTITIONS" \
      "$DEEP_BASE" "$DEEP_QUERY" "$DEEP_GT" \
      100000000 96 "25 50 100 200 400" 3 P1_P2 128 5000000 16
  done
  verify_controlled_partition_count_match "$root"
  log "Deep100M ACE pair root: $root"
}

run_formal_deep100m_cagra_auto() {
  exclusive_host_gate
  local timestamp root
  timestamp="$(date +%Y%m%d_%H%M%S)"
  root="${CAGRA_AUTO_RUN_ROOT:-$EVAL_ROOT/p1_p2/deep100m_cagra_auto/$timestamp}"
  fresh_dir "$root"
  printf '%s\n' "$root" >"$EVAL_ROOT/latest_deep100m_cagra_auto_root.txt"
  printf 'cagra_auto: hot_path/run_1\n' >"$root/formal_schedule.txt"
  printf 'ordinal\tmethod\tcache_path\trun_root\n' >"$root/formal_schedule.tsv"
  printf '1\tcagra_auto\thot\t%s\n' "$root/hot_path/cagra_auto/run_1" >>"$root/formal_schedule.tsv"
  run_hot_method "$root" cagra_auto 0 "$DEEP_BASE" "$DEEP_QUERY" "$DEEP_GT" \
    100000000 96 "25 50 100 200 400" 3 P1_P2 128 5000000 16
  log "Deep100M CAGRA-auto system-level root: $root"
}

usage() {
  cat <<'EOF'
Usage: p1_p2_runbook.sh [smoke|smoke-tagore-vamana|prepare|formal-deep100m|formal-deep100m-ace-pair|formal-deep100m-cagra-auto]
smoke                 Run all three methods on a deterministic 65,536-row fixture.
smoke-tagore-vamana   Run only Tagore-Vamana build, validation, and search on that fixture.
prepare               Verify the shared Release cuVS/toolchain contract and compile the graph validator.
formal-deep100m        Run one controlled warm-cache build/search per DEEP100M method.
formal-deep100m-ace-pair
                       Run the independently partitioned matched E2E hot-only pair.
formal-deep100m-cagra-auto
                       Run one warm-cache CAGRA-auto system-level result.
EOF
}

case "$MODE" in
  smoke) verify_prerequisites; run_smoke ;;
  smoke-tagore-vamana) verify_tagore_vamana_prerequisites; run_smoke_tagore_vamana ;;
  prepare) verify_prerequisites; log "prerequisites passed" ;;
  formal-deep100m) verify_prerequisites; run_formal_deep100m ;;
  formal-deep100m-ace-pair) verify_prerequisites; run_formal_deep100m_ace_pair ;;
  formal-deep100m-cagra-auto) verify_prerequisites; run_formal_deep100m_cagra_auto ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
```
<!-- END RUNBOOK_SCRIPT -->

## Smoke Gate

The default smoke uses one deterministic `65536 x 32` float32 dataset, 100
queries, and exact GT10. All three methods use final degree 64 and common `L=100`
search. Tagore-Vamana smoke alone uses eight clusters, `max_points=32768`, and
buffer size four. Every smoke row is `SMOKE_ONLY` and `paper_eligible=false`.

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re
source = Path("docs/runbooks/p1_p2_shear_cagra_tagore.md").read_text()
begin = "<!-- BEGIN " + "RUNBOOK_SCRIPT -->"
end = "<!-- END " + "RUNBOOK_SCRIPT -->"
block = source.split("\n" + begin + "\n", 1)[1].split("\n" + end + "\n", 1)[0]
script = re.search(r"```bash\n(.*?)\n```", block, re.S).group(1)
Path("/tmp/p1_p2_shear_cagra_tagore.sh").write_text(script + "\n")
PY
chmod +x /tmp/p1_p2_shear_cagra_tagore.sh
/tmp/p1_p2_shear_cagra_tagore.sh smoke-tagore-vamana
```

## Method A: controlled cuVS ACE/CAGRA-NN-Descent

The controlled CAGRA row has `comparison_class=matched_end_to_end`. It starts
from the original vectors, requests ACE automatic partitioning with
`npartitions=0`, fixes degree 128 to 64, and explicitly forces the local CAGRA
solver to NN-Descent. With `ef_construction=176`, the pinned mapping yields 16
iterations. `partition_count_audit.json` records the positive actual count;
a missing forced-solver record or any `solver=ivf_pq` rejects the row.

## Method B: Tagore-CAGRA

Tagore-CAGRA independently executes the same ACE automatic partition policy,
then uses Tagore for local graph construction. The controlled pair is accepted
only when its actual counts match on the hot path. It fixes
16 iterations and seed 20260713 and uses the persistent workspace plugin.
cuVS NN-Descent does not expose an external construction seed: its pinned
implementation initializes deterministically from each segment index. The
runbook therefore records both seed contracts but does not claim identical
random initial graphs. Common ACE stages, validation, remapping, persistence,
monitoring, and search remain identical.

## Method C: Tagore-Vamana

Tagore-Vamana includes DiskANN-compatible two-assignment partitioning, bounded
NumPy-memmap conversion, single-GPU `Tagore.largeIndex`, CPU merge, local-index
eviction, and final persistence. DEEP100M fixes 128 clusters,
`max_points=5000000`, and `buffer_size=16`. The measured pipeline receives
eight OpenMP threads and single-threaded BLAS; the Tagore algorithm and its
native merge implementation are otherwise unchanged.

## Separate system-level result: CAGRA-auto

The optional CAGRA-auto row has `comparison_class=system_level_auto`. It keeps
the native `npartitions=0`, `ef_construction=120`, and plan-disabled solver
heuristic, so some partitions may use IVF-PQ. It runs only through
`formal-deep100m-cagra-auto` and must not be placed in the controlled local
builder table.

## Structural Validation

The strict validator scans every node and edge, checking header, degree and ID
bounds, self edges, duplicates, file length, and start node. `graph_quality`
records reachability and sampled graph diagnostics outside the corresponding
measured build wall.

## P2 Search Sweep

DEEP100M uses `25 50 100 200 400`, eight threads, `--storage mmap`, one warm-up,
and three measured repetitions. Recall@10 must be stable and nondecreasing.
Matched-recall interpolation is allowed only inside measured brackets. The
first `L=25` warm-up starts from the declared prewarmed state;
the three measured repetitions are post-warmup steady-state measurements.

## DEEP100M Formal Schedule

`formal-deep100m` requires an exclusive GPU and passwordless cache control.
For each method it writes one `hot_path/<method>/run_1` root. Every measured
build drops page cache and prewarms only the original base. Every measured
search drops page cache and prewarms the base, final graph, query, and GT.
Prewarm time is outside P1/P2. Each root must be new. Failures are retained and
never automatically retried, resumed, or repaired by changing parameters.

## Result Aggregation

Each accepted root contains status, time, process-tree RSS, temporary storage,
GPU monitoring, graph validation/quality, pre-cleanup index size, search
summaries, and a cleanup manifest/summary.
Report the DEEP warm-cache build for each method. For each search-list size,
report the mean, standard deviation, and min--max across the three measured
searches; do not calculate paper values from incomplete rows. These rows are
steady warm-cache results, not first-run latency measurements.
Page cache is system memory outside process RSS. Use the existing paired
cache-path evidence, not this hot-only schedule, for cache-sensitivity claims.
The CAGRA-NN-Descent and Tagore-CAGRA rows are matched end-to-end walls. Keep
the separately labeled `system_level_auto` row distinct because its partition
count, construction effort, and solver selection are automatic.

## Failure and Cleanup Rules

On failure, retain commands, logs, profiles, partitions, local indexes, monitor
traces, and partial output. Never automatically retry or delete failure
evidence. Every hot run performs cleanup immediately after its
successful search sweep. It records target sizes, removes its final index and
method-specific large intermediate data, and verifies both target absence and
retained timing, validation, and search evidence. The run is not marked
accepted, and the scheduler does not start the next run, until this cleanup gate
passes. A cleanup failure records `failed_stage=cleanup` and stops the schedule.

## Acceptance Checklist

- All three smoke paths complete, validate, and search.
- Every formal method has one accepted `hot_path/<method>/run_1` with a fresh
  root. Controlled CAGRA and Tagore-CAGRA each start from the original vectors
  and independently execute ACE partitioning.
- Every formal run records `cache_path=hot` in matching `cache_path.txt`,
  `status.json`, directory,
  `formal_schedule.tsv`, and `cache_protocol.tsv` evidence. The measured build
  prewarms only the base; measured search prewarms base, final graph, query,
  and GT.
- Controlled CAGRA and Tagore-CAGRA record `TAGORE_ACE_PLAN_MODE=disabled`,
  `timing_scope=from_scratch`, `npartitions_requested=0`,
  `TAGORE_ACE_EXECUTION_MODE=pipelined`, and the correct local builder.
- Every controlled ACE run records a positive actual count in
  `partition_count_audit.json`; `controlled_partition_count_match.json`
  proves that CAGRA and Tagore-CAGRA match on the hot path.
- Controlled CAGRA records `TAGORE_ACE_CAGRA_LOCAL_SOLVER=nn-descent`, only
  `solver=nn_descent`, and exactly 16 iterations from `ef_construction=176`.
  Tagore-CAGRA keeps the CAGRA-solver option at `auto` because its local
  builder is the Tagore plugin.
- Every CAGRA and Tagore-CAGRA row contains `cuvs_release_contract.txt`; its
  Release/O3/OpenMP/SM89 contract passes, and both methods record identical
  installed `libcuvs.so` and `libcuvs_c.so` hashes and ELF build IDs.
- Tagore-CAGRA fixes `TAGORE_CAGRA_ITERATIONS=16` and
  `TAGORE_CAGRA_SEED=20260713`.
- CAGRA-NN-Descent, Tagore-CAGRA, and Tagore-Vamana command records show OMP 8
  and BLAS 1. CAGRA-auto, when run, is reported separately.
- Every reported build passes graph validation and common search.
- Every accepted hot run contains `cleanup_manifest.tsv` and
  `cleanup_summary.txt`, the summary contains `verified=true`, and every target
  named in the manifest no longer exists.
- No primary controlled schedule creates or replays a frozen partition plan;
  each failed method retains its own from-scratch partition evidence.
