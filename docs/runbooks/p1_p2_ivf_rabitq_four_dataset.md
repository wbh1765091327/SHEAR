# IVF-RaBitQ 四数据集 P1/P2 Runbook

本 runbook 在同一套 hot-only、drop-then-prewarm 协议下测量 Wiki35M、LAION100M、
MSMARCO70M 和 DEEP100M。P1 使用 GPU 分批构建 RaBitQ-Library 原生 CPU IVF 索引；P2
使用 CPU 的八 query-worker search，单条 query 内部串行，无 rerank。

## 固定实验合同

```text
primary_build_metric=gpu_build_seconds
secondary_build_metrics=end_to_end_wall_seconds,external_wall_seconds
quantization=exact
bits_per_dim=7
search_backend=RaBitQ-Library_CPU
query_parallelism=8
per_query_parallelism=1
centroid_hnsw_build_parallelism=8
rerank=false
cache_path=hot_only
```

`gpu_build_seconds` 只等于 K-means、全量 assignment 和 GPU quantization 三个 GPU 阶段；
`end_to_end_wall_seconds` 是 builder 内从 `.fbin` 到 native index、sidecar 和验证的独立 wall；
`external_wall_seconds` 来自外层 `/usr/bin/time -v`。三个口径必须分列，不能相加后仍命名为
GPU build。RaBitQ-Library 的 centroid HNSW 插入并发固定上限为 8；这只影响包含 sidecar 的
端到端时间，不计入 `gpu_build_seconds`。

| Dataset | Rows | Dim | nlist | Base | Query | Ground truth |
|---|---:|---:|---:|---|---|---|
| Wiki35M | 35,000,000 | 1024 | 32,768 | `wiki-1024-35M/en_base.fbin` | `wiki-1024-35M/query.fbin` | `wiki-1024-35M/gt100.bin` |
| LAION100M | 100,000,000 | 512 | 65,536 | `laion-512-100M/base.fbin` | `laion-512-100M/query.fbin` | `laion-512-100M/gt100.bin` |
| MSMARCO70M | 70,000,000 | 768 | 65,536 | `msmarco-768-70M/base.fbin` | `msmarco-768-114M/query.fbin` | `msmarco-768-70M/gt100.bin` |
| DEEP100M | 100,000,000 | 96 | 131,072 | `deep-96-100M/base.100M.fbin` | `deep-96-100M/query.public.10K.fbin` | `deep-96-100M/groundtruth.public.10K.ibin` |

四个 formal build 都固定为 `bits=7`、`niter=20`、
`max_points_per_cluster=64`、`batch_vectors=131072`、`fast_quantize=false`。每条正式
Recall--QPS 曲线保留核心 `nprobe=64,128,256,512`，不足时按 1.5 倍向下/向上扩展，直到
最低点 Recall@10≤0.85 且最高点 Recall@10≥0.97；上限为 4096。discovery 每点一次，
formal 每点三次，均使用全部 10,000 queries、100-query warm-up 和 8 个 query worker。

## 模式与输出

```text
help
prepare
smoke
formal
wiki35m | laion100m | msmarco70m | deep100m
```

默认正式输出为：

```text
/data_staging/wbh/ivf_rabitq_four_dataset/runs/<timestamp>/hot_path/<dataset>/ivf_rabitq_exact/run_1
```

`prepare` 只编译、测试和记录 provenance；`smoke` 使用确定性的 65,536×96 fixture，标记
`paper_eligible=false`，不 drop page cache。formal 失败后不重试、不改参数、不清理现场；
成功后先记录主索引和 HNSW sidecar 的 size/SHA-256，再只删除当前 run root 拥有的大型索引。
若 `nprobe=1` 仍无法覆盖低端，状态为 `quality_lower_bound_not_reached`；若 4096 仍无法覆盖
高端，状态为 `quality_upper_bound_not_reached`，两者都不会启动正式 P2。

提取并运行 runner：

```bash
awk '/^[<]!-- BEGIN RUNBOOK_SCRIPT --[>]$/{on=1;next}/^[<]!-- END RUNBOOK_SCRIPT --[>]$/{on=0}on' \
  docs/runbooks/p1_p2_ivf_rabitq_four_dataset.md | sed '1d;$d' | bash -s -- help
```

<!-- BEGIN RUNBOOK_SCRIPT -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit
export LC_ALL=C

MODE="${1:-help}"
RUNNER_BASHPID="$BASHPID"
DATA_ROOT="${DATA_ROOT:-/data_nvme_kioxia/tyj/data}"
EVAL_ROOT="${EVAL_ROOT:-/data_staging/wbh/ivf_rabitq_four_dataset}"
CUVS_ROOT="${CUVS_ROOT:-/path/to/cuvs_rabitq}"
RABITQ_ROOT="${RABITQ_ROOT:-/path/to/RaBitQ-Library}"
SHEAR_ROOT="${SHEAR_ROOT:-/path/to/SHEAR}"
CUVS_CMAKE="${CUVS_CMAKE:-/path/to/cmake}"
RUNBOOK_PATH="$SHEAR_ROOT/docs/runbooks/p1_p2_ivf_rabitq_four_dataset.md"
RUNBOOK_TEST="$SHEAR_ROOT/script/test_ivf_rabitq_four_dataset_runbook.py"
GPU_BUILDER="$CUVS_ROOT/cpp/build/ivf_rabitq_cpu_index_build"
CPU_SEARCH="$RABITQ_ROOT/bin/ivf_rabitq_fbin_benchmark"
FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)
CORE_NPROBES="64,128,256,512"
LOW_RECALL=0.85
HIGH_RECALL=0.97
MAX_NPROBE=4096
LOCK_FILE="${LOCK_FILE:-/data_staging/wbh/.ivf_rabitq_four_dataset.lock}"
BATCH_ROOT="${BATCH_ROOT:-}"
BATCH_KIND=""
RUN_ROOT=""
DATASET=""
CURRENT_STAGE=""
PAPER_ELIGIBLE=true
EXPERIMENT_ROLE=formal
EXPECT_HNSW=true
FAILURE_MESSAGE=""
ACTIVE_BUILD_PGID=""
ACTIVE_MONITOR_PIDS=()
EXPERIMENT_LOCK_FD=""

usage() {
  cat <<'USAGE'
Usage: ivf_rabitq_four_dataset.sh MODE

Modes:
  help        Show this contract without checking tools or data.
  prepare     Configure, build, test, and freeze provenance.
  smoke       Run deterministic 65,536-row exact-build/native-search smoke.
  formal      Run Wiki35M, LAION100M, MSMARCO70M, then DEEP100M.
  wiki35m     Run only Wiki35M in a fresh batch root.
  laion100m   Run only LAION100M in a fresh batch root.
  msmarco70m  Run only MSMARCO70M in a fresh batch root.
  deep100m    Run only DEEP100M in a fresh batch root.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

die() {
  log "ERROR: $*"
  return 1
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "required executable is missing: $1"
}

require_commands() {
  local command_name
  for command_name in "$@"; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command is missing: $command_name"
  done
}

fresh_dir() {
  [[ ! -e "$1" ]] || die "refusing to overwrite existing path: $1"
  mkdir -p "$1"
}

cleanup_monitors() {
  local pid
  if [[ -n "$ACTIVE_BUILD_PGID" ]]; then
    kill -TERM -- "-$ACTIVE_BUILD_PGID" 2>/dev/null || true
  fi
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_MONITOR_PIDS=()
  ACTIVE_BUILD_PGID=""
}

write_status() {
  local path="$1" phase="$2" exit_code="$3" failed_stage="${4:-}" message="${5:-}"
  local paper_eligible="${6:-$PAPER_ELIGIBLE}"
  mkdir -p "$(dirname "$path")"
  python3 - "$path" "$phase" "$exit_code" "$failed_stage" "$message" "$paper_eligible" <<'PY_WRITE_STATUS'
import datetime
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
phase = sys.argv[2]
exit_code = int(sys.argv[3])
failed_stage = sys.argv[4] or None
message = sys.argv[5] or None
paper_eligible = sys.argv[6].lower() == "true"
if path.exists():
    value = json.loads(path.read_text())
else:
    value = {"schema": "ivf_rabitq_four_dataset_status_v1", "history": []}
event = {
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": phase,
    "exit_code": exit_code,
    "failed_stage": failed_stage,
    "message": message,
}
value.update(event)
value["paper_eligible"] = paper_eligible
value["history"].append(event)
temporary = path.with_suffix(path.suffix + ".partial")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY_WRITE_STATUS
}

update_schedule() {
  local dataset="$1" state="$2" reason="${3:-}"
  [[ -n "$BATCH_ROOT" && -f "$BATCH_ROOT/formal_schedule.tsv" ]] || return 0
  python3 - "$BATCH_ROOT/formal_schedule.tsv" "$dataset" "$state" "$reason" <<'PY_UPDATE_SCHEDULE'
import csv
import datetime
from pathlib import Path
import sys

path = Path(sys.argv[1])
dataset, state, reason = sys.argv[2:5]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
with path.open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
for row in rows:
    if row["dataset"] != dataset:
        continue
    row["state"] = state
    if state == "running" and not row["started_at"]:
        row["started_at"] = now
    if state in {"accepted", "failed"}:
        row["ended_at"] = now
    row["reason"] = reason
    break
else:
    raise SystemExit(f"dataset absent from schedule: {dataset}")
temporary = path.with_suffix(path.suffix + ".partial")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=("dataset", "ordinal", "state", "run_root", "started_at", "ended_at", "reason"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(path)
PY_UPDATE_SCHEDULE
}

generate_batch_summary() {
  [[ -n "$BATCH_ROOT" && -d "$BATCH_ROOT" ]] || return 0
  python3 - "$BATCH_ROOT" <<'PY_BATCH_SUMMARY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
datasets = ("wiki35m", "laion100m", "msmarco70m", "deep100m", "smoke")
build_fields = (
    "dataset", "rows", "dim", "nlist", "bits_per_dim", "fast_quantize",
    "kmeans_gpu_seconds", "assignment_gpu_seconds", "quantization_gpu_seconds",
    "gpu_build_seconds", "host_gather_seconds", "conversion_seconds",
    "validation_seconds", "end_to_end_wall_seconds", "external_wall_seconds",
    "peak_process_tree_rss_bytes", "peak_hbm_bytes", "main_index_bytes", "hnsw_index_bytes",
    "status",
)
build_rows = []
search_rows = []
for dataset in datasets:
    run = root / "hot_path" / dataset / "ivf_rabitq_exact" / "run_1"
    if not run.exists():
        continue
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    metrics = json.loads((run / "build_metrics.json").read_text()) if (run / "build_metrics.json").exists() else {}
    external = json.loads((run / "external_time.json").read_text()) if (run / "external_time.json").exists() else {}
    resources = json.loads((run / "resource_summary.json").read_text()) if (run / "resource_summary.json").exists() else {}
    params, timings = metrics.get("params", {}), metrics.get("timings", {})
    output = metrics.get("output", {})
    build_rows.append({
        "dataset": dataset,
        "rows": metrics.get("dataset", {}).get("rows", ""),
        "dim": metrics.get("dataset", {}).get("dim", ""),
        "nlist": params.get("nlist", ""),
        "bits_per_dim": params.get("bits_per_dim", ""),
        "fast_quantize": params.get("fast_quantize", ""),
        **{name: timings.get(name, "") for name in build_fields if name.endswith("_seconds")},
        "external_wall_seconds": external.get("external_wall_seconds", ""),
        "peak_process_tree_rss_bytes": resources.get("peak_process_tree_rss_bytes", ""),
        "peak_hbm_bytes": resources.get("peak_hbm_bytes", ""),
        "main_index_bytes": output.get("main_index_bytes", ""),
        "hnsw_index_bytes": output.get("hnsw_index_bytes", ""),
        "status": status.get("phase", "missing"),
    })
    summary = run / "search_summary.tsv"
    if summary.exists():
        with summary.open(newline="") as stream:
            search_rows.extend(csv.DictReader(stream, delimiter="\t"))

def write_tsv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)

write_tsv(root / "build_summary.tsv", build_fields, build_rows)
search_fields = (
    "dataset", "nprobe", "recall_at_10", "qps_mean", "qps_sample_std",
    "qps_median", "qps_min", "qps_max", "repeat_count", "query_workers", "rerank",
)
write_tsv(root / "search_summary.tsv", search_fields, search_rows)
lines = [
    "# IVF-RaBitQ four-dataset summary", "",
    "Primary IVF build metric: `gpu_build_seconds` (GPU K-means + assignment + quantization).",
    "Secondary deployment metrics: builder `end_to_end_wall_seconds` and external `/usr/bin/time` wall.",
    "Search: CPU RaBitQ-Library, eight query workers, per-query serial, rerank=false.", "",
    "## Dataset state", "", "| Dataset | State |", "|---|---|",
]
for row in build_rows:
    lines.append(f"| {row['dataset']} | {row['status']} |")
lines.append("")
(root / "summary.md").write_text("\n".join(lines))
PY_BATCH_SUMMARY
}

on_error() {
  local exit_code=$?
  if [[ "$BASHPID" != "$RUNNER_BASHPID" ]]; then
    return "$exit_code"
  fi
  trap - ERR INT TERM
  cleanup_monitors
  if [[ -n "$RUN_ROOT" && -d "$RUN_ROOT" ]]; then
    write_status "$RUN_ROOT/status.json" failed "$exit_code" "${CURRENT_STAGE:-unknown}" "${FAILURE_MESSAGE:-command failed}" || true
    [[ -z "$DATASET" ]] || update_schedule "$DATASET" failed "${CURRENT_STAGE:-unknown}" || true
  fi
  generate_batch_summary || true
  exit "$exit_code"
}

trap on_error ERR
trap 'CURRENT_STAGE=interrupted; false' INT TERM

load_dataset() {
  DATASET="$1"
  QUERY_ROWS=10000
  GT_ROWS=10000
  case "$DATASET" in
    wiki35m)
      ROWS=35000000 DIM=1024 NLIST=32768
      BASE="$DATA_ROOT/wiki-1024-35M/en_base.fbin"
      QUERY="$DATA_ROOT/wiki-1024-35M/query.fbin"
      GT="$DATA_ROOT/wiki-1024-35M/gt100.bin"
      ;;
    laion100m)
      ROWS=100000000 DIM=512 NLIST=65536
      BASE="$DATA_ROOT/laion-512-100M/base.fbin"
      QUERY="$DATA_ROOT/laion-512-100M/query.fbin"
      GT="$DATA_ROOT/laion-512-100M/gt100.bin"
      ;;
    msmarco70m)
      ROWS=70000000 DIM=768 NLIST=65536
      BASE="$DATA_ROOT/msmarco-768-70M/base.fbin"
      QUERY="$DATA_ROOT/msmarco-768-114M/query.fbin"
      GT="$DATA_ROOT/msmarco-768-70M/gt100.bin"
      ;;
    deep100m)
      ROWS=100000000 DIM=96 NLIST=131072
      BASE="$DATA_ROOT/deep-96-100M/base.100M.fbin"
      QUERY="$DATA_ROOT/deep-96-100M/query.public.10K.fbin"
      GT="$DATA_ROOT/deep-96-100M/groundtruth.public.10K.ibin"
      ;;
    *) die "unknown dataset: $DATASET" ;;
  esac
  RUN_ROOT="$BATCH_ROOT/hot_path/$DATASET/ivf_rabitq_exact/run_1"
  INDEX_DIR="$RUN_ROOT/index"
  INDEX_FILE="$INDEX_DIR/ivf_rabitq.cpu.index"
  HNSW_FILE="$INDEX_FILE.hnsw"
  PAPER_ELIGIBLE=true
  EXPERIMENT_ROLE=formal
  EXPECT_HNSW=true
}

initialize_schedule() {
  local ordinal=0 dataset
  printf 'dataset\tordinal\tstate\trun_root\tstarted_at\tended_at\treason\n' >"$BATCH_ROOT/formal_schedule.tsv"
  for dataset in "$@"; do
    ordinal=$((ordinal + 1))
    printf '%s\t%s\tpending\t%s\t\t\t\n' \
      "$dataset" "$ordinal" "$BATCH_ROOT/hot_path/$dataset/ivf_rabitq_exact/run_1" \
      >>"$BATCH_ROOT/formal_schedule.tsv"
  done
}

record_repository_provenance() {
  local root="$BATCH_ROOT/provenance" repository
  mkdir -p "$root"
  for repository in "$RABITQ_ROOT" "$CUVS_ROOT" "$SHEAR_ROOT"; do
    local name
    name="$(basename "$repository")"
    git -C "$repository" rev-parse HEAD >"$root/${name}_git_head.txt"
    git -C "$repository" status --short >"$root/${name}_git_status.txt"
    git -C "$repository" diff --binary >"$root/${name}_dirty.diff"
  done
  {
    sha256sum "$GPU_BUILDER" "$CPU_SEARCH" "$RUNBOOK_PATH" "$RUNBOOK_TEST"
  } >"$root/binary_hashes.txt"
  {
    printf '## GPU builder\n'
    ldd "$GPU_BUILDER"
    printf '\n## CPU search\n'
    ldd "$CPU_SEARCH"
  } >"$root/ldd.txt" 2>&1
  cp "$CUVS_ROOT/cpp/build/CMakeCache.txt" "$root/cuvs_CMakeCache.txt"
  cp "$RABITQ_ROOT/build/CMakeCache.txt" "$root/rabitq_CMakeCache.txt"
  nvidia-smi -q >"$root/nvidia_smi_q.txt" 2>&1
  lscpu >"$root/lscpu.txt" 2>&1
}

record_environment() {
  {
    date --iso-8601=seconds
    uname -a
    lscpu
    free -h
    cat /proc/meminfo
    nvidia-smi
    mount
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
  } >"$RUN_ROOT/environment.txt" 2>&1
  cp "$BATCH_ROOT/provenance/binary_hashes.txt" "$RUN_ROOT/binary_hashes.txt"
}

verify_tool_prerequisites() {
  require_commands bash python3 git cmake ctest sha256sum dd sync sudo nvidia-smi \
    stat du awk sed grep find flock setsid ps vmstat pidstat iostat realpath /usr/bin/time
  python3 -c 'import numpy' >/dev/null
  require_executable "$CUVS_CMAKE"
  require_executable "$GPU_BUILDER"
  require_executable "$CPU_SEARCH"
  "$CPU_SEARCH" --help | grep -F -- '--nprobes-exact' >/dev/null \
    || die "CPU search binary does not support --nprobes-exact"
}

verify_gpu_idle() {
  local active
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$active" ]] || die "GPU 0 has active compute PIDs: $active"
}

verify_output_space() {
  local required available
  required=$((ROWS * DIM + ROWS * DIM / 4 + 10 * 1024 * 1024 * 1024))
  available="$(df -PB1 "$EVAL_ROOT" | awk 'NR==2 {print $4}')"
  ((available >= required)) || die "insufficient output space: need=$required available=$available"
}

verify_formal_prerequisites() {
  verify_tool_prerequisites
  require_file "$BASE"
  require_file "$QUERY"
  require_file "$GT"
  verify_gpu_idle
  sudo -n true >/dev/null || die "passwordless sudo is required for formal cache drop"
  mkdir -p "$EVAL_ROOT"
  verify_output_space
}

verify_smoke_prerequisites() {
  verify_tool_prerequisites
  verify_gpu_idle
  mkdir -p "$EVAL_ROOT/smoke"
}

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_FILE")"
  exec {EXPERIMENT_LOCK_FD}>"$LOCK_FILE"
  flock -n "$EXPERIMENT_LOCK_FD" || die "another IVF-RaBitQ batch holds $LOCK_FILE"
}

initialize_batch_root() {
  local kind="$1"
  fresh_dir "$BATCH_ROOT"
  BATCH_KIND="$kind"
  mkdir -p "$BATCH_ROOT/provenance"
  printf '%s\n' "$kind" >"$BATCH_ROOT/requested_mode.txt"
  record_repository_provenance
}

record_cache_state() {
  local phase="$1" label="$2" root="$3"
  [[ -f "$root/cache_protocol.tsv" ]] \
    || printf 'sample_epoch\tphase\tlabel\tmetric\tvalue_kib\n' >"$root/cache_protocol.tsv"
  awk -v epoch="$(date +%s.%N)" -v phase="$phase" -v label="$label" '
    /^(MemFree|MemAvailable|Cached|Buffers|SReclaimable):/ {
      key=$1; sub(":$", "", key); printf "%s\t%s\t%s\t%s\t%s\n", epoch, phase, label, key, $2
    }' /proc/meminfo >>"$root/cache_protocol.tsv"
}

drop_page_cache() {
  local phase="$1" root="$2"
  record_cache_state "$phase" before_drop "$root"
  sync
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  record_cache_state "$phase" after_drop "$root"
}

prewarm_paths() {
  local phase="$1" root="$2"
  shift 2
  local path bytes started ended timing
  printf 'phase\tpath\tbytes\tstarted_epoch_s\tended_epoch_s\tstatus\n' \
    >"$root/${phase}_prewarm_manifest.tsv"
  timing="$root/${phase}_prewarm.time.log"
  : >"$timing"
  for path in "$@"; do
    require_file "$path"
    bytes="$(stat -Lc %s -- "$path")"
    started="$(date +%s.%N)"
    /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M' -o "$timing" -a \
      dd if="$path" of=/dev/null bs=64M status=none
    ended="$(date +%s.%N)"
    printf '%s\t%s\t%s\t%s\t%s\tcomplete\n' \
      "$phase" "$(realpath -e "$path")" "$bytes" "$started" "$ended" \
      >>"$root/${phase}_prewarm_manifest.tsv"
  done
  record_cache_state "$phase" after_warm "$root"
}

prepare_build_cache() {
  drop_page_cache build "$RUN_ROOT"
  prewarm_paths build "$RUN_ROOT" "$BASE"
}

prepare_search_cache() {
  local phase="$1"
  drop_page_cache "$phase" "$RUN_ROOT"
  prewarm_paths "$phase" "$RUN_ROOT" \
    "$INDEX_FILE" "$HNSW_FILE" "$QUERY" "$GT"
}

validate_dataset() {
  python3 - "$BASE" "$QUERY" "$GT" "$ROWS" "$DIM" "$QUERY_ROWS" "$GT_ROWS" \
    "$RUN_ROOT/dataset_manifest.json" <<'PY_DATASET_AUDIT'
import hashlib
import json
from pathlib import Path
import struct
import sys

base, query, groundtruth = map(Path, sys.argv[1:4])
rows, dim, query_rows, gt_rows = map(int, sys.argv[4:8])
output = Path(sys.argv[8])

def header(path, item_bytes):
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise SystemExit(f"short header: {path}")
    matrix_rows, matrix_dim = struct.unpack("<II", raw)
    minimum = 8 + matrix_rows * matrix_dim * item_bytes
    if path.stat().st_size < minimum:
        raise SystemExit(f"short payload: {path}: minimum={minimum} actual={path.stat().st_size}")
    return matrix_rows, matrix_dim, minimum

base_rows, base_dim, base_bytes = header(base, 4)
query_actual_rows, query_dim, query_bytes = header(query, 4)
gt_actual_rows, gt_width, gt_id_bytes = header(groundtruth, 4)
if (base_rows, base_dim) != (rows, dim) or base.stat().st_size != base_bytes:
    raise SystemExit(f"base shape/size mismatch: {(base_rows, base_dim, base.stat().st_size)}")
if (query_actual_rows, query_dim) != (query_rows, dim) or query.stat().st_size != query_bytes:
    raise SystemExit(f"query shape/size mismatch: {(query_actual_rows, query_dim, query.stat().st_size)}")
if gt_actual_rows != gt_rows or gt_width < 10:
    raise SystemExit(f"GT shape mismatch: {(gt_actual_rows, gt_width)}")
valid_gt_sizes = {gt_id_bytes, 8 + 2 * gt_actual_rows * gt_width * 4}
if groundtruth.stat().st_size not in valid_gt_sizes:
    raise SystemExit(f"GT size mismatch: {groundtruth.stat().st_size}")

maximum_id = -1
remaining = gt_actual_rows * gt_width
with groundtruth.open("rb") as stream:
    stream.seek(8)
    while remaining:
        count = min(remaining, 1_048_576)
        payload = stream.read(count * 4)
        if len(payload) != count * 4:
            raise SystemExit("short GT ID payload")
        for (identifier,) in struct.iter_unpack("<I", payload):
            if identifier >= rows:
                raise SystemExit(f"GT id {identifier} is outside [0,{rows})")
            maximum_id = max(maximum_id, identifier)
        remaining -= count

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = {
    "schema": "ivf_rabitq_dataset_manifest_v1",
    "base": {"path": str(base.resolve()), "rows": rows, "dim": dim, "bytes": base.stat().st_size},
    "query": {
        "path": str(query.resolve()), "rows": query_rows, "dim": dim,
        "bytes": query.stat().st_size, "sha256": sha256(query),
    },
    "ground_truth": {
        "path": str(groundtruth.resolve()), "rows": gt_actual_rows, "width": gt_width,
        "bytes": groundtruth.stat().st_size, "maximum_id": maximum_id,
        "sha256": sha256(groundtruth),
    },
}
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY_DATASET_AUDIT
}

sample_process_group() {
  local stop_file="$1" pgid="$2" output="$3"
  printf 'sample_epoch\tpgid\tmembers\trss_kib\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    ps -eo pgid=,rss= | awk -v epoch="$(date +%s.%N)" -v wanted="$pgid" '
      $1 == wanted {members += 1; rss += $2}
      END {printf "%s\t%s\t%s\t%s\n", epoch, wanted, members + 0, rss + 0}
    ' >>"$output"
    sleep 1
  done
}

sample_gpu_memory() {
  local stop_file="$1" output="$2"
  printf 'sample_epoch\tmemory_used_mib\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    local used
    used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    printf '%s\t%s\n' "$(date +%s.%N)" "$used" >>"$output"
    sleep 1
  done
}

sample_meminfo() {
  local stop_file="$1" output="$2"
  printf 'sample_epoch\tMemTotal_kB\tMemFree_kB\tMemAvailable_kB\tBuffers_kB\tCached_kB\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    awk -v epoch="$(date +%s.%N)" '
      /^MemTotal:/ {total=$2} /^MemFree:/ {free=$2} /^MemAvailable:/ {available=$2}
      /^Buffers:/ {buffers=$2} /^Cached:/ {cached=$2}
      END {printf "%s\t%s\t%s\t%s\t%s\t%s\n",epoch,total,free,available,buffers,cached}
    ' /proc/meminfo >>"$output"
    sleep 1
  done
}

sample_temp_usage() {
  local stop_file="$1" output="$2"
  printf 'sample_epoch\tallocated_bytes\tapparent_bytes\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    local allocated apparent
    allocated="$(du -s -B1 "$RUN_ROOT" 2>/dev/null | awk '{print $1}' || true)"
    apparent="$(du -sb "$RUN_ROOT" 2>/dev/null | awk '{print $1}' || true)"
    printf '%s\t%s\t%s\n' "$(date +%s.%N)" "${allocated:-0}" "${apparent:-0}" >>"$output"
    sleep 1
  done
}

start_build_monitors() {
  local pgid="$1" stop_file="$RUN_ROOT/.monitor_stop"
  rm -f -- "$stop_file"
  mkdir -p "$RUN_ROOT/gpu_monitor"
  sample_process_group "$stop_file" "$pgid" "$RUN_ROOT/process_tree_rss.tsv" &
  ACTIVE_MONITOR_PIDS+=("$!")
  sample_gpu_memory "$stop_file" "$RUN_ROOT/gpu_monitor/gpu.tsv" &
  ACTIVE_MONITOR_PIDS+=("$!")
  sample_meminfo "$stop_file" "$RUN_ROOT/meminfo.tsv" &
  ACTIVE_MONITOR_PIDS+=("$!")
  sample_temp_usage "$stop_file" "$RUN_ROOT/temp_usage.tsv" &
  ACTIVE_MONITOR_PIDS+=("$!")
  vmstat 1 >"$RUN_ROOT/vmstat.log" & ACTIVE_MONITOR_PIDS+=("$!")
  pidstat -p ALL 1 >"$RUN_ROOT/pidstat.log" & ACTIVE_MONITOR_PIDS+=("$!")
  iostat -dx 1 >"$RUN_ROOT/iostat.log" & ACTIVE_MONITOR_PIDS+=("$!")
}

stop_build_monitors() {
  local pid
  touch "$RUN_ROOT/.monitor_stop"
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_MONITOR_PIDS=()
  rm -f -- "$RUN_ROOT/.monitor_stop"
}

write_command() {
  local output="$1"
  shift
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n' >"$output"
  printf '%q ' "$@" >>"$output"
  printf '\n' >>"$output"
  chmod +x "$output"
}

parse_external_time() {
  python3 - "$RUN_ROOT/time.log" "$RUN_ROOT/external_time.json" <<'PY_EXTERNAL_TIME'
import json
import re
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text()
match = re.search(r"Elapsed .*?:\s*([0-9:.]+)\s*$", text, re.MULTILINE)
if not match:
    raise SystemExit("external elapsed wall is absent")
parts = [float(value) for value in match.group(1).split(":")]
seconds = 0.0
for value in parts:
    seconds = seconds * 60.0 + value
rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
value = {
    "schema": "ivf_rabitq_external_time_v1",
    "external_wall_seconds": seconds,
    "time_max_rss_kib": int(rss.group(1)) if rss else None,
}
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY_EXTERNAL_TIME
}

summarize_build_resources() {
  python3 - "$RUN_ROOT/process_tree_rss.tsv" "$RUN_ROOT/gpu_monitor/gpu.tsv" \
    "$RUN_ROOT/temp_usage.tsv" "$RUN_ROOT/resource_summary.json" <<'PY_RESOURCE_SUMMARY'
import csv
import json
from pathlib import Path
import sys

rss_path, gpu_path, temp_path, output = map(Path, sys.argv[1:5])
def maximum(path, column):
    with path.open(newline="") as stream:
        values = [int(float(row[column])) for row in csv.DictReader(stream, delimiter="\t")]
    return max(values, default=0)
value = {
    "schema": "ivf_rabitq_resource_summary_v1",
    "peak_process_tree_rss_bytes": maximum(rss_path, "rss_kib") * 1024,
    "peak_hbm_bytes": maximum(gpu_path, "memory_used_mib") * 1024 * 1024,
    "peak_temp_allocated_bytes": maximum(temp_path, "allocated_bytes"),
    "peak_temp_apparent_bytes": maximum(temp_path, "apparent_bytes"),
}
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY_RESOURCE_SUMMARY
}

run_gpu_build() {
  local niter=20 batch_vectors=131072
  if [[ "$EXPERIMENT_ROLE" == smoke ]]; then
    niter=5 batch_vectors=8192
  fi
  mkdir -p "$INDEX_DIR"
  local -a command=(
    "$GPU_BUILDER"
    --base "$BASE"
    --output "$INDEX_FILE"
    --metrics-json "$RUN_ROOT/build_metrics.json"
    --nlist "$NLIST"
    --bits 7
    --niter "$niter"
    --max-points-per-cluster 64
    --batch-vectors "$batch_vectors"
    --fast-quantize false
  )
  write_command "$RUN_ROOT/command.sh" "${command[@]}"
  setsid /usr/bin/time -v -o "$RUN_ROOT/time.log" \
    "${command[@]}" >"$RUN_ROOT/stdout.log" 2>"$RUN_ROOT/stderr.log" &
  local build_pid=$! build_rc
  ACTIVE_BUILD_PGID="$build_pid"
  start_build_monitors "$build_pid"
  set +e
  wait "$build_pid"
  build_rc=$?
  set -e
  stop_build_monitors
  ACTIVE_BUILD_PGID=""
  parse_external_time
  summarize_build_resources
  ((build_rc == 0)) || die "GPU builder failed with exit code $build_rc"
}

validate_build_metrics() {
  local niter=20 batch_vectors=131072
  [[ "$EXPERIMENT_ROLE" != smoke ]] || { niter=5; batch_vectors=8192; }
  python3 - "$RUN_ROOT/build_metrics.json" "$INDEX_FILE" "$HNSW_FILE" \
    "$ROWS" "$DIM" "$NLIST" "$niter" "$batch_vectors" "$EXPECT_HNSW" \
    "$RUN_ROOT/build_audit.json" <<'PY_BUILD_AUDIT'
import json
import math
from pathlib import Path
import sys

metrics_path, main_path, hnsw_path = map(Path, sys.argv[1:4])
rows, dim, nlist, niter, batch_vectors = map(int, sys.argv[4:9])
expect_hnsw = sys.argv[9].lower() == "true"
output = Path(sys.argv[10])
metrics = json.loads(metrics_path.read_text())
if metrics.get("schema_version") != 1:
    raise SystemExit("unexpected build schema")
if metrics.get("dataset", {}).get("rows") != rows or metrics.get("dataset", {}).get("dim") != dim:
    raise SystemExit("build dataset metadata mismatch")
params = metrics.get("params", {})
expected = {
    "nlist": nlist,
    "bits_per_dim": 7,
    "ex_bits": 6,
    "niter": niter,
    "max_train_points_per_cluster": 64,
    "batch_vectors": batch_vectors,
    "fast_quantize": False,
}
for key, wanted in expected.items():
    if params.get(key) != wanted:
        raise SystemExit(f"build parameter mismatch: {key}: {params.get(key)!r} != {wanted!r}")
timings = metrics.get("timings", {})
timing_names = (
    "kmeans_gpu_seconds", "assignment_gpu_seconds", "quantization_gpu_seconds",
    "gpu_build_seconds", "host_gather_seconds", "d2h_seconds", "cpu_repack_seconds",
    "index_write_seconds", "centroid_hnsw_seconds", "conversion_seconds",
    "validation_seconds", "end_to_end_wall_seconds",
)
for name in timing_names:
    value = timings.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise SystemExit(f"invalid build timing: {name}={value!r}")
gpu_sum = sum(timings[name] for name in (
    "kmeans_gpu_seconds", "assignment_gpu_seconds", "quantization_gpu_seconds"
))
if not math.isclose(gpu_sum, timings["gpu_build_seconds"], rel_tol=1e-12, abs_tol=1e-9):
    raise SystemExit("gpu_build_seconds does not equal its three GPU phases")
if metrics.get("memory", {}).get("max_batch_vectors", 0) > batch_vectors:
    raise SystemExit("builder exceeded batch_vectors")
if not main_path.is_file() or main_path.stat().st_size == 0:
    raise SystemExit("main index is missing or empty")
reported = metrics.get("output", {})
if main_path.stat().st_size != reported.get("main_index_bytes"):
    raise SystemExit("main index byte count mismatch")
if expect_hnsw:
    if not hnsw_path.is_file() or hnsw_path.stat().st_size == 0:
        raise SystemExit("required HNSW sidecar is missing")
    if hnsw_path.stat().st_size != reported.get("hnsw_index_bytes"):
        raise SystemExit("HNSW byte count mismatch")
else:
    if hnsw_path.exists() or reported.get("hnsw_index_bytes") != 0:
        raise SystemExit("Flat smoke unexpectedly produced an HNSW sidecar")
for partial in (Path(str(main_path) + ".partial"), Path(str(main_path) + ".partial.hnsw")):
    if partial.exists():
        raise SystemExit(f"partial remains after successful build: {partial}")
audit = {
    "schema": "ivf_rabitq_build_audit_v1",
    "gpu_build_seconds": timings["gpu_build_seconds"],
    "main_index_bytes": main_path.stat().st_size,
    "hnsw_index_bytes": hnsw_path.stat().st_size if hnsw_path.exists() else 0,
    "expect_hnsw": expect_hnsw,
}
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY_BUILD_AUDIT
}

validate_native_load() {
  "$CPU_SEARCH" \
    --index "$INDEX_FILE" \
    --query "$QUERY" \
    --groundtruth "$GT" \
    --json "$RUN_ROOT/native_load_validation.json" \
    --topk 10 \
    --target-recall 0.97 \
    --nprobes-exact 1 \
    --warmup-queries 0 \
    --repeats 1 \
    --query-limit 1 \
    --query-threads 1 \
    --use-hacc true \
    >"$RUN_ROOT/native_load_validation.stdout.log" \
    2>"$RUN_ROOT/native_load_validation.stderr.log"
  python3 - "$RUN_ROOT/native_load_validation.json" "$ROWS" "$DIM" "$NLIST" <<'PY_NATIVE_AUDIT'
import json
import math
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text())
rows, dim, nlist = map(int, sys.argv[2:5])
metadata = value.get("index_metadata", {})
if (metadata.get("num_vectors"), metadata.get("dim"), metadata.get("nlist"), metadata.get("bits_per_dim")) != (rows, dim, nlist, 7):
    raise SystemExit("native index metadata mismatch")
if value.get("execution") != {"query_parallelism": 1, "omp_threads": 1, "rerank": False, "num_queries": 1}:
    raise SystemExit("native validation execution metadata mismatch")
results = value.get("results", [])
if len(results) != 1 or results[0].get("nprobe") != 1:
    raise SystemExit("native validation did not execute nprobe=1")
if not math.isfinite(results[0].get("recall_at_10", float("nan"))):
    raise SystemExit("native validation recall is non-finite")
PY_NATIVE_AUDIT
}

validate_build() {
  require_file "$RUN_ROOT/build_metrics.json"
  require_file "$RUN_ROOT/time.log"
  require_file "$RUN_ROOT/external_time.json"
  require_file "$RUN_ROOT/resource_summary.json"
  validate_build_metrics
  validate_native_load
}

audit_search_json() {
  local json_path="$1" repeats="$2" nprobes="$3"
  python3 - "$json_path" "$repeats" "$nprobes" "$QUERY_ROWS" "$ROWS" "$DIM" "$NLIST" <<'PY_SEARCH_AUDIT'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_repeats = int(sys.argv[2])
expected_nprobes = [int(value) for value in sys.argv[3].split(",")]
query_rows, rows, dim, nlist = map(int, sys.argv[4:8])
value = json.loads(path.read_text())
if value.get("schema_version") != 1:
    raise SystemExit("unexpected search schema")
params = value.get("params", {})
expected_params = {
    "topk": 10,
    "target_recall": 0.97,
    "warmup_queries": 100,
    "repeats": expected_repeats,
    "use_hacc": True,
    "query_limit": 0,
    "nprobe_mode": "explicit_exact",
    "requested_nprobes": expected_nprobes,
}
for key, expected in expected_params.items():
    if params.get(key) != expected:
        raise SystemExit(f"search parameter mismatch: {key}: {params.get(key)!r} != {expected!r}")
execution = value.get("execution", {})
if execution != {
    "query_parallelism": 8,
    "omp_threads": 8,
    "rerank": False,
    "num_queries": query_rows,
}:
    raise SystemExit(f"search execution mismatch: {execution!r}")
metadata = value.get("index_metadata", {})
if (metadata.get("num_vectors"), metadata.get("dim"), metadata.get("nlist"), metadata.get("bits_per_dim")) != (rows, dim, nlist, 7):
    raise SystemExit("search index metadata mismatch")
results = value.get("results", [])
if [point.get("nprobe") for point in results] != expected_nprobes:
    raise SystemExit("exact-list result sequence differs from request")
for point in results:
    recall = point.get("recall_at_10")
    seconds, qps = point.get("repeat_seconds", []), point.get("repeat_qps", [])
    if not isinstance(recall, (int, float)) or not math.isfinite(recall) or not 0 <= recall <= 1:
        raise SystemExit("invalid Recall@10")
    if len(seconds) != expected_repeats or len(qps) != expected_repeats:
        raise SystemExit("repeat count mismatch")
    if any(not math.isfinite(item) or item <= 0 for item in [*seconds, *qps]):
        raise SystemExit("invalid search timing/QPS")
PY_SEARCH_AUDIT
}

run_search_json() {
  local output="$1" repeats="$2" nprobes="$3" stem
  [[ ! -e "$output" && ! -e "$output.partial" ]] \
    || die "refusing to overwrite search JSON: $output"
  mkdir -p "$(dirname "$output")"
  stem="${output%.json}"
  "$CPU_SEARCH" \
    --index "$INDEX_FILE" \
    --query "$QUERY" \
    --groundtruth "$GT" \
    --json "$output" \
    --topk 10 \
    --target-recall 0.97 \
    --nprobes-exact "$nprobes" \
    --warmup-queries 100 \
    --repeats "$repeats" \
    --query-threads 8 \
    --use-hacc true \
    >"${stem}.stdout.log" 2>"${stem}.stderr.log"
  audit_search_json "$output" "$repeats" "$nprobes"
}

recall_for_nprobe() {
  local nprobe="$1"
  python3 - "$RUN_ROOT/discovery" "$nprobe" <<'PY_RECALL_LOOKUP'
import json
from pathlib import Path
import sys

root, wanted = Path(sys.argv[1]), int(sys.argv[2])
values = []
for path in sorted(root.glob("*.json")):
    value = json.loads(path.read_text())
    for point in value.get("results", []):
        if int(point["nprobe"]) == wanted:
            values.append(float(point["recall_at_10"]))
if len(values) != 1:
    raise SystemExit(f"expected one Recall for nprobe={wanted}, found {len(values)}")
print(values[0])
PY_RECALL_LOOKUP
}

recall_gt() {
  local nprobe="$1" threshold="$2" recall
  recall="$(recall_for_nprobe "$nprobe")"
  python3 - "$recall" "$threshold" <<'PY_COMPARE_GT'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY_COMPARE_GT
}

recall_lt() {
  local nprobe="$1" threshold="$2" recall
  recall="$(recall_for_nprobe "$nprobe")"
  python3 - "$recall" "$threshold" <<'PY_COMPARE_LT'
import sys
raise SystemExit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY_COMPARE_LT
}

fail_quality() {
  local quality="$1"
  CURRENT_STAGE=discovery_quality
  FAILURE_MESSAGE="$quality"
  write_status "$RUN_ROOT/status.json" failed 1 discovery_quality "$quality"
  die "$quality"
}

merge_discovery() {
  python3 - "$RUN_ROOT/discovery" "$RUN_ROOT/selected_nprobes.tsv" \
    "$RUN_ROOT/discovery/discovery_merged.tsv" "$LOW_RECALL" "$HIGH_RECALL" <<'PY_DISCOVERY_MERGE'
import csv
import json
import math
from pathlib import Path
import sys

root, selected_path, merged_path = map(Path, sys.argv[1:4])
low, high = map(float, sys.argv[4:6])
points = {}
for path in sorted(root.glob("*.json")):
    value = json.loads(path.read_text())
    params, execution = value.get("params", {}), value.get("execution", {})
    if params.get("nprobe_mode") != "explicit_exact":
        raise SystemExit(f"non-exact discovery JSON: {path}")
    if execution.get("query_parallelism") != 8 or execution.get("rerank") is not False:
        raise SystemExit(f"invalid discovery execution: {path}")
    requested = params.get("requested_nprobes")
    results = value.get("results", [])
    if [row.get("nprobe") for row in results] != requested:
        raise SystemExit(f"discovery list mismatch: {path}")
    for row in results:
        nprobe, recall = int(row["nprobe"]), float(row["recall_at_10"])
        if not math.isfinite(recall) or not 0 <= recall <= 1:
            raise SystemExit(f"invalid discovery Recall: {path}")
        if nprobe in points and not math.isclose(points[nprobe][0], recall, abs_tol=1e-12):
            raise SystemExit(f"conflicting duplicate nprobe={nprobe}")
        qps = row.get("repeat_qps", [])
        if len(qps) != 1 or not math.isfinite(float(qps[0])) or float(qps[0]) <= 0:
            raise SystemExit(f"invalid discovery QPS: {path}")
        points[nprobe] = (recall, path.name, float(qps[0]))

core = {64, 128, 256, 512}
if not core.issubset(points):
    raise SystemExit(f"missing core nprobes: {sorted(core - points.keys())}")
rows = [(nprobe, *points[nprobe]) for nprobe in sorted(points)]
if rows[0][1] > low:
    raise SystemExit("quality_lower_bound_not_reached")
if rows[-1][1] < high:
    raise SystemExit("quality_upper_bound_not_reached")

def write(path, fieldnames, records):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)

write(merged_path, ("nprobe", "recall_at_10", "source", "discovery_qps"), [
    {"nprobe": nprobe, "recall_at_10": recall, "source": source, "discovery_qps": qps}
    for nprobe, recall, source, qps in rows
])
write(selected_path, ("nprobe", "discovery_recall_at_10", "source"), [
    {"nprobe": nprobe, "discovery_recall_at_10": recall, "source": source}
    for nprobe, recall, source, _ in rows
])
PY_DISCOVERY_MERGE
}

run_discovery() {
  mkdir -p "$RUN_ROOT/discovery"
  run_search_json "$RUN_ROOT/discovery/core.json" 1 "$CORE_NPROBES"

  local current=64 next output
  while recall_gt "$current" "$LOW_RECALL"; do
    ((current != 1)) || fail_quality quality_lower_bound_not_reached
    next=$((current * 2 / 3))
    ((next >= 1)) || next=1
    ((next < current)) || next=$((current - 1))
    current="$next"
    output="$RUN_ROOT/discovery/lower_$(printf '%04d' "$current").json"
    run_search_json "$output" 1 "$current"
  done

  current=512
  while recall_lt "$current" "$HIGH_RECALL"; do
    ((current != MAX_NPROBE)) || fail_quality quality_upper_bound_not_reached
    next=$(((current * 3 + 1) / 2))
    if ((next >= MAX_NPROBE)); then
      current=4096
    else
      current="$next"
    fi
    output="$RUN_ROOT/discovery/upper_$(printf '%04d' "$current").json"
    run_search_json "$output" 1 "$current"
  done
  merge_discovery
}

selected_nprobe_csv() {
  python3 - "$RUN_ROOT/selected_nprobes.tsv" <<'PY_SELECTED_CSV'
import csv
from pathlib import Path
import sys
with Path(sys.argv[1]).open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if not rows:
    raise SystemExit("selected nprobe list is empty")
print(",".join(row["nprobe"] for row in rows))
PY_SELECTED_CSV
}

run_formal_search() {
  local selected
  selected="$(selected_nprobe_csv)"
  run_search_json "$RUN_ROOT/formal_search.json" 3 "$selected"
}

summarize_search() {
  local expected_repeats=3 require_coverage=true
  if [[ "$EXPERIMENT_ROLE" == smoke ]]; then
    expected_repeats=1
    require_coverage=false
  fi
  python3 - "$RUN_ROOT/formal_search.json" "$RUN_ROOT/selected_nprobes.tsv" "$DATASET" \
    "$RUN_ROOT/search_sweep.tsv" "$RUN_ROOT/search_summary.tsv" \
    "$RUN_ROOT/search_anomalies.tsv" "$expected_repeats" "$require_coverage" <<'PY_SEARCH_SUMMARY'
import csv
import json
import math
from pathlib import Path
import statistics
import sys

formal_path, selected_path = map(Path, sys.argv[1:3])
dataset = sys.argv[3]
sweep_path, summary_path, anomaly_path = map(Path, sys.argv[4:7])
expected_repeats = int(sys.argv[7])
require_coverage = sys.argv[8].lower() == "true"
with selected_path.open(newline="") as stream:
    selected_rows = list(csv.DictReader(stream, delimiter="\t"))
selected = [int(row["nprobe"]) for row in selected_rows]
value = json.loads(formal_path.read_text())
params, execution = value.get("params", {}), value.get("execution", {})
if params.get("nprobe_mode") != "explicit_exact" or params.get("requested_nprobes") != selected:
    raise SystemExit("formal exact-list provenance mismatch")
if params.get("repeats") != expected_repeats:
    raise SystemExit("formal repeat parameter mismatch")
if execution.get("query_parallelism") != 8 or execution.get("omp_threads") != 8 or execution.get("rerank") is not False:
    raise SystemExit("formal execution metadata mismatch")
points = value.get("results", [])
if [int(point["nprobe"]) for point in points] != selected:
    raise SystemExit("formal result list differs from selected_nprobes.tsv")
if require_coverage and (float(points[0]["recall_at_10"]) > 0.85 or float(points[-1]["recall_at_10"]) < 0.97):
    raise SystemExit("formal curve does not cover Recall@10 0.85--0.97")

sweep_rows, summary_rows, anomaly_rows = [], [], []
previous = None
for point in points:
    nprobe, recall = int(point["nprobe"]), float(point["recall_at_10"])
    seconds = [float(item) for item in point.get("repeat_seconds", [])]
    qps = [float(item) for item in point.get("repeat_qps", [])]
    if len(seconds) != expected_repeats or len(qps) != expected_repeats:
        raise SystemExit(f"repeat count mismatch at nprobe={nprobe}")
    if not math.isfinite(recall) or not 0 <= recall <= 1 or any(
        not math.isfinite(item) or item <= 0 for item in [*seconds, *qps]
    ):
        raise SystemExit(f"invalid formal measurement at nprobe={nprobe}")
    for repeat, (duration, throughput) in enumerate(zip(seconds, qps), start=1):
        sweep_rows.append({
            "dataset": dataset, "nprobe": nprobe, "repeat": repeat,
            "recall_at_10": recall, "seconds": duration, "qps": throughput,
            "query_workers": 8, "rerank": "false",
        })
    summary_rows.append({
        "dataset": dataset,
        "nprobe": nprobe,
        "recall_at_10": recall,
        "qps_mean": statistics.fmean(qps),
        "qps_sample_std": statistics.stdev(qps) if len(qps) > 1 else 0.0,
        "qps_median": statistics.median(qps),
        "qps_min": min(qps),
        "qps_max": max(qps),
        "repeat_count": len(qps),
        "query_workers": 8,
        "rerank": "false",
    })
    if previous is not None and recall < previous[1]:
        anomaly_rows.append({
            "previous_nprobe": previous[0], "previous_recall_at_10": previous[1],
            "nprobe": nprobe, "recall_at_10": recall, "kind": "recall_decrease",
        })
    previous = (nprobe, recall)

def write(path, fieldnames, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

write(sweep_path, (
    "dataset", "nprobe", "repeat", "recall_at_10", "seconds", "qps", "query_workers", "rerank"
), sweep_rows)
write(summary_path, (
    "dataset", "nprobe", "recall_at_10", "qps_mean", "qps_sample_std", "qps_median",
    "qps_min", "qps_max", "repeat_count", "query_workers", "rerank"
), summary_rows)
write(anomaly_path, (
    "previous_nprobe", "previous_recall_at_10", "nprobe", "recall_at_10", "kind"
), anomaly_rows)
PY_SEARCH_SUMMARY
}

cleanup_successful_dataset() {
  local index_root resolved bytes digest hash_line
  index_root="$(realpath -m "$INDEX_DIR")"
  resolved="$(realpath -m "$INDEX_FILE")"
  [[ "$(dirname "$resolved")" == "$index_root" ]] || die "main index escapes run root: $resolved"
  require_file "$INDEX_FILE"
  if [[ "$EXPECT_HNSW" == true ]]; then
    resolved="$(realpath -m "$HNSW_FILE")"
    [[ "$(dirname "$resolved")" == "$index_root" ]] || die "HNSW index escapes run root: $resolved"
    require_file "$HNSW_FILE"
  else
    [[ ! -e "$HNSW_FILE" ]] || die "unexpected HNSW sidecar before cleanup"
  fi

  printf 'kind\tpath\tbytes\tsha256\n' >"$RUN_ROOT/index_artifact_manifest.tsv"
  : >"$RUN_ROOT/index_sha256.txt"
  bytes="$(stat -Lc %s -- "$INDEX_FILE")"
  hash_line="$(sha256sum "$INDEX_FILE")"
  printf '%s\n' "$hash_line" >>"$RUN_ROOT/index_sha256.txt"
  digest="${hash_line%% *}"
  printf 'ivf_rabitq_index\t%s\t%s\t%s\n' "$INDEX_FILE" "$bytes" "$digest" \
    >>"$RUN_ROOT/index_artifact_manifest.tsv"
  if [[ -f "$HNSW_FILE" ]]; then
    bytes="$(stat -Lc %s -- "$HNSW_FILE")"
    hash_line="$(sha256sum "$HNSW_FILE")"
    printf '%s\n' "$hash_line" >>"$RUN_ROOT/index_sha256.txt"
    digest="${hash_line%% *}"
    printf 'ivf_rabitq_index\t%s\t%s\t%s\n' "$HNSW_FILE" "$bytes" "$digest" \
      >>"$RUN_ROOT/index_artifact_manifest.tsv"
  fi
  cp "$RUN_ROOT/index_artifact_manifest.tsv" "$RUN_ROOT/cleanup_manifest.tsv"
  {
    printf 'dataset=%s\n' "$DATASET"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'verified=false\n'
  } >"$RUN_ROOT/cleanup_summary.txt"

  rm -f -- "$INDEX_FILE"
  if [[ -e "$HNSW_FILE" ]]; then
    rm -f -- "$HNSW_FILE"
  fi
  for path in "$INDEX_FILE.partial" "$INDEX_FILE.partial.hnsw"; do
    resolved="$(realpath -m "$path")"
    [[ "$(dirname "$resolved")" == "$index_root" ]] || die "partial escapes run root: $resolved"
    [[ ! -e "$path" ]] || rm -f -- "$path"
  done
}

verify_dataset_cleanup() {
  local required=(
    status.json command.sh dataset_manifest.json environment.txt binary_hashes.txt
    time.log stdout.log stderr.log build_metrics.json build_audit.json external_time.json
    resource_summary.json process_tree_rss.tsv meminfo.tsv temp_usage.tsv
    native_load_validation.json formal_search.json selected_nprobes.tsv
    search_sweep.tsv search_summary.tsv search_anomalies.tsv
    index_artifact_manifest.tsv index_sha256.txt cleanup_manifest.tsv cleanup_summary.txt
  )
  local evidence
  for evidence in "${required[@]}"; do
    require_file "$RUN_ROOT/$evidence"
  done
  if [[ "$EXPERIMENT_ROLE" == formal ]]; then
    require_file "$RUN_ROOT/cache_protocol.tsv"
    require_file "$RUN_ROOT/build_prewarm_manifest.tsv"
    require_file "$RUN_ROOT/discovery_prewarm_manifest.tsv"
    require_file "$RUN_ROOT/formal_search_prewarm_manifest.tsv"
    require_file "$RUN_ROOT/discovery/discovery_merged.tsv"
  fi
  [[ ! -e "$INDEX_FILE" ]] || die "main index remains after cleanup"
  [[ ! -e "$HNSW_FILE" ]] || die "HNSW sidecar remains after cleanup"
  [[ ! -e "$INDEX_FILE.partial" ]] || die "main partial remains after cleanup"
  [[ ! -e "$INDEX_FILE.partial.hnsw" ]] || die "HNSW partial remains after cleanup"
  python3 - "$RUN_ROOT/cleanup_summary.txt" <<'PY_CLEANUP_VERIFY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
lines = path.read_text().splitlines()
if "verified=false" not in lines:
    raise SystemExit("cleanup summary is not awaiting verification")
path.write_text("\n".join("verified=true" if line == "verified=false" else line for line in lines) + "\n")
PY_CLEANUP_VERIFY
}

finalize_successful_dataset() {
  CURRENT_STAGE=cleaning
  write_status "$RUN_ROOT/status.json" cleaning 0 "" "recording and removing owned index artifacts"
  cleanup_successful_dataset
  verify_dataset_cleanup
  write_status "$RUN_ROOT/status.json" accepted 0 "" "cleanup verified"
  update_schedule "$DATASET" accepted "cleanup verified"
}

run_dataset_configured() {
  FAILURE_MESSAGE=""
  CURRENT_STAGE=preflight
  fresh_dir "$RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  write_status "$RUN_ROOT/status.json" created 0 "" "run root created"
  update_schedule "$DATASET" running ""
  validate_dataset
  record_environment

  CURRENT_STAGE=build_prewarm
  prepare_build_cache
  CURRENT_STAGE=building
  write_status "$RUN_ROOT/status.json" building 0 "" "exact 7-bit GPU build"
  run_gpu_build
  CURRENT_STAGE=build_validation
  validate_build

  CURRENT_STAGE=discovery_prewarm
  prepare_search_cache discovery
  CURRENT_STAGE=discovering
  write_status "$RUN_ROOT/status.json" discovering 0 "" "adaptive 1.5x nprobe discovery"
  run_discovery

  CURRENT_STAGE=formal_search_prewarm
  prepare_search_cache formal_search
  CURRENT_STAGE=formal_search
  write_status "$RUN_ROOT/status.json" formal_search 0 "" "three-repeat exact-list search"
  run_formal_search
  CURRENT_STAGE=summarizing
  write_status "$RUN_ROOT/status.json" summarizing 0 "" "search and batch summaries"
  summarize_search
  generate_batch_summary
  finalize_successful_dataset
  RUN_ROOT=""
  DATASET=""
  CURRENT_STAGE=""
  generate_batch_summary
}

run_dataset() {
  load_dataset "$1"
  verify_formal_prerequisites
  run_dataset_configured
}

run_prepare() {
  require_commands bash python3 git cmake ctest grep sha256sum ldd nvidia-smi lscpu
  require_executable "$CUVS_CMAKE"
  mkdir -p "$EVAL_ROOT/prepare"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  [[ -n "$BATCH_ROOT" ]] || BATCH_ROOT="$EVAL_ROOT/prepare/$timestamp"
  fresh_dir "$BATCH_ROOT"
  mkdir -p "$BATCH_ROOT/provenance"
  BATCH_KIND=prepare
  {
    cmake -S "$RABITQ_ROOT" -B "$RABITQ_ROOT/build" \
      -DRABITQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$RABITQ_ROOT/build" \
      --target rabitq_tests ivf_rabitq_fbin_benchmark -j8
    ctest --test-dir "$RABITQ_ROOT/build" --output-on-failure \
      -R 'NprobeSchedule|RecallAtK|BenchmarkStats|IvfBatchSearch'

    "$CUVS_CMAKE" -S "$CUVS_ROOT/cpp" -B "$CUVS_ROOT/cpp/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_IVF_RABITQ_CPU_EXPORTER=ON \
      -DRABITQ_LIBRARY_ROOT="$RABITQ_ROOT" \
      -DCMAKE_CUDA_ARCHITECTURES=89-real
    "$CUVS_CMAKE" --build "$CUVS_ROOT/cpp/build" \
      --target ivf_rabitq_cpu_index_build \
               NEIGHBORS_ANN_IVF_RABITQ_CPU_LAYOUT_TEST \
               NEIGHBORS_ANN_IVF_RABITQ_STREAM_EXPORT_TEST -j8
    "$CUVS_ROOT/cpp/build/gtests/NEIGHBORS_ANN_IVF_RABITQ_CPU_LAYOUT_TEST"
    "$CUVS_ROOT/cpp/build/gtests/NEIGHBORS_ANN_IVF_RABITQ_STREAM_EXPORT_TEST"
    python3 "$RUNBOOK_TEST"
  } 2>&1 | tee "$BATCH_ROOT/prepare.log"
  verify_tool_prerequisites
  record_repository_provenance
  python3 - "$CUVS_ROOT/cpp/build/CMakeCache.txt" <<'PY_PREPARE_AUDIT'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
required = {
    "CMAKE_BUILD_TYPE": "Release",
    "BUILD_IVF_RABITQ_CPU_EXPORTER": "ON",
    "CMAKE_CUDA_ARCHITECTURES": "89-real",
}
cache = {}
for line in text.splitlines():
    if "=" not in line or ":" not in line:
        continue
    key_with_type, value = line.split("=", 1)
    key = key_with_type.split(":", 1)[0]
    cache[key] = value
mismatched = {
    key: {"expected": expected, "actual": cache.get(key)}
    for key, expected in required.items()
    if cache.get(key) != expected
}
if mismatched:
    raise SystemExit(f"cuVS CMake contract mismatch: {mismatched}")
PY_PREPARE_AUDIT
  write_status "$BATCH_ROOT/prepare_status.json" accepted 0 "" "prepare passed" false
  log "prepare root: $BATCH_ROOT"
}

generate_smoke_data() {
  local output="$1"
  mkdir -p "$output"
  python3 - "$output" <<'PY_SMOKE_DATA'
import json
from pathlib import Path
import struct
import sys

import numpy as np

root = Path(sys.argv[1])
rows, dim, query_rows, width = 65_536, 96, 256, 100
rng = np.random.default_rng(42)
base = rng.standard_normal((rows, dim), dtype=np.float32)
query_ids = (np.arange(query_rows, dtype=np.uint64) * 257 % rows).astype(np.int64)
queries = (base[query_ids] + rng.normal(0.0, 1e-3, (query_rows, dim))).astype(np.float32)
base_norm = np.einsum("ij,ij->i", base, base)
groundtruth = np.empty((query_rows, width), dtype=np.uint32)
for start in range(0, query_rows, 32):
    block = queries[start:start + 32]
    query_norm = np.einsum("ij,ij->i", block, block)[:, None]
    distances = query_norm + base_norm[None, :] - 2.0 * (block @ base.T)
    np.maximum(distances, 0.0, out=distances)
    candidates = np.argpartition(distances, kth=width - 1, axis=1)[:, :width]
    for offset, identifiers in enumerate(candidates):
        order = np.lexsort((identifiers, distances[offset, identifiers]))
        groundtruth[start + offset] = identifiers[order].astype(np.uint32)

def write_matrix(path, matrix):
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", matrix.shape[0], matrix.shape[1]))
        matrix.tofile(stream)

write_matrix(root / "base.fbin", base)
write_matrix(root / "query.fbin", queries)
write_matrix(root / "gt100.ibin", groundtruth)
(root / "smoke_contract.json").write_text(json.dumps({
    "schema": "ivf_rabitq_smoke_contract_v1", "seed": 42,
    "rows": rows, "dim": dim, "query_rows": query_rows, "gt_width": width,
    "ground_truth": "exact squared L2, bounded NumPy blocks",
}, indent=2, sort_keys=True) + "\n")
PY_SMOKE_DATA
}

write_smoke_selected() {
  python3 - "$RUN_ROOT/formal_search.json" "$RUN_ROOT/selected_nprobes.tsv" <<'PY_SMOKE_SELECTED'
import csv
import json
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
value = json.loads(source.read_text())
temporary = output.with_suffix(output.suffix + ".partial")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=("nprobe", "discovery_recall_at_10", "source"),
        delimiter="\t", lineterminator="\n",
    )
    writer.writeheader()
    for point in value["results"]:
        writer.writerow({
            "nprobe": point["nprobe"],
            "discovery_recall_at_10": point["recall_at_10"],
            "source": "smoke",
        })
temporary.replace(output)
PY_SMOKE_SELECTED
}

run_smoke_configured() {
  FAILURE_MESSAGE=""
  CURRENT_STAGE=smoke_data
  generate_smoke_data "$SMOKE_DATA_DIR"
  fresh_dir "$RUN_ROOT"
  write_status "$RUN_ROOT/status.json" created 0 "" "smoke run root created" false
  update_schedule smoke running ""
  validate_dataset
  record_environment
  printf 'sample_epoch\tphase\tlabel\tmetric\tvalue_kib\n' >"$RUN_ROOT/cache_protocol.tsv"
  printf '%s\tsmoke\tnone\tpaper_eligible\t0\n' "$(date +%s.%N)" >>"$RUN_ROOT/cache_protocol.tsv"

  CURRENT_STAGE=building
  write_status "$RUN_ROOT/status.json" building 0 "" "smoke exact 7-bit GPU build" false
  run_gpu_build
  CURRENT_STAGE=build_validation
  validate_build
  CURRENT_STAGE=formal_search
  run_search_json "$RUN_ROOT/formal_search.json" 1 "8,16,32,64"
  write_smoke_selected
  CURRENT_STAGE=summarizing
  summarize_search
  generate_batch_summary
  finalize_successful_dataset
  RUN_ROOT=""
  DATASET=""
  CURRENT_STAGE=""
  generate_batch_summary
}

configure_smoke() {
  DATASET=smoke
  ROWS=65536 DIM=96 NLIST=128 QUERY_ROWS=256 GT_ROWS=256
  BASE="$SMOKE_DATA_DIR/base.fbin"
  QUERY="$SMOKE_DATA_DIR/query.fbin"
  GT="$SMOKE_DATA_DIR/gt100.ibin"
  RUN_ROOT="$BATCH_ROOT/hot_path/smoke/ivf_rabitq_exact/run_1"
  INDEX_DIR="$RUN_ROOT/index"
  INDEX_FILE="$INDEX_DIR/ivf_rabitq.cpu.index"
  HNSW_FILE="$INDEX_FILE.hnsw"
  PAPER_ELIGIBLE=false
  EXPERIMENT_ROLE=smoke
  EXPECT_HNSW=false
}

run_smoke_entry() {
  verify_smoke_prerequisites
  acquire_lock
  mkdir -p "$EVAL_ROOT/smoke"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  [[ -n "$BATCH_ROOT" ]] || BATCH_ROOT="$EVAL_ROOT/smoke/$timestamp"
  initialize_batch_root smoke
  initialize_schedule smoke
  SMOKE_DATA_DIR="$BATCH_ROOT/data"
  configure_smoke
  run_smoke_configured
  log "smoke root: $BATCH_ROOT"
}

run_formal_entry() {
  verify_tool_prerequisites
  acquire_lock
  mkdir -p "$EVAL_ROOT/runs"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  [[ -n "$BATCH_ROOT" ]] || BATCH_ROOT="$EVAL_ROOT/runs/$timestamp"
  initialize_batch_root formal
  initialize_schedule "${FORMAL_DATASETS[@]}"
  run_dataset "wiki35m"
  run_dataset "laion100m"
  run_dataset "msmarco70m"
  run_dataset "deep100m"
  generate_batch_summary
  log "formal root: $BATCH_ROOT"
}

run_single_entry() {
  local requested="$1" timestamp
  verify_tool_prerequisites
  acquire_lock
  mkdir -p "$EVAL_ROOT/runs"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  [[ -n "$BATCH_ROOT" ]] || BATCH_ROOT="$EVAL_ROOT/runs/$timestamp"
  initialize_batch_root "$requested"
  initialize_schedule "$requested"
  run_dataset "$requested"
  generate_batch_summary
  log "single-dataset root: $BATCH_ROOT"
}

case "$MODE" in
  help) usage ;;
  prepare) run_prepare ;;
  smoke) run_smoke_entry ;;
  formal) run_formal_entry ;;
  wiki35m|laion100m|msmarco70m|deep100m) run_single_entry "$MODE" ;;
  *) usage >&2; exit 2 ;;
esac
```
<!-- END RUNBOOK_SCRIPT -->

## 结果验收

- `prepare` 的定向 RaBitQ/cuVS tests 全部通过，并记录两个二进制与三个仓库的 provenance。
- formal 的 `build_metrics.json` 必须是 exact 7-bit，且主索引与 `.hnsw` 均通过 native load。
- discovery 必须保留核心四点并可重放 1.5 倍扩展；formal JSON 的结果列表必须与
  `selected_nprobes.tsv` 完全一致。
- 每个 formal nprobe 恰好三次 QPS；表格报告 arithmetic mean、sample std、median 和
  min--max，Recall 不平滑、不插值。
- accepted 状态只能出现在 cleanup `verified=true` 之后。失败状态保留索引、partial、日志和
  monitor trace，不自动进入下一数据集。
