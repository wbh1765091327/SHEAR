# SHEAR Four-Dataset Warm-Cache Runbook

This runbook measures SHEAR only. It covers Wiki35M, LAION100M,
MSMARCO70M, and DEEP100M in the fixed order shown below. Build and search each
receive an independent drop-then-prewarm phase. Prewarm duration is recorded
but excluded from the measured build or search command.

Formal order:

```text
Wiki35M -> LAION100M -> MSMARCO70M -> DEEP100M
```

External `/usr/bin/time -v` wall time is the primary build time. Driver wall,
profile stage total, persistent-kernel event time, and the post-last-GPU
CPU-tail metric are diagnostic timing layers and must not be substituted for
external wall time.

## Material passport

| Dataset | Rows | Dim | Base | Query | GT | Expected SQ8 leaf mode |
|---|---:|---:|---|---|---|---|
| Wiki35M | 35,000,000 | 1024 | `/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin` | `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin` | `/data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin` | staged |
| LAION100M | 100,000,000 | 512 | `/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin` | `/data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin` | `/data_nvme_kioxia/tyj/data/laion-512-100M/gt100.bin` | staged |
| MSMARCO70M | 70,000,000 | 768 | `/data_nvme_kioxia/tyj/data/msmarco-768-70M/base.fbin` | `/data_nvme_kioxia/tyj/data/msmarco-768-114M/query.fbin` | `/data_nvme_kioxia/tyj/data/msmarco-768-70M/gt100.bin` | staged |
| DEEP100M | 100,000,000 | 96 | `/data_nvme_kioxia/tyj/data/deep-96-100M/base.100M.fbin` | `/data_nvme_kioxia/tyj/data/deep-96-100M/query.public.10K.fbin` | `/data_nvme_kioxia/tyj/data/deep-96-100M/groundtruth.public.10K.ibin` | row-owned |

All formal builds use squared L2, `R=64`, `Lbuild=100`, eight build threads,
single-threaded BLAS, leaf sizes 256--1280, `leaf-knn=2`, sampling fraction
0.05, fanout 8/3/1, seed 42, buffered OOM I/O with automatic host budget,
16 GiB GPU workspace, SQ8 GPU-D0 staging, fused persistent execution, and an
8%/8%/8% D0 capacity budget. The formal command explicitly requests D0
`wmma-batch`, D1+ SQ8 assignment `wmma`, and persistent ticket batch `8`.
SQ8 leaf mode and scheduler remain `auto` so the profile can select row-owned
leaf execution for DEEP96, staged leaf execution for the high-dimensional
datasets, and the baseline scheduler.

The profile audit requires requested and actual D0 SQ8 WMMA batch-8, requested
and actual D1+ INT8 WMMA assignment and ticket batch 8, baseline scheduler,
row-owned leaf for DEEP96, and staged leaf for 512D/768D/1024D. It also
requires zero overflow/fallback indicators and both copies of the new
post-last-GPU CPU-tail metric to be observed and equal. The tail remains a
diagnostic value outside `serial_timing.rows`.

Search independently drops and prewarms base, final graph, query, and ground
truth, in that order. It uses mmap and eight threads at
`L=25,50,100,200,400`, with one warmup and three measured repetitions per L.

The default output root is:

```text
/data_staging/wbh/shear_four_dataset_warm_cache/runs/<timestamp>
```

Each formal result is written under:

```text
<batch-root>/hot_path/<dataset>/shear/run_1
```

After profile, graph, and search validation succeeds, the run records the
original index size, deletes the index, OOM temporary directory, and SQ8
sidecar directory, verifies that evidence remains, and only then marks the
dataset accepted. A failed or interrupted run retains its scene and stops the
schedule.

## Executable program

Extract the fenced Bash program and run one of `help`, `prepare`, `smoke`,
`formal`, `wiki35m`, `laion100m`, `msmarco70m`, or `deep100m`.

<!-- BEGIN RUNBOOK_SCRIPT -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-help}"
SHEAR_ROOT="${SHEAR_ROOT:-/path/to/SHEAR}"
DATA_ROOT="${DATA_ROOT:-/data_nvme_kioxia/tyj/data}"
OUTPUT_BASE="${OUTPUT_BASE:-/data_staging/wbh/shear_four_dataset_warm_cache/runs}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BATCH_ROOT="${BATCH_ROOT:-$OUTPUT_BASE/$RUN_STAMP}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUNBOOK_PATH="$SHEAR_ROOT/docs/runbooks/p1_p2_shear_four_dataset.md"
LOCK_PATH="${LOCK_PATH:-/data_staging/wbh/.shear_four_dataset_warm_cache.lock}"
FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)
SEARCH_LISTS=(25 50 100 200 400)
ACTIVE_MONITOR_PIDS=()
ACTIVE_BUILD_PGID=""
BATCH_INITIALIZED=0
BATCH_KIND=""
CURRENT_RUN_ROOT=""
CURRENT_DATASET=""
CURRENT_STAGE=""
EXPERIMENT_LOCK_FD=""

export PATH="$HOME/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LC_ALL=C

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
  mkdir -p "$(dirname "$path")"
  python3 - "$path" "$phase" "$exit_code" "$failed_stage" "$message" <<'PY_WRITE_STATUS'
import datetime
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
phase = sys.argv[2]
exit_code = int(sys.argv[3])
failed_stage = sys.argv[4] or None
message = sys.argv[5] or None
if path.exists():
    value = json.loads(path.read_text())
else:
    value = {"schema": "shear_four_dataset_status_v1", "history": []}
event = {
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": phase,
    "exit_code": exit_code,
    "failed_stage": failed_stage,
    "message": message,
}
value.update(event)
value["history"].append(event)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY_WRITE_STATUS
}

load_dataset() {
  DATASET="$1"
  QUERY_ROWS=10000
  GT_ROWS=10000
  GT_WIDTH=100
  case "$DATASET" in
    wiki35m)
      ROWS=35000000 DIM=1024 LEAF_MODE=staged SPLIT_RECURSION_LIMIT=8
      BASE="$DATA_ROOT/wiki-1024-35M/en_base.fbin"
      QUERY="$DATA_ROOT/wiki-1024-35M/query.fbin"
      GT="$DATA_ROOT/wiki-1024-35M/gt100.bin"
      ;;
    laion100m)
      ROWS=100000000 DIM=512 LEAF_MODE=staged SPLIT_RECURSION_LIMIT=8
      BASE="$DATA_ROOT/laion-512-100M/base.fbin"
      QUERY="$DATA_ROOT/laion-512-100M/query.fbin"
      GT="$DATA_ROOT/laion-512-100M/gt100.bin"
      ;;
    msmarco70m)
      ROWS=70000000 DIM=768 LEAF_MODE=staged SPLIT_RECURSION_LIMIT=32
      BASE="$DATA_ROOT/msmarco-768-70M/base.fbin"
      QUERY="$DATA_ROOT/msmarco-768-114M/query.fbin"
      GT="$DATA_ROOT/msmarco-768-70M/gt100.bin"
      ;;
    deep100m)
      ROWS=100000000 DIM=96 LEAF_MODE=row-owned SPLIT_RECURSION_LIMIT=8
      BASE="$DATA_ROOT/deep-96-100M/base.100M.fbin"
      QUERY="$DATA_ROOT/deep-96-100M/query.public.10K.fbin"
      GT="$DATA_ROOT/deep-96-100M/groundtruth.public.10K.ibin"
      ;;
    *) die "unknown dataset: $DATASET" ;;
  esac
  RUN_ROOT="$BATCH_ROOT/hot_path/$DATASET/shear/run_1"
  INDEX_DIR="$RUN_ROOT/index"
  INDEX_FILE="$INDEX_DIR/_mem.index"
  OOM_TEMP_DIR="$RUN_ROOT/oom_temp"
  SQ_SIDECAR_DIR="$RUN_ROOT/sq_sidecar"
  SQ_SIDECAR_PREFIX="$SQ_SIDECAR_DIR/sq_wmma"
  EXPERIMENT_ROLE=formal
}

record_repository_provenance() {
  local root="$BATCH_ROOT/provenance"
  mkdir -p "$root"
  git -C "$SHEAR_ROOT" rev-parse HEAD >"$root/git_head.txt"
  git -C "$SHEAR_ROOT" status --short >"$root/git_status.txt"
  git -C "$SHEAR_ROOT" diff --binary >"$root/dirty.diff"
  {
    sha256sum \
      "$SHEAR_ROOT/target/release/build_shear_index" \
      "$SHEAR_ROOT/target/release/search_shear_index" \
      "$SHEAR_ROOT/target/release/graph_quality"
    [[ -f "$RUNBOOK_PATH" ]] && sha256sum "$RUNBOOK_PATH"
  } >"$root/binary_sha256.txt"
  rustc -Vv >"$root/rustc.txt" 2>&1 || true
  cargo -V >"$root/cargo.txt" 2>&1 || true
  nvidia-smi -q >"$root/nvidia_smi_q.txt" 2>&1 || true
  lscpu >"$root/lscpu.txt" 2>&1 || true
  free -b >"$root/free_bytes.txt" 2>&1 || true
}

write_batch_manifest() {
  python3 - "$BATCH_ROOT" "$BATCH_KIND" "$SHEAR_ROOT" "$DATA_ROOT" <<'PY_BATCH_MANIFEST'
import datetime
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
value = {
    "schema": "shear_four_dataset_batch_v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "batch_root": str(root),
    "requested_mode": sys.argv[2],
    "shear_root": sys.argv[3],
    "data_root": sys.argv[4],
    "cache_path": "hot",
    "formal_order": ["wiki35m", "laion100m", "msmarco70m", "deep100m"],
    "build_prewarm": ["base"],
    "search_prewarm": ["base", "index", "query", "ground_truth"],
    "search_lists": [25, 50, 100, 200, 400],
    "search_warmups_per_l": 1,
    "search_measured_repetitions_per_l": 3,
}
(root / "batch_manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY_BATCH_MANIFEST
}

initialize_batch_root() {
  local requested_mode="$1"
  [[ ! -e "$BATCH_ROOT" ]] || die "BATCH_ROOT already exists: $BATCH_ROOT"
  mkdir -p "$BATCH_ROOT/provenance"
  BATCH_KIND="$requested_mode"
  BATCH_INITIALIZED=1
  printf '%s\n' "$requested_mode" >"$BATCH_ROOT/requested_mode.txt"
  record_repository_provenance
  write_batch_manifest
}

record_environment() {
  local run_root="$1"
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
  cp "$BATCH_ROOT/provenance/git_head.txt" "$run_root/git_head.txt"
  cp "$BATCH_ROOT/provenance/git_status.txt" "$run_root/git_status.txt"
  cp "$BATCH_ROOT/provenance/binary_sha256.txt" "$run_root/binary_sha256.txt"
}

write_dataset_validator() {
  local output="$1"
  cat >"$output" <<'PY_DATASET_VALIDATOR'
#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import struct


def header(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II", raw)


def edge_digest(path: Path) -> dict[str, str]:
    size = path.stat().st_size
    width = min(size, 1024 * 1024)
    with path.open("rb") as stream:
        first = stream.read(width)
        stream.seek(max(0, size - width))
        last = stream.read(width)
    return {
        "first_1mib_sha256": hashlib.sha256(first).hexdigest(),
        "last_1mib_sha256": hashlib.sha256(last).hexdigest(),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--query", type=Path, required=True)
parser.add_argument("--query-rows", type=int, required=True)
parser.add_argument("--gt", type=Path, required=True)
parser.add_argument("--gt-rows", type=int, required=True)
parser.add_argument("--gt-width", type=int, required=True)
args = parser.parse_args()

for path in (args.base, args.query, args.gt):
    if not path.is_file():
        raise RuntimeError(f"missing material: {path}")

base_rows, base_dim = header(args.base)
query_rows, query_dim = header(args.query)
gt_rows, gt_width = header(args.gt)
expected_base_size = 8 + args.rows * args.dim * 4
expected_query_size = 8 + args.query_rows * args.dim * 4
gt_id_bytes = args.gt_rows * args.gt_width * 4
valid_gt_sizes = {8 + gt_id_bytes, 8 + 2 * gt_id_bytes}
gt_size = args.gt.stat().st_size
checks = (
    (base_rows == args.rows, f"base rows {base_rows} != {args.rows}"),
    (base_dim == args.dim, f"base dim {base_dim} != {args.dim}"),
    (args.base.stat().st_size == expected_base_size, "base byte size mismatch"),
    (query_rows == args.query_rows, f"query rows {query_rows} != {args.query_rows}"),
    (query_dim == args.dim, f"query dim {query_dim} != {args.dim}"),
    (args.query.stat().st_size == expected_query_size, "query byte size mismatch"),
    (gt_rows == args.gt_rows, f"GT rows {gt_rows} != {args.gt_rows}"),
    (gt_width == args.gt_width, f"GT width {gt_width} != {args.gt_width}"),
    (gt_size in valid_gt_sizes, f"GT byte size {gt_size} not in {sorted(valid_gt_sizes)}"),
)
for valid, message in checks:
    if not valid:
        raise RuntimeError(message)

maximum_id = -1
remaining = args.gt_rows * args.gt_width
with args.gt.open("rb") as stream:
    stream.seek(8)
    while remaining:
        count = min(remaining, 1024 * 1024)
        payload = stream.read(count * 4)
        if len(payload) != count * 4:
            raise RuntimeError("short GT ID payload")
        for (value,) in struct.iter_unpack("<I", payload):
            if value >= args.rows:
                raise RuntimeError(f"GT id {value} is outside [0, {args.rows})")
            maximum_id = max(maximum_id, value)
        remaining -= count

result = {
    "schema": "shear_dataset_validation_v1",
    "base": {"path": str(args.base.resolve()), "rows": base_rows, "dim": base_dim,
             "bytes": args.base.stat().st_size, **edge_digest(args.base)},
    "query": {"path": str(args.query.resolve()), "rows": query_rows, "dim": query_dim,
              "bytes": args.query.stat().st_size, **edge_digest(args.query)},
    "ground_truth": {"path": str(args.gt.resolve()), "rows": gt_rows, "width": gt_width,
                     "bytes": gt_size, "maximum_id": maximum_id,
                     "has_distance_payload": gt_size == 8 + 2 * gt_id_bytes,
                     **edge_digest(args.gt)},
}
print(json.dumps(result, indent=2, sort_keys=True))
PY_DATASET_VALIDATOR
  chmod +x "$output"
}

validate_dataset() {
  local validator="$RUN_ROOT/tools/validate_dataset.py"
  mkdir -p "$RUN_ROOT/tools"
  write_dataset_validator "$validator"
  python3 "$validator" \
    --base "$BASE" --rows "$ROWS" --dim "$DIM" \
    --query "$QUERY" --query-rows "$QUERY_ROWS" \
    --gt "$GT" --gt-rows "$GT_ROWS" --gt-width "$GT_WIDTH" \
    >"$RUN_ROOT/dataset_validation.json"
}

record_cache_state() {
  local phase="$1" label="$2" root="$3"
  if [[ ! -f "$root/cache_protocol.tsv" ]]; then
    printf 'sample_epoch\tphase\tlabel\tmetric\tvalue_kib\n' >"$root/cache_protocol.tsv"
  fi
  awk -v epoch="$(date +%s.%N)" -v phase="$phase" -v label="$label" '
    /^(MemFree|MemAvailable|Cached|Buffers|SReclaimable):/ {
      key=$1; sub(":$", "", key); printf "%s\t%s\t%s\t%s\t%s\n", epoch, phase, label, key, $2
    }' /proc/meminfo >>"$root/cache_protocol.tsv"
}

drop_page_cache() {
  local phase="$1" root="$2"
  record_cache_state "$phase" before_drop "$root"
  sync
  printf 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  record_cache_state "$phase" after_drop "$root"
}

prewarm_paths() {
  local phase="$1" root="$2"
  shift 2
  local path started ended bytes elapsed elapsed_file
  printf 'phase\tpath\tbytes\tstarted_epoch_s\tended_epoch_s\telapsed_s\n' \
    >"$root/${phase}_prewarm_manifest.tsv"
  for path in "$@"; do
    require_file "$path"
    bytes="$(stat -Lc %s -- "$path")"
    started="$(date +%s.%N)"
    elapsed_file="$root/.${phase}_prewarm_elapsed"
    /usr/bin/time -f '%e' -o "$elapsed_file" \
      dd if="$path" of=/dev/null bs=16M status=none
    elapsed="$(<"$elapsed_file")"
    rm -f -- "$elapsed_file"
    ended="$(date +%s.%N)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$phase" "$path" "$bytes" "$started" "$ended" "$elapsed" \
      >>"$root/${phase}_prewarm_manifest.tsv"
  done
  record_cache_state "$phase" after_warm "$root"
}

record_cache_path() {
  printf 'hot\n' >"$RUN_ROOT/cache_path"
  printf 'cache_path=hot\n' >"$RUN_ROOT/cache_path.txt"
}

prepare_build_cache() {
  drop_page_cache build "$RUN_ROOT"
  prewarm_paths build "$RUN_ROOT" "$BASE"
}

prepare_search_cache() {
  drop_page_cache search "$RUN_ROOT"
  prewarm_paths search "$RUN_ROOT" "$BASE" "$INDEX_FILE" "$QUERY" "$GT"
}

verify_prerequisites() {
  require_commands bash python3 git sha256sum jq dd sync sudo nvidia-smi \
    stat du awk sed grep find flock setsid ps vmstat pidstat iostat g++ /usr/bin/time
  require_file "$SHEAR_ROOT/script/monitor_gpu_build.sh"
  require_file "$SHEAR_ROOT/target/release/build_shear_index"
  require_file "$SHEAR_ROOT/target/release/search_shear_index"
  require_file "$SHEAR_ROOT/target/release/graph_quality"
}

exclusive_host_gate() {
  sudo -n true >/dev/null || die "passwordless sudo is required to drop page cache"
  local active
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$active" ]] || die "GPU has active compute processes: $active"
  mkdir -p "$(dirname "$LOCK_PATH")"
  exec {EXPERIMENT_LOCK_FD}>"$LOCK_PATH"
  flock -n "$EXPERIMENT_LOCK_FD" || die "another SHEAR four-dataset run holds $LOCK_PATH"
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
  python3 - "$run_root" <<'PY_RESOURCE_SUMMARY'
import csv
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
process_path = root / "process_tree_rss.tsv"
valid = []
empty = 0
epochs = []
if process_path.is_file():
    with process_path.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            epochs.append(float(row["sample_epoch"]))
            if int(row["members"]):
                valid.append(row)
            else:
                empty += 1
peak = max(valid, key=lambda row: int(row["rss_kib"])) if valid else None
temp_rows = []
temp_path = root / "temp_usage.tsv"
if temp_path.is_file():
    with temp_path.open() as stream:
        temp_rows = list(csv.DictReader(stream, delimiter="\t"))
peak_hbm_mib = 0.0
gpu_summary = root / "gpu_monitor" / "summary.txt"
if gpu_summary.is_file():
    match = re.search(r"^memory_used_max_mib=([0-9.]+)$", gpu_summary.read_text(), re.M)
    if match:
        peak_hbm_mib = float(match.group(1))
summary = {
    "schema": "shear_resource_summary_v1",
    "valid_process_samples": len(valid),
    "empty_process_samples": empty,
    "missed_process_intervals": sum((b - a) > 2.5 for a, b in zip(epochs, epochs[1:])),
    "peak_process_tree_rss_bytes": int(peak["rss_kib"]) * 1024 if peak else 0,
    "peak_process_tree_members": peak["pid_rss_comm"] if peak else "",
    "peak_hbm_bytes": int(peak_hbm_mib * 1024 * 1024),
    "peak_temp_allocated_bytes": max((int(row["allocated_bytes"]) for row in temp_rows), default=0),
    "peak_temp_apparent_bytes": max((int(row["apparent_bytes"]) for row in temp_rows), default=0),
}
(root / "resource_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not valid:
    raise SystemExit("no valid process-tree RSS samples")
if peak_hbm_mib <= 0:
    raise SystemExit("no valid GPU HBM samples")
PY_RESOURCE_SUMMARY
}

write_wrapper_timing() {
  local start_epoch="$1" end_epoch="$2" exit_code="$3"
  python3 - "$RUN_ROOT/build_wrapper_timing.json" "$start_epoch" "$end_epoch" "$exit_code" <<'PY_WRAPPER_TIMING'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
start = float(sys.argv[2])
end = float(sys.argv[3])
value = {
    "schema": "shear_build_wrapper_timing_v1",
    "start_epoch_s": start,
    "end_epoch_s": end,
    "elapsed_s": end - start,
    "exit_code": int(sys.argv[4]),
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY_WRAPPER_TIMING
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
  local start_epoch end_epoch build_rc gpu_wrapper_pid rss_pid pgid attempt
  rm -f "$stop_file" "$pgid_file"
  write_status "$run_root/status.json" running 0 "" build
  vmstat -t 1 >"$run_root/vmstat.log" 2>&1 & local vmstat_pid=$!
  pidstat -r -d -u -p ALL 1 >"$run_root/pidstat.log" 2>&1 & local pidstat_pid=$!
  iostat -dxz -t 1 >"$run_root/iostat.log" 2>&1 & local iostat_pid=$!
  sample_meminfo "$stop_file" "$run_root/meminfo.tsv" & local meminfo_pid=$!
  sample_temp "$stop_file" "$temp_root" "$run_root/temp_usage.tsv" & local temp_pid=$!
  ACTIVE_MONITOR_PIDS=("$vmstat_pid" "$pidstat_pid" "$iostat_pid" "$meminfo_pid" "$temp_pid")

  start_epoch="$(date +%s.%N)"
  bash "$SHEAR_ROOT/script/monitor_gpu_build.sh" \
    --out-dir "$run_root/gpu_monitor" --gpu "$CUDA_DEVICE" --interval 1 -- \
    setsid bash -c '
      pgid_file=$1; time_log=$2; stdout_log=$3; stderr_log=$4; payload=$5
      printf "%s\n" "$$" >"$pgid_file"
      exec /usr/bin/time -v -o "$time_log" "$payload" >"$stdout_log" 2>"$stderr_log"
    ' _ "$pgid_file" "$run_root/time.log" "$run_root/build.stdout.log" \
      "$run_root/build.stderr.log" "$command_file" &
  gpu_wrapper_pid=$!

  for attempt in $(seq 1 200); do
    [[ -s "$pgid_file" ]] && break
    kill -0 "$gpu_wrapper_pid" 2>/dev/null || break
    sleep 0.05
  done
  if [[ ! -s "$pgid_file" ]]; then
    set +e
    wait "$gpu_wrapper_pid"
    build_rc=$?
    set -e
    end_epoch="$(date +%s.%N)"
    touch "$stop_file"
    cleanup_monitors
    write_wrapper_timing "$start_epoch" "$end_epoch" "$build_rc"
    return "${build_rc:-1}"
  fi

  pgid="$(<"$pgid_file")"
  ACTIVE_BUILD_PGID="$pgid"
  sample_pgid "$pgid" "$run_root/process_tree_rss.tsv" &
  rss_pid=$!
  ACTIVE_MONITOR_PIDS+=("$rss_pid")
  set +e
  wait "$gpu_wrapper_pid"
  build_rc=$?
  set -e
  wait "$rss_pid" 2>/dev/null || true
  ACTIVE_BUILD_PGID=""
  end_epoch="$(date +%s.%N)"
  touch "$stop_file"
  cleanup_monitors
  write_wrapper_timing "$start_epoch" "$end_epoch" "$build_rc"
  if [[ "$build_rc" -eq 0 ]]; then
    summarize_resources "$run_root"
    write_status "$run_root/status.json" built 0 "" "build completed"
  else
    summarize_resources "$run_root" || true
  fi
  return "$build_rc"
}

run_shear_build() {
  mkdir -p "$INDEX_DIR" "$OOM_TEMP_DIR" "$SQ_SIDECAR_DIR"
  run_measured_build "$RUN_ROOT" "$OOM_TEMP_DIR" env \
    SHEAR_PIPNN_PROFILE=1 \
    SHEAR_PIPNN_PHASE_MONITOR=1 \
    SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 \
    SHEAR_PIPNN_WMMA_CTA_SHARED_B=0 \
    SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0 \
    SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0 \
    SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0 \
    SHEAR_PIPNN_PRUNE_BACKEND=hashprune \
    SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$SHEAR_ROOT/target/release/build_shear_index" \
    --data-type float --dist-fn l2 \
    --data-path "$BASE" --index-path-prefix "$INDEX_DIR" --overwrite \
    --oom-enable --oom-memory-budget-gb 0 --oom-io-mode buffered \
    --oom-temp-dir "$OOM_TEMP_DIR" \
    --oom-profile-json "$RUN_ROOT/profile.json" \
    --driver-timing-json "$RUN_ROOT/driver_timing.json" \
    --max-degree 64 --lbuild 100 --num-threads 8 \
    --leaf-size-min 256 --leaf-size-max 1280 --leaf-knn 2 \
    --psamp-fraction 0.05 --fanout-top 8 --fanout-second 3 \
    --root-risk-candidate-rs 1,2,4,8 --seed 42 \
    --partitioner gpu-subtree \
    --gpu-subtree-root-assignment-backend cpu \
    --gpu-subtree-assignment-backend persistent-wmma-tf32 \
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 \
    --gpu-subtree-execution-mode fused-persistent \
    --gpu-subtree-d0-capacity-enable \
    --gpu-subtree-d0-capacity-split-recursion-limit "$SPLIT_RECURSION_LIMIT" \
    --gpu-subtree-d0-capacity-high-ratio 0.08 \
    --gpu-subtree-d0-capacity-target-ratio 0.08 \
    --gpu-subtree-d0-capacity-low-ratio 0.08 \
    --gpu-subtree-d0-partition-mode gpu-kmeans-coarse \
    --gpu-subtree-sq-staging-mode gpu-d0 \
    --gpu-subtree-sq-staging-upload-shard-rows 262144 \
    --gpu-subtree-sq-wmma-mode build \
    --gpu-subtree-sq-wmma-sidecar-prefix "$SQ_SIDECAR_PREFIX" \
    --gpu-subtree-sq-wmma-sidecar-mode rebuild \
    --gpu-subtree-persistent-large-ticket-batch 8 \
    --gpu-subtree-sq8-assignment-kernel wmma \
    --gpu-subtree-sq8-d0-assignment-kernel wmma-batch \
    --gpu-subtree-sq8-leaf-mode auto \
    --gpu-subtree-persistent-scheduler-mode auto \
    --pipnn-backend cuda --pipnn-gpu-device 0 \
    --pipnn-gpu-workspace-budget-gb 16
}

validate_shear_profile() {
  python3 - "$RUN_ROOT/profile.json" "$DATASET" "$ROWS" "$DIM" "$LEAF_MODE" \
    "$EXPERIMENT_ROLE" "$RUN_ROOT/profile_audit.json" <<'PY_PROFILE_AUDIT'
import json
import math
from pathlib import Path
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


profile_path = Path(sys.argv[1])
dataset = sys.argv[2]
rows_expected = int(sys.argv[3])
dim_expected = int(sys.argv[4])
leaf_expected = sys.argv[5]
role = sys.argv[6]
output = Path(sys.argv[7])
profile = json.loads(profile_path.read_text())
gpu = profile["gpu_subtree"]

require(profile["oom_io_mode"] == "buffered", "OOM I/O mode is not buffered")
require(profile["partitioner"]["kind"] == "gpu-subtree", "partitioner is not gpu-subtree")
require(gpu["enabled"] is True, "GPU subtree is not enabled")
require(gpu["root_assignment_backend"] == "cpu", "root assignment backend mismatch")
require(gpu["d0_partition_mode"] == "gpu-kmeans-coarse", "D0 partition mode mismatch")
require(gpu["sq_staging_mode"] == "gpu-d0", "SQ8 staging mode mismatch")
require(gpu["assignment_backend"] == "persistent-wmma-tf32", "assignment backend mismatch")
require(gpu["subtree_assignment_backend"] == "persistent-wmma-tf32", "subtree assignment backend mismatch")
require(gpu["leaf_topk_backend"] == "persistent-wmma-tf32", "leaf backend mismatch")
require(gpu["sq_wmma_mode"] == "build", "SQ8 WMMA mode mismatch")
require(gpu["sq_wmma_sidecar_mode"] == "rebuild", "SQ8 sidecar mode mismatch")
require(int(gpu["persistent_effective_deeper_fanout"]) == 1, "deeper fanout is not 1")

capacity = gpu["d0_capacity"]
require(capacity["enabled"] is True, "D0 capacity policy is disabled")
c_gpu = int(capacity["c_gpu"])
require(c_gpu > 0, "invalid c_gpu")
tolerance = 1.0 / c_gpu
for field in ("c_high", "c_target", "c_low"):
    ratio = int(capacity[field]) / c_gpu
    require(abs(ratio - 0.08) <= tolerance, f"{field} ratio {ratio} is not 8%")

sq_d0 = gpu["sq_gpu_staging"]
requested = {
    "ticket_batch": gpu["persistent_large_ticket_batch_requested"],
    "assignment": gpu["persistent_sq8_assignment_kernel_requested"],
    "d0_assignment": sq_d0["d0_assign_kernel_requested"],
    "leaf": gpu["persistent_sq8_leaf_mode_requested"],
    "scheduler": gpu["persistent_scheduler_mode_requested"],
}
expected_requested = {
    "ticket_batch": "8",
    "assignment": "wmma",
    "d0_assignment": "wmma-batch",
    "leaf": "auto",
    "scheduler": "auto",
}
require(requested == expected_requested,
        f"requested modes {requested} != {expected_requested}")
actual = {
    "ticket_batch": int(gpu["persistent_large_ticket_batch_actual"]),
    "assignment": gpu["persistent_sq8_assignment_kernel_actual"],
    "leaf": gpu["persistent_sq8_leaf_mode_actual"],
    "scheduler": gpu["persistent_scheduler_mode_actual"],
}
require(actual["ticket_batch"] == 8, f"actual ticket batch is {actual['ticket_batch']}")
require(actual["assignment"] == "wmma", f"actual assignment kernel is {actual['assignment']}")
require(actual["leaf"] == leaf_expected, f"actual leaf mode {actual['leaf']} != {leaf_expected}")
require(actual["scheduler"] == "baseline", f"actual scheduler is {actual['scheduler']}")
d0_actual = {
    "kernel": sq_d0["d0_assign_kernel_mode"],
    "row_tile": int(sq_d0["d0_assign_row_tile"]),
    "distance_kernel_ms": float(sq_d0["d0_assign_distance_kernel_ms"]),
    "rows": int(sq_d0["d0_assign_rows"]),
}
require(d0_actual["kernel"] == "cuda-wmma-batch",
        f"actual D0 kernel is {d0_actual['kernel']}")
require(d0_actual["row_tile"] == 8,
        f"actual D0 row tile is {d0_actual['row_tile']}")
require(math.isfinite(d0_actual["distance_kernel_ms"])
        and d0_actual["distance_kernel_ms"] > 0,
        f"invalid D0 distance kernel time: {d0_actual['distance_kernel_ms']}")
require(d0_actual["rows"] == rows_expected,
        f"D0 rows {d0_actual['rows']} != {rows_expected}")
wmma_tiles = int(gpu["persistent_large_sq8_wmma_tiles"])
dp4a_tiles = int(gpu["persistent_large_sq8_dp4a_tiles"])
require(wmma_tiles > 0, "no SQ8 WMMA assignment tiles were executed")
require(dp4a_tiles == 0, f"unexpected DP4A tiles: {dp4a_tiles}")
if leaf_expected == "row-owned":
    require(int(gpu["persistent_leaf_row_lock_merge_clocks"]) == 0,
            "row-owned leaf performed row-lock merge work")

physical_tiles = int(gpu["persistent_large_physical_tiles"])
range_claims = int(gpu["persistent_large_ticket_range_claims"])
require(physical_tiles > 0, "no physical large-task tiles")
require(range_claims > 0, "no ticket-range claims")
require(range_claims * 8 >= physical_tiles, "range claims undercount physical tiles")
claim_ratio = range_claims / physical_tiles
require(claim_ratio >= 0.125, f"ticket claim ratio below 1/8: {claim_ratio}")
if role == "formal":
    require(claim_ratio <= 0.15, f"formal ticket claim ratio is not near 1/8: {claim_ratio}")

zero_fields = (
    (gpu, "persistent_queue_overflow_flags"),
    (gpu, "persistent_fallback_count"),
    (gpu, "cpu_fallbacks"),
    (gpu, "async_flush_errors"),
    (profile["gpu"], "gpu_cpu_fallbacks"),
    (profile["scheduler"], "gpu_cpu_fallbacks"),
)
for owner, field in zero_fields:
    require(int(owner[field]) == 0, f"{field} is nonzero: {owner[field]}")

root_tail = float(gpu["post_last_gpu_cpu_tail_ms"])
diag = profile["serial_timing"]["diagnostics"]
diag_tail = float(diag["post_last_gpu_cpu_tail_ms"])
require(gpu["post_last_gpu_cpu_tail_observed"] is True, "root CPU tail was not observed")
require(diag["post_last_gpu_cpu_tail_observed"] is True, "diagnostic CPU tail was not observed")
tail_tolerance = max(1e-6, 1e-9 * abs(root_tail))
require(abs(root_tail - diag_tail) <= tail_tolerance, "root and diagnostic CPU tails differ")
async_wait = float(gpu["async_flush_join_wait_ms"])
require(root_tail >= async_wait, f"CPU tail {root_tail} < async join wait {async_wait}")

serial = profile["serial_timing"]
serial_rows = serial["rows"]
require("post_last_gpu_cpu_tail_ms" not in serial_rows, "CPU tail was added to serial rows")
require("post_last_gpu_cpu_tail_observed" not in serial_rows, "CPU-tail observed flag was added to serial rows")
accounted = int(serial["accounted_sum_ms"])
stage_total = int(serial["stage_total_ms"])
require(accounted == stage_total, "accounted sum differs from stage total")
require(serial["sum_matches_stage_total"] is True, "serial sum flag is false")
row_sum = sum(int(value) for value in serial_rows.values())
require(abs(row_sum - accounted) <= len(serial_rows),
        f"serial row sum {row_sum} is inconsistent with accounted {accounted}")

require(int(profile["num_points"]) == rows_expected, "profile row count mismatch")
require(int(profile["dim"]) == dim_expected, "profile dimension mismatch")
audit = {
    "schema": "shear_profile_audit_v1",
    "dataset": dataset,
    "rows": rows_expected,
    "dim": dim_expected,
    "requested": requested,
    "actual": actual,
    "d0_assignment": d0_actual,
    "persistent_kernel_event_ms": float(gpu["persistent_kernel_event_ms"]),
    "post_last_gpu_cpu_tail_ms": root_tail,
    "post_last_gpu_cpu_tail_observed": True,
    "cpu_compact_flush_ms": float(gpu["cpu_compact_flush_ms"]),
    "async_flush_join_wait_ms": async_wait,
    "ticket_range_claims": range_claims,
    "physical_tiles": physical_tiles,
    "ticket_range_claim_ratio": claim_ratio,
    "wmma_tiles": wmma_tiles,
    "dp4a_tiles": dp4a_tiles,
    "serial_accounted_sum_ms": accounted,
    "serial_stage_total_ms": stage_total,
}
for value in audit.values():
    if isinstance(value, float):
        require(math.isfinite(value), "profile audit contains non-finite value")
output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
PY_PROFILE_AUDIT
}

initialize_schedule() {
  local datasets=("$@") dataset ordinal=0
  printf 'dataset\tordinal\tstate\trun_root\tstarted_at\tended_at\treason\n' >"$BATCH_ROOT/formal_schedule.tsv"
  for dataset in "${datasets[@]}"; do
    ordinal=$((ordinal + 1))
    printf '%s\t%s\tpending\t%s\t\t\t\n' \
      "$dataset" "$ordinal" "$BATCH_ROOT/hot_path/$dataset/shear/run_1" \
      >>"$BATCH_ROOT/formal_schedule.tsv"
  done
}

update_schedule() {
  local dataset="$1" state="$2" reason="${3:-}"
  [[ -f "$BATCH_ROOT/formal_schedule.tsv" ]] || return 0
  python3 - "$BATCH_ROOT/formal_schedule.tsv" "$dataset" "$state" "$reason" <<'PY_UPDATE_SCHEDULE'
import csv
import datetime
from pathlib import Path
import sys

path = Path(sys.argv[1])
dataset = sys.argv[2]
state = sys.argv[3]
reason = sys.argv[4]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
with path.open() as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
matched = False
for row in rows:
    if row["dataset"] != dataset:
        continue
    matched = True
    row["state"] = state
    if state == "running" and not row["started_at"]:
        row["started_at"] = now
    if state in {"accepted", "failed", "interrupted"}:
        row["ended_at"] = now
    row["reason"] = reason
if not matched:
    raise RuntimeError(f"dataset is absent from schedule: {dataset}")
tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=(
        "dataset", "ordinal", "state", "run_root", "started_at", "ended_at", "reason"
    ), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
tmp.replace(path)
PY_UPDATE_SCHEDULE
}

cleanup_successful_dataset() {
  local manifest="$RUN_ROOT/cleanup_manifest.tsv"
  local summary="$RUN_ROOT/cleanup_summary.txt"
  local targets=("$INDEX_DIR" "$OOM_TEMP_DIR" "$SQ_SIDECAR_DIR")
  if [[ "$EXPERIMENT_ROLE" == smoke ]]; then
    targets+=("$SMOKE_DATA_DIR")
  fi
  local target bytes total=0
  printf 'kind\tbytes\tpath\n' >"$manifest"
  for target in "${targets[@]}"; do
    bytes=0
    if [[ -e "$target" || -L "$target" ]]; then
      bytes="$(du -sb -- "$target" | awk '{print $1}')"
    fi
    printf 'shear\t%s\t%s\n' "$bytes" "$target" >>"$manifest"
    total=$((total + bytes))
  done
  {
    printf 'dataset=%s\n' "$DATASET"
    printf 'recorded_bytes=%s\n' "$total"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'verified=false\n'
  } >"$summary"
  for target in "${targets[@]}"; do
    if [[ -e "$target" || -L "$target" ]]; then
      rm -rf -- "$target"
    fi
  done
}

verify_dataset_cleanup() {
  local required_evidence=(
    time.log profile.json driver_timing.json build_wrapper_timing.json profile_audit.json resource_summary.json
    graph_validation.json graph_quality.log graph_quality.tsv index_size_before_cleanup.json
    search_sweep.tsv search_summary.tsv cache_protocol.tsv
    build_prewarm_manifest.tsv search_prewarm_manifest.tsv
    command.sh status.json dataset_validation.json environment.txt
    cache_path cache_path.txt binary_sha256.txt
  )
  local evidence row_kind row_bytes row_path rows=0
  require_file "$RUN_ROOT/cleanup_manifest.tsv"
  require_file "$RUN_ROOT/cleanup_summary.txt"
  for evidence in "${required_evidence[@]}"; do
    require_file "$RUN_ROOT/$evidence"
  done
  while IFS=$'\t' read -r row_kind row_bytes row_path; do
    [[ "$row_kind" == kind ]] && continue
    [[ "$row_kind" == shear ]] || die "cleanup manifest kind mismatch: $row_kind"
    [[ "$row_bytes" =~ ^[0-9]+$ ]] || die "invalid cleanup byte count: $row_bytes"
    [[ ! -e "$row_path" && ! -L "$row_path" ]] || die "cleanup target remains: $row_path"
    rows=$((rows + 1))
  done <"$RUN_ROOT/cleanup_manifest.tsv"
  ((rows >= 3)) || die "cleanup manifest is incomplete"
  python3 - "$RUN_ROOT/cleanup_summary.txt" <<'PY_CLEANUP_VERIFY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
if "verified=false" not in lines:
    raise RuntimeError("cleanup summary is not in pre-verification state")
path.write_text("\n".join("verified=true" if line == "verified=false" else line for line in lines) + "\n")
PY_CLEANUP_VERIFY
}

finalize_successful_dataset() {
  local evidence
  local required_before_cleanup=(
    time.log profile.json driver_timing.json build_wrapper_timing.json profile_audit.json resource_summary.json
    graph_validation.json graph_quality.log graph_quality.tsv index_size_before_cleanup.json
    search_sweep.tsv search_summary.tsv cache_protocol.tsv
    build_prewarm_manifest.tsv search_prewarm_manifest.tsv
    command.sh status.json dataset_validation.json environment.txt
    cache_path cache_path.txt binary_sha256.txt
  )
  CURRENT_STAGE=cleanup
  for evidence in "${required_before_cleanup[@]}"; do
    require_file "$RUN_ROOT/$evidence"
  done
  cleanup_successful_dataset
  verify_dataset_cleanup
  write_status "$RUN_ROOT/status.json" accepted 0 "" "cleanup verified"
  update_schedule "$DATASET" accepted "cleanup verified"
  log "accepted dataset: $DATASET ($RUN_ROOT)"
}

generate_batch_summary() {
  [[ -d "$BATCH_ROOT" ]] || return 0
  python3 - "$BATCH_ROOT" <<'PY_BATCH_SUMMARY'
import csv
import json
import math
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
formal_datasets = ["wiki35m", "laion100m", "msmarco70m", "deep100m"]
requested_mode_path = root / "requested_mode.txt"
requested_mode = requested_mode_path.read_text().strip() if requested_mode_path.is_file() else "formal"
datasets = formal_datasets
if requested_mode == "smoke" or (root / "hot_path/smoke").exists():
    datasets = ["smoke"]
elif requested_mode in formal_datasets:
    datasets = [requested_mode]


def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(value)


def parse_time(path: Path) -> dict[str, float | int | None]:
    result = {"external_build_wall_s": None, "maximum_rss_bytes": None,
              "filesystem_inputs": None, "filesystem_outputs": None}
    if not path.is_file():
        return result
    text = path.read_text(errors="replace")
    patterns = {
        "external_build_wall_s": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)",
        "maximum_rss_bytes": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "filesystem_inputs": r"File system inputs:\s*(\d+)",
        "filesystem_outputs": r"File system outputs:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        if key == "external_build_wall_s":
            result[key] = elapsed_seconds(match.group(1))
        elif key == "maximum_rss_bytes":
            result[key] = int(match.group(1)) * 1024
        else:
            result[key] = int(match.group(1))
    return result


def search_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open() as stream:
        return {int(row["l"]): row for row in csv.DictReader(stream, delimiter="\t")}


rows = []
for dataset in datasets:
    run = root / "hot_path" / dataset / "shear" / "run_1"
    status = load_json(run / "status.json")
    state = status["phase"] if status else "missing"
    timing = parse_time(run / "time.log")
    driver = load_json(run / "driver_timing.json") or {}
    profile = load_json(run / "profile.json") or {}
    audit = load_json(run / "profile_audit.json") or {}
    resources = load_json(run / "resource_summary.json") or {}
    index_size = load_json(run / "index_size_before_cleanup.json") or {}
    searches = search_rows(run / "search_summary.tsv")
    wall = timing["external_build_wall_s"]
    tail = audit.get("post_last_gpu_cpu_tail_ms")
    row = {
        "dataset": dataset,
        "status": state,
        "external_build_wall_s": wall,
        "driver_wall_ms": driver.get("total_wall_ms"),
        "profile_stage_total_ms": driver.get("profile_stage_total_ms") or
                                  (profile.get("serial_timing") or {}).get("stage_total_ms"),
        "persistent_kernel_event_ms": audit.get("persistent_kernel_event_ms"),
        "post_last_gpu_cpu_tail_ms": tail,
        "cpu_compact_flush_ms": audit.get("cpu_compact_flush_ms"),
        "async_flush_join_wait_ms": audit.get("async_flush_join_wait_ms"),
        "post_last_gpu_cpu_tail_over_build_wall":
            tail / (wall * 1000.0) if tail is not None and wall else None,
        "peak_process_tree_rss_bytes": resources.get("peak_process_tree_rss_bytes") or
                                       timing["maximum_rss_bytes"],
        "peak_hbm_bytes": resources.get("peak_hbm_bytes"),
        "filesystem_inputs": timing["filesystem_inputs"],
        "filesystem_outputs": timing["filesystem_outputs"],
        "index_bytes_before_cleanup": index_size.get("bytes"),
        "actual_ticket_batch": (audit.get("actual") or {}).get("ticket_batch"),
        "actual_assignment_kernel": (audit.get("actual") or {}).get("assignment"),
        "actual_leaf_mode": (audit.get("actual") or {}).get("leaf"),
        "actual_scheduler": (audit.get("actual") or {}).get("scheduler"),
    }
    for l in (25, 50, 100, 200, 400):
        values = searches.get(l, {})
        for metric in ("recall_mean", "recall_stdev", "qps_mean", "qps_stdev"):
            row[f"L{l}_{metric}"] = values.get(metric)
    rows.append(row)

base_fields = [
    "dataset", "status", "external_build_wall_s", "driver_wall_ms",
    "profile_stage_total_ms", "persistent_kernel_event_ms",
    "post_last_gpu_cpu_tail_ms", "cpu_compact_flush_ms", "async_flush_join_wait_ms",
    "post_last_gpu_cpu_tail_over_build_wall", "peak_process_tree_rss_bytes",
    "peak_hbm_bytes", "filesystem_inputs", "filesystem_outputs",
    "index_bytes_before_cleanup", "actual_ticket_batch", "actual_assignment_kernel",
    "actual_leaf_mode", "actual_scheduler",
]
search_fields = [f"L{l}_{metric}" for l in (25, 50, 100, 200, 400)
                 for metric in ("recall_mean", "recall_stdev", "qps_mean", "qps_stdev")]
fields = base_fields + search_fields
with (root / "summary.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# SHEAR Four-Dataset Warm-Cache Summary",
    "",
    "External `/usr/bin/time -v` wall time is the primary build time. Prewarm is excluded.",
    "Filesystem input/output columns are the raw GNU time counters.",
    "",
    "## Build and telemetry",
    "",
    "| Dataset | Status | Build wall (s) | Driver (ms) | Stage total (ms) | Persistent (ms) | CPU tail (ms) | Tail / wall | Peak RSS (bytes) | Peak HBM (bytes) | Index bytes | Actual modes |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    modes = "/".join(str(row[key] or "-") for key in (
        "actual_ticket_batch", "actual_assignment_kernel", "actual_leaf_mode", "actual_scheduler"
    ))
    ratio = row["post_last_gpu_cpu_tail_over_build_wall"]
    lines.append("| {dataset} | {status} | {wall} | {driver} | {stage} | {persistent} | {tail} | {ratio} | {rss} | {hbm} | {index} | {modes} |".format(
        dataset=row["dataset"], status=row["status"],
        wall=row["external_build_wall_s"] if row["external_build_wall_s"] is not None else "-",
        driver=row["driver_wall_ms"] if row["driver_wall_ms"] is not None else "-",
        stage=row["profile_stage_total_ms"] if row["profile_stage_total_ms"] is not None else "-",
        persistent=row["persistent_kernel_event_ms"] if row["persistent_kernel_event_ms"] is not None else "-",
        tail=row["post_last_gpu_cpu_tail_ms"] if row["post_last_gpu_cpu_tail_ms"] is not None else "-",
        ratio=f"{ratio:.6f}" if ratio is not None and math.isfinite(ratio) else "-",
        rss=row["peak_process_tree_rss_bytes"] or "-", hbm=row["peak_hbm_bytes"] or "-",
        index=row["index_bytes_before_cleanup"] or "-", modes=modes,
    ))
lines.extend(["", "## Search", "",
              "| Dataset | L | Recall@10 mean | Recall std | QPS mean | QPS std |",
              "|---|---:|---:|---:|---:|---:|"])
for row in rows:
    for l in (25, 50, 100, 200, 400):
        values = [row[f"L{l}_{name}"] for name in
                  ("recall_mean", "recall_stdev", "qps_mean", "qps_stdev")]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row["dataset"], l, *(value if value not in (None, "") else "-" for value in values)))
failed = [row["dataset"] for row in rows if row["status"] not in {"accepted"}]
lines.extend(["", "## Failed or missing", "", ", ".join(failed) if failed else "None", ""])
(root / "summary.md").write_text("\n".join(lines))
PY_BATCH_SUMMARY
}

run_dataset_configured() {
  CURRENT_DATASET="$DATASET"
  CURRENT_RUN_ROOT="$RUN_ROOT"
  CURRENT_STAGE=preflight
  fresh_dir "$RUN_ROOT"
  mkdir -p "$RUN_ROOT/tools"
  record_cache_path
  write_status "$RUN_ROOT/status.json" running 0 "" preflight
  update_schedule "$DATASET" running ""
  validate_dataset
  record_environment "$RUN_ROOT"
  require_file "$BATCH_ROOT/provenance/git_head.txt"

  CURRENT_STAGE=build_prewarm
  prepare_build_cache
  CURRENT_STAGE=build
  run_shear_build
  require_file "$RUN_ROOT/profile.json"
  require_file "$RUN_ROOT/driver_timing.json"

  CURRENT_STAGE=profile_audit
  validate_shear_profile
  CURRENT_STAGE=graph_validation
  validate_graph

  CURRENT_STAGE=search_prewarm
  prepare_search_cache
  CURRENT_STAGE=search
  run_search_sweep
  summarize_search

  finalize_successful_dataset
  generate_batch_summary
  CURRENT_RUN_ROOT=""
  CURRENT_DATASET=""
  CURRENT_STAGE=""
}

run_dataset() {
  load_dataset "$1"
  run_dataset_configured
}

run_formal() {
  run_dataset wiki35m
  run_dataset laion100m
  run_dataset msmarco70m
  run_dataset deep100m
}

run_prepare() {
  [[ ! -e "$BATCH_ROOT" ]] || die "BATCH_ROOT already exists: $BATCH_ROOT"
  mkdir -p "$BATCH_ROOT/provenance"
  BATCH_KIND=prepare
  BATCH_INITIALIZED=1
  printf 'prepare\n' >"$BATCH_ROOT/requested_mode.txt"
  (
    cd "$SHEAR_ROOT"
    cargo test -p shear_core --lib post_last_gpu_cpu_tail -- --nocapture
    cargo check -p shear_core --features pipnn_cuda
    cargo build --locked --release -p build_shear_index --features build_shear_index/cuda
    cargo build --locked --release -p search_shear_index
    cargo build --locked --release -p search_shear_index --bin graph_quality
  ) 2>&1 | tee "$BATCH_ROOT/provenance/prepare.log"
  verify_prerequisites
  record_repository_provenance
  write_batch_manifest
  write_status "$BATCH_ROOT/prepare_status.json" accepted 0 "" "prepare passed"
}

write_smoke_data() {
  mkdir -p "$SMOKE_DATA_DIR"
  python3 - "$SMOKE_DATA_DIR" <<'PY_SMOKE_DATA'
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
noise = rng.normal(0.0, 1e-3, size=(query_rows, dim)).astype(np.float32)
queries = (base[query_ids] + noise).astype(np.float32)
base_norm = np.einsum("ij,ij->i", base, base)
ground_truth = np.empty((query_rows, width), dtype=np.uint32)
for start in range(0, query_rows, 32):
    block = queries[start:start + 32]
    query_norm = np.einsum("ij,ij->i", block, block)[:, None]
    distances = query_norm + base_norm[None, :] - 2.0 * (block @ base.T)
    np.maximum(distances, 0.0, out=distances)
    candidates = np.argpartition(distances, kth=width - 1, axis=1)[:, :width]
    for row_offset, candidate_ids in enumerate(candidates):
        candidate_distances = distances[row_offset, candidate_ids]
        order = np.lexsort((candidate_ids, candidate_distances))
        ground_truth[start + row_offset] = candidate_ids[order].astype(np.uint32)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", matrix.shape[0], matrix.shape[1]))
        matrix.tofile(stream)


write_matrix(root / "base.fbin", base)
write_matrix(root / "query.fbin", queries)
write_matrix(root / "gt100.ibin", ground_truth)
(root / "smoke_contract.json").write_text(json.dumps({
    "schema": "shear_smoke_contract_v1",
    "seed": 42,
    "rows": rows,
    "dim": dim,
    "query_rows": query_rows,
    "gt_width": width,
    "ground_truth": "exact squared L2, blockwise",
}, indent=2, sort_keys=True) + "\n")
PY_SMOKE_DATA
}

run_smoke() {
  DATASET=smoke
  ROWS=65536
  DIM=96
  QUERY_ROWS=256
  GT_ROWS=256
  GT_WIDTH=100
  LEAF_MODE=row-owned
  SPLIT_RECURSION_LIMIT=8
  EXPERIMENT_ROLE=smoke
  SMOKE_DATA_DIR="$BATCH_ROOT/smoke/data"
  BASE="$SMOKE_DATA_DIR/base.fbin"
  QUERY="$SMOKE_DATA_DIR/query.fbin"
  GT="$SMOKE_DATA_DIR/gt100.ibin"
  RUN_ROOT="$BATCH_ROOT/hot_path/smoke/shear/run_1"
  INDEX_DIR="$RUN_ROOT/index"
  INDEX_FILE="$INDEX_DIR/_mem.index"
  OOM_TEMP_DIR="$RUN_ROOT/oom_temp"
  SQ_SIDECAR_DIR="$RUN_ROOT/sq_sidecar"
  SQ_SIDECAR_PREFIX="$SQ_SIDECAR_DIR/sq_wmma"
  write_smoke_data
  cp "$SMOKE_DATA_DIR/smoke_contract.json" "$BATCH_ROOT/smoke_contract.json"
  run_dataset_configured
}

run_formal_entry() {
  verify_prerequisites
  exclusive_host_gate
  initialize_batch_root formal
  initialize_schedule "${FORMAL_DATASETS[@]}"
  run_formal
}

run_single_entry() {
  verify_prerequisites
  exclusive_host_gate
  initialize_batch_root "$1"
  initialize_schedule "$1"
  run_dataset "$1"
}

run_smoke_entry() {
  verify_prerequisites
  exclusive_host_gate
  initialize_batch_root smoke
  initialize_schedule smoke
  run_smoke
}

on_error() {
  local rc=$? line="${BASH_LINENO[0]:-unknown}"
  trap - ERR
  cleanup_monitors
  if [[ -n "$CURRENT_RUN_ROOT" && -d "$CURRENT_RUN_ROOT" ]]; then
    write_status "$CURRENT_RUN_ROOT/status.json" failed "$rc" "$CURRENT_STAGE" "line $line"
  fi
  if [[ -n "$CURRENT_DATASET" && "$BATCH_INITIALIZED" -eq 1 ]]; then
    update_schedule "$CURRENT_DATASET" failed "$CURRENT_STAGE at line $line" || true
  fi
  if [[ "$BATCH_INITIALIZED" -eq 1 && "$BATCH_KIND" != prepare ]]; then
    generate_batch_summary || true
  fi
  log "failed: dataset=${CURRENT_DATASET:-none} stage=${CURRENT_STAGE:-none} rc=$rc line=$line"
  exit "$rc"
}

on_signal() {
  local signal="$1" rc=130
  [[ "$signal" == TERM ]] && rc=143
  trap - ERR INT TERM
  cleanup_monitors
  if [[ -n "$CURRENT_RUN_ROOT" && -d "$CURRENT_RUN_ROOT" ]]; then
    write_status "$CURRENT_RUN_ROOT/status.json" interrupted "$rc" "$CURRENT_STAGE" "$signal"
  fi
  if [[ -n "$CURRENT_DATASET" && "$BATCH_INITIALIZED" -eq 1 ]]; then
    update_schedule "$CURRENT_DATASET" interrupted "$signal during $CURRENT_STAGE" || true
  fi
  if [[ "$BATCH_INITIALIZED" -eq 1 && "$BATCH_KIND" != prepare ]]; then
    generate_batch_summary || true
  fi
  exit "$rc"
}

on_exit() {
  cleanup_monitors
  if [[ "$BATCH_INITIALIZED" -eq 1 && "$BATCH_KIND" != prepare ]]; then
    generate_batch_summary || true
  fi
}

usage() {
  cat <<'EOF'
Usage: p1_p2_shear_four_dataset.sh MODE
  help       print this contract without prerequisite checks
  prepare    test/build the required binaries and record provenance
  smoke      run a 65,536-row GPU build/search/cleanup smoke test
  formal     run wiki35m, laion100m, msmarco70m, deep100m sequentially
  wiki35m | laion100m | msmarco70m | deep100m
             run exactly one formal dataset in a fresh batch root
EOF
}

ensure_graph_validator() {
  local tool_root="$BATCH_ROOT/tools"
  local source="$tool_root/strict_mem_graph_validator.cpp"
  local binary="$tool_root/strict_mem_graph_validator"
  mkdir -p "$tool_root"
  cat >"$source" <<'CPP_GRAPH_VALIDATOR'
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

template <class T>
bool read_exact(std::ifstream& in, T& value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

int main(int argc, char** argv) {
  if (argc != 5) return 2;
  const std::string path = argv[1];
  const uint64_t expected_nodes = std::stoull(argv[2]);
  const uint32_t allowed_degree = static_cast<uint32_t>(std::stoul(argv[3]));
  const std::string output = argv[4];
  std::ifstream in(path, std::ios::binary);
  if (!in) return 3;
  uint64_t header_size = 0;
  uint64_t frozen_points = 0;
  uint32_t header_degree = 0;
  uint32_t start_node = 0;
  if (!read_exact(in, header_size) || !read_exact(in, header_degree) ||
      !read_exact(in, start_node) || !read_exact(in, frozen_points)) return 4;
  const uint64_t actual_size = std::filesystem::file_size(path);
  uint64_t invalid_degree_nodes = 0;
  uint64_t invalid_edges = 0;
  uint64_t self_edges = 0;
  uint64_t duplicate_edges = 0;
  uint64_t edges = 0;
  uint32_t observed_max_degree = 0;
  std::array<uint32_t, 64> neighbors{};
  for (uint64_t node = 0; node < expected_nodes; ++node) {
    uint32_t degree = 0;
    if (!read_exact(in, degree)) return 5;
    if (degree == 0 || degree > allowed_degree || degree > neighbors.size()) {
      ++invalid_degree_nodes;
      if (degree > neighbors.size()) return 6;
    }
    observed_max_degree = std::max(observed_max_degree, degree);
    edges += degree;
    if (!in.read(reinterpret_cast<char*>(neighbors.data()), degree * sizeof(uint32_t))) return 7;
    for (uint32_t i = 0; i < degree; ++i) {
      invalid_edges += neighbors[i] >= expected_nodes;
      self_edges += neighbors[i] == node;
    }
    std::sort(neighbors.begin(), neighbors.begin() + degree);
    for (uint32_t i = 1; i < degree; ++i) {
      duplicate_edges += neighbors[i] == neighbors[i - 1];
    }
  }
  const auto position = in.tellg();
  const uint64_t consumed = position < 0 ? 0 : static_cast<uint64_t>(position);
  const bool valid = header_size == actual_size && consumed == actual_size &&
      header_degree <= allowed_degree && observed_max_degree == header_degree &&
      start_node < expected_nodes && invalid_degree_nodes == 0 && invalid_edges == 0 &&
      self_edges == 0 && duplicate_edges == 0;
  std::ofstream out(output);
  out << "{\n  \"valid\": " << (valid ? "true" : "false")
      << ",\n  \"nodes\": " << expected_nodes
      << ",\n  \"edges\": " << edges
      << ",\n  \"header_size\": " << header_size
      << ",\n  \"actual_size\": " << actual_size
      << ",\n  \"header_max_degree\": " << header_degree
      << ",\n  \"observed_max_degree\": " << observed_max_degree
      << ",\n  \"start_node\": " << start_node
      << ",\n  \"frozen_points\": " << frozen_points
      << ",\n  \"invalid_degree_nodes\": " << invalid_degree_nodes
      << ",\n  \"invalid_edges\": " << invalid_edges
      << ",\n  \"self_edges\": " << self_edges
      << ",\n  \"duplicate_edges\": " << duplicate_edges << "\n}\n";
  return valid ? 0 : 1;
}
CPP_GRAPH_VALIDATOR
  if [[ ! -x "$binary" || "$source" -nt "$binary" ]]; then
    g++ -O3 -std=c++20 -Wall -Wextra "$source" -o "$binary"
  fi
  printf '%s\n' "$binary"
}

record_index_size() {
  python3 - "$INDEX_FILE" "$RUN_ROOT/index_size_before_cleanup.json" <<'PY_INDEX_SIZE'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
output = Path(sys.argv[2])
output.write_text(json.dumps({
    "schema": "shear_index_size_v1",
    "path": str(source),
    "bytes": source.stat().st_size,
}, indent=2, sort_keys=True) + "\n")
PY_INDEX_SIZE
}

validate_graph() {
  local validator
  require_file "$INDEX_FILE"
  validator="$(ensure_graph_validator)"
  "$validator" "$INDEX_FILE" "$ROWS" 64 "$RUN_ROOT/graph_validation.json"
  "$SHEAR_ROOT/target/release/graph_quality" \
    --index-path-prefix "$INDEX_DIR" --data-path "$BASE" \
    --output "$RUN_ROOT/graph_quality.tsv" --max-edge-samples 1000000 \
    >"$RUN_ROOT/graph_quality.log" 2>&1
  record_index_size
  write_status "$RUN_ROOT/status.json" validated 0 "" "graph validation passed"
}

metric_value() {
  local metrics="$1" column="$2"
  awk -F'\t' -v wanted="$column" '
    NR == 1 {for (i=1; i<=NF; ++i) if ($i == wanted) column_number=i; next}
    NR == 2 {if (!column_number) exit 2; print $column_number}
  ' "$metrics"
}

run_search_once() {
  local l="$1" kind="$2" repetition="$3"
  local stem metrics result stdout_log stderr_log wall_file wall_seconds recall qps
  if [[ "$kind" == warmup ]]; then
    stem="warmup_L${l}"
  else
    stem="L${l}_r${repetition}"
  fi
  metrics="$RUN_ROOT/search/${stem}.tsv"
  result="$RUN_ROOT/search/${stem}.fbin"
  stdout_log="$RUN_ROOT/search/${stem}.stdout.log"
  stderr_log="$RUN_ROOT/search/${stem}.stderr.log"
  wall_file="$RUN_ROOT/search/${stem}.wall_seconds"
  /usr/bin/time -f '%e' -o "$wall_file" \
    "$SHEAR_ROOT/target/release/search_shear_index" \
    --index-path-prefix "$INDEX_DIR" --data-path "$BASE" \
    --query-file "$QUERY" --gt-file "$GT" --result-path "$result" \
    --recall-at 10 --search-list-size "$l" --num-threads 8 --storage mmap \
    --metrics-path "$metrics" \
    >"$stdout_log" 2>"$stderr_log"
  require_file "$metrics"
  if [[ "$kind" == measured ]]; then
    recall="$(metric_value "$metrics" recall)"
    qps="$(metric_value "$metrics" qps)"
    wall_seconds="$(<"$wall_file")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$DATASET" "$l" "$repetition" "$recall" "$qps" "$wall_seconds" "$metrics" \
      >>"$RUN_ROOT/search_sweep.tsv"
  fi
}

run_search_sweep() {
  mkdir -p "$RUN_ROOT/search"
  printf 'dataset\tl\trepetition\trecall_at_10\tqps\twall_seconds\tmetrics_path\n' \
    >"$RUN_ROOT/search_sweep.tsv"
  local l rep
  for l in 25 50 100 200 400; do
    run_search_once "$l" warmup 0
    for rep in 1 2 3; do
      run_search_once "$l" measured "$rep"
    done
  done
}

summarize_search() {
  python3 - "$RUN_ROOT/search_sweep.tsv" "$RUN_ROOT/search_summary.tsv" <<'PY_SEARCH_SUMMARY'
import csv
import math
from pathlib import Path
import statistics
import sys

source = Path(sys.argv[1])
output = Path(sys.argv[2])
groups: dict[int, list[dict[str, str]]] = {}
with source.open() as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        groups.setdefault(int(row["l"]), []).append(row)
if sorted(groups) != [25, 50, 100, 200, 400]:
    raise RuntimeError(f"unexpected search L values: {sorted(groups)}")
summary = []
previous_recall = -1.0
for l in sorted(groups):
    rows = groups[l]
    if len(rows) != 3:
        raise RuntimeError(f"L={l} has {len(rows)} measured repetitions")
    if sorted(int(row["repetition"]) for row in rows) != [1, 2, 3]:
        raise RuntimeError(f"L={l} repetition IDs are invalid")
    recalls = [float(row["recall_at_10"]) for row in rows]
    qps_values = [float(row["qps"]) for row in rows]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in recalls):
        raise RuntimeError(f"L={l} has invalid recall")
    if not all(math.isfinite(value) and value > 0.0 for value in qps_values):
        raise RuntimeError(f"L={l} has invalid QPS")
    if max(recalls) - min(recalls) > 1e-6:
        raise RuntimeError(f"L={l} recall changed across repetitions: {recalls}")
    recall_mean = statistics.mean(recalls)
    if recall_mean + 1e-6 < previous_recall:
        raise RuntimeError(f"recall decreased at L={l}")
    previous_recall = recall_mean
    summary.append({
        "l": l,
        "recall_mean": recall_mean,
        "recall_stdev": statistics.stdev(recalls),
        "qps_mean": statistics.mean(qps_values),
        "qps_stdev": statistics.stdev(qps_values),
        "repetitions": len(rows),
    })
with output.open("w") as stream:
    stream.write("l\trecall_mean\trecall_stdev\tqps_mean\tqps_stdev\trepetitions\n")
    for row in summary:
        stream.write("\t".join(str(row[key]) for key in (
            "l", "recall_mean", "recall_stdev", "qps_mean", "qps_stdev", "repetitions"
        )) + "\n")
PY_SEARCH_SUMMARY
  write_status "$RUN_ROOT/status.json" searched 0 "" "search sweep validated"
}

trap on_error ERR
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

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
