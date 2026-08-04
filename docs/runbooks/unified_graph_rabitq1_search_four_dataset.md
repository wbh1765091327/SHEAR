# 统一图搜索 RaBitQ1 四数据集 Runbook

本 runbook 使用同一套图搜索实现比较 SHEAR、CAGRA NN-Descent、Tagore-CAGRA
和 Tagore-Vamana。图只提供邻接关系；导航距离分别采用 `rabitq1-post-rerank`
和 `rabitq1-adaptive-exact`，两者最终返回的结果都由原始 `f32` 向量精确距离决定。

正式流程不会构建、改写或删除任何数据集图索引。调用者必须通过 `GRAPH_MANIFEST`
提供物料护照；该 TSV 的列必须严格为：

```text
dataset	method	index_path_prefix
```

四个数据集与四种方法的 16 个组合必须各出现一次。`index_path_prefix` 是包含
`_mem.index` 的目录。运行器在搜索前验证清单唯一性、图头文件字节数、逐点记录数、
邻居 ID 范围和 SHA-256。它也会审计既有报告，但不会把未写入本次清单的旧路径当作输入。

固定搜索合同为 Recall@10、8 线程、mmap、完整 query 集合，扫描
`L=25,50,100,200,400`。每个点先进行一次完整 warmup，再进行恰好两次计量；只有两次
结果 SHA-256 和 Recall@10 都一致时才进入 accepted 汇总。失败不自动重试。

```bash
python3 script/test_unified_graph_rabitq1_search_runbook.py --extract >/tmp/unified_graph_rabitq1.sh
chmod 700 /tmp/unified_graph_rabitq1.sh
/tmp/unified_graph_rabitq1.sh help
```

<!-- BEGIN RUNBOOK_SCRIPT -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit
export LC_ALL=C

MODE="${1:-help}"
DATA_ROOT="${DATA_ROOT:-/data_nvme_kioxia/tyj/data}"
EVAL_ROOT="${EVAL_ROOT:-/data_staging/wbh/unified_graph_rabitq1}"
SIDECAR_ROOT="${SIDECAR_ROOT:-$EVAL_ROOT/sidecars}" # default: /data_staging/wbh/unified_graph_rabitq1/sidecars
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$EVAL_ROOT/runs}" # default: /data_staging/wbh/unified_graph_rabitq1/runs
SHEAR_ROOT="${SHEAR_ROOT:-/path/to/SHEAR}"
SEARCH_BIN="${SEARCH_BIN:-$SHEAR_ROOT/target/release/search_shear_index}"
SIDECAR_BUILDER="${SIDECAR_BUILDER:-$SHEAR_ROOT/target/release/build_graph_rabitq_1bit}"
GRAPH_MANIFEST="${GRAPH_MANIFEST:-}"
# GRAPH_MANIFEST header: dataset\tmethod\tindex_path_prefix
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$RUN_ROOT_BASE/$RUN_ID"
LOCK_FILE="${LOCK_FILE:-$EVAL_ROOT/.unified_graph_rabitq1.lock}"
FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)
CURRENT_STAGE=initializing
LOCK_FD=""
DATASET=""
ROWS=0
DIM=0
BASE=""
QUERY=""
GT=""
SIDECAR=""
INDEX_PREFIX=""
INDEX_FILE=""

usage() {
  cat <<'USAGE'
Usage: unified_graph_rabitq1.sh MODE

Modes:
  help            Show this contract.
  prepare         Build binaries, run focused tests, and record provenance.
  smoke           Run a deterministic synthetic two-encoding smoke.
  build-sidecars  Build Wiki35M, LAION100M, MSMARCO70M, DEEP100M sidecars.
  formal          Search all four datasets using the 16-row graph manifest.
  wiki35m         Search Wiki35M only.
  laion100m       Search LAION100M only.
  msmarco70m      Search MSMARCO70M only.
  deep100m        Search DEEP100M only.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

die() {
  log "ERROR: $*"
  return 1
}

on_error() {
  local exit_code="$?"
  trap - ERR
  if [[ -n "${RUN_ROOT:-}" ]]; then
    mkdir -p "$RUN_ROOT"
    printf 'failed\t%s\t%s\n' "$CURRENT_STAGE" "$exit_code" >"$RUN_ROOT/status"
  fi
  log "failed at stage=$CURRENT_STAGE exit_code=$exit_code"
  exit "$exit_code"
}
trap 'on_error' ERR

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "required executable is missing: $1"
}

acquire_lock() {
  mkdir -p "$EVAL_ROOT"
  exec {LOCK_FD}>"$LOCK_FILE"
  flock -n "$LOCK_FD" || die "another unified graph RaBitQ1 run holds $LOCK_FILE"
}

load_dataset() {
  DATASET="$1"
  case "$DATASET" in
    wiki35m)
      ROWS=35000000; DIM=1024
      BASE="$DATA_ROOT/wiki-1024-35M/en_base.fbin"
      QUERY="$DATA_ROOT/wiki-1024-35M/query.fbin"
      GT="$DATA_ROOT/wiki-1024-35M/gt100.bin"
      ;;
    laion100m)
      ROWS=100000000; DIM=512
      BASE="$DATA_ROOT/laion-512-100M/base.fbin"
      QUERY="$DATA_ROOT/laion-512-100M/query.fbin"
      GT="$DATA_ROOT/laion-512-100M/gt100.bin"
      ;;
    msmarco70m)
      ROWS=70000000; DIM=768
      BASE="$DATA_ROOT/msmarco-768-70M/base.fbin"
      QUERY="$DATA_ROOT/msmarco-768-114M/query.fbin"
      GT="$DATA_ROOT/msmarco-768-70M/gt100.bin"
      ;;
    deep100m)
      ROWS=100000000; DIM=96
      BASE="$DATA_ROOT/deep-96-100M/base.100M.fbin"
      QUERY="$DATA_ROOT/deep-96-100M/query.public.10K.fbin"
      GT="$DATA_ROOT/deep-96-100M/groundtruth.public.10K.ibin"
      ;;
    *) die "unknown dataset: $DATASET" ;;
  esac
  SIDECAR="$SIDECAR_ROOT/$DATASET.rabitq1"
}

validate_graph_manifest() {
  require_file "$GRAPH_MANIFEST"
  python3 - "$GRAPH_MANIFEST" <<'PY_GRAPH_MANIFEST'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
with path.open(newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    if reader.fieldnames != ["dataset", "method", "index_path_prefix"]:
        raise SystemExit("GRAPH_MANIFEST columns must be dataset, method, index_path_prefix")
    rows = list(reader)
datasets = ("wiki35m", "laion100m", "msmarco70m", "deep100m")
methods = ("shear", "cagra_nn_descent", "tagore_cagra", "tagore_vamana")
expected = {(dataset, method) for dataset in datasets for method in methods}
seen = [(row["dataset"], row["method"]) for row in rows]
if len(seen) != 16 or len(set(seen)) != 16 or set(seen) != expected:
    raise SystemExit("GRAPH_MANIFEST must contain each of 16 dataset/method rows exactly once")
for row in rows:
    graph = Path(row["index_path_prefix"]) / "_mem.index"
    if not graph.is_file():
        raise SystemExit(f"manifest graph is missing: {graph}")
PY_GRAPH_MANIFEST
}

validate_graph_file() {
  local path="$1" expected_rows="$2" report="$3"
  if [[ "${SKIP_GRAPH_BINARY_VALIDATION:-0}" == 1 ]]; then
    sha256sum "$path" >"$report"
    return
  fi
  python3 - "$path" "$expected_rows" "$report" <<'PY_GRAPH_FILE'
from pathlib import Path
import hashlib
import struct
import sys

path, rows, report = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
data = path.read_bytes()
if len(data) < 24:
    raise SystemExit(f"graph header is truncated: {path}")
declared, max_degree, start = struct.unpack_from("<QII", data, 0)
if declared != len(data) or start >= rows:
    raise SystemExit(f"invalid graph header: {path}")

def scan(explicit):
    offset, count, observed = 24, 0, 0
    seen = set()
    while count < rows and offset < len(data):
        if explicit:
            if offset + 8 > len(data): return None
            uid, degree = struct.unpack_from("<II", data, offset)
            if uid >= rows or uid in seen: return None
            seen.add(uid); degree_offset = offset + 4; offset += 8
        else:
            if offset + 4 > len(data): return None
            degree = struct.unpack_from("<I", data, offset)[0]
            degree_offset = offset; offset += 4
        if degree == 0 or offset + degree * 4 > len(data): return None
        neighbors = struct.unpack_from(f"<{degree}I", data, offset)
        if any(value >= rows for value in neighbors): return None
        offset += degree * 4; count += 1; observed = max(observed, degree)
    if offset != len(data) or count != rows or observed != max_degree: return None
    if explicit and len(seen) != rows: return None
    return "explicit-vertex-ids" if explicit else "sequential-degree-only"

encoding = scan(False) or scan(True)
if encoding is None:
    raise SystemExit(f"graph record scan failed: {path}")
report.write_text(
    f"sha256\t{hashlib.sha256(data).hexdigest()}\nbytes\t{len(data)}\nrows\t{rows}\nencoding\t{encoding}\n"
)
PY_GRAPH_FILE
}

resolve_graph() {
  local dataset="$1" method="$2"
  INDEX_PREFIX="$(awk -F '\t' -v d="$dataset" -v m="$method" 'NR>1 && $1==d && $2==m {print $3}' "$GRAPH_MANIFEST")"
  [[ -n "$INDEX_PREFIX" ]] || die "graph manifest row is absent: $dataset/$method"
  INDEX_FILE="$INDEX_PREFIX/_mem.index"
  require_file "$INDEX_FILE"
  mkdir -p "$RUN_ROOT/graph_passports/$dataset"
  validate_graph_file "$INDEX_FILE" "$ROWS" "$RUN_ROOT/graph_passports/$dataset/$method.tsv"
}

prior_result_audit() {
  mkdir -p "$RUN_ROOT"
  find "$RUN_ROOT_BASE" -mindepth 2 -maxdepth 2 -name status -type f -print 2>/dev/null \
    | sort >"$RUN_ROOT/prior_result_audit.txt" || true
}

record_provenance() {
  mkdir -p "$RUN_ROOT/provenance"
  {
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'mode\t%s\n' "$MODE"
    printf 'git_head\t'
    git -C "$SHEAR_ROOT" rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
  } >"$RUN_ROOT/provenance/context.tsv"
  for path in "$SEARCH_BIN" "$SIDECAR_BUILDER" "$GRAPH_MANIFEST"; do
    [[ -f "$path" ]] && sha256sum "$path" >>"$RUN_ROOT/provenance/sha256.txt"
  done
}

prepare_search_cache() {
  local path
  for path in "$BASE" "$INDEX_FILE" "$QUERY" "$GT"; do
    dd if="$path" of=/dev/null bs=8M status=none
  done
}

run_search_once() {
  local method="$1" backend="$2" l="$3" kind="$4" rep="$5"
  local label="$kind"
  [[ "$kind" == measured ]] && label="rep_$rep"
  local run_dir="$RUN_ROOT/search/$DATASET/$method/$backend/L$l/$label"
  mkdir -p "$run_dir"
  CURRENT_STAGE="search:$DATASET:$method:$backend:L$l:$label"
  "$SEARCH_BIN" \
    --index-path-prefix "$INDEX_PREFIX" \
    --data-path "$BASE" \
    --query-file "$QUERY" \
    --result-path "$run_dir/result.fbin" \
    --gt-file "$GT" \
    --recall-at 10 \
    --search-list-size "$l" \
    --num-threads 8 \
    --storage mmap \
    --rabitq1-sidecar "$SIDECAR" \
    --search-backend "$backend" \
    --metrics-path "$run_dir/metrics.tsv" \
    >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
  sha256sum "$run_dir/result.fbin" >"$run_dir/result.sha256"
  if [[ "$kind" == measured ]]; then
    python3 - "$RUN_ROOT/raw_search.tsv" "$DATASET" "$method" "$backend" "$l" "$rep" "$run_dir" <<'PY_RAW'
import csv
from pathlib import Path
import sys

raw, dataset, method, backend, l, rep, run_dir = sys.argv[1:]
run_dir = Path(run_dir)
with (run_dir / "metrics.tsv").open(newline="") as stream:
    metric = next(csv.DictReader(stream, delimiter="\t"))
digest = (run_dir / "result.sha256").read_text().split()[0]
fields = ("dataset", "method", "backend", "L", "rep", "recall", "qps", "result_sha256",
          "total_one_bit_evaluations", "total_lower_bound_rejects",
          "total_navigation_exact_evaluations", "total_adaptive_exact_refinements",
          "total_post_rerank_exact_evaluations", "total_exact_evaluations",
          "logical_exact_vector_bytes")
row = {"dataset": dataset, "method": method, "backend": backend, "L": l, "rep": rep,
       "recall": metric["recall"], "qps": metric["qps"], "result_sha256": digest}
for field in fields[8:]: row[field] = metric[field]
path = Path(raw)
with path.open("a", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    if path.stat().st_size == 0: writer.writeheader()
    writer.writerow(row)
PY_RAW
  fi
}

validate_repetitions() {
  local method="$1" backend="$2" l="$3"
  local root="$RUN_ROOT/search/$DATASET/$method/$backend/L$l"
  local hash_1 hash_2 recall_1 recall_2
  hash_1="$(cut -d ' ' -f1 "$root/rep_1/result.sha256")"
  hash_2="$(cut -d ' ' -f1 "$root/rep_2/result.sha256")"
  [[ "$hash_1" == "$hash_2" ]] || die "result hash mismatch: $DATASET/$method/$backend/L$l"
  recall_1="$(awk -F '\t' 'NR==2 {print $5}' "$root/rep_1/metrics.tsv")"
  recall_2="$(awk -F '\t' 'NR==2 {print $5}' "$root/rep_2/metrics.tsv")"
  [[ "$recall_1" == "$recall_2" ]] || die "Recall@10 mismatch: $DATASET/$method/$backend/L$l"
  printf '%s\t%s\t%s\t%s\t%s\n' "$DATASET" "$method" "$backend" "$l" "$hash_1" >>"$RUN_ROOT/accepted.tsv"
}

summarize_search() {
  python3 - "$RUN_ROOT/raw_search.tsv" "$RUN_ROOT/summary.tsv" <<'PY_SUMMARY'
import csv
from collections import defaultdict
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
with source.open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
groups = defaultdict(list)
for row in rows: groups[(row["dataset"], row["method"], row["backend"], row["L"])].append(row)
fields = ("dataset", "method", "backend", "L", "recall", "qps_1", "qps_2", "qps_median", "qps_min", "qps_max",
          "total_exact_evaluations", "logical_exact_vector_bytes")
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: int(row["rep"]))
        if len(group) != 2: raise SystemExit(f"expected exactly two measured rows for {key}")
        qps = [float(row["qps"]) for row in group]
        writer.writerow(dict(zip(fields[:4], key)) | {"recall": group[0]["recall"], "qps_1": qps[0], "qps_2": qps[1],
            "qps_median": (qps[0] + qps[1]) / 2, "qps_min": min(qps), "qps_max": max(qps),
            "total_exact_evaluations": group[0]["total_exact_evaluations"],
            "logical_exact_vector_bytes": group[0]["logical_exact_vector_bytes"]})
PY_SUMMARY
}

run_formal_dataset() {
  load_dataset "$1"
  require_file "$BASE"; require_file "$QUERY"; require_file "$GT"; require_file "$SIDECAR"
  for method in shear cagra_nn_descent tagore_cagra tagore_vamana; do
    resolve_graph "$DATASET" "$method"
    prepare_search_cache
    for backend in rabitq1-post-rerank rabitq1-adaptive-exact; do
      for l in 25 50 100 200 400; do
        run_search_once "$method" "$backend" "$l" warmup 0
        for rep in 1 2; do
          run_search_once "$method" "$backend" "$l" measured "$rep"
        done
        validate_repetitions "$method" "$backend" "$l"
      done
    done
  done
}

run_formal_entry() {
  local datasets=("$@")
  acquire_lock
  if [[ -f "$RUN_ROOT/status" && "$(head -n1 "$RUN_ROOT/status")" == accepted ]]; then
    log "prior accepted provenance found for RUN_ID=$RUN_ID; refusing to rerun silently"
    return 0
  fi
  mkdir -p "$RUN_ROOT"
  prior_result_audit
  validate_graph_manifest
  require_executable "$SEARCH_BIN"
  record_provenance
  : >"$RUN_ROOT/raw_search.tsv"
  local dataset
  for dataset in "${datasets[@]}"; do run_formal_dataset "$dataset"; done
  summarize_search
  printf 'accepted\n' >"$RUN_ROOT/status"
  log "accepted: $RUN_ROOT"
}

expected_sidecar_bytes() {
  local rows="$1" dim="$2" padded stride
  padded=$(( (dim + 63) / 64 * 64 ))
  stride=$(( padded / 8 + 12 ))
  printf '%s\n' $((256 + padded / 2 + 16 * padded * 4 + rows * (1 + stride)))
}

monitor_process_tree_rss() {
  local pid="$1" output="$2" peak=0 current
  while kill -0 "$pid" 2>/dev/null; do
    current="$(ps -o rss= -p "$pid" --ppid "$pid" 2>/dev/null | awk '{sum+=$1} END {print sum+0}')"
    (( current > peak )) && peak="$current"
    sleep 1
  done
  printf '%s\n' $((peak * 1024)) >"$output"
}

build_one_sidecar() {
  load_dataset "$1"
  require_file "$BASE"
  mkdir -p "$SIDECAR_ROOT" "$RUN_ROOT/sidecar_build/$DATASET"
  if [[ -e "$SIDECAR" ]]; then
    "$SIDECAR_BUILDER" --data-path "$BASE" --output "$SIDECAR" --validate-only \
      >"$RUN_ROOT/sidecar_build/$DATASET/validation.json"
    log "validated and reused existing sidecar: $SIDECAR"
    return
  fi
  local expected persistent free required
  expected="$(expected_sidecar_bytes "$ROWS" "$DIM")"
  persistent="$(find "$SIDECAR_ROOT" -maxdepth 1 -type f -name '*.rabitq1' -printf '%s\n' 2>/dev/null | awk '{sum+=$1} END {print sum+0}')"
  free="$(df -B1 --output=avail "$EVAL_ROOT" | tail -n1 | tr -d ' ')"
  required=$((persistent + 2 * expected))
  (( free > required )) || die "insufficient free space for $DATASET sidecar: free=$free required>$required"
  local run_dir="$RUN_ROOT/sidecar_build/$DATASET"
  printf '%q ' "$SIDECAR_BUILDER" --data-path "$BASE" --output "$SIDECAR" --centroids 16 --sample-rows 262144 --kmeans-iters 20 --seed 42 --threads 8 >"$run_dir/command.sh"
  printf '\n' >>"$run_dir/command.sh"
  CURRENT_STAGE="build-sidecar:$DATASET"
  /usr/bin/time -v -o "$run_dir/time.log" "$SIDECAR_BUILDER" \
    --data-path "$BASE" \
    --output "$SIDECAR" \
    --centroids 16 \
    --sample-rows 262144 \
    --kmeans-iters 20 \
    --seed 42 \
    --threads 8 \
    >"$run_dir/stdout.log" 2>"$run_dir/stderr.log" &
  local pid="$!"
  monitor_process_tree_rss "$pid" "$run_dir/process_tree_peak_rss_bytes" &
  local monitor_pid="$!"
  wait "$pid"
  wait "$monitor_pid"
  require_file "$SIDECAR"
  sha256sum "$BASE" "$SIDECAR" "$SIDECAR.json" >"$run_dir/sha256.txt"
  du -B1 "$SIDECAR" "$SIDECAR.json" >"$run_dir/disk_bytes.tsv"
  printf 'accepted\n' >"$run_dir/status"
}

run_build_sidecars() {
  acquire_lock
  mkdir -p "$RUN_ROOT"
  require_executable "$SIDECAR_BUILDER"
  prior_result_audit
  local dataset
  for dataset in wiki35m laion100m msmarco70m deep100m; do build_one_sidecar "$dataset"; done
  printf 'accepted\n' >"$RUN_ROOT/status"
}

create_smoke_fixture() {
  local root="$1"
  python3 - "$root" <<'PY_SMOKE_FIXTURE'
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1]); root.mkdir(parents=True)
rows, dim = 16, 64
vectors = []
for i in range(rows):
    row = [0.0] * dim; row[0] = float(i); row[1] = float(i * i) / 16; row[2] = 1.0; vectors.append(row)
with (root / "base.fbin").open("wb") as stream:
    stream.write(struct.pack("<II", rows, dim))
    for row in vectors: stream.write(struct.pack(f"<{dim}f", *row))
queries = [vectors[3], vectors[12]]
with (root / "query.fbin").open("wb") as stream:
    stream.write(struct.pack("<II", 2, dim))
    for row in queries: stream.write(struct.pack(f"<{dim}f", *row))
with (root / "gt.bin").open("wb") as stream:
    stream.write(struct.pack("<II", 2, 10))
    for target in (3, 12): stream.write(struct.pack("<10I", *([target] + [i for i in range(10) if i != target][:9])))
neighbors = [[j for j in range(rows) if j != i] for i in range(rows)]
for encoding in ("sequential", "explicit"):
    prefix = root / encoding; prefix.mkdir()
    payload = bytearray()
    for uid, values in enumerate(neighbors):
        if encoding == "explicit": payload += struct.pack("<I", uid)
        payload += struct.pack("<I", len(values)) + struct.pack(f"<{len(values)}I", *values)
    header = struct.pack("<QIIQ", 24 + len(payload), 15, 0, 0)
    (prefix / "_mem.index").write_bytes(header + payload)
PY_SMOKE_FIXTURE
}

run_smoke() {
  acquire_lock
  require_executable "$SEARCH_BIN"; require_executable "$SIDECAR_BUILDER"
  local root="$RUN_ROOT/smoke"
  [[ ! -e "$root" ]] || die "refusing to overwrite smoke root: $root"
  create_smoke_fixture "$root"
  "$SIDECAR_BUILDER" --data-path "$root/base.fbin" --output "$root/base.rabitq1" \
    --centroids 16 --sample-rows 16 --kmeans-iters 20 --seed 42 --batch-rows 5 --threads 2
  local encoding backend threads output metrics
  for encoding in sequential explicit; do
    for backend in rabitq1-post-rerank rabitq1-adaptive-exact; do
      for threads in 1 8; do
        output="$root/$encoding.$backend.T$threads.fbin"
        metrics="$root/$encoding.$backend.T$threads.tsv"
        "$SEARCH_BIN" --index-path-prefix "$root/$encoding" --data-path "$root/base.fbin" \
          --query-file "$root/query.fbin" --result-path "$output" --gt-file "$root/gt.bin" \
          --recall-at 2 --search-list-size 16 --num-threads "$threads" --storage mmap \
          --rabitq1-sidecar "$root/base.rabitq1" --search-backend "$backend" --metrics-path "$metrics"
      done
      cmp "$root/$encoding.$backend.T1.fbin" "$root/$encoding.$backend.T8.fbin"
    done
  done
  cmp "$root/sequential.rabitq1-post-rerank.T1.fbin" "$root/explicit.rabitq1-post-rerank.T1.fbin"
  cmp "$root/sequential.rabitq1-adaptive-exact.T1.fbin" "$root/explicit.rabitq1-adaptive-exact.T1.fbin"
  sha256sum "$root"/*.fbin >"$root/result_sha256.txt"
  printf 'accepted\n' >"$RUN_ROOT/status"
}

run_prepare() {
  acquire_lock
  mkdir -p "$RUN_ROOT"
  (cd "$SHEAR_ROOT" && cargo build --locked --release -p search_shear_index --bin build_graph_rabitq_1bit --bin search_shear_index)
  (cd "$SHEAR_ROOT" && cargo test -p search_shear_index --bin build_graph_rabitq_1bit --bin search_shear_index)
  record_provenance
  printf 'accepted\n' >"$RUN_ROOT/status"
}

case "$MODE" in
  help) usage ;;
  prepare) run_prepare ;;
  smoke) run_smoke ;;
  build-sidecars) run_build_sidecars ;;
  formal) run_formal_entry "${FORMAL_DATASETS[@]}" ;;
  wiki35m) run_formal_entry wiki35m ;;
  laion100m) run_formal_entry laion100m ;;
  msmarco70m) run_formal_entry msmarco70m ;;
  deep100m) run_formal_entry deep100m ;;
  *) usage; die "unknown mode: $MODE" ;;
esac
```
<!-- END RUNBOOK_SCRIPT -->
