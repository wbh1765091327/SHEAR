#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {wiki10m|laion100m|smoke} RUN_ROOT" >&2
  exit 2
}

dataset=${1:-}
run_root=${2:-}
[[ $# -eq 2 ]] || usage
[[ $dataset =~ ^(wiki10m|laion100m|smoke)$ ]] || usage
[[ ! -e $run_root ]] || { echo "RUN_ROOT already exists: $run_root" >&2; exit 1; }
mkdir -p "$run_root"
printf 'running\n' >"$run_root/STATUS"

accepted=0
on_exit() {
  status=$?
  if (( accepted == 0 )); then
    printf 'failed\n' >"$run_root/STATUS"
    echo "failure: preserve the failing transaction at $run_root; no automatic retry" >&2
  fi
  exit "$status"
}
trap on_exit EXIT

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
search_l_values=(25 50 100 200 400)
search_repetitions=2

run_smoke() {
  mkdir -p "$run_root"/{plain,soft/diagnostics/shadow_plain,search/{plain,shadow_plain,soft}/L25}
  printf 'plain-graph\n' >"$run_root/plain/_mem.index"
  printf 'soft-graph\n' >"$run_root/soft/_mem.index"
  printf 'shadow-graph\n' >"$run_root/soft/diagnostics/shadow_plain/_mem.index"
  printf '{"state":"plain-soft-shadow"}\n' >"$run_root/soft/diagnostics/construction_metrics.json"
  for method in plain shadow_plain soft; do
    for repeat in measured1 measured2; do
      target="$run_root/search/$method/L25/$repeat"
      mkdir -p "$target"
      printf 'result-%s\n' "$method" >"$target/results.fbin"
      printf 'recall\tqps\n1.0\t1.0\n' >"$target/metrics.tsv"
      printf 'query_id\tL\n0\t25\n' >"$target/per_query_metrics.tsv"
    done
  done
  diagnostic="$run_root/search/soft/L25/diagnostic"
  mkdir -p "$diagnostic"
  cp "$run_root/search/soft/L25/measured1/results.fbin" "$diagnostic/results.fbin"
  printf 'query_id\tL\n0\t25\n' >"$diagnostic/per_query_metrics.tsv"
  if [[ ${HASHPRUNE_SMOKE_MISMATCH:-} == diagnostic ]]; then
    printf 'mismatch\n' >>"$diagnostic/results.fbin"
  fi
  cmp "$run_root/search/soft/L25/measured1/results.fbin" "$diagnostic/results.fbin"
  sha256sum "$diagnostic/results.fbin" >"$run_root/diagnostic_result_sha256.txt"
  printf '{"dataset":"smoke","status":"accepted"}\n' >"$run_root/experiment_manifest.json"
  printf '# Smoke summary\n' >"$run_root/summary.md"
  printf 'accepted\n' >"$run_root/STATUS"
  accepted=1
}

if [[ $dataset == smoke ]]; then
  run_smoke
  trap - EXIT
  exit 0
fi

case "$run_root" in
  /data_staging/wbh/*) ;;
  *) echo "formal RUN_ROOT must be a new descendant of /data_staging/wbh" >&2; exit 1 ;;
esac

case "$dataset" in
  wiki10m)
    base=/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
    query=/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
    gt=/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
    rows=10000000
    dim=1024
    rabitq1_sidecar=/data_staging/wbh/unified_graph_rabitq1/sidecars/wiki10m.rabitq1
    ;;
  laion100m)
    base=/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin
    query=/data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin
    gt=/data_nvme_kioxia/tyj/data/laion-512-100M/gt100.bin
    rows=100000000
    dim=512
    rabitq1_sidecar=/data_staging/wbh/unified_graph_rabitq1/sidecars/laion100m.rabitq1
    ;;
esac

lock_path=/data_staging/wbh/.shear_experiment.lock
exec 9>"$lock_path"
flock -n 9 || { echo "SHEAR experiment lock is held" >&2; exit 1; }
command -v nvidia-smi >/dev/null
[[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l) -eq 0 ]] || {
  echo "GPU is not idle" >&2
  exit 1
}
sudo -n true

build_bin="$repo/target/release/build_shear_index"
search_bin="$repo/target/release/search_shear_index"
shadow_bin="$repo/target/release/build_hashprune_shadow_plain"
rabitq_builder="$repo/target/release/build_graph_rabitq_1bit"
graph_quality_bin="$repo/target/release/graph_quality"
for required in "$build_bin" "$search_bin" "$shadow_bin" "$rabitq_builder" "$graph_quality_bin" "$base" "$query" "$gt"; do
  [[ -e $required ]] || { echo "missing required input: $required" >&2; exit 1; }
done

validate_fbin_header() {
  python3 - "$1" "$2" "$3" <<'PY'
import struct, sys
path, expected_rows, expected_dim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, 'rb') as stream:
    header = stream.read(8)
if len(header) != 8 or struct.unpack('<II', header) != (expected_rows, expected_dim):
    raise SystemExit(f"header mismatch: {path}")
PY
}
validate_fbin_header "$base" "$rows" "$dim"
validate_fbin_header "$query" 10000 "$dim"
python3 - "$gt" "$rows" <<'PY'
import math, struct, sys
path, base_rows = sys.argv[1], int(sys.argv[2])
with open(path, 'rb') as stream:
    header = stream.read(8)
    if len(header) != 8:
        raise SystemExit(f"truncated GT header: {path}")
    queries, width = struct.unpack('<II', header)
    payload = stream.read()
count = queries * width
id_only_bytes = count * 4
id_distance_bytes = count * 8
if queries != 10000 or width < 10 or len(payload) not in (id_only_bytes, id_distance_bytes):
    raise SystemExit(f"GT shape or length mismatch: {path}")
ids = struct.unpack_from(f'<{count}I', payload)
if any(value >= base_rows for value in ids):
    raise SystemExit(f"GT contains an ID outside [0,{base_rows}): {path}")
if len(payload) == id_distance_bytes:
    distances = struct.iter_unpack('<f', payload[id_only_bytes:])
    if any(not math.isfinite(value[0]) for value in distances):
        raise SystemExit(f"GT contains a non-finite distance: {path}")
PY

if [[ -e $rabitq1_sidecar ]]; then
  "$rabitq_builder" --data-path "$base" --output "$rabitq1_sidecar" --validate-only
elif [[ $dataset == wiki10m ]]; then
  mkdir -p "$(dirname "$rabitq1_sidecar")"
  "$rabitq_builder" --data-path "$base" --output "$rabitq1_sidecar" --centroids 16 \
    --sample-rows 262144 --kmeans-iters 20 --seed 42 --threads 8
  "$rabitq_builder" --data-path "$base" --output "$rabitq1_sidecar" --validate-only
else
  echo "missing frozen LAION RaBitQ1 sidecar" >&2
  exit 1
fi
if [[ $dataset == laion100m ]]; then
  expected=46a7758e2008313f1d31893966db2afe10c46d12103780b4ee06b796c975fae0
  actual=$(sha256sum "$rabitq1_sidecar" | awk '{print $1}')
  [[ $actual == "$expected" ]] || { echo "LAION RaBitQ1 sidecar SHA-256 mismatch" >&2; exit 1; }
fi

sha256sum "$build_bin" "$search_bin" "$shadow_bin" "$rabitq_builder" "$graph_quality_bin" \
  "$query" "$gt" "$rabitq1_sidecar" >"$run_root/preflight_sha256.txt"
dd if="$base" bs=4096 count=1 status=none | sha256sum >"$run_root/base_header_window_sha256.txt"

drop_and_prewarm_base() {
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  dd if="$base" of=/dev/null bs=16M status=none
}

validate_graph_strict() {
  local graph=$1 output=$2
  python3 - "$graph" "$rows" "$output" <<'PY'
import json, mmap, os, struct, sys
path, rows, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, 'rb') as stream:
    data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    declared, cap, start, frozen = struct.unpack_from('<QIIQ', data, 0)
    if declared != len(data) or start >= rows or cap > 64:
        raise SystemExit(f"invalid graph header: {path}")
    offset = 24; edges = 0; observed = 0
    for source in range(rows):
        if offset + 4 > len(data): raise SystemExit(f"truncated row {source}")
        degree, = struct.unpack_from('<I', data, offset); offset += 4
        if degree > cap or offset + 4 * degree > len(data): raise SystemExit(f"invalid degree at {source}")
        neighbors = struct.unpack_from(f'<{degree}I', data, offset); offset += 4 * degree
        if source in neighbors or len(set(neighbors)) != len(neighbors) or any(v >= rows for v in neighbors):
            raise SystemExit(f"invalid adjacency row {source}")
        edges += degree; observed = max(observed, degree)
    if offset != len(data) or observed != cap: raise SystemExit(f"graph length/cap mismatch: {path}")
report = {"rows": rows, "edges": edges, "max_degree": observed, "start": start, "frozen": frozen}
temporary = output + '.partial'
with open(temporary, 'w') as stream:
    json.dump(report, stream, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
os.replace(temporary, output)
PY
}

build_case() {
  local name=$1 mode=$2
  local case_dir="$run_root/$name"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar}
  local diagnostic_arg=()
  if [[ $name == soft ]]; then
    diagnostic_arg=(--hashprune-backfill-diagnostics-prefix "$case_dir/diagnostics")
  fi
  local command=(env
    SHEAR_PIPNN_PROFILE=1 SHEAR_PIPNN_PHASE_MONITOR=1
    SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 SHEAR_PIPNN_WMMA_CTA_SHARED_B=0
    SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0 SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0
    SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0 SHEAR_PIPNN_PRUNE_BACKEND=hashprune
    SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$build_bin" --data-type float --dist-fn l2 --data-path "$base"
    --index-path-prefix "$case_dir/index" --overwrite --oom-enable --oom-memory-budget-gb 0
    --oom-io-mode buffered --oom-temp-dir "$case_dir/oom" --oom-profile-json "$case_dir/profile.json"
    --driver-timing-json "$case_dir/driver_timing.json" --max-degree 64 --lbuild 100 --num-threads 8
    --m-hash-bits 12 --leaf-size-min 256 --leaf-size-max 1280 --leaf-knn 2 --psamp-fraction 0.05
    --fanout-top 8 --fanout-second 3 --root-risk-candidate-rs 1,2,4,8 --seed 42
    --partitioner gpu-subtree --gpu-subtree-root-assignment-backend cpu
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 --gpu-subtree-execution-mode fused-persistent
    --gpu-subtree-d0-capacity-enable --gpu-subtree-d0-capacity-split-recursion-limit 8
    --gpu-subtree-d0-capacity-high-ratio 0.08 --gpu-subtree-d0-capacity-target-ratio 0.08
    --gpu-subtree-d0-capacity-low-ratio 0.08 --gpu-subtree-d0-partition-mode gpu-kmeans-coarse
    --gpu-subtree-sq-staging-mode gpu-d0 --gpu-subtree-sq-staging-upload-shard-rows 262144
    --gpu-subtree-sq-wmma-mode build --gpu-subtree-sq-wmma-sidecar-prefix "$case_dir/sq_sidecar/sq_wmma"
    --gpu-subtree-sq-wmma-sidecar-mode rebuild --gpu-subtree-persistent-large-ticket-batch auto
    --gpu-subtree-sq8-assignment-kernel auto --gpu-subtree-sq8-leaf-mode auto
    --gpu-subtree-persistent-scheduler-mode auto --hashprune-evidence-mode off
    --hashprune-evidence-overflow-budget 0 --hashprune-backfill-mode "$mode"
    --hashprune-backfill-ablation none --hashprune-backfill-budget 0 --hashprune-backfill-alpha 1.0
    --pipnn-backend cuda --pipnn-gpu-device 0 --pipnn-gpu-workspace-budget-gb 16 "${diagnostic_arg[@]}")
  printf '%q ' "${command[@]}" >"$case_dir/command.sh"
  printf '\n' >>"$case_dir/command.sh"
  drop_and_prewarm_base
  nvidia-smi dmon -s pucvmet -d 1 >"$case_dir/gpu_dmon.log" 2>&1 &
  local monitor_pid=$!
  set +e
  /usr/bin/time -v -o "$case_dir/time.log" "${command[@]}" >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
  local status=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  printf '%s\n' "$status" >"$case_dir/status.txt"
  (( status == 0 )) || return "$status"
  validate_graph_strict "$case_dir/index/_mem.index" "$case_dir/graph_validation.json"
  "$graph_quality_bin" --index-path-prefix "$case_dir/index" --data-path "$base" \
    --output "$case_dir/graph_quality.tsv" --max-edge-samples 1000000 >"$case_dir/graph_quality.log" 2>&1
}

build_case plain off
build_case soft rhalf-soft-tier-q3
python3 - "$run_root/soft/profile.json" <<'PY'
import json, sys
profile = json.load(open(sys.argv[1]))
text = json.dumps(profile)
if 'soft-tier-q3' not in text or '1.1' not in text:
    raise SystemExit('SoftTier profile does not audit priority_rule=soft-tier-q3 and priority_alpha=1.1')
PY

mkdir -p "$run_root/soft/diagnostics/shadow_plain"
"$shadow_bin" \
  --source-graph "$run_root/soft/index/_mem.index" \
  --primary-count-sidecar "$run_root/soft/diagnostics/primary_count.sidecar" \
  --output-graph "$run_root/soft/diagnostics/shadow_plain/_mem.index" \
  --validation-json "$run_root/soft/diagnostics/shadow_plain_validation.json"
validate_graph_strict "$run_root/soft/diagnostics/shadow_plain/_mem.index" \
  "$run_root/soft/diagnostics/shadow_plain_graph_validation.json"
"$graph_quality_bin" --index-path-prefix "$run_root/soft/diagnostics/shadow_plain" --data-path "$base" \
  --output "$run_root/soft/diagnostics/shadow_plain_graph_quality.tsv" --max-edge-samples 1000000

run_search() {
  local method=$1 graph_dir=$2 l_value=$3 output_dir=$4
  mkdir -p "$output_dir"
  "$search_bin" --index-path-prefix "$graph_dir" --data-path "$base" --query-file "$query" --gt-file "$gt" \
    --result-path "$output_dir/results.fbin" --metrics-path "$output_dir/metrics.tsv" \
    --query-metrics-path "$output_dir/per_query_metrics.tsv" --recall-at 10 --search-list-size "$l_value" \
    --num-threads 8 --storage mmap --search-backend rabitq1-adaptive-exact --rabitq1-sidecar "$rabitq1_sidecar"
}

for method in plain shadow_plain soft; do
  case "$method" in
    plain) graph_dir="$run_root/plain/index" ;;
    shadow_plain) graph_dir="$run_root/soft/diagnostics/shadow_plain" ;;
    soft) graph_dir="$run_root/soft/index" ;;
  esac
  for l_value in "${search_l_values[@]}"; do
    search_root="$run_root/search/$method/L$l_value"
    mkdir -p "$search_root"
    for warm_file in "$base" "$graph_dir/_mem.index" "$rabitq1_sidecar" "$query" "$gt"; do
      dd if="$warm_file" of=/dev/null bs=16M status=none
    done
    run_search "$method" "$graph_dir" "$l_value" "$search_root/warmup" # one warm-up
    for repeat in $(seq 1 "$search_repetitions"); do
      run_search "$method" "$graph_dir" "$l_value" "$search_root/measured$repeat"
    done
    cmp "$search_root/measured1/results.fbin" "$search_root/measured2/results.fbin"
  done
done

for l_value in "${search_l_values[@]}"; do
  diagnostic="$run_root/search/soft/L$l_value/diagnostic"
  mkdir -p "$diagnostic"
  "$search_bin" --index-path-prefix "$run_root/soft/index" --data-path "$base" --query-file "$query" --gt-file "$gt" \
    --result-path "$diagnostic/results.fbin" --metrics-path "$diagnostic/metrics.tsv" \
    --query-metrics-path "$diagnostic/per_query_metrics.tsv" --recall-at 10 --search-list-size "$l_value" \
    --num-threads 8 --storage mmap --search-backend rabitq1-adaptive-exact --rabitq1-sidecar "$rabitq1_sidecar" \
    --hashprune-primary-count-sidecar "$run_root/soft/diagnostics/primary_count.sidecar"
  cmp "$diagnostic/results.fbin" "$run_root/search/soft/L$l_value/measured1/results.fbin" || {
    echo "diagnostic result SHA-256 differs from standard Soft" >&2
    exit 1
  }
done

python3 "$repo/script/summarize_hashprune_diagnostic_quality.py" --experiment-root "$run_root" --gt-file "$gt" \
  --construction-metrics "$run_root/soft/diagnostics/construction_metrics.json" \
  --output-dir "$run_root/report" --l-values 25,50,100,200,400 --k 10

python3 - "$run_root" "$rows" <<'PY'
import json, sys
root, rows = sys.argv[1], int(sys.argv[2])
plain = json.load(open(f'{root}/plain/graph_validation.json'))
soft = json.load(open(f'{root}/soft/graph_validation.json'))
shadow = json.load(open(f'{root}/soft/diagnostics/shadow_plain_graph_validation.json'))
construction = json.load(open(f'{root}/soft/diagnostics/construction_metrics.json'))
if any(item['rows'] != rows or item['max_degree'] > 64 for item in (plain, soft, shadow)):
    raise SystemExit('graph row or maximum-degree conservation failed')
if soft['edges'] - shadow['edges'] != construction['backfill_edges']:
    raise SystemExit('Soft-minus-shadow edge conservation failed')
if shadow['edges'] != construction['primary_edges']:
    raise SystemExit('shadow primary-edge conservation failed')
PY

sha256sum "$run_root/plain/index/_mem.index" "$run_root/soft/index/_mem.index" \
  "$run_root/soft/diagnostics/primary_count.sidecar" \
  "$run_root/soft/diagnostics/shadow_plain/_mem.index" >"$run_root/graph_sha256.txt"
python3 - "$run_root" "$dataset" "$rows" "$dim" <<'PY'
import json, os, sys
root, dataset, rows, dim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
manifest = {"dataset": dataset, "rows": rows, "dim": dim, "status": "accepted"}
for name in ("soft/diagnostics/manifest.json", "experiment_manifest.json"):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w") as stream:
        json.dump(manifest, stream, sort_keys=True)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
PY

printf 'accepted_pending_cleanup\n' >"$run_root/STATUS"
python3 - "$run_root" <<'PY'
import hashlib, json, os, shutil, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
relative_targets = [
    'plain/index', 'plain/oom', 'plain/sq_sidecar',
    'soft/index', 'soft/oom', 'soft/sq_sidecar',
    'soft/diagnostics/shadow_plain/_mem.index',
]
records = []
for relative in relative_targets:
    target = (root / relative).resolve()
    if root not in target.parents:
        raise SystemExit(f'cleanup target escapes run root: {target}')
    files = [target] if target.is_file() else sorted(path for path in target.rglob('*') if path.is_file())
    size = 0; digest = hashlib.sha256()
    for path in files:
        payload = path.read_bytes(); size += len(payload)
        digest.update(str(path.relative_to(root)).encode()); digest.update(b'\0'); digest.update(payload)
    records.append({'path': relative, 'bytes': size, 'sha256': digest.hexdigest(), 'removed': False})
manifest = root / 'cleanup_manifest.json'; temporary = manifest.with_suffix('.json.partial')
temporary.write_text(json.dumps({'targets': records}, indent=2, sort_keys=True) + '\n')
with temporary.open('rb') as stream: os.fsync(stream.fileno())
os.replace(temporary, manifest)
for record in records:
    target = root / record['path']
    if target.is_dir(): shutil.rmtree(target)
    elif target.exists(): target.unlink()
    record['removed'] = not target.exists()
temporary.write_text(json.dumps({'targets': records}, indent=2, sort_keys=True) + '\n')
with temporary.open('rb') as stream: os.fsync(stream.fileno())
os.replace(temporary, manifest)
if not all(record['removed'] for record in records): raise SystemExit('cleanup verification failed')
PY
printf 'accepted\n' >"$run_root/STATUS"
sync
accepted=1
trap - EXIT
