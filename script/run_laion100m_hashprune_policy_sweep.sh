#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-help}
OUTPUT_BASE=${OUTPUT_BASE:-/data_staging/wbh/hashprune_policy_sweep/laion100m}
RUN_ID=${RUN_ID:-}
EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE:-legacy-15}
BASE=/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin
QUERY=/data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin
GT=/data_nvme_kioxia/tyj/data/laion-512-100M/gt100.bin
RABITQ1_SIDECAR=/data_staging/wbh/unified_graph_rabitq1/sidecars/laion100m.rabitq1
EXPECTED_SIDECAR_SHA256=46a7758e2008313f1d31893966db2afe10c46d12103780b4ee06b796c975fae0
ROWS=100000000
DIM=512
PASS1_L=(25 50 100 200 400)
PASS2_L=(400 200 100 50 25)

# BEGIN FROZEN MATRIX
MATRIX=(
  '001|hashslot_q1|rhalf-hash-slot-1|1|1.00'
  '002|distance_first_q2|rhalf-distance-first-q3|2|1.00'
  '003|distance_first_q3|rhalf-distance-first-q3|3|1.00'
  '004|hard_tier_q2|rhalf-hard-tier-q3|2|strict'
  '005|hard_tier_q3|rhalf-hard-tier-q3|3|strict'
  '006|soft_tier_q2_a1p05|rhalf-soft-tier-q3|2|1.05'
  '007|soft_tier_q2_a1p1|rhalf-soft-tier-q3|2|1.10'
  '008|soft_tier_q2_a1p2|rhalf-soft-tier-q3|2|1.20'
  '009|soft_tier_q2_a1p4|rhalf-soft-tier-q3|2|1.40'
  '010|soft_tier_q2_a2p0|rhalf-soft-tier-q3|2|2.00'
  '011|soft_tier_q3_a1p05|rhalf-soft-tier-q3|3|1.05'
  '012|soft_tier_q3_a1p1|rhalf-soft-tier-q3|3|1.10'
  '013|soft_tier_q3_a1p2|rhalf-soft-tier-q3|3|1.20'
  '014|soft_tier_q3_a1p4|rhalf-soft-tier-q3|3|1.40'
  '015|soft_tier_q3_a2p0|rhalf-soft-tier-q3|3|2.00'
)
# END FROZEN MATRIX

if [[ $EXPERIMENT_PROFILE == four-policy-cap3 ]]; then
  PASS1_L=(10 16 25 32 50 64 80 100 128 160 200 300 400)
  PASS2_L=()
  MATRIX=(
    '001|plain|off|1|NA'
    '002|distance_first_q3_cap3|rhalf-distance-first-q3|3|NA'
    '003|hard_tier_q3_cap3|rhalf-hard-tier-q3|3|NA'
    '004|soft_tier_q3_cap3_a1p02|rhalf-soft-tier-q3|3|1.02'
  )
elif [[ $EXPERIMENT_PROFILE != legacy-15 ]]; then
  echo "error: unknown EXPERIMENT_PROFILE: $EXPERIMENT_PROFILE" >&2
  exit 1
fi

BUILD_BIN="$REPO/target/release/build_shear_index"
SEARCH_BIN="$REPO/target/release/search_shear_index"
RABITQ_BUILDER="$REPO/target/release/build_graph_rabitq_1bit"
GRAPH_QUALITY_BIN="$REPO/target/release/graph_quality"
GPU_MONITOR="$REPO/script/monitor_gpu_build.sh"
SUMMARIZER="$REPO/script/summarize_laion100m_hashprune_policy_sweep.py"
FOUR_POLICY_SUMMARIZER="$REPO/script/summarize_laion100m_hashprune_four_policy_cap3.py"
CARGO_BIN=${CARGO_BIN:-cargo}

usage() {
  cat <<'USAGE'
Usage: bash script/run_laion100m_hashprune_policy_sweep.sh MODE

Modes:
  help       Show this contract.
  smoke      Execute 15 synthetic transactions under OUTPUT_BASE/RUN_ID.
  prepare    Build the required release binaries.
  preflight  Validate inputs, binaries, locks, cache control, GPU, and free space.
  formal     Execute the frozen sequential LAION100M sweep; RUN_ID is required.
  resume-four-after-plain
             Resume a stopped four-policy run after accepting Plain's pass1.
  resume-four-after-distance
             Resume a stopped four-policy run after validating DistanceFirst.
  summarize Regenerate the report for BATCH_ROOT or OUTPUT_BASE/RUN_ID.

Formal execution never retries or skips a failed transaction. Failed payloads
are preserved. Accepted transaction-local index, OOM, and SQ payloads are
removed only after validation, evidence fsync, and metric capture.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

atomic_status() {
  local path=$1 value=$2 temporary
  temporary="${path}.partial.$$"
  printf '%s\n' "$value" >"$temporary"
  mv -f -- "$temporary" "$path"
}

require_new_batch_root() {
  [[ -n $RUN_ID ]] || die "RUN_ID is required"
  [[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: $RUN_ID"
  BATCH_ROOT="$OUTPUT_BASE/$RUN_ID"
  [[ ! -e $BATCH_ROOT ]] || die "run root already exists: $BATCH_ROOT"
  mkdir -p "$BATCH_ROOT/transactions" "$BATCH_ROOT/report" "$BATCH_ROOT/provenance"
  atomic_status "$BATCH_ROOT/STATUS" running
}

write_matrix() {
  local destination=$1 row ordinal label policy quota alpha
  printf 'ordinal\tlabel\tpolicy_mode\thash_quota\tpriority_alpha\n' >"$destination"
  for row in "${MATRIX[@]}"; do
    IFS='|' read -r ordinal label policy quota alpha <<<"$row"
    printf '%s\t%s\t%s\t%s\t%s\n' "$ordinal" "$label" "$policy" "$quota" "$alpha" >>"$destination"
  done
}

record_smoke_call() {
  local case_dir=$1 kind=$2 pass=$3 l_value=$4
  printf '%s\t%s\t%s\n' "$kind" "$pass" "$l_value" >>"$case_dir/search_calls.tsv"
}

run_smoke_case() {
  local row=$1 ordinal label policy quota alpha case_dir l_value
  IFS='|' read -r ordinal label policy quota alpha <<<"$row"
  case_dir="$BATCH_ROOT/transactions/${ordinal}_${label}"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar,search}
  atomic_status "$case_dir/STATUS" running
  printf '1\n' >"$case_dir/attempt_count"
  printf 'synthetic graph for %s\n' "$label" >"$case_dir/index/_mem.index"
  printf '{"label":"%s","mode":"%s","requested_hash_quota":%s,"priority_alpha":"%s"}\n' \
    "$label" "$policy" "$quota" "$alpha" >"$case_dir/profile.json"
  printf 'kind\tpass\tL\n' >"$case_dir/search_calls.tsv"
  record_smoke_call "$case_dir" warmup warmup 400
  for l_value in "${PASS1_L[@]}"; do
    record_smoke_call "$case_dir" measured pass1 "$l_value"
  done
  for l_value in "${PASS2_L[@]}"; do
    record_smoke_call "$case_dir" measured pass2 "$l_value"
  done
  if [[ ${SWEEP_SMOKE_MISMATCH_LABEL:-} == "$label" ]]; then
    printf 'paired result mismatch\n' >"$case_dir/failure.txt"
    atomic_status "$case_dir/STATUS" failed
    return 1
  fi
  printf '{"label":"%s","graph_bytes":%s,"avg_out_degree":44.0}\n' \
    "$label" "$(stat -Lc %s -- "$case_dir/index/_mem.index")" >"$case_dir/accepted_metrics.json"
  atomic_status "$case_dir/STATUS" accepted
  rm -rf -- "$case_dir/index" "$case_dir/oom" "$case_dir/sq_sidecar"
}

run_smoke() {
  local row
  require_new_batch_root
  write_matrix "$BATCH_ROOT/frozen_matrix.tsv"
  for row in "${MATRIX[@]}"; do
    if ! run_smoke_case "$row"; then
      atomic_status "$BATCH_ROOT/STATUS" failed
      return 1
    fi
  done
  printf '{"schema":"laion100m_hashprune_policy_sweep_smoke_v1","accepted":%s,"profile":"%s"}\n' \
    "${#MATRIX[@]}" "$EXPERIMENT_PROFILE" >"$BATCH_ROOT/experiment_manifest.json"
  atomic_status "$BATCH_ROOT/STATUS" accepted
}

prepare_binaries() {
  [[ -x $CARGO_BIN ]] || die "missing cargo: $CARGO_BIN"
  PATH="$(dirname "$CARGO_BIN"):$PATH" "$CARGO_BIN" build --locked --release \
    -p build_shear_index --features build_shear_index/cuda
  PATH="$(dirname "$CARGO_BIN"):$PATH" "$CARGO_BIN" build --locked --release \
    -p search_shear_index --bins
}

acquire_locks() {
  mkdir -p "$OUTPUT_BASE"
  exec 8>"$OUTPUT_BASE/.batch.lock"
  flock -n 8 || die "HashPrune policy sweep lock is held"
  exec 9>/data_staging/wbh/.shear_experiment.lock
  flock -n 9 || die "global SHEAR experiment lock is held"
}

validate_inputs() {
  local required actual free_bytes minimum_free=$((300 * 1024 * 1024 * 1024))
  for required in "$BUILD_BIN" "$SEARCH_BIN" "$RABITQ_BUILDER" "$GRAPH_QUALITY_BIN" \
    "$GPU_MONITOR" "$SUMMARIZER" "$FOUR_POLICY_SUMMARIZER" "$BASE" "$QUERY" "$GT" "$RABITQ1_SIDECAR"; do
    [[ -e $required ]] || die "missing required input: $required"
  done
  [[ -x $BUILD_BIN && -x $SEARCH_BIN && -x $RABITQ_BUILDER && -x $GRAPH_QUALITY_BIN ]] || \
    die "one or more release binaries are not executable"
  if strings "$BUILD_BIN" | grep -Fq 'SHEAR was built without CUDA support'; then
    die "build_shear_index lacks the required CUDA feature; run prepare"
  fi
  python3 - "$BASE" "$QUERY" "$GT" "$ROWS" "$DIM" <<'PY_INPUTS'
import math
import struct
import sys

base, query, gt, rows, dim = sys.argv[1:]
rows, dim = int(rows), int(dim)
for path, expected in ((base, (rows, dim)), (query, (10000, dim))):
    with open(path, "rb") as stream:
        header = stream.read(8)
    if len(header) != 8 or struct.unpack("<II", header) != expected:
        raise SystemExit(f"fbin header mismatch: {path}")
with open(gt, "rb") as stream:
    header = stream.read(8)
    payload = stream.read()
if len(header) != 8:
    raise SystemExit("truncated ground truth header")
queries, width = struct.unpack("<II", header)
count = queries * width
if queries != 10000 or width < 10 or len(payload) not in (count * 4, count * 8):
    raise SystemExit("ground truth shape or length mismatch")
ids = struct.unpack_from(f"<{count}I", payload)
if any(point >= rows for point in ids):
    raise SystemExit("ground truth contains an out-of-range point ID")
if len(payload) == count * 8:
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload[count * 4:])):
        raise SystemExit("ground truth contains a non-finite distance")
PY_INPUTS
  "$RABITQ_BUILDER" --data-path "$BASE" --output "$RABITQ1_SIDECAR" --validate-only >/dev/null
  actual=$(sha256sum "$RABITQ1_SIDECAR" | awk '{print $1}')
  [[ $actual == "$EXPECTED_SIDECAR_SHA256" ]] || die "RaBitQ1 sidecar SHA-256 mismatch: $actual"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l) -eq 0 ]] || \
    die "GPU 0 is not idle"
  sudo -n true || die "passwordless sudo is required for controlled page-cache drops"
  free_bytes=$(df -PB1 "$OUTPUT_BASE" | awk 'NR==2 {print $4}')
  (( free_bytes >= minimum_free )) || die "less than 300 GiB free below OUTPUT_BASE"
}

run_preflight() {
  acquire_locks
  validate_inputs
  printf 'preflight accepted: inputs, sidecar, binaries, GPU, cache control, locks, and free space\n'
}

drop_cache() {
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

prewarm_file() {
  local path=$1 manifest=$2 elapsed_file
  elapsed_file="${manifest}.elapsed.$$"
  /usr/bin/time -f '%e' -o "$elapsed_file" dd if="$path" of=/dev/null bs=16M status=none
  printf '%s\t%s\t%s\n' "$path" "$(stat -Lc %s -- "$path")" "$(<"$elapsed_file")" >>"$manifest"
  rm -f -- "$elapsed_file"
}

validate_graph_strict() {
  local graph=$1 output=$2
  python3 - "$graph" "$ROWS" "$output" <<'PY_GRAPH'
import json
import mmap
import os
import struct
import sys

path, rows, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, "rb") as stream:
    data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    if len(data) < 24:
        raise SystemExit("truncated graph header")
    declared, cap, start, frozen = struct.unpack_from("<QIIQ", data, 0)
    if declared != len(data) or cap > 64 or start >= rows:
        raise SystemExit("invalid graph header")
    offset, edges, observed = 24, 0, 0
    for source in range(rows):
        if offset + 4 > len(data):
            raise SystemExit(f"truncated graph row {source}")
        degree, = struct.unpack_from("<I", data, offset)
        offset += 4
        if degree > cap or offset + degree * 4 > len(data):
            raise SystemExit(f"invalid degree at row {source}")
        neighbors = struct.unpack_from(f"<{degree}I", data, offset)
        offset += degree * 4
        if source in neighbors or len(neighbors) != len(set(neighbors)) or any(v >= rows for v in neighbors):
            raise SystemExit(f"invalid neighbors at row {source}")
        edges += degree
        observed = max(observed, degree)
    if offset != len(data) or observed != cap:
        raise SystemExit("graph byte consumption or degree cap mismatch")
report = {
    "rows": rows,
    "edges": edges,
    "max_degree": observed,
    "avg_out_degree": edges / rows,
    "start": start,
    "frozen": frozen,
    "graph_bytes": os.path.getsize(path),
}
temporary = output + ".partial"
with open(temporary, "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY_GRAPH
}

validate_profile() {
  local profile=$1 policy=$2 quota=$3 alpha=$4 output=$5
  python3 - "$profile" "$policy" "$quota" "$alpha" "$EXPERIMENT_PROFILE" "$output" <<'PY_PROFILE'
import json
import math
import os
import sys

profile_path, policy, quota, alpha, experiment_profile, output = sys.argv[1:]
quota = int(quota)
profile = json.load(open(profile_path))
backfill = profile.get("hashprune_backfill")
if not isinstance(backfill, dict):
    raise SystemExit("missing hashprune_backfill profile")
expected_alpha = float(alpha) if policy == "rhalf-soft-tier-q3" else 1.0
checks = {
    "mode": backfill.get("mode") == policy,
    "requested_hash_quota": backfill.get("requested_hash_quota") == quota,
    "effective_hash_quota": backfill.get("effective_hash_quota") == quota,
    "derived_budget": backfill.get("derived_budget") == 32,
    "uses_alpha_gate": backfill.get("uses_alpha_gate") is False,
    "effective_priority_alpha": math.isclose(float(backfill.get("effective_priority_alpha")), expected_alpha),
}
if experiment_profile == "four-policy-cap3":
    expected_cap = None if policy == "off" else 3
    checks["requested_max_backfill_per_row"] = backfill.get("requested_max_backfill_per_row") == expected_cap
    checks["effective_max_backfill_per_row"] = backfill.get("effective_max_backfill_per_row") == (expected_cap or 0)
    checks["requested_hash_quota"] = backfill.get("requested_hash_quota") is None
    if policy == "off":
        checks["derived_budget"] = backfill.get("derived_budget") == 0
        checks["effective_hash_quota"] = backfill.get("effective_hash_quota") == 1
if policy == "rhalf-soft-tier-q3":
    checks["requested_priority_alpha"] = math.isclose(float(backfill.get("requested_priority_alpha")), expected_alpha)
else:
    checks["requested_priority_alpha"] = backfill.get("requested_priority_alpha") is None
if quota < 3:
    checks["no_layer_3"] = int(backfill.get("emitted_layer_3", 0)) == 0
if not all(checks.values()):
    raise SystemExit(f"HashPrune profile mismatch: {checks}")
value = {"checks": checks, "accepted": True, "backfill": backfill}
temporary = output + ".partial"
with open(temporary, "w") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY_PROFILE
}

capture_resource_summary() {
  local case_dir=$1
  python3 - "$case_dir/time.log" "$case_dir/gpu_monitor/summary.txt" \
    "$case_dir/resource_summary.json" <<'PY_RESOURCE'
import json
from pathlib import Path
import re
import sys

time_path, gpu_path, output = map(Path, sys.argv[1:])
time_text = time_path.read_text()
gpu = {}
for line in gpu_path.read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        gpu[key] = value
rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_text)
elapsed = re.search(r"Elapsed \(wall clock\) time.*: (.+)", time_text)
value = {
    "peak_process_rss_bytes": int(rss.group(1)) * 1024 if rss else None,
    "peak_hbm_mib": float(gpu["memory_used_max_mib"]) if "memory_used_max_mib" in gpu else None,
    "monitor_elapsed_seconds": float(gpu["elapsed_seconds"]) if "elapsed_seconds" in gpu else None,
    "time_elapsed_display": elapsed.group(1) if elapsed else None,
}
temporary = output.with_suffix(output.suffix + ".partial")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY_RESOURCE
}

run_build() {
  local case_dir=$1 policy=$2 quota=$3 alpha=$4
  local alpha_args=() cap_args=() quota_args=()
  if [[ $policy == rhalf-soft-tier-q3 ]]; then
    alpha_args=(--hashprune-soft-tier-priority-alpha "$alpha")
  fi
  if [[ $EXPERIMENT_PROFILE == four-policy-cap3 && $policy != off ]]; then
    cap_args=(--hashprune-rhalf-max-backfill-per-row 3)
  fi
  if [[ $EXPERIMENT_PROFILE != four-policy-cap3 ]]; then
    quota_args=(--hashprune-rhalf-hash-quota "$quota")
  fi
  local command=(env
    SHEAR_PIPNN_PROFILE=1 SHEAR_PIPNN_PHASE_MONITOR=1
    SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 SHEAR_PIPNN_WMMA_CTA_SHARED_B=0
    SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0 SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0
    SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0 SHEAR_PIPNN_PRUNE_BACKEND=hashprune
    SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$BUILD_BIN" --data-type float --dist-fn l2 --data-path "$BASE"
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
    --hashprune-evidence-overflow-budget 0 --hashprune-backfill-mode "$policy"
    --hashprune-backfill-ablation none --hashprune-backfill-budget 0 --hashprune-backfill-alpha 1.0
    "${quota_args[@]}" "${cap_args[@]}" "${alpha_args[@]}"
    --pipnn-backend cuda --pipnn-gpu-device 0 --pipnn-gpu-workspace-budget-gb 16)
  printf '%q ' "${command[@]}" >"$case_dir/build_command.sh"
  printf '\n' >>"$case_dir/build_command.sh"
  drop_cache
  printf 'path\tbytes\telapsed_seconds\n' >"$case_dir/build_prewarm.tsv"
  prewarm_file "$BASE" "$case_dir/build_prewarm.tsv"
  set +e
  "$GPU_MONITOR" --out-dir "$case_dir/gpu_monitor" --gpu 0 --interval 1 -- \
    /usr/bin/time -v -o "$case_dir/time.log" "${command[@]}"
  local status=$?
  set -e
  printf '%s\n' "$status" >"$case_dir/build_exit_code"
  (( status == 0 )) || return "$status"
  capture_resource_summary "$case_dir"
}

run_search_once() {
  local case_dir=$1 l_value=$2 label=$3 kind=$4
  local run_dir="$case_dir/search/$label/L$l_value"
  mkdir -p "$run_dir"
  "$SEARCH_BIN" --index-path-prefix "$case_dir/index" --data-path "$BASE" \
    --query-file "$QUERY" --gt-file "$GT" --result-path "$run_dir/results.fbin" \
    --metrics-path "$run_dir/metrics.tsv" --query-metrics-path "$run_dir/per_query_metrics.tsv" \
    --recall-at 10 --search-list-size "$l_value" --num-threads 8 --storage mmap \
    --search-backend rabitq1-adaptive-exact --rabitq1-sidecar "$RABITQ1_SIDECAR" \
    >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
  sha256sum "$run_dir/results.fbin" >"$run_dir/results.sha256"
  printf '%s\t%s\t%s\t%s\n' "$kind" "$label" "$l_value" \
    "$(awk 'NR==1 {print $1}' "$run_dir/results.sha256")" >>"$case_dir/search_calls.tsv"
}

run_search_sweep() {
  local case_dir=$1 l_value path
  drop_cache
  printf 'path\tbytes\telapsed_seconds\n' >"$case_dir/search_prewarm.tsv"
  for path in "$BASE" "$case_dir/index/_mem.index" "$RABITQ1_SIDECAR" "$QUERY" "$GT"; do
    prewarm_file "$path" "$case_dir/search_prewarm.tsv"
  done
  printf 'kind\tpass\tL\tresult_sha256\n' >"$case_dir/search_calls.tsv"
  run_search_once "$case_dir" 400 warmup warmup

  for l_value in "${PASS1_L[@]}"; do
    run_search_once "$case_dir" "$l_value" pass1 measured
  done
  for l_value in "${PASS2_L[@]}"; do
    run_search_once "$case_dir" "$l_value" pass2 measured
  done
  local expected_l_csv
  expected_l_csv=$(IFS=,; echo "${PASS1_L[*]}")
  python3 - "$case_dir" "$expected_l_csv" "$EXPERIMENT_PROFILE" <<'PY_SEARCH'
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

case = Path(sys.argv[1])
expected_l = tuple(int(value) for value in sys.argv[2].split(","))
experiment_profile = sys.argv[3]
pass_names = ("pass1",) if experiment_profile == "four-policy-cap3" else ("pass1", "pass2")
fields = None
rows = []
for l_value in expected_l:
    pair = []
    for pass_name in pass_names:
        root = case / "search" / pass_name / f"L{l_value}"
        with (root / "metrics.tsv").open(newline="") as stream:
            metric = next(csv.DictReader(stream, delimiter="\t"))
        digest = hashlib.sha256((root / "results.fbin").read_bytes()).hexdigest()
        pair.append((metric, digest))
        row = {"pass": pass_name, "L": l_value, "result_sha256": digest, **metric}
        rows.append(row)
        fields = list(row)
    if len(pair) == 2 and (pair[0][1] != pair[1][1] or pair[0][0]["recall"] != pair[1][0]["recall"]):
        raise SystemExit(f"paired result or recall mismatch at L={l_value}")
output = case / "search_measurements.tsv"
temporary = output.with_suffix(output.suffix + ".partial")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(output)
PY_SEARCH
}

fsync_evidence() {
  local case_dir=$1
  python3 - "$case_dir" <<'PY_FSYNC'
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for path in root.rglob("*"):
    if path.is_file():
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
descriptor = os.open(root, os.O_RDONLY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY_FSYNC
}

cleanup_accepted_payload() {
  local case_dir=$1 resolved root_resolved target
  resolved=$(realpath -e "$case_dir")
  root_resolved=$(realpath -e "$BATCH_ROOT/transactions")
  [[ $resolved == "$root_resolved"/* ]] || die "cleanup target escapes transaction root: $resolved"
  [[ $(<"$case_dir/STATUS") == accepted ]] || die "refusing cleanup for non-accepted transaction"
  for target in "$case_dir/index" "$case_dir/oom" "$case_dir/sq_sidecar"; do
    [[ $target == "$case_dir"/* ]] || die "unsafe cleanup target: $target"
  done
  rm -rf -- "$case_dir/index" "$case_dir/oom" "$case_dir/sq_sidecar"
  printf 'removed transaction-local index, oom, and sq_sidecar after acceptance\n' \
    >"$case_dir/cleanup_manifest.txt"
  python3 - "$case_dir/cleanup_manifest.txt" "$case_dir" <<'PY_CLEANUP_FSYNC'
import os
import sys
for path in sys.argv[1:]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY_CLEANUP_FSYNC
}

run_formal_case() {
  local row=$1 ordinal label policy quota alpha case_dir graph
  IFS='|' read -r ordinal label policy quota alpha <<<"$row"
  case_dir="$BATCH_ROOT/transactions/${ordinal}_${label}"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar,search}
  atomic_status "$case_dir/STATUS" running
  printf '1\n' >"$case_dir/attempt_count"
  printf '{"ordinal":%s,"label":"%s","policy_mode":"%s","requested_hash_quota":%s,"priority_alpha":"%s"}\n' \
    "$((10#$ordinal))" "$label" "$policy" "$quota" "$alpha" >"$case_dir/parameter_passport.json"
  run_build "$case_dir" "$policy" "$quota" "$alpha"
  graph="$case_dir/index/_mem.index"
  validate_graph_strict "$graph" "$case_dir/graph_validation.json"
  sha256sum "$graph" >"$case_dir/graph.sha256"
  "$GRAPH_QUALITY_BIN" --index-path-prefix "$case_dir/index" --data-path "$BASE" \
    --output "$case_dir/graph_quality.tsv" --max-edge-samples 1000000 \
    >"$case_dir/graph_quality.stdout.log" 2>"$case_dir/graph_quality.stderr.log"
  validate_profile "$case_dir/profile.json" "$policy" "$quota" "$alpha" \
    "$case_dir/policy_validation.json"
  run_search_sweep "$case_dir"
  python3 - "$case_dir" <<'PY_ACCEPT'
import csv
import json
from pathlib import Path
import sys

case = Path(sys.argv[1])
graph = json.load(open(case / "graph_validation.json"))
quality = next(csv.DictReader((case / "graph_quality.tsv").open(), delimiter="\t"))
profile = json.load(open(case / "profile.json"))["hashprune_backfill"]
resource = json.load(open(case / "resource_summary.json"))
value = {"graph": graph, "graph_quality": quality, "hashprune_backfill": profile, "resource": resource}
(case / "accepted_metrics.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY_ACCEPT
  fsync_evidence "$case_dir"
  atomic_status "$case_dir/STATUS" accepted
  fsync_evidence "$case_dir"
  cleanup_accepted_payload "$case_dir"
}

write_provenance() {
  write_matrix "$BATCH_ROOT/frozen_matrix.tsv"
  sha256sum "$BUILD_BIN" "$SEARCH_BIN" "$RABITQ_BUILDER" "$GRAPH_QUALITY_BIN" \
    "$QUERY" "$GT" "$RABITQ1_SIDECAR" >"$BATCH_ROOT/provenance/sha256.txt"
  dd if="$BASE" bs=4096 count=1 status=none | sha256sum >"$BATCH_ROOT/provenance/base_header_window.sha256"
  git -C "$REPO" rev-parse HEAD >"$BATCH_ROOT/provenance/git_commit.txt"
  git -C "$REPO" status --short >"$BATCH_ROOT/provenance/git_status_short.txt"
  nvidia-smi >"$BATCH_ROOT/provenance/nvidia_smi.txt"
}

CURRENT_CASE=""

on_formal_exit() {
  local status=$?
  set +e
  if [[ -n ${BATCH_ROOT:-} && -d ${BATCH_ROOT:-} && -f $BATCH_ROOT/STATUS ]]; then
    if [[ $(<"$BATCH_ROOT/STATUS") != accepted ]]; then
      atomic_status "$BATCH_ROOT/STATUS" failed
    fi
  fi
  if [[ -n $CURRENT_CASE && -d $CURRENT_CASE && -f $CURRENT_CASE/STATUS ]]; then
    if [[ $(<"$CURRENT_CASE/STATUS") == running ]]; then
      atomic_status "$CURRENT_CASE/STATUS" failed
    fi
  fi
  exit "$status"
}

run_formal() {
  local row
  acquire_locks
  validate_inputs
  require_new_batch_root
  write_provenance
  trap on_formal_exit EXIT
  for row in "${MATRIX[@]}"; do
    IFS='|' read -r ordinal label _ <<<"$row"
    CURRENT_CASE="$BATCH_ROOT/transactions/${ordinal}_${label}"
    run_formal_case "$row"
    CURRENT_CASE=""
  done
  if [[ $EXPERIMENT_PROFILE == four-policy-cap3 ]]; then
    python3 "$FOUR_POLICY_SUMMARIZER" --batch-root "$BATCH_ROOT"
  else
    python3 "$SUMMARIZER" --batch-root "$BATCH_ROOT"
  fi
  atomic_status "$BATCH_ROOT/STATUS" accepted
  trap - EXIT
}

resume_four_after_plain() {
  [[ $EXPERIMENT_PROFILE == four-policy-cap3 ]] || die "resume-four-after-plain requires four-policy-cap3"
  [[ -n $RUN_ID ]] || die "RUN_ID is required"
  acquire_locks
  validate_inputs
  BATCH_ROOT="$OUTPUT_BASE/$RUN_ID"
  [[ -d $BATCH_ROOT ]] || die "missing run root: $BATCH_ROOT"
  atomic_status "$BATCH_ROOT/STATUS" running
  trap on_formal_exit EXIT

  local plain="$BATCH_ROOT/transactions/001_plain"
  [[ -f $plain/graph_validation.json && -f $plain/graph_quality.tsv ]] || die "Plain validation evidence is incomplete"
  [[ -f $plain/policy_validation.json && -f $plain/resource_summary.json ]] || die "Plain policy/resource evidence is incomplete"
  [[ -d $plain/search/pass1 ]] || die "Plain pass1 search results are missing"
  rm -rf -- "$plain/search/pass2"
  local expected_l_csv
  expected_l_csv=$(IFS=,; echo "${PASS1_L[*]}")
  python3 - "$plain" "$expected_l_csv" <<'PY_RESUME_PLAIN'
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

case = Path(sys.argv[1])
expected_l = tuple(int(value) for value in sys.argv[2].split(","))
rows = []
fields = None
for l_value in expected_l:
    root = case / "search" / "pass1" / f"L{l_value}"
    with (root / "metrics.tsv").open(newline="") as stream:
        metric = next(csv.DictReader(stream, delimiter="\t"))
    digest = hashlib.sha256((root / "results.fbin").read_bytes()).hexdigest()
    row = {"pass": "pass1", "L": l_value, "result_sha256": digest, **metric}
    rows.append(row)
    fields = list(row)
output = case / "search_measurements.tsv"
temporary = output.with_suffix(output.suffix + ".partial")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(output)

graph = json.load(open(case / "graph_validation.json"))
quality = next(csv.DictReader((case / "graph_quality.tsv").open(), delimiter="\t"))
profile = json.load(open(case / "profile.json"))["hashprune_backfill"]
resource = json.load(open(case / "resource_summary.json"))
value = {"graph": graph, "graph_quality": quality, "hashprune_backfill": profile, "resource": resource}
(case / "accepted_metrics.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY_RESUME_PLAIN
  fsync_evidence "$plain"
  atomic_status "$plain/STATUS" accepted
  fsync_evidence "$plain"
  cleanup_accepted_payload "$plain"

  local row ordinal label
  for row in "${MATRIX[@]:1}"; do
    IFS='|' read -r ordinal label _ <<<"$row"
    CURRENT_CASE="$BATCH_ROOT/transactions/${ordinal}_${label}"
    run_formal_case "$row"
    CURRENT_CASE=""
  done
  python3 "$FOUR_POLICY_SUMMARIZER" --batch-root "$BATCH_ROOT"
  atomic_status "$BATCH_ROOT/STATUS" accepted
  trap - EXIT
}

resume_four_after_distance() {
  [[ $EXPERIMENT_PROFILE == four-policy-cap3 ]] || die "resume-four-after-distance requires four-policy-cap3"
  [[ -n $RUN_ID ]] || die "RUN_ID is required"
  acquire_locks
  validate_inputs
  BATCH_ROOT="$OUTPUT_BASE/$RUN_ID"
  [[ -d $BATCH_ROOT ]] || die "missing run root: $BATCH_ROOT"
  atomic_status "$BATCH_ROOT/STATUS" running
  trap on_formal_exit EXIT

  local distance="$BATCH_ROOT/transactions/002_distance_first_q3_cap3"
  [[ -f $distance/graph_validation.json && -f $distance/graph_quality.tsv ]] || \
    die "DistanceFirst validation evidence is incomplete"
  validate_profile "$distance/profile.json" rhalf-distance-first-q3 3 NA \
    "$distance/policy_validation.json"
  atomic_status "$distance/STATUS" running
  run_search_sweep "$distance"
  python3 - "$distance" <<'PY_RESUME_DISTANCE'
import csv
import json
from pathlib import Path
import sys

case = Path(sys.argv[1])
graph = json.load(open(case / "graph_validation.json"))
quality = next(csv.DictReader((case / "graph_quality.tsv").open(), delimiter="\t"))
profile = json.load(open(case / "profile.json"))["hashprune_backfill"]
resource = json.load(open(case / "resource_summary.json"))
value = {"graph": graph, "graph_quality": quality, "hashprune_backfill": profile, "resource": resource}
(case / "accepted_metrics.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY_RESUME_DISTANCE
  fsync_evidence "$distance"
  atomic_status "$distance/STATUS" accepted
  fsync_evidence "$distance"
  cleanup_accepted_payload "$distance"

  local row ordinal label
  for row in "${MATRIX[@]:2}"; do
    IFS='|' read -r ordinal label _ <<<"$row"
    CURRENT_CASE="$BATCH_ROOT/transactions/${ordinal}_${label}"
    run_formal_case "$row"
    CURRENT_CASE=""
  done
  python3 "$FOUR_POLICY_SUMMARIZER" --batch-root "$BATCH_ROOT"
  atomic_status "$BATCH_ROOT/STATUS" accepted
  trap - EXIT
}

run_summarize() {
  BATCH_ROOT=${BATCH_ROOT:-${RUN_ID:+$OUTPUT_BASE/$RUN_ID}}
  [[ -n ${BATCH_ROOT:-} && -d $BATCH_ROOT ]] || die "set BATCH_ROOT or OUTPUT_BASE and RUN_ID"
  if [[ $EXPERIMENT_PROFILE == four-policy-cap3 ]]; then
    python3 "$FOUR_POLICY_SUMMARIZER" --batch-root "$BATCH_ROOT"
  else
    python3 "$SUMMARIZER" --batch-root "$BATCH_ROOT"
  fi
}

case "$MODE" in
  help|-h|--help) usage ;;
  smoke) run_smoke ;;
  prepare) prepare_binaries ;;
  preflight) run_preflight ;;
  formal) run_formal ;;
  resume-four-after-plain) resume_four_after_plain ;;
  resume-four-after-distance) resume_four_after_distance ;;
  summarize) run_summarize ;;
  *) usage; die "unknown mode: $MODE" ;;
esac
