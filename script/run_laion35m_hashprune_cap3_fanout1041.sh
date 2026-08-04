#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-help}
OUTPUT_BASE=${OUTPUT_BASE:-/data_staging/wbh/hashprune_softtier_cap6/laion35m_f421}
RUN_ID=${RUN_ID:-}
BATCH_ROOT=${BATCH_ROOT:-}
BASE=/data_staging/wbh/laion-512-35M/base.fbin
QUERY=/data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin
GT=/data_staging/wbh/laion-512-35M/gt100.bin
RABITQ1_SIDECAR=/data_staging/wbh/unified_graph_rabitq1/sidecars/laion35m.rabitq1
RABITQ1_MANIFEST=/data_staging/wbh/unified_graph_rabitq1/sidecars/laion35m.rabitq1.json
ROWS=35000000
DIM=512
MEASURED_L=(10 16 25 32 50 100 200)

# BEGIN FROZEN MATRIX
MATRIX=(
  '001|plain|plain|off|NA|NA'
  '002|c6_hashslot_q1|hashslot|rhalf-hash-slot-1|6|NA'
  '003|c6_distance_first_q3|distance_first|rhalf-distance-first-q3|6|NA'
  '004|c6_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|6|1.02'
  '005|c6_hard_tier_q3|hard_tier|rhalf-hard-tier-q3|6|NA'
)
# END FROZEN MATRIX

BUILD_BIN="$REPO/target/release/build_shear_index"
SEARCH_BIN="$REPO/target/release/search_shear_index"
RABITQ_BUILDER="$REPO/target/release/build_graph_rabitq_1bit"
GRAPH_QUALITY_BIN="$REPO/target/release/graph_quality"
GPU_MONITOR="$REPO/script/monitor_gpu_build.sh"
SUMMARIZER="$REPO/script/summarize_laion35m_hashprune_cap3_fanout1041.py"
CARGO_BIN=${CARGO_BIN:-cargo}

usage() {
  cat <<'USAGE'
Usage: bash script/run_laion35m_hashprune_cap3_fanout1041.sh MODE

Modes:
  help       Show this contract.
  prepare    Build the required release binaries.
  build-sidecar Build and validate the dedicated LAION35M RaBitQ1 sidecar.
  preflight  Validate inputs, sidecar manifest, binaries, locks, GPU, and space.
  smoke      Execute five synthetic transactions under OUTPUT_BASE/RUN_ID.
  formal     Execute the frozen sequential LAION35M ablation; RUN_ID is required.
  summarize Regenerate the report for BATCH_ROOT or OUTPUT_BASE/RUN_ID.

Frozen contract: fanout 4/2/1; Plain plus four cap=6 policies; SoftTier
priority alpha=1.02; adaptive-exact search; one L=200 graph warmup followed by
one measured pass at L=10,16,25,32,50,100,200.

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

validate_new_run_target() {
  [[ -n $RUN_ID ]] || die "RUN_ID is required"
  [[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: $RUN_ID"
  BATCH_ROOT="$OUTPUT_BASE/$RUN_ID"
  [[ ! -e $BATCH_ROOT ]] || die "run root already exists: $BATCH_ROOT"
}

require_new_batch_root() {
  validate_new_run_target
  mkdir -p "$BATCH_ROOT/transactions" "$BATCH_ROOT/report" "$BATCH_ROOT/provenance"
  atomic_status "$BATCH_ROOT/STATUS" running
}

write_matrix() {
  local destination=$1 row ordinal label method policy cap alpha
  printf 'ordinal\tlabel\tmethod\tpolicy_mode\trow_cap\tpriority_alpha\n' \
    >"$destination"
  for row in "${MATRIX[@]}"; do
    IFS='|' read -r ordinal label method policy cap alpha <<<"$row"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ordinal" "$label" "$method" "$policy" "$cap" "$alpha" \
      >>"$destination"
  done
}

write_smoke_build_command() {
  local case_dir=$1 policy=$2 cap=$3 alpha=$4
  {
    printf '%q ' "$BUILD_BIN" --max-degree 64 --lbuild 100 --num-threads 8 \
      --m-hash-bits 12 --fanout-top 4 --fanout-second 2 \
      --root-risk-candidate-rs 1,2,4 --seed 42 \
      --hashprune-backfill-mode "$policy" --hashprune-backfill-alpha 1.0
    if [[ $policy != off ]]; then
      printf '%q ' --hashprune-rhalf-max-backfill-per-row "$cap"
    fi
    if [[ $policy == rhalf-soft-tier-q3 ]]; then
      printf '%q ' --hashprune-soft-tier-priority-alpha "$alpha"
    fi
    printf '\n'
  } >"$case_dir/build_command.sh"
}

record_smoke_call() {
  local case_dir=$1 kind=$2 pass=$3 l_value=$4
  printf '%s\t%s\t%s\n' "$kind" "$pass" "$l_value" >>"$case_dir/search_calls.tsv"
}

write_smoke_evidence() {
  local case_dir=$1 ordinal=$2 label=$3 method=$4 policy=$5 cap=$6 alpha=$7
  python3 - "$case_dir" "$ordinal" "$label" "$method" "$policy" "$cap" "$alpha" <<'PY_SMOKE'
import csv
import json
import sys
from pathlib import Path

case = Path(sys.argv[1])
ordinal = int(sys.argv[2], 10)
label, method, policy, cap_text, alpha_text = sys.argv[3:]
enabled = policy != "off"
cap = int(cap_text) if enabled else None
priority_alpha = float(alpha_text) if method == "soft_tier" else None
effective_alpha = priority_alpha if priority_alpha is not None else 1.0
quota = 1 if method in ("plain", "hashslot") else 3
avg_degree = 40.0 if not enabled else 42.0 + ordinal / 100.0
graph_bytes = int(5_600_000_000 + avg_degree * 140_000_000)
backfill = {
    "mode": policy,
    "ablation": "none",
    "derived_budget": 32 if enabled else 0,
    "requested_hash_quota": None,
    "effective_hash_quota": quota,
    "requested_max_backfill_per_row": cap,
    "effective_max_backfill_per_row": cap if enabled else 0,
    "capacity_rule": (
        "min-derived-budget-row-cap-spare"
        if enabled
        else "min-derived-budget-spare"
    ),
    "requested_priority_alpha": priority_alpha,
    "effective_priority_alpha": effective_alpha,
    "rows_seen": 35_000_000,
    "rows_underfull_before": 12_000_000 if enabled else 0,
    "rows_with_backfill": 10_000_000 if enabled else 0,
    "rows_with_nonempty_shortlist": 11_000_000 if enabled else 0,
    "shortlist_admissions": 30_000_000 if enabled else 0,
    "shortlist_replacements": 2_000_000 + ordinal if enabled else 0,
    "shortlist_unchanged": 4_000_000 if enabled else 0,
    "shortlist_capacity_rejections": 3_000_000 + ordinal if enabled else 0,
    "shortlisted_layer_1": 24_000_000 if enabled else 0,
    "shortlisted_layer_2": 9_000_000 if enabled else 0,
    "shortlisted_layer_3": 6_000_000 if enabled else 0,
    "emitted_layer_1": 18_000_000 if enabled else 0,
    "emitted_layer_2": 7_000_000 if enabled else 0,
    "emitted_layer_3": 4_000_000 if enabled else 0,
    "rows_with_deeper_backfill": 5_000_000 if enabled else 0,
    "primary_edges": 1_400_000_000,
    "backfill_edges": 29_000_000 + ordinal if enabled else 0,
    "mean_degree_before": 40.0,
    "mean_degree_after": avg_degree,
}
profile = {"hashprune_backfill": backfill}
(case / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
graph = {
    "rows": 35_000_000,
    "edges": int(avg_degree * 35_000_000),
    "max_degree": 64,
    "avg_out_degree": avg_degree,
    "start": 0,
    "frozen": 0,
    "graph_bytes": graph_bytes,
}
(case / "graph_validation.json").write_text(
    json.dumps(graph, indent=2, sort_keys=True) + "\n"
)
quality_fields = [
    "nodes", "edges", "start", "max_out_degree", "avg_out_degree",
    "p50_out_degree", "p95_out_degree", "zero_out_nodes", "avg_in_degree",
    "p50_in_degree", "p95_in_degree", "zero_in_nodes",
    "reciprocal_sampled_edges", "reciprocal_directed_edges",
    "reciprocal_directed_ratio", "directed_reachable_from_start",
    "directed_reachable_from_start_ratio", "edge_distance_samples",
    "edge_distance_p50", "edge_distance_p95", "edge_distance_p99",
]
quality = {
    "nodes": 35_000_000,
    "edges": graph["edges"],
    "start": 0,
    "max_out_degree": 64,
    "avg_out_degree": avg_degree,
    "p50_out_degree": 42,
    "p95_out_degree": 60,
    "zero_out_nodes": 0,
    "avg_in_degree": avg_degree,
    "p50_in_degree": 40,
    "p95_in_degree": 70,
    "zero_in_nodes": 10,
    "reciprocal_sampled_edges": 1_000_000,
    "reciprocal_directed_edges": 900_000,
    "reciprocal_directed_ratio": 0.9,
    "directed_reachable_from_start": 34_990_000,
    "directed_reachable_from_start_ratio": 0.999714,
    "edge_distance_samples": 1_000_000,
    "edge_distance_p50": 0.3,
    "edge_distance_p95": 0.6,
    "edge_distance_p99": 0.8,
}
with (case / "graph_quality.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=quality_fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(quality)
resource = {
    "peak_process_rss_bytes": 48_000_000_000 + ordinal,
    "peak_hbm_mib": 15_000.0,
    "monitor_elapsed_seconds": 100.0 + ordinal,
    "time_elapsed_display": f"1:{40 + ordinal:02d}.00",
    "build_wall_seconds": 100.0 + ordinal,
}
(case / "resource_summary.json").write_text(
    json.dumps(resource, indent=2, sort_keys=True) + "\n"
)
search_fields = [
    "pass", "L", "storage", "queries", "recall_at", "search_list_size",
    "recall", "search_seconds", "qps", "avg_visited_nodes",
    "avg_expanded_nodes", "avg_distance_comparisons",
    "total_one_bit_evaluations", "total_navigation_exact_evaluations",
    "total_adaptive_exact_refinements", "total_post_rerank_exact_evaluations",
    "total_exact_evaluations", "logical_exact_vector_bytes",
    "search_backend", "graph_bytes", "result_sha256",
]
rows = []
method_bonus = {
    "plain": 0.0,
    "hashslot": 0.004,
    "distance_first": 0.006,
    "soft_tier": 0.010,
    "hard_tier": 0.005,
}[method]
for l_value in (10, 16, 25, 32, 50, 100, 200):
    recall = min(0.99, 0.72 + l_value / 1000.0 + method_bonus)
    qps = 30_000.0 / (1.0 + l_value / 20.0) - ordinal
    rows.append({
        "pass": "measured",
        "L": l_value,
        "storage": "mmap",
        "queries": 10_000,
        "recall_at": 10,
        "search_list_size": l_value,
        "recall": recall,
        "search_seconds": 10_000 / qps,
        "qps": qps,
        "avg_visited_nodes": l_value * 2,
        "avg_expanded_nodes": l_value,
        "avg_distance_comparisons": l_value * 3,
        "total_one_bit_evaluations": l_value * 30_000,
        "total_navigation_exact_evaluations": l_value * 5_000,
        "total_adaptive_exact_refinements": l_value * 2_000,
        "total_post_rerank_exact_evaluations": 100_000,
        "total_exact_evaluations": l_value * 7_000 + 100_000,
        "logical_exact_vector_bytes": (l_value * 7_000 + 100_000) * 512 * 4,
        "search_backend": "rabitq1-adaptive-exact",
        "graph_bytes": graph_bytes,
        "result_sha256": f"synthetic-{label}-L{l_value}",
    })
with (case / "search_measurements.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=search_fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
accepted = {
    "label": label,
    "method": method,
    "graph": graph,
    "graph_quality": quality,
    "hashprune_backfill": backfill,
    "resource": resource,
}
(case / "accepted_metrics.json").write_text(
    json.dumps(accepted, indent=2, sort_keys=True) + "\n"
)
PY_SMOKE
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
}

run_smoke_case() {
  local row=$1 ordinal label method policy cap alpha case_dir l_value
  IFS='|' read -r ordinal label method policy cap alpha <<<"$row"
  case_dir="$BATCH_ROOT/transactions/${ordinal}_${label}"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar,search}
  atomic_status "$case_dir/STATUS" running
  printf '1\n' >"$case_dir/attempt_count"
  printf 'synthetic graph for %s\n' "$label" >"$case_dir/index/_mem.index"
  if [[ $policy == off ]]; then
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":null,"priority_alpha":null}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" \
      >"$case_dir/parameter_passport.json"
  else
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":%s,"priority_alpha":"%s"}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" "$cap" "$alpha" \
      >"$case_dir/parameter_passport.json"
  fi
  write_smoke_build_command "$case_dir" "$policy" "$cap" "$alpha"
  printf 'kind\tpass\tL\n' >"$case_dir/search_calls.tsv"
  record_smoke_call "$case_dir" warmup warmup 200
  for l_value in 10 16 25 32 50 100 200; do
    record_smoke_call "$case_dir" measured measured "$l_value"
  done
  if [[ ${LAION35M_SMOKE_FAIL_LABEL:-} == "$label" ]]; then
    printf 'forced synthetic failure\n' >"$case_dir/failure.txt"
    atomic_status "$case_dir/STATUS" failed
    return 1
  fi
  write_smoke_evidence "$case_dir" "$ordinal" "$label" "$method" "$policy" "$cap" "$alpha"
  atomic_status "$case_dir/STATUS" accepted
  cleanup_accepted_payload "$case_dir"
}

run_smoke() {
  local row graph_count=${#MATRIX[@]} measured_count
  measured_count=$((graph_count * ${#MEASURED_L[@]}))
  require_new_batch_root
  write_matrix "$BATCH_ROOT/frozen_matrix.tsv"
  for row in "${MATRIX[@]}"; do
    if ! run_smoke_case "$row"; then
      atomic_status "$BATCH_ROOT/STATUS" failed
      return 1
    fi
  done
  printf '{"schema":"laion35m_hashprune_cap6_fanout421_smoke_v1","accepted":%s,"graphs":%s,"warmups":%s,"measured_rows":%s,"search_calls":%s}\n' \
    "$graph_count" "$graph_count" "$graph_count" "$measured_count" \
    "$((graph_count + measured_count))" >"$BATCH_ROOT/experiment_manifest.json"
  python3 "$SUMMARIZER" --batch-root "$BATCH_ROOT"
  atomic_status "$BATCH_ROOT/STATUS" accepted
}

prepare_binaries() {
  [[ -x $CARGO_BIN ]] || die "missing cargo: $CARGO_BIN"
  PATH="$(dirname "$CARGO_BIN"):$PATH" "$CARGO_BIN" build --locked --release \
    -p build_shear_index --features build_shear_index/cuda
  PATH="$(dirname "$CARGO_BIN"):$PATH" "$CARGO_BIN" build --locked --release \
    -p search_shear_index --bins
}

expected_sidecar_bytes() {
  local padded stride
  padded=$(( (DIM + 63) / 64 * 64 ))
  stride=$(( padded / 8 + 12 ))
  printf '%s\n' $((256 + padded / 2 + 16 * padded * 4 + ROWS * (1 + stride)))
}

validate_dataset_headers() {
  python3 - "$BASE" "$QUERY" "$GT" "$ROWS" "$DIM" <<'PY_HEADERS'
import math
import os
import struct
import sys

base, query, gt, rows, dim = sys.argv[1:]
rows, dim = int(rows), int(dim)
for path, expected in ((base, (rows, dim)), (query, (10_000, dim))):
    with open(path, "rb") as stream:
        header = stream.read(8)
    if len(header) != 8 or struct.unpack("<II", header) != expected:
        raise SystemExit(f"fbin header mismatch: {path}")
    expected_bytes = 8 + expected[0] * expected[1] * 4
    if os.path.getsize(path) != expected_bytes:
        raise SystemExit(f"fbin length mismatch: {path}")
with open(gt, "rb") as stream:
    header = stream.read(8)
    payload = stream.read()
if len(header) != 8:
    raise SystemExit("truncated ground truth header")
queries, width = struct.unpack("<II", header)
count = queries * width
if queries != 10_000 or width < 10 or len(payload) not in (count * 4, count * 8):
    raise SystemExit("ground truth shape or length mismatch")
ids = struct.unpack_from(f"<{count}I", payload)
if any(point >= rows for point in ids):
    raise SystemExit("ground truth contains an out-of-range point ID")
if len(payload) == count * 8:
    distances = struct.iter_unpack("<f", payload[count * 4:])
    if any(not math.isfinite(value[0]) for value in distances):
        raise SystemExit("ground truth contains a non-finite distance")
PY_HEADERS
}

validate_sidecar_contract() {
  local validation_output=${1:-/dev/null} expected
  expected=$(expected_sidecar_bytes)
  python3 - "$RABITQ1_SIDECAR" "$RABITQ1_MANIFEST" "$ROWS" "$DIM" "$expected" <<'PY_SIDECAR'
import json
import os
import sys

sidecar, manifest_path, rows, dim, expected_bytes = sys.argv[1:]
rows, dim, expected_bytes = int(rows), int(dim), int(expected_bytes)
manifest = json.load(open(manifest_path))
expected = {
    "schema": "graph_rabitq1_manifest_v1",
    "rows": rows,
    "dim": dim,
    "padded_dim": ((dim + 63) // 64) * 64,
    "center_count": 16,
    "code_bytes": ((dim + 63) // 64) * 64 // 8,
    "bin_stride": ((dim + 63) // 64) * 64 // 8 + 12,
    "ex_data_bytes": 0,
    "sidecar_bytes": expected_bytes,
}
checks = {key: manifest.get(key) == value for key, value in expected.items()}
checks["sidecar_file_bytes"] = os.path.getsize(sidecar) == expected_bytes
if not all(checks.values()):
    raise SystemExit(f"RaBitQ1 sidecar contract mismatch: {checks}")
PY_SIDECAR
  "$RABITQ_BUILDER" --data-path "$BASE" --output "$RABITQ1_SIDECAR" \
    --validate-only >"$validation_output"
}

run_build_sidecar() {
  [[ -x $RABITQ_BUILDER ]] || die "missing executable sidecar builder: $RABITQ_BUILDER"
  [[ -f $BASE ]] || die "missing LAION35M base: $BASE"
  validate_dataset_headers
  mkdir -p "$(dirname "$RABITQ1_SIDECAR")" "$OUTPUT_BASE/sidecar_build"
  acquire_locks
  if [[ -f $RABITQ1_SIDECAR && -f $RABITQ1_MANIFEST && ${REBUILD_SIDECAR:-0} != 1 ]]; then
    validate_sidecar_contract "$OUTPUT_BASE/sidecar_build/validation.json"
    sha256sum "$BASE" "$RABITQ1_SIDECAR" "$RABITQ1_MANIFEST" \
      >"$OUTPUT_BASE/sidecar_build/sha256.txt"
    printf 'accepted\n' >"$OUTPUT_BASE/sidecar_build/STATUS"
    printf 'validated and reused existing sidecar: %s\n' "$RABITQ1_SIDECAR"
    return
  fi
  if [[ -e $RABITQ1_SIDECAR || -e $RABITQ1_MANIFEST ]]; then
    [[ ${REBUILD_SIDECAR:-0} == 1 ]] || \
      die "partial or invalid sidecar exists; set REBUILD_SIDECAR=1 after inspection"
  fi
  local temporary_root temporary_sidecar temporary_manifest status
  temporary_root=$(mktemp -d "$(dirname "$RABITQ1_SIDECAR")/.laion35m.rabitq1.build.XXXXXX")
  temporary_sidecar="$temporary_root/laion35m.rabitq1"
  temporary_manifest="${temporary_sidecar}.json"
  printf '%q ' "$RABITQ_BUILDER" --data-path "$BASE" --output "$temporary_sidecar" \
    --centroids 16 --sample-rows 262144 --kmeans-iters 20 --seed 42 --threads 8 \
    >"$OUTPUT_BASE/sidecar_build/command.sh"
  printf '\n' >>"$OUTPUT_BASE/sidecar_build/command.sh"
  set +e
  /usr/bin/time -v -o "$OUTPUT_BASE/sidecar_build/time.log" \
    "$RABITQ_BUILDER" --data-path "$BASE" --output "$temporary_sidecar" \
      --centroids 16 --sample-rows 262144 --kmeans-iters 20 --seed 42 --threads 8 \
      >"$OUTPUT_BASE/sidecar_build/stdout.log" \
      2>"$OUTPUT_BASE/sidecar_build/stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$OUTPUT_BASE/sidecar_build/exit_code"
  if (( status != 0 )); then
    printf 'sidecar build failed; preserved temporary root: %s\n' "$temporary_root" >&2
    return "$status"
  fi
  [[ -f $temporary_sidecar && -f $temporary_manifest ]] || \
    die "sidecar builder did not produce both payload and manifest"
  "$RABITQ_BUILDER" --data-path "$BASE" --output "$temporary_sidecar" \
    --validate-only >"$OUTPUT_BASE/sidecar_build/validation.json"
  local expected
  expected=$(expected_sidecar_bytes)
  [[ $(stat -Lc %s -- "$temporary_sidecar") == "$expected" ]] || \
    die "temporary sidecar byte length mismatch"
  python3 - "$temporary_sidecar" "$temporary_manifest" "$(dirname "$RABITQ1_SIDECAR")" <<'PY_FSYNC_SIDECAR'
import os
import sys

for path in sys.argv[1:3]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
descriptor = os.open(sys.argv[3], os.O_RDONLY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY_FSYNC_SIDECAR
  mv -f -- "$temporary_sidecar" "$RABITQ1_SIDECAR"
  mv -f -- "$temporary_manifest" "$RABITQ1_MANIFEST"
  rmdir -- "$temporary_root"
  validate_sidecar_contract "$OUTPUT_BASE/sidecar_build/validation.json"
  sha256sum "$BASE" "$RABITQ1_SIDECAR" "$RABITQ1_MANIFEST" \
    >"$OUTPUT_BASE/sidecar_build/sha256.txt"
  printf 'accepted\n' >"$OUTPUT_BASE/sidecar_build/STATUS"
}

acquire_locks() {
  mkdir -p "$OUTPUT_BASE"
  exec 8>"$OUTPUT_BASE/.batch.lock"
  flock -n 8 || die "LAION35M HashPrune cap-3 batch lock is held"
  exec 9>/data_staging/wbh/.shear_experiment.lock
  flock -n 9 || die "global SHEAR experiment lock is held"
}

validate_inputs() {
  local required free_bytes minimum_free=$((100 * 1024 * 1024 * 1024))
  for required in "$BUILD_BIN" "$SEARCH_BIN" "$RABITQ_BUILDER" "$GRAPH_QUALITY_BIN" \
    "$GPU_MONITOR" "$SUMMARIZER" "$BASE" "$QUERY" "$GT" "$RABITQ1_SIDECAR" \
    "$RABITQ1_MANIFEST"; do
    [[ -e $required ]] || die "missing required input: $required"
  done
  [[ -x $BUILD_BIN && -x $SEARCH_BIN && -x $RABITQ_BUILDER && -x $GRAPH_QUALITY_BIN ]] || \
    die "one or more release binaries are not executable"
  if strings "$BUILD_BIN" | grep -Fq 'SHEAR was built without CUDA support'; then
    die "build_shear_index lacks the required CUDA feature; run prepare"
  fi
  validate_dataset_headers
  validate_sidecar_contract
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l) -eq 0 ]] || \
    die "GPU 0 is not idle"
  sudo -n true || die "passwordless sudo is required for controlled page-cache drops"
  free_bytes=$(df -PB1 "$OUTPUT_BASE" | awk 'NR==2 {print $4}')
  (( free_bytes >= minimum_free )) || die "less than 100 GiB free below OUTPUT_BASE"
  printf 'RaBitQ1 sidecar sha256=%s\n' "$(sha256sum "$RABITQ1_SIDECAR" | awk '{print $1}')"
}

run_preflight() {
  validate_new_run_target
  acquire_locks
  validate_inputs
  printf 'preflight accepted: LAION35M inputs, GT, sidecar manifest, binaries, GPU, locks, and space\n'
}

drop_cache() {
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

prewarm_file() {
  local path=$1 manifest=$2 elapsed_file
  elapsed_file="${manifest}.elapsed.$$"
  /usr/bin/time -f '%e' -o "$elapsed_file" dd if="$path" of=/dev/null bs=16M status=none
  printf '%s\t%s\t%s\n' "$path" "$(stat -Lc %s -- "$path")" "$(<"$elapsed_file")" \
    >>"$manifest"
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
    if declared != len(data) or cap != 64 or start >= rows:
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
        if source in neighbors or len(neighbors) != len(set(neighbors)):
            raise SystemExit(f"invalid neighbors at row {source}")
        if any(neighbor >= rows for neighbor in neighbors):
            raise SystemExit(f"out-of-range neighbor at row {source}")
        edges += degree
        observed = max(observed, degree)
    if offset != len(data):
        raise SystemExit("graph byte consumption mismatch")
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
  local profile=$1 policy=$2 cap=$3 alpha=$4 output=$5
  python3 - "$profile" "$policy" "$cap" "$alpha" "$output" <<'PY_PROFILE'
import json
import math
import os
import sys

profile_path, policy, cap, alpha, output = sys.argv[1:]
backfill = json.load(open(profile_path)).get("hashprune_backfill")
if not isinstance(backfill, dict):
    raise SystemExit("missing hashprune_backfill profile")
selection_pressure_observed = False
if policy == "off":
    checks = {
        "mode": backfill.get("mode") == "off",
        "requested_hash_quota_default": backfill.get("requested_hash_quota") is None,
        "effective_hash_quota": backfill.get("effective_hash_quota") == 1,
        "derived_budget": backfill.get("derived_budget") == 0,
        "requested_row_cap": backfill.get("requested_max_backfill_per_row") is None,
        "effective_row_cap": backfill.get("effective_max_backfill_per_row") == 0,
        "capacity_rule": backfill.get("capacity_rule")
        == "min-derived-budget-spare",
        "requested_priority_alpha": backfill.get("requested_priority_alpha") is None,
        "effective_priority_alpha": backfill.get("effective_priority_alpha") == 1,
        "backfill_edges": backfill.get("backfill_edges") == 0,
    }
else:
    cap = int(cap)
    expected_quota = 1 if policy == "rhalf-hash-slot-1" else 3
    expected_alpha = float(alpha) if policy == "rhalf-soft-tier-q3" else 1.0
    checks = {
        "mode": backfill.get("mode") == policy,
        "requested_hash_quota_default": backfill.get("requested_hash_quota") is None,
        "effective_hash_quota": backfill.get("effective_hash_quota") == expected_quota,
        "derived_budget": backfill.get("derived_budget") == 32,
        "requested_row_cap": backfill.get("requested_max_backfill_per_row") == cap,
        "effective_row_cap": backfill.get("effective_max_backfill_per_row") == cap,
        "capacity_rule": backfill.get("capacity_rule")
        == "min-derived-budget-row-cap-spare",
        "effective_priority_alpha": math.isclose(
            float(backfill.get("effective_priority_alpha")), expected_alpha
        ),
    }
    pressure_fields = (
        "shortlist_replacements",
        "shortlist_capacity_rejections",
    )
    pressure_fields_present = all(
        type(backfill.get(field)) is int and backfill[field] >= 0
        for field in pressure_fields
    )
    checks["selection_pressure_fields_present"] = pressure_fields_present
    selection_pressure_observed = (
        pressure_fields_present
        and (
            backfill["shortlist_replacements"] > 0
            or backfill["shortlist_capacity_rejections"] > 0
        )
    )
    if policy == "rhalf-soft-tier-q3":
        checks["requested_priority_alpha"] = math.isclose(
            float(backfill.get("requested_priority_alpha")), expected_alpha
        )
    else:
        checks["requested_priority_alpha_default"] = (
            backfill.get("requested_priority_alpha") is None
        )
if not all(checks.values()):
    raise SystemExit(f"HashPrune profile mismatch: {checks}")
value = {
    "accepted": True,
    "checks": checks,
    "selection_pressure_observed": selection_pressure_observed,
    "backfill": backfill,
}
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
  local case_dir=$1 policy=$2 cap=$3 alpha=$4
  local alpha_args=() cap_args=()
  if [[ $policy != off ]]; then
    cap_args=(--hashprune-rhalf-max-backfill-per-row "$cap")
  fi
  if [[ $policy == rhalf-soft-tier-q3 ]]; then
    alpha_args=(--hashprune-soft-tier-priority-alpha "$alpha")
  fi
  local command=(env
    SHEAR_PIPNN_PROFILE=1 SHEAR_PIPNN_PHASE_MONITOR=1
    SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1 SHEAR_PIPNN_WMMA_CTA_SHARED_B=0
    SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0 SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0
    SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0 SHEAR_PIPNN_PRUNE_BACKEND=hashprune
    SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0 OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$BUILD_BIN" --data-type float --dist-fn l2 --data-path "$BASE"
    --index-path-prefix "$case_dir/index" --overwrite --oom-enable
    --oom-memory-budget-gb 0 --oom-io-mode buffered --oom-temp-dir "$case_dir/oom"
    --oom-profile-json "$case_dir/profile.json"
    --driver-timing-json "$case_dir/driver_timing.json"
    --max-degree 64 --lbuild 100 --num-threads 8 --m-hash-bits 12
    --leaf-size-min 256 --leaf-size-max 1280 --leaf-knn 2 --psamp-fraction 0.05
    --fanout-top 4 --fanout-second 2 --root-risk-candidate-rs 1,2,4 --seed 42
    --partitioner gpu-subtree --gpu-subtree-root-assignment-backend cpu
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32
    --gpu-subtree-execution-mode fused-persistent --gpu-subtree-d0-capacity-enable
    --gpu-subtree-d0-capacity-split-recursion-limit 8
    --gpu-subtree-d0-capacity-high-ratio 0.08
    --gpu-subtree-d0-capacity-target-ratio 0.08
    --gpu-subtree-d0-capacity-low-ratio 0.08
    --gpu-subtree-d0-partition-mode gpu-kmeans-coarse
    --gpu-subtree-sq-staging-mode gpu-d0
    --gpu-subtree-sq-staging-upload-shard-rows 262144
    --gpu-subtree-sq-wmma-mode build
    --gpu-subtree-sq-wmma-sidecar-prefix "$case_dir/sq_sidecar/sq_wmma"
    --gpu-subtree-sq-wmma-sidecar-mode rebuild
    --gpu-subtree-persistent-large-ticket-batch auto
    --gpu-subtree-sq8-assignment-kernel auto --gpu-subtree-sq8-leaf-mode auto
    --gpu-subtree-persistent-scheduler-mode auto --hashprune-evidence-mode off
    --hashprune-evidence-overflow-budget 0 --hashprune-backfill-mode "$policy"
    --hashprune-backfill-ablation none --hashprune-backfill-budget 0
    --hashprune-backfill-alpha 1.0
    "${cap_args[@]}" "${alpha_args[@]}"
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
    --metrics-path "$run_dir/metrics.tsv" \
    --query-metrics-path "$run_dir/per_query_metrics.tsv" \
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
  run_search_once "$case_dir" 200 warmup warmup
  for l_value in "${MEASURED_L[@]}"; do
    run_search_once "$case_dir" "$l_value" measured measured
  done
  python3 - "$case_dir" <<'PY_SEARCH'
import csv
import os
from pathlib import Path
import sys

case = Path(sys.argv[1])
rows, fields = [], None
for l_value in (10, 16, 25, 32, 50, 100, 200):
    root = case / "search" / "measured" / f"L{l_value}"
    with (root / "metrics.tsv").open(newline="") as stream:
        metric = next(csv.DictReader(stream, delimiter="\t"))
    digest = (root / "results.sha256").read_text().split()[0]
    row = {"pass": "measured", "L": l_value, "result_sha256": digest, **metric}
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

run_formal_case() {
  local row=$1 ordinal label method policy cap alpha case_dir graph
  IFS='|' read -r ordinal label method policy cap alpha <<<"$row"
  case_dir="$BATCH_ROOT/transactions/${ordinal}_${label}"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar,search}
  atomic_status "$case_dir/STATUS" running
  printf '1\n' >"$case_dir/attempt_count"
  if [[ $policy == off ]]; then
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":null,"priority_alpha":null}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" \
      >"$case_dir/parameter_passport.json"
  else
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":%s,"priority_alpha":"%s"}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" "$cap" "$alpha" \
      >"$case_dir/parameter_passport.json"
  fi
  run_build "$case_dir" "$policy" "$cap" "$alpha"
  graph="$case_dir/index/_mem.index"
  validate_graph_strict "$graph" "$case_dir/graph_validation.json"
  sha256sum "$graph" >"$case_dir/graph.sha256"
  "$GRAPH_QUALITY_BIN" --index-path-prefix "$case_dir/index" --data-path "$BASE" \
    --output "$case_dir/graph_quality.tsv" --max-edge-samples 1000000 \
    >"$case_dir/graph_quality.stdout.log" 2>"$case_dir/graph_quality.stderr.log"
  validate_profile "$case_dir/profile.json" "$policy" "$cap" "$alpha" \
    "$case_dir/policy_validation.json"
  run_search_sweep "$case_dir"
  python3 - "$case_dir" <<'PY_ACCEPT'
import csv
import json
import os
from pathlib import Path
import sys

case = Path(sys.argv[1])
value = {
    "graph": json.load(open(case / "graph_validation.json")),
    "graph_quality": next(
        csv.DictReader((case / "graph_quality.tsv").open(), delimiter="\t")
    ),
    "hashprune_backfill": json.load(open(case / "profile.json"))["hashprune_backfill"],
    "resource": json.load(open(case / "resource_summary.json")),
}
output = case / "accepted_metrics.json"
temporary = output.with_suffix(output.suffix + ".partial")
with temporary.open("w") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(output)
PY_ACCEPT
  fsync_evidence "$case_dir"
  atomic_status "$case_dir/STATUS" accepted
  fsync_evidence "$case_dir"
  cleanup_accepted_payload "$case_dir"
  fsync_evidence "$case_dir"
}

write_provenance() {
  local graph_count=${#MATRIX[@]} measured_count
  measured_count=$((graph_count * ${#MEASURED_L[@]}))
  write_matrix "$BATCH_ROOT/frozen_matrix.tsv"
  sha256sum "$BUILD_BIN" "$SEARCH_BIN" "$RABITQ_BUILDER" "$GRAPH_QUALITY_BIN" \
    "$BASE" "$QUERY" "$GT" "$RABITQ1_SIDECAR" "$RABITQ1_MANIFEST" \
    >"$BATCH_ROOT/provenance/sha256.txt"
  git -C "$REPO" rev-parse HEAD >"$BATCH_ROOT/provenance/git_commit.txt"
  git -C "$REPO" status --short >"$BATCH_ROOT/provenance/git_status_short.txt"
  nvidia-smi >"$BATCH_ROOT/provenance/nvidia_smi.txt"
  printf '{"schema":"laion35m_hashprune_cap6_fanout421_v1","graphs":%s,"warmups":%s,"measured_rows":%s,"single_run_qps":true,"fanout":[4,2,1],"row_cap":6}\n' \
    "$graph_count" "$graph_count" "$measured_count" \
    >"$BATCH_ROOT/experiment_manifest.json"
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
  local row ordinal label
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
  python3 "$SUMMARIZER" --batch-root "$BATCH_ROOT"
  atomic_status "$BATCH_ROOT/STATUS" accepted
  trap - EXIT
}

run_summarize() {
  BATCH_ROOT=${BATCH_ROOT:-${RUN_ID:+$OUTPUT_BASE/$RUN_ID}}
  [[ -n ${BATCH_ROOT:-} && -d $BATCH_ROOT ]] || \
    die "set BATCH_ROOT or OUTPUT_BASE and RUN_ID"
  python3 "$SUMMARIZER" --batch-root "$BATCH_ROOT"
}

case "$MODE" in
  help|-h|--help) usage ;;
  prepare) prepare_binaries ;;
  build-sidecar) run_build_sidecar ;;
  preflight) run_preflight ;;
  smoke) run_smoke ;;
  formal) run_formal ;;
  summarize) run_summarize ;;
  *) usage; die "unknown mode: $MODE" ;;
esac
