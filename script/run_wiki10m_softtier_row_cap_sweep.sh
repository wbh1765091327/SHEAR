#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-help}
SWEEP_VARIANT=${SWEEP_VARIANT:-base}
case "$SWEEP_VARIANT" in
  base)
    OUTPUT_BASE=${OUTPUT_BASE:-/data_staging/wbh/hashprune_softtier_row_cap/wiki10m}
    ;;
  extension)
    OUTPUT_BASE=${OUTPUT_BASE:-/data_staging/wbh/hashprune_softtier_row_cap_extension/wiki10m}
    ;;
  *)
    echo "error: unknown SWEEP_VARIANT: $SWEEP_VARIANT" >&2
    exit 1
    ;;
esac
RUN_ID=${RUN_ID:-}
BATCH_ROOT=${BATCH_ROOT:-}
BASE=/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
QUERY=/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
GT=/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
RABITQ1_SIDECAR=/data_staging/wbh/unified_graph_rabitq1/sidecars/wiki10m.rabitq1
RABITQ1_MANIFEST=/data_staging/wbh/unified_graph_rabitq1/sidecars/wiki10m.rabitq1.json
ROWS=10000000
DIM=1024
MEASURED_L=(10 16 25 32 50 100 200)

# BEGIN FROZEN MATRIX
MATRIX=(
  '001|c2_hashslot_q1|hashslot|rhalf-hash-slot-1|2|NA|0'
  '002|c2_distance_first_q3|distance_first|rhalf-distance-first-q3|2|NA|0'
  '003|c2_hard_tier_q3|hard_tier|rhalf-hard-tier-q3|2|NA|0'
  '004|c2_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|2|1.02|0'
  '005|c2_soft_tier_q3_a1p05|soft_tier|rhalf-soft-tier-q3|2|1.05|0'
  '006|c2_soft_tier_q3_a1p10|soft_tier|rhalf-soft-tier-q3|2|1.10|1'
  '007|c3_hashslot_q1|hashslot|rhalf-hash-slot-1|3|NA|0'
  '008|c3_distance_first_q3|distance_first|rhalf-distance-first-q3|3|NA|0'
  '009|c3_hard_tier_q3|hard_tier|rhalf-hard-tier-q3|3|NA|0'
  '010|c3_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|3|1.02|0'
  '011|c3_soft_tier_q3_a1p05|soft_tier|rhalf-soft-tier-q3|3|1.05|0'
  '012|c3_soft_tier_q3_a1p10|soft_tier|rhalf-soft-tier-q3|3|1.10|1'
  '013|c4_hashslot_q1|hashslot|rhalf-hash-slot-1|4|NA|0'
  '014|c4_distance_first_q3|distance_first|rhalf-distance-first-q3|4|NA|0'
  '015|c4_hard_tier_q3|hard_tier|rhalf-hard-tier-q3|4|NA|0'
  '016|c4_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|4|1.02|0'
  '017|c4_soft_tier_q3_a1p05|soft_tier|rhalf-soft-tier-q3|4|1.05|0'
  '018|c4_soft_tier_q3_a1p10|soft_tier|rhalf-soft-tier-q3|4|1.10|1'
)
# END FROZEN MATRIX

if [[ $SWEEP_VARIANT == extension ]]; then
  MATRIX=(
    '001|plain|plain|off|NA|NA|0'
    '002|c1_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|1|1.02|0'
    '003|c5_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|5|1.02|0'
    '004|c6_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|6|1.02|0'
    '005|c8_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|8|1.02|0'
    '006|c10_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|10|1.02|0'
  )
fi

BUILD_BIN="$REPO/target/release/build_shear_index"
SEARCH_BIN="$REPO/target/release/search_shear_index"
RABITQ_BUILDER="$REPO/target/release/build_graph_rabitq_1bit"
GRAPH_QUALITY_BIN="$REPO/target/release/graph_quality"
GPU_MONITOR="$REPO/script/monitor_gpu_build.sh"
if [[ $SWEEP_VARIANT == extension ]]; then
  SUMMARIZER="$REPO/script/summarize_wiki10m_softtier_row_cap_extension.py"
else
  SUMMARIZER="$REPO/script/summarize_wiki10m_softtier_row_cap_sweep.py"
fi
CARGO_BIN=${CARGO_BIN:-cargo}

usage() {
  cat <<'USAGE'
Usage: bash script/run_wiki10m_softtier_row_cap_sweep.sh MODE

Modes:
  help       Show this contract.
  smoke      Execute 18 synthetic transactions under OUTPUT_BASE/RUN_ID.
  prepare    Build the required release binaries.
  preflight  Validate inputs, sidecar manifest, binaries, locks, GPU, and space.
  formal     Execute the frozen sequential Wiki10M sweep; RUN_ID is required.
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
  local destination=$1 row ordinal label method policy cap alpha diagnostics
  printf 'ordinal\tlabel\tmethod\tpolicy_mode\trow_cap\tpriority_alpha\tdiagnostics\n' \
    >"$destination"
  for row in "${MATRIX[@]}"; do
    IFS='|' read -r ordinal label method policy cap alpha diagnostics <<<"$row"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ordinal" "$label" "$method" "$policy" "$cap" "$alpha" "$diagnostics" \
      >>"$destination"
  done
}

write_smoke_build_command() {
  local case_dir=$1 policy=$2 cap=$3 alpha=$4 diagnostics=$5
  {
    printf '%q ' "$BUILD_BIN" --max-degree 64 --lbuild 100 --num-threads 8 \
      --m-hash-bits 12 --fanout-top 8 --fanout-second 3 \
      --root-risk-candidate-rs 1,2,4,8 --seed 42 \
      --hashprune-backfill-mode "$policy"
    if [[ $policy != off ]]; then
      printf '%q ' --hashprune-rhalf-max-backfill-per-row "$cap"
    fi
    if [[ $policy == rhalf-soft-tier-q3 && $diagnostics == 0 ]]; then
      printf '%q ' --hashprune-soft-tier-priority-alpha "$alpha"
    fi
    if [[ $diagnostics == 1 ]]; then
      printf '%q ' --hashprune-backfill-diagnostics-prefix "$case_dir/diagnostics"
    fi
    printf '\n'
  } >"$case_dir/build_command.sh"
}

record_smoke_call() {
  local case_dir=$1 kind=$2 pass=$3 l_value=$4
  printf '%s\t%s\t%s\n' "$kind" "$pass" "$l_value" >>"$case_dir/search_calls.tsv"
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
  local row=$1 ordinal label method policy cap alpha diagnostics case_dir l_value
  IFS='|' read -r ordinal label method policy cap alpha diagnostics <<<"$row"
  case_dir="$BATCH_ROOT/transactions/${ordinal}_${label}"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar,search}
  atomic_status "$case_dir/STATUS" running
  printf '1\n' >"$case_dir/attempt_count"
  printf 'synthetic graph for %s\n' "$label" >"$case_dir/index/_mem.index"
  if [[ $policy == off ]]; then
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":null,"priority_alpha":null,"diagnostics":%s}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" "$diagnostics" \
      >"$case_dir/parameter_passport.json"
  else
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":%s,"priority_alpha":"%s","diagnostics":%s}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" "$cap" "$alpha" "$diagnostics" \
      >"$case_dir/parameter_passport.json"
  fi
  write_smoke_build_command "$case_dir" "$policy" "$cap" "$alpha" "$diagnostics"
  printf 'kind\tpass\tL\n' >"$case_dir/search_calls.tsv"
  record_smoke_call "$case_dir" warmup warmup 200
  for l_value in 10 16 25 32 50 100 200; do
    record_smoke_call "$case_dir" measured measured "$l_value"
  done
  if [[ ${SWEEP_SMOKE_FAIL_LABEL:-} == "$label" ]]; then
    printf 'forced synthetic failure\n' >"$case_dir/failure.txt"
    atomic_status "$case_dir/STATUS" failed
    return 1
  fi
  printf '{"label":"%s","graph":{"graph_bytes":1800000000,"avg_out_degree":48.0}}\n' \
    "$label" >"$case_dir/accepted_metrics.json"
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
  printf '{"schema":"wiki10m_softtier_row_cap_smoke_v1","variant":"%s","accepted":%s,"graphs":%s,"warmups":%s,"measured_rows":%s,"search_calls":%s}\n' \
    "$SWEEP_VARIANT" "$graph_count" "$graph_count" "$graph_count" "$measured_count" \
    "$((graph_count + measured_count))" >"$BATCH_ROOT/experiment_manifest.json"
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
  flock -n 8 || die "Wiki10M SoftTier row-cap sweep lock is held"
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
  python3 - "$BASE" "$QUERY" "$GT" "$RABITQ1_SIDECAR" "$RABITQ1_MANIFEST" \
    "$ROWS" "$DIM" <<'PY_INPUTS'
import json
import math
import os
import struct
import sys

base, query, gt, sidecar, manifest_path, rows, dim = sys.argv[1:]
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
manifest = json.load(open(manifest_path))
expected = {
    "schema": "graph_rabitq1_manifest_v1",
    "rows": rows,
    "dim": dim,
    "padded_dim": dim,
    "ex_data_bytes": 0,
    "sidecar_bytes": os.path.getsize(sidecar),
}
checks = {key: manifest.get(key) == value for key, value in expected.items()}
if not all(checks.values()):
    raise SystemExit(f"RaBitQ1 manifest mismatch: {checks}")
PY_INPUTS
  "$RABITQ_BUILDER" --data-path "$BASE" --output "$RABITQ1_SIDECAR" \
    --validate-only >/dev/null
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l) -eq 0 ]] || \
    die "GPU 0 is not idle"
  sudo -n true || die "passwordless sudo is required for controlled page-cache drops"
  free_bytes=$(df -PB1 "$OUTPUT_BASE" | awk 'NR==2 {print $4}')
  (( free_bytes >= minimum_free )) || die "less than 100 GiB free below OUTPUT_BASE"
  printf 'RaBitQ1 sidecar sha256=%s\n' "$(sha256sum "$RABITQ1_SIDECAR" | awk '{print $1}')"
}

run_preflight() {
  acquire_locks
  validate_inputs
  printf 'preflight accepted: Wiki10M inputs, GT, sidecar manifest, binaries, GPU, locks, and space\n'
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
  local profile=$1 policy=$2 cap=$3 alpha=$4 diagnostics=$5 output=$6
  python3 - "$profile" "$policy" "$cap" "$alpha" "$diagnostics" "$output" <<'PY_PROFILE'
import json
import math
import os
import sys

profile_path, policy, cap, alpha, diagnostics, output = sys.argv[1:]
diagnostics = int(diagnostics)
backfill = json.load(open(profile_path)).get("hashprune_backfill")
if not isinstance(backfill, dict):
    raise SystemExit("missing hashprune_backfill profile")
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
    if policy == "rhalf-soft-tier-q3" and not diagnostics:
        checks["requested_priority_alpha"] = math.isclose(
            float(backfill.get("requested_priority_alpha")), expected_alpha
        )
    else:
        checks["requested_priority_alpha_default"] = (
            backfill.get("requested_priority_alpha") is None
        )
if not all(checks.values()):
    raise SystemExit(f"HashPrune profile mismatch: {checks}")
value = {"accepted": True, "checks": checks, "backfill": backfill}
temporary = output + ".partial"
with open(temporary, "w") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
PY_PROFILE
}

validate_diagnostics() {
  local case_dir=$1 cap=$2 diagnostics=$3
  [[ $diagnostics == 1 ]] || return 0
  python3 - "$case_dir/diagnostics" "$cap" <<'PY_DIAGNOSTICS'
import json
from pathlib import Path
import sys

root, cap = Path(sys.argv[1]), int(sys.argv[2])
required = (
    "construction_metrics.json",
    "counterfactual_policy_summary.tsv",
    "alpha_sensitivity.tsv",
    "distance_layer_histograms.tsv",
    "primary_count.sidecar",
    "primary_count_manifest.json",
)
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"missing diagnostics: {missing}")
metrics = json.load(open(root / "construction_metrics.json"))
checks = {
    "requested_cap": metrics.get("requested_max_backfill_per_row") == cap,
    "effective_cap": metrics.get("effective_max_backfill_per_row") == cap,
    "capacity_rule": metrics.get("capacity_rule")
    == "min-derived-budget-row-cap-spare",
    "alpha_102": any(
        line.startswith("1.02\t")
        for line in (root / "alpha_sensitivity.tsv").read_text().splitlines()
    ),
}
if not all(checks.values()):
    raise SystemExit(f"diagnostic mismatch: {checks}")
PY_DIAGNOSTICS
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
  local case_dir=$1 policy=$2 cap=$3 alpha=$4 diagnostics=$5
  local alpha_args=() cap_args=() diagnostic_args=()
  if [[ $policy != off ]]; then
    cap_args=(--hashprune-rhalf-max-backfill-per-row "$cap")
  fi
  if [[ $policy == rhalf-soft-tier-q3 && $diagnostics == 0 ]]; then
    alpha_args=(--hashprune-soft-tier-priority-alpha "$alpha")
  fi
  if [[ $diagnostics == 1 ]]; then
    diagnostic_args=(--hashprune-backfill-diagnostics-prefix "$case_dir/diagnostics")
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
    --fanout-top 8 --fanout-second 3 --root-risk-candidate-rs 1,2,4,8 --seed 42
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
    "${cap_args[@]}" "${alpha_args[@]}" "${diagnostic_args[@]}"
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
  local row=$1 ordinal label method policy cap alpha diagnostics case_dir graph
  IFS='|' read -r ordinal label method policy cap alpha diagnostics <<<"$row"
  case_dir="$BATCH_ROOT/transactions/${ordinal}_${label}"
  mkdir -p "$case_dir"/{index,oom,sq_sidecar,search}
  atomic_status "$case_dir/STATUS" running
  printf '1\n' >"$case_dir/attempt_count"
  if [[ $policy == off ]]; then
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":null,"priority_alpha":null,"diagnostics":%s}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" "$diagnostics" \
      >"$case_dir/parameter_passport.json"
  else
    printf '{"ordinal":%s,"label":"%s","method":"%s","policy_mode":"%s","row_cap":%s,"priority_alpha":"%s","diagnostics":%s}\n' \
      "$((10#$ordinal))" "$label" "$method" "$policy" "$cap" "$alpha" "$diagnostics" \
      >"$case_dir/parameter_passport.json"
  fi
  run_build "$case_dir" "$policy" "$cap" "$alpha" "$diagnostics"
  graph="$case_dir/index/_mem.index"
  validate_graph_strict "$graph" "$case_dir/graph_validation.json"
  sha256sum "$graph" >"$case_dir/graph.sha256"
  "$GRAPH_QUALITY_BIN" --index-path-prefix "$case_dir/index" --data-path "$BASE" \
    --output "$case_dir/graph_quality.tsv" --max-edge-samples 1000000 \
    >"$case_dir/graph_quality.stdout.log" 2>"$case_dir/graph_quality.stderr.log"
  validate_profile "$case_dir/profile.json" "$policy" "$cap" "$alpha" "$diagnostics" \
    "$case_dir/policy_validation.json"
  validate_diagnostics "$case_dir" "$cap" "$diagnostics"
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
  printf '{"schema":"wiki10m_softtier_row_cap_formal_v1","variant":"%s","graphs":%s,"warmups":%s,"measured_rows":%s,"single_run_qps":true}\n' \
    "$SWEEP_VARIANT" "$graph_count" "$graph_count" "$measured_count" \
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
  smoke) run_smoke ;;
  prepare) prepare_binaries ;;
  preflight) run_preflight ;;
  formal) run_formal ;;
  summarize) run_summarize ;;
  *) usage; die "unknown mode: $MODE" ;;
esac
