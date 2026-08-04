#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUVS_SOURCE_ROOT="${CUVS_SOURCE_ROOT:-$REPO_ROOT/../cuvs_tagore_ace}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data_staging/wbh/laion35m_ace_disk_motivation}"
BASE="${BASE:-/data_staging/wbh/laion-512-35M/base.fbin}"
CANONICAL_BASE="${CANONICAL_BASE:-/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin}"
CUVS_PYTHON="${CUVS_PYTHON:-/data_staging/wbh/cuvs_tagore_ace_venv/bin/python}"
CUVS_INSTALL_ROOT="${CUVS_INSTALL_ROOT:-/data_staging/wbh/shear_paper_evaluation/build/cuvs_release_shared_ace/install}"
CUVS_SITE_PACKAGES="$("$CUVS_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUVS_RUNTIME_LIBRARY_PATH="$CUVS_INSTALL_ROOT/lib:$CUVS_SITE_PACKAGES/librmm/lib64:$CUVS_SITE_PACKAGES/libraft/lib64:$CUVS_SITE_PACKAGES/rapids_logger/lib64:/usr/local/cuda-13.0/lib64"
DRIVER="$REPO_ROOT/script/run_laion35m_ace_disk_motivation.py"
ANALYZER="$REPO_ROOT/script/analyze_laion35m_ace_nsys.py"
FORMAL_ROWS=35000000
FORMAL_DIM=512
FORMAL_PAYLOAD_BYTES=71680000000
GRAPH_DEGREE=64
INTERMEDIATE_GRAPH_DEGREE=128
EF_CONSTRUCTION=176
FORMAL_PARTITIONS=40
MONITOR_PIDS=()

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

die() {
  log "error: $*"
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

stop_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  MONITOR_PIDS=()
}
trap stop_monitors EXIT INT TERM

basic_preflight() {
  local command_name
  for command_name in python3 cmp swapon nvidia-smi nsys pidstat vmstat iostat \
    ps du df dd find ldd readlink sha256sum git; do
    require_command "$command_name"
  done
  [[ -x "$CUVS_PYTHON" ]] || die "missing cuVS Python: $CUVS_PYTHON"
  [[ -x /usr/bin/time ]] || die "missing /usr/bin/time"
  [[ -f "$DRIVER" && -f "$ANALYZER" ]] || die "missing driver or analyzer"
  [[ -z "$(swapon --show --noheadings)" ]] || die "swap must be disabled"
  [[ "$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)" == "0" ]] || die "SwapTotal is not zero"

  local active_gpu_pids
  active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ -z "$active_gpu_pids" ]] || die "GPU has active compute processes: $active_gpu_pids"

  local free_hbm_mib
  free_hbm_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$free_hbm_mib" =~ ^[0-9]+$ ]] || die "cannot read free HBM"
  (( free_hbm_mib >= 23000 )) || die "less than 23,000 MiB HBM is free"

  local extension resolved_cpp resolved_c
  extension="$(find "$CUVS_SITE_PACKAGES/cuvs/neighbors/cagra" -maxdepth 1 \
    -type f -name 'cagra*.so' -print -quit)"
  [[ -n "$extension" ]] || die "missing cuVS CAGRA Python extension"
  resolved_cpp="$(env LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" ldd "$extension" |
    awk '$1 == "libcuvs.so" {print $3; exit}')"
  resolved_c="$(env LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" ldd "$extension" |
    awk '$1 == "libcuvs_c.so" {print $3; exit}')"
  [[ "$(readlink -f "$resolved_cpp")" == "$(readlink -f "$CUVS_INSTALL_ROOT/lib/libcuvs.so")" ]] ||
    die "Python extension resolves the wrong libcuvs.so"
  [[ "$(readlink -f "$resolved_c")" == "$(readlink -f "$CUVS_INSTALL_ROOT/lib/libcuvs_c.so")" ]] ||
    die "Python extension resolves the wrong libcuvs_c.so"
  env TAGORE_CUVS_DISABLE_MG_IMPORT=1 LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    PYTHONPATH="$REPO_ROOT" "$CUVS_PYTHON" -c \
    'from script.run_laion35m_ace_disk_motivation import _import_cagra; _import_cagra()'
}

formal_preflight() {
  basic_preflight
  [[ -f "$BASE" ]] || die "missing LAION35M base: $BASE"
  [[ -f "$CANONICAL_BASE" ]] || die "missing canonical LAION100M base: $CANONICAL_BASE"
  PYTHONPATH="$REPO_ROOT" python3 - "$BASE" "$FORMAL_ROWS" "$FORMAL_DIM" <<'PY'
import sys
from pathlib import Path
from script.run_laion35m_ace_disk_motivation import read_fbin_shape
shape = read_fbin_shape(Path(sys.argv[1]))
expected = (int(sys.argv[2]), int(sys.argv[3]))
if shape != expected:
    raise SystemExit(f"LAION35M shape mismatch: {shape} != {expected}")
PY
  log "verifying that the existing LAION35M payload equals the canonical 100M prefix"
  cmp --ignore-initial=8:8 --bytes=71680000000 "$BASE" "$CANONICAL_BASE"

  mkdir -p "$OUTPUT_ROOT"
  local free_bytes
  free_bytes="$(df -PB1 "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
  (( free_bytes >= 220 * 1024 * 1024 * 1024 )) ||
    die "formal run requires at least 220 GiB free under $OUTPUT_ROOT"
  log "preflight passed: LAION35M=$FORMAL_ROWS x $FORMAL_DIM, HBM=${free_hbm_mib:-checked}, swap=off"
}

start_monitors() {
  local case_root="$1"
  mkdir -p "$case_root/monitors"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm \
    --format=csv -l 1 >"$case_root/monitors/gpu.csv" 2>"$case_root/monitors/gpu.err" &
  MONITOR_PIDS+=("$!")
  vmstat -t 1 >"$case_root/monitors/vmstat.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  iostat -dx 1 >"$case_root/monitors/iostat.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  pidstat -r -d -u -h -G 'python|nsys|time' 1 \
    >"$case_root/monitors/pidstat.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  (
    while :; do
      date --iso-8601=ns
      ps -eo pid,ppid,rss,pcpu,comm,args --sort=-rss |
        awk 'NR == 1 || /nsys|run_laion35m_ace_disk_motivation.py|cuvs_tagore_ace_venv/'
      sleep 1
    done
  ) >"$case_root/monitors/process_tree_rss.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  (
    while :; do
      printf '%s\t' "$(date --iso-8601=ns)"
      du -sb "$case_root/output" 2>/dev/null || printf '0\t%s\n' "$case_root/output"
      sleep 1
    done
  ) >"$case_root/monitors/temp_bytes.log" 2>&1 &
  MONITOR_PIDS+=("$!")
}

prewarm_input() {
  local input="$1" case_root="$2"
  log "prewarming $input outside the measured build"
  /usr/bin/time -v -o "$case_root/prewarm_time.log" \
    dd if="$input" of=/dev/null bs=16M iflag=fullblock status=progress \
    >"$case_root/prewarm_stdout.log" 2>"$case_root/prewarm_stderr.log"
}

run_case() {
  local run_root="$1" case_name="$2" input="$3" npartitions="$4" use_disk="$5"
  local partition_plan_mode="$6" partition_plan_dir="$7"
  local case_root="$run_root/$case_name"
  [[ ! -e "$case_root" ]] || die "case already exists: $case_root"
  mkdir -p "$case_root"
  prewarm_input "$input" "$case_root"
  cp /proc/meminfo "$case_root/meminfo.before"
  nvidia-smi -q >"$case_root/nvidia-smi.before.txt"

  local -a driver_command=(
    "$CUVS_PYTHON" "$DRIVER" build
    --input-fbin "$input"
    --output-dir "$case_root/output"
    --profile-json "$case_root/driver_profile.json"
    --generated-artifacts "$case_root/generated_artifacts.txt"
    --graph-degree "$GRAPH_DEGREE"
    --intermediate-graph-degree "$INTERMEDIATE_GRAPH_DEGREE"
    --ef-construction "$EF_CONSTRUCTION"
    --npartitions "$npartitions"
    --device 0
    --partition-plan-mode "$partition_plan_mode"
    --partition-plan-dir "$partition_plan_dir"
  )
  if [[ "$use_disk" == "true" ]]; then
    driver_command+=(--use-disk)
  fi
  {
    printf 'case=%s\n' "$case_name"
    printf 'prewarm_excluded=true\n'
    printf 'command='
    printf '%q ' env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
      "LD_LIBRARY_PATH=$CUVS_RUNTIME_LIBRARY_PATH" "${driver_command[@]}"
    printf '\n'
  } >"$case_root/command.txt"

  start_monitors "$case_root"
  log "starting $case_name (partitions=$npartitions, use_disk=$use_disk, partition_plan_mode=$partition_plan_mode)"
  set +e
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 TAGORE_CUVS_DISABLE_MG_IMPORT=1 \
    LD_LIBRARY_PATH="$CUVS_RUNTIME_LIBRARY_PATH" \
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --force-overwrite=false --output="$case_root/trace" \
      /usr/bin/time -v -o "$case_root/time.log" \
      "${driver_command[@]}" \
      >"$case_root/stdout.log" 2>"$case_root/stderr.log"
  local exit_code=$?
  set -e
  stop_monitors
  printf '%s\n' "$exit_code" >"$case_root/exit_code.txt"
  cp /proc/meminfo "$case_root/meminfo.after"
  nvidia-smi -q >"$case_root/nvidia-smi.after.txt"
  (( exit_code == 0 )) || die "$case_name failed with exit code $exit_code"
  [[ -f "$case_root/trace.nsys-rep" ]] || die "$case_name produced no Nsight report"
  [[ -f "$case_root/driver_profile.json" ]] || die "$case_name produced no profile"
  if [[ "$partition_plan_mode" == "assignment_create" ]]; then
    [[ -f "$partition_plan_dir/partition_assignment.bin" ]] ||
      die "$case_name did not create the frozen partition assignment"
  fi

  if [[ "$use_disk" == "true" ]]; then
    grep -Fq "disk mode is forced. Using disk-mode" "$case_root/stderr.log" ||
      die "$case_name did not enter forced disk mode"
  else
    grep -Fq "Using in-memory mode." "$case_root/stderr.log" ||
      die "$case_name did not remain in in-memory mode"
  fi

  nsys export --type=sqlite --force-overwrite=true \
    --output="$case_root/trace.sqlite" "$case_root/trace.nsys-rep" \
    >"$case_root/nsys_export.log" 2>&1
  python3 "$ANALYZER" --sqlite "$case_root/trace.sqlite" \
    --output "$case_root/timeline.json"
  log "completed $case_name"
}

record_provenance() {
  local run_root="$1"
  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'base=%s\n' "$BASE"
    printf 'canonical_base=%s\n' "$CANONICAL_BASE"
    printf 'rows=%s\n' "$FORMAL_ROWS"
    printf 'dim=%s\n' "$FORMAL_DIM"
    printf 'graph_degree=%s\n' "$GRAPH_DEGREE"
    printf 'intermediate_graph_degree=%s\n' "$INTERMEDIATE_GRAPH_DEGREE"
    printf 'ef_construction=%s\n' "$EF_CONSTRUCTION"
    printf 'npartitions=%s\n' "$FORMAL_PARTITIONS"
    printf 'local_builder=cagra\n'
    printf 'cagra_local_solver=nn_descent\n'
    printf 'execution_mode=pipelined\n'
    printf 'OMP_NUM_THREADS=8\n'
    printf 'BLAS_threads=1\n'
    nsys --version
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  } >"$run_root/provenance.txt"
  sha256sum "$DRIVER" "$ANALYZER" "$CUVS_INSTALL_ROOT/lib/libcuvs.so" \
    >"$run_root/artifact_sha256.txt"
  git -C "$REPO_ROOT" status --short >"$run_root/shear_git_status.txt"
  git -C "$CUVS_SOURCE_ROOT" status --short \
    >"$run_root/cuvs_git_status.txt"
}

summarize_run() {
  local run_root="$1" allow_nonformal="${2:-false}"
  local -a command=(
    python3 "$DRIVER" summarize --run-root "$run_root" --report-dir "$run_root/report"
  )
  if [[ "$allow_nonformal" == "true" ]]; then
    command+=(--allow-nonformal)
  fi
  PYTHONPATH="$REPO_ROOT" "${command[@]}"
  sync "$run_root/report/summary.json" "$run_root/report/report.md" \
    "$run_root/report/partition_timeline.tsv"
}

cleanup_run() {
  local run_root="$1" manifest artifact resolved allowed_prefix
  [[ -d "$run_root" ]] || die "run root does not exist: $run_root"
  : >"$run_root/cleanup.log"
  for manifest in "$run_root"/ace_*/generated_artifacts.txt; do
    [[ -f "$manifest" ]] || continue
    while IFS= read -r artifact; do
      [[ -n "$artifact" ]] || continue
      resolved="$(readlink -f "$artifact")"
      allowed_prefix="$(readlink -f "$run_root")/"
      [[ "$resolved" == "$allowed_prefix"* ]] ||
        die "refusing cleanup outside run root: $resolved"
      if [[ -f "$resolved" ]]; then
        stat -Lc 'remove bytes=%s path=%n' "$resolved" >>"$run_root/cleanup.log"
        rm -f -- "$resolved"
      fi
    done <"$manifest"
  done
  log "registered generated graph payloads cleaned under $run_root"
}

run_smoke() {
  basic_preflight
  local run_id="${RUN_ID:-laion35m-ace-disk-smoke-$(date +%Y%m%d-%H%M%S)}"
  local run_root="$OUTPUT_ROOT/$run_id"
  [[ ! -e "$run_root" ]] || die "run ID already exists: $run_id"
  mkdir -p "$run_root"
  local smoke_input="$run_root/input_8192x512.fbin"
  local partition_plan_dir="$run_root/partition_assignment"
  mkdir -p "$partition_plan_dir"
  python3 - "$smoke_input" <<'PY'
import struct
import sys
from pathlib import Path
import numpy as np
path = Path(sys.argv[1])
data = np.random.default_rng(20260727).standard_normal((8192, 512), dtype=np.float32)
with path.open("wb") as stream:
    stream.write(struct.pack("<ii", *data.shape))
    data.tofile(stream)
PY
  run_case "$run_root" ace_non_disk "$smoke_input" 4 false \
    assignment_create "$partition_plan_dir"
  run_case "$run_root" ace_disk "$smoke_input" 4 true \
    assignment_replay "$partition_plan_dir"
  summarize_run "$run_root" true
  cleanup_run "$run_root"
  printf '%s\n' "$run_id" >"$OUTPUT_ROOT/latest_smoke_run_id.txt"
  log "smoke accepted: $run_root"
}

run_formal() {
  formal_preflight
  local run_id="${RUN_ID:-laion35m-ace-disk-motivation-$(date +%Y%m%d-%H%M%S)}"
  local run_root="$OUTPUT_ROOT/$run_id"
  [[ ! -e "$run_root" ]] || die "run ID already exists: $run_id"
  mkdir -p "$run_root"
  local partition_plan_dir="$run_root/partition_assignment"
  mkdir -p "$partition_plan_dir"
  record_provenance "$run_root"
  printf '%s\n' "$run_id" >"$OUTPUT_ROOT/latest_run_id.txt"
  run_case "$run_root" ace_non_disk "$BASE" "$FORMAL_PARTITIONS" false \
    assignment_create "$partition_plan_dir"
  run_case "$run_root" ace_disk "$BASE" "$FORMAL_PARTITIONS" true \
    assignment_replay "$partition_plan_dir"
  summarize_run "$run_root" false
  cleanup_run "$run_root"
  log "formal comparison accepted: $run_root"
}

resolve_run_root() {
  local run_id="${RUN_ID:-}"
  if [[ -z "$run_id" && -f "$OUTPUT_ROOT/latest_run_id.txt" ]]; then
    run_id="$(<"$OUTPUT_ROOT/latest_run_id.txt")"
  fi
  [[ -n "$run_id" ]] || die "set RUN_ID or create a formal run first"
  printf '%s/%s\n' "$OUTPUT_ROOT" "$run_id"
}

usage() {
  printf 'usage: %s {prepare|preflight|smoke|formal|summarize|cleanup}\n' "$0" >&2
}

case "${1:-}" in
  prepare)
    formal_preflight
    python3 -m unittest -v \
      script/test_analyze_laion35m_ace_nsys.py \
      script/test_laion35m_ace_disk_motivation.py
    ;;
  preflight)
    formal_preflight
    ;;
  smoke)
    run_smoke
    ;;
  formal)
    run_formal
    ;;
  summarize)
    summarize_run "$(resolve_run_root)" false
    ;;
  cleanup)
    cleanup_run "$(resolve_run_root)"
    ;;
  *)
    usage
    exit 2
    ;;
esac
