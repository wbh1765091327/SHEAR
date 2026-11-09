#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_env.sh"
set -Eeuo pipefail
shopt -s inherit_errexit
export LC_ALL=C

MODE="${1:-help}"
readonly EXPECTED_BASE_BYTES=204800000008
readonly PAYLOAD_BYTES=204800000000
readonly SIDECAR_BYTES=52000000000
CAP_ORDER=(p50 p40 p30 p20)
declare -A CAP_BYTES=(
  [p50]=102400000000
  [p40]=81920000000
  [p30]=61440000000
  [p20]=40960000000
)
readonly CAP_ORDER CAP_BYTES

SHEAR_ROOT="${SHEAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
DATA_ROOT="${DATA_ROOT:-${SHEAR_DATA_ROOT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${SHEAR_OUTPUT_ROOT}/laion100m_shear_memory_scaling/runs}"
BASE="${BASE:-${SHEAR_DATA_ROOT}/laion-512-100M/base.fbin}"
BATCH_ROOT="${BATCH_ROOT:-}"
LOCK_PATH="${LOCK_PATH:-${SHEAR_OUTPUT_ROOT}/.laion100m_shear_memory_scaling.lock}"
ACTIVE_MONITOR_PIDS=()
EXPERIMENT_LOCK_FD=""
SELF_TEST_MODE=0

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

die() {
  log "ERROR: $*"
  return 1
}

usage() {
  cat <<'USAGE'
Usage: laion100m_shear_memory_scaling.sh MODE [BATCH_ROOT]

Modes:
  help        Show this frozen construction-only contract.
  preflight   Validate LAION100M, cgroup v2, swap, binaries, GPU, and space.
  self-test   Exercise cgroup success/OOM classification without using the dataset or GPU.
  formal      Run SHEAR at 50%, 40%, 30%, and 20% memory caps in that order.
  summarize   Rebuild summary.tsv and summary.md for BATCH_ROOT.

This runner does not run search. The formal mode is the only mode that starts
LAION100M construction and requires separate explicit approval.
USAGE
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "required executable is missing: $1"
}

require_commands() {
  local name
  for name in "$@"; do
    command -v "$name" >/dev/null 2>&1 || die "required command is missing: $name"
  done
}

fresh_dir() {
  [[ ! -e "$1" ]] || die "refusing to overwrite existing path: $1"
  mkdir -p "$1"
}

write_status() {
  local path="$1" state="$2" exit_code="$3" message="${4:-}"
  python3 - "$path" "$state" "$exit_code" "$message" <<'PY_STATUS'
import datetime
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
event = {
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "state": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "message": sys.argv[4] or None,
}
if path.exists():
    value = json.loads(path.read_text())
else:
    value = {
        "schema": "laion100m_shear_memory_scaling_status_v1",
        "history": [],
    }
value.update(event)
value["history"].append(event)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY_STATUS
}

validate_base_contract() {
  python3 - "$BASE" "$EXPECTED_BASE_BYTES" "$PAYLOAD_BYTES" \
    "${CAP_BYTES[p50]}" "${CAP_BYTES[p40]}" "${CAP_BYTES[p30]}" "${CAP_BYTES[p20]}" <<'PY_BASE'
from pathlib import Path
import struct
import sys

base = Path(sys.argv[1])
expected = int(sys.argv[2])
payload_bytes = int(sys.argv[3])
caps = dict(zip((50, 40, 30, 20), map(int, sys.argv[4:]), strict=True))
if not base.is_file():
    raise SystemExit(f"missing base: {base}")
if base.stat().st_size != expected:
    raise SystemExit(f"base bytes {base.stat().st_size} != {expected}")
with base.open("rb") as stream:
    raw = stream.read(8)
if len(raw) != 8:
    raise SystemExit(f"short base header: {base}")
header = struct.unpack("<II", raw)
if header != (100_000_000, 512):
    raise SystemExit(f"unexpected base header: {header}")
if payload_bytes != header[0] * header[1] * 4:
    raise SystemExit("payload byte contract does not match rows x dim x float32")
if expected != payload_bytes + 8:
    raise SystemExit("fbin size does not equal payload plus header")
for percent, cap in caps.items():
    theoretical = payload_bytes * percent // 100
    aligned = theoretical // 4096 * 4096
    if cap != aligned:
        raise SystemExit(
            f"p{percent} cap {cap} != lower page-aligned value {aligned}"
        )
    print(f"p{percent}_cap_bytes={cap}")
print(f"base_bytes={expected}")
print(f"payload_bytes={payload_bytes}")
PY_BASE
}

current_cgroup_dir() {
  local relative
  relative="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
  [[ -n "$relative" ]] || die "unable to resolve current cgroup v2 path"
  printf '/sys/fs/cgroup%s\n' "$relative"
}

verify_binary_cli_contract() {
  "$SHEAR_ROOT/target/release/build_shear_index" \
    --gpu-subtree-sq-staging-mode gpu-d0 \
    --gpu-subtree-sq-wmma-sidecar-mode rebuild \
    --gpu-subtree-sq8-d0-assignment-kernel wmma-batch \
    --hashprune-backfill-mode rhalf-soft-tier-q3 \
    --hashprune-rhalf-max-backfill-per-row 3 \
    --hashprune-soft-tier-priority-alpha 1.02 \
    --help >/dev/null
}

run_preflight() {
  require_commands bash python3 awk sed stat sha256sum systemd-run systemctl \
    journalctl vmstat iostat pidstat nvidia-smi ps pgrep flock sudo g++ realpath \
    git du find /usr/bin/time
  [[ "$(stat -fc %T /sys/fs/cgroup)" == cgroup2fs ]] || die "cgroup v2 is required"
  local current_cgroup_path
  current_cgroup_path="$(current_cgroup_dir)"
  [[ -f "$current_cgroup_path/memory.swap.max" ]] \
    || die "memory.swap.max is unavailable in the current cgroup subtree"
  sudo -n true || die "passwordless sudo is required"
  awk 'NR > 1 {active=1} END {exit active ? 1 : 0}' /proc/swaps \
    || die "host swap must be disabled"
  validate_base_contract
  require_executable "$SHEAR_ROOT/target/release/build_shear_index"
  require_executable "$SHEAR_ROOT/target/release/graph_quality"
  require_executable "$SHEAR_ROOT/script/monitor_gpu_build.sh"
  verify_binary_cli_contract

  local free_bytes active_gpu
  free_bytes="$(df -B1 --output=avail ${SHEAR_OUTPUT_ROOT} | tail -n1 | tr -d ' ')"
  ((free_bytes >= 350000000000)) \
    || die "at least 350 GB free staging space is required"
  active_gpu="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$active_gpu" ]] || die "GPU 0 has active compute processes: $active_gpu"
  if pgrep -x build_shear_index >/dev/null 2>&1; then
    die "another build_shear_index process is active"
  fi
  log "preflight passed: free_staging_bytes=$free_bytes"
}

drop_page_cache() {
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

memory_stat_value() {
  local path="$1" field="$2"
  awk -v wanted="$field" \
    '$1 == wanted {print $2; found=1} END {if (!found) print 0}' "$path"
}

io_stat_totals() {
  local path="$1"
  awk '
    { for (i=2; i<=NF; ++i) { split($i, pair, "="); sum[pair[1]] += pair[2] } }
    END { printf "%d %d %d %d\n", sum["rbytes"], sum["wbytes"], sum["rios"], sum["wios"] }
  ' "$path"
}

sample_cgroup() {
  local unit="$1" run_root="$2" cap_bytes="$3"
  local control_group="" cgroup_dir="" attempt active final_written=0
  for attempt in $(seq 1 200); do
    control_group="$(sudo -n systemctl show "$unit.service" -p ControlGroup \
      --value 2>/dev/null || true)"
    if [[ -n "$control_group" && -d "/sys/fs/cgroup$control_group" ]]; then
      cgroup_dir="/sys/fs/cgroup$control_group"
      break
    fi
    sleep 0.05
  done
  [[ -n "$cgroup_dir" ]] || {
    printf 'unable to resolve live cgroup for %s\n' "$unit" \
      >"$run_root/cgroup_sampler_error.txt"
    return 1
  }

  local actual_max actual_swap
  actual_max="$(<"$cgroup_dir/memory.max")"
  actual_swap="$(<"$cgroup_dir/memory.swap.max")"
  {
    printf 'control_group=%s\n' "$control_group"
    printf 'memory.max=%s\n' "$actual_max"
    printf 'memory.swap.max=%s\n' "$actual_swap"
  } >"$run_root/cgroup_limits.txt"
  [[ "$actual_max" == "$cap_bytes" ]] || return 2
  [[ "$actual_swap" == 0 ]] || return 3

  printf 'epoch\tmemory_current\tmemory_peak\tanon\tfile\tfile_dirty\tfile_writeback\tevent_high\tevent_max\tevent_oom\tevent_oom_kill\trbytes\twbytes\trios\twios\n' \
    >"$run_root/cgroup_samples.tsv"

  while [[ -d "$cgroup_dir" ]]; do
    local current peak anon file file_dirty file_writeback high maximum oom oom_kill
    local rbytes wbytes rios wios io_totals
    if ! current="$(cat "$cgroup_dir/memory.current" 2>/dev/null)"; then
      break
    fi
    if ! peak="$(cat "$cgroup_dir/memory.peak" 2>/dev/null)"; then
      break
    fi
    if ! anon="$(memory_stat_value "$cgroup_dir/memory.stat" anon 2>/dev/null)" \
      || ! file="$(memory_stat_value "$cgroup_dir/memory.stat" file 2>/dev/null)" \
      || ! file_dirty="$(memory_stat_value "$cgroup_dir/memory.stat" file_dirty 2>/dev/null)" \
      || ! file_writeback="$(memory_stat_value "$cgroup_dir/memory.stat" file_writeback 2>/dev/null)" \
      || ! high="$(memory_stat_value "$cgroup_dir/memory.events" high 2>/dev/null)" \
      || ! maximum="$(memory_stat_value "$cgroup_dir/memory.events" max 2>/dev/null)" \
      || ! oom="$(memory_stat_value "$cgroup_dir/memory.events" oom 2>/dev/null)" \
      || ! oom_kill="$(memory_stat_value "$cgroup_dir/memory.events" oom_kill 2>/dev/null)"; then
      break
    fi
    if ! io_totals="$(io_stat_totals "$cgroup_dir/io.stat" 2>/dev/null)"; then
      break
    fi
    read -r rbytes wbytes rios wios <<<"$io_totals"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%s.%N)" "$current" "$peak" "$anon" "$file" "$file_dirty" \
      "$file_writeback" "$high" "$maximum" "$oom" "$oom_kill" \
      "$rbytes" "$wbytes" "$rios" "$wios" >>"$run_root/cgroup_samples.tsv"

    active="$(sudo -n systemctl show "$unit.service" -p ActiveState \
      --value 2>/dev/null || true)"
    if [[ "$active" != active && "$active" != activating && "$active" != deactivating ]]; then
      if ((final_written)); then
        break
      fi
      final_written=1
    fi
    sleep 1
  done
}

start_host_monitors() {
  local run_root="$1"
  ((SELF_TEST_MODE == 0)) || return 0
  mkdir -p "$run_root/host_monitor"
  vmstat -w -t 1 >"$run_root/host_monitor/vmstat.log" 2>&1 &
  ACTIVE_MONITOR_PIDS+=("$!")
  iostat -dxz -y -t 1 >"$run_root/host_monitor/iostat.log" 2>&1 &
  ACTIVE_MONITOR_PIDS+=("$!")
  pidstat -r -d -u -h -p ALL 1 >"$run_root/host_monitor/pidstat.log" 2>&1 &
  ACTIVE_MONITOR_PIDS+=("$!")
}

stop_host_monitors() {
  local pid
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_MONITOR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_MONITOR_PIDS=()
}

sample_temp() {
  local stop_file="$1" payload="$2" output="$3"
  printf 'sample_epoch\tallocated_bytes\tapparent_bytes\n' >"$output"
  while [[ ! -e "$stop_file" ]]; do
    local allocated apparent
    allocated="$(du -s -B1 "$payload" 2>/dev/null | awk '{print $1}' || true)"
    apparent="$(du -sb "$payload" 2>/dev/null | awk '{print $1}' || true)"
    printf '%s\t%s\t%s\n' "$(date +%s.%N)" "${allocated:-0}" "${apparent:-0}" \
      >>"$output"
    sleep 1
  done
}

classify_service() {
  local run_root="$1" exit_code="$2"
  python3 - "$run_root" "$exit_code" <<'PY_CLASSIFY'
import csv
import json
from pathlib import Path
import re
import signal
import sys

root = Path(sys.argv[1])
launcher_exit_code = int(sys.argv[2])
samples = []
sample_path = root / "cgroup_samples.tsv"
if sample_path.is_file():
    with sample_path.open() as stream:
        samples = list(csv.DictReader(stream, delimiter="\t"))
oom = max((int(row["event_oom"]) for row in samples), default=0)
oom_kill = max((int(row["event_oom_kill"]) for row in samples), default=0)
journal_path = root / "cgroup_journal.log"
journal = journal_path.read_text(errors="replace") if journal_path.is_file() else ""
journal_oom = bool(re.search(r"oom-kill|out of memory|memory pressure", journal, re.I))
systemd = {}
show_path = root / "systemd_show.txt"
if show_path.is_file():
    for line in show_path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            systemd[key] = value
systemd_result = systemd.get("Result", "")
exec_main_code = systemd.get("ExecMainCode", "")
try:
    exec_main_status = int(systemd.get("ExecMainStatus", ""))
except ValueError:
    exec_main_status = None
run_output = ""
for name in ("systemd_run.stdout.log", "systemd_run.stderr.log"):
    path = root / name
    if path.is_file():
        run_output += path.read_text(errors="replace") + "\n"
result_match = re.search(r"Finished with result:\s*(\S+)", run_output)
main_match = re.search(
    r"Main processes terminated with:\s*code=(\w+)/status=(\w+)",
    run_output,
)
if result_match:
    systemd_result = result_match.group(1)
if main_match:
    exec_main_code = main_match.group(1)
    status_text = main_match.group(2)
    try:
        exec_main_status = int(status_text)
    except ValueError:
        try:
            exec_main_status = signal.Signals["SIG" + status_text].value
        except (KeyError, ValueError):
            exec_main_status = None
    if exec_main_code == "killed" and systemd_result == "success":
        systemd_result = "signal"
systemd_observed = bool(systemd_result or exec_main_code or exec_main_status is not None)
main_succeeded = (
    systemd_result == "success"
    and exec_main_code in {"0", "exited"}
    and exec_main_status == 0
)
if systemd_observed and exec_main_status is not None:
    if exec_main_code == "killed":
        exit_code = 128 + exec_main_status
    else:
        exit_code = exec_main_status
else:
    exit_code = launcher_exit_code
if oom_kill > 0 or journal_oom or systemd_result == "oom-kill":
    state = "oom"
elif (main_succeeded if systemd_observed else launcher_exit_code == 0):
    state = "completed"
else:
    state = "failed"
value = {
    "schema": "laion100m_shear_cgroup_service_v1",
    "state": state,
    "exit_code": exit_code,
    "launcher_exit_code": launcher_exit_code,
    "systemd_result": systemd_result,
    "exec_main_code": exec_main_code,
    "exec_main_status": exec_main_status,
    "event_oom": oom,
    "event_oom_kill": oom_kill,
    "journal_oom": journal_oom,
    "sample_count": len(samples),
}
(root / "service_result.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n"
)
(root / "service_exit_code.txt").write_text(f"{exit_code}\n")
print(state)
PY_CLASSIFY
}

summarize_resources() {
  local run_root="$1"
  python3 - "$run_root" <<'PY_RESOURCES'
import csv
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
payload = root / "payload"
time_text = (payload / "time.log").read_text(errors="replace") if (payload / "time.log").is_file() else ""
rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
gpu_text = (root / "gpu_monitor/summary.txt").read_text(errors="replace") \
    if (root / "gpu_monitor/summary.txt").is_file() else ""
hbm_match = re.search(r"^memory_used_max_mib=([0-9.]+)$", gpu_text, re.M)
temp_rows = []
if (root / "temp_usage.tsv").is_file():
    with (root / "temp_usage.tsv").open() as stream:
        temp_rows = list(csv.DictReader(stream, delimiter="\t"))
value = {
    "schema": "laion100m_shear_resource_summary_v1",
    "peak_process_tree_rss_bytes": int(rss_match.group(1)) * 1024 if rss_match else 0,
    "peak_hbm_bytes": int(float(hbm_match.group(1)) * 1024 * 1024) if hbm_match else 0,
    "peak_temp_allocated_bytes": max(
        (int(row["allocated_bytes"]) for row in temp_rows), default=0
    ),
    "peak_temp_apparent_bytes": max(
        (int(row["apparent_bytes"]) for row in temp_rows), default=0
    ),
}
(payload / "resource_summary.json").parent.mkdir(parents=True, exist_ok=True)
(payload / "resource_summary.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n"
)
PY_RESOURCES
}

launch_cgroup_service() {
  local unit="$1" run_root="$2" cap_bytes="$3"
  shift 3
  mkdir -p "$run_root"
  local stop_file="$run_root/.monitor_stop"
  rm -f "$stop_file"
  start_host_monitors "$run_root"
  sample_temp "$stop_file" "$run_root/payload" "$run_root/temp_usage.tsv" &
  local temp_pid=$!

  local -a service=(
    sudo -n systemd-run
    --unit="$unit"
    --wait
    --pipe
    --uid="$(id -un)"
    --gid="$(id -gn)"
    --working-directory="$SHEAR_ROOT"
    --property="MemoryMax=$cap_bytes"
    --property="MemorySwapMax=0"
    --property="OOMPolicy=stop"
    "$@"
  )
  set +e
  if ((SELF_TEST_MODE)); then
    "${service[@]}" >"$run_root/systemd_run.stdout.log" \
      2>"$run_root/systemd_run.stderr.log" &
  else
    "$SHEAR_ROOT/script/monitor_gpu_build.sh" \
      --out-dir "$run_root/gpu_monitor" --gpu 0 --interval 1 -- \
      "${service[@]}" >"$run_root/systemd_run.stdout.log" \
      2>"$run_root/systemd_run.stderr.log" &
  fi
  local launcher_pid=$!
  set -e
  sample_cgroup "$unit" "$run_root" "$cap_bytes" &
  local sampler_pid=$!

  set +e
  wait "$launcher_pid"
  local service_exit=$?
  wait "$sampler_pid"
  local sampler_exit=$?
  set -e
  touch "$stop_file"
  wait "$temp_pid" 2>/dev/null || true
  stop_host_monitors

  printf '%s\n' "$service_exit" >"$run_root/launcher_exit_code.txt"
  printf '%s\n' "$sampler_exit" >"$run_root/cgroup_sampler_exit_code.txt"
  sudo -n systemctl show "$unit.service" \
    -p Result -p ExecMainCode -p ExecMainStatus -p ActiveState -p SubState \
    -p MemoryCurrent -p MemoryPeak -p MemoryMax -p MemorySwapCurrent \
    >"$run_root/systemd_show.txt" 2>&1 || true
  sudo -n journalctl -u "$unit.service" --no-pager \
    >"$run_root/cgroup_journal.log" 2>&1 || true
  if ((sampler_exit != 0)); then
    printf 'sampler_exit=%s\n' "$sampler_exit" >"$run_root/runtime_error.txt"
    service_exit=1
  fi
  classify_service "$run_root" "$service_exit" >"$run_root/service_state.txt"
  summarize_resources "$run_root"
}

record_environment() {
  local cap_root="$1" label="$2" cap_bytes="$3"
  {
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'host=%s\n' "$(hostname)"
    printf 'cap_label=%s\n' "$label"
    printf 'cap_bytes=%s\n' "$cap_bytes"
    printf 'base=%s\n' "$BASE"
    printf 'base_bytes=%s\n' "$(stat -Lc %s "$BASE")"
    uname -a
    lscpu
    free -b
    cat /proc/meminfo
    nvidia-smi
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
    mount
    df -B1 ${SHEAR_OUTPUT_ROOT} ${SHEAR_DATA_ROOT}
  } >"$cap_root/environment.txt" 2>&1
}

record_provenance() {
  local root="$1"
  mkdir -p "$root"
  git -C "$SHEAR_ROOT" rev-parse HEAD >"$root/git_head.txt"
  git -C "$SHEAR_ROOT" status --short >"$root/git_status.txt"
  git -C "$SHEAR_ROOT" diff --binary >"$root/dirty.diff"
  sha256sum \
    "$SHEAR_ROOT/target/release/build_shear_index" \
    "$SHEAR_ROOT/target/release/graph_quality" \
    "$SHEAR_ROOT/script/laion100m_shear_memory_scaling.sh" \
    >"$root/sha256.txt"
}

write_shear_wrapper() {
  local wrapper="$1"
  cat >"$wrapper" <<'SHEAR_WRAPPER'
#!/usr/bin/env bash
set -Eeuo pipefail
payload="$1"
project_root="$2"
base="$3"
mkdir -p "$payload/index" "$payload/oom_temp" "$payload/sq_sidecar"
exec /usr/bin/time -v -o "$payload/time.log" \
  env \
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
  "$project_root/target/release/build_shear_index" \
    --data-type float --dist-fn l2 \
    --data-path "$base" --index-path-prefix "$payload/index" --overwrite \
    --oom-enable --oom-memory-budget-gb 0 --oom-io-mode direct \
    --oom-temp-dir "$payload/oom_temp" \
    --oom-profile-json "$payload/profile.json" \
    --driver-timing-json "$payload/driver_timing.json" \
    --max-degree 64 --lbuild 100 --num-threads 8 \
    --leaf-size-min 256 --leaf-size-max 1280 --leaf-knn 2 \
    --psamp-fraction 0.05 --fanout-top 8 --fanout-second 3 \
    --root-risk-candidate-rs 1,2,4,8 --seed 42 \
    --hashprune-backfill-mode rhalf-soft-tier-q3 \
    --hashprune-rhalf-max-backfill-per-row 3 \
    --hashprune-soft-tier-priority-alpha 1.02 \
    --hashprune-backfill-alpha 1.0 \
    --partitioner gpu-subtree \
    --gpu-subtree-root-assignment-backend cpu \
    --gpu-subtree-assignment-backend persistent-wmma-tf32 \
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32 \
    --gpu-subtree-execution-mode fused-persistent \
    --gpu-subtree-d0-capacity-enable \
    --gpu-subtree-d0-capacity-split-recursion-limit 8 \
    --gpu-subtree-d0-capacity-high-ratio 0.08 \
    --gpu-subtree-d0-capacity-target-ratio 0.08 \
    --gpu-subtree-d0-capacity-low-ratio 0.08 \
    --gpu-subtree-d0-partition-mode gpu-kmeans-coarse \
    --gpu-subtree-sq-staging-mode gpu-d0 \
    --gpu-subtree-sq-staging-upload-shard-rows 262144 \
    --gpu-subtree-sq-wmma-mode build \
    --gpu-subtree-sq-wmma-sidecar-prefix "$payload/sq_sidecar/sq_wmma" \
    --gpu-subtree-sq-wmma-sidecar-mode rebuild \
    --gpu-subtree-persistent-large-ticket-batch 8 \
    --gpu-subtree-sq8-assignment-kernel wmma \
    --gpu-subtree-sq8-d0-assignment-kernel wmma-batch \
    --gpu-subtree-sq8-leaf-mode auto \
    --gpu-subtree-persistent-scheduler-mode auto \
    --pipnn-backend cuda --pipnn-gpu-device 0 \
    --pipnn-gpu-workspace-budget-gb 16 \
    >"$payload/stdout.log" 2>"$payload/stderr.log"
test -s "$payload/index/_mem.index"
test -s "$payload/profile.json"
test -s "$payload/driver_timing.json"
SHEAR_WRAPPER
  chmod +x "$wrapper"
}

extract_shear_metrics() {
  local payload="$1" cap_bytes="$2"
  python3 - "$payload/profile.json" "$payload/shear_metrics.json" \
    "$cap_bytes" "$SIDECAR_BYTES" <<'PY_METRICS'
import json
import math
from pathlib import Path
import sys

profile_path = Path(sys.argv[1])
output = Path(sys.argv[2])
cap_bytes = int(sys.argv[3])
sidecar_bytes = int(sys.argv[4])
profile = json.loads(profile_path.read_text())

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

require(int(profile["num_points"]) == 100_000_000, "profile row mismatch")
require(int(profile["dim"]) == 512, "profile dim mismatch")
require(profile["memory_budget_source"] == "auto-host-cgroup", "budget source mismatch")
actual_bh = int(profile["memory_budget_bytes"])
require(0 < actual_bh <= cap_bytes * 80 // 100, "actual B_h exceeds cgroup policy")
require(profile["oom_io_mode"] == "direct", "OOM I/O mode mismatch")
require(profile["prune_backend"] == "hashprune", "prune backend mismatch")
backfill = profile["hashprune_backfill"]
require(backfill["mode"] == "rhalf-soft-tier-q3", "HashPrune mode mismatch")
require(int(backfill["effective_max_backfill_per_row"]) == 3, "HashPrune cap mismatch")
require(
    math.isclose(float(backfill["effective_priority_alpha"]), 1.02, abs_tol=1e-9),
    "HashPrune priority alpha mismatch",
)
gpu = profile["gpu_subtree"]
require(gpu["enabled"] is True, "GPU subtree disabled")
require(gpu["d0_partition_mode"] == "gpu-kmeans-coarse", "D0 mode mismatch")
require(gpu["sq_staging_mode"] == "gpu-d0", "SQ staging mismatch")
require(gpu["sq_wmma_sidecar_mode"] == "rebuild", "sidecar mode mismatch")
memory_plan = profile["memory_plan"]
require(memory_plan["sidecar_resident_kind"] == "sq_wmma", "resident kind mismatch")
resident_bytes = int(gpu["sq_wmma_sidecar_resident_bytes"])
require(0 <= resident_bytes <= sidecar_bytes, "invalid sidecar resident bytes")
sidecar_file_rows = int(gpu["sq_wmma_sidecar_file_rows"])
io = profile["gpu_subtree_io"]
serial = profile["serial_timing"]["rows"]
metrics = {
    "schema": "laion100m_shear_memory_metrics_v1",
    "actual_bh_bytes": actual_bh,
    "reservoir_resident_bytes": int(profile["resident_reservoir"]["bytes"]),
    "sidecar_theoretical_bytes": sidecar_bytes,
    "sidecar_resident_bytes": resident_bytes,
    "sidecar_resident_fraction": min(1.0, resident_bytes / sidecar_bytes),
    "sidecar_admitted_rows": int(memory_plan["sidecar_resident_admitted_rows"]),
    "sidecar_file_rows": sidecar_file_rows,
    "sidecar_file_logical_bytes": sidecar_file_rows * 520,
    "sidecar_file_gather_ms": int(gpu["sq_wmma_sidecar_file_gather_ms"]),
    "candidate_spill_bytes": int(io["candidate_spill_bytes"]),
    "candidate_spill_files": int(io["candidate_spill_files"]),
    "candidate_reread_bytes": None,
    "reduce_segments_scanned": int(io["reduce_segments_scanned"]),
    "reduce_part_files_opened": int(io["reduce_part_files_opened"]),
    "external_reduce_ms": int(serial["external_reduce_ms"]),
}
output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
PY_METRICS
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
#include <iomanip>
#include <iostream>
#include <limits>
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
  uint32_t observed_min_degree = std::numeric_limits<uint32_t>::max();
  uint32_t observed_max_degree = 0;
  std::array<uint32_t, 64> neighbors{};
  for (uint64_t node = 0; node < expected_nodes; ++node) {
    uint32_t degree = 0;
    if (!read_exact(in, degree)) return 5;
    if (degree == 0 || degree > allowed_degree || degree > neighbors.size()) {
      ++invalid_degree_nodes;
      if (degree > neighbors.size()) return 6;
    }
    observed_min_degree = std::min(observed_min_degree, degree);
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
  const double mean_degree = expected_nodes == 0
      ? 0.0 : static_cast<double>(edges) / static_cast<double>(expected_nodes);
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
      << ",\n  \"observed_min_degree\": " << observed_min_degree
      << ",\n  \"mean_degree\": " << std::fixed << std::setprecision(6) << mean_degree
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
  local index="$1" output="$2"
  python3 - "$index" "$output" <<'PY_INDEX_SIZE'
import json
from pathlib import Path
import sys

index = Path(sys.argv[1])
output = Path(sys.argv[2])
output.write_text(
    json.dumps(
        {
            "schema": "laion100m_shear_index_size_v1",
            "path": str(index),
            "bytes": index.stat().st_size,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY_INDEX_SIZE
}

validate_graph() {
  local payload="$1" validator
  local index="$payload/index/_mem.index"
  require_file "$index"
  validator="$(ensure_graph_validator)"
  "$validator" "$index" 100000000 64 "$payload/graph_validation.json"
  "$SHEAR_ROOT/target/release/graph_quality" \
    --index-path-prefix "$payload/index" \
    --data-path "$BASE" \
    --output "$payload/graph_quality.tsv" \
    --max-edge-samples 1000000 \
    >"$payload/graph_quality.log" 2>&1
  record_index_size "$index" "$payload/index_size_before_cleanup.json"
}

cleanup_successful_payload() {
  local payload="$1"
  local manifest="$payload/cleanup_manifest.tsv"
  local summary="$payload/cleanup_summary.txt"
  local -a targets=("$payload/index" "$payload/oom_temp" "$payload/sq_sidecar")
  local target resolved bytes total=0
  printf 'kind\tbytes\tpath\n' >"$manifest"
  for target in "${targets[@]}"; do
    resolved="$(realpath -m -- "$target")"
    [[ "$resolved" == "$payload"/* ]] || die "cleanup target escapes payload: $resolved"
    bytes=0
    if [[ -e "$target" || -L "$target" ]]; then
      bytes="$(du -sb -- "$target" | awk '{print $1}')"
    fi
    total=$((total + bytes))
    printf 'shear\t%s\t%s\n' "$bytes" "$target" >>"$manifest"
  done
  for target in "${targets[@]}"; do
    if [[ -e "$target" || -L "$target" ]]; then
      rm -rf -- "$target"
    fi
  done
  for target in "${targets[@]}"; do
    [[ ! -e "$target" && ! -L "$target" ]] || die "cleanup target remains: $target"
  done
  {
    printf 'recorded_bytes=%s\n' "$total"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'verified=true\n'
  } >"$summary"
}

postprocess_completed_cap() {
  local cap_root="$1" cap_bytes="$2"
  local payload="$cap_root/payload"
  extract_shear_metrics "$payload" "$cap_bytes"
  validate_graph "$payload"
  for evidence in \
    profile.json driver_timing.json shear_metrics.json graph_validation.json \
    graph_quality.tsv graph_quality.log index_size_before_cleanup.json \
    resource_summary.json; do
    require_file "$payload/$evidence"
  done
  cleanup_successful_payload "$payload"
}

update_schedule() {
  local label="$1" state="$2" message="${3:-}"
  python3 - "$BATCH_ROOT/schedule.tsv" "$label" "$state" "$message" <<'PY_SCHEDULE'
import csv
import datetime
from pathlib import Path
import sys

path = Path(sys.argv[1])
label, state, message = sys.argv[2:]
if path.is_file():
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
else:
    rows = [
        {"cap_label": name, "state": "not_run", "updated_at": "", "message": ""}
        for name in ("p50", "p40", "p30", "p20")
    ]
for row in rows:
    if row["cap_label"] == label:
        row["state"] = state
        row["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        row["message"] = message
tmp = path.with_suffix(".tmp")
with tmp.open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=("cap_label", "state", "updated_at", "message"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
tmp.replace(path)
PY_SCHEDULE
}

write_batch_manifest() {
  python3 - "$BATCH_ROOT" "$BASE" <<'PY_MANIFEST'
import datetime
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
value = {
    "schema": "laion100m_shear_memory_scaling_batch_v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "dataset": "laion100m",
    "base": sys.argv[2],
    "base_payload_bytes": 204_800_000_000,
    "sidecar_theoretical_bytes": 52_000_000_000,
    "cap_order": ["p50", "p40", "p30", "p20"],
    "cap_bytes": {
        "p50": 102_400_000_000,
        "p40": 81_920_000_000,
        "p30": 61_440_000_000,
        "p20": 40_960_000_000,
    },
    "cache_path": "cold",
    "oom_io_mode": "direct",
    "swap_enabled": False,
    "search_enabled": False,
}
(root / "run_manifest.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n"
)
PY_MANIFEST
}

initialize_batch() {
  if [[ -z "$BATCH_ROOT" ]]; then
    BATCH_ROOT="$OUTPUT_BASE/$(date +%Y%m%d_%H%M%S)"
  fi
  fresh_dir "$BATCH_ROOT"
  mkdir -p "$BATCH_ROOT/provenance"
  record_provenance "$BATCH_ROOT/provenance"
  write_batch_manifest
  update_schedule p50 not_run
}

run_cap() {
  local label="$1"
  local cap_bytes="${CAP_BYTES[$label]}"
  local cap_root="$BATCH_ROOT/$label"
  local payload="$cap_root/payload"
  local wrapper="$cap_root/shear_wrapper.sh"
  local unit="shear-laion100m-shear-${label}-$(date +%s)-$$"
  fresh_dir "$cap_root"
  record_environment "$cap_root" "$label" "$cap_bytes"
  write_shear_wrapper "$wrapper"
  sha256sum "$wrapper" >"$cap_root/shear_wrapper.sha256"
  write_status "$cap_root/status.json" running 0 "cold cgroup build"
  update_schedule "$label" running
  drop_page_cache
  launch_cgroup_service "$unit" "$cap_root" "$cap_bytes" \
    "$wrapper" "$payload" "$SHEAR_ROOT" "$BASE"

  local service_state service_exit
  service_state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' \
    "$cap_root/service_result.json")"
  service_exit="$(<"$cap_root/service_exit_code.txt")"
  case "$service_state" in
    completed)
      set +e
      ( set -e; postprocess_completed_cap "$cap_root" "$cap_bytes" ) \
        >"$cap_root/postprocess.stdout.log" 2>"$cap_root/postprocess.stderr.log"
      local post_exit=$?
      set -e
      printf '%s\n' "$post_exit" >"$cap_root/postprocess_exit_code.txt"
      if ((post_exit == 0)); then
        write_status "$cap_root/status.json" completed 0 "graph validation and cleanup passed"
        update_schedule "$label" completed
      else
        write_status "$cap_root/status.json" failed_validation "$post_exit" \
          "profile or graph validation failed"
        update_schedule "$label" failed_validation
      fi
      ;;
    oom)
      write_status "$cap_root/status.json" oom "$service_exit" "cgroup OOM evidence observed"
      update_schedule "$label" oom
      ;;
    *)
      write_status "$cap_root/status.json" failed_runtime "$service_exit" \
        "non-OOM construction or sampler failure"
      update_schedule "$label" failed_runtime
      ;;
  esac
}

generate_summary() {
  [[ -n "$BATCH_ROOT" && -d "$BATCH_ROOT" ]] || die "summary root is missing: $BATCH_ROOT"
  python3 - "$BATCH_ROOT" "$SIDECAR_BYTES" \
    "${CAP_BYTES[p50]}" "${CAP_BYTES[p40]}" "${CAP_BYTES[p30]}" "${CAP_BYTES[p20]}" <<'PY_SUMMARY'
import csv
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
sidecar_bytes = int(sys.argv[2])
labels = ("p50", "p40", "p30", "p20")
percent = {"p50": 50, "p40": 40, "p30": 30, "p20": 20}
caps = dict(zip(labels, map(int, sys.argv[3:]), strict=True))

fields = (
    "cap_label", "cap_percent", "cgroup_cap_bytes", "status", "exit_code",
    "event_oom", "event_oom_kill", "e2e_wall_s", "cgroup_peak_dram_bytes",
    "peak_process_tree_rss_bytes", "peak_hbm_bytes", "peak_temp_allocated_bytes",
    "actual_bh_bytes", "reservoir_resident_bytes", "sidecar_theoretical_bytes",
    "sidecar_resident_bytes", "sidecar_resident_fraction", "sidecar_admitted_rows",
    "sidecar_file_rows", "sidecar_file_logical_bytes", "sidecar_file_gather_ms",
    "candidate_spill_bytes", "candidate_spill_files", "candidate_reread_bytes",
    "external_reduce_ms", "reduce_segments_scanned", "reduce_part_files_opened",
    "cgroup_physical_read_bytes", "cgroup_physical_write_bytes",
    "cgroup_read_ios", "cgroup_write_ios", "index_bytes", "graph_rows",
    "graph_edges", "degree_min", "degree_mean", "degree_max", "invalid_edges",
    "self_edges", "duplicate_edges", "cleanup_verified",
)

def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else {}

def elapsed_seconds(path: Path):
    if not path.is_file():
        return None
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)",
        path.read_text(errors="replace"),
    )
    if not match:
        return None
    parts = [float(value) for value in match.group(1).split(":")]
    return sum(value * 60 ** index for index, value in enumerate(reversed(parts)))

def cgroup_metrics(path: Path):
    if not path.is_file():
        return {}
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    return {
        "cgroup_peak_dram_bytes": max(int(row["memory_peak"]) for row in rows),
        "cgroup_physical_read_bytes": max(0, int(last["rbytes"]) - int(first["rbytes"])),
        "cgroup_physical_write_bytes": max(0, int(last["wbytes"]) - int(first["wbytes"])),
        "cgroup_read_ios": max(0, int(last["rios"]) - int(first["rios"])),
        "cgroup_write_ios": max(0, int(last["wios"]) - int(first["wios"])),
    }

def cleanup_verified(path: Path):
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line == "verified=true":
            return True
        if line == "verified=false":
            return False
    return None

def render(value, *, fraction=False):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if fraction:
        return f"{float(value):.6f}"
    return str(value)

rows = []
for label in labels:
    cap_root = root / label
    payload = cap_root / "payload"
    status = load_json(cap_root / "status.json")
    service = load_json(cap_root / "service_result.json")
    resources = load_json(payload / "resource_summary.json")
    metrics = load_json(payload / "shear_metrics.json")
    graph = load_json(payload / "graph_validation.json")
    index = load_json(payload / "index_size_before_cleanup.json")
    cgroup = cgroup_metrics(cap_root / "cgroup_samples.tsv")
    row = {field: "N/A" for field in fields}
    row.update({
        "cap_label": label,
        "cap_percent": str(percent[label]),
        "cgroup_cap_bytes": str(caps[label]),
        "status": status.get("state", "not_run"),
        "exit_code": render(status.get("exit_code")),
        "event_oom": render(service.get("event_oom")),
        "event_oom_kill": render(service.get("event_oom_kill")),
        "e2e_wall_s": render(elapsed_seconds(payload / "time.log")),
        "sidecar_theoretical_bytes": str(sidecar_bytes),
        "cleanup_verified": render(cleanup_verified(payload / "cleanup_summary.txt")),
    })
    for key in (
        "peak_process_tree_rss_bytes", "peak_hbm_bytes", "peak_temp_allocated_bytes"
    ):
        row[key] = render(resources.get(key))
    for key in (
        "actual_bh_bytes", "reservoir_resident_bytes", "sidecar_resident_bytes",
        "sidecar_admitted_rows", "sidecar_file_rows", "sidecar_file_logical_bytes",
        "sidecar_file_gather_ms", "candidate_spill_bytes", "candidate_spill_files",
        "candidate_reread_bytes", "external_reduce_ms", "reduce_segments_scanned",
        "reduce_part_files_opened",
    ):
        row[key] = render(metrics.get(key))
    row["sidecar_resident_fraction"] = render(
        metrics.get("sidecar_resident_fraction"), fraction=True
    )
    for key, value in cgroup.items():
        row[key] = render(value)
    row.update({
        "index_bytes": render(index.get("bytes")),
        "graph_rows": render(graph.get("nodes")),
        "graph_edges": render(graph.get("edges")),
        "degree_min": render(graph.get("observed_min_degree")),
        "degree_mean": render(graph.get("mean_degree"), fraction=True),
        "degree_max": render(graph.get("observed_max_degree")),
        "invalid_edges": render(graph.get("invalid_edges")),
        "self_edges": render(graph.get("self_edges")),
        "duplicate_edges": render(graph.get("duplicate_edges")),
    })
    rows.append(row)

with (root / "summary.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# LAION100M SHEAR Memory Scaling Summary",
    "",
    "| Cap | Status | E2E (s) | Peak DRAM (B) | Actual B_h (B) | Sidecar resident | Sidecar file logical (B) | Spill (B) | External reduce (ms) | Physical read/write (B) | Graph valid |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    graph_valid = (
        "true"
        if row["invalid_edges"] == "0"
        and row["self_edges"] == "0"
        and row["duplicate_edges"] == "0"
        and row["graph_rows"] != "N/A"
        else "N/A"
    )
    lines.append(
        f'| {row["cap_percent"]}% | {row["status"]} | {row["e2e_wall_s"]} | '
        f'{row["cgroup_peak_dram_bytes"]} | {row["actual_bh_bytes"]} | '
        f'{row["sidecar_resident_fraction"]} | {row["sidecar_file_logical_bytes"]} | '
        f'{row["candidate_spill_bytes"]} | {row["external_reduce_ms"]} | '
        f'{row["cgroup_physical_read_bytes"]}/{row["cgroup_physical_write_bytes"]} | '
        f'{graph_valid} |'
    )
lines.extend([
    "",
    "`sidecar_file_logical_bytes = file_rows × 520 B`; it is not physical NVMe traffic.",
    "Candidate reread bytes are not independently exposed by the current profile and remain `N/A`.",
    "Cgroup physical I/O includes all files accessed by the process tree.",
    "No search is part of this experiment.",
])
(root / "summary.md").write_text("\n".join(lines) + "\n")
PY_SUMMARY
}

run_formal() {
  run_preflight
  mkdir -p "$(dirname "$LOCK_PATH")"
  exec {EXPERIMENT_LOCK_FD}>"$LOCK_PATH"
  flock -n "$EXPERIMENT_LOCK_FD" || die "another memory-scaling run holds $LOCK_PATH"
  initialize_batch
  run_cap "p50"
  generate_summary
  run_cap "p40"
  generate_summary
  run_cap "p30"
  generate_summary
  run_cap "p20"
  generate_summary
  log "formal experiment completed: $BATCH_ROOT"
}

run_self_test() {
  require_commands bash python3 awk systemd-run systemctl journalctl sudo stat du
  [[ "$(stat -fc %T /sys/fs/cgroup)" == cgroup2fs ]] || die "cgroup v2 is required"
  sudo -n true || die "passwordless sudo is required"
  local root success_root oom_root success_script oom_script
  root="$(mktemp -d ${SHEAR_OUTPUT_ROOT}/laion100m_shear_memory_scaling-selftest.XXXXXX)"
  [[ "$root" == ${SHEAR_OUTPUT_ROOT}/laion100m_shear_memory_scaling-selftest.* ]] \
    || die "unexpected self-test root: $root"
  success_root="$root/success"
  oom_root="$root/oom"
  success_script="$root/success.sh"
  oom_script="$root/oom.sh"
  mkdir -p "$success_root/payload" "$oom_root/payload"
  cat >"$success_script" <<'SELF_SUCCESS'
#!/usr/bin/env bash
set -Eeuo pipefail
python3 - <<'PY'
import time

payload = bytearray(8 * 1024 * 1024)
assert len(payload) == 8 * 1024 * 1024
time.sleep(2)
PY
SELF_SUCCESS
  cat >"$oom_script" <<'SELF_OOM'
#!/usr/bin/env bash
set -Eeuo pipefail
python3 - <<'PY'
import time

time.sleep(1)
payload = bytearray(256 * 1024 * 1024)
assert len(payload) == 256 * 1024 * 1024
PY
SELF_OOM
  chmod +x "$success_script" "$oom_script"
  SELF_TEST_MODE=1
  launch_cgroup_service "shear-shear-self-success-$$" "$success_root" \
    268435456 "$success_script"
  launch_cgroup_service "shear-shear-self-oom-$$" "$oom_root" \
    67108864 "$oom_script"
  local success_state oom_state
  success_state="$(<"$success_root/service_state.txt")"
  oom_state="$(<"$oom_root/service_state.txt")"
  [[ "$success_state" == completed ]] || die "self-test success classified as $success_state"
  [[ "$oom_state" == oom ]] || die "self-test OOM classified as $oom_state"
  grep -F 'memory.max=268435456' "$success_root/cgroup_limits.txt" >/dev/null
  grep -F 'memory.swap.max=0' "$success_root/cgroup_limits.txt" >/dev/null
  grep -F 'memory.max=67108864' "$oom_root/cgroup_limits.txt" >/dev/null
  grep -F 'memory.swap.max=0' "$oom_root/cgroup_limits.txt" >/dev/null
  rm -rf -- "$root"
  log "self-test passed"
}

case "$MODE" in
  help)
    usage
    ;;
  preflight)
    run_preflight
    ;;
  self-test)
    run_self_test
    ;;
  formal)
    run_formal
    ;;
  summarize)
    BATCH_ROOT="${2:-${BATCH_ROOT:-}}"
    [[ -n "$BATCH_ROOT" ]] || die "summarize requires BATCH_ROOT"
    generate_summary
    ;;
  *)
    usage
    die "unknown mode: $MODE"
    ;;
esac
