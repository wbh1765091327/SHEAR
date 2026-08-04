# Five-Method Four-Dataset Build and Search Runbook

This runbook creates one provenance-bound report for five methods on Wiki35M,
LAION100M, MSMARCO70M, and DEEP100M. It executes four graph methods in the
fixed order SHEAR, CAGRA NN-Descent, Tagore-CAGRA, and Tagore-Vamana. The
already accepted IVF-RaBitQ batch is imported as reused evidence; IVF-RaBitQ
is not executed by this runbook.

Each graph transaction uses the accepted hot-only construction contract,
validates the resulting graph, searches it with the shared
`rabitq1-adaptive-exact` backend, records construction and Recall--QPS
evidence, and then removes only that transaction's large graph artifacts.
The next method does not start until cleanup is verified. A failed transaction
is retained and stops the schedule.

SHEAR explicitly selects the evaluated HashPrune SoftTier production tuple:
`rhalf-soft-tier-q3`, at most three retained backfill edges per row, and
`priority_alpha=1.02`. The legacy configuration sentinel remains
`alpha=1.0`; all effective and requested tuning values are audited from the
profile.

Formal graph search uses Recall@10, eight threads, mmap, the complete query
set, `L=25,50,100,200,400`, one warmup, and two measured repetitions. The two
measured repetitions must have identical Recall@10 and result SHA-256. The
imported IVF rows retain `nprobe` as their parameter and are recomputed from
repetitions 1 and 2 only.

The formal experiment is not run by extracting this document or invoking
`help`. To extract the executable payload:

```bash
python3 script/test_five_method_four_dataset_runbook.py --extract \
  >/tmp/five_method_four_dataset.sh
chmod 700 /tmp/five_method_four_dataset.sh
/tmp/five_method_four_dataset.sh help
```

<!-- BEGIN RUNBOOK_SCRIPT -->
```bash
#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit
export LC_ALL=C

MODE="${MODE:-${1:-help}}"
SHEAR_ROOT="${SHEAR_ROOT:-/path/to/SHEAR}"
DATA_ROOT="${DATA_ROOT:-/data_nvme_kioxia/tyj/data}"
OUTPUT_BASE="${OUTPUT_BASE:-/data_staging/wbh/five_method_four_dataset/runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ROOT="${BATCH_ROOT:-$OUTPUT_BASE/$RUN_ID}"
LOCK_FILE="${LOCK_FILE:-/data_staging/wbh/.five_method_four_dataset.lock}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/data_staging/wbh/unified_graph_rabitq1/sidecars}"
SIDECAR_MANIFEST="${SIDECAR_MANIFEST:-$SIDECAR_ROOT/sidecar_manifest.tsv}"
IVF_REUSE_ROOT="${IVF_REUSE_ROOT:-/data_staging/wbh/ivf_rabitq_four_dataset/runs/20260721_235157}"
SEARCH_BIN="${SEARCH_BIN:-$SHEAR_ROOT/target/release/search_shear_index}"
TEST_SCRIPT="$SHEAR_ROOT/script/test_five_method_four_dataset_runbook.py"

SHEAR_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_shear_four_dataset.md"
WIKI_ACE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_wiki35m_shear_cagra_tagore.md"
LAION_ACE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_laion100m_shear_cagra_tagore.md"
MSMARCO_ACE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_msmarco70m_shear_cagra_tagore.md"
DEEP_ACE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_shear_cagra_tagore.md"
UNIFIED_SEARCH_RUNBOOK="$SHEAR_ROOT/docs/runbooks/unified_graph_rabitq1_search_four_dataset.md"
IVF_SCHEMA_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_ivf_rabitq_four_dataset.md"

FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)
GRAPH_METHODS=(shear cagra_nn_descent tagore_cagra tagore_vamana)
SEARCH_LISTS=(25 50 100 200 400)
ACTIVE_DATASETS=("${FORMAL_DATASETS[@]}")

DATASET=""
ROWS=0
DIM=0
BASE=""
QUERY=""
GT=""
TAGORE_CLUSTERS=0
TAGORE_MAX_POINTS=0
SPLIT_RECURSION_LIMIT=8
ACE_RUNBOOK=""
SIDECAR=""
CURRENT_METHOD_ROOT=""
CURRENT_STAGE="initializing"
BATCH_INITIALIZED=0
LOCK_FD=""
SHEAR_LIB=""
WIKI_ACE_LIB=""
LAION_ACE_LIB=""
MSMARCO_ACE_LIB=""
DEEP_ACE_LIB=""
PROCESS_MONITOR=""

export PATH="$HOME/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

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

require_dir() {
  [[ -d "$1" ]] || die "required directory is missing: $1"
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

usage() {
  cat <<'USAGE'
Usage: five_method_four_dataset.sh MODE

Modes:
  help        Show this contract without touching experiment state.
  prepare     Build/test required binaries and freeze provenance only.
  preflight   Validate data, sidecars, reused IVF evidence, and runtimes.
  smoke       Run the fixture-only orchestration contract test.
  formal      Execute four graph methods on all four datasets and report five methods.
  wiki35m     Execute the four graph methods for Wiki35M only.
  laion100m   Execute the four graph methods for LAION100M only.
  msmarco70m  Execute the four graph methods for MSMARCO70M only.
  deep100m    Execute the four graph methods for DEEP100M only.
  summarize   Rebuild summaries for an existing BATCH_ROOT; never build or search.

IVF-RaBitQ is imported from an accepted batch and is never executed here.
USAGE
}

write_status() {
  local path="$1" phase="$2" exit_code="$3" stage="${4:-}" message="${5:-}"
  python3 - "$path" "$phase" "$exit_code" "$stage" "$message" <<'PY_STATUS'
import datetime
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
event = {
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "stage": sys.argv[4] or None,
    "message": sys.argv[5] or None,
}
if path.exists():
    value = json.loads(path.read_text())
else:
    value = {"schema": "five_method_four_dataset_status_v1", "history": []}
value.update(event)
value["history"].append(event)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY_STATUS
}

initialize_schedule() {
  local datasets_csv
  datasets_csv="$(IFS=,; printf '%s' "${ACTIVE_DATASETS[*]}")"
  python3 - "$BATCH_ROOT/formal_schedule.tsv" "$BATCH_ROOT" "$datasets_csv" "$IVF_REUSE_ROOT" <<'PY_SCHEDULE_INIT'
import csv
from pathlib import Path
import sys

path, root = Path(sys.argv[1]), Path(sys.argv[2])
datasets = [value for value in sys.argv[3].split(",") if value]
ivf_root = sys.argv[4]
methods = ("shear", "cagra_nn_descent", "tagore_cagra", "tagore_vamana")
rows = []
ordinal = 0
for dataset in datasets:
    for method in methods:
        ordinal += 1
        rows.append({
            "ordinal": ordinal,
            "dataset": dataset,
            "method": method,
            "execution_origin": "current",
            "state": "scheduled",
            "run_root": str(root / "hot_path" / dataset / method / "run_1"),
            "source_root": "",
            "message": "",
        })
    ordinal += 1
    rows.append({
        "ordinal": ordinal,
        "dataset": dataset,
        "method": "ivf_rabitq",
        "execution_origin": "reused",
        "state": "reuse_pending",
        "run_root": "",
        "source_root": ivf_root,
        "message": "",
    })
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
PY_SCHEDULE_INIT
}

update_schedule() {
  local dataset="$1" method="$2" state="$3" message="${4:-}"
  python3 - "$BATCH_ROOT/formal_schedule.tsv" "$dataset" "$method" "$state" "$message" <<'PY_SCHEDULE_UPDATE'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
dataset, method, state, message = sys.argv[2:]
with path.open(newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    fields = reader.fieldnames
    rows = list(reader)
matches = 0
for row in rows:
    if row["dataset"] == dataset and row["method"] == method:
        row["state"] = state
        row["message"] = message
        matches += 1
if matches != 1:
    raise SystemExit(f"schedule row count is {matches} for {dataset}/{method}")
temporary = path.with_suffix(path.suffix + ".tmp")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(path)
PY_SCHEDULE_UPDATE
}

load_dataset() {
  DATASET="$1"
  case "$DATASET" in
    wiki35m)
      ROWS=35000000; DIM=1024
      BASE="$DATA_ROOT/wiki-1024-35M/en_base.fbin"
      QUERY="$DATA_ROOT/wiki-1024-35M/query.fbin"
      GT="$DATA_ROOT/wiki-1024-35M/gt100.bin"
      TAGORE_CLUSTERS=128; TAGORE_MAX_POINTS=1200000
      SPLIT_RECURSION_LIMIT=8; ACE_RUNBOOK="$WIKI_ACE_RUNBOOK"
      ;;
    laion100m)
      ROWS=100000000; DIM=512
      BASE="$DATA_ROOT/laion-512-100M/base.fbin"
      QUERY="$DATA_ROOT/laion-512-100M/query.fbin"
      GT="$DATA_ROOT/laion-512-100M/gt100.bin"
      TAGORE_CLUSTERS=256; TAGORE_MAX_POINTS=3000000
      SPLIT_RECURSION_LIMIT=8; ACE_RUNBOOK="$LAION_ACE_RUNBOOK"
      ;;
    msmarco70m)
      ROWS=70000000; DIM=768
      BASE="$DATA_ROOT/msmarco-768-70M/base.fbin"
      QUERY="$DATA_ROOT/msmarco-768-114M/query.fbin"
      GT="$DATA_ROOT/msmarco-768-70M/gt100.bin"
      TAGORE_CLUSTERS=256; TAGORE_MAX_POINTS=2500000
      SPLIT_RECURSION_LIMIT=32; ACE_RUNBOOK="$MSMARCO_ACE_RUNBOOK"
      ;;
    deep100m)
      ROWS=100000000; DIM=96
      BASE="$DATA_ROOT/deep-96-100M/base.100M.fbin"
      QUERY="$DATA_ROOT/deep-96-100M/query.public.10K.fbin"
      GT="$DATA_ROOT/deep-96-100M/groundtruth.public.10K.ibin"
      TAGORE_CLUSTERS=128; TAGORE_MAX_POINTS=5000000
      SPLIT_RECURSION_LIMIT=8; ACE_RUNBOOK="$DEEP_ACE_RUNBOOK"
      ;;
    *) die "unknown dataset: $DATASET" ;;
  esac
}

ace_library_for_dataset() {
  case "$1" in
    wiki35m) printf '%s\n' "$WIKI_ACE_LIB" ;;
    laion100m) printf '%s\n' "$LAION_ACE_LIB" ;;
    msmarco70m) printf '%s\n' "$MSMARCO_ACE_LIB" ;;
    deep100m) printf '%s\n' "$DEEP_ACE_LIB" ;;
    *) die "unknown dataset for ACE library: $1" ;;
  esac
}

extract_runbook_library() {
  local source="$1" output="$2"
  python3 - "$source" "$output" <<'PY_EXTRACT_RUNBOOK'
from pathlib import Path
import re
import sys

source, output = map(Path, sys.argv[1:])
text = source.read_text()
marker = "<!-- {} RUNBOOK_SCRIPT -->"
begin_marker = re.search(rf"^{re.escape(marker.format('BEGIN'))}[ \t]*$", text, re.M)
end_marker = re.search(rf"^{re.escape(marker.format('END'))}[ \t]*$", text, re.M)
if not begin_marker or not end_marker or end_marker.start() <= begin_marker.end():
    raise SystemExit(f"missing standalone runbook markers: {source}")
begin = begin_marker.end()
end = end_marker.start()
match = re.search(r"```bash\n(?P<body>.*?)\n```", text[begin:end], re.S)
if not match:
    raise SystemExit(f"missing Bash payload: {source}")
script = match.group("body") + "\n"
terminal = re.search(r'^case "\$MODE" in\n', script, re.M)
if not terminal:
    raise SystemExit(f"missing terminal dispatch: {source}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(script[:terminal.start()])
PY_EXTRACT_RUNBOOK
  chmod 700 "$output"
}

write_process_tree_monitor() {
  local output="$1"
  cat >"$output" <<'PY_PROCESS_MONITOR'
#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import time

root_pid = int(sys.argv[1])
samples_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
maxima = {}
peak_rss = 0
valid_samples = 0

def snapshot():
    records = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            close = raw.rfind(")")
            rest = raw[close + 2 :].split()
            pid = int(entry.name)
            records[pid] = (int(rest[1]), int(rest[19]))
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    descendants = set()
    for pid in records:
        cursor = pid
        seen = set()
        while cursor in records and cursor not in seen:
            if cursor == root_pid:
                descendants.add(pid)
                break
            seen.add(cursor)
            cursor = records[cursor][0]
    return records, descendants

samples_path.parent.mkdir(parents=True, exist_ok=True)
with samples_path.open("w") as samples:
    samples.write("epoch\tprocesses\trss_bytes\tread_bytes_seen\twrite_bytes_seen\n")
    while True:
        records, descendants = snapshot()
        if root_pid not in records:
            break
        rss_total = 0
        for pid in descendants:
            try:
                status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
                rss_kib = next(
                    (int(line.split()[1]) for line in status if line.startswith("VmRSS:")), 0
                )
                values = {}
                for line in (Path("/proc") / str(pid) / "io").read_text().splitlines():
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip())
            except (
                FileNotFoundError,
                ProcessLookupError,
                PermissionError,
                StopIteration,
                ValueError,
            ):
                continue
            identity = f"{pid}:{records[pid][1]}"
            previous = maxima.get(identity, {"read_bytes": 0, "write_bytes": 0})
            maxima[identity] = {
                "read_bytes": max(previous["read_bytes"], values.get("read_bytes", 0)),
                "write_bytes": max(previous["write_bytes"], values.get("write_bytes", 0)),
            }
            rss_total += rss_kib * 1024
        peak_rss = max(peak_rss, rss_total)
        read_seen = sum(value["read_bytes"] for value in maxima.values())
        write_seen = sum(value["write_bytes"] for value in maxima.values())
        samples.write(f"{time.time()}\t{len(descendants)}\t{rss_total}\t{read_seen}\t{write_seen}\n")
        samples.flush()
        valid_samples += 1
        time.sleep(0.1)
summary = {
    "schema": "five_method_process_tree_io_v1",
    "root_pid": root_pid,
    "valid_samples": valid_samples,
    "peak_process_tree_rss_bytes": peak_rss,
    "read_bytes": sum(value["read_bytes"] for value in maxima.values()),
    "write_bytes": sum(value["write_bytes"] for value in maxima.values()),
}
temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(summary_path)
PY_PROCESS_MONITOR
  chmod 700 "$output"
}

prepare_runbook_libraries() {
  local root="$BATCH_ROOT/provenance/runbook_libraries"
  mkdir -p "$root"
  SHEAR_LIB="$root/shear_four_dataset.lib.sh"
  WIKI_ACE_LIB="$root/wiki35m_cagra_tagore.lib.sh"
  LAION_ACE_LIB="$root/laion100m_cagra_tagore.lib.sh"
  MSMARCO_ACE_LIB="$root/msmarco70m_cagra_tagore.lib.sh"
  DEEP_ACE_LIB="$root/deep100m_cagra_tagore.lib.sh"
  extract_runbook_library "$SHEAR_RUNBOOK" "$SHEAR_LIB"
  extract_runbook_library "$WIKI_ACE_RUNBOOK" "$WIKI_ACE_LIB"
  extract_runbook_library "$LAION_ACE_RUNBOOK" "$LAION_ACE_LIB"
  extract_runbook_library "$MSMARCO_ACE_RUNBOOK" "$MSMARCO_ACE_LIB"
  extract_runbook_library "$DEEP_ACE_RUNBOOK" "$DEEP_ACE_LIB"
  extract_runbook_library "$UNIFIED_SEARCH_RUNBOOK" "$root/unified_graph_search.reference.lib.sh"
  extract_runbook_library "$IVF_SCHEMA_RUNBOOK" "$root/ivf_rabitq.reference.lib.sh"
  PROCESS_MONITOR="$BATCH_ROOT/provenance/process_tree_io_monitor.py"
  write_process_tree_monitor "$PROCESS_MONITOR"
  sha256sum \
    "$SHEAR_RUNBOOK" "$WIKI_ACE_RUNBOOK" "$LAION_ACE_RUNBOOK" \
    "$MSMARCO_ACE_RUNBOOK" "$DEEP_ACE_RUNBOOK" "$UNIFIED_SEARCH_RUNBOOK" \
    "$IVF_SCHEMA_RUNBOOK" "$SHEAR_LIB" "$WIKI_ACE_LIB" "$LAION_ACE_LIB" \
    "$MSMARCO_ACE_LIB" "$DEEP_ACE_LIB" "$PROCESS_MONITOR" \
    >"$BATCH_ROOT/provenance/source_runbooks.sha256"
}

record_provenance() {
  mkdir -p "$BATCH_ROOT/provenance"
  {
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'mode\t%s\n' "$MODE"
    printf 'datasets\t%s\n' "$(IFS=,; printf '%s' "${ACTIVE_DATASETS[*]}")"
    printf 'ivf_execution_origin\treused\n'
    printf 'ivf_source_root\t%s\n' "$IVF_REUSE_ROOT"
    printf 'graph_search_backend\trabitq1-adaptive-exact\n'
    printf 'graph_search_repetitions\t2\n'
    printf 'git_head\t'
    git -C "$SHEAR_ROOT" rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
  } >"$BATCH_ROOT/provenance/context.tsv"
  git -C "$SHEAR_ROOT" status --short >"$BATCH_ROOT/provenance/git_status.txt" || true
  if ! git -C "$SHEAR_ROOT" diff --binary >"$BATCH_ROOT/provenance/working_tree.diff"; then
    rm -f "$BATCH_ROOT/provenance/working_tree.diff"
  fi
  sha256sum "$SEARCH_BIN" "$SIDECAR_MANIFEST" >"$BATCH_ROOT/provenance/runtime_inputs.sha256"
}

select_sidecar() {
  local passport="$BATCH_ROOT/hot_path/$DATASET/sidecar_passport.json"
  mkdir -p "$(dirname "$passport")"
  python3 - "$SIDECAR_MANIFEST" "$DATASET" "$BASE" "$ROWS" "$DIM" "$passport" <<'PY_SIDECAR'
import csv
import hashlib
import json
from pathlib import Path
import sys

manifest, dataset, base, rows, dim, output = sys.argv[1:]
rows, dim = int(rows), int(dim)
with Path(manifest).open(newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    required = [
        "dataset", "base", "sidecar", "sidecar_bytes", "sha256", "format_version",
        "rows", "dim", "padded_dim", "validation_status",
    ]
    if reader.fieldnames != required:
        raise SystemExit("sidecar manifest schema mismatch")
    matches = [row for row in reader if row["dataset"] == dataset]
if len(matches) != 1:
    raise SystemExit(f"sidecar manifest row count is {len(matches)} for {dataset}")
row = matches[0]
sidecar = Path(row["sidecar"])
if row["base"] != base or int(row["rows"]) != rows or int(row["dim"]) != dim:
    raise SystemExit(f"sidecar dataset contract mismatch for {dataset}")
if row["format_version"] != "graph_rabitq1_v1" or row["validation_status"] != "accepted":
    raise SystemExit(f"sidecar is not accepted for {dataset}")
if not sidecar.is_file() or sidecar.stat().st_size != int(row["sidecar_bytes"]):
    raise SystemExit(f"sidecar file contract mismatch: {sidecar}")
digest = hashlib.sha256()
with sidecar.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != row["sha256"]:
    raise SystemExit(f"sidecar SHA-256 mismatch: {sidecar}")
record = dict(row)
record["rows"] = int(record["rows"])
record["dim"] = int(record["dim"])
record["padded_dim"] = int(record["padded_dim"])
record["sidecar_bytes"] = int(record["sidecar_bytes"])
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(sidecar)
PY_SIDECAR
  SIDECAR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sidecar"])' "$passport")"
}

write_shear_wrapper() {
  local output="$1"
  cat >"$output" <<'SHEAR_WRAPPER'
#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

library="$1"
dataset="$2"
payload="$3"
batch_root="$4"
process_monitor="$5"
export SHEAR_ROOT="$6"
export DATA_ROOT="$7"
export BATCH_ROOT="$batch_root"

source "$library"
trap - ERR INT TERM EXIT
load_dataset "$dataset"
RUN_ROOT="$payload"
INDEX_DIR="$RUN_ROOT/index"
INDEX_FILE="$INDEX_DIR/_mem.index"
OOM_TEMP_DIR="$RUN_ROOT/oom_temp"
SQ_SIDECAR_DIR="$RUN_ROOT/sq_sidecar"
SQ_SIDECAR_PREFIX="$SQ_SIDECAR_DIR/sq_wmma"
EXPERIMENT_ROLE=formal

eval "$(declare -f run_measured_build | sed '1s/run_measured_build/source_run_measured_build/')"
run_measured_build() {
  local run_root="$1" build_pid monitor_pid build_rc monitor_rc
  source_run_measured_build "$@" &
  build_pid=$!
  python3 "$process_monitor" "$build_pid" \
    "$run_root/process_tree_io.tsv" "$run_root/process_tree_io_summary.json" &
  monitor_pid=$!
  set +e
  wait "$build_pid"
  build_rc=$?
  wait "$monitor_pid"
  monitor_rc=$?
  set -e
  if [[ "$build_rc" -eq 0 && "$monitor_rc" -ne 0 ]]; then
    return "$monitor_rc"
  fi
  [[ "$build_rc" -ne 0 || -s "$run_root/process_tree_io_summary.json" ]] || return 1
  return "$build_rc"
}

fresh_dir "$RUN_ROOT"
mkdir -p "$RUN_ROOT/tools"
record_cache_path
validate_dataset
{
  date --iso-8601=seconds
  uname -a
  free -h
  nvidia-smi
} >"$RUN_ROOT/environment.txt" 2>&1
prepare_build_cache

# The named mode keeps 1.0 as its legacy configuration sentinel. The evaluated
# production tuning is explicit here and is audited below.
export SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE=rhalf-soft-tier-q3
export SHEAR_PIPNN_HASHPRUNE_BACKFILL_ABLATION=none
export SHEAR_PIPNN_HASHPRUNE_BACKFILL_BUDGET=0
export SHEAR_PIPNN_HASHPRUNE_BACKFILL_ALPHA=1.0
export SHEAR_PIPNN_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA=1.02
export SHEAR_PIPNN_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW=3
run_shear_build
validate_shear_profile

python3 - "$RUN_ROOT/profile.json" "$ROWS" "$RUN_ROOT/soft_tail_audit.json" <<'PY_SOFT_TAIL_AUDIT'
import json
import math
from pathlib import Path
import sys

profile_path, rows, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
profile = json.loads(profile_path.read_text())
value = profile.get("hashprune_backfill")
if not isinstance(value, dict):
    raise RuntimeError("profile has no hashprune_backfill object")
expected = {
    "mode": "rhalf-soft-tier-q3",
    "ablation": "none",
    "budget": 0,
    "alpha": 1.0,
    "priority_rule": "soft-tier-q3",
    "priority_alpha": 1.02,
    "effective_priority_alpha": 1.02,
    "requested_priority_alpha": 1.02,
    "effective_max_backfill_per_row": 3,
    "requested_max_backfill_per_row": 3,
    "derived_budget": 32,
    "backup_hash_quota": 3,
    "uses_alpha_gate": False,
    "rows_seen": rows,
}
for key, wanted in expected.items():
    actual = value.get(key)
    if isinstance(wanted, float):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"hashprune_backfill.{key}={actual!r}, expected {wanted!r}")
    elif actual != wanted:
        raise RuntimeError(f"hashprune_backfill.{key}={actual!r}, expected {wanted!r}")
audit = {
    "schema": "five_method_soft_tail_audit_v1",
    "accepted": True,
    "legacy_configuration": {
        "mode": expected["mode"],
        "ablation": expected["ablation"],
        "budget": expected["budget"],
        "alpha": expected["alpha"],
    },
    "effective_policy": {
        "priority_rule": expected["priority_rule"],
        "priority_alpha": float(value["priority_alpha"]),
        "effective_priority_alpha": float(value["effective_priority_alpha"]),
        "requested_priority_alpha": float(value["requested_priority_alpha"]),
        "effective_max_backfill_per_row": value["effective_max_backfill_per_row"],
        "requested_max_backfill_per_row": value["requested_max_backfill_per_row"],
        "derived_budget": expected["derived_budget"],
        "backup_hash_quota": expected["backup_hash_quota"],
        "uses_alpha_gate": expected["uses_alpha_gate"],
        "rows_seen": expected["rows_seen"],
    },
}
output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
PY_SOFT_TAIL_AUDIT

validate_graph
SHEAR_WRAPPER
  chmod 700 "$output"
}

write_ace_wrapper() {
  local output="$1"
  cat >"$output" <<'ACE_WRAPPER'
#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

library="$1"
method="$2"
payload="$3"
base="$4"
query="$5"
gt="$6"
rows="$7"
dim="$8"
clusters="$9"
max_points="${10}"
batch_root="${11}"
process_monitor="${12}"
export SHEAR_ROOT="${13}"

source "$library"
trap - ERR INT TERM EXIT
eval "$(declare -f run_measured_build | sed '1s/run_measured_build/source_run_measured_build/')"
run_measured_build() {
  local run_root="$1" build_pid monitor_pid build_rc monitor_rc
  source_run_measured_build "$@" &
  build_pid=$!
  python3 "$process_monitor" "$build_pid" \
    "$run_root/process_tree_io.tsv" "$run_root/process_tree_io_summary.json" &
  monitor_pid=$!
  set +e
  wait "$build_pid"
  build_rc=$?
  wait "$monitor_pid"
  monitor_rc=$?
  set -e
  if [[ "$build_rc" -eq 0 && "$monitor_rc" -ne 0 ]]; then
    return "$monitor_rc"
  fi
  [[ "$build_rc" -ne 0 || -s "$run_root/process_tree_io_summary.json" ]] || return 1
  return "$build_rc"
}
run_search_sweep() { :; }
finalize_successful_method() { :; }

run_method "$method" "$payload" "$base" "$query" "$gt" "$rows" "$dim" \
  "25 50 100 200 400" 2 P1_P2 "$clusters" "$max_points" 16 hot 0
ACE_WRAPPER
  chmod 700 "$output"
}

write_tagore_wrapper() {
  local output="$1"
  cat >"$output" <<'TAGORE_WRAPPER'
#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

library="$1"
payload="$2"
base="$3"
query="$4"
gt="$5"
rows="$6"
dim="$7"
clusters="$8"
max_points="$9"
batch_root="${10}"
process_monitor="${11}"
export SHEAR_ROOT="${12}"

source "$library"
trap - ERR INT TERM EXIT
eval "$(declare -f run_measured_build | sed '1s/run_measured_build/source_run_measured_build/')"
run_measured_build() {
  local run_root="$1" build_pid monitor_pid build_rc monitor_rc
  source_run_measured_build "$@" &
  build_pid=$!
  python3 "$process_monitor" "$build_pid" \
    "$run_root/process_tree_io.tsv" "$run_root/process_tree_io_summary.json" &
  monitor_pid=$!
  set +e
  wait "$build_pid"
  build_rc=$?
  wait "$monitor_pid"
  monitor_rc=$?
  set -e
  if [[ "$build_rc" -eq 0 && "$monitor_rc" -ne 0 ]]; then
    return "$monitor_rc"
  fi
  [[ "$build_rc" -ne 0 || -s "$run_root/process_tree_io_summary.json" ]] || return 1
  return "$build_rc"
}
run_search_sweep() { :; }
finalize_successful_method() { :; }

run_method tagore_vamana "$payload" "$base" "$query" "$gt" "$rows" "$dim" \
  "25 50 100 200 400" 2 P1_P2 "$clusters" "$max_points" 16 hot 0
TAGORE_WRAPPER
  chmod 700 "$output"
}

write_build_wrappers() {
  local root="$BATCH_ROOT/provenance/wrappers"
  mkdir -p "$root"
  write_shear_wrapper "$root/shear.sh"
  write_ace_wrapper "$root/ace.sh"
  write_tagore_wrapper "$root/tagore_vamana.sh"
  sha256sum "$root/shear.sh" "$root/ace.sh" "$root/tagore_vamana.sh" \
    >"$BATCH_ROOT/provenance/wrappers.sha256"
}

run_build_adapter() {
  local method="$1" method_root="$2"
  local payload="$method_root/payload"
  local wrapper library
  mkdir -p "$method_root"
  case "$method" in
    shear)
      wrapper="$BATCH_ROOT/provenance/wrappers/shear.sh"
      bash "$wrapper" "$SHEAR_LIB" "$DATASET" "$payload" "$BATCH_ROOT" \
        "$PROCESS_MONITOR" "$SHEAR_ROOT" "$DATA_ROOT" \
        >"$method_root/build_adapter.stdout.log" 2>"$method_root/build_adapter.stderr.log"
      ;;
    cagra_nn_descent|tagore_cagra)
      wrapper="$BATCH_ROOT/provenance/wrappers/ace.sh"
      library="$(ace_library_for_dataset "$DATASET")"
      bash "$wrapper" "$library" "$method" "$payload" "$BASE" "$QUERY" "$GT" \
        "$ROWS" "$DIM" "$TAGORE_CLUSTERS" "$TAGORE_MAX_POINTS" "$BATCH_ROOT" \
        "$PROCESS_MONITOR" "$SHEAR_ROOT" \
        >"$method_root/build_adapter.stdout.log" 2>"$method_root/build_adapter.stderr.log"
      ;;
    tagore_vamana)
      wrapper="$BATCH_ROOT/provenance/wrappers/tagore_vamana.sh"
      library="$(ace_library_for_dataset "$DATASET")"
      bash "$wrapper" "$library" "$payload" "$BASE" "$QUERY" "$GT" "$ROWS" "$DIM" \
        "$TAGORE_CLUSTERS" "$TAGORE_MAX_POINTS" "$BATCH_ROOT" "$PROCESS_MONITOR" \
        "$SHEAR_ROOT" >"$method_root/build_adapter.stdout.log" \
        2>"$method_root/build_adapter.stderr.log"
      ;;
    *) die "unknown graph method: $method" ;;
  esac
  require_file "$payload/index/_mem.index"
}

record_graph_passport() {
  local method="$1" method_root="$2"
  local graph="$method_root/payload/index/_mem.index"
  python3 - "$graph" "$DATASET" "$method" "$ROWS" "$DIM" \
    "$method_root/graph_passport.json" <<'PY_GRAPH_PASSPORT'
import hashlib
import json
from pathlib import Path
import sys

graph, dataset, method, rows, dim, output = sys.argv[1:]
graph_path = Path(graph)
digest = hashlib.sha256()
with graph_path.open("rb") as stream:
    for block in iter(lambda: stream.read(64 * 1024 * 1024), b""):
        digest.update(block)
value = {
    "schema": "five_method_graph_passport_v1",
    "dataset": dataset,
    "method": method,
    "rows": int(rows),
    "dim": int(dim),
    "path": str(graph_path.absolute()),
    "bytes": graph_path.stat().st_size,
    "sha256": digest.hexdigest(),
}
Path(output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY_GRAPH_PASSPORT
}

validate_graph_quality_gate() {
  local method_root="$1"
  python3 - "$method_root/payload/graph_quality.tsv" <<'PY_GRAPH_QUALITY_GATE'
import csv
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise RuntimeError(f"missing graph quality report: {path}")
with path.open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if len(rows) != 1:
    raise RuntimeError(f"expected one graph quality row in {path}, found {len(rows)}")
row = rows[0]
ratio = float(row["directed_reachable_from_start_ratio"])
if not math.isfinite(ratio) or ratio < 0.95:
    raise RuntimeError(
        f"graph entry reaches only {ratio:.6f} of nodes; required minimum is 0.95"
    )
PY_GRAPH_QUALITY_GATE
}

prepare_graph_search_cache() {
  local method_root="$1" path
  local graph="$method_root/payload/index/_mem.index"
  local manifest="$method_root/search_prewarm_manifest.tsv"
  printf 'path\tbytes\telapsed_seconds\n' >"$manifest"
  sync
  printf 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  for path in "$BASE" "$graph" "$QUERY" "$GT" "$SIDECAR"; do
    local elapsed_file="$method_root/.prewarm_elapsed"
    /usr/bin/time -f '%e' -o "$elapsed_file" dd if="$path" of=/dev/null bs=16M status=none
    printf '%s\t%s\t%s\n' "$path" "$(stat -Lc %s -- "$path")" "$(<"$elapsed_file")" \
      >>"$manifest"
    rm -f "$elapsed_file"
  done
}

run_graph_search() {
  local method="$1" method_root="$2"
  local index_prefix="$method_root/payload/index"
  local search_root="$method_root/unified_search" raw="$method_root/graph_search_sweep.tsv"
  local l rep kind label run_dir metrics result
  mkdir -p "$search_root"
  printf 'dataset\tmethod\texecution_origin\tparameter_kind\tparameter_value\trepeat\trecall_at_10\tqps\tresult_sha256\ttotal_one_bit_evaluations\ttotal_lower_bound_rejects\ttotal_navigation_exact_evaluations\ttotal_adaptive_exact_refinements\ttotal_exact_evaluations\tlogical_exact_vector_bytes\n' >"$raw"
  prepare_graph_search_cache "$method_root"
  for l in 25 50 100 200 400; do
    for kind in warmup; do
      run_dir="$search_root/L$l/$kind"
      mkdir -p "$run_dir"
      "$SEARCH_BIN" \
        --index-path-prefix "$index_prefix" \
        --data-path "$BASE" \
        --query-file "$QUERY" \
        --result-path "$run_dir/result.fbin" \
        --gt-file "$GT" \
        --recall-at 10 \
        --search-list-size "$l" \
        --num-threads 8 \
        --storage mmap \
        --search-backend rabitq1-adaptive-exact \
        --rabitq1-sidecar "$SIDECAR" \
        --metrics-path "$run_dir/metrics.tsv" \
        >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
    done
    for rep in 1 2; do
      label="rep_$rep"
      run_dir="$search_root/L$l/$label"
      mkdir -p "$run_dir"
      metrics="$run_dir/metrics.tsv"
      result="$run_dir/result.fbin"
      "$SEARCH_BIN" \
        --index-path-prefix "$index_prefix" \
        --data-path "$BASE" \
        --query-file "$QUERY" \
        --result-path "$result" \
        --gt-file "$GT" \
        --recall-at 10 \
        --search-list-size "$l" \
        --num-threads 8 \
        --storage mmap \
        --search-backend rabitq1-adaptive-exact \
        --rabitq1-sidecar "$SIDECAR" \
        --metrics-path "$metrics" \
        >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
      sha256sum "$result" >"$run_dir/result.sha256"
      python3 - "$raw" "$DATASET" "$method" "$l" "$rep" "$metrics" \
        "$run_dir/result.sha256" <<'PY_GRAPH_SEARCH_ROW'
import csv
from pathlib import Path
import sys

raw, dataset, method, parameter, repeat, metrics, digest = sys.argv[1:]
with Path(metrics).open(newline="") as stream:
    metric = next(csv.DictReader(stream, delimiter="\t"))
fields = (
    "dataset", "method", "execution_origin", "parameter_kind", "parameter_value",
    "repeat", "recall_at_10", "qps", "result_sha256",
    "total_one_bit_evaluations", "total_lower_bound_rejects",
    "total_navigation_exact_evaluations", "total_adaptive_exact_refinements",
    "total_exact_evaluations", "logical_exact_vector_bytes",
)
row = {
    "dataset": dataset,
    "method": method,
    "execution_origin": "current",
    "parameter_kind": "L",
    "parameter_value": parameter,
    "repeat": repeat,
    "recall_at_10": metric["recall"],
    "qps": metric["qps"],
    "result_sha256": Path(digest).read_text().split()[0],
}
for field in fields[9:]:
    row[field] = metric.get(field, "")
with Path(raw).open("a", newline="") as stream:
    csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n").writerow(row)
PY_GRAPH_SEARCH_ROW
    done
  done
  python3 - "$raw" "$method_root/graph_search_summary.tsv" <<'PY_GRAPH_SEARCH_SUMMARY'
import csv
import math
from pathlib import Path
import statistics
import sys

source, output = map(Path, sys.argv[1:])
with source.open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
groups = {}
for row in rows:
    groups.setdefault(int(row["parameter_value"]), []).append(row)
if sorted(groups) != [25, 50, 100, 200, 400]:
    raise RuntimeError(f"unexpected L values: {sorted(groups)}")
fields = (
    "dataset", "method", "execution_origin", "parameter_kind", "parameter_value",
    "recall_at_10", "qps_mean", "qps_sample_std", "qps_median", "qps_min",
    "qps_max", "repeat_count", "source_root",
)
summary = []
last_recall = -1.0
for parameter in sorted(groups):
    group = groups[parameter]
    if sorted(int(row["repeat"]) for row in group) != [1, 2]:
        raise RuntimeError(f"L={parameter} does not have repetitions 1 and 2")
    recalls = [float(row["recall_at_10"]) for row in group]
    qps = [float(row["qps"]) for row in group]
    hashes = {row["result_sha256"] for row in group}
    if len(hashes) != 1 or max(recalls) - min(recalls) > 1e-12:
        raise RuntimeError(f"L={parameter} repetitions changed result or recall")
    if not all(math.isfinite(value) and value > 0 for value in qps):
        raise RuntimeError(f"L={parameter} has invalid QPS")
    recall = recalls[0]
    if not math.isfinite(recall) or not 0 <= recall <= 1 or recall + 1e-12 < last_recall:
        raise RuntimeError(f"L={parameter} has invalid or decreasing recall")
    if parameter == 25 and recall <= 0:
        raise RuntimeError("L=25 recall must be positive")
    last_recall = recall
    summary.append({
        "dataset": group[0]["dataset"],
        "method": group[0]["method"],
        "execution_origin": "current",
        "parameter_kind": "L",
        "parameter_value": parameter,
        "recall_at_10": recall,
        "qps_mean": statistics.mean(qps),
        "qps_sample_std": statistics.stdev(qps),
        "qps_median": statistics.median(qps),
        "qps_min": min(qps),
        "qps_max": max(qps),
        "repeat_count": 2,
        "source_root": "",
    })
temporary = output.with_suffix(output.suffix + ".tmp")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
temporary.replace(output)
PY_GRAPH_SEARCH_SUMMARY
}

write_graph_construction_row() {
  local method="$1" method_root="$2"
  python3 - "$DATASET" "$method" "$method_root" \
    "$method_root/construction_row.tsv" <<'PY_GRAPH_CONSTRUCTION'
import csv
import json
import re
from pathlib import Path
import sys

dataset, method, root, output = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
payload = root / "payload"

def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}

def elapsed_from_time(path):
    if not path.is_file():
        return ""
    match = re.search(r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+?)\s*$", path.read_text(errors="replace"), re.M)
    if not match:
        return ""
    parts = match.group(1).split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return ""

wrapper = load_json(payload / "build_wrapper_timing.json")
wall = wrapper.get("elapsed_s", "")
if wall == "":
    wall = elapsed_from_time(payload / "time.log")
io = load_json(payload / "process_tree_io_summary.json")
resource = load_json(payload / "resource_summary.json")
rss = load_json(payload / "process_tree_rss_summary.json")
temp = load_json(payload / "temp_usage_summary.json")
passport = load_json(root / "graph_passport.json")
peak_dram = io.get("peak_process_tree_rss_bytes")
if not peak_dram:
    peak_dram = resource.get("peak_process_tree_rss_bytes") or int(rss.get("peak_rss_kib", 0)) * 1024
peak_temp = resource.get("peak_temp_allocated_bytes", temp.get("peak_allocated_bytes", 0))
fields = (
    "dataset", "method", "execution_origin", "status", "external_wall_seconds",
    "peak_dram_bytes", "read_bytes", "write_bytes", "peak_temp_bytes",
    "index_bytes", "source_root",
)
row = {
    "dataset": dataset,
    "method": method,
    "execution_origin": "current",
    "status": "accepted",
    "external_wall_seconds": wall,
    "peak_dram_bytes": peak_dram or 0,
    "read_bytes": io.get("read_bytes", ""),
    "write_bytes": io.get("write_bytes", ""),
    "peak_temp_bytes": peak_temp or 0,
    "index_bytes": passport.get("bytes", 0),
    "source_root": "",
}
temporary = output.with_suffix(output.suffix + ".tmp")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
temporary.replace(output)
PY_GRAPH_CONSTRUCTION
}

append_tsv() {
  local source="$1" destination="$2"
  python3 - "$source" "$destination" <<'PY_APPEND_TSV'
import csv
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
with source.open(newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    fields = reader.fieldnames
    rows = list(reader)
if destination.exists():
    with destination.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != fields:
            raise RuntimeError(f"TSV schema mismatch: {source} -> {destination}")
with destination.open("a", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    if destination.stat().st_size == 0:
        writer.writeheader()
    writer.writerows(rows)
PY_APPEND_TSV
}

cleanup_successful_method() {
  local method="$1" method_root="$2"
  local payload="$method_root/payload"
  local target resolved bytes total=0
  local manifest="$method_root/cleanup_manifest.tsv"
  local -a targets=()
  case "$method" in
    shear)
      targets=("$payload/index" "$payload/oom_temp" "$payload/sq_sidecar")
      ;;
    cagra_nn_descent|tagore_cagra)
      targets=("$payload/index" "$payload/ace_export")
      ;;
    tagore_vamana)
      targets=(
        "$payload/index"
        "$payload/diskann_part"
        "$payload/tagore_input/clusters"
        "$payload/tagore_input/centers.bin"
        "$payload/tagore_run/local_index"
        "$payload/tagore_run/index"
      )
      ;;
    *) die "unknown cleanup method: $method" ;;
  esac

  # Resolve the complete allowlist before removing any path. A top-level
  # symlink is rejected if its referent leaves this transaction's payload.
  for target in "${targets[@]}"; do
    resolved="$(realpath -m -- "$target")"
    [[ "$resolved" == "$payload"/* ]] || die "cleanup target escapes payload: $target -> $resolved"
  done
  for target in \
    "$method_root/graph_passport.json" \
    "$method_root/graph_search_sweep.tsv" \
    "$method_root/graph_search_summary.tsv"; do
    require_file "$target"
  done

  printf 'method\tbytes\tpath\tresolved_path\n' >"$manifest"
  for target in "${targets[@]}"; do
    resolved="$(realpath -m -- "$target")"
    bytes=0
    if [[ -e "$target" || -L "$target" ]]; then
      bytes="$(du -sb -- "$target" | awk '{print $1}')"
    fi
    total=$((total + bytes))
    printf '%s\t%s\t%s\t%s\n' "$method" "$bytes" "$target" "$resolved" >>"$manifest"
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
    printf 'method=%s\n' "$method"
    printf 'recorded_bytes=%s\n' "$total"
    printf 'verified=true\n'
    printf 'verified_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$method_root/cleanup_summary.txt"
}

write_artifact_row() {
  local method="$1" method_root="$2"
  python3 - "$method_root/graph_passport.json" "$method_root/cleanup_summary.txt" \
    "$method_root/artifact_row.tsv" <<'PY_ARTIFACT_ROW'
import csv
import json
from pathlib import Path
import sys

passport_path, cleanup_path, output = map(Path, sys.argv[1:])
passport = json.loads(passport_path.read_text())
cleanup = dict(
    line.split("=", 1) for line in cleanup_path.read_text().splitlines() if "=" in line
)
if cleanup.get("verified") != "true":
    raise RuntimeError(f"cleanup is not verified: {cleanup_path}")
fields = (
    "dataset", "method", "artifact_kind", "bytes", "sha256", "original_path",
    "retained", "cleanup_verified",
)
row = {
    "dataset": passport["dataset"],
    "method": passport["method"],
    "artifact_kind": "graph_index",
    "bytes": passport["bytes"],
    "sha256": passport["sha256"],
    "original_path": passport["path"],
    "retained": "false",
    "cleanup_verified": "true",
}
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
PY_ARTIFACT_ROW
}

verify_controlled_partition_pair() {
  local dataset_root="$BATCH_ROOT/hot_path/$DATASET"
  python3 - "$dataset_root/cagra_nn_descent/run_1/payload/partition_count_audit.json" \
    "$dataset_root/tagore_cagra/run_1/payload/partition_count_audit.json" \
    "$dataset_root/controlled_partition_count_match.json" <<'PY_PARTITION_PAIR'
import json
from pathlib import Path
import sys

cagra_path, tagore_path, output = map(Path, sys.argv[1:])
cagra = json.loads(cagra_path.read_text())
tagore = json.loads(tagore_path.read_text())
accepted = (
    cagra.get("npartitions_requested") == 0
    and tagore.get("npartitions_requested") == 0
    and cagra.get("partition_policy") == "ace_auto"
    and tagore.get("partition_policy") == "ace_auto"
    and int(cagra.get("npartitions_actual", 0)) > 0
    and cagra.get("npartitions_actual") == tagore.get("npartitions_actual")
)
record = {
    "schema": "five_method_controlled_partition_pair_v1",
    "accepted": accepted,
    "cagra_npartitions_actual": cagra.get("npartitions_actual"),
    "tagore_npartitions_actual": tagore.get("npartitions_actual"),
}
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if not accepted:
    raise RuntimeError(f"CAGRA/Tagore partition counts do not match: {record}")
PY_PARTITION_PAIR
}

import_ivf_results() {
  local datasets_csv
  datasets_csv="$(IFS=,; printf '%s' "${ACTIVE_DATASETS[*]}")"
  python3 - "$IVF_REUSE_ROOT" "$BATCH_ROOT" "$DATA_ROOT" "$datasets_csv" <<'PY_IVF_IMPORT'
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

source, destination, data_root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
datasets = [value for value in sys.argv[4].split(",") if value]
contracts = {
    "wiki35m": (35_000_000, 1024, 32768, "wiki-1024-35M/en_base.fbin", "wiki-1024-35M/query.fbin", "wiki-1024-35M/gt100.bin"),
    "laion100m": (100_000_000, 512, 65536, "laion-512-100M/base.fbin", "laion-512-100M/query.fbin", "laion-512-100M/gt100.bin"),
    "msmarco70m": (70_000_000, 768, 65536, "msmarco-768-70M/base.fbin", "msmarco-768-114M/query.fbin", "msmarco-768-70M/gt100.bin"),
    "deep100m": (100_000_000, 96, 131072, "deep-96-100M/base.100M.fbin", "deep-96-100M/query.public.10K.fbin", "deep-96-100M/groundtruth.public.10K.ibin"),
}
if not datasets or any(dataset not in contracts for dataset in datasets):
    raise RuntimeError(f"invalid imported dataset list: {datasets}")

build_path = source / "build_summary.tsv"
summary_path = source / "summary.md"
if not build_path.is_file() or not summary_path.is_file():
    raise RuntimeError(f"IVF source is incomplete: {source}")
with build_path.open(newline="") as stream:
    build_rows = {row["dataset"]: row for row in csv.DictReader(stream, delimiter="\t")}

construction_fields = (
    "dataset", "method", "execution_origin", "status", "external_wall_seconds",
    "peak_dram_bytes", "read_bytes", "write_bytes", "peak_temp_bytes",
    "index_bytes", "source_root",
)
sweep_fields = (
    "dataset", "method", "execution_origin", "parameter_kind", "parameter_value",
    "repeat", "recall_at_10", "qps", "source_root",
)
search_fields = (
    "dataset", "method", "execution_origin", "parameter_kind", "parameter_value",
    "recall_at_10", "qps_mean", "qps_sample_std", "qps_median", "qps_min",
    "qps_max", "repeat_count", "source_root",
)
construction, sweep, search = [], [], []
hashed = [build_path, summary_path]
for dataset in datasets:
    rows, dim, nlist, base_suffix, query_suffix, gt_suffix = contracts[dataset]
    run_root = source / "hot_path" / dataset / "ivf_rabitq_exact" / "run_1"
    status_path = run_root / "status.json"
    cleanup_path = run_root / "cleanup_summary.txt"
    manifest_path = run_root / "dataset_manifest.json"
    raw_path = run_root / "search_sweep.tsv"
    for path in (status_path, cleanup_path, manifest_path, raw_path):
        if not path.is_file():
            raise RuntimeError(f"missing IVF evidence: {path}")
    status = json.loads(status_path.read_text())
    if status.get("phase") != "accepted" or int(status.get("exit_code", -1)) != 0:
        raise RuntimeError(f"IVF run is not accepted: {run_root}")
    cleanup = dict(
        line.split("=", 1) for line in cleanup_path.read_text().splitlines() if "=" in line
    )
    if cleanup.get("verified") != "true":
        raise RuntimeError(f"IVF cleanup is not verified: {run_root}")
    manifest = json.loads(manifest_path.read_text())
    base = manifest.get("base", {})
    query = manifest.get("query", {})
    ground_truth = manifest.get("ground_truth", {})
    expected_base = str(data_root / base_suffix)
    if (base.get("path"), int(base.get("rows", -1)), int(base.get("dim", -1))) != (expected_base, rows, dim):
        raise RuntimeError(f"IVF dataset manifest mismatch for {dataset}")
    if (
        query.get("path") != str(data_root / query_suffix)
        or int(query.get("rows", -1)) != 10_000
        or int(query.get("dim", -1)) != dim
        or ground_truth.get("path") != str(data_root / gt_suffix)
        or int(ground_truth.get("rows", -1)) != 10_000
        or int(ground_truth.get("width", -1)) < 10
    ):
        raise RuntimeError(f"IVF query/ground-truth manifest mismatch for {dataset}")
    build = build_rows.get(dataset)
    if build is None:
        raise RuntimeError(f"IVF build row is missing for {dataset}")
    if (
        int(build["rows"]) != rows
        or int(build["dim"]) != dim
        or int(build["nlist"]) != nlist
        or int(build["bits_per_dim"]) != 7
        or build["fast_quantize"].lower() != "false"
        or build["status"] != "accepted"
    ):
        raise RuntimeError(f"IVF build contract mismatch for {dataset}")
    construction.append({
        "dataset": dataset,
        "method": "ivf_rabitq",
        "execution_origin": "reused",
        "status": "accepted",
        "external_wall_seconds": build["external_wall_seconds"],
        "peak_dram_bytes": build["peak_process_tree_rss_bytes"],
        "read_bytes": "",
        "write_bytes": "",
        "peak_temp_bytes": "",
        "index_bytes": int(build["main_index_bytes"]) + int(build["hnsw_index_bytes"]),
        "source_root": str(source),
    })
    with raw_path.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream, delimiter="\t"))
    groups = {}
    for row in all_rows:
        if row.get("dataset") != dataset:
            raise RuntimeError(f"IVF search dataset mismatch in {raw_path}")
        if int(row["repeat"]) not in (1, 2):
            continue
        if int(row["query_workers"]) != 8 or row["rerank"].lower() != "false":
            raise RuntimeError(f"IVF search contract mismatch in {raw_path}")
        groups.setdefault(int(row["nprobe"]), []).append(row)
    if not groups:
        raise RuntimeError(f"IVF search sweep is empty for {dataset}")
    for nprobe in sorted(groups):
        group = groups[nprobe]
        if sorted(int(row["repeat"]) for row in group) != [1, 2]:
            raise RuntimeError(f"IVF nprobe={nprobe} lacks repetitions 1 and 2")
        recalls = [float(row["recall_at_10"]) for row in group]
        qps = [float(row["qps"]) for row in group]
        if max(recalls) - min(recalls) > 1e-12:
            raise RuntimeError(f"IVF recall changed at nprobe={nprobe}")
        if not all(math.isfinite(value) and value > 0 for value in qps):
            raise RuntimeError(f"IVF QPS is invalid at nprobe={nprobe}")
        for row in group:
            sweep.append({
                "dataset": dataset,
                "method": "ivf_rabitq",
                "execution_origin": "reused",
                "parameter_kind": "nprobe",
                "parameter_value": nprobe,
                "repeat": row["repeat"],
                "recall_at_10": row["recall_at_10"],
                "qps": row["qps"],
                "source_root": str(source),
            })
        search.append({
            "dataset": dataset,
            "method": "ivf_rabitq",
            "execution_origin": "reused",
            "parameter_kind": "nprobe",
            "parameter_value": nprobe,
            "recall_at_10": recalls[0],
            "qps_mean": statistics.mean(qps),
            "qps_sample_std": statistics.stdev(qps),
            "qps_median": statistics.median(qps),
            "qps_min": min(qps),
            "qps_max": max(qps),
            "repeat_count": 2,
            "source_root": str(source),
        })
    hashed.extend((status_path, cleanup_path, manifest_path, raw_path))

destination.mkdir(parents=True, exist_ok=True)
def write(name, fields, rows):
    path = destination / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

write("ivf_reused_construction.tsv", construction_fields, construction)
write("ivf_reused_search_sweep.tsv", sweep_fields, sweep)
write("ivf_reused_search_summary.tsv", search_fields, search)
(destination / "ivf_reused_provenance.json").write_text(json.dumps({
    "schema": "five_method_ivf_reuse_v1",
    "execution_origin": "reused",
    "source_root": str(source),
    "source_run_id": source.name,
    "datasets": datasets,
    "raw_repetitions_used": [1, 2],
    "executed_by_current_batch": False,
}, indent=2, sort_keys=True) + "\n")
with (destination / "ivf_reused_source_sha256.tsv").open("w", newline="") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("sha256", "path"))
    for path in hashed:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        writer.writerow((digest, str(path)))

schedule_path = destination / "formal_schedule.tsv"
if schedule_path.is_file():
    with schedule_path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        schedule = list(reader)
    for row in schedule:
        if row["dataset"] in datasets and row["method"] == "ivf_rabitq":
            row["state"] = "reuse_accepted"
            row["message"] = "imported repetitions 1 and 2; no execution"
    temporary = schedule_path.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(schedule)
    temporary.replace(schedule_path)
PY_IVF_IMPORT
}

generate_batch_report() {
  python3 - "$BATCH_ROOT" <<'PY_BATCH_REPORT'
import csv
from pathlib import Path
import sys

root = Path(sys.argv[1])
construction_fields = (
    "dataset", "method", "execution_origin", "status", "external_wall_seconds",
    "peak_dram_bytes", "read_bytes", "write_bytes", "peak_temp_bytes",
    "index_bytes", "source_root",
)
search_fields = (
    "dataset", "method", "execution_origin", "parameter_kind", "parameter_value",
    "recall_at_10", "qps_mean", "qps_sample_std", "qps_median", "qps_min",
    "qps_max", "repeat_count", "source_root",
)
method_order = {
    "shear": 0,
    "cagra_nn_descent": 1,
    "tagore_cagra": 2,
    "tagore_vamana": 3,
    "ivf_rabitq": 4,
}
dataset_order = {name: index for index, name in enumerate(("wiki35m", "laion100m", "msmarco70m", "deep100m"))}
labels = {
    "shear": "SHEAR",
    "cagra_nn_descent": "CAGRA NN-Descent",
    "tagore_cagra": "Tagore-CAGRA",
    "tagore_vamana": "Tagore-Vamana",
    "ivf_rabitq": "IVF-RaBitQ",
}

def read(path, expected):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected):
            raise RuntimeError(f"schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)

def write(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

construction = read(root / "graph_construction_summary.tsv", construction_fields)
construction += read(root / "ivf_reused_construction.tsv", construction_fields)
construction.sort(key=lambda row: (dataset_order.get(row["dataset"], 99), method_order.get(row["method"], 99)))
search = read(root / "graph_search_summary.tsv", search_fields)
search += read(root / "ivf_reused_search_summary.tsv", search_fields)
search.sort(key=lambda row: (
    dataset_order.get(row["dataset"], 99),
    method_order.get(row["method"], 99),
    row["parameter_kind"],
    int(row["parameter_value"]),
))
write(root / "construction_summary.tsv", construction_fields, construction)
write(root / "search_summary.tsv", search_fields, search)
schedule = []
schedule_path = root / "formal_schedule.tsv"
if schedule_path.is_file():
    with schedule_path.open(newline="") as stream:
        schedule = list(csv.DictReader(stream, delimiter="\t"))

def number(value, digits=3):
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)

def integer(value):
    if value in (None, ""):
        return "n/a"
    try:
        return f"{int(float(value)):,}"
    except ValueError:
        return str(value)

lines = [
    "# Five-Method Four-Dataset Construction and Search Report",
    "",
    "## Contract and provenance",
    "",
    "The graph rows use the current batch's hot-only, uncapped construction transactions and the shared `rabitq1-adaptive-exact` search backend. Graph search uses Recall@10, eight threads, mmap, one warmup, and two measured repetitions at each L.",
    "",
    "IVF-RaBitQ was not executed in this batch; its accepted construction and search evidence was reused from the provenance-bound source batch, and QPS was recomputed from raw repetitions 1 and 2 only.",
    "",
    "Graph `L` and IVF `nprobe` remain distinct parameters. Blank I/O fields mean that the reused source did not expose a directly comparable counter; they are not zero.",
    "",
    "## Completion and reuse state",
    "",
    "| Dataset | Method | Origin | State |",
    "|---|---|---|---|",
]
if schedule:
    for row in schedule:
        lines.append(
            f"| {row['dataset']} | {labels.get(row['method'], row['method'])} | "
            f"{row['execution_origin']} | {row['state']} |"
        )
else:
    for row in construction:
        lines.append(f"| {row['dataset']} | {labels.get(row['method'], row['method'])} | {row['execution_origin']} | {row['status']} |")

lines.extend((
    "",
    "## Construction comparison",
    "",
    "External wall time is the common E2E construction measure. Method-local GPU and phase timers remain in each transaction directory as diagnostics.",
    "",
    "| Dataset | Method | Origin | E2E (s) | Peak DRAM (B) | Read (B) | Write (B) | Peak temp (B) | Index (B) |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
))
for row in construction:
    lines.append(
        f"| {row['dataset']} | {labels.get(row['method'], row['method'])} | {row['execution_origin']} | "
        f"{number(row['external_wall_seconds'])} | {integer(row['peak_dram_bytes'])} | "
        f"{integer(row['read_bytes'])} | {integer(row['write_bytes'])} | "
        f"{integer(row['peak_temp_bytes'])} | {integer(row['index_bytes'])} |"
    )

lines.extend(("", "## Recall--QPS", ""))
for dataset in sorted({row["dataset"] for row in search}, key=lambda value: dataset_order.get(value, 99)):
    lines.extend((
        f"### {dataset}",
        "",
        "| Method | Origin | Parameter | Value | Recall@10 | QPS mean | QPS sample SD | QPS median |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ))
    for row in (candidate for candidate in search if candidate["dataset"] == dataset):
        lines.append(
            f"| {labels.get(row['method'], row['method'])} | {row['execution_origin']} | "
            f"{row['parameter_kind']} | {row['parameter_value']} | {number(row['recall_at_10'], 6)} | "
            f"{number(row['qps_mean'])} | {number(row['qps_sample_std'])} | {number(row['qps_median'])} |"
        )
    lines.append("")

lines.extend((
    "## Diagnostics and exclusions",
    "",
    "- SHEAR is accepted only when `soft_tail_audit.json` confirms legacy alpha 1.0, `priority_alpha=1.02`, and a three-edge per-row backfill cap.",
    "- CAGRA NN-Descent and Tagore-CAGRA retain their run-local partition and solver audits; the dataset-level pair audit requires the same positive ACE-auto partition count.",
    "- Reused IVF-RaBitQ rows do not claim graph-build physical I/O or temporary-space measurements when the source schema did not provide them.",
    "- Failed graph transactions retain their artifacts and stop the fixed schedule; they are not silently retried or parameter-tuned.",
    "",
))
temporary = root / "report.md.tmp"
temporary.write_text("\n".join(lines))
temporary.replace(root / "report.md")
PY_BATCH_REPORT
}

validate_dataset_headers() {
  python3 - "$BASE" "$QUERY" "$GT" "$ROWS" "$DIM" <<'PY_DATASET_HEADERS'
from pathlib import Path
import struct
import sys

base, query, gt = map(Path, sys.argv[1:4])
rows, dim = map(int, sys.argv[4:])
for path in (base, query, gt):
    if not path.is_file():
        raise FileNotFoundError(path)
def header(path):
    raw = path.open("rb").read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II", raw)
if header(base) != (rows, dim):
    raise RuntimeError(f"base header mismatch: {base}")
query_shape = header(query)
if query_shape[1] != dim:
    raise RuntimeError(f"query dimension mismatch: {query}")
gt_shape = header(gt)
if gt_shape[0] != query_shape[0] or gt_shape[1] < 10:
    raise RuntimeError(f"ground-truth header mismatch: {gt}")
if base.stat().st_size != 8 + rows * dim * 4:
    raise RuntimeError(f"base byte length mismatch: {base}")
PY_DATASET_HEADERS
}

verify_common_prerequisites() {
  require_commands bash python3 cargo git sha256sum jq dd sync sudo nvidia-smi \
    stat du awk sed find flock realpath /usr/bin/time
  local path
  for path in \
    "$SHEAR_RUNBOOK" "$WIKI_ACE_RUNBOOK" "$LAION_ACE_RUNBOOK" \
    "$MSMARCO_ACE_RUNBOOK" "$DEEP_ACE_RUNBOOK" "$UNIFIED_SEARCH_RUNBOOK" \
    "$IVF_SCHEMA_RUNBOOK" "$SIDECAR_MANIFEST" "$TEST_SCRIPT" \
    "$SHEAR_ROOT/target/release/build_shear_index" "$SEARCH_BIN" \
    "$SHEAR_ROOT/target/release/graph_quality"; do
    require_file "$path"
  done
}

exclusive_host_gate() {
  local active
  sudo -n true >/dev/null 2>&1 || die "passwordless sudo is required for cache control"
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)"
  [[ -z "$active" ]] || die "GPU has active compute processes: $active"
  mkdir -p "$(dirname "$LOCK_FILE")"
  exec {LOCK_FD}>"$LOCK_FILE"
  flock -n "$LOCK_FD" || die "another unified graph batch holds $LOCK_FILE"
}

initialize_batch() {
  fresh_dir "$BATCH_ROOT"
  BATCH_INITIALIZED=1
  mkdir -p "$BATCH_ROOT/provenance"
  initialize_schedule
  prepare_runbook_libraries
  write_build_wrappers
  record_provenance
  import_ivf_results
  generate_batch_report
  write_status "$BATCH_ROOT/status.json" initialized 0 initialization "batch initialized"
}

run_method_transaction() {
  local method="$1"
  local method_root="$BATCH_ROOT/hot_path/$DATASET/$method/run_1"
  CURRENT_METHOD_ROOT="$method_root"
  CURRENT_STAGE="initializing:$DATASET:$method"
  fresh_dir "$method_root"
  cp "$BATCH_ROOT/hot_path/$DATASET/sidecar_passport.json" "$method_root/sidecar_passport.json"
  write_status "$method_root/transaction_status.json" running 0 "$CURRENT_STAGE" ""
  update_schedule "$DATASET" "$method" running "$CURRENT_STAGE"

  CURRENT_STAGE="building:$DATASET:$method"
  write_status "$method_root/transaction_status.json" running 0 "$CURRENT_STAGE" ""
  update_schedule "$DATASET" "$method" running "$CURRENT_STAGE"
  run_build_adapter "$method" "$method_root"

  CURRENT_STAGE="build_validation:$DATASET:$method"
  validate_graph_quality_gate "$method_root"
  record_graph_passport "$method" "$method_root"

  CURRENT_STAGE="searching:$DATASET:$method"
  write_status "$method_root/transaction_status.json" running 0 "$CURRENT_STAGE" ""
  update_schedule "$DATASET" "$method" running "$CURRENT_STAGE"
  run_graph_search "$method" "$method_root"

  CURRENT_STAGE="reporting:$DATASET:$method"
  if [[ "$method" == tagore_cagra ]]; then
    verify_controlled_partition_pair
  fi
  write_graph_construction_row "$method" "$method_root"

  CURRENT_STAGE="cleaning:$DATASET:$method"
  write_status "$method_root/transaction_status.json" running 0 "$CURRENT_STAGE" ""
  cleanup_successful_method "$method" "$method_root"
  write_artifact_row "$method" "$method_root"

  append_tsv "$method_root/construction_row.tsv" "$BATCH_ROOT/graph_construction_summary.tsv"
  append_tsv "$method_root/graph_search_sweep.tsv" "$BATCH_ROOT/graph_search_sweep.tsv"
  append_tsv "$method_root/graph_search_summary.tsv" "$BATCH_ROOT/graph_search_summary.tsv"
  append_tsv "$method_root/artifact_row.tsv" "$BATCH_ROOT/artifact_manifest.tsv"
  write_status "$method_root/transaction_status.json" accepted 0 accepted "cleanup verified"
  update_schedule "$DATASET" "$method" accepted "build, search, report, and cleanup verified"
  generate_batch_report
  CURRENT_METHOD_ROOT=""
  CURRENT_STAGE="idle"
}

run_dataset() {
  local method
  load_dataset "$1"
  validate_dataset_headers
  select_sidecar
  for method in "${GRAPH_METHODS[@]}"; do
    run_method_transaction "$method"
  done
}

run_formal() {
  local dataset
  for dataset in "${FORMAL_DATASETS[@]}"; do
    run_dataset "$dataset"
  done
}

on_formal_error() {
  local rc=$?
  trap - ERR
  if [[ -n "$CURRENT_METHOD_ROOT" && -d "$CURRENT_METHOD_ROOT" ]]; then
    write_status "$CURRENT_METHOD_ROOT/transaction_status.json" failed "$rc" "$CURRENT_STAGE" \
      "scene retained; no retry" || true
    if [[ -n "$DATASET" ]]; then
      local method
      method="$(basename "$(dirname "$CURRENT_METHOD_ROOT")")"
      update_schedule "$DATASET" "$method" failed "$CURRENT_STAGE; scene retained" || true
    fi
  fi
  if [[ "$BATCH_INITIALIZED" -eq 1 ]]; then
    write_status "$BATCH_ROOT/status.json" failed "$rc" "$CURRENT_STAGE" \
      "formal schedule stopped" || true
    generate_batch_report || true
  fi
  log "formal execution failed at $CURRENT_STAGE (exit $rc); scene retained"
  exit "$rc"
}

run_formal_entry() {
  verify_common_prerequisites
  exclusive_host_gate
  initialize_batch
  trap on_formal_error ERR
  run_formal
  trap - ERR
  write_status "$BATCH_ROOT/status.json" accepted 0 complete "fixed schedule completed"
  generate_batch_report
  log "accepted unified batch: $BATCH_ROOT"
}

run_single_entry() {
  ACTIVE_DATASETS=("$1")
  verify_common_prerequisites
  exclusive_host_gate
  initialize_batch
  trap on_formal_error ERR
  run_dataset "$1"
  trap - ERR
  write_status "$BATCH_ROOT/status.json" accepted 0 complete "single-dataset schedule completed"
  generate_batch_report
  log "accepted unified dataset batch: $BATCH_ROOT"
}

run_prepare() {
  require_commands bash python3 cargo
  (cd "$SHEAR_ROOT" && cargo build --locked --release -p build_shear_index --features build_shear_index/cuda)
  (cd "$SHEAR_ROOT" && cargo build --locked --release -p search_shear_index --bin search_shear_index --bin graph_quality)
  (cd "$SHEAR_ROOT" && cargo test -p search_shear_index --bin search_shear_index)
  verify_common_prerequisites
  initialize_batch
  write_status "$BATCH_ROOT/status.json" prepared 0 prepare "binaries and provenance frozen; no experiment run"
}

run_preflight() {
  verify_common_prerequisites
  exclusive_host_gate
  initialize_batch
  local dataset
  for dataset in "${ACTIVE_DATASETS[@]}"; do
    load_dataset "$dataset"
    validate_dataset_headers
    select_sidecar >/dev/null
  done
  bash -n "$BATCH_ROOT/provenance/wrappers/shear.sh"
  bash -n "$BATCH_ROOT/provenance/wrappers/ace.sh"
  bash -n "$BATCH_ROOT/provenance/wrappers/tagore_vamana.sh"
  write_status "$BATCH_ROOT/status.json" preflight_accepted 0 preflight \
    "inputs, sidecars, reused IVF evidence, wrappers, sudo, and GPU gate accepted"
}

run_smoke() {
  python3 "$TEST_SCRIPT"
}

run_summarize() {
  require_dir "$BATCH_ROOT"
  require_file "$BATCH_ROOT/formal_schedule.tsv"
  mapfile -t ACTIVE_DATASETS < <(
    python3 - "$BATCH_ROOT/formal_schedule.tsv" <<'PY_SUMMARY_DATASETS'
import csv
from pathlib import Path
import sys

order = ("wiki35m", "laion100m", "msmarco70m", "deep100m")
with Path(sys.argv[1]).open(newline="") as stream:
    present = {row["dataset"] for row in csv.DictReader(stream, delimiter="\t")}
for dataset in order:
    if dataset in present:
        print(dataset)
PY_SUMMARY_DATASETS
  )
  ((${#ACTIVE_DATASETS[@]} > 0)) || die "formal schedule contains no datasets"
  import_ivf_results
  generate_batch_report
  log "summaries regenerated: $BATCH_ROOT"
}

case "$MODE" in
  help) usage ;;
  prepare) run_prepare ;;
  preflight) run_preflight ;;
  smoke) run_smoke ;;
  formal) run_formal_entry ;;
  wiki35m|laion100m|msmarco70m|deep100m) run_single_entry "$MODE" ;;
  summarize) run_summarize ;;
  *) usage >&2; exit 2 ;;
esac
```
<!-- END RUNBOOK_SCRIPT -->
