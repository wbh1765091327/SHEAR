# Tagore-Vamana Four-Dataset Adaptive Runbook

This independent runbook builds and searches Tagore-Vamana indexes for
Wiki35M, LAION100M, MSMARCO70M, and DEEP100M. It discovers a capacity-safe
DiskANN partition count from the actual two-assignment shard sizes, freezes the
first accepted materialization, and never reruns randomized partitioning for
the formal build.

The executable payload is extracted rather than copied manually:

```bash
python3 script/test_tagore_vamana_four_dataset_runbook.py --extract \
  >/tmp/tagore_vamana_four_dataset.sh
chmod 700 /tmp/tagore_vamana_four_dataset.sh
/tmp/tagore_vamana_four_dataset.sh help
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
OUTPUT_BASE="${OUTPUT_BASE:-/data_staging/wbh/tagore_vamana_four_dataset/runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_ROOT="${BATCH_ROOT:-$OUTPUT_BASE/$RUN_ID}"
DISKANN_ROOT="${DISKANN_ROOT:-/path/to/DiskANN-OOM}"
TAGORE_ROOT="${TAGORE_ROOT:-/path/to/Tagore}"
TAGORE_BUILD_ROOT="${TAGORE_BUILD_ROOT:-/data_staging/wbh/shear_paper_evaluation/build/tagore}"
TAGORE_PYTHON="${TAGORE_PYTHON:-/data_staging/wbh/tagore_wiki10m_measurement/venv/bin/python}"
CUVS_PYTHON="${CUVS_PYTHON:-/data_staging/wbh/cuvs_tagore_ace_venv/bin/python}"
PARTITION_BIN="${PARTITION_BIN:-$DISKANN_ROOT/build/apps/utils/partition_data}"
SEARCH_BIN="${SEARCH_BIN:-$SHEAR_ROOT/target/release/search_shear_index}"
GRAPH_QUALITY_BIN="${GRAPH_QUALITY_BIN:-$SHEAR_ROOT/target/release/graph_quality}"
SIDECAR_MANIFEST="${SIDECAR_MANIFEST:-/data_staging/wbh/unified_graph_rabitq1/sidecars/sidecar_manifest.tsv}"
TEST_SCRIPT="$SHEAR_ROOT/script/test_tagore_vamana_four_dataset_runbook.py"

FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)
ACTIVE_DATASETS=("${FORMAL_DATASETS[@]}")
TAGORE_K=96
TAGORE_FINAL_DEGREE=64
TAGORE_BUFFER_SIZE=16
PARTITION_REPLICATION=2
PARTITION_SAMPLE_RATE=0.01
MAX_DISCOVERY_ATTEMPTS="${MAX_DISCOVERY_ATTEMPTS:-8}"
MAX_DISCOVERY_CLUSTERS="${MAX_DISCOVERY_CLUSTERS:-4096}"
REFERENCE_GPU_MEMORY_MIB=24564
SESSION_ID="${SESSION_ID:-}"

DATASET=""
ROWS=0
DIM=0
MAX_POINTS=0
BASE=""
QUERY=""
GT=""
SOURCE_RUNBOOK=""
SIDECAR=""

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

fresh_dir() {
  [[ ! -e "$1" ]] || die "refusing to overwrite: $1"
  mkdir -p "$1"
}

load_dataset() {
  DATASET="$1"
  case "$DATASET" in
    wiki35m)
      ROWS=35000000; DIM=1024; MAX_POINTS=1200000
      BASE="$DATA_ROOT/wiki-1024-35M/en_base.fbin"
      QUERY="$DATA_ROOT/wiki-1024-35M/query.fbin"
      GT="$DATA_ROOT/wiki-1024-35M/gt100.bin"
      SOURCE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_wiki35m_shear_cagra_tagore.md"
      ;;
    laion100m)
      ROWS=100000000; DIM=512; MAX_POINTS=3000000
      BASE="$DATA_ROOT/laion-512-100M/base.fbin"
      QUERY="$DATA_ROOT/laion-512-100M/query.fbin"
      GT="$DATA_ROOT/laion-512-100M/gt100.bin"
      SOURCE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_laion100m_shear_cagra_tagore.md"
      ;;
    msmarco70m)
      ROWS=70000000; DIM=768; MAX_POINTS=2500000
      BASE="$DATA_ROOT/msmarco-768-70M/base.fbin"
      QUERY="$DATA_ROOT/msmarco-768-114M/query.fbin"
      GT="$DATA_ROOT/msmarco-768-70M/gt100.bin"
      SOURCE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_msmarco70m_shear_cagra_tagore.md"
      ;;
    deep100m)
      ROWS=100000000; DIM=96; MAX_POINTS=5000000
      BASE="$DATA_ROOT/deep-96-100M/base.100M.fbin"
      QUERY="$DATA_ROOT/deep-96-100M/query.public.10K.fbin"
      GT="$DATA_ROOT/deep-96-100M/groundtruth.public.10K.ibin"
      SOURCE_RUNBOOK="$SHEAR_ROOT/docs/runbooks/p1_p2_shear_cagra_tagore.md"
      ;;
    *) die "unknown dataset: $DATASET" ;;
  esac
}

write_partition_policy() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_PARTITION_POLICY'
#!/usr/bin/env python3
import argparse


def round_up_8(value: int) -> int:
    if value <= 0:
        raise ValueError("cluster count must be positive")
    return ((value + 7) // 8) * 8


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
initial = subparsers.add_parser("initial")
initial.add_argument("--rows", type=int, required=True)
initial.add_argument("--max-points", type=int, required=True)
initial.add_argument("--buffer-size", type=int, required=True)
next_value = subparsers.add_parser("next")
next_value.add_argument("--current", type=int, required=True)
next_value.add_argument("--observed-max", type=int, required=True)
next_value.add_argument("--max-points", type=int, required=True)
args = parser.parse_args()

if args.command == "initial":
    lower_bound = (2 * args.rows + args.max_points - 1) // args.max_points
    print(round_up_8(max(args.buffer_size, lower_bound)))
else:
    numerator = args.current * args.observed_max * 110
    denominator = args.max_points * 100
    ratio_jump = (numerator + denominator - 1) // denominator
    print(round_up_8(max(args.current + 8, ratio_jump)))
PY_PARTITION_POLICY
  chmod 700 "$output"
}

write_partition_inspector() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_INSPECT_PARTITION'
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II", raw)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=Path, required=True)
parser.add_argument("--clusters", type=int, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--max-points", type=int, required=True)
parser.add_argument("--json", type=Path, required=True)
parser.add_argument("--counts", type=Path, required=True)
args = parser.parse_args()

if min(args.clusters, args.rows, args.dim, args.max_points) <= 0:
    raise RuntimeError("partition dimensions must be positive")

centroids_path = Path(f"{args.prefix}_centroids.bin")
centroid_rows, centroid_dim = read_header(centroids_path)
if (centroid_rows, centroid_dim) != (args.clusters, args.dim):
    raise RuntimeError(
        f"centroids header mismatch: {(centroid_rows, centroid_dim)} "
        f"!= {(args.clusters, args.dim)}"
    )
expected_centroid_bytes = 8 + args.clusters * args.dim * 4
if centroids_path.stat().st_size != expected_centroid_bytes:
    raise RuntimeError(f"centroids payload mismatch: {centroids_path}")

counts: list[int] = []
shards: list[dict[str, object]] = []
for cluster in range(args.clusters):
    data_path = Path(f"{args.prefix}_subshard-{cluster}.bin")
    ids_path = Path(f"{args.prefix}_subshard-{cluster}_ids_uint32.bin")
    n_data, data_dim = read_header(data_path)
    n_ids, id_dim = read_header(ids_path)
    if n_data == 0:
        raise RuntimeError(f"empty shard: {cluster}")
    if n_data != n_ids or data_dim != args.dim or id_dim != 1:
        raise RuntimeError(f"header mismatch in shard {cluster}")
    if data_path.stat().st_size != 8 + n_data * args.dim * 4:
        raise RuntimeError(f"data payload mismatch in shard {cluster}")
    if ids_path.stat().st_size != 8 + n_ids * 4:
        raise RuntimeError(f"id payload mismatch in shard {cluster}")
    counts.append(n_data)
    shards.append(
        {
            "cluster": cluster,
            "count": n_data,
            "data": identity(data_path),
            "ids": identity(ids_path),
        }
    )

total = sum(counts)
if total != 2 * args.rows:
    raise RuntimeError(f"assignment count {total} != {2 * args.rows}")

metadata = {
    "schema": "tagore_vamana_partition_inspection_v1",
    "prefix": str(args.prefix.resolve()),
    "clusters": args.clusters,
    "rows": args.rows,
    "dim": args.dim,
    "max_points": args.max_points,
    "min_shard_points": min(counts),
    "max_shard_points": max(counts),
    "total_shard_points": total,
    "fits_capacity": max(counts) <= args.max_points,
    "centroids": identity(centroids_path),
    "centroids_sha256": digest(centroids_path),
    "shards": shards,
}

args.json.parent.mkdir(parents=True, exist_ok=True)
json_tmp = args.json.with_suffix(args.json.suffix + ".tmp")
json_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
os.replace(json_tmp, args.json)

args.counts.parent.mkdir(parents=True, exist_ok=True)
counts_tmp = args.counts.with_suffix(args.counts.suffix + ".tmp")
with counts_tmp.open("w") as stream:
    stream.write("cluster\tcount\tdata_bytes\tid_bytes\n")
    for shard in shards:
        stream.write(
            f"{shard['cluster']}\t{shard['count']}\t"
            f"{shard['data']['bytes']}\t{shard['ids']['bytes']}\n"
        )
os.replace(counts_tmp, args.counts)
PY_INSPECT_PARTITION
  chmod 700 "$output"
}

write_discovery_summary_tool() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_DISCOVERY_SUMMARY'
#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

attempt_paths = sorted(
    (args.dataset_root / "discovery/attempts").glob("attempt_*/attempt.json")
)
attempts = [json.loads(path.read_text()) for path in attempt_paths]
if not attempts:
    raise RuntimeError("discovery has no recorded attempts")
accepted = [value for value in attempts if value["status"] == "accepted"]
if len(accepted) != 1 or attempts[-1]["status"] != "accepted":
    raise RuntimeError("discovery must end with exactly one accepted attempt")
accepted_value = accepted[0]
manifest = json.loads(
    (args.dataset_root / "frozen/partition_manifest.json").read_text()
)
frozen_ns = int(manifest["frozen_ns"])
partition_sum = sum(float(value["partition_seconds"]) for value in attempts)
rejected_sum = sum(
    float(value["partition_seconds"])
    for value in attempts
    if value["status"] == "rejected_capacity"
)
summary = {
    "schema": "tagore_vamana_partition_discovery_v1",
    "session_id": accepted_value["session_id"],
    "attempt_count": len(attempts),
    "accepted_clusters": accepted_value["clusters"],
    "accepted_attempt": accepted_value["attempt"],
    "discovery_started_ns": attempts[0]["command_started_ns"],
    "accepted_command_started_ns": accepted_value["command_started_ns"],
    "frozen_ns": frozen_ns,
    "discovery_wall_seconds": (
        frozen_ns - attempts[0]["command_started_ns"]
    ) / 1e9,
    "all_attempt_partition_seconds": partition_sum,
    "rejected_partition_seconds": rejected_sum,
    "accepted_partition_seconds": float(accepted_value["partition_seconds"]),
}
summary["discovery_overhead_seconds"] = max(
    0.0,
    summary["discovery_wall_seconds"] - summary["all_attempt_partition_seconds"],
)
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(temporary, args.output)
PY_DISCOVERY_SUMMARY
  chmod 700 "$output"
}

write_tagore_timed_runner() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_TAGORE_TIMED_RUNNER'
#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time


def capture(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise RuntimeError("capture requires a command after --")

    paths = {
        "stdout": Path(args.stdout_log),
        "stderr": Path(args.stderr_log),
    }
    event_path = Path(args.events)
    for path in (*paths.values(), event_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to capture child output")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    pending = {"stdout": bytearray(), "stderr": bytearray()}

    with (
        paths["stdout"].open("wb", buffering=0) as stdout_log,
        paths["stderr"].open("wb", buffering=0) as stderr_log,
        event_path.open("w", encoding="utf-8", buffering=1) as events,
    ):
        raw_logs = {"stdout": stdout_log, "stderr": stderr_log}

        def emit(channel: str, raw: bytes) -> None:
            events.write(
                json.dumps(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "channel": channel,
                        "text": raw.decode("utf-8", errors="replace"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            events.flush()

        while selector.get_map():
            for key, _ in selector.select():
                channel = key.data
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    if pending[channel]:
                        emit(channel, bytes(pending[channel]))
                        pending[channel].clear()
                    continue
                raw_logs[channel].write(chunk)
                pending[channel].extend(chunk)
                while True:
                    newline = pending[channel].find(b"\n")
                    if newline < 0:
                        break
                    raw = bytes(pending[channel][:newline])
                    del pending[channel][: newline + 1]
                    if raw.endswith(b"\r"):
                        raw = raw[:-1]
                    emit(channel, raw)

    return process.wait()


def worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, args.build_root)
    import Tagore

    input_root = args.run_root / "tagore_input"
    tagore_root = args.run_root / "tagore_run"
    parameters = {
        "devicelist": [0],
        "dim": args.dim,
        "k": 96,
        "cluster_num": args.clusters,
        "max_points": args.max_points,
        "max_degree": 64,
        "buffer_size": 16,
        "num": args.rows,
        "threshold": 1.2,
        "final_degree": 64,
        "topm": 64,
    }
    print("TAGORE_PARAMS_START", flush=True)
    for key, value in parameters.items():
        print(f"{key}={value}", flush=True)
    print("TAGORE_PARAMS_END", flush=True)
    started = time.perf_counter()
    Tagore.largeIndex(
        [0],
        args.dim,
        96,
        args.clusters,
        args.max_points,
        str(input_root / "clusters/cluster_"),
        str(tagore_root / "local_index/local_"),
        str(tagore_root / "index/final.tagore.vamana"),
        64,
        16,
        args.rows,
        str(input_root / "centers.bin").encode(),
        1.2,
        64,
        64,
    )
    print(
        f"TAGORE_LARGEINDEX_SECONDS={time.perf_counter() - started:.6f}",
        flush=True,
    )
    return 0


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="mode", required=True)
capture_parser = subparsers.add_parser("capture")
capture_parser.add_argument("--stdout-log", required=True)
capture_parser.add_argument("--stderr-log", required=True)
capture_parser.add_argument("--events", required=True)
capture_parser.add_argument("command", nargs=argparse.REMAINDER)
worker_parser = subparsers.add_parser("worker")
worker_parser.add_argument("--build-root", required=True)
worker_parser.add_argument("--run-root", type=Path, required=True)
worker_parser.add_argument("--rows", type=int, required=True)
worker_parser.add_argument("--dim", type=int, required=True)
worker_parser.add_argument("--clusters", type=int, required=True)
worker_parser.add_argument("--max-points", type=int, required=True)
arguments = parser.parse_args()
if arguments.mode == "capture":
    raise SystemExit(capture(arguments))
raise SystemExit(worker(arguments))
PY_TAGORE_TIMED_RUNNER
  chmod 700 "$output"
}

write_timeline_analyzer() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_ANALYZE_TAGORE_TIMELINE'
#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re


def seconds(start: int, end: int) -> float:
    if end < start:
        raise RuntimeError(f"negative interval: {start} -> {end}")
    return (end - start) / 1e9


def insert_once(values: dict[int, int], ordinal: int, stamp: int, name: str) -> None:
    if ordinal in values:
        raise RuntimeError(f"duplicate {name} event for ordinal {ordinal}")
    values[ordinal] = stamp


parser = argparse.ArgumentParser()
parser.add_argument("--events", type=Path, required=True)
parser.add_argument("--clusters", type=int, required=True)
parser.add_argument("--buffer-size", type=int, required=True)
parser.add_argument("--largeindex-started-ns", type=int, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--timeline", type=Path, required=True)
args = parser.parse_args()
if args.clusters <= 0 or args.buffer_size <= 0:
    raise RuntimeError("clusters and buffer size must be positive")

patterns = {
    "moved": re.compile(r"Points num:(\d+) has been move to GPU"),
    "ready": re.compile(r"Waiting for transfering No\.(\d+) to CPU"),
    "transferred": re.compile(r"No\.(\d+) has been transfered to CPU"),
    "mapped": re.compile(r"No\.(\d+) has been mapped"),
    "wait_merge": re.compile(r"^Waiting to merge No\.(\d+)$"),
    "begin_merge": re.compile(r"^Begin to merge No\.(\d+)$"),
    "end_merge": re.compile(r"^End to merge No\.(\d+)$"),
    "index_size": re.compile(r"^Index size:(\d+)$"),
}
moved: list[int] = []
points: list[int] = []
ready: dict[int, int] = {}
transferred: dict[int, int] = {}
mapped: dict[int, int] = {}
wait_merge: dict[int, int] = {}
begin_merge: dict[int, int] = {}
end_merge: dict[int, int] = {}
final_begin: int | None = None
index_size_stamp: int | None = None
index_size_bytes: int | None = None

for line_number, line in enumerate(
    args.events.read_text(encoding="utf-8").splitlines(), start=1
):
    event = json.loads(line)
    if event.get("channel") != "stdout":
        continue
    stamp = int(event["monotonic_ns"])
    text = str(event["text"]).strip()
    match = patterns["moved"].search(text)
    if match:
        moved.append(stamp)
        points.append(int(match.group(1)))
        continue
    matched = False
    for name, values in (
        ("ready", ready),
        ("transferred", transferred),
        ("mapped", mapped),
        ("wait_merge", wait_merge),
        ("begin_merge", begin_merge),
        ("end_merge", end_merge),
    ):
        match = patterns[name].search(text)
        if match:
            insert_once(values, int(match.group(1)), stamp, name)
            matched = True
            break
    if matched:
        continue
    if text == "Begin to merge":
        if final_begin is not None:
            raise RuntimeError("duplicate unnumbered final merge begin")
        final_begin = stamp
        continue
    match = patterns["index_size"].search(text)
    if match:
        if index_size_stamp is not None:
            raise RuntimeError("duplicate index-size event")
        index_size_stamp = stamp
        index_size_bytes = int(match.group(1))

if len(moved) != args.clusters:
    raise RuntimeError(
        f"moved event count {len(moved)} != clusters {args.clusters}"
    )
expected = set(range(args.clusters))
for name, values in (
    ("ready", ready),
    ("transferred", transferred),
    ("mapped", mapped),
    ("wait_merge", wait_merge),
    ("begin_merge", begin_merge),
    ("end_merge", end_merge),
):
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise RuntimeError(f"{name} ordinal mismatch: missing={missing}, extra={extra}")
if final_begin is None or index_size_stamp is None or index_size_bytes is None:
    raise RuntimeError("final global merge events are incomplete")

rows: list[dict[str, object]] = []
for ordinal in range(args.clusters):
    eligible = ordinal >= args.buffer_size
    ready_before_previous = (
        eligible and ready[ordinal] < end_merge[ordinal - 1]
    )
    rows.append(
        {
            "ordinal": ordinal,
            "points": points[ordinal],
            "local_build_to_transfer_gate_seconds": seconds(
                moved[ordinal], ready[ordinal]
            ),
            "transfer_gate_combined_seconds": seconds(
                ready[ordinal], transferred[ordinal]
            ),
            "mapping_seconds": seconds(
                transferred[ordinal], mapped[ordinal]
            ),
            "cpu_wait_seconds": seconds(
                wait_merge[ordinal], begin_merge[ordinal]
            ),
            "online_merge_seconds": seconds(
                begin_merge[ordinal], end_merge[ordinal]
            ),
            "eligible_for_backpressure": eligible,
            "ready_before_previous_merge_end": ready_before_previous,
        }
    )

seconds(args.largeindex_started_ns, moved[0])
seconds(moved[0], end_merge[args.clusters - 1])
seconds(end_merge[args.clusters - 1], final_begin)
seconds(final_begin, index_size_stamp)
eligible_count = max(0, args.clusters - args.buffer_size)
summary = {
    "schema": "tagore_vamana_overlap_summary_v1",
    "clusters": args.clusters,
    "buffer_size": args.buffer_size,
    "eligible_backpressure_events": eligible_count,
    "gpu_ready_before_previous_merge_end_count": sum(
        bool(row["ready_before_previous_merge_end"]) for row in rows
    ),
    "cpu_wait_count": len(wait_merge),
    "cpu_wait_seconds": sum(float(row["cpu_wait_seconds"]) for row in rows),
    "online_merge_seconds": sum(
        float(row["online_merge_seconds"]) for row in rows
    ),
    "startup_seconds": seconds(args.largeindex_started_ns, moved[0]),
    "online_window_seconds": seconds(
        moved[0], end_merge[args.clusters - 1]
    ),
    "drain_flush_seconds": seconds(
        end_merge[args.clusters - 1], final_begin
    ),
    "final_global_post_seconds": seconds(final_begin, index_size_stamp),
    "index_size_bytes": index_size_bytes,
}

args.timeline.parent.mkdir(parents=True, exist_ok=True)
timeline_tmp = args.timeline.with_suffix(args.timeline.suffix + ".tmp")
columns = [
    "ordinal",
    "points",
    "local_build_to_transfer_gate_seconds",
    "transfer_gate_combined_seconds",
    "mapping_seconds",
    "cpu_wait_seconds",
    "online_merge_seconds",
    "eligible_for_backpressure",
    "ready_before_previous_merge_end",
]
with timeline_tmp.open("w", encoding="utf-8") as stream:
    stream.write("\t".join(columns) + "\n")
    for row in rows:
        stream.write(
            "\t".join(
                str(int(row[column]))
                if isinstance(row[column], bool)
                else str(row[column])
                for column in columns
            )
            + "\n"
        )
os.replace(timeline_tmp, args.timeline)

args.summary.parent.mkdir(parents=True, exist_ok=True)
summary_tmp = args.summary.with_suffix(args.summary.suffix + ".tmp")
summary_tmp.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(summary_tmp, args.summary)
PY_ANALYZE_TAGORE_TIMELINE
  chmod 700 "$output"
}

extract_runbook_library() {
  local source="$1" output="$2"
  python3 - "$source" "$output" <<'PY_EXTRACT_RUNBOOK'
from pathlib import Path
import os
import re
import sys


source, output = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
begin_text = "<!-- BEGIN " + "RUNBOOK_SCRIPT -->"
end_text = "<!-- END " + "RUNBOOK_SCRIPT -->"
begin_matches = list(
    re.finditer(rf"^{re.escape(begin_text)}[ \t]*$", text, re.MULTILINE)
)
end_matches = list(
    re.finditer(rf"^{re.escape(end_text)}[ \t]*$", text, re.MULTILINE)
)
if len(begin_matches) != 1 or len(end_matches) != 1:
    raise SystemExit(f"missing standalone runbook markers: {source}")
begin = begin_matches[0]
end = end_matches[0]
if end.start() <= begin.end():
    raise SystemExit(f"reversed standalone runbook markers: {source}")
region = text[begin.end() : end.start()]
fences = list(re.finditer(r"```bash\n(?P<body>.*?)\n```", region, re.DOTALL))
if len(fences) != 1:
    raise SystemExit(f"expected one Bash payload: {source}")
script = fences[0].group("body") + "\n"
terminal = re.search(r'^case "\$MODE" in\n', script, re.MULTILINE)
if terminal is None:
    raise SystemExit(f"missing terminal dispatch: {source}")
library = script[: terminal.start()]
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    if output.read_text(encoding="utf-8") != library:
        raise SystemExit(f"frozen source library changed: {output}")
else:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(library, encoding="utf-8")
    os.replace(temporary, output)
PY_EXTRACT_RUNBOOK
  chmod 700 "$output"
}

freeze_source_libraries() {
  local root="$BATCH_ROOT/provenance/source_libraries"
  local manifest="$root/manifest.tsv"
  local temporary="$root/manifest.tsv.tmp.$$"
  local dataset output source_sha library_sha
  mkdir -p "$root"
  printf 'dataset\tsource\tsource_sha256\tlibrary\tlibrary_sha256\n' \
    >"$temporary"
  for dataset in "${FORMAL_DATASETS[@]}"; do
    load_dataset "$dataset"
    require_file "$SOURCE_RUNBOOK"
    output="$root/$dataset.sh"
    extract_runbook_library "$SOURCE_RUNBOOK" "$output"
    source_sha="$(sha256sum "$SOURCE_RUNBOOK" | awk '{print $1}')"
    library_sha="$(sha256sum "$output" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$dataset" "$SOURCE_RUNBOOK" "$source_sha" "$output" "$library_sha" \
      >>"$temporary"
  done
  if [[ -f "$manifest" ]]; then
    cmp -s "$temporary" "$manifest" \
      || die "source library provenance changed: $manifest"
    rm -f -- "$temporary"
  else
    mv -- "$temporary" "$manifest"
  fi
}

write_frozen_partition_validator() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_VALIDATE_FROZEN_PARTITION'
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


IDENTITY_KEYS = ("path", "bytes", "device", "inode", "mtime_ns")


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II", raw)


def identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def require_identity(
    label: str, expected: dict[str, object], path: Path
) -> dict[str, int | str]:
    current = identity(path)
    normalized = {key: expected.get(key) for key in IDENTITY_KEYS}
    if current != normalized:
        raise RuntimeError(
            f"{label} identity changed: expected={normalized}, current={current}"
        )
    return current


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--max-points", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
expected = {
    "dataset": args.dataset,
    "base": str(args.base.resolve()),
    "rows": args.rows,
    "dim": args.dim,
    "max_points": args.max_points,
    "status": "accepted",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(
            f"frozen manifest mismatch for {key}: "
            f"{manifest.get(key)!r} != {value!r}"
        )
if manifest.get("schema") != "tagore_vamana_frozen_partition_v1":
    raise RuntimeError("unexpected frozen manifest schema")

base_rows, base_dim = read_header(args.base)
if (base_rows, base_dim) != (args.rows, args.dim):
    raise RuntimeError("base header changed")
if args.base.stat().st_size != 8 + args.rows * args.dim * 4:
    raise RuntimeError("base payload length changed")
require_identity("base", manifest["base_identity"], args.base)

clusters = int(manifest["clusters"])
prefix = Path(manifest["prefix"])
expected_partition = (args.manifest.parent / "partition").resolve()
if prefix.resolve().parent != expected_partition or prefix.name != "dataset":
    raise RuntimeError(f"frozen prefix escapes accepted partition: {prefix}")
centroids = Path(f"{prefix}_centroids.bin")
require_identity("centroids", manifest["centroids"], centroids)
centroid_rows, centroid_dim = read_header(centroids)
if (centroid_rows, centroid_dim) != (clusters, args.dim):
    raise RuntimeError("centroids header changed")
if centroids.stat().st_size != 8 + clusters * args.dim * 4:
    raise RuntimeError("centroids payload length changed")
if digest(centroids) != manifest["centroids_sha256"]:
    raise RuntimeError("centroids digest changed")

manifest_shards = manifest.get("shards")
if not isinstance(manifest_shards, list) or len(manifest_shards) != clusters:
    raise RuntimeError("frozen shard inventory is incomplete")
counts: list[int] = []
for ordinal, shard in enumerate(manifest_shards):
    if shard.get("cluster") != ordinal:
        raise RuntimeError(f"unexpected shard ordinal: {shard.get('cluster')}")
    data_path = Path(f"{prefix}_subshard-{ordinal}.bin")
    ids_path = Path(f"{prefix}_subshard-{ordinal}_ids_uint32.bin")
    require_identity(f"shard {ordinal} data", shard["data"], data_path)
    require_identity(f"shard {ordinal} ids", shard["ids"], ids_path)
    data_rows, data_dim = read_header(data_path)
    id_rows, id_dim = read_header(ids_path)
    if data_rows <= 0:
        raise RuntimeError(f"empty shard: {ordinal}")
    if data_rows != id_rows or data_dim != args.dim or id_dim != 1:
        raise RuntimeError(f"shard header changed: {ordinal}")
    if data_path.stat().st_size != 8 + data_rows * args.dim * 4:
        raise RuntimeError(f"shard data length changed: {ordinal}")
    if ids_path.stat().st_size != 8 + id_rows * 4:
        raise RuntimeError(f"shard ids length changed: {ordinal}")
    if data_rows != shard.get("count"):
        raise RuntimeError(f"shard count changed: {ordinal}")
    counts.append(data_rows)
if sum(counts) != 2 * args.rows:
    raise RuntimeError("frozen assignment total changed")
if max(counts) > args.max_points:
    raise RuntimeError("frozen partition exceeds capacity")

result = {
    "schema": "tagore_vamana_frozen_revalidation_v1",
    "valid": True,
    "dataset": args.dataset,
    "base": identity(args.base),
    "clusters": clusters,
    "rows": args.rows,
    "dim": args.dim,
    "max_points": args.max_points,
    "partition_prefix": str(prefix.resolve()),
    "total_shard_points": sum(counts),
    "min_shard_points": min(counts),
    "max_shard_points": max(counts),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, args.output)
print(clusters)
PY_VALIDATE_FROZEN_PARTITION
  chmod 700 "$output"
}

write_sidecar_selector() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_SELECT_SIDECAR'
#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
required = [
    "dataset",
    "base",
    "sidecar",
    "sidecar_bytes",
    "sha256",
    "format_version",
    "rows",
    "dim",
    "padded_dim",
    "validation_status",
]
with args.manifest.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    if reader.fieldnames != required:
        raise RuntimeError("sidecar manifest schema mismatch")
    matches = [row for row in reader if row["dataset"] == args.dataset]
if len(matches) != 1:
    raise RuntimeError(
        f"sidecar manifest row count is {len(matches)} for {args.dataset}"
    )
row = matches[0]
sidecar = Path(row["sidecar"])
if (
    Path(row["base"]).resolve() != args.base.resolve()
    or int(row["rows"]) != args.rows
    or int(row["dim"]) != args.dim
):
    raise RuntimeError(f"sidecar dataset contract mismatch for {args.dataset}")
if (
    row["format_version"] != "graph_rabitq1_v1"
    or row["validation_status"] != "accepted"
):
    raise RuntimeError(f"sidecar is not accepted for {args.dataset}")
if not sidecar.is_file() or sidecar.stat().st_size != int(row["sidecar_bytes"]):
    raise RuntimeError(f"sidecar file contract mismatch: {sidecar}")
digest = hashlib.sha256()
with sidecar.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != row["sha256"]:
    raise RuntimeError(f"sidecar SHA-256 mismatch: {sidecar}")
record = dict(row)
record.update(
    {
        "base": str(args.base.resolve()),
        "sidecar": str(sidecar.resolve()),
        "sidecar_bytes": int(row["sidecar_bytes"]),
        "rows": int(row["rows"]),
        "dim": int(row["dim"]),
        "padded_dim": int(row["padded_dim"]),
    }
)
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, args.output)
print(sidecar.resolve())
PY_SELECT_SIDECAR
  chmod 700 "$output"
}

write_search_row_tool() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_APPEND_SEARCH_ROW'
#!/usr/bin/env python3
import argparse
import csv
import hashlib
from pathlib import Path


FIELDS = (
    "dataset",
    "parameter_value",
    "repeat",
    "recall_at_10",
    "qps",
    "result_sha256",
    "total_one_bit_evaluations",
    "total_lower_bound_rejects",
    "total_navigation_exact_evaluations",
    "total_adaptive_exact_refinements",
    "total_exact_evaluations",
    "logical_exact_vector_bytes",
)
parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--l", type=int, required=True)
parser.add_argument("--repeat", type=int, required=True)
parser.add_argument("--metrics", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
with args.metrics.open(newline="", encoding="utf-8") as stream:
    metrics = list(csv.DictReader(stream, delimiter="\t"))
if len(metrics) != 1:
    raise RuntimeError(f"metrics row count is {len(metrics)}: {args.metrics}")
metric = metrics[0]
digest = hashlib.sha256()
with args.result.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)
row = {
    "dataset": args.dataset,
    "parameter_value": args.l,
    "repeat": args.repeat,
    "recall_at_10": metric["recall"],
    "qps": metric["qps"],
    "result_sha256": digest.hexdigest(),
}
for field in FIELDS[6:]:
    row[field] = metric.get(field, "")
with args.raw.open("a", newline="", encoding="utf-8") as stream:
    csv.DictWriter(
        stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
    ).writerow(row)
PY_APPEND_SEARCH_ROW
  chmod 700 "$output"
}

write_search_summarizer() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_SUMMARIZE_SEARCH'
#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import statistics


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with args.input.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
required = {
    "dataset",
    "parameter_value",
    "repeat",
    "recall_at_10",
    "qps",
    "result_sha256",
}
if not rows or not required.issubset(rows[0]):
    raise RuntimeError("search sweep schema mismatch")
groups: dict[int, list[dict[str, str]]] = {}
for row in rows:
    groups.setdefault(int(row["parameter_value"]), []).append(row)
expected = [25, 50, 100, 200, 400]
if sorted(groups) != expected:
    raise RuntimeError(f"unexpected L values: {sorted(groups)}")
fields = (
    "dataset",
    "parameter_value",
    "recall_at_10",
    "qps_mean",
    "qps_sample_std",
    "qps_median",
    "qps_min",
    "qps_max",
    "repeat_count",
    "result_sha256",
)
summary: list[dict[str, object]] = []
last_recall = -1.0
for value in expected:
    group = groups[value]
    if sorted(int(row["repeat"]) for row in group) != [1, 2]:
        raise RuntimeError(f"L={value} does not have repetitions 1 and 2")
    recalls = [float(row["recall_at_10"]) for row in group]
    qps = [float(row["qps"]) for row in group]
    hashes = {row["result_sha256"] for row in group}
    datasets = {row["dataset"] for row in group}
    if len(datasets) != 1:
        raise RuntimeError(f"L={value} mixes datasets")
    if len(hashes) != 1 or max(recalls) - min(recalls) > 1e-12:
        raise RuntimeError(f"L={value} repetitions changed result or recall")
    if not all(math.isfinite(number) and number > 0 for number in qps):
        raise RuntimeError(f"L={value} has invalid QPS")
    recall = recalls[0]
    if (
        not math.isfinite(recall)
        or not 0 <= recall <= 1
        or recall + 1e-12 < last_recall
    ):
        raise RuntimeError(f"L={value} has invalid or decreasing recall")
    if value == 25 and recall <= 0:
        raise RuntimeError("L=25 recall must be positive")
    last_recall = recall
    summary.append(
        {
            "dataset": group[0]["dataset"],
            "parameter_value": value,
            "recall_at_10": recall,
            "qps_mean": statistics.mean(qps),
            "qps_sample_std": statistics.stdev(qps),
            "qps_median": statistics.median(qps),
            "qps_min": min(qps),
            "qps_max": max(qps),
            "repeat_count": 2,
            "result_sha256": next(iter(hashes)),
        }
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
with temporary.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(summary)
temporary.replace(args.output)
PY_SUMMARIZE_SEARCH
  chmod 700 "$output"
}

write_graph_passport_tool() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_GRAPH_PASSPORT'
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--graph", type=Path, required=True)
parser.add_argument("--validation", type=Path, required=True)
parser.add_argument("--quality", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
validation = json.loads(args.validation.read_text(encoding="utf-8"))
if validation.get("valid") is not True:
    raise RuntimeError("graph structural validation was not accepted")
if not args.graph.is_file() or args.graph.stat().st_size <= 0:
    raise RuntimeError("graph is missing or empty")
if not args.quality.is_file() or args.quality.stat().st_size <= 0:
    raise RuntimeError("graph quality evidence is missing")
digest = hashlib.sha256()
with args.graph.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)
result = {
    "schema": "tagore_vamana_graph_passport_v1",
    "valid": True,
    "dataset": args.dataset,
    "path": str(args.graph),
    "resolved_path": str(args.graph.resolve()),
    "bytes": args.graph.stat().st_size,
    "sha256": digest.hexdigest(),
    "validation": str(args.validation.resolve()),
    "quality": str(args.quality.resolve()),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, args.output)
PY_GRAPH_PASSPORT
  chmod 700 "$output"
}

write_dataset_status_tool() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_DATASET_STATUS'
#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=Path, required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--state", required=True)
parser.add_argument("--session-id", required=True)
args = parser.parse_args()
history: list[dict[str, str]] = []
if args.path.exists():
    current = json.loads(args.path.read_text(encoding="utf-8"))
    history = list(current.get("history", []))
event = {
    "state": args.state,
    "session_id": args.session_id,
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
history.append(event)
value = {
    "schema": "tagore_vamana_dataset_status_v1",
    "dataset": args.dataset,
    "state": args.state,
    "session_id": args.session_id,
    "history": history,
}
args.path.parent.mkdir(parents=True, exist_ok=True)
temporary = args.path.with_suffix(args.path.suffix + ".tmp")
temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, args.path)
PY_DATASET_STATUS
  chmod 700 "$output"
}

write_dataset_summarizer() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_SUMMARIZE_DATASET'
#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ns(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.dataset_root
formal = root / "formal"
status = load_json(root / "status.json")
if status.get("state") != "accepted":
    raise RuntimeError("dataset is not accepted")
discovery = load_json(root / "discovery/discovery_summary.json")
overlap = load_json(formal / "overlap_summary.json")
graph = load_json(formal / "graph_passport.json")
if graph.get("valid") is not True:
    raise RuntimeError("graph passport is not accepted")
formal_session = (formal / "session_id.txt").read_text(encoding="utf-8").strip()
conversion_started = read_ns(formal / "conversion_started_ns.txt")
conversion_completed = read_ns(formal / "conversion_completed_ns.txt")
largeindex_started = read_ns(formal / "largeindex_started_ns.txt")
largeindex_completed = read_ns(formal / "largeindex_completed_ns.txt")
if not (
    conversion_started
    <= conversion_completed
    <= largeindex_started
    <= largeindex_completed
):
    raise RuntimeError("formal stage timestamps are not monotonic")
conversion_seconds = (conversion_completed - conversion_started) / 1e9
largeindex_seconds = (largeindex_completed - largeindex_started) / 1e9

with (formal / "search/search_summary.tsv").open(
    newline="", encoding="utf-8"
) as stream:
    search = list(csv.DictReader(stream, delimiter="\t"))
if [int(row["parameter_value"]) for row in search] != [25, 50, 100, 200, 400]:
    raise RuntimeError("search summary has incomplete L values")
if any(int(row["repeat_count"]) != 2 for row in search):
    raise RuntimeError("search summary has incomplete repetitions")

with (formal / "gpu_monitor/gpu_samples.csv").open(
    newline="", encoding="utf-8"
) as stream:
    gpu_rows = list(csv.DictReader(stream))
if not gpu_rows:
    raise RuntimeError("GPU monitor has no samples")
gpu_util = [float(row["gpu_util_percent"]) for row in gpu_rows]
gpu_memory = [float(row["memory_used_mib"]) for row in gpu_rows]
gpu_summary = {
    "sample_count": len(gpu_rows),
    "active_sample_count": sum(value > 0 for value in gpu_util),
    "active_sample_ratio": sum(value > 0 for value in gpu_util) / len(gpu_util),
    "mean_gpu_util_percent": sum(gpu_util) / len(gpu_util),
    "peak_memory_used_mib": max(gpu_memory),
}

same_session = discovery["session_id"] == formal_session
accepted_pipeline = (
    float(discovery["accepted_partition_seconds"])
    + conversion_seconds
    + largeindex_seconds
)
discovery_inclusive = (
    float(discovery["discovery_wall_seconds"])
    + conversion_seconds
    + largeindex_seconds
)
accepted_wall: float | str
full_wall: float | str
if same_session:
    accepted_wall = (
        largeindex_completed - int(discovery["accepted_command_started_ns"])
    ) / 1e9
    full_wall = (
        largeindex_completed - int(discovery["discovery_started_ns"])
    ) / 1e9
else:
    accepted_wall = "not_applicable"
    full_wall = "not_applicable"
summary = {
    "schema": "tagore_vamana_dataset_summary_v1",
    "dataset": args.dataset,
    "state": "accepted",
    "discovery_session_id": discovery["session_id"],
    "formal_session_id": formal_session,
    "same_session_continuous_wall": same_session,
    "attempt_count": discovery["attempt_count"],
    "accepted_clusters": discovery["accepted_clusters"],
    "discovery_wall_seconds": discovery["discovery_wall_seconds"],
    "all_attempt_partition_seconds": discovery[
        "all_attempt_partition_seconds"
    ],
    "rejected_partition_seconds": discovery["rejected_partition_seconds"],
    "accepted_partition_seconds": discovery["accepted_partition_seconds"],
    "discovery_overhead_seconds": discovery["discovery_overhead_seconds"],
    "conversion_seconds": conversion_seconds,
    "largeindex_seconds": largeindex_seconds,
    "gpu_cpu_pipeline_seconds": largeindex_seconds,
    "accepted_pipeline_seconds": accepted_pipeline,
    "discovery_inclusive_seconds": discovery_inclusive,
    "accepted_configuration_wall_seconds": accepted_wall,
    "full_discovery_execution_wall_seconds": full_wall,
    "conversion_started_ns": conversion_started,
    "conversion_completed_ns": conversion_completed,
    "largeindex_started_ns": largeindex_started,
    "largeindex_completed_ns": largeindex_completed,
    "overlap": overlap,
    "gpu_monitor": gpu_summary,
    "graph": graph,
    "search": search,
}
timing_fields = {
    key: value
    for key, value in summary.items()
    if key.endswith("_seconds")
    or key.endswith("_ns")
    or key
    in {
        "schema",
        "dataset",
        "discovery_session_id",
        "formal_session_id",
        "same_session_continuous_wall",
    }
}
timing_path = formal / "timing_summary.json"
timing_tmp = timing_path.with_suffix(timing_path.suffix + ".tmp")
timing_tmp.write_text(
    json.dumps(timing_fields, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(timing_tmp, timing_path)
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, args.output)
PY_SUMMARIZE_DATASET
  chmod 700 "$output"
}

write_batch_summarizer() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_SUMMARIZE_BATCH'
#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--batch-root", type=Path, required=True)
parser.add_argument("--datasets", nargs="+", required=True)
parser.add_argument("--tsv", type=Path, required=True)
parser.add_argument("--markdown", type=Path, required=True)
args = parser.parse_args()
fields = (
    "dataset",
    "state",
    "accepted_clusters",
    "attempt_count",
    "accepted_pipeline_seconds",
    "discovery_inclusive_seconds",
    "accepted_configuration_wall_seconds",
    "full_discovery_execution_wall_seconds",
    "largeindex_seconds",
)
rows: list[dict[str, object]] = []
for dataset in args.datasets:
    root = args.batch_root / dataset
    summary_path = root / "summary.json"
    if summary_path.exists():
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({field: value.get(field, "") for field in fields})
        continue
    state = "incomplete"
    status_path = root / "status.json"
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8")).get(
            "state", "incomplete"
        )
    row = {field: "" for field in fields}
    row.update({"dataset": dataset, "state": state})
    rows.append(row)
args.tsv.parent.mkdir(parents=True, exist_ok=True)
tsv_tmp = args.tsv.with_suffix(args.tsv.suffix + ".tmp")
with tsv_tmp.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
os.replace(tsv_tmp, args.tsv)
lines = [
    "# Tagore-Vamana adaptive four-dataset summary",
    "",
    "Stage sums and continuous wall clocks are separate quantities. "
    "Continuous walls are `not_applicable` when discovery and formal "
    "execution use different session IDs.",
    "",
    "| Dataset | State | P | Attempts | Accepted stage sum (s) | "
    "Discovery-inclusive stage sum (s) | Accepted continuous wall (s) | "
    "Full continuous wall (s) | largeIndex (s) |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {dataset} | {state} | {accepted_clusters} | {attempt_count} | "
        "{accepted_pipeline_seconds} | {discovery_inclusive_seconds} | "
        "{accepted_configuration_wall_seconds} | "
        "{full_discovery_execution_wall_seconds} | {largeindex_seconds} |"
        .format(**row)
    )
markdown_tmp = args.markdown.with_suffix(args.markdown.suffix + ".tmp")
markdown_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
os.replace(markdown_tmp, args.markdown)
PY_SUMMARIZE_BATCH
  chmod 700 "$output"
}

write_preflight_dataset_tool() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  cat >"$output" <<'PY_PREFLIGHT_DATASET'
#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil
import struct


def header(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short header: {path}")
    return struct.unpack("<II", raw)


def nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(f"no existing parent for staging path: {path}")
        candidate = candidate.parent
    return candidate


def file_record(path: Path, shape: tuple[int, int]) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "shape": list(shape),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--query", type=Path, required=True)
parser.add_argument("--gt", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--max-points", type=int, required=True)
parser.add_argument("--staging-path", type=Path, required=True)
parser.add_argument("--available-bytes", type=int)
parser.add_argument("--already-allocated-bytes", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.already_allocated_bytes < 0:
    raise RuntimeError("already allocated bytes must be nonnegative")
for path in (args.base, args.query, args.gt):
    if not path.is_file():
        raise FileNotFoundError(path)
base_header = header(args.base)
query_header = header(args.query)
gt_header = header(args.gt)
if base_header != (args.rows, args.dim):
    raise RuntimeError(f"base header mismatch: {base_header}")
if query_header[0] <= 0 or query_header[1] != args.dim:
    raise RuntimeError(f"query header mismatch: {query_header}")
if gt_header[0] != query_header[0] or gt_header[1] < 10:
    raise RuntimeError(f"ground-truth header mismatch: {gt_header}")
if args.base.stat().st_size != 8 + args.rows * args.dim * 4:
    raise RuntimeError("base payload length mismatch")
if args.query.stat().st_size != 8 + query_header[0] * args.dim * 4:
    raise RuntimeError("query payload length mismatch")
gt_payload = gt_header[0] * gt_header[1] * 4
if args.gt.stat().st_size not in {8 + gt_payload, 8 + 2 * gt_payload}:
    raise RuntimeError("ground-truth payload length mismatch")

replicated_partition = 2 * args.rows * (4 * args.dim + 4)
centers = 8 * args.rows
local_index_upper = 2 * args.rows * (4 + 64 * 4)
final_index_upper = 24 + args.rows * (4 + 64 * 4)
required = int(
    (
        replicated_partition
        + centers
        + local_index_upper
        + final_index_upper
    )
    * 1.10
) + 10 * 1024**3
staging_probe = nearest_existing(args.staging_path)
available = (
    args.available_bytes
    if args.available_bytes is not None
    else shutil.disk_usage(staging_probe).free
)
required_additional = max(0, required - args.already_allocated_bytes)
accepted = available >= required_additional
value = {
    "schema": "tagore_vamana_dataset_preflight_v1",
    "dataset": args.dataset,
    "accepted": accepted,
    "rows": args.rows,
    "dim": args.dim,
    "max_points": args.max_points,
    "base": file_record(args.base, base_header),
    "query": file_record(args.query, query_header),
    "ground_truth": file_record(args.gt, gt_header),
    "staging_probe": str(staging_probe),
    "available_free_bytes": available,
    "required_free_bytes": required,
    "already_allocated_bytes": args.already_allocated_bytes,
    "required_additional_free_bytes": required_additional,
    "space_components": {
        "replicated_partition_bytes": replicated_partition,
        "centers_bytes": centers,
        "local_index_upper_bytes": local_index_upper,
        "final_index_upper_bytes": final_index_upper,
        "fixed_headroom_bytes": 10 * 1024**3,
        "multiplicative_headroom": 1.10,
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, args.output)
if not accepted:
    raise SystemExit(
        f"insufficient staging space for {args.dataset}: "
        f"required_additional={required_additional}, "
        f"required_from_scratch={required}, available={available}"
    )
PY_PREFLIGHT_DATASET
  chmod 700 "$output"
}

populate_tools() {
  local root="$1"
  mkdir -p "$root"
  write_partition_policy "$root/partition_policy.py"
  write_partition_inspector "$root/inspect_partition.py"
  write_discovery_summary_tool "$root/summarize_discovery.py"
  write_tagore_timed_runner "$root/tagore_timed_runner.py"
  write_timeline_analyzer "$root/analyze_tagore_timeline.py"
  write_frozen_partition_validator "$root/validate_frozen_partition.py"
  write_sidecar_selector "$root/select_sidecar.py"
  write_search_row_tool "$root/append_search_row.py"
  write_search_summarizer "$root/summarize_search.py"
  write_graph_passport_tool "$root/write_graph_passport.py"
  write_dataset_status_tool "$root/write_dataset_status.py"
  write_dataset_summarizer "$root/summarize_dataset.py"
  write_batch_summarizer "$root/summarize_batch.py"
  write_preflight_dataset_tool "$root/preflight_dataset.py"
  (
    cd "$root"
    sha256sum ./*.py >manifest.sha256
    chmod 600 manifest.sha256
  )
}

write_tools() {
  local root="$BATCH_ROOT/provenance/tools"
  local candidate
  mkdir -p "$BATCH_ROOT/provenance"
  candidate="$(mktemp -d \
    "$BATCH_ROOT/provenance/.tools_candidate.XXXXXXXX")"
  populate_tools "$candidate"
  if [[ -e "$root" ]]; then
    "$CUVS_PYTHON" - "$root" "$candidate" <<'PY_VERIFY_IMMUTABLE_TOOLS'
from pathlib import Path
import stat
import sys


current, candidate = map(Path, sys.argv[1:])
current_names = sorted(
    path.relative_to(current) for path in current.rglob("*") if path.is_file()
)
candidate_names = sorted(
    path.relative_to(candidate)
    for path in candidate.rglob("*")
    if path.is_file()
)
if current_names != candidate_names:
    raise RuntimeError("immutable tool inventory changed")
for name in current_names:
    left = current / name
    right = candidate / name
    if left.read_bytes() != right.read_bytes():
        raise RuntimeError(f"immutable tool content changed: {name}")
    if stat.S_IMODE(left.stat().st_mode) != stat.S_IMODE(right.stat().st_mode):
        raise RuntimeError(f"immutable tool mode changed: {name}")
PY_VERIFY_IMMUTABLE_TOOLS
    [[ "$candidate" == \
      "$BATCH_ROOT/provenance/.tools_candidate."* ]] \
      || die "tool candidate path escaped provenance: $candidate"
    rm -rf -- "$candidate"
  else
    mv -- "$candidate" "$root"
  fi
}

freeze_orchestrator_sources() {
  local output="$BATCH_ROOT/provenance/orchestrator_sources.sha256"
  local temporary="$output.tmp.$$"
  local runbook_source="$SHEAR_ROOT/docs/runbooks/tagore_vamana_four_dataset.md"
  require_file "$runbook_source"
  require_file "$TEST_SCRIPT"
  sha256sum "$runbook_source" "$TEST_SCRIPT" >"$temporary"
  if [[ -f "$output" ]]; then
    cmp -s "$temporary" "$output" \
      || die "immutable orchestrator source changed: $output"
    rm -f -- "$temporary"
  else
    mv -- "$temporary" "$output"
  fi
}

initialize_batch() {
  if [[ -z "$SESSION_ID" ]]; then
    SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  fi
  mkdir -p "$BATCH_ROOT/provenance"
  write_tools
  freeze_orchestrator_sources
  freeze_source_libraries
  if [[ ! -f "$BATCH_ROOT/provenance/session.tsv" ]]; then
    {
      printf 'run_id\t%s\n' "$RUN_ID"
      printf 'session_id\t%s\n' "$SESSION_ID"
      printf 'created_at\t%s\n' "$(date --iso-8601=seconds)"
    } >"$BATCH_ROOT/provenance/session.tsv"
  fi
}

monotonic_ns() {
  python3 -c 'import time; print(time.monotonic_ns())'
}

run_partition_attempt() {
  local attempt_root="$1" prefix="$2" clusters="$3"
  local started_ns completed_ns rc
  cat >"$attempt_root/command.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
exec $(printf '%q ' "$PARTITION_BIN" float "$BASE" "$prefix" \
  "$PARTITION_SAMPLE_RATE" "$clusters" "$PARTITION_REPLICATION")
EOF
  chmod 700 "$attempt_root/command.sh"
  started_ns="$(monotonic_ns)"
  printf '%s\n' "$started_ns" >"$attempt_root/command_started_ns.txt"
  set +e
  /usr/bin/time -v -o "$attempt_root/partition.time.log" \
    "$PARTITION_BIN" float "$BASE" "$prefix" "$PARTITION_SAMPLE_RATE" \
    "$clusters" "$PARTITION_REPLICATION" \
    >"$attempt_root/partition.stdout.log" \
    2>"$attempt_root/partition.stderr.log"
  rc=$?
  set -e
  completed_ns="$(monotonic_ns)"
  printf '%s\n' "$completed_ns" >"$attempt_root/command_completed_ns.txt"
  printf '%s\n' "$rc" >"$attempt_root/partition_exit_code.txt"
  return "$rc"
}

write_attempt_record() {
  local attempt_root="$1" attempt="$2" clusters="$3" status="$4" max_shard="$5"
  python3 - "$attempt_root" "$attempt" "$clusters" "$status" "$max_shard" \
    "$SESSION_ID" <<'PY_ATTEMPT_RECORD'
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
started = int((root / "command_started_ns.txt").read_text())
completed = int((root / "command_completed_ns.txt").read_text())
value = {
    "schema": "tagore_vamana_partition_attempt_v1",
    "attempt": int(sys.argv[2]),
    "clusters": int(sys.argv[3]),
    "status": sys.argv[4],
    "max_shard_points": int(sys.argv[5]),
    "session_id": sys.argv[6],
    "command_started_ns": started,
    "command_completed_ns": completed,
    "partition_seconds": (completed - started) / 1e9,
    "partition_exit_code": int((root / "partition_exit_code.txt").read_text()),
}
temporary = root / "attempt.json.tmp"
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "attempt.json")
PY_ATTEMPT_RECORD
}

cleanup_rejected_payload() {
  local attempt_root="$1" payload="$2"
  local resolved_attempt resolved_payload bytes
  resolved_attempt="$(realpath -m -- "$attempt_root")"
  resolved_payload="$(realpath -m -- "$payload")"
  [[ "$resolved_payload" == "$resolved_attempt/payload" ]] \
    || die "rejected cleanup target escapes attempt: $payload"
  [[ -d "$payload" && ! -L "$payload" ]] \
    || die "rejected cleanup target is not a directory: $payload"
  bytes="$(du -sb -- "$payload" | awk '{print $1}')"
  {
    printf 'status\tbytes\tpath\tresolved_path\n'
    printf 'rejected_capacity\t%s\t%s\t%s\n' \
      "$bytes" "$payload" "$resolved_payload"
  } >"$attempt_root/cleanup_manifest.tsv"
  rm -rf -- "$payload"
  [[ ! -e "$payload" && ! -L "$payload" ]] \
    || die "rejected cleanup target remains: $payload"
}

accept_partition_attempt() {
  local dataset_root="$1" attempt_root="$2" attempt="$3" clusters="$4"
  local frozen_root="$dataset_root/frozen"
  local inspector="$BATCH_ROOT/provenance/tools/inspect_partition.py"
  local frozen_ns
  mkdir -p "$frozen_root"
  [[ ! -e "$frozen_root/partition" ]] \
    || die "frozen partition already exists: $frozen_root/partition"
  mv -- "$attempt_root/payload" "$frozen_root/partition"
  "$CUVS_PYTHON" "$inspector" \
    --prefix "$frozen_root/partition/dataset" \
    --clusters "$clusters" --rows "$ROWS" --dim "$DIM" \
    --max-points "$MAX_POINTS" \
    --json "$frozen_root/accepted_inspection.json" \
    --counts "$frozen_root/shard_counts.tsv"
  frozen_ns="$(monotonic_ns)"
  python3 - "$frozen_root/accepted_inspection.json" \
    "$attempt_root/attempt.json" "$frozen_root/partition_manifest.json" \
    "$DATASET" "$BASE" "$SOURCE_RUNBOOK" "$SESSION_ID" "$frozen_ns" <<'PY_FROZEN_MANIFEST'
import json
import os
from pathlib import Path
import sys

inspection_path = Path(sys.argv[1])
attempt_path = Path(sys.argv[2])
output = Path(sys.argv[3])
inspection = json.loads(inspection_path.read_text())
attempt = json.loads(attempt_path.read_text())
base_path = Path(sys.argv[5]).resolve()
base_stat = base_path.stat()
value = dict(inspection)
value.update({
    "schema": "tagore_vamana_frozen_partition_v1",
    "status": "accepted",
    "dataset": sys.argv[4],
    "base": str(base_path),
    "base_identity": {
        "path": str(base_path),
        "bytes": base_stat.st_size,
        "device": base_stat.st_dev,
        "inode": base_stat.st_ino,
        "mtime_ns": base_stat.st_mtime_ns,
    },
    "source_runbook": str(Path(sys.argv[6]).resolve()),
    "session_id": sys.argv[7],
    "attempt": attempt["attempt"],
    "accepted_command_started_ns": attempt["command_started_ns"],
    "frozen_ns": int(sys.argv[8]),
})
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY_FROZEN_MANIFEST
}

summarize_discovery() {
  local dataset_root="$1"
  "$CUVS_PYTHON" "$BATCH_ROOT/provenance/tools/summarize_discovery.py" \
    --dataset-root "$dataset_root" \
    --output "$dataset_root/discovery/discovery_summary.json"
}

discover_dataset() {
  local dataset="$1" dataset_root="$BATCH_ROOT/$1"
  local discovery_root="$dataset_root/discovery"
  local attempts_root="$discovery_root/attempts"
  local frozen_root="$dataset_root/frozen"
  local policy="$BATCH_ROOT/provenance/tools/partition_policy.py"
  local inspector="$BATCH_ROOT/provenance/tools/inspect_partition.py"
  local clusters attempt attempt_root payload prefix inspection max_shard next

  load_dataset "$dataset"
  mkdir -p "$attempts_root"
  [[ ! -e "$frozen_root/partition_manifest.json" ]] \
    || die "frozen partition already exists: $dataset"
  clusters="$("$CUVS_PYTHON" "$policy" initial \
    --rows "$ROWS" --max-points "$MAX_POINTS" \
    --buffer-size "$TAGORE_BUFFER_SIZE")"

  for attempt in $(seq 1 "$MAX_DISCOVERY_ATTEMPTS"); do
    (( clusters <= MAX_DISCOVERY_CLUSTERS )) \
      || die "cluster limit exceeded: $clusters"
    attempt_root="$attempts_root/$(printf 'attempt_%03d_p%04d' \
      "$attempt" "$clusters")"
    fresh_dir "$attempt_root"
    payload="$attempt_root/payload"
    mkdir -p "$payload"
    prefix="$payload/dataset"
    run_partition_attempt "$attempt_root" "$prefix" "$clusters"
    inspection="$attempt_root/inspection.json"
    "$CUVS_PYTHON" "$inspector" \
      --prefix "$prefix" --clusters "$clusters" \
      --rows "$ROWS" --dim "$DIM" --max-points "$MAX_POINTS" \
      --json "$inspection" --counts "$attempt_root/shard_counts.tsv"
    max_shard="$("$CUVS_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["max_shard_points"])' \
      "$inspection")"
    if "$CUVS_PYTHON" -c \
      'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["fits_capacity"])' \
      "$inspection"; then
      write_attempt_record "$attempt_root" "$attempt" "$clusters" \
        accepted "$max_shard"
      accept_partition_attempt "$dataset_root" "$attempt_root" \
        "$attempt" "$clusters"
      summarize_discovery "$dataset_root"
      return
    fi
    write_attempt_record "$attempt_root" "$attempt" "$clusters" \
      rejected_capacity "$max_shard"
    cleanup_rejected_payload "$attempt_root" "$payload"
    next="$("$CUVS_PYTHON" "$policy" next \
      --current "$clusters" --observed-max "$max_shard" \
      --max-points "$MAX_POINTS")"
    (( next > clusters )) \
      || die "partition policy did not increase cluster count"
    clusters="$next"
  done
  die "no capacity-safe partition after $MAX_DISCOVERY_ATTEMPTS attempts"
}

run_conversion() {
  local dataset="$1" formal_root="$2" frozen_prefix="$3" clusters="$4"
  local library="$BATCH_ROOT/provenance/source_libraries/$dataset.sh"
  local converter="$formal_root/tools/convert_diskann_partition_to_tagore.py"
  local started_ns completed_ns
  require_file "$library"
  mkdir -p "$formal_root/tools" "$formal_root/tagore_input"
  (
    # shellcheck disable=SC1090
    source "$library"
    write_converter "$converter"
  )
  require_executable "$converter"
  cat >"$formal_root/conversion.command.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
exec $(printf '%q ' "$CUVS_PYTHON" "$converter" \
  --diskann-prefix "$frozen_prefix" \
  --tagore-out "$formal_root/tagore_input" \
  --num-points "$ROWS" --dim "$DIM" --clusters "$clusters" \
  --max-points "$MAX_POINTS" --consume-source)
EOF
  chmod 700 "$formal_root/conversion.command.sh"
  started_ns="$(monotonic_ns)"
  printf '%s\n' "$started_ns" >"$formal_root/conversion_started_ns.txt"
  /usr/bin/time -v -o "$formal_root/conversion.time.log" \
    "$CUVS_PYTHON" "$converter" \
    --diskann-prefix "$frozen_prefix" \
    --tagore-out "$formal_root/tagore_input" \
    --num-points "$ROWS" --dim "$DIM" --clusters "$clusters" \
    --max-points "$MAX_POINTS" --consume-source \
    >"$formal_root/conversion.stdout.log" \
    2>"$formal_root/conversion.stderr.log"
  completed_ns="$(monotonic_ns)"
  printf '%s\n' "$completed_ns" >"$formal_root/conversion_completed_ns.txt"
  require_file "$formal_root/tagore_input/convert_metadata.json"
}

run_largeindex() {
  local dataset="$1" formal_root="$2" clusters="$3"
  local runner="$BATCH_ROOT/provenance/tools/tagore_timed_runner.py"
  local monitor="$SHEAR_ROOT/script/monitor_gpu_build.sh"
  local final_graph="$formal_root/tagore_run/index/final.tagore.vamana"
  local started_ns completed_ns
  require_executable "$runner"
  require_file "$monitor"
  mkdir -p \
    "$formal_root/tagore_run/local_index" \
    "$formal_root/tagore_run/index" \
    "$formal_root/index"
  started_ns="$(monotonic_ns)"
  printf '%s\n' "$started_ns" >"$formal_root/largeindex_started_ns.txt"
  bash "$monitor" \
    --out-dir "$formal_root/gpu_monitor" --interval 0.1 --gpu 0 -- \
    /usr/bin/time -v -o "$formal_root/largeindex.time.log" \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH="/usr/local/cuda-13.0/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}" \
    "$TAGORE_PYTHON" "$runner" capture \
    --stdout-log "$formal_root/largeindex.stdout.log" \
    --stderr-log "$formal_root/largeindex.stderr.log" \
    --events "$formal_root/largeindex_events.jsonl" -- \
    "$TAGORE_PYTHON" "$runner" worker \
    --build-root "$TAGORE_BUILD_ROOT" --run-root "$formal_root" \
    --rows "$ROWS" --dim "$DIM" --clusters "$clusters" \
    --max-points "$MAX_POINTS"
  completed_ns="$(monotonic_ns)"
  printf '%s\n' "$completed_ns" >"$formal_root/largeindex_completed_ns.txt"
  [[ -s "$final_graph" ]] || die "Tagore did not emit a non-empty graph"
  [[ ! -e "$formal_root/index/_mem.index" ]] \
    || die "final graph link already exists"
  ln -s ../tagore_run/index/final.tagore.vamana \
    "$formal_root/index/_mem.index"
  find "$formal_root/tagore_run/local_index" -type f -printf '%s\t%p\n' \
    >"$formal_root/local_index_inventory.tsv"
}

analyze_largeindex() {
  local formal_root="$1" clusters="$2"
  local started_ns
  started_ns="$(<"$formal_root/largeindex_started_ns.txt")"
  "$CUVS_PYTHON" \
    "$BATCH_ROOT/provenance/tools/analyze_tagore_timeline.py" \
    --events "$formal_root/largeindex_events.jsonl" \
    --clusters "$clusters" --buffer-size "$TAGORE_BUFFER_SIZE" \
    --largeindex-started-ns "$started_ns" \
    --summary "$formal_root/overlap_summary.json" \
    --timeline "$formal_root/shard_timeline.tsv"
}

validate_no_internal_split() {
  local formal_root="$1" accepted_clusters="$2"
  local values=()
  mapfile -t values < <(
    sed -n \
      's/.*Cluster num after being adjusted:\([0-9][0-9]*\).*/\1/p' \
      "$formal_root/largeindex.stdout.log"
  )
  (( ${#values[@]} == 1 )) \
    || die "expected exactly one adjusted-cluster record"
  [[ "${values[0]}" == "$accepted_clusters" ]] \
    || die "Tagore internally changed clusters: ${values[0]} != $accepted_clusters"
}

validate_final_graph() {
  local dataset="$1" formal_root="$2"
  local graph="$formal_root/index/_mem.index"
  local library="$BATCH_ROOT/provenance/source_libraries/$dataset.sh"
  [[ -s "$graph" ]] || die "final graph is missing or empty: $graph"
  require_file "$library"
  (
    # shellcheck disable=SC1090
    source "$library"
    export EXPERIMENT_ROLE=tagore_vamana_adaptive
    validate_graph "$formal_root" "$formal_root/index" "$BASE" "$ROWS"
  )
  require_file "$formal_root/graph_validation.json"
  require_file "$formal_root/graph_quality.tsv"
  "$CUVS_PYTHON" \
    "$BATCH_ROOT/provenance/tools/write_graph_passport.py" \
    --dataset "$dataset" --graph "$graph" \
    --validation "$formal_root/graph_validation.json" \
    --quality "$formal_root/graph_quality.tsv" \
    --output "$formal_root/graph_passport.json"
}

select_sidecar() {
  local dataset="$1" formal_root="$2"
  local passport="$formal_root/sidecar_passport.json"
  load_dataset "$dataset"
  require_file "$SIDECAR_MANIFEST"
  SIDECAR="$("$CUVS_PYTHON" \
    "$BATCH_ROOT/provenance/tools/select_sidecar.py" \
    --manifest "$SIDECAR_MANIFEST" --dataset "$dataset" \
    --base "$BASE" --rows "$ROWS" --dim "$DIM" --output "$passport")"
  require_file "$SIDECAR"
}

run_graph_search() {
  local dataset="$1" formal_root="$2"
  local search_root="$formal_root/search"
  local raw="$search_root/search_sweep.tsv"
  local l rep run_dir metrics result
  load_dataset "$dataset"
  require_executable "$SEARCH_BIN"
  require_file "$formal_root/index/_mem.index"
  require_file "$QUERY"
  require_file "$GT"
  select_sidecar "$dataset" "$formal_root"
  mkdir -p "$search_root"
  printf '%s\n' \
    $'dataset\tparameter_value\trepeat\trecall_at_10\tqps\tresult_sha256\ttotal_one_bit_evaluations\ttotal_lower_bound_rejects\ttotal_navigation_exact_evaluations\ttotal_adaptive_exact_refinements\ttotal_exact_evaluations\tlogical_exact_vector_bytes' \
    >"$raw"
  for l in 25 50 100 200 400; do
    run_dir="$search_root/L$l/warmup"
    fresh_dir "$run_dir"
    "$SEARCH_BIN" \
      --index-path-prefix "$formal_root/index" \
      --data-path "$BASE" \
      --query-file "$QUERY" \
      --gt-file "$GT" \
      --result-path "$run_dir/result.fbin" \
      --recall-at 10 \
      --search-list-size "$l" \
      --num-threads 8 \
      --storage mmap \
      --search-backend rabitq1-adaptive-exact \
      --rabitq1-sidecar "$SIDECAR" \
      --metrics-path "$run_dir/metrics.tsv" \
      >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
    for rep in 1 2; do
      run_dir="$search_root/L$l/rep_$rep"
      fresh_dir "$run_dir"
      metrics="$run_dir/metrics.tsv"
      result="$run_dir/result.fbin"
      /usr/bin/time -v -o "$run_dir/time.log" \
        "$SEARCH_BIN" \
        --index-path-prefix "$formal_root/index" \
        --data-path "$BASE" \
        --query-file "$QUERY" \
        --gt-file "$GT" \
        --result-path "$result" \
        --recall-at 10 \
        --search-list-size "$l" \
        --num-threads 8 \
        --storage mmap \
        --search-backend rabitq1-adaptive-exact \
        --rabitq1-sidecar "$SIDECAR" \
        --metrics-path "$metrics" \
        >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
      "$CUVS_PYTHON" \
        "$BATCH_ROOT/provenance/tools/append_search_row.py" \
        --raw "$raw" --dataset "$dataset" --l "$l" --repeat "$rep" \
        --metrics "$metrics" --result "$result"
    done
  done
  "$CUVS_PYTHON" "$BATCH_ROOT/provenance/tools/summarize_search.py" \
    --input "$raw" --output "$search_root/search_summary.tsv"
}

write_dataset_status() {
  local dataset_root="$1" state="$2"
  local dataset
  dataset="$(basename "$dataset_root")"
  "$CUVS_PYTHON" "$BATCH_ROOT/provenance/tools/write_dataset_status.py" \
    --path "$dataset_root/status.json" \
    --dataset "$dataset" \
    --state "$state" --session-id "$SESSION_ID"
  "$CUVS_PYTHON" - "$BATCH_ROOT/formal_schedule.tsv" \
    "$dataset" "$state" "$SESSION_ID" <<'PY_SCHEDULE'
import csv
import datetime
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
fields = ("utc", "dataset", "state", "session_id")
rows = []
if path.exists():
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError("formal schedule schema mismatch")
        rows = list(reader)
rows.append(
    {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": sys.argv[2],
        "state": sys.argv[3],
        "session_id": sys.argv[4],
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
with temporary.open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, path)
PY_SCHEDULE
}

write_dataset_summary() {
  local dataset_root="$1" dataset
  dataset="$(basename "$dataset_root")"
  "$CUVS_PYTHON" "$BATCH_ROOT/provenance/tools/summarize_dataset.py" \
    --dataset-root "$dataset_root" --dataset "$dataset" \
    --output "$dataset_root/summary.json"
}

cleanup_successful_dataset() {
  local dataset_root="$1"
  local formal_root="$dataset_root/formal"
  local resolved_batch resolved_root target resolved bytes total=0
  local keep="${KEEP_LARGE_ARTIFACTS:-0}"
  local manifest="$dataset_root/cleanup_manifest.tsv"
  local summary="$dataset_root/cleanup_summary.txt"
  local -a targets=(
    "$dataset_root/frozen/partition"
    "$formal_root/tagore_input/clusters"
    "$formal_root/tagore_input/centers.bin"
    "$formal_root/tagore_run/local_index"
    "$formal_root/tagore_run/index"
    "$formal_root/index"
  )
  [[ "$keep" == 0 || "$keep" == 1 ]] \
    || die "KEEP_LARGE_ARTIFACTS must be 0 or 1"
  [[ -d "$dataset_root" && ! -L "$dataset_root" ]] \
    || die "dataset cleanup root is not a real directory: $dataset_root"
  resolved_batch="$(realpath -m -- "$BATCH_ROOT")"
  resolved_root="$(realpath -m -- "$dataset_root")"
  [[ "$resolved_root" == "$resolved_batch"/* ]] \
    || die "dataset cleanup root escapes batch: $dataset_root -> $resolved_root"
  for target in "${targets[@]}"; do
    resolved="$(realpath -m -- "$target")"
    [[ "$resolved" == "$resolved_root"/* ]] \
      || die "cleanup target escapes dataset root: $target -> $resolved"
  done
  "$CUVS_PYTHON" - "$dataset_root/status.json" \
    "$formal_root/graph_validation.json" \
    "$formal_root/graph_passport.json" \
    "$formal_root/search/search_summary.tsv" <<'PY_CLEANUP_GATE'
import csv
import json
from pathlib import Path
import sys

status_path, validation_path, passport_path, search_path = map(
    Path, sys.argv[1:]
)
status = json.loads(status_path.read_text())
validation = json.loads(validation_path.read_text())
passport = json.loads(passport_path.read_text())
if status.get("state") != "accepted":
    raise RuntimeError("cleanup requires accepted dataset status")
if validation.get("valid") is not True or passport.get("valid") is not True:
    raise RuntimeError("cleanup requires accepted graph evidence")
with search_path.open(newline="") as stream:
    search = list(csv.DictReader(stream, delimiter="\t"))
if [int(row["parameter_value"]) for row in search] != [25, 50, 100, 200, 400]:
    raise RuntimeError("cleanup requires complete search L values")
if any(int(row["repeat_count"]) != 2 for row in search):
    raise RuntimeError("cleanup requires two measured search repetitions")
PY_CLEANUP_GATE
  for target in \
    "$formal_root/timing_summary.json" \
    "$formal_root/overlap_summary.json" \
    "$formal_root/graph_quality.tsv"; do
    require_file "$target"
  done
  printf 'action\tbytes\tpath\tresolved_path\n' >"$manifest"
  for target in "${targets[@]}"; do
    resolved="$(realpath -m -- "$target")"
    bytes=0
    if [[ -e "$target" || -L "$target" ]]; then
      bytes="$(du -sb -- "$target" | awk '{print $1}')"
    fi
    total=$((total + bytes))
    printf '%s\t%s\t%s\t%s\n' \
      "$([[ "$keep" == 1 ]] && printf retain || printf remove)" \
      "$bytes" "$target" "$resolved" >>"$manifest"
  done
  if [[ "$keep" == 0 ]]; then
    for target in "${targets[@]}"; do
      if [[ -e "$target" || -L "$target" ]]; then
        rm -rf -- "$target"
      fi
    done
    for target in "${targets[@]}"; do
      [[ ! -e "$target" && ! -L "$target" ]] \
        || die "cleanup target remains: $target"
    done
  fi
  {
    printf 'keep_large_artifacts=%s\n' "$keep"
    printf 'recorded_bytes=%s\n' "$total"
    printf 'verified=true\n'
    printf 'verified_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$summary"
}

run_summarize() {
  require_executable "$BATCH_ROOT/provenance/tools/summarize_batch.py"
  "$CUVS_PYTHON" "$BATCH_ROOT/provenance/tools/summarize_batch.py" \
    --batch-root "$BATCH_ROOT" --datasets "${FORMAL_DATASETS[@]}" \
    --tsv "$BATCH_ROOT/summary.tsv" \
    --markdown "$BATCH_ROOT/summary.md"
}

record_cgroup_limits() {
  local output="$1" relative cgroup_root name
  relative="$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)"
  [[ -n "$relative" ]] || die "unable to resolve inherited cgroup v2 path"
  cgroup_root="/sys/fs/cgroup$relative"
  {
    printf 'relative_path\t%s\n' "$relative"
    printf 'resolved_path\t%s\n' "$cgroup_root"
    for name in memory.max memory.high memory.swap.max; do
      printf '%s\t' "$name"
      if [[ -r "$cgroup_root/$name" ]]; then
        tr -d '\n' <"$cgroup_root/$name"
        printf '\n'
      else
        printf 'unavailable\n'
      fi
    done
  } >"$output"
}

run_gpu_preflight() {
  local output_root="$1"
  local gpu_rows="$output_root/gpu.tsv"
  local process_rows="$output_root/gpu_compute_processes.tsv"
  local memory_total row_count
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi is required"
  mkdir -p "$output_root"
  nvidia-smi -i 0 \
    --query-gpu=index,name,memory.total,memory.free,driver_version \
    --format=csv,noheader,nounits >"$gpu_rows"
  row_count="$(awk 'NF {count++} END {print count+0}' "$gpu_rows")"
  [[ "$row_count" == 1 ]] || die "expected exactly one GPU 0 row"
  memory_total="$(awk -F, '{
    value=$3
    gsub(/^[ \t]+|[ \t]+$/, "", value)
    print value
  }' "$gpu_rows")"
  [[ "$memory_total" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || die "invalid GPU memory.total: $memory_total"
  awk -v actual="$memory_total" -v required="$REFERENCE_GPU_MEMORY_MIB" \
    'BEGIN { exit !(actual + 0 >= required + 0) }' \
    || die "GPU 0 memory is below ${REFERENCE_GPU_MEMORY_MIB} MiB"
  nvidia-smi -i 0 --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d;/No running processes found/d' \
    >"$process_rows"
  [[ ! -s "$process_rows" ]] \
    || die "GPU 0 already has a compute process: $process_rows"
}

run_common_preflight() {
  local output_root="$BATCH_ROOT/preflight"
  local dataset dataset_root manifest already_allocated
  mkdir -p "$output_root" "$BATCH_ROOT/provenance"
  write_tools
  require_executable "$PARTITION_BIN"
  require_executable "$SEARCH_BIN"
  require_executable "$GRAPH_QUALITY_BIN"
  require_executable "$TAGORE_PYTHON"
  require_executable "$CUVS_PYTHON"
  require_file "$SIDECAR_MANIFEST"
  require_file "$SHEAR_ROOT/script/monitor_gpu_build.sh"
  compgen -G "$TAGORE_BUILD_ROOT/Tagore*.so" >/dev/null \
    || die "Tagore Python extension is missing under $TAGORE_BUILD_ROOT"
  "$TAGORE_PYTHON" -c \
    'import sys; sys.path.insert(0, sys.argv[1]); import Tagore' \
    "$TAGORE_BUILD_ROOT"
  for dataset in "${ACTIVE_DATASETS[@]}"; do
    load_dataset "$dataset"
    require_file "$SOURCE_RUNBOOK"
    dataset_root="$BATCH_ROOT/$dataset"
    manifest="$dataset_root/frozen/partition_manifest.json"
    already_allocated=0
    if [[ -f "$manifest" ]]; then
      "$CUVS_PYTHON" \
        "$BATCH_ROOT/provenance/tools/validate_frozen_partition.py" \
        --manifest "$manifest" --dataset "$dataset" --base "$BASE" \
        --rows "$ROWS" --dim "$DIM" --max-points "$MAX_POINTS" \
        --output "$output_root/${dataset}_frozen_revalidation.json" \
        >"$output_root/${dataset}_accepted_clusters.txt"
      already_allocated="$("$CUVS_PYTHON" - "$manifest" \
        "$ROWS" "$DIM" <<'PY_ALLOCATED_PARTITION'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text())
total = 0
for shard in value["shards"]:
    total += int(shard["data"]["bytes"]) + int(shard["ids"]["bytes"])
modeled_partition = 2 * int(sys.argv[2]) * (4 * int(sys.argv[3]) + 4)
print(min(total, modeled_partition))
PY_ALLOCATED_PARTITION
)"
    fi
    "$CUVS_PYTHON" "$BATCH_ROOT/provenance/tools/preflight_dataset.py" \
      --dataset "$dataset" --base "$BASE" --query "$QUERY" --gt "$GT" \
      --rows "$ROWS" --dim "$DIM" --max-points "$MAX_POINTS" \
      --staging-path "$BATCH_ROOT" \
      --already-allocated-bytes "$already_allocated" \
      --output "$output_root/$dataset.json"
  done
  record_cgroup_limits "$output_root/cgroup.tsv"
  {
    date --iso-8601=seconds
    uname -a
    lscpu
    free -h
    df -B1 "$BATCH_ROOT"
  } >"$output_root/host.txt" 2>&1
}

run_preflight() {
  run_common_preflight
  run_gpu_preflight "$BATCH_ROOT/preflight"
}

record_runtime_provenance() {
  local root="$BATCH_ROOT/provenance"
  mkdir -p "$root"
  {
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'session_id\t%s\n' "$SESSION_ID"
    printf 'mode\t%s\n' "$MODE"
    printf 'datasets\t%s\n' "$(IFS=,; printf '%s' "${ACTIVE_DATASETS[*]}")"
    printf 'keep_large_artifacts\t%s\n' "${KEEP_LARGE_ARTIFACTS:-0}"
  } >"$root/context.tsv"
  git -C "$SHEAR_ROOT" rev-parse HEAD >"$root/git_head.txt"
  git -C "$SHEAR_ROOT" status --short >"$root/git_status.txt"
  git -C "$SHEAR_ROOT" diff --binary >"$root/working_tree.diff"
  sha256sum \
    "$PARTITION_BIN" "$SEARCH_BIN" "$GRAPH_QUALITY_BIN" \
    "$SIDECAR_MANIFEST" "$SHEAR_ROOT/script/monitor_gpu_build.sh" \
    >"$root/runtime_inputs.sha256"
  find "$TAGORE_BUILD_ROOT" -maxdepth 1 -type f -name 'Tagore*.so' \
    -print0 | sort -z | xargs -0 -r sha256sum \
    >"$root/tagore_extensions.sha256"
}

run_prepare() {
  initialize_batch
  (
    cd "$SHEAR_ROOT"
    cargo build --locked --release -p search_shear_index \
      --bin search_shear_index --bin graph_quality
  )
  require_executable "$PARTITION_BIN"
  require_executable "$SEARCH_BIN"
  require_executable "$GRAPH_QUALITY_BIN"
  require_file "$SIDECAR_MANIFEST"
  record_runtime_provenance
}

ACTIVE_TRANSACTION_ROOT=""

record_transaction_failure() {
  local rc=$?
  trap - ERR
  set +e
  if [[ -n "$ACTIVE_TRANSACTION_ROOT" \
    && -x "$BATCH_ROOT/provenance/tools/write_dataset_status.py" ]]; then
    write_dataset_status "$ACTIVE_TRANSACTION_ROOT" failed
  fi
  set -e
  return "$rc"
}

run_discovery_for_active_datasets() {
  local dataset dataset_root
  initialize_batch
  trap record_transaction_failure ERR
  for dataset in "${ACTIVE_DATASETS[@]}"; do
    dataset_root="$BATCH_ROOT/$dataset"
    ACTIVE_TRANSACTION_ROOT="$dataset_root"
    mkdir -p "$dataset_root"
    write_dataset_status "$dataset_root" scheduled
    if [[ ! -f "$dataset_root/frozen/partition_manifest.json" ]]; then
      write_dataset_status "$dataset_root" discovering
      discover_dataset "$dataset"
    fi
    write_dataset_status "$dataset_root" partition_frozen
    ACTIVE_TRANSACTION_ROOT=""
  done
  trap - ERR
}

run_formal_for_active_datasets() {
  local dataset dataset_root
  initialize_batch
  run_common_preflight
  run_gpu_preflight "$BATCH_ROOT/preflight"
  trap record_transaction_failure ERR
  for dataset in "${ACTIVE_DATASETS[@]}"; do
    dataset_root="$BATCH_ROOT/$dataset"
    ACTIVE_TRANSACTION_ROOT="$dataset_root"
    mkdir -p "$dataset_root"
    write_dataset_status "$dataset_root" scheduled
    if [[ ! -f "$dataset_root/frozen/partition_manifest.json" ]]; then
      write_dataset_status "$dataset_root" discovering
      discover_dataset "$dataset"
    fi
    write_dataset_status "$dataset_root" partition_frozen
    run_formal_dataset "$dataset"
    ACTIVE_TRANSACTION_ROOT=""
  done
  trap - ERR
  run_summarize
}

run_formal_dataset() {
  local dataset="$1" dataset_root="$BATCH_ROOT/$1"
  local formal_root="$dataset_root/formal"
  local frozen_root="$dataset_root/frozen"
  local manifest="$frozen_root/partition_manifest.json"
  local accepted_clusters

  load_dataset "$dataset"
  require_file "$manifest"
  fresh_dir "$formal_root"
  printf '%s\n' "$SESSION_ID" >"$formal_root/session_id.txt"
  accepted_clusters="$("$CUVS_PYTHON" \
    "$BATCH_ROOT/provenance/tools/validate_frozen_partition.py" \
    --manifest "$manifest" --dataset "$dataset" --base "$BASE" \
    --rows "$ROWS" --dim "$DIM" --max-points "$MAX_POINTS" \
    --output "$formal_root/frozen_revalidation.json")"
  write_dataset_status "$dataset_root" converting
  run_conversion "$dataset" "$formal_root" \
    "$frozen_root/partition/dataset" "$accepted_clusters"
  write_dataset_status "$dataset_root" building
  run_largeindex "$dataset" "$formal_root" "$accepted_clusters"
  analyze_largeindex "$formal_root" "$accepted_clusters"
  validate_no_internal_split "$formal_root" "$accepted_clusters"
  write_dataset_status "$dataset_root" validating
  validate_final_graph "$dataset" "$formal_root"
  write_dataset_status "$dataset_root" searching
  run_graph_search "$dataset" "$formal_root"
  write_dataset_status "$dataset_root" accepted
  write_dataset_summary "$dataset_root"
  cleanup_successful_dataset "$dataset_root"
}

usage() {
  cat <<'USAGE'
Usage: tagore_vamana_four_dataset.sh MODE

Modes:
  help        Show this contract without touching experiment state.
  prepare     Build or validate dependencies and freeze provenance.
  preflight   Perform read-only data, space, GPU, and runtime checks.
  smoke       Run fixture-only orchestration and timeline tests.
  discover    Discover and freeze partitions for all active datasets.
  formal      Discover, build, validate, and search all four datasets.
  wiki35m     Execute the complete Wiki35M transaction.
  laion100m   Execute the complete LAION100M transaction.
  msmarco70m  Execute the complete MSMARCO70M transaction.
  deep100m    Execute the complete DEEP100M transaction.
  summarize   Rebuild summaries from an existing BATCH_ROOT.
USAGE
}

case "$MODE" in
  help|-h|--help) usage ;;
  prepare) run_prepare ;;
  preflight) run_preflight ;;
  smoke) python3 "$TEST_SCRIPT" --fixture-only ;;
  discover) run_discovery_for_active_datasets ;;
  formal) run_formal_for_active_datasets ;;
  wiki35m|laion100m|msmarco70m|deep100m)
    ACTIVE_DATASETS=("$MODE")
    run_formal_for_active_datasets
    ;;
  summarize) run_summarize ;;
  *) usage >&2; exit 2 ;;
esac
```
<!-- END RUNBOOK_SCRIPT -->

## 实验合同

四个数据集使用同一套 Tagore 参数，但显存容量上限来自各自已经验证过的
`max_points`：

| 数据集 | `N` | 维度 | `max_points` |
|---|---:|---:|---:|
| Wiki35M | 35,000,000 | 1024 | 1,200,000 |
| LAION100M | 100,000,000 | 512 | 3,000,000 |
| MSMARCO70M | 70,000,000 | 768 | 2,500,000 |
| DEEP100M | 100,000,000 | 96 | 5,000,000 |

固定参数为：DiskANN sample rate `0.01`、replication `2`、Tagore
`K=96`、`max_degree=64`、`final_degree=64`、`TOPM=64`、
threshold `1.2`、buffer size `16`、GPU 0、OMP 8、BLAS 1。
runbook 不动态增大 `max_points`，也不修改 Tagore 的构图或 merge
实现。

正式执行要求 GPU 0 总显存不少于 24,564 MiB，且启动时没有其他
compute process。runbook 只记录当前进程继承的 cgroup v2
`memory.max`、`memory.high` 和 `memory.swap.max`；它不会创建或修改
cgroup，因此带内存限制的实验应由调用者在外层建立限制。

preflight 的从零开始空间上界为：

```text
1.10 * (
  2N * (4*DIM + 4)
  + 8N
  + 2N * (4 + 64*4)
  + 24 + N * (4 + 64*4)
) + 10 GiB
```

它分别覆盖 replicated partition、centers、local-index 上界和 final
index 上界。如果同一 batch 已有通过逐文件复验的 frozen partition，
总上界仍原样记录，但正式续跑只要求尚未分配的增量空间，避免把已经占盘
且 conversion 将逐 shard 消费的 partition 重复计算。

## 自适应分区与冻结

DiskANN 将每个点分配给两个中心，因此初始分区数为：

```text
P0 = round_up_8(max(16, ceil(2N / max_points)))
```

每次分区后，runbook 从实际 shard header 读取最大 shard 大小 `S`。
若 `S > max_points`，下一次使用：

```text
Pnext = round_up_8(max(P + 8, ceil(P * S / max_points * 1.10)))
```

增加分区数的目的是降低每个 shard 的点数；数据集总行数相近并不意味
着分区数应相同，因为维度、数据分布和各自验证过的 workspace capacity
不同。`1.10` 只是减少昂贵重试的余量，接受条件始终是实际
`max(shard_counts) <= max_points`。默认最多尝试 8 次且不超过 4096
个分区。

`partition_data` 的抽样和 k-means 初始化具有随机性。因此 runbook
冻结第一次真正通过容量检查的 centroids、data/id shards 和文件身份，
而不是只记录一个分区数后重新运行。正式 conversion 在消费前再次检查
base 路径与 stat、每个文件的 inode/mtime/大小/header、总分配数
`2N` 和最大 shard；随后直接消费冻结文件，正式路径不会再次调用
`partition_data`。

正常但超限的尝试记录为 `rejected_capacity`。其命令、计时、日志、
counts 和清理清单会保留，只删除该尝试目录内的大 payload。命令失败、
空 shard、损坏 header、错误总数或越界清理都属于真实失败，立即停止并
保留现场。

## 时间边界

构建阶段分别记录：

```text
accepted_pipeline_seconds
  = accepted_partition_seconds + conversion_seconds + largeindex_seconds

discovery_inclusive_seconds
  = discovery_wall_seconds + conversion_seconds + largeindex_seconds
```

其中：

- `discovery_wall_seconds` 是第一次 partition 启动到 accepted
  manifest 发布的连续时间，包括全部失败尝试、检查、清理和冻结；
- `all_attempt_partition_seconds` 是所有 partition 命令时间之和；
- `rejected_partition_seconds` 和 `accepted_partition_seconds`
  分别给出被拒绝尝试与最终成功尝试；
- `discovery_overhead_seconds` 是 discovery wall 减去所有 partition
  命令时间；
- `gpu_cpu_pipeline_seconds` 等于完整 `largeIndex` wall，明确不包含
  partition 和格式转换；
- 图验证与搜索不计入构建时间。

如果 discovery 和正式构建处于同一 invocation session，还报告：

```text
accepted_configuration_wall_seconds
  = accepted partition 进程启动 -> largeIndex 完成

full_discovery_execution_wall_seconds
  = 第一次 partition 进程启动 -> largeIndex 完成
```

若先执行 `discover`、之后用另一次脚本调用继续 `formal`，阶段和仍然
有效，但这两个连续墙钟字段写为 `not_applicable`，不会把两次调用之间
的空闲时间伪装成实验时间。

## CPU–GPU 重叠解释

父进程同时 drain Tagore 的 stdout 和 stderr，保留原始日志，并为每个
完整日志行记录单调时钟与 UTC。对每个 shard 输出：

- H2D 后的 local-build-start 到回传门；
- 回传门到 `transfered` 的组合区间；
- transfer 完成到 mapped；
- CPU 等待 merge 与编号 online merge；
- GPU 是否在前一编号 merge 结束前到达下一回传门。

前 16 个局部图只是在填充 buffer，不具备稳态背压判定条件。因此
`gpu_ready_before_previous_merge_end_count` 的分母固定为 `P-16`，只检查
ordinal `i >= 16`。该计数是日志可见的 producer/consumer 顺序，不等于
GPU 一定发生了硬阻塞。

尤其是 `Waiting for transfering` 到 `transfered` 同时可能包含 buffer
slot 等待、约 100 ms 的轮询和 D2H copy。runbook 只将其称为
`transfer_gate_combined_seconds`，不声称它是纯 D2H、纯 kernel 或纯
backpressure 时间。最后一个编号 merge 到无编号 `Begin to merge`
是 drain/flush；无编号 begin 到 `Index size` 是最终全局 post。

## 使用方法

先提取并检查帮助；`help` 和 `smoke` 不创建大数据实验目录：

```bash
python3 script/test_tagore_vamana_four_dataset_runbook.py --extract \
  >/tmp/tagore_vamana_four_dataset.sh
chmod 700 /tmp/tagore_vamana_four_dataset.sh
/tmp/tagore_vamana_four_dataset.sh help
/tmp/tagore_vamana_four_dataset.sh smoke
```

选择一个固定 batch root，依次执行依赖准备、只读 preflight 和正式
实验：

```bash
export BATCH_ROOT=/data_staging/wbh/tagore_vamana_four_dataset/runs/<run-id>
/tmp/tagore_vamana_four_dataset.sh prepare
/tmp/tagore_vamana_four_dataset.sh preflight
/tmp/tagore_vamana_four_dataset.sh formal
```

也可以先只发现并冻结四个数据集的分区，之后在同一 `BATCH_ROOT`
继续；第二步不会重新分区：

```bash
/tmp/tagore_vamana_four_dataset.sh discover
/tmp/tagore_vamana_four_dataset.sh formal
```

单数据集完整 transaction 使用 `wiki35m`、`laion100m`、
`msmarco70m` 或 `deep100m`。从已有记录重建批次报告：

```bash
/tmp/tagore_vamana_four_dataset.sh wiki35m
/tmp/tagore_vamana_four_dataset.sh summarize
```

默认在成功验证和完整搜索后删除大型中间/最终图。若确实需要保留，
必须显式设置：

```bash
KEEP_LARGE_ARTIFACTS=1 /tmp/tagore_vamana_four_dataset.sh formal
```

## 输出、搜索与清理

每个 dataset transaction 的主要结构为：

```text
<batch>/<dataset>/
  status.json
  discovery/
    discovery_summary.json
    attempts/attempt_NNN_pPPPP/
  frozen/
    partition_manifest.json
    accepted_inspection.json
    partition/
  formal/
    frozen_revalidation.json
    conversion.{time,stdout,stderr}.log
    largeindex.{time,stdout,stderr}.log
    largeindex_events.jsonl
    shard_timeline.tsv
    overlap_summary.json
    gpu_monitor/
    graph_validation.json
    graph_quality.tsv
    graph_passport.json
    search/
    timing_summary.json
  summary.json
  cleanup_manifest.tsv
  cleanup_summary.txt
```

搜索固定使用 accepted `graph_rabitq1_v1` sidecar、
`rabitq1-adaptive-exact`、Recall@10、8 线程、mmap，以及
`L=25,50,100,200,400`。每个 L 先 warmup 一次，再测量两次；两次
recall 与结果 SHA-256 必须一致，且 recall 不得随 L 增大而下降。

成功清理严格限于当前 dataset root 下的冻结 partition、Tagore
clusters/centers、local index、Tagore final index 和通用 index 目录。
清理前先记录每个目标的解析路径和大小，并要求 accepted status、
timing、overlap、图验证/passport 和五个 L 的完整搜索证据。路径逃逸或
任一证据缺失都会停止；失败 transaction 从不自动清理或重试。

## 非声明与实施范围

这个 runbook 测量日志可观察的流水线行为，不把 Tagore 已有的 CPU–GPU
并发本身当作新机制，也不从日志推导不可观测的 CUDA kernel 或纯 I/O
时间。分区数适配只保证当前真实 materialization 满足容量合同，不保证
它在图质量或构建时间上全局最优。

本次实现验证只运行静态检查、合成事件流、伪 partition、搜索/清理小
夹具、Bash/Python 语法检查和条件允许时的只读 preflight；不会自动启动
Wiki35M、LAION100M、MSMARCO70M 或 DEEP100M 的正式构建。
