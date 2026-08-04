#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable


Interval = tuple[int, int]
PARTITION_RE = re.compile(r"\bpartition=(\d+)\b")
CPU_MARKERS = (
    "SHEAR_MOTIVATION/CPU_GRAPH_INIT",
    "SHEAR_MOTIVATION/CPU_GRAPH_UPDATE_SAMPLE",
    "SHEAR_MOTIVATION/CPU_SAMPLE_NEW",
    "SHEAR_MOTIVATION/CPU_FINAL_UPDATE_SORT",
)
COORDINATION_PREFIX = "SHEAR_MOTIVATION/COORD_"


def normalize_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(left: Iterable[Interval], right: Iterable[Interval]) -> list[Interval]:
    left_normalized = normalize_intervals(left)
    right_normalized = normalize_intervals(right)
    intersections: list[Interval] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_normalized) and right_index < len(right_normalized):
        left_start, left_end = left_normalized[left_index]
        right_start, right_end = right_normalized[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            intersections.append((start, end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return intersections


def subtract_intervals(left: Iterable[Interval], right: Iterable[Interval]) -> list[Interval]:
    remaining: list[Interval] = []
    right_normalized = normalize_intervals(right)
    right_index = 0
    for left_start, left_end in normalize_intervals(left):
        cursor = left_start
        while right_index < len(right_normalized) and right_normalized[right_index][1] <= cursor:
            right_index += 1
        scan_index = right_index
        while scan_index < len(right_normalized):
            right_start, right_end = right_normalized[scan_index]
            if right_start >= left_end:
                break
            if right_start > cursor:
                remaining.append((cursor, min(right_start, left_end)))
            cursor = max(cursor, right_end)
            if cursor >= left_end:
                break
            scan_index += 1
        if cursor < left_end:
            remaining.append((cursor, left_end))
    return normalize_intervals(remaining)


def duration_ns(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in normalize_intervals(intervals))


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _read_nvtx_events(connection: sqlite3.Connection) -> list[tuple[int, int, str]]:
    if not _table_exists(connection, "NVTX_EVENTS"):
        raise ValueError("Nsight SQLite export does not contain NVTX_EVENTS")
    if _table_exists(connection, "StringIds"):
        query = """
            SELECT event.start, event.end, COALESCE(event.text, string.value)
            FROM NVTX_EVENTS AS event
            LEFT JOIN StringIds AS string ON event.textId = string.id
            WHERE event.end IS NOT NULL
              AND COALESCE(event.text, string.value) LIKE 'SHEAR_MOTIVATION/%'
        """
    else:
        query = """
            SELECT start, end, text
            FROM NVTX_EVENTS
            WHERE end IS NOT NULL AND text LIKE 'SHEAR_MOTIVATION/%'
        """
    return [
        (int(start), int(end), str(text))
        for start, end, text in connection.execute(query)
        if text is not None and int(end) > int(start)
    ]


def _read_activity_intervals(connection: sqlite3.Connection, table: str) -> list[Interval]:
    if not _table_exists(connection, table):
        return []
    return normalize_intervals(
        (int(start), int(end))
        for start, end in connection.execute(f'SELECT start, end FROM "{table}"')
    )


def _milliseconds(intervals: Iterable[Interval]) -> float:
    return duration_ns(intervals) / 1_000_000.0


def _summarize_window(
    local_window: list[Interval],
    gpu_kernels: list[Interval],
    gpu_memcpy: list[Interval],
    cpu_graph: list[Interval],
    coordination: list[Interval],
    gnnd: list[Interval],
    post_gnnd_optimize: list[Interval],
) -> dict:
    local = normalize_intervals(local_window)
    gpu_compute = intersect_intervals(gpu_kernels, local)
    gpu_transfer = intersect_intervals(gpu_memcpy, local)
    cpu = intersect_intervals(cpu_graph, local)
    coord = intersect_intervals(coordination, local)

    gpu_transfer_exclusive = subtract_intervals(gpu_transfer, gpu_compute)
    device = normalize_intervals([*gpu_compute, *gpu_transfer])
    cpu_exclusive = subtract_intervals(cpu, device)
    coordination_exclusive = subtract_intervals(coord, [*device, *cpu])
    accounted = normalize_intervals([*device, *cpu, *coord])
    other = subtract_intervals(local, accounted)

    exclusive = {
        "gpu_compute": _milliseconds(gpu_compute),
        "gpu_transfer": _milliseconds(gpu_transfer_exclusive),
        "cpu_graph": _milliseconds(cpu_exclusive),
        "coordination": _milliseconds(coordination_exclusive),
        "other": _milliseconds(other),
    }
    local_wall_ms = _milliseconds(local)
    return {
        "local_build_wall_ms": local_wall_ms,
        "gpu_compute_active_ms": _milliseconds(gpu_compute),
        "gpu_transfer_active_ms": _milliseconds(gpu_transfer),
        "cpu_graph_active_ms": _milliseconds(cpu),
        "coordination_active_ms": _milliseconds(coord),
        "gpu_cpu_graph_overlap_ms": _milliseconds(intersect_intervals(gpu_compute, cpu)),
        "gpu_coordination_overlap_ms": _milliseconds(intersect_intervals(gpu_compute, coord)),
        "cpu_graph_coordination_overlap_ms": _milliseconds(intersect_intervals(cpu, coord)),
        "gnnd_wall_ms": _milliseconds(intersect_intervals(gnnd, local)),
        "post_gnnd_optimize_wall_ms": _milliseconds(
            intersect_intervals(post_gnnd_optimize, local)
        ),
        "critical_path_exclusive_ms": exclusive,
        "critical_path_closure_error_ms": abs(sum(exclusive.values()) - local_wall_ms),
    }


def analyze_sqlite(path: Path) -> dict:
    path = Path(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        events = _read_nvtx_events(connection)
        gpu_kernels = _read_activity_intervals(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        gpu_memcpy = _read_activity_intervals(connection, "CUPTI_ACTIVITY_KIND_MEMCPY")
    finally:
        connection.close()

    local_events = [
        (start, end, text)
        for start, end, text in events
        if text.startswith("SHEAR_MOTIVATION/ACE_LOCAL_BUILD")
    ]
    if not local_events:
        raise ValueError("trace contains no SHEAR_MOTIVATION/ACE_LOCAL_BUILD ranges")

    cpu_graph = normalize_intervals(
        (start, end)
        for start, end, text in events
        if text.startswith(CPU_MARKERS)
    )
    coordination = normalize_intervals(
        (start, end)
        for start, end, text in events
        if text.startswith(COORDINATION_PREFIX)
    )
    gnnd = normalize_intervals(
        (start, end)
        for start, end, text in events
        if text.startswith("SHEAR_MOTIVATION/GNND_LOCAL_BUILD")
    )
    post_gnnd_optimize = normalize_intervals(
        (start, end)
        for start, end, text in events
        if text.startswith("SHEAR_MOTIVATION/CAGRA_POST_GNND_OPTIMIZE")
    )
    local_windows = normalize_intervals((start, end) for start, end, _ in local_events)

    aggregate = _summarize_window(
        local_windows,
        gpu_kernels,
        gpu_memcpy,
        cpu_graph,
        coordination,
        gnnd,
        post_gnnd_optimize,
    )
    aggregate["partition_count"] = len(local_events)

    partitions = []
    for sequence, (start, end, text) in enumerate(sorted(local_events)):
        match = PARTITION_RE.search(text)
        partition = _summarize_window(
            [(start, end)],
            gpu_kernels,
            gpu_memcpy,
            cpu_graph,
            coordination,
            gnnd,
            post_gnnd_optimize,
        )
        partition["sequence"] = sequence
        partition["partition_id"] = None if match is None else int(match.group(1))
        partitions.append(partition)

    marker_counts: dict[str, int] = {}
    marker_intervals: dict[str, list[Interval]] = {}
    for _, _, text in events:
        marker = text.split(" ", 1)[0]
        marker_counts[marker] = marker_counts.get(marker, 0) + 1
    for start, end, text in events:
        marker = text.split(" ", 1)[0]
        marker_intervals.setdefault(marker, []).append((start, end))

    return {
        "schema": "laion35m-ace-nsys-breakdown-v1",
        "source_sqlite": str(path.resolve()),
        "aggregate": aggregate,
        "partitions": partitions,
        "marker_counts": dict(sorted(marker_counts.items())),
        "marker_active_ms": {
            marker: _milliseconds(intervals)
            for marker, intervals in sorted(marker_intervals.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_sqlite(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
