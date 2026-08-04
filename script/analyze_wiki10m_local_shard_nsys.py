#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable


Interval = tuple[int, int]
METHODS = ("cagra", "tagore_cagra", "tagore_vamana")
DENSE_PATTERNS = (
    r"local_join.*wmma",
    r"nn_descent_opt_cal",
    r"compute_l2_norms_kernel",
    r"cal_power",
)
IRREGULAR_PATTERNS = (
    r"initialize_graph",
    r"nn_descent_opt_(sample|reverse_sample|merge)",
    r"add_rev_edges_kernel",
    r"kern_(fused_prune|make_rev_graph|merge_graph)",
    r"do_reverse_graph",
    r"merge_reverse_plus",
    r"reset_",
    r"cal_ep",
    r"select_(path|1hop)",
    r"filter_reverse",
)
MIXED_PATTERNS = (
    r"sample_kernel6",
    r"kern_sort",
)
KNOWN_OTHER_PATTERNS = (
    r"convert_copy_kernel",
    r"raft::linalg::detail::map_kernel",
)
CPU_GRAPH_PREFIXES = (
    "SHEAR_MOTIVATION/CPU_GRAPH_",
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
    intersections = []
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
    right_normalized = normalize_intervals(right)
    remaining = []
    for left_start, left_end in normalize_intervals(left):
        cursor = left_start
        for right_start, right_end in right_normalized:
            if right_end <= cursor:
                continue
            if right_start >= left_end:
                break
            if right_start > cursor:
                remaining.append((cursor, min(right_start, left_end)))
            cursor = max(cursor, right_end)
            if cursor >= left_end:
                break
        if cursor < left_end:
            remaining.append((cursor, left_end))
    return normalize_intervals(remaining)


def duration_ns(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in normalize_intervals(intervals))


def _milliseconds(intervals: Iterable[Interval]) -> float:
    return duration_ns(intervals) / 1_000_000.0


def _summed_intersection_ms(
    intervals: Iterable[Interval],
    window: Iterable[Interval],
) -> float:
    return (
        sum(
            duration_ns(intersect_intervals([interval], window))
            for interval in intervals
        )
        / 1_000_000.0
    )


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _read_nvtx_events(connection: sqlite3.Connection) -> list[tuple[int, int, str]]:
    table = "NVTX_EVENTS"
    if not _table_exists(connection, table):
        raise ValueError("Nsight SQLite export does not contain NVTX_EVENTS")
    columns = _columns(connection, table)
    if not {"start", "end"}.issubset(columns):
        raise ValueError("NVTX_EVENTS has no start/end columns")

    has_strings = _table_exists(connection, "StringIds")
    if "text" in columns and "textId" in columns and has_strings:
        query = """
            SELECT event.start, event.end, COALESCE(event.text, string.value)
            FROM NVTX_EVENTS AS event
            LEFT JOIN StringIds AS string ON event.textId = string.id
            WHERE event.end IS NOT NULL
        """
    elif "text" in columns:
        query = """
            SELECT start, end, text
            FROM NVTX_EVENTS
            WHERE end IS NOT NULL
        """
    elif "textId" in columns and has_strings:
        query = """
            SELECT event.start, event.end, string.value
            FROM NVTX_EVENTS AS event
            JOIN StringIds AS string ON event.textId = string.id
            WHERE event.end IS NOT NULL
        """
    else:
        raise ValueError("NVTX_EVENTS has no supported text column")
    return [
        (int(start), int(end), str(text))
        for start, end, text in connection.execute(query)
        if text is not None and int(end) > int(start)
    ]


def _read_kernel_events(connection: sqlite3.Connection) -> list[dict]:
    candidates = (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    )
    table = next((name for name in candidates if _table_exists(connection, name)), None)
    if table is None:
        raise ValueError("Nsight SQLite export contains no supported CUDA kernel table")
    columns = _columns(connection, table)
    if not {"start", "end"}.issubset(columns):
        raise ValueError(f"{table} has no start/end columns")

    inline = next((name for name in ("text", "name", "kernelName") if name in columns), None)
    identifier = next(
        (name for name in ("demangledName", "shortName") if name in columns),
        None,
    )
    if inline is not None:
        query = f'SELECT start, end, "{inline}" FROM "{table}"'
    elif identifier is not None and _table_exists(connection, "StringIds"):
        query = f"""
            SELECT activity.start, activity.end, string.value
            FROM "{table}" AS activity
            JOIN StringIds AS string ON activity."{identifier}" = string.id
        """
    else:
        diagnostic = ", ".join(sorted(columns))
        raise ValueError(f"{table} has no supported kernel name column; columns={diagnostic}")
    return [
        {"start": int(start), "end": int(end), "kernel_name": str(name)}
        for start, end, name in connection.execute(query)
        if name is not None and int(end) > int(start)
    ]


def _read_activity_intervals(
    connection: sqlite3.Connection, tables: Iterable[str]
) -> list[Interval]:
    intervals = []
    for table in tables:
        if not _table_exists(connection, table):
            continue
        columns = _columns(connection, table)
        if {"start", "end"}.issubset(columns):
            intervals.extend(
                (int(start), int(end))
                for start, end in connection.execute(
                    f'SELECT start, end FROM "{table}"'
                )
            )
    return normalize_intervals(intervals)


def _contains_kernel(name: str, kernel: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(kernel)}(?![A-Za-z0-9_])", name) is not None


def semantic_kernel_class(name: str, method: str) -> str:
    lowered = name.lower()
    if method == "tagore_vamana" and (
        _contains_kernel(lowered, "select_path")
        or _contains_kernel(lowered, "filter_reverse")
    ):
        return "mixed_cfs"
    if method == "tagore_cagra" and (
        "select_1hop_cagra" in lowered or "filter_reverse_1hop" in lowered
    ):
        return "irregular_cfs"
    return classify_kernel(name, method)


def is_known_other_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(re.search(pattern, lowered) for pattern in KNOWN_OTHER_PATTERNS)


def classify_kernel(name: str, method: str | None = None) -> str:
    lowered = name.lower()
    if method == "tagore_vamana" and (
        _contains_kernel(lowered, "select_path")
        or _contains_kernel(lowered, "filter_reverse")
    ):
        return "mixed"
    if method == "tagore_cagra" and (
        "select_1hop_cagra" in lowered or "filter_reverse_1hop" in lowered
    ):
        return "irregular"
    for pattern in DENSE_PATTERNS:
        if re.search(pattern, lowered):
            return "dense"
    for pattern in IRREGULAR_PATTERNS:
        if re.search(pattern, lowered):
            return "irregular"
    for pattern in MIXED_PATTERNS:
        if re.search(pattern, lowered):
            return "mixed"
    return "other_gpu"


def _marker_intervals(
    events: Iterable[tuple[int, int, str]], prefix: str
) -> list[Interval]:
    return normalize_intervals(
        (start, end) for start, end, text in events if text.startswith(prefix)
    )


def _exclusive_critical_path(
    local: list[Interval], categories: list[tuple[str, list[Interval]]]
) -> dict[str, float]:
    remaining = normalize_intervals(local)
    result = {}
    for name, intervals in categories:
        assigned = intersect_intervals(remaining, intervals)
        result[name] = _milliseconds(assigned)
        remaining = subtract_intervals(remaining, assigned)
    result["residual"] = _milliseconds(remaining)
    return result


def analyze_sqlite(path: Path, method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    path = Path(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        nvtx_events = _read_nvtx_events(connection)
        kernels = _read_kernel_events(connection)
        memcpy = _read_activity_intervals(
            connection,
            (
                "CUPTI_ACTIVITY_KIND_MEMCPY",
                "CUPTI_ACTIVITY_KIND_MEMCPY2",
            ),
        )
    finally:
        connection.close()

    local_prefix = f"WIKI10M_LOCAL_SHARD/{method}/LOCAL_BUILD"
    local = _marker_intervals(nvtx_events, local_prefix)
    if not local:
        raise ValueError(f"trace contains no {local_prefix} range")
    gnn = _marker_intervals(nvtx_events, f"WIKI10M_LOCAL_SHARD/{method}/GNN")
    if not gnn:
        raise ValueError(f"trace contains no WIKI10M_LOCAL_SHARD/{method}/GNN range")
    prune = _marker_intervals(nvtx_events, f"WIKI10M_LOCAL_SHARD/{method}/PRUNE")
    if method != "cagra" and not prune:
        raise ValueError(f"trace contains no WIKI10M_LOCAL_SHARD/{method}/PRUNE range")

    class_intervals: dict[str, list[Interval]] = {
        "dense": [],
        "irregular": [],
        "mixed": [],
        "other_gpu": [],
    }
    unclassified_intervals = []
    kernel_aggregates: dict[tuple[str, str], dict] = {}
    setup_intervals = []
    cfs_primary_intervals = []
    cfs_reverse_intervals = []
    for event in kernels:
        name = event["kernel_name"]
        kernel_class = classify_kernel(name, method)
        semantic_class = semantic_kernel_class(name, method)
        clipped = intersect_intervals(
            [(event["start"], event["end"])],
            local,
        )
        if not clipped:
            continue
        class_intervals[kernel_class].extend(clipped)
        classification_status = (
            "unclassified"
            if kernel_class == "other_gpu" and not is_known_other_kernel(name)
            else "known"
        )
        if classification_status == "unclassified":
            unclassified_intervals.extend(clipped)
        key = (name, kernel_class)
        aggregate = kernel_aggregates.setdefault(
            key,
            {
                "kernel_name": name,
                "kernel_class": kernel_class,
                "semantic_class": semantic_class,
                "classification_status": classification_status,
                "launch_count": 0,
                "total_time_ms": 0.0,
            },
        )
        aggregate["launch_count"] += 1
        aggregate["total_time_ms"] += _milliseconds(clipped)
        lowered = name.lower()
        if "cal_ep_gpu" in lowered:
            setup_intervals.extend(clipped)
        if (
            method == "tagore_vamana"
            and _contains_kernel(lowered, "select_path")
        ) or (
            method == "tagore_cagra" and "select_1hop_cagra" in lowered
        ):
            cfs_primary_intervals.extend(clipped)
        if (
            method == "tagore_vamana"
            and _contains_kernel(lowered, "filter_reverse")
        ) or (
            method == "tagore_cagra" and "filter_reverse_1hop" in lowered
        ):
            cfs_reverse_intervals.extend(clipped)

    for kernel_class in class_intervals:
        class_intervals[kernel_class] = normalize_intervals(class_intervals[kernel_class])
    gpu_kernel = normalize_intervals(
        interval
        for intervals in class_intervals.values()
        for interval in intervals
    )
    transfer = intersect_intervals(memcpy, local)
    cpu_graph = intersect_intervals(
        normalize_intervals(
            (start, end)
            for start, end, text in nvtx_events
            if text.startswith(CPU_GRAPH_PREFIXES)
        ),
        local,
    )
    coordination = intersect_intervals(
        normalize_intervals(
            (start, end)
            for start, end, text in nvtx_events
            if text.startswith(COORDINATION_PREFIX)
        ),
        local,
    )

    critical_path = _exclusive_critical_path(
        local,
        [
            ("dense_gpu", class_intervals["dense"]),
            ("irregular_graph_state_gpu", class_intervals["irregular"]),
            ("mixed_gpu", class_intervals["mixed"]),
            ("other_gpu", class_intervals["other_gpu"]),
            ("transfer", transfer),
            ("cpu_graph", cpu_graph),
            ("coordination", coordination),
        ],
    )
    gpu_path = _exclusive_critical_path(
        gpu_kernel,
        [
            ("dense_gpu", class_intervals["dense"]),
            ("irregular_graph_state_gpu", class_intervals["irregular"]),
            ("mixed_gpu", class_intervals["mixed"]),
            ("other_gpu", class_intervals["other_gpu"]),
        ],
    )
    local_wall_ms = _milliseconds(local)
    gpu_kernel_ms = _milliseconds(gpu_kernel)
    class_active_ms = {
        name: _milliseconds(intervals) for name, intervals in class_intervals.items()
    }
    kernel_rows = sorted(
        kernel_aggregates.values(),
        key=lambda row: (row["kernel_class"], row["kernel_name"]),
    )
    gpu_kernel_elapsed_sum_ms = sum(
        float(row["total_time_ms"]) for row in kernel_rows
    )
    unclassified = [
        row for row in kernel_rows if row["classification_status"] == "unclassified"
    ]
    unclassified_time_ms = _milliseconds(unclassified_intervals)
    critical_path_total = sum(critical_path.values())
    prune_window = intersect_intervals(prune, local)
    prune_api_wall_ms = None if method == "cagra" else _milliseconds(prune_window)
    prune_kernels = intersect_intervals(gpu_kernel, prune_window)
    prune_transfer = intersect_intervals(transfer, prune_window)
    prune_host_residual = (
        []
        if method == "cagra"
        else subtract_intervals(prune_window, [*prune_kernels, *prune_transfer])
    )
    setup_gpu_ms = _summed_intersection_ms(
        setup_intervals,
        prune_window,
    )
    cfs_primary_gpu_ms = _summed_intersection_ms(
        cfs_primary_intervals,
        prune_window,
    )
    cfs_reverse_gpu_ms = _summed_intersection_ms(
        cfs_reverse_intervals,
        prune_window,
    )
    cfs_gpu_ms = cfs_primary_gpu_ms + cfs_reverse_gpu_ms

    return {
        "schema": "wiki10m-local-shard-nsys-v1",
        "method": method,
        "source_sqlite": str(path.resolve()),
        "local_marker_count": sum(
            1 for _, _, text in nvtx_events if text.startswith(local_prefix)
        ),
        "local_build_wall_ms": local_wall_ms,
        "gnn_wall_ms": _milliseconds(intersect_intervals(gnn, local)),
        "prune_wall_ms": prune_api_wall_ms,
        "prune_api_wall_ms": prune_api_wall_ms,
        "prune_setup_gpu_ms": setup_gpu_ms,
        "cfs_primary_gpu_ms": cfs_primary_gpu_ms,
        "cfs_reverse_finalize_gpu_ms": cfs_reverse_gpu_ms,
        "cfs_gpu_ms": cfs_gpu_ms,
        "cfs_time_basis": "summed_nsys_kernel_elapsed_time",
        "cfs_share_of_gpu_kernel_time": (
            0.0
            if gpu_kernel_elapsed_sum_ms == 0
            else cfs_gpu_ms / gpu_kernel_elapsed_sum_ms
        ),
        "cfs_share_of_local_build_wall": (
            0.0 if local_wall_ms == 0 else cfs_gpu_ms / local_wall_ms
        ),
        "prune_transfer_ms": _milliseconds(prune_transfer),
        "prune_host_residual_ms": _milliseconds(prune_host_residual),
        "cfs_logical_stage_split_supported": False,
        "cfs_implementation": (
            "not_applicable"
            if method == "cagra"
            else "collect_filter_fused_plus_reverse_finalize"
        ),
        "gpu_kernel_active_ms": gpu_kernel_ms,
        "gpu_kernel_elapsed_sum_ms": gpu_kernel_elapsed_sum_ms,
        "gpu_transfer_active_ms": _milliseconds(transfer),
        "cpu_graph_active_ms": _milliseconds(cpu_graph),
        "coordination_active_ms": _milliseconds(coordination),
        "gpu_duty": 0.0 if local_wall_ms == 0 else gpu_kernel_ms / local_wall_ms,
        "gpu_class_active_ms": class_active_ms,
        "gpu_class_wall_share": {
            name: 0.0 if local_wall_ms == 0 else value / local_wall_ms
            for name, value in class_active_ms.items()
        },
        "critical_path_exclusive_ms": critical_path,
        "critical_path_share": {
            name: 0.0 if local_wall_ms == 0 else value / local_wall_ms
            for name, value in critical_path.items()
        },
        "critical_path_closure_error_ms": abs(critical_path_total - local_wall_ms),
        "gpu_path_exclusive_ms": gpu_path,
        "gpu_path_exclusive_share": {
            name: 0.0 if gpu_kernel_ms == 0 else value / gpu_kernel_ms
            for name, value in gpu_path.items()
        },
        "gpu_path_closure_error_ms": abs(sum(gpu_path.values()) - gpu_kernel_ms),
        "overlap_ms": {
            "kernel_class_overlap": max(
                0.0, sum(class_active_ms.values()) - gpu_kernel_ms
            ),
            "dense_irregular": _milliseconds(
                intersect_intervals(
                    class_intervals["dense"], class_intervals["irregular"]
                )
            ),
            "dense_mixed": _milliseconds(
                intersect_intervals(class_intervals["dense"], class_intervals["mixed"])
            ),
            "kernel_transfer": _milliseconds(intersect_intervals(gpu_kernel, transfer)),
            "gpu_cpu_graph": _milliseconds(intersect_intervals(gpu_kernel, cpu_graph)),
            "gpu_coordination": _milliseconds(
                intersect_intervals(gpu_kernel, coordination)
            ),
            "cpu_graph_coordination": _milliseconds(
                intersect_intervals(cpu_graph, coordination)
            ),
        },
        "kernel_aggregates": kernel_rows,
        "unclassified_kernels": unclassified,
        "unclassified_kernel_time_ms": unclassified_time_ms,
        "unclassified_gpu_share": (
            0.0 if gpu_kernel_ms == 0 else unclassified_time_ms / gpu_kernel_ms
        ),
    }


def _write_unclassified_tsv(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("method", "kernel_name", "kernel_class", "launch_count", "total_time_ms"))
        for row in report["unclassified_kernels"]:
            writer.writerow(
                (
                    report["method"],
                    row["kernel_name"],
                    row["kernel_class"],
                    row["launch_count"],
                    f"{row['total_time_ms']:.9f}",
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unclassified-tsv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = analyze_sqlite(args.sqlite, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_unclassified_tsv(args.unclassified_tsv, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
