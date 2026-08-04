#!/usr/bin/env python3
"""Validate and summarize the preregistered Wiki10M TESSA ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODES = ("A0", "A1", "A2", "A3")
EXTENSION_MODES = ("W1", "D8")
RECORD_MODES = MODES + EXTENSION_MODES
SQ8_2X2_MODE_MAP = {"D1": "A2", "W1": "W1", "D8": "D8", "W8": "A3"}
FRESH_2X2_MODES = tuple(SQ8_2X2_MODE_MAP.values())
FRESH_DP4A_MODE_MAP = {"D1": "A2", "D8": "D8"}
FRESH_DP4A_MODES = tuple(FRESH_DP4A_MODE_MAP.values())
SEARCH_L_VALUES = (25, 50, 100, 200, 400)
HASH_FIELDS = ("root_leader_hash", "d0_partition_hash", "d0_task_list_hash")


def median_range(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty value list")
    return {"median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1]}


def paired_gain_decision(
    a2_values: Iterable[float], a3_values: Iterable[float], threshold: float = 0.03
) -> dict[str, Any]:
    a2 = list(map(float, a2_values))
    a3 = list(map(float, a3_values))
    if len(a2) != 3 or len(a3) != 3:
        raise ValueError("paired gain requires exactly three A2 and A3 repetitions")
    gains = []
    for baseline, optimized in zip(a2, a3, strict=True):
        if baseline <= 0:
            raise ValueError("paired gain baseline must be positive")
        gains.append((baseline - optimized) / baseline)
    stats = median_range(gains)
    return {"gains": gains, **stats, "stable": stats["median"] >= threshold and stats["min"] > 0.0}


def matched_recall_qps(
    points: Iterable[tuple[float, float]], target: float = 0.95
) -> float | None:
    ordered = [(float(recall), float(qps)) for recall, qps in points]
    for recall, qps in ordered:
        if math.isclose(recall, target, rel_tol=0.0, abs_tol=1e-12):
            return qps
    for (left_recall, left_qps), (right_recall, right_qps) in zip(ordered, ordered[1:]):
        low = min(left_recall, right_recall)
        high = max(left_recall, right_recall)
        if low <= target <= high and not math.isclose(left_recall, right_recall):
            weight = (target - left_recall) / (right_recall - left_recall)
            return left_qps + weight * (right_qps - left_qps)
    return None


def validate_build_records(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mode = str(record.get("mode"))
        if mode not in MODES:
            raise ValueError(f"unknown build mode {mode!r}")
        grouped[mode].append(record)
        for field in ("fallback_count", "overflow_count", "watchdog_count"):
            if int(record.get(field, 0)) != 0:
                raise ValueError(f"nonzero safety counter {field} in {mode} repeat {record.get('repeat')}")

    for mode in MODES:
        repeats = sorted(int(row["repeat"]) for row in grouped[mode])
        if repeats != [1, 2, 3]:
            raise ValueError(f"{mode} must have exactly three repetitions numbered 1,2,3")

    for field in HASH_FIELDS:
        values = {str(row.get(field, "")) for row in records}
        if "" in values or "0" in values or len(values) != 1:
            raise ValueError(f"D0 hash mismatch or missing value for {field}: {sorted(values)}")


def validate_extension_build_records(
    records: list[dict[str, Any]], expected_repeats: tuple[int, ...] = (1, 2, 3)
) -> None:
    expected_modes = ("A2", "A3", "W1", "D8")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mode = str(record.get("mode"))
        if mode not in expected_modes:
            raise ValueError(f"unknown 2x2 build mode {mode!r}")
        grouped[mode].append(record)
        for field in ("fallback_count", "overflow_count", "watchdog_count"):
            if int(record.get(field, 0)) != 0:
                raise ValueError(f"nonzero safety counter {field} in {mode} repeat {record.get('repeat')}")

    for mode in expected_modes:
        repeats = sorted(int(row["repeat"]) for row in grouped[mode])
        if repeats != list(expected_repeats):
            if expected_repeats == (1, 2, 3):
                raise ValueError(f"{mode} must have exactly three repetitions numbered 1,2,3")
            raise ValueError(f"{mode} repetitions {repeats} do not equal {list(expected_repeats)}")

    for field in HASH_FIELDS:
        values = {str(row.get(field, "")) for row in records}
        if "" in values or "0" in values or len(values) != 1:
            raise ValueError(f"D0 hash mismatch or missing value for {field}: {sorted(values)}")

    expected_executor = {
        "A2": "dp4a",
        "A3": "wmma",
        "W1": "wmma",
        "D8": "dp4a",
    }
    for mode, mode_records in grouped.items():
        executor = expected_executor[mode]
        for record in mode_records:
            assignment_dp4a = int(record.get("assignment_dp4a_tiles", 0))
            assignment_wmma = int(record.get("assignment_wmma_tiles", 0))
            leaf_dp4a = int(record.get("leaf_dp4a_tiles", 0))
            leaf_wmma = int(record.get("leaf_wmma_tiles", 0))
            if executor == "dp4a":
                counters_match = (
                    assignment_dp4a > 0
                    and assignment_wmma == 0
                    and leaf_dp4a > 0
                    and leaf_wmma == 0
                )
            else:
                counters_match = (
                    assignment_wmma > 0
                    and assignment_dp4a == 0
                    and leaf_wmma > 0
                    and leaf_dp4a == 0
                )
            if not counters_match:
                raise ValueError(f"{mode} executor tile counters do not match its contract")


def extension_speedup_ratios(cell_medians: dict[str, float]) -> dict[str, float]:
    required = ("D1", "W1", "D8", "W8")
    values = {name: float(cell_medians[name]) for name in required}
    if any(value <= 0 for value in values.values()):
        raise ValueError("all 2x2 cell medians must be positive")
    wmma_batch1 = values["D1"] / values["W1"]
    wmma_batch8 = values["D8"] / values["W8"]
    return {
        "wmma_at_batch1": wmma_batch1,
        "wmma_at_batch8": wmma_batch8,
        "batch8_under_dp4a": values["D1"] / values["D8"],
        "batch8_under_wmma": values["W1"] / values["W8"],
        "executor_batch_interaction": wmma_batch8 / wmma_batch1,
    }


def quality_decision(records: list[dict[str, Any]], tolerance: float = 0.005) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["mode"]), int(record["l"]))].append(record)
    losses = []
    for search_l in sorted({key[1] for key in grouped}):
        baseline = grouped.get(("A0", search_l), [])
        if len(baseline) != 3:
            raise ValueError(f"A0 L={search_l} must have three measured searches")
        baseline_mean = statistics.mean(float(row["recall"]) for row in baseline)
        for mode in MODES[1:]:
            rows = grouped.get((mode, search_l), [])
            if len(rows) != 3:
                raise ValueError(f"{mode} L={search_l} must have three measured searches")
            mode_mean = statistics.mean(float(row["recall"]) for row in rows)
            losses.append({"mode": mode, "l": search_l, "loss": baseline_mean - mode_mean})
    max_loss = max((row["loss"] for row in losses), default=0.0)
    return {"preserved": max_loss <= tolerance, "max_loss": max_loss, "comparisons": losses}


def _nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _sum_named_counters(value: Any, names: set[str]) -> int:
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            if key in names and isinstance(child, (int, float)):
                total += int(child)
            else:
                total += _sum_named_counters(child, names)
        return total
    if isinstance(value, list):
        return sum(_sum_named_counters(child, names) for child in value)
    return 0


def _read_key_values(path: Path) -> dict[str, str]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if "\t" in line:
                key, value = line.split("\t", 1)
                values[key] = value
    return values


def _read_single_tsv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected one TSV data row in {path}, found {len(rows)}")
    return rows[0]


def _max_gpu_memory_mib(path: Path) -> float | None:
    if not path.exists():
        return None
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("memory_used_mib", "").strip()
            if raw:
                values.append(float(raw))
    return max(values) if values else None


def _expected_mode_contract(mode: str) -> dict[str, Any]:
    contracts = {
        "A0": {"assignment_backend": "persistent-tiled-gemm-fp32", "leaf_topk_backend": "persistent-tiled-gemm-fp32", "batch": 1, "sq_assignment": "none", "sq_leaf": "none"},
        "A1": {"assignment_backend": "persistent-wmma-tf32", "leaf_topk_backend": "persistent-wmma-tf32", "batch": 1, "sq_assignment": "none", "sq_leaf": "none"},
        "A2": {"assignment_backend": "persistent-wmma-tf32", "leaf_topk_backend": "persistent-wmma-tf32", "batch": 1, "sq_assignment": "warp-dp4a", "sq_leaf": "warp-dp4a"},
        "A3": {"assignment_backend": "persistent-wmma-tf32", "leaf_topk_backend": "persistent-wmma-tf32", "batch": 8, "sq_assignment": "wmma", "sq_leaf": "wmma"},
        "W1": {"assignment_backend": "persistent-wmma-tf32", "leaf_topk_backend": "persistent-wmma-tf32", "batch": 1, "sq_assignment": "wmma", "sq_leaf": "wmma"},
        "D8": {"assignment_backend": "persistent-wmma-tf32", "leaf_topk_backend": "persistent-wmma-tf32", "batch": 8, "sq_assignment": "warp-dp4a", "sq_leaf": "warp-dp4a"},
    }
    return contracts[mode]


def record_build(case_dir: Path, mode: str, repeat: int) -> dict[str, Any]:
    profile_path = case_dir / "profile.json"
    graph_path = case_dir / "graph_quality.tsv"
    time_path = case_dir / "time.tsv"
    for path in (profile_path, graph_path, time_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing required build artifact: {path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    graph = _read_single_tsv(graph_path)
    timing = _read_key_values(time_path)
    gpu = profile.get("gpu_subtree", {})
    expected = _expected_mode_contract(mode)

    if int(graph["nodes"]) != 10_000_000 or int(graph["self_loop_edges"]) != 0:
        raise ValueError(f"invalid graph structure in {graph_path}")
    for field in ("assignment_backend", "leaf_topk_backend"):
        if gpu.get(field) != expected[field]:
            raise ValueError(f"{mode} actual {field}={gpu.get(field)!r}, expected {expected[field]!r}")
    if int(gpu.get("persistent_large_ticket_batch_actual", 0)) != expected["batch"]:
        raise ValueError(f"{mode} actual ticket batch does not match the frozen contract")
    if gpu.get("persistent_sq8_assignment_kernel_actual", "none") != expected["sq_assignment"]:
        raise ValueError(f"{mode} actual SQ8 assignment executor does not match the frozen contract")
    if gpu.get("persistent_sq8_leaf_executor_actual", "none") != expected["sq_leaf"]:
        raise ValueError(f"{mode} actual SQ8 leaf executor does not match the frozen contract")

    fallback_count = _sum_named_counters(
        profile,
        {"cpu_fallbacks", "persistent_fallback_count", "fallback_count", "assignment_dco_fallback_nodes"},
    )
    overflow_count = _sum_named_counters(
        profile,
        {"persistent_queue_overflow_flags", "queue_overflow_flags", "async_flush_errors"},
    )
    watchdog_count = _sum_named_counters(profile, {"persistent_watchdog_trips", "watchdog_trips"})
    if fallback_count or overflow_count or watchdog_count:
        raise ValueError(
            f"nonzero safety counters: fallback={fallback_count}, overflow={overflow_count}, watchdog={watchdog_count}"
        )

    index_path = case_dir / "index" / "_mem.index"
    if not index_path.is_file() or index_path.stat().st_size == 0:
        raise ValueError(f"missing generated index: {index_path}")
    peak_hbm_mib = _max_gpu_memory_mib(case_dir / "gpu_monitor" / "gpu_samples.csv")
    if peak_hbm_mib is None:
        raise ValueError(f"missing GPU memory samples for {case_dir}")
    record = {
        "mode": mode,
        "repeat": repeat,
        "wall_s": float(timing["wall_s"]),
        "max_rss_kb": int(timing["max_rss_kb"]),
        "stage_total_ms": float(_nested(profile, "stage_ms", "total")),
        "persistent_event_ms": float(_nested(profile, "gpu_subtree_timing", "persistent_kernel_event_ms")),
        "gpu_h2d_bytes": int(_nested(profile, "gpu_subtree_io", "gpu_h2d_bytes", default=0)),
        "gpu_d2h_bytes": int(_nested(profile, "gpu_subtree_io", "gpu_d2h_bytes", default=0)),
        "peak_hbm_mib": peak_hbm_mib,
        "index_bytes": index_path.stat().st_size,
        "avg_out_degree": float(graph["avg_out_degree"]),
        "candidate_artifact_bytes": int(_nested(profile, "temp_files", "candidate", "bytes", default=0)),
        "persistent_topk_rows": int(gpu.get("persistent_topk_rows", 0)),
        "root_leader_hash": str(_nested(profile, "root_fanout", "root_leader_hash", default="")),
        "d0_partition_hash": str(gpu.get("d0_partition_hash", "")),
        "d0_task_list_hash": str(gpu.get("d0_task_list_hash", "")),
        "fallback_count": fallback_count,
        "overflow_count": overflow_count,
        "watchdog_count": watchdog_count,
        "assignment_backend": gpu.get("assignment_backend"),
        "leaf_topk_backend": gpu.get("leaf_topk_backend"),
        "sq_assignment_actual": gpu.get("persistent_sq8_assignment_kernel_actual"),
        "sq_leaf_actual": gpu.get("persistent_sq8_leaf_executor_actual"),
        "ticket_batch_actual": gpu.get("persistent_large_ticket_batch_actual"),
        "assignment_dp4a_tiles": int(gpu.get("persistent_large_sq8_dp4a_tiles", 0)),
        "assignment_wmma_tiles": int(gpu.get("persistent_large_sq8_wmma_tiles", 0)),
        "leaf_dp4a_tiles": int(gpu.get("persistent_leaf_sq8_dp4a_tiles", 0)),
        "leaf_wmma_tiles": int(gpu.get("persistent_leaf_sq8_wmma_tiles", 0)),
        "ticket_claims": int(gpu.get("persistent_large_ticket_range_claims", 0)),
        "physical_tiles": int(gpu.get("persistent_large_physical_tiles", 0)),
    }
    if any(record[field] in ("", "0") for field in HASH_FIELDS):
        raise ValueError(f"missing nonzero D0 hash in {profile_path}")
    output = case_dir / "build_record.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def record_search(mode: str, search_l: int, repeat: int, log_path: Path, result_path: Path, output: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8")
    search = re.search(r"Search:\s+(\d+)\s+queries in\s+([0-9.]+)s\s+\(([0-9.]+)\s+QPS\)", text)
    recall = re.search(r"Recall@(\d+):\s+([0-9.]+)", text)
    if search is None or recall is None or int(recall.group(1)) != 10:
        raise ValueError(f"failed to parse Recall@10 and QPS from {log_path}")
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise ValueError(f"missing search result {result_path}")
    record = {
        "mode": mode,
        "l": search_l,
        "repeat": repeat,
        "queries": int(search.group(1)),
        "wall_s": float(search.group(2)),
        "qps": float(search.group(3)),
        "recall": float(recall.group(2)),
        "result_bytes": result_path.stat().st_size,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def load_records(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    builds = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_root / "full_builds").glob("A*/run*/build_record.json"))]
    searches = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_root / "search").glob("A*/L*/rep[123].json"))]
    return builds, searches


def load_2x2_build_records(run_root: Path) -> list[dict[str, Any]]:
    paths = []
    for mode, base in (
        ("A2", run_root / "full_builds" / "A2"),
        ("A3", run_root / "full_builds" / "A3"),
        ("W1", run_root / "full_build_2x2" / "full_builds" / "W1"),
        ("D8", run_root / "full_build_2x2" / "full_builds" / "D8"),
    ):
        paths.extend(base / f"run{repeat}" / "build_record.json" for repeat in (1, 2, 3))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_2x2_search_records(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for mode in ("A0", "A2", "A3", "W1", "D8"):
        if mode in ("W1", "D8"):
            base = run_root / "full_build_2x2" / "search" / mode
        else:
            base = run_root / "search" / mode
        for search_l in SEARCH_L_VALUES:
            for repeat in (1, 2, 3):
                path = base / f"L{search_l}" / f"rep{repeat}.json"
                records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def load_fresh_2x2_build_records(run_root: Path) -> list[dict[str, Any]]:
    paths = [
        run_root / "full_builds" / mode / f"run{repeat}" / "build_record.json"
        for mode in FRESH_2X2_MODES
        for repeat in (1, 2, 3)
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_fresh_2x2_search_records(run_root: Path) -> list[dict[str, Any]]:
    paths = [
        run_root / "search" / mode / f"L{search_l}" / f"rep{repeat}.json"
        for mode in FRESH_2X2_MODES
        for search_l in SEARCH_L_VALUES
        for repeat in (1, 2, 3)
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_fresh_dp4a_build_records(run_root: Path) -> list[dict[str, Any]]:
    paths = [
        run_root / "full_builds" / mode / f"run{repeat}" / "build_record.json"
        for mode in FRESH_DP4A_MODES
        for repeat in (1, 2, 3)
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_fresh_dp4a_search_records(run_root: Path) -> list[dict[str, Any]]:
    paths = [
        run_root / "search" / mode / f"L{search_l}" / f"rep{repeat}.json"
        for mode in FRESH_DP4A_MODES
        for search_l in SEARCH_L_VALUES
        for repeat in (1, 2, 3)
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def validate_dp4a_build_records(
    records: list[dict[str, Any]], expected_repeats: tuple[int, ...] = (1, 2, 3)
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mode = str(record.get("mode"))
        if mode not in FRESH_DP4A_MODES:
            raise ValueError(f"unknown fresh DP4A build mode {mode!r}")
        grouped[mode].append(record)
        for field in ("fallback_count", "overflow_count", "watchdog_count"):
            if int(record.get(field, 0)) != 0:
                raise ValueError(
                    f"nonzero safety counter {field} in {mode} repeat {record.get('repeat')}"
                )
        if not (
            int(record.get("assignment_dp4a_tiles", 0)) > 0
            and int(record.get("assignment_wmma_tiles", 0)) == 0
            and int(record.get("leaf_dp4a_tiles", 0)) > 0
            and int(record.get("leaf_wmma_tiles", 0)) == 0
        ):
            raise ValueError(f"{mode} executor tile counters do not match packed DP4A")

    for mode in FRESH_DP4A_MODES:
        repeats = sorted(int(row["repeat"]) for row in grouped[mode])
        if repeats != list(expected_repeats):
            raise ValueError(f"{mode} repetitions {repeats} do not equal {list(expected_repeats)}")
    for field in HASH_FIELDS:
        values = {str(row.get(field, "")) for row in records}
        if "" in values or "0" in values or len(values) != 1:
            raise ValueError(f"D0 hash mismatch or missing value for {field}: {sorted(values)}")


def validate_2x2_search_records(
    records: list[dict[str, Any]],
    modes: tuple[str, ...] = ("A0", "A2", "A3", "W1", "D8"),
) -> None:
    expected_keys = {
        (mode, search_l, repeat)
        for mode in modes
        for search_l in SEARCH_L_VALUES
        for repeat in (1, 2, 3)
    }
    actual_keys = {
        (str(row.get("mode")), int(row.get("l", 0)), int(row.get("repeat", 0))) for row in records
    }
    if len(records) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("2x2 search matrix is incomplete or contains duplicate cells")
    for row in records:
        mode = str(row["mode"])
        search_l = int(row["l"])
        repeat = int(row["repeat"])
        if int(row.get("queries", 0)) != 10_000:
            raise ValueError(f"{mode} L={search_l} repeat={repeat} must contain 10000 queries")
        if float(row.get("qps", 0.0)) <= 0 or int(row.get("result_bytes", 0)) <= 0:
            raise ValueError(f"invalid search throughput or result size in {mode} L={search_l} repeat={repeat}")
        recall = float(row.get("recall", -1.0))
        if not 0.0 <= recall <= 1.0:
            raise ValueError(f"invalid recall in {mode} L={search_l} repeat={repeat}: {recall}")


def summarize(run_root: Path) -> None:
    builds, searches = load_records(run_root)
    validate_build_records(builds)
    expected_searches = len(MODES) * len(SEARCH_L_VALUES) * 3
    if len(searches) != expected_searches:
        raise ValueError(f"expected {expected_searches} measured searches, found {len(searches)}")
    search_keys = {(row["mode"], int(row["l"]), int(row["repeat"])) for row in searches}
    expected_keys = {(mode, search_l, repeat) for mode in MODES for search_l in SEARCH_L_VALUES for repeat in (1, 2, 3)}
    if search_keys != expected_keys:
        raise ValueError("search matrix is incomplete or contains duplicate cells")

    quality = quality_decision(searches)
    by_mode = {mode: sorted((row for row in builds if row["mode"] == mode), key=lambda row: row["repeat"]) for mode in MODES}
    wall_gain = paired_gain_decision([row["wall_s"] for row in by_mode["A2"]], [row["wall_s"] for row in by_mode["A3"]])
    event_gain = paired_gain_decision([row["persistent_event_ms"] for row in by_mode["A2"]], [row["persistent_event_ms"] for row in by_mode["A3"]])

    rows = []
    for mode in MODES:
        for metric in ("wall_s", "stage_total_ms", "persistent_event_ms", "gpu_h2d_bytes", "gpu_d2h_bytes", "peak_hbm_mib", "max_rss_kb", "index_bytes", "avg_out_degree", "candidate_artifact_bytes", "persistent_topk_rows"):
            stats = median_range(row[metric] for row in by_mode[mode])
            rows.append({"section": "build", "mode": mode, "metric": metric, **stats, "n": 3, "note": ""})

    search_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in searches:
        search_grouped[(row["mode"], int(row["l"]))].append(row)
    matched = {}
    for mode in MODES:
        points = []
        for search_l in SEARCH_L_VALUES:
            cell = search_grouped[(mode, search_l)]
            recall_values = [float(row["recall"]) for row in cell]
            qps_values = [float(row["qps"]) for row in cell]
            recall_stats = median_range(recall_values)
            qps_stats = median_range(qps_values)
            recall_stats["median"] = statistics.mean(recall_values)
            qps_stats["median"] = statistics.mean(qps_values)
            rows.extend([
                {"section": "search", "mode": mode, "metric": f"recall_at_10_L{search_l}", **recall_stats, "n": 3, "note": "central column is mean; range is min/max"},
                {"section": "search", "mode": mode, "metric": f"qps_L{search_l}", **qps_stats, "n": 3, "note": "central column is mean; range is min/max"},
            ])
            points.append((statistics.mean(recall_values), statistics.mean(qps_values)))
        matched[mode] = matched_recall_qps(points, 0.95)
        value = matched[mode]
        rows.append({"section": "search", "mode": mode, "metric": "matched_recall_0.95_qps", "median": value if value is not None else "NA", "min": value if value is not None else "NA", "max": value if value is not None else "NA", "n": 5, "note": "adjacent measured-point interpolation; no extrapolation"})

    for metric, decision in (("a3_vs_a2_wall_gain", wall_gain), ("a3_vs_a2_persistent_event_gain", event_gain)):
        rows.append({"section": "decision", "mode": "A3", "metric": metric, "median": decision["median"], "min": decision["min"], "max": decision["max"], "n": 3, "note": "stable" if decision["stable"] else "no stable system gain"})
    rows.append({"section": "decision", "mode": "all", "metric": "quality_preserved", "median": int(quality["preserved"]), "min": quality["max_loss"], "max": quality["max_loss"], "n": len(quality["comparisons"]), "note": "loss threshold <= 0.005"})

    operator_summary_path = run_root / "operator_replay" / "summary.json"
    if not operator_summary_path.is_file():
        raise ValueError(f"missing operator replay summary: {operator_summary_path}")
    operator_summary = json.loads(operator_summary_path.read_text(encoding="utf-8"))
    if not operator_summary.get("a2_a3_candidate_ids_and_distance_bits_exact", False):
        raise ValueError("operator replay did not prove A2/A3 exact candidate IDs and distance bits")
    operator_assignment_2x2 = {}
    for mode in operator_summary.get("modes", []):
        mode_name = str(mode["mode"])
        operator_assignment_2x2[mode_name] = float(mode["assignment_event_ms"]["median"])
        for key in ("assignment_event_ms", "leaf_event_ms", "total_event_ms"):
            distribution = mode.get(key)
            if distribution is None:
                continue
            rows.append({"section": "operator", "mode": mode_name, "metric": key, "median": distribution["median"], "min": distribution["min"], "max": distribution["max"], "n": 20, "note": "GPU event time (ms)"})
        for key in ("assignment_throughput_pairs_per_s", "leaf_throughput_pairs_per_s", "assignment_logical_payload_bytes", "leaf_logical_payload_bytes"):
            value = mode.get(key)
            if value is not None:
                rows.append({"section": "operator", "mode": mode_name, "metric": key, "median": value, "min": value, "max": value, "n": 1, "note": "logical payload, not hardware DRAM transactions" if "payload" in key else ""})
        for key in ("assignment_overlap_at_rh", "leaf_overlap_at_k"):
            overlap = mode.get(key)
            if overlap is None:
                continue
            for overlap_key in ("macro_mean", "pair_weighted_mean", "p5", "p50", "p95", "exact_task_fraction"):
                rows.append({"section": "operator", "mode": mode_name, "metric": f"{key}_{overlap_key}", "median": overlap[overlap_key], "min": overlap[overlap_key], "max": overlap[overlap_key], "n": overlap["task_count"], "note": "A0 deterministic FP32 reference"})

    result_path = run_root / "results.tsv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "mode", "metric", "median", "min", "max", "n", "note"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    stable = wall_gain["stable"] and event_gain["stable"]
    lines = [
        "# TESSA Wiki10M progressive ablation summary",
        "",
        f"- Build matrix: valid (A0--A3, three from-scratch repetitions each; D0 hashes identical).",
        f"- A3 vs A2 external-wall paired gain: median {wall_gain['median']:.3%}, range [{wall_gain['min']:.3%}, {wall_gain['max']:.3%}].",
        f"- A3 vs A2 persistent-event paired gain: median {event_gain['median']:.3%}, range [{event_gain['min']:.3%}, {event_gain['max']:.3%}].",
        f"- Stability decision: {'stable >=3% gain with all paired gains positive' if stable else 'no stable system gain'}.",
        f"- Quality decision: {'preserved' if quality['preserved'] else 'not preserved'}; maximum fixed-L Recall@10 loss vs A0 is {quality['max_loss']:.6f} (threshold 0.005).",
        "",
        "## Matched Recall@10 = 0.95 QPS",
        "",
    ]
    for mode in MODES:
        lines.append(f"- {mode}: {matched[mode]:.3f}" if matched[mode] is not None else f"- {mode}: NA (target not bracketed; no extrapolation)")
    lines.extend([
        "",
        "## Frozen assignment replay: executor × ticket batch",
        "",
        "| Executor | batch-1 median event (ms) | batch-8 median event (ms) |",
        "|---|---:|---:|",
        f"| DP4A | {operator_assignment_2x2['a2-sq8-dp4a-batch1']:.6f} | {operator_assignment_2x2['sq8-dp4a-batch8']:.6f} |",
        f"| WMMA | {operator_assignment_2x2['sq8-wmma-batch1']:.6f} | {operator_assignment_2x2['a3-sq8-wmma-batch8']:.6f} |",
        "",
        "## Attribution limits",
        "",
        "A0--A1 includes both the hardware path and TF32 rounding. A1--A2 changes both representation and executor. A2--A3 is the coupled TESSA package (DP4A→WMMA plus batch-1→batch-8); only the frozen assignment 2×2 replay may separate executor, batching, and interaction effects. Batch-8 applies only to large assignment tasks and is not attributed to leaf scoring.",
        "",
        f"Operator replay details: `{operator_summary_path}`.",
    ])
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_root / "summary.json").write_text(json.dumps({"wall_gain": wall_gain, "event_gain": event_gain, "quality": quality, "matched_recall_qps": matched, "operator_assignment_2x2_median_ms": operator_assignment_2x2}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_2x2(run_root: Path, *, fresh: bool) -> None:
    builds = (
        load_fresh_2x2_build_records(run_root)
        if fresh
        else load_2x2_build_records(run_root)
    )
    validate_extension_build_records(builds)
    searches = (
        load_fresh_2x2_search_records(run_root)
        if fresh
        else load_2x2_search_records(run_root)
    )
    search_modes = FRESH_2X2_MODES if fresh else ("A0", "A2", "A3", "W1", "D8")
    validate_2x2_search_records(searches, search_modes)

    by_mode = {
        mode: sorted((row for row in builds if row["mode"] == mode), key=lambda row: row["repeat"])
        for mode in ("A2", "A3", "W1", "D8")
    }
    cell_records = {cell: by_mode[mode] for cell, mode in SQ8_2X2_MODE_MAP.items()}
    cell_stats = {}
    result_rows = []
    for cell, rows in cell_records.items():
        source_mode = SQ8_2X2_MODE_MAP[cell]
        wall = median_range(row["wall_s"] for row in rows)
        event = median_range(row["persistent_event_ms"] for row in rows)
        cell_stats[cell] = {
            "source_mode": source_mode,
            "wall_s": wall,
            "persistent_event_ms": event,
        }
        for metric, stats in (("wall_s", wall), ("persistent_event_ms", event)):
            result_rows.append(
                {
                    "section": "build",
                    "mode": cell,
                    "metric": metric,
                    **stats,
                    "n": 3,
                    "note": f"source mode {source_mode}",
                }
            )

    wall_ratios = extension_speedup_ratios(
        {cell: float(stats["wall_s"]["median"]) for cell, stats in cell_stats.items()}
    )
    event_ratios = extension_speedup_ratios(
        {cell: float(stats["persistent_event_ms"]["median"]) for cell, stats in cell_stats.items()}
    )

    search_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in searches:
        search_grouped[(str(row["mode"]), int(row["l"]))].append(row)

    matched = {}
    search_stats = {}
    for cell, source_mode in SQ8_2X2_MODE_MAP.items():
        points = []
        search_stats[cell] = {}
        for search_l in SEARCH_L_VALUES:
            rows = search_grouped[(source_mode, search_l)]
            recalls = [float(row["recall"]) for row in rows]
            qps_values = [float(row["qps"]) for row in rows]
            recall_stats = median_range(recalls)
            qps_stats = median_range(qps_values)
            recall_stats["mean"] = statistics.mean(recalls)
            qps_stats["mean"] = statistics.mean(qps_values)
            search_stats[cell][str(search_l)] = {"recall": recall_stats, "qps": qps_stats}
            points.append((recall_stats["mean"], qps_stats["mean"]))
            result_rows.extend(
                [
                    {
                        "section": "search",
                        "mode": cell,
                        "metric": f"recall_at_10_L{search_l}",
                        "median": recall_stats["mean"],
                        "min": recall_stats["min"],
                        "max": recall_stats["max"],
                        "n": 3,
                        "note": f"mean; source mode {source_mode}",
                    },
                    {
                        "section": "search",
                        "mode": cell,
                        "metric": f"qps_L{search_l}",
                        "median": qps_stats["mean"],
                        "min": qps_stats["min"],
                        "max": qps_stats["max"],
                        "n": 3,
                        "note": f"mean; source mode {source_mode}",
                    },
                ]
            )
        matched[cell] = matched_recall_qps(points, 0.95)

    quality_comparisons = []
    baseline_mode = "A2" if fresh else "A0"
    compared_modes = ("W1", "D8", "A3") if fresh else EXTENSION_MODES
    for search_l in SEARCH_L_VALUES:
        baseline = statistics.mean(
            float(row["recall"]) for row in search_grouped[(baseline_mode, search_l)]
        )
        for mode in compared_modes:
            observed = statistics.mean(float(row["recall"]) for row in search_grouped[(mode, search_l)])
            quality_comparisons.append({"mode": mode, "l": search_l, "loss": baseline - observed})
    max_loss = max((row["loss"] for row in quality_comparisons), default=0.0)
    quality = {
        "preserved": max_loss <= 0.005,
        "max_loss": max_loss,
        "tolerance": 0.005,
        "comparisons": quality_comparisons,
    }

    output_root = run_root if fresh else run_root / "full_build_2x2"
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "results.tsv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "mode", "metric", "median", "min", "max", "n", "note"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "schema": (
            "tessa-full-build-2x2-fresh-v1"
            if fresh
            else "tessa-full-build-2x2-extension-v1"
        ),
        "post_hoc": not fresh,
        "cells": cell_stats,
        "wall_speedup_ratios": wall_ratios,
        "event_speedup_ratios": event_ratios,
        "quality": quality,
        "matched_recall_qps": matched,
        "search": search_stats,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    batch1_claim = (
        "WMMA is faster at batch 1 on both measured layers."
        if wall_ratios["wmma_at_batch1"] > 1.0 and event_ratios["wmma_at_batch1"] > 1.0
        else "WMMA is not faster at batch 1 on both measured layers."
    )
    batch8_claim = (
        "WMMA is faster at batch 8 on both measured layers."
        if wall_ratios["wmma_at_batch8"] > 1.0 and event_ratios["wmma_at_batch8"] > 1.0
        else "WMMA is not faster at batch 8 on both measured layers."
    )
    title = (
        "# TESSA Wiki10M fresh full-build 2×2 summary"
        if fresh
        else "# TESSA Wiki10M full-build 2×2 extension summary"
    )
    provenance = (
        "All four cells were built from scratch with the same release binary and interleaved execution order."
        if fresh
        else "This is a post-hoc extension. D1/A2 and W8/A3 reuse the accepted parent run; W1 and D8 were measured later, so repeat ordinals are not paired across all four cells."
    )
    lines = [
        title,
        "",
        provenance,
        "",
        "## Build medians and ranges",
        "",
        "| Cell | Executor | Batch | Wall median [min, max] (s) | Persistent event median [min, max] (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "D1": ("DP4A", 1),
        "W1": ("WMMA", 1),
        "D8": ("DP4A", 8),
        "W8": ("WMMA", 8),
    }
    for cell in ("D1", "W1", "D8", "W8"):
        executor, batch = labels[cell]
        wall = cell_stats[cell]["wall_s"]
        event = cell_stats[cell]["persistent_event_ms"]
        lines.append(
            f"| {cell} | {executor} | {batch} | {wall['median']:.3f} [{wall['min']:.3f}, {wall['max']:.3f}] | "
            f"{event['median']:.3f} [{event['min']:.3f}, {event['max']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Speedup ratios",
            "",
            "A ratio above 1 means the denominator/optimized cell is faster. The interaction ratio above 1 means batch 8 increases WMMA's relative advantage over DP4A.",
            "",
            "| Contrast | Wall ratio | Persistent-event ratio |",
            "|---|---:|---:|",
        ]
    )
    ratio_labels = (
        ("WMMA at batch 1", "wmma_at_batch1"),
        ("WMMA at batch 8", "wmma_at_batch8"),
        ("batch 8 under DP4A", "batch8_under_dp4a"),
        ("batch 8 under WMMA", "batch8_under_wmma"),
        ("executor×batch interaction", "executor_batch_interaction"),
    )
    for label, key in ratio_labels:
        lines.append(f"| {label} | {wall_ratios[key]:.6f} | {event_ratios[key]:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {batch1_claim}",
            f"- {batch8_claim}",
            f"- Search quality: {'preserved' if quality['preserved'] else 'not preserved'}; maximum fixed-L Recall@10 loss vs {baseline_mode} is {max_loss:.6f} (threshold 0.005).",
            "- Ticket batching applies only to large assignment work and is not attributed to leaf scoring.",
            "",
            "## Matched Recall@10 = 0.95 QPS",
            "",
        ]
    )
    for cell in ("D1", "W1", "D8", "W8"):
        value = matched[cell]
        lines.append(f"- {cell}: {value:.3f}" if value is not None else f"- {cell}: NA (target not bracketed)")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_2x2(run_root: Path) -> None:
    _summarize_2x2(run_root, fresh=False)


def summarize_fresh_2x2(run_root: Path) -> None:
    _summarize_2x2(run_root, fresh=True)


def summarize_fresh_dp4a(run_root: Path) -> None:
    builds = load_fresh_dp4a_build_records(run_root)
    validate_dp4a_build_records(builds)
    searches = load_fresh_dp4a_search_records(run_root)
    validate_2x2_search_records(searches, FRESH_DP4A_MODES)

    by_mode = {
        mode: sorted(
            (row for row in builds if row["mode"] == mode), key=lambda row: row["repeat"]
        )
        for mode in FRESH_DP4A_MODES
    }
    cells = {}
    result_rows = []
    for cell, mode in FRESH_DP4A_MODE_MAP.items():
        wall = median_range(row["wall_s"] for row in by_mode[mode])
        event = median_range(row["persistent_event_ms"] for row in by_mode[mode])
        cells[cell] = {
            "source_mode": mode,
            "wall_s": wall,
            "persistent_event_ms": event,
        }
        for metric, stats in (("wall_s", wall), ("persistent_event_ms", event)):
            result_rows.append(
                {
                    "section": "build",
                    "mode": cell,
                    "metric": metric,
                    **stats,
                    "n": 3,
                    "note": f"source mode {mode}",
                }
            )

    search_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in searches:
        search_grouped[(str(row["mode"]), int(row["l"]))].append(row)
    search_stats = {}
    matched = {}
    for cell, mode in FRESH_DP4A_MODE_MAP.items():
        search_stats[cell] = {}
        points = []
        for search_l in SEARCH_L_VALUES:
            rows = search_grouped[(mode, search_l)]
            recalls = [float(row["recall"]) for row in rows]
            qps_values = [float(row["qps"]) for row in rows]
            recall = {**median_range(recalls), "mean": statistics.mean(recalls)}
            qps = {**median_range(qps_values), "mean": statistics.mean(qps_values)}
            search_stats[cell][str(search_l)] = {"recall": recall, "qps": qps}
            points.append((recall["mean"], qps["mean"]))
            for metric, stats in (
                (f"recall_at_10_L{search_l}", recall),
                (f"qps_L{search_l}", qps),
            ):
                result_rows.append(
                    {
                        "section": "search",
                        "mode": cell,
                        "metric": metric,
                        "median": stats["mean"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "n": 3,
                        "note": f"mean; source mode {mode}",
                    }
                )
        matched[cell] = matched_recall_qps(points, 0.95)

    quality_comparisons = []
    for search_l in SEARCH_L_VALUES:
        d1 = statistics.mean(
            float(row["recall"]) for row in search_grouped[("A2", search_l)]
        )
        d8 = statistics.mean(
            float(row["recall"]) for row in search_grouped[("D8", search_l)]
        )
        quality_comparisons.append({"l": search_l, "loss": d1 - d8})
    max_loss = max([0.0, *(row["loss"] for row in quality_comparisons)])
    batch8_speedup = {
        "wall": cells["D1"]["wall_s"]["median"] / cells["D8"]["wall_s"]["median"],
        "persistent_event": (
            cells["D1"]["persistent_event_ms"]["median"]
            / cells["D8"]["persistent_event_ms"]["median"]
        ),
    }

    with (run_root / "results.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "mode", "metric", "median", "min", "max", "n", "note"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "schema": "tessa-full-build-dp4a-fresh-v1",
        "post_hoc": False,
        "cells": cells,
        "batch8_speedup": batch8_speedup,
        "quality": {
            "preserved": max_loss <= 0.005,
            "max_loss": max_loss,
            "tolerance": 0.005,
            "comparisons": quality_comparisons,
        },
        "matched_recall_qps": matched,
        "search": search_stats,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# TESSA Wiki10M fresh DP4A summary",
        "",
        "D1 and D8 were built from scratch with the same release binary and interleaved execution order.",
        "",
        "| Cell | Batch | Wall median [min, max] (s) | Persistent event median [min, max] (ms) |",
        "|---|---:|---:|---:|",
    ]
    for cell, batch in (("D1", 1), ("D8", 8)):
        wall = cells[cell]["wall_s"]
        event = cells[cell]["persistent_event_ms"]
        lines.append(
            f"| {cell} | {batch} | {wall['median']:.3f} [{wall['min']:.3f}, {wall['max']:.3f}] | "
            f"{event['median']:.3f} [{event['min']:.3f}, {event['max']:.3f}] |"
        )
    lines.extend(
        [
            "",
            f"- Batch-8 speedup: {batch8_speedup['wall']:.6f}× by wall and "
            f"{batch8_speedup['persistent_event']:.6f}× by persistent event.",
            f"- Search quality: {'preserved' if max_loss <= 0.005 else 'not preserved'}; "
            f"maximum fixed-L Recall@10 loss of D8 versus D1 is {max_loss:.6f}.",
        ]
    )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_build_parser = subparsers.add_parser("record-build")
    record_build_parser.add_argument("--case-dir", type=Path, required=True)
    record_build_parser.add_argument("--mode", choices=RECORD_MODES, required=True)
    record_build_parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    record_search_parser = subparsers.add_parser("record-search")
    record_search_parser.add_argument("--mode", choices=RECORD_MODES, required=True)
    record_search_parser.add_argument("--search-l", type=int, choices=SEARCH_L_VALUES, required=True)
    record_search_parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    record_search_parser.add_argument("--log", type=Path, required=True)
    record_search_parser.add_argument("--result", type=Path, required=True)
    record_search_parser.add_argument("--output", type=Path, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("run_root", type=Path)
    summarize_2x2_parser = subparsers.add_parser("summarize-2x2")
    summarize_2x2_parser.add_argument("run_root", type=Path)
    summarize_fresh_2x2_parser = subparsers.add_parser("summarize-fresh-2x2")
    summarize_fresh_2x2_parser.add_argument("run_root", type=Path)
    summarize_fresh_dp4a_parser = subparsers.add_parser("summarize-fresh-dp4a")
    summarize_fresh_dp4a_parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "record-build":
        print(json.dumps(record_build(args.case_dir, args.mode, args.repeat), sort_keys=True))
    elif args.command == "record-search":
        print(json.dumps(record_search(args.mode, args.search_l, args.repeat, args.log, args.result, args.output), sort_keys=True))
    elif args.command == "summarize":
        summarize(args.run_root.resolve())
    elif args.command == "summarize-2x2":
        summarize_2x2(args.run_root.resolve())
    elif args.command == "summarize-fresh-2x2":
        summarize_fresh_2x2(args.run_root.resolve())
    else:
        summarize_fresh_dp4a(args.run_root.resolve())


if __name__ == "__main__":
    main()
