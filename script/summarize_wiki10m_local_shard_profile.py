#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
import struct
from pathlib import Path

try:
    from script.analyze_wiki10m_local_shard_ncu import (
        aggregate_ncu,
        read_ncu_csv,
    )
    from script.analyze_wiki10m_local_shard_nsys import (
        analyze_sqlite,
        semantic_kernel_class,
    )
except ModuleNotFoundError:
    from analyze_wiki10m_local_shard_ncu import aggregate_ncu, read_ncu_csv
    from analyze_wiki10m_local_shard_nsys import (
        analyze_sqlite,
        semantic_kernel_class,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_SOURCE_PATHS = tuple(
    REPO_ROOT / relative
    for relative in (
        "script/wiki10m_local_shard_profile.py",
        "script/analyze_wiki10m_local_shard_nsys.py",
        "script/analyze_wiki10m_local_shard_ncu.py",
        "script/summarize_wiki10m_local_shard_profile.py",
        "script/run_wiki10m_local_shard_gpu_profile.sh",
        "docs/runbooks/wiki10m_local_shard_gpu_path_profile.md",
    )
)
METHODS = ("cagra", "tagore_cagra", "tagore_vamana")
CONTRACT_FIELDS = (
    "shard_sha256",
    "rows",
    "dim",
    "metric",
    "intermediate_graph_degree",
    "graph_degree",
    "iterations",
    "topm",
    "threshold",
    "seed",
    "entry_point",
)
EXPECTED_NCU_CLASSES = {
    "cagra": ("dense", "irregular", "mixed"),
    "tagore_cagra": ("dense", "irregular_cfs", "mixed"),
    "tagore_vamana": ("dense", "mixed_cfs", "mixed"),
}
NCU_COVERAGE_THRESHOLD = 0.80
NCU_FIELDS = (
    "sm_pct",
    "tensor_pct",
    "dram_pct",
    "l2_pct",
    "occupancy_pct",
    "long_scoreboard_pct",
    "barrier_pct",
    "atomic_instructions",
)
CRITICAL_PATH_FIELDS = (
    "dense_gpu",
    "irregular_graph_state_gpu",
    "mixed_gpu",
    "other_gpu",
    "transfer",
    "cpu_graph",
    "coordination",
    "residual",
)
GPU_PATH_FIELDS = (
    "dense_gpu",
    "irregular_graph_state_gpu",
    "mixed_gpu",
    "other_gpu",
    "residual",
)


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {label} evidence: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} evidence: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {label} evidence object: {path}")
    return payload


def _sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label} evidence: {path}")


def _validate_provenance(run_root: Path) -> None:
    path = run_root / "provenance/sha256.txt"
    _require_nonempty_file(path, "provenance SHA-256")
    recorded = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
        ):
            raise ValueError(f"invalid provenance SHA-256 row: {line!r}")
        artifact = Path(parts[1]).resolve()
        key = str(artifact)
        if key in recorded:
            raise ValueError(f"duplicate provenance SHA-256 path: {artifact}")
        _require_nonempty_file(artifact, "provenance artifact")
        if _sha256_file(artifact) != parts[0]:
            raise ValueError(f"provenance artifact hash mismatch: {artifact}")
        recorded[key] = parts[0]
    for source in PROVENANCE_SOURCE_PATHS:
        key = str(source.resolve())
        if recorded.get(key) != _sha256_file(source):
            raise ValueError(f"provenance source hash mismatch: {source}")


def _require_success_evidence(
    case_root: Path,
    label: str,
    *,
    require_time_log: bool = False,
) -> str:
    exit_path = case_root / "exit_code"
    command_path = case_root / "command.sh"
    _require_nonempty_file(exit_path, f"{label} exit")
    _require_nonempty_file(command_path, f"{label} command")
    if exit_path.read_text().strip() != "0":
        raise ValueError(f"{label} exit status is not zero: {exit_path}")
    command = command_path.read_text().strip()
    if require_time_log:
        time_path = case_root / "time.log"
        _require_nonempty_file(time_path, f"{label} time log")
        if "Exit status: 0" not in time_path.read_text():
            raise ValueError(f"{label} time log lacks successful exit: {time_path}")
    return command


def _profile_contract(profile: dict) -> dict:
    return {field: profile.get(field) for field in CONTRACT_FIELDS}


def _validate_profile_contract(
    profile: dict,
    contract: dict,
    path: Path,
) -> None:
    observed = _profile_contract(profile)
    missing = [field for field, value in observed.items() if value is None]
    if missing:
        raise ValueError(f"profile is missing parameters {missing}: {path}")
    for field in CONTRACT_FIELDS:
        if observed[field] != contract[field]:
            raise ValueError(
                f"{field} mismatch: {observed[field]!r} != "
                f"{contract[field]!r} in {path}"
            )


def _load_gpu_monitor(path: Path) -> dict:
    _require_nonempty_file(path, "GPU memory monitor")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"GPU memory monitor has no header: {path}")
        index_field = next(
            (
                field
                for field in reader.fieldnames
                if field.strip().lower() == "index"
            ),
            None,
        )
        if index_field is None:
            raise ValueError(f"GPU memory monitor lacks GPU index: {path}")
        memory_field = next(
            (
                field
                for field in reader.fieldnames
                if field.strip().lower() == "memory.used [mib]"
            ),
            None,
        )
        if memory_field is None:
            raise ValueError(f"GPU memory monitor lacks memory.used [MiB]: {path}")
        values = []
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            raw_index = (row.get(index_field) or "").strip()
            try:
                gpu_index = int(raw_index)
            except ValueError as error:
                raise ValueError(f"invalid GPU index in {path}") from error
            if gpu_index != 0:
                raise ValueError(
                    f"GPU memory monitor contains nonzero device index in {path}"
                )
            raw = (row.get(memory_field) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError as error:
                raise ValueError(f"invalid GPU memory sample in {path}") from error
    if len(values) < 2:
        raise ValueError(
            f"GPU memory monitor requires baseline plus peak samples: {path}"
        )
    baseline = values[0]
    peak = max(values)
    return {
        "sample_count": len(values),
        "baseline_mib": baseline,
        "peak_mib": peak,
        "peak_delta_mib": max(0.0, peak - baseline),
    }


def _load_timing(
    run_root: Path, allow_single_repeat: bool
) -> tuple[dict[str, dict], dict]:
    methods = {}
    reference_contract = None
    reference_hbm = None
    for method in METHODS:
        paths = sorted((run_root / "timing" / method).glob("repeat_*/profile.json"))
        expected = 1 if allow_single_repeat else 3
        if len(paths) != expected:
            raise ValueError(
                f"{method} has {len(paths)} timing repeats; expected {expected}"
            )
        profiles = []
        elapsed = []
        memory_samples = []
        for path in paths:
            case_root = path.parent
            command = _require_success_evidence(
                case_root,
                f"timing {method}",
                require_time_log=True,
            )
            if f"--method {method}" not in command:
                raise ValueError(f"timing command method mismatch: {case_root}")
            profile = _load_json(path, "timing")
            if profile.get("method") != method:
                raise ValueError(f"timing method mismatch in {path}")
            if profile.get("profiled_timing") is not False:
                raise ValueError(f"profiled timing was used as canonical timing: {path}")
            value = profile.get("local_build_wall_ms")
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"invalid local build timing in {path}")
            contract = _profile_contract(profile)
            missing = [field for field, item in contract.items() if item is None]
            if missing:
                raise ValueError(f"timing profile is missing parameters {missing}: {path}")
            if reference_contract is None:
                reference_contract = contract
            else:
                for field in CONTRACT_FIELDS:
                    if contract[field] != reference_contract[field]:
                        raise ValueError(
                            f"{field} mismatch: {contract[field]!r} != "
                            f"{reference_contract[field]!r} in {path}"
                        )
            hbm = profile.get("analytical_hbm_bytes")
            if not isinstance(hbm, dict) or not hbm:
                raise ValueError(f"missing analytical HBM accounting in {path}")
            if any(
                not isinstance(value, (int, float)) or value < 0
                for value in hbm.values()
            ):
                raise ValueError(f"invalid analytical HBM accounting in {path}")
            expected_hbm = {
                "vector_storage_bytes": int(contract["rows"])
                * int(contract["dim"])
                * 2,
                "graph_state_lower_bound_bytes": int(contract["rows"])
                * (
                    int(contract["intermediate_graph_degree"])
                    + int(contract["intermediate_graph_degree"])
                )
                * 4,
                "candidate_distance_lower_bound_bytes": int(contract["rows"])
                * (
                    int(contract["intermediate_graph_degree"])
                    + int(contract["intermediate_graph_degree"])
                )
                * 4,
                "output_graph_neighbor_bytes": int(contract["rows"])
                * int(contract["graph_degree"])
                * 4,
            }
            if any(hbm.get(field) != expected for field, expected in expected_hbm.items()):
                raise ValueError(f"analytical HBM accounting mismatch in {path}")
            if reference_hbm is None:
                reference_hbm = hbm
            elif hbm != reference_hbm:
                raise ValueError(f"analytical HBM accounting differs across repeats: {path}")
            profiles.append(profile)
            elapsed.append(float(value))
            memory_samples.append(_load_gpu_monitor(case_root / "gpu-monitor.csv"))
        peaks = [sample["peak_mib"] for sample in memory_samples]
        deltas = [sample["peak_delta_mib"] for sample in memory_samples]
        methods[method] = {
            "profiles": profiles,
            "timing": {
                "repeat_count": len(elapsed),
                "median_ms": statistics.median(elapsed),
                "min_ms": min(elapsed),
                "max_ms": max(elapsed),
                "samples_ms": elapsed,
            },
            "analytical_hbm_bytes": profiles[0]["analytical_hbm_bytes"],
            "observed_gpu_memory": {
                "monitor_scope": "whole_device_on_preflight-idle_gpu",
                "samples": memory_samples,
                "median_peak_mib": statistics.median(peaks),
                "max_peak_mib": max(peaks),
                "median_peak_delta_mib": statistics.median(deltas),
                "max_peak_delta_mib": max(deltas),
            },
        }
    assert reference_contract is not None
    return methods, reference_contract


def _load_nsys(
    run_root: Path,
    methods: dict[str, dict],
    contract: dict,
) -> None:
    for method in METHODS:
        case_root = run_root / "nsys" / method
        command = _require_success_evidence(case_root, f"Nsys {method}")
        if f"--method {method}" not in command:
            raise ValueError(f"Nsys command method mismatch: {case_root}")
        export_exit = case_root / "export.exit_code"
        _require_nonempty_file(export_exit, f"Nsys {method} export exit")
        if export_exit.read_text().strip() != "0":
            raise ValueError(f"Nsys export exit status is not zero: {export_exit}")
        trace_path = case_root / "trace.sqlite"
        _require_nonempty_file(trace_path, f"Nsys {method} trace")
        profile_path = case_root / "profile.json"
        profile = _load_json(profile_path, "Nsys build profile")
        if profile.get("method") != method or profile.get("profiled_timing") is not True:
            raise ValueError(f"Nsys build profile mismatch in {profile_path}")
        _validate_profile_contract(profile, contract, profile_path)

        path = case_root / "timeline.json"
        timeline = _load_json(path, "Nsys timeline")
        if timeline.get("method") != method:
            raise ValueError(f"Nsys method mismatch in {path}")
        if Path(timeline.get("source_sqlite", "")).resolve() != trace_path.resolve():
            raise ValueError(f"Nsys timeline source mismatch in {path}")
        try:
            recomputed_timeline = analyze_sqlite(trace_path, method)
        except (ValueError, sqlite3.Error) as error:
            raise ValueError(
                f"cannot reanalyze Nsys trace for {method}: {trace_path}"
            ) from error
        if timeline != recomputed_timeline:
            raise ValueError(
                f"Nsys timeline does not match trace reanalysis for {method}: {path}"
            )
        timeline = recomputed_timeline
        local_wall = timeline.get("local_build_wall_ms")
        if not isinstance(local_wall, (int, float)) or local_wall <= 0:
            raise ValueError(f"invalid Nsys local wall in {path}")
        exclusive = timeline.get("critical_path_exclusive_ms")
        if not isinstance(exclusive, dict):
            raise ValueError(f"missing critical path in {path}")
        missing = [field for field in CRITICAL_PATH_FIELDS if field not in exclusive]
        if missing:
            raise ValueError(f"critical path is missing {missing}: {path}")
        recomputed_closure = abs(
            sum(float(exclusive[field]) for field in CRITICAL_PATH_FIELDS)
            - float(local_wall)
        )
        reported_closure = float(timeline.get("critical_path_closure_error_ms", float("inf")))
        closure = max(recomputed_closure, reported_closure)
        tolerance = max(2.0, float(local_wall) * 0.005)
        if closure > tolerance:
            raise ValueError(
                f"critical path closure {closure:.6f} ms exceeds "
                f"{tolerance:.6f} ms for {method}"
            )
        gpu_active = timeline.get("gpu_kernel_active_ms")
        gpu_path = timeline.get("gpu_path_exclusive_ms")
        if not isinstance(gpu_active, (int, float)) or gpu_active <= 0:
            raise ValueError(f"invalid GPU active time in {path}")
        gpu_elapsed_sum = timeline.get("gpu_kernel_elapsed_sum_ms")
        kernel_elapsed_sum = sum(
            float(row.get("total_time_ms", 0.0))
            for row in timeline.get("kernel_aggregates", [])
        )
        if not isinstance(gpu_elapsed_sum, (int, float)) or gpu_elapsed_sum <= 0:
            raise ValueError(f"invalid summed GPU kernel time in {path}")
        elapsed_tolerance = max(0.001, float(gpu_elapsed_sum) * 1e-6)
        if abs(float(gpu_elapsed_sum) - kernel_elapsed_sum) > elapsed_tolerance:
            raise ValueError(f"summed GPU kernel time mismatch in {path}")
        if not isinstance(gpu_path, dict):
            raise ValueError(f"missing GPU-active path split in {path}")
        missing_gpu = [field for field in GPU_PATH_FIELDS if field not in gpu_path]
        if missing_gpu:
            raise ValueError(f"GPU-active path is missing {missing_gpu}: {path}")
        gpu_closure = max(
            abs(
                sum(float(gpu_path[field]) for field in GPU_PATH_FIELDS)
                - float(gpu_active)
            ),
            float(timeline.get("gpu_path_closure_error_ms", float("inf"))),
        )
        gpu_tolerance = max(0.05, float(gpu_active) * 0.005)
        if gpu_closure > gpu_tolerance:
            raise ValueError(
                f"GPU-active path closure {gpu_closure:.6f} ms exceeds "
                f"{gpu_tolerance:.6f} ms for {method}"
            )
        unclassified = timeline.get("unclassified_gpu_share")
        if not isinstance(unclassified, (int, float)) or not 0 <= unclassified <= 0.05:
            raise ValueError(
                f"unclassified GPU share exceeds 5% or is invalid for {method}: "
                f"{unclassified!r}"
            )
        unclassified_time = timeline.get("unclassified_kernel_time_ms")
        if (
            not isinstance(unclassified_time, (int, float))
            or not math.isfinite(float(unclassified_time))
            or float(unclassified_time) < 0
        ):
            raise ValueError(f"invalid unclassified GPU time in {path}")
        expected_unclassified_share = float(unclassified_time) / float(gpu_active)
        if not math.isclose(
            float(unclassified),
            expected_unclassified_share,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"unclassified GPU share mismatch in {path}")
        shares = {
            field: float(exclusive[field]) / float(local_wall)
            for field in CRITICAL_PATH_FIELDS
        }
        reported_shares = timeline.get("critical_path_share")
        if reported_shares is not None:
            if not isinstance(reported_shares, dict):
                raise ValueError(f"invalid critical path shares in {path}")
            for field, expected in shares.items():
                observed = reported_shares.get(field)
                if (
                    not isinstance(observed, (int, float))
                    or not math.isfinite(float(observed))
                    or not math.isclose(
                        float(observed),
                        expected,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"critical path share mismatch for {field} in {path}"
                    )
        methods[method]["timeline"] = timeline
        methods[method]["critical_path_share"] = shares
        gpu_shares = {
            field: float(gpu_path[field]) / float(gpu_active)
            for field in GPU_PATH_FIELDS
        }
        reported_gpu_shares = timeline.get("gpu_path_exclusive_share")
        if reported_gpu_shares is not None:
            if not isinstance(reported_gpu_shares, dict):
                raise ValueError(f"invalid GPU-active path shares in {path}")
            for field, expected in gpu_shares.items():
                observed = reported_gpu_shares.get(field)
                if (
                    not isinstance(observed, (int, float))
                    or not math.isfinite(float(observed))
                    or not math.isclose(
                        float(observed),
                        expected,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"GPU-active path share mismatch for {field} in {path}"
                    )
        methods[method]["gpu_path_exclusive_share"] = gpu_shares
        if method != "cagra":
            if timeline.get("cfs_logical_stage_split_supported") is not False:
                raise ValueError(
                    f"CFS logical Collect/Filter/Store split must remain unsupported: {path}"
                )
            if (
                timeline.get("cfs_implementation")
                != "collect_filter_fused_plus_reverse_finalize"
            ):
                raise ValueError(f"missing fused CFS implementation label: {path}")
            if timeline.get("cfs_time_basis") != "summed_nsys_kernel_elapsed_time":
                raise ValueError(f"invalid CFS time basis: {path}")
        cfs_primary = timeline.get("cfs_primary_gpu_ms")
        cfs_reverse = timeline.get("cfs_reverse_finalize_gpu_ms")
        cfs_gpu = timeline.get("cfs_gpu_ms")
        for label, value in (
            ("CFS primary time", cfs_primary),
            ("CFS reverse/finalize time", cfs_reverse),
            ("CFS total time", cfs_gpu),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"invalid {label} in {path}")
        expected_cfs_gpu = float(cfs_primary) + float(cfs_reverse)
        cfs_tolerance = max(0.001, expected_cfs_gpu * 1e-6)
        if abs(float(cfs_gpu) - expected_cfs_gpu) > cfs_tolerance:
            raise ValueError(f"CFS total time mismatch in {path}")
        expected_cfs_kernel_share = expected_cfs_gpu / float(gpu_elapsed_sum)
        expected_cfs_wall_share = expected_cfs_gpu / float(local_wall)
        for label, observed, expected in (
            (
                "CFS GPU-kernel share",
                timeline.get("cfs_share_of_gpu_kernel_time"),
                expected_cfs_kernel_share,
            ),
            (
                "CFS local-build share",
                timeline.get("cfs_share_of_local_build_wall"),
                expected_cfs_wall_share,
            ),
        ):
            if (
                not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or not math.isclose(
                    float(observed),
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"{label} mismatch in {path}")
        for field in (
            "prune_api_wall_ms",
            "prune_setup_gpu_ms",
            "cfs_primary_gpu_ms",
            "cfs_reverse_finalize_gpu_ms",
            "cfs_gpu_ms",
            "cfs_time_basis",
            "cfs_share_of_gpu_kernel_time",
            "cfs_share_of_local_build_wall",
            "prune_transfer_ms",
            "prune_host_residual_ms",
            "cfs_logical_stage_split_supported",
            "cfs_implementation",
        ):
            methods[method][field] = timeline.get(field)
        methods[method]["cfs_gpu_ms"] = expected_cfs_gpu
        methods[method][
            "cfs_share_of_gpu_kernel_time"
        ] = expected_cfs_kernel_share
        methods[method]["cfs_share_of_local_build_wall"] = expected_cfs_wall_share


def _load_ncu(
    run_root: Path,
    methods: dict[str, dict],
    contract: dict,
    allow_ncu_incomplete: bool,
) -> str:
    status_path = run_root / "ncu/status.txt"
    if not status_path.is_file():
        raise ValueError(f"missing NCU status evidence: {status_path}")
    status = status_path.read_text().strip()
    if status not in {"NCU_READY", "NCU_INCOMPLETE"}:
        raise ValueError(f"invalid NCU status {status!r}")
    if status == "NCU_INCOMPLETE":
        if not allow_ncu_incomplete:
            raise ValueError("NCU_INCOMPLETE is not allowed for this summary")
        probe_root = run_root / "ncu/probe"
        _require_nonempty_file(probe_root / "command.sh", "NCU probe command")
        _require_nonempty_file(probe_root / "exit_code", "NCU probe exit")
        probe_exit = probe_root.joinpath("exit_code").read_text().strip()
        failure_roots = []
        if probe_exit != "0":
            failure_roots.append(probe_root)
        else:
            for exit_path in (run_root / "ncu").glob(
                "*/**/captures/*/exit_code"
            ):
                if exit_path.read_text().strip() != "0":
                    _require_nonempty_file(
                        exit_path.parent / "command.sh",
                        "failed NCU capture command",
                    )
                    failure_roots.append(exit_path.parent)
        permission_failure = False
        for failure_root in failure_roots:
            logs = ""
            for name in ("stdout.log", "stderr.log"):
                log_path = failure_root / name
                if log_path.is_file():
                    logs += log_path.read_text(errors="replace")
            permission_failure |= "ERR_NVGPUCTRPERM" in logs
        if not permission_failure:
            raise ValueError("NCU_INCOMPLETE lacks ERR_NVGPUCTRPERM evidence")
        for method in METHODS:
            methods[method]["ncu"] = {}
        return status

    for method in METHODS:
        captures = {}
        for kernel_class in EXPECTED_NCU_CLASSES[method]:
            path = run_root / "ncu" / method / kernel_class / "metrics.json"
            report = _load_json(path, "NCU")
            if report.get("method") != method or report.get("kernel_class") != kernel_class:
                raise ValueError(f"NCU label mismatch in {path}")
            metrics = report.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"NCU metrics missing in {path}")
            for field in NCU_FIELDS:
                metrics.setdefault(field, None)
            per_kernel = report.get("per_kernel")
            if not isinstance(per_kernel, dict) or not per_kernel:
                raise ValueError(f"NCU per-kernel signatures missing in {path}")
            for kernel_name, kernel_report in per_kernel.items():
                if not kernel_name or not isinstance(kernel_report, dict):
                    raise ValueError(f"invalid NCU per-kernel signature in {path}")
                if not isinstance(kernel_report.get("counter_signature"), dict):
                    raise ValueError(
                        f"missing NCU counter signature for {kernel_name!r} in {path}"
                    )
            source_csv = report.get("source_csv")
            if not isinstance(source_csv, list) or not source_csv:
                raise ValueError(f"NCU source CSV evidence missing in {path}")
            capture_root = run_root / "ncu" / method / kernel_class / "captures"
            source_paths = []
            for source in source_csv:
                source_path = Path(source)
                try:
                    source_path.resolve().relative_to(capture_root.resolve())
                except ValueError as error:
                    raise ValueError(f"NCU source CSV is outside capture root: {source}") from error
                _require_nonempty_file(source_path, "NCU source CSV")
                case_root = source_path.parent
                command = _require_success_evidence(
                    case_root, f"NCU {method}/{kernel_class}"
                )
                if f"--method {method}" not in command:
                    raise ValueError(f"NCU capture command method mismatch: {case_root}")
                profile_path = case_root / "profile.json"
                profile = _load_json(profile_path, "NCU build profile")
                if (
                    profile.get("method") != method
                    or profile.get("profiled_timing") is not True
                ):
                    raise ValueError(f"NCU build profile mismatch in {profile_path}")
                _validate_profile_contract(profile, contract, profile_path)
                source_paths.append(source_path)
            try:
                recomputed_report = aggregate_ncu(
                    [
                        row
                        for source_path in source_paths
                        for row in read_ncu_csv(source_path)
                    ],
                    method=method,
                    kernel_class=kernel_class,
                )
            except ValueError as error:
                raise ValueError(
                    f"cannot reanalyze NCU CSV for {method}/{kernel_class}"
                ) from error
            recomputed_report["source_csv"] = [
                str(source_path.resolve()) for source_path in source_paths
            ]
            if report != recomputed_report:
                raise ValueError(
                    f"NCU report does not match raw CSV reanalysis: {path}"
                )
            nsys_kernels = methods[method]["timeline"].get(
                "kernel_aggregates",
                [],
            )
            for kernel_name in recomputed_report["per_kernel"]:
                observed_class = semantic_kernel_class(kernel_name, method)
                if observed_class != kernel_class:
                    raise ValueError(
                        f"NCU kernel semantic class mismatch for {kernel_name!r}: "
                        f"{observed_class!r} != {kernel_class!r}"
                    )
                matching_nsys = [
                    kernel
                    for kernel in nsys_kernels
                    if _kernel_name_matches(
                        str(kernel.get("kernel_name", "")),
                        kernel_name,
                    )
                ]
                if not matching_nsys:
                    raise ValueError(
                        f"NCU kernel has no matching Nsys aggregate: {kernel_name!r}"
                    )
                if any(
                    kernel.get("semantic_class") != kernel_class
                    for kernel in matching_nsys
                ):
                    raise ValueError(
                        f"NCU/Nsys semantic class mismatch for {kernel_name!r}"
                    )
            captures[kernel_class] = recomputed_report
        methods[method]["ncu"] = captures
    return status


def _recall_field(fieldnames: list[str]) -> str | None:
    normalized = {
        name.lower().replace("@", "_at_").replace("-", "_").replace(" ", "_"): name
        for name in fieldnames
    }
    for candidate in (
        "recall_at_10",
        "recall_at_10_mean",
        "mean_recall_at_10",
        "recall",
    ):
        if candidate in normalized:
            return normalized[candidate]
    return None


def _read_fbin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"short fbin header: {path}")
    rows, dim = struct.unpack("<ii", header)
    if rows <= 0 or dim <= 0 or path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"invalid fbin shape or byte length: {path}")
    return rows, dim


def _load_quality_contract(run_root: Path, construction_contract: dict) -> dict:
    path = run_root / "quality/common/quality_contract.json"
    contract = _load_json(path, "quality contract")
    if contract.get("schema") != "wiki10m-local-shard-quality-contract-v1":
        raise ValueError(f"invalid quality contract schema: {path}")
    if contract.get("shard_sha256") != construction_contract["shard_sha256"]:
        raise ValueError("quality contract shard does not match construction")
    if contract.get("metric") != construction_contract["metric"]:
        raise ValueError("quality contract metric does not match construction")

    bindings = (
        ("construction_manifest", "construction_manifest_sha256"),
        ("shard_fbin", "shard_sha256"),
        ("query_fbin", "query_sha256"),
        ("query_manifest", "query_manifest_sha256"),
        ("groundtruth", "groundtruth_sha256"),
    )
    resolved = {}
    for path_field, hash_field in bindings:
        value = contract.get(path_field)
        expected_hash = contract.get(hash_field)
        if not isinstance(value, str) or not isinstance(expected_hash, str):
            raise ValueError(f"quality contract lacks {path_field}/{hash_field}")
        artifact = Path(value)
        _require_nonempty_file(artifact, f"quality {path_field}")
        if _sha256_file(artifact) != expected_hash:
            raise ValueError(f"quality {path_field} SHA-256 mismatch")
        resolved[path_field] = artifact

    shard_rows, shard_dim = _read_fbin_shape(resolved["shard_fbin"])
    query_rows, query_dim = _read_fbin_shape(resolved["query_fbin"])
    if (shard_rows, shard_dim) != (
        contract.get("shard_rows"),
        contract.get("shard_dim"),
    ):
        raise ValueError("quality contract shard shape mismatch")
    if (query_rows, query_dim) != (
        contract.get("query_rows"),
        contract.get("query_dim"),
    ):
        raise ValueError("quality contract query shape mismatch")
    if query_dim != shard_dim:
        raise ValueError("quality query dimension does not match shard")

    query_manifest = _load_json(resolved["query_manifest"], "query manifest")
    query_source_kind = query_manifest.get("source_kind", "shard_subset")
    if contract.get("query_source_kind", query_source_kind) != query_source_kind:
        raise ValueError("query source kind disagrees with quality contract")
    if query_source_kind == "shard_subset":
        if Path(query_manifest.get("source_fbin", "")).resolve() != resolved[
            "shard_fbin"
        ].resolve():
            raise ValueError("query manifest source does not match quality shard")
    elif query_source_kind == "external_query_copy":
        origin = Path(query_manifest.get("source_query_fbin", ""))
        _require_nonempty_file(origin, "canonical query origin")
        origin_sha256 = _sha256_file(origin)
        if (
            query_manifest.get("source_query_sha256") != origin_sha256
            or contract.get("query_origin_sha256") != origin_sha256
            or Path(contract.get("query_origin", "")).resolve() != origin.resolve()
        ):
            raise ValueError("canonical query origin SHA-256 mismatch")
    else:
        raise ValueError(f"unsupported quality query source: {query_source_kind}")
    if Path(query_manifest.get("output_fbin", "")).resolve() != resolved[
        "query_fbin"
    ].resolve():
        raise ValueError("query manifest output does not match quality query")
    if query_manifest.get("query_sha256") != contract.get("query_sha256"):
        raise ValueError("query manifest SHA-256 disagrees with quality contract")

    with resolved["groundtruth"].open("rb") as stream:
        gt_header = stream.read(8)
    if len(gt_header) != 8:
        raise ValueError("short quality groundtruth header")
    gt_rows, gt_k = struct.unpack("<II", gt_header)
    if (gt_rows, gt_k) != (
        contract.get("groundtruth_rows"),
        contract.get("groundtruth_k"),
    ):
        raise ValueError("quality groundtruth header mismatch")
    if gt_rows != query_rows or gt_k != contract.get("recall_at"):
        raise ValueError("quality groundtruth does not match query/Recall@K")
    if resolved["groundtruth"].stat().st_size != 8 + gt_rows * gt_k * 8:
        raise ValueError("quality groundtruth byte length mismatch")

    common = path.parent
    command_path = common / "groundtruth.command.sh"
    exit_path = common / "groundtruth.exit_code"
    _require_nonempty_file(command_path, "groundtruth command")
    _require_nonempty_file(exit_path, "groundtruth exit")
    if exit_path.read_text().strip() != "0":
        raise ValueError("groundtruth generation exit status is not zero")
    command = command_path.read_text()
    if (
        f"--base_file {resolved['shard_fbin']}" not in command
        or f"--query_file {resolved['query_fbin']}" not in command
        or f"--gt_file {resolved['groundtruth']}" not in command
        or "--dist_fn l2" not in command
        or "--K 10" not in command
        or "--data_type float" not in command
    ):
        raise ValueError("groundtruth command is not bound to quality inputs")
    contract["quality_contract_sha256"] = _sha256_file(path)
    return contract


def _load_selected_graph(
    root: Path,
    validation: dict,
    profile: dict,
    expected_rows: int,
) -> Path:
    command_path = root / "graph_validation.command.sh"
    exit_path = root / "graph_validation.exit_code"
    _require_nonempty_file(command_path, "graph validation command")
    _require_nonempty_file(exit_path, "graph validation exit")
    if exit_path.read_text().strip() != "0":
        raise ValueError(f"graph validation exit status is not zero: {exit_path}")
    graph_value = validation.get("path")
    if not isinstance(graph_value, str):
        raise ValueError(f"graph validation lacks path: {root}")
    graph_path = Path(graph_value)
    profile_graph = profile.get("output_graph")
    if (
        not isinstance(profile_graph, str)
        or graph_path.resolve() != Path(profile_graph).resolve()
    ):
        raise ValueError(
            "graph validation is not bound to the selected timing output graph"
        )
    _require_nonempty_file(graph_path, "selected timing graph")
    expected_sha256 = validation.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or _sha256_file(graph_path) != expected_sha256
        or validation.get("bytes") != graph_path.stat().st_size
    ):
        raise ValueError("graph validation path/SHA/size mismatch")
    command = command_path.read_text()
    if (
        f"--graph {graph_path.resolve()}" not in command
        or f"--rows {expected_rows}" not in command
        or "--expected-max-degree 64" not in command
    ):
        raise ValueError("graph validation command is not bound to selected graph")
    return graph_path


def _load_graph_quality(
    root: Path,
    validation: dict,
    expected_rows: int,
    graph_path: Path,
    shard_path: Path,
) -> dict:
    path = root / "graph_quality.tsv"
    _require_nonempty_file(path, "graph quality")
    command_path = root / "graph_quality.command.sh"
    exit_path = root / "graph_quality.exit_code"
    _require_nonempty_file(command_path, "graph quality command")
    _require_nonempty_file(exit_path, "graph quality exit")
    if exit_path.read_text().strip() != "0":
        raise ValueError(f"graph quality exit status is not zero: {exit_path}")
    command = command_path.read_text()
    if (
        f"--index-path-prefix {graph_path.parent.resolve()}" not in command
        or f"--data-path {shard_path.resolve()}" not in command
    ):
        raise ValueError(f"graph quality command/input mismatch: {command_path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"graph quality must contain exactly one row: {path}")
    row = rows[0]
    required = (
        "nodes",
        "edges",
        "self_loop_edges",
        "start",
        "max_out_degree",
        "directed_reachable_from_start",
        "directed_reachable_from_start_ratio",
    )
    if any(field not in row for field in required):
        raise ValueError(f"graph quality lacks required fields: {path}")
    try:
        parsed = {
            "nodes": int(row["nodes"]),
            "edges": int(row["edges"]),
            "self_loop_edges": int(row["self_loop_edges"]),
            "start": int(row["start"]),
            "max_out_degree": int(row["max_out_degree"]),
            "directed_reachable_from_start": int(
                row["directed_reachable_from_start"]
            ),
            "directed_reachable_from_start_ratio": float(
                row["directed_reachable_from_start_ratio"]
            ),
        }
    except ValueError as error:
        raise ValueError(f"invalid graph quality row: {path}") from error
    if (
        parsed["nodes"] != expected_rows
        or parsed["edges"] <= 0
        or parsed["self_loop_edges"] != 0
        or not 0 <= parsed["start"] < expected_rows
        or not 0 <= parsed["directed_reachable_from_start"] <= expected_rows
        or not 0 <= parsed["directed_reachable_from_start_ratio"] <= 1
    ):
        raise ValueError(f"invalid graph quality invariants: {path}")
    expected_reachable_ratio = (
        parsed["directed_reachable_from_start"] / parsed["nodes"]
    )
    if not math.isclose(
        parsed["directed_reachable_from_start_ratio"],
        expected_reachable_ratio,
        rel_tol=1e-9,
        abs_tol=5.1e-7,
    ):
        raise ValueError(f"graph reachable ratio mismatch: {path}")
    if validation.get("rows") != parsed["nodes"] or validation.get("edges") != parsed["edges"]:
        raise ValueError(f"graph validation/quality mismatch: {path}")
    return {**row, **parsed}


def _load_quality(
    run_root: Path,
    methods: dict[str, dict],
    contract: dict,
    allow_single_repeat: bool,
) -> None:
    expected_l = {25, 50, 100, 200, 400}
    expected_repetitions = {1} if allow_single_repeat else {1, 2, 3}
    query_path = Path(contract["query_fbin"])
    groundtruth_path = Path(contract["groundtruth"])
    shard_path = Path(contract["shard_fbin"])
    for method in METHODS:
        root = run_root / "quality" / method
        validation = _load_json(root / "graph_validation.json", "quality graph validation")
        if validation.get("valid") is not True:
            raise ValueError(f"quality graph validation failed for {method}")
        selected_graph = _load_selected_graph(
            root,
            validation,
            methods[method]["profiles"][0 if allow_single_repeat else 1],
            int(contract["shard_rows"]),
        )
        graph_quality = _load_graph_quality(
            root,
            validation,
            int(contract["shard_rows"]),
            selected_graph,
            shard_path,
        )
        search_path = root / "search_summary.tsv"
        if not search_path.is_file():
            raise ValueError(f"missing quality search evidence: {search_path}")
        with search_path.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"quality search has no header: {search_path}")
            recall_name = _recall_field(reader.fieldnames)
            l_name = next((name for name in reader.fieldnames if name.lower() == "l"), None)
            repetition_name = next(
                (name for name in reader.fieldnames if name.lower() == "repetition"),
                None,
            )
            qps_name = next(
                (name for name in reader.fieldnames if name.lower() == "qps"),
                None,
            )
            if (
                recall_name is None
                or l_name is None
                or repetition_name is None
                or qps_name is None
            ):
                raise ValueError(
                    f"quality search lacks L/repetition/Recall@10/QPS: {search_path}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"quality search is empty: {search_path}")
        parsed = []
        observed_grid = set()
        for row in rows:
            try:
                work = int(float(row[l_name]))
                repetition = int(float(row[repetition_name]))
                recall = float(row[recall_name])
                qps = float(row[qps_name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid quality search row in {search_path}") from error
            if not 0 <= recall <= 1:
                raise ValueError(f"invalid Recall@10 in {search_path}: {recall}")
            pair = (work, repetition)
            if pair in observed_grid:
                raise ValueError(f"duplicate quality grid point {pair}: {search_path}")
            observed_grid.add(pair)
            parsed.append(
                {
                    **row,
                    "L": work,
                    "repetition": repetition,
                    "recall_at_10": recall,
                    "qps": qps,
                }
            )
        expected_grid = {
            (work, repetition)
            for work in expected_l
            for repetition in expected_repetitions
        }
        if observed_grid != expected_grid:
            raise ValueError(
                f"quality L×repeat grid mismatch in {search_path}: "
                f"{sorted(observed_grid)} != {sorted(expected_grid)}"
            )
        for row in parsed:
            work, repetition = row["L"], row["repetition"]
            prefix = root / f"L{work}.repeat_{repetition}"
            exit_path = Path(f"{prefix}.exit_code")
            command_path = Path(f"{prefix}.command.sh")
            metrics_path = Path(f"{prefix}.metrics.tsv")
            _require_nonempty_file(exit_path, "quality search exit")
            _require_nonempty_file(command_path, "quality search command")
            _require_nonempty_file(metrics_path, "quality search metrics")
            if exit_path.read_text().strip() != "0":
                raise ValueError(f"quality search exit status is not zero: {exit_path}")
            command = command_path.read_text()
            if (
                f"--index-path-prefix {selected_graph.parent.resolve()}"
                not in command
                or f"--data-path {shard_path.resolve()}" not in command
                or f"--query-file {query_path}" not in command
                or f"--gt-file {groundtruth_path}" not in command
                or f"--search-list-size {work}" not in command
            ):
                raise ValueError(f"quality search command/input mismatch: {command_path}")
            with metrics_path.open(newline="") as stream:
                metric_rows = list(csv.DictReader(stream, delimiter="\t"))
            if len(metric_rows) != 1:
                raise ValueError(f"quality metrics must have one row: {metrics_path}")
            metric_row = metric_rows[0]
            try:
                metric_queries = int(metric_row["queries"])
                metric_recall_at = int(metric_row["recall_at"])
                metric_l = int(metric_row["search_list_size"])
                metric_recall = float(metric_row["recall"])
                metric_qps = float(metric_row["qps"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid quality metrics: {metrics_path}") from error
            if (
                metric_queries != contract["query_rows"]
                or metric_recall_at != contract["recall_at"]
                or metric_l != work
                or abs(metric_recall - row["recall_at_10"]) > 1e-9
                or abs(metric_qps - row["qps"]) > 1e-6
            ):
                raise ValueError(f"quality summary/metrics mismatch: {metrics_path}")
        max_l = max(row["L"] for row in parsed)
        canonical = statistics.median(
            row["recall_at_10"] for row in parsed if row["L"] == max_l
        )
        methods[method]["quality"] = {
            "graph_validation": validation,
            "graph_quality": graph_quality,
            "search_summary": parsed,
            "canonical_L": max_l,
            "recall_at_10": canonical,
        }


def _resource_signature_supported(methods: dict[str, dict], ncu_status: str) -> bool:
    if ncu_status != "NCU_READY":
        return False
    cfs_classes = {
        "tagore_cagra": "irregular_cfs",
        "tagore_vamana": "mixed_cfs",
    }
    for method, cfs_class in cfs_classes.items():
        dense_reports = methods[method]["ncu"]["dense"]["per_kernel"].values()
        graph_reports = methods[method]["ncu"][cfs_class]["per_kernel"].values()
        dense_signatures = [
            report.get("counter_signature", {}) for report in dense_reports
        ]
        graph_signatures = [
            report.get("counter_signature", {}) for report in graph_reports
        ]
        if not dense_signatures or not graph_signatures:
            return False
        if not all(
            signature.get("classification")
            in {"compute_bound", "mixed_compute_memory_latency"}
            and signature.get("compute_signal") is True
            for signature in dense_signatures
        ):
            return False
        if not all(
            signature.get("classification")
            in {"memory_latency_bound", "mixed_compute_memory_latency"}
            and signature.get("memory_latency_signal") is True
            for signature in graph_signatures
        ):
            return False
    return True


def _kernel_name_matches(nsys_name: str, ncu_name: str) -> bool:
    left = " ".join(str(nsys_name).lower().split())
    right = " ".join(str(ncu_name).lower().split())
    if not left or not right:
        return False
    return left == right or (len(left) >= 6 and left in right) or (
        len(right) >= 6 and right in left
    )


def _capture_has_definitive_signature(capture: dict) -> bool:
    return capture.get("counter_signature", {}).get("classification") in {
        "compute_bound",
        "memory_latency_bound",
        "mixed_compute_memory_latency",
    }


def _capture_is_memory_latency_bound(capture: dict) -> bool:
    return (
        capture.get("counter_signature", {}).get("classification")
        == "memory_latency_bound"
    )


def _apply_ncu_coverage(methods: dict[str, dict], ncu_status: str) -> str:
    statuses = []
    for method in METHODS:
        timeline = methods[method]["timeline"]
        denominator = float(timeline.get("gpu_kernel_elapsed_sum_ms", 0.0))
        classified_time = 0.0
        memory_time = 0.0
        if ncu_status == "NCU_READY" and denominator > 0:
            per_kernel_captures = [
                (ncu_name, kernel_report)
                for capture in methods[method]["ncu"].values()
                for ncu_name, kernel_report in capture["per_kernel"].items()
            ]
            for kernel in timeline.get("kernel_aggregates", []):
                kernel_name = str(kernel.get("kernel_name", ""))
                kernel_time = float(kernel.get("total_time_ms", 0.0))
                matches = [
                    kernel_report
                    for ncu_name, kernel_report in per_kernel_captures
                    if _kernel_name_matches(kernel_name, ncu_name)
                ]
                if len(matches) > 1:
                    raise ValueError(
                        f"ambiguous NCU per-kernel match for {method}: {kernel_name}"
                    )
                matched_capture = matches[0] if matches else None
                if matched_capture is None or not _capture_has_definitive_signature(
                    matched_capture
                ):
                    continue
                classified_time += kernel_time
                if _capture_is_memory_latency_bound(matched_capture):
                    memory_time += kernel_time
        coverage = (
            0.0
            if denominator <= 0
            else classified_time / denominator
        )
        status = (
            "SUPPORTED"
            if ncu_status == "NCU_READY" and coverage >= NCU_COVERAGE_THRESHOLD
            else "INCOMPLETE"
        )
        methods[method]["ncu_classified_nsys_time_ms"] = classified_time
        methods[method][
            "ncu_coverage_denominator_kind"
        ] = "summed_nsys_kernel_elapsed_time"
        methods[method]["ncu_coverage_denominator_ms"] = denominator
        methods[method][
            "ncu_coverage_of_gpu_kernel_elapsed_time"
        ] = coverage
        # Retained for the requested report contract; the denominator kind above
        # makes clear that this is additive kernel-time coverage, not interval union.
        methods[method]["ncu_coverage_of_gpu_active_time"] = coverage
        methods[method]["memory_bound_nsys_time_ms"] = memory_time
        methods[method]["memory_bound_execution_share"] = (
            memory_time / denominator if status == "SUPPORTED" else None
        )
        methods[method]["memory_bound_share_status"] = status
        statuses.append(status)
    global_status = (
        "SUPPORTED"
        if all(status == "SUPPORTED" for status in statuses)
        else "INCOMPLETE"
    )
    if global_status == "INCOMPLETE":
        for method in METHODS:
            methods[method]["memory_bound_execution_share"] = None
            methods[method]["memory_bound_share_status"] = "INCOMPLETE"
    return global_status


def _shared_budget_supported(methods: dict[str, dict], ncu_status: str) -> tuple[bool, dict]:
    reference_dense = methods["cagra"]["gpu_path_exclusive_share"]["dense_gpu"]
    path_checks = {}
    for method in ("tagore_cagra", "tagore_vamana"):
        shares = methods[method]["gpu_path_exclusive_share"]
        path_checks[method] = {
            "graph_state_share_positive": (
                shares["irregular_graph_state_gpu"] + shares["mixed_gpu"] > 0
            ),
            "dense_share_below_cagra": shares["dense_gpu"] < reference_dense,
        }
    resource_signature = _resource_signature_supported(methods, ncu_status)
    supported = resource_signature and all(
        all(checks.values()) for checks in path_checks.values()
    )
    return supported, {
        "cagra_dense_gpu_active_share": reference_dense,
        "path_checks": path_checks,
        "resource_signature_supported": resource_signature,
    }


def _contention_supported(run_root: Path, methods: dict[str, dict]) -> tuple[bool, dict]:
    overlap_by_method = {}
    for method in METHODS:
        overlap = methods[method]["timeline"].get("overlap_ms", {})
        overlap_by_method[method] = (
            float(overlap.get("dense_irregular", 0.0))
            + float(overlap.get("dense_mixed", 0.0))
        )
    return False, {
        "status": "UNSUPPORTED_NO_BOUND_DENSE_REPLAY",
        "dense_graph_state_overlap_ms_by_method": overlap_by_method,
        "normalized_dense_slowdown_ratio": None,
        "slowdown_evidence_valid": False,
        "reason": (
            "the current runner has no schema-bound standalone/overlap dense "
            "replay tied to method, kernel, trace, and construction contract"
        ),
    }


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(run_root: Path, summary: dict) -> None:
    report_root = run_root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    timing_rows = []
    critical_rows = []
    ncu_rows = []
    hbm_rows = []
    for method in METHODS:
        data = summary["methods"][method]
        timing_rows.append(
            {
                "method": method,
                "repeats": data["timing"]["repeat_count"],
                "median_ms": data["timing"]["median_ms"],
                "min_ms": data["timing"]["min_ms"],
                "max_ms": data["timing"]["max_ms"],
                "median_peak_gpu_memory_mib": data["observed_gpu_memory"][
                    "median_peak_mib"
                ],
                "max_peak_gpu_memory_mib": data["observed_gpu_memory"][
                    "max_peak_mib"
                ],
                "median_peak_gpu_memory_delta_mib": data[
                    "observed_gpu_memory"
                ]["median_peak_delta_mib"],
                "max_peak_gpu_memory_delta_mib": data["observed_gpu_memory"][
                    "max_peak_delta_mib"
                ],
            }
        )
        critical_rows.append(
            {
                "method": method,
                "local_build_wall_ms": data["timeline"]["local_build_wall_ms"],
                "gpu_kernel_active_ms": data["timeline"]["gpu_kernel_active_ms"],
                "gpu_kernel_elapsed_sum_ms": data["timeline"][
                    "gpu_kernel_elapsed_sum_ms"
                ],
                "gpu_duty": data["timeline"].get("gpu_duty"),
                **{
                    f"{field}_share": data["critical_path_share"][field]
                    for field in CRITICAL_PATH_FIELDS
                },
                **{
                    f"gpu_active_{field}_share": data[
                        "gpu_path_exclusive_share"
                    ][field]
                    for field in GPU_PATH_FIELDS
                },
                "unclassified_gpu_share": data["timeline"]["unclassified_gpu_share"],
                "prune_api_wall_ms": data["prune_api_wall_ms"],
                "prune_setup_gpu_ms": data["prune_setup_gpu_ms"],
                "cfs_primary_gpu_ms": data["cfs_primary_gpu_ms"],
                "cfs_reverse_finalize_gpu_ms": data[
                    "cfs_reverse_finalize_gpu_ms"
                ],
                "cfs_gpu_ms": data["cfs_gpu_ms"],
                "cfs_time_basis": data["cfs_time_basis"],
                "cfs_share_of_gpu_kernel_time": data[
                    "cfs_share_of_gpu_kernel_time"
                ],
                "cfs_share_of_local_build_wall": data[
                    "cfs_share_of_local_build_wall"
                ],
                "prune_transfer_ms": data["prune_transfer_ms"],
                "prune_host_residual_ms": data["prune_host_residual_ms"],
                "cfs_logical_stage_split_supported": data[
                    "cfs_logical_stage_split_supported"
                ],
                "cfs_implementation": data["cfs_implementation"],
            }
        )
        for kernel_class, capture in data["ncu"].items():
            for kernel_name, kernel_report in capture["per_kernel"].items():
                ncu_rows.append(
                    {
                        "method": method,
                        "kernel_class": kernel_class,
                        "kernel_name": kernel_name,
                        "aggregation_scope": "per_kernel",
                        "launch_count": kernel_report.get("launch_count"),
                        "total_duration_ms": kernel_report.get("total_duration_ms"),
                        **{
                            field: kernel_report["metrics"].get(field)
                            for field in NCU_FIELDS
                        },
                        "counter_signature": kernel_report.get(
                            "counter_signature", {}
                        ).get("classification"),
                        "ncu_classified_nsys_time_ms": data[
                            "ncu_classified_nsys_time_ms"
                        ],
                        "ncu_coverage_denominator_kind": data[
                            "ncu_coverage_denominator_kind"
                        ],
                        "ncu_coverage_denominator_ms": data[
                            "ncu_coverage_denominator_ms"
                        ],
                        "ncu_coverage_of_gpu_kernel_elapsed_time": data[
                            "ncu_coverage_of_gpu_kernel_elapsed_time"
                        ],
                        "ncu_coverage_of_gpu_active_time": data[
                            "ncu_coverage_of_gpu_active_time"
                        ],
                        "memory_bound_nsys_time_ms": data[
                            "memory_bound_nsys_time_ms"
                        ],
                        "memory_bound_execution_share": data[
                            "memory_bound_execution_share"
                        ],
                        "memory_bound_share_status": data[
                            "memory_bound_share_status"
                        ],
                        "missing_fields": ",".join(
                            kernel_report.get("missing_fields", [])
                        ),
                    }
                )
        hbm_rows.append(
            {
                "method": method,
                "measurement_scope": data["observed_gpu_memory"]["monitor_scope"],
                "median_observed_peak_mib": data["observed_gpu_memory"][
                    "median_peak_mib"
                ],
                "max_observed_peak_mib": data["observed_gpu_memory"]["max_peak_mib"],
                "median_observed_peak_delta_mib": data["observed_gpu_memory"][
                    "median_peak_delta_mib"
                ],
                "max_observed_peak_delta_mib": data["observed_gpu_memory"][
                    "max_peak_delta_mib"
                ],
                **data["analytical_hbm_bytes"],
            }
        )

    _write_tsv(
        report_root / "timing.tsv",
        list(timing_rows[0]),
        timing_rows,
    )
    _write_tsv(
        report_root / "critical_path.tsv",
        list(critical_rows[0]),
        critical_rows,
    )
    ncu_fields = [
        "method",
        "kernel_class",
        "kernel_name",
        "aggregation_scope",
        "launch_count",
        "total_duration_ms",
        *NCU_FIELDS,
        "counter_signature",
        "ncu_classified_nsys_time_ms",
        "ncu_coverage_denominator_kind",
        "ncu_coverage_denominator_ms",
        "ncu_coverage_of_gpu_kernel_elapsed_time",
        "ncu_coverage_of_gpu_active_time",
        "memory_bound_nsys_time_ms",
        "memory_bound_execution_share",
        "memory_bound_share_status",
        "missing_fields",
    ]
    _write_tsv(report_root / "ncu_metrics.tsv", ncu_fields, ncu_rows)
    _write_tsv(report_root / "hbm.tsv", list(hbm_rows[0]), hbm_rows)

    conclusion = {
        "SUPPORTED": "SUPPORTED: graph-state work shares the device budget and lowers dense duty.",
        "NOT_SUPPORTED": "NOT SUPPORTED: the required critical-path/resource pattern was not observed.",
        "INCOMPLETE": "INCOMPLETE: NCU performance counters are unavailable or incomplete.",
    }[summary["conclusion_status"]]
    markdown = [
        "# Wiki10M local-shard GPU path profile",
        "",
        conclusion,
        "",
        f"- Acceptance: {summary['acceptance_status']}",
        f"- NCU status: {summary['ncu_status']}",
        f"- Shared-device-budget gate: {summary['shared_device_budget_supported']}",
        f"- Stronger overlap-and-slowdown gate: {summary['contention_claim_supported']}",
        f"- Memory-bound share status: {summary['memory_bound_share_status']}",
        "",
        "## Fixed-work timing and quality",
        "",
        "| Method | Median ms [min, max] | Recall@10 | Peak GPU MiB | Peak delta MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        data = summary["methods"][method]
        markdown.append(
            f"| {method} | {data['timing']['median_ms']:.3f} "
            f"[{data['timing']['min_ms']:.3f}, {data['timing']['max_ms']:.3f}] | "
            f"{data['quality']['recall_at_10']:.6f} | "
            f"{data['observed_gpu_memory']['max_peak_mib']:.3f} | "
            f"{data['observed_gpu_memory']['max_peak_delta_mib']:.3f} |"
        )
    markdown.extend(
        [
            "",
            "## GPU-active class split",
            "",
            "| Method | GPU active ms | Dense | Irregular graph-state | Mixed | Other | Unclassified |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        data = summary["methods"][method]
        shares = data["gpu_path_exclusive_share"]
        markdown.append(
            f"| {method} | {data['timeline']['gpu_kernel_active_ms']:.3f} | "
            f"{shares['dense_gpu']:.6f} | "
            f"{shares['irregular_graph_state_gpu']:.6f} | "
            f"{shares['mixed_gpu']:.6f} | "
            f"{shares['other_gpu']:.6f} | "
            f"{data['timeline']['unclassified_gpu_share']:.6f} |"
        )
    markdown.extend(
        [
            "",
            "## CFS and NCU coverage",
            "",
            "| Method | CFS primary ms | CFS reverse/finalize ms | CFS/GPU | "
            "NCU kernel-time coverage | Memory-bound status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for method in METHODS:
        data = summary["methods"][method]
        primary = data["cfs_primary_gpu_ms"] or 0.0
        reverse = data["cfs_reverse_finalize_gpu_ms"] or 0.0
        cfs_share = data["cfs_share_of_gpu_kernel_time"] or 0.0
        markdown.append(
            f"| {method} | {primary:.3f} | {reverse:.3f} | {cfs_share:.6f} | "
            f"{data['ncu_coverage_of_gpu_kernel_elapsed_time']:.6f} | "
            f"{data['memory_bound_share_status']} |"
        )
    markdown.extend(
        [
            "",
            "## Per-kernel NCU signatures",
            "",
            "| Method | Class | Kernel | Signature | SM % | DRAM % | Long scoreboard % |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    if ncu_rows:
        for row in ncu_rows:
            markdown.append(
                f"| {row['method']} | {row['kernel_class']} | "
                f"`{row['kernel_name']}` | {row['counter_signature']} | "
                f"{row['sm_pct'] if row['sm_pct'] is not None else 'NA'} | "
                f"{row['dram_pct'] if row['dram_pct'] is not None else 'NA'} | "
                f"{row['long_scoreboard_pct'] if row['long_scoreboard_pct'] is not None else 'NA'} |"
            )
    else:
        markdown.append("| all | — | — | NCU_INCOMPLETE | — | — | — |")
    markdown.extend(
        [
            "",
            "This is one-shard mechanism evidence. Timing is fixed-work; it is not a matched-quality speed comparison.",
            (
                "NCU coverage and memory-bound shares use summed Nsys kernel "
                "elapsed time as both numerator and denominator; they are not "
                "GPU-active interval-union shares."
            ),
            (
                "CFS is reported as fused primary and reverse/finalize kernels; "
                "a logical Collect/Filter/Store stage split is unsupported."
            ),
            (
                "A memory-bound execution share may be reported only when "
                "`memory_bound_share_status=SUPPORTED`; otherwise use the "
                "irregular/mixed graph-state shares."
            ),
            "",
        ]
    )
    (report_root / "report.md").write_text("\n".join(markdown))
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def summarize_run(
    run_root: Path,
    allow_ncu_incomplete: bool = True,
    allow_single_repeat: bool = False,
) -> dict:
    run_root = Path(run_root)
    _validate_provenance(run_root)
    methods, contract = _load_timing(run_root, allow_single_repeat)
    _load_nsys(run_root, methods, contract)
    ncu_status = _load_ncu(run_root, methods, contract, allow_ncu_incomplete)
    memory_bound_share_status = _apply_ncu_coverage(methods, ncu_status)
    quality_contract = _load_quality_contract(run_root, contract)
    _load_quality(run_root, methods, quality_contract, allow_single_repeat)
    shared_supported, shared_evidence = _shared_budget_supported(methods, ncu_status)
    contention_supported, contention_evidence = _contention_supported(run_root, methods)
    conclusion_status = (
        "INCOMPLETE"
        if ncu_status == "NCU_INCOMPLETE"
        else ("SUPPORTED" if shared_supported else "NOT_SUPPORTED")
    )
    summary = {
        "schema": "wiki10m-local-shard-summary-v1",
        "run_root": str(run_root.resolve()),
        "acceptance_status": "ACCEPTED",
        "ncu_status": ncu_status,
        "ncu_coverage_threshold": NCU_COVERAGE_THRESHOLD,
        "memory_bound_share_status": memory_bound_share_status,
        "construction_contract": contract,
        "quality_contract": quality_contract,
        "methods": methods,
        "shared_device_budget_supported": shared_supported,
        "shared_device_budget_evidence": shared_evidence,
        "contention_claim_supported": contention_supported,
        "contention_evidence": contention_evidence,
        "conclusion_status": conclusion_status,
    }
    _write_reports(run_root, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-ncu-incomplete", action="store_true")
    parser.add_argument("--allow-single-repeat", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = summarize_run(
        args.run_root,
        allow_ncu_incomplete=args.allow_ncu_incomplete,
        allow_single_repeat=args.allow_single_repeat,
    )
    print(args.run_root / "report/summary.json")
    return 0 if summary["acceptance_status"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
