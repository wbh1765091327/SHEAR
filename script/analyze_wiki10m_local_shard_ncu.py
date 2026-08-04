#!/usr/bin/env python3
import argparse
import csv
import io
import json
import math
import re
from pathlib import Path


DURATION_METRIC = "gpu__time_duration.sum"
STABLE_METRICS = {
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "barrier_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "atomic_instructions": "smsp__sass_inst_executed_op_global_atom.sum",
}
PERCENTAGE_FIELDS = {
    "sm_pct",
    "tensor_pct",
    "dram_pct",
    "l2_pct",
    "occupancy_pct",
    "long_scoreboard_pct",
    "barrier_pct",
}
ADDITIVE_FIELDS = {"atomic_instructions"}
MISSING_VALUES = {"", "n/a", "nan", "not available", "--"}


def parse_metric_value(value: str, unit: str) -> float | None:
    value = str(value).strip()
    if value.lower() in MISSING_VALUES:
        return None
    cleaned = value.replace(",", "").replace("%", "").strip()
    cleaned = re.sub(r"^[~≈<>]+\s*", "", cleaned)
    multiplier = 1.0
    if cleaned and cleaned[-1].upper() in {"K", "M", "G", "T"}:
        multiplier = {
            "K": 1e3,
            "M": 1e6,
            "G": 1e9,
            "T": 1e12,
        }[cleaned[-1].upper()]
        cleaned = cleaned[:-1]
    try:
        number = float(cleaned) * multiplier
    except ValueError as error:
        raise ValueError(f"cannot parse NCU metric value {value!r} ({unit!r})") from error

    normalized_unit = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    if normalized_unit in {"nsecond", "nanosecond", "nanoseconds", "ns"}:
        return number / 1_000_000.0
    if normalized_unit in {"usecond", "microsecond", "microseconds", "us"}:
        return number / 1_000.0
    if normalized_unit in {"msecond", "millisecond", "milliseconds", "ms"}:
        return number
    if normalized_unit in {"second", "seconds", "s"}:
        return number * 1_000.0
    return number


def read_ncu_csv(path: Path) -> list[dict]:
    lines = Path(path).read_text(errors="replace").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            parsed = next(csv.reader([line]))
            if "Metric Name" in parsed and "Metric Value" in parsed:
                header_index = index
                break
    if header_index is None:
        raise ValueError(f"NCU CSV contains no Metric Name header: {path}")

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    required = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"NCU CSV header is missing required columns: {path}")
    rows = []
    for source in reader:
        metric_name = (source.get("Metric Name") or "").strip()
        if not metric_name:
            continue
        unit = (source.get("Metric Unit") or "").strip()
        raw_value = (source.get("Metric Value") or "").strip()
        rows.append(
            {
                "source_csv": str(Path(path).resolve()),
                "launch_id": (source.get("ID") or "").strip(),
                "process_id": (source.get("Process ID") or "").strip(),
                "context": (source.get("Context") or "").strip(),
                "stream": (source.get("Stream") or "").strip(),
                "kernel_name": (source.get("Kernel Name") or "").strip(),
                "metric_name": metric_name,
                "unit": unit,
                "raw_value": raw_value,
                "value": parse_metric_value(raw_value, unit),
            }
        )
    if not rows:
        raise ValueError(f"NCU CSV contains no metric rows: {path}")
    return rows


def _deduplicated_metric(values: list[float | None], name: str) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if any(not math.isclose(first, value, rel_tol=1e-9, abs_tol=1e-12) for value in present[1:]):
        raise ValueError(f"inconsistent duplicate NCU metric {name}")
    return first


def classify_counter_signature(metrics: dict[str, float | None]) -> dict:
    sm = metrics.get("sm_pct")
    tensor = metrics.get("tensor_pct")
    dram = metrics.get("dram_pct")
    l2 = metrics.get("l2_pct")
    scoreboard = metrics.get("long_scoreboard_pct")
    atomic = metrics.get("atomic_instructions")
    compute_signal = bool(
        (sm is not None and sm >= 60.0)
        or (tensor is not None and tensor >= 30.0)
    )
    memory_throughput_signal = bool(
        max(
            value
            for value in (dram, l2, 0.0)
            if value is not None
        )
        >= 60.0
    )
    latency_signal = bool(scoreboard is not None and scoreboard >= 25.0)
    memory_latency_signal = memory_throughput_signal or latency_signal
    if compute_signal and memory_latency_signal:
        classification = "mixed_compute_memory_latency"
    elif memory_latency_signal:
        classification = "memory_latency_bound"
    elif compute_signal:
        classification = "compute_bound"
    else:
        classification = "insufficient_or_balanced"
    return {
        "schema": "wiki10m-ncu-counter-signature-v2",
        "classification": classification,
        "compute_signal": compute_signal,
        "memory_latency_signal": memory_latency_signal,
        "memory_throughput_signal": memory_throughput_signal,
        "long_scoreboard_signal": latency_signal,
        "atomic_signal": bool(atomic is not None and atomic > 0),
        "atomic_signal_role": "diagnostic_only_not_bound_classifier",
        "thresholds": {
            "sm_compute_pct": 60.0,
            "tensor_compute_pct": 30.0,
            "dram_or_l2_memory_pct": 60.0,
            "long_scoreboard_latency_pct": 25.0,
        },
    }


def _summarize_launches(normalized_launches: list[dict]) -> dict:
    if not normalized_launches:
        raise ValueError("cannot aggregate an empty NCU capture")
    stable = {}
    for field, metric_name in STABLE_METRICS.items():
        if field in PERCENTAGE_FIELDS:
            weighted_sum = 0.0
            duration_sum = 0.0
            for launch in normalized_launches:
                duration = launch["duration_ms"]
                value = launch["metrics"].get(metric_name)
                if duration is not None and value is not None:
                    weighted_sum += duration * value
                    duration_sum += duration
            stable[field] = None if duration_sum == 0 else weighted_sum / duration_sum
        elif field in ADDITIVE_FIELDS:
            values = [
                launch["metrics"].get(metric_name)
                for launch in normalized_launches
                if launch["metrics"].get(metric_name) is not None
            ]
            stable[field] = None if not values else sum(values)
        else:
            raise AssertionError(f"no aggregation rule for {field}")
    durations = [
        launch["duration_ms"]
        for launch in normalized_launches
        if launch["duration_ms"] is not None
    ]
    return {
        "launch_count": len(normalized_launches),
        "launches_with_duration": len(durations),
        "total_duration_ms": sum(durations),
        "metrics": stable,
        "counter_signature": classify_counter_signature(stable),
        "missing_fields": sorted(name for name, value in stable.items() if value is None),
    }


def aggregate_ncu(rows: list[dict], method: str, kernel_class: str) -> dict:
    launches: dict[tuple[str, ...], dict] = {}
    raw_metrics: dict[str, dict] = {}
    for row in rows:
        key = (
            row.get("source_csv", ""),
            row["process_id"],
            row["context"],
            row["stream"],
            row["launch_id"],
            row["kernel_name"],
        )
        launch = launches.setdefault(
            key,
            {
                "kernel_name": row["kernel_name"],
                "metrics": {},
            },
        )
        launch["metrics"].setdefault(row["metric_name"], []).append(row["value"])
        raw = raw_metrics.setdefault(
            row["metric_name"],
            {"sample_count": 0, "present_count": 0, "units": set(), "parsed_sum": 0.0},
        )
        raw["sample_count"] += 1
        raw["units"].add(row["unit"])
        if row["value"] is not None:
            raw["present_count"] += 1
            raw["parsed_sum"] += row["value"]

    normalized_launches = []
    for launch in launches.values():
        metrics = {
            name: _deduplicated_metric(values, name)
            for name, values in launch["metrics"].items()
        }
        normalized_launches.append(
            {
                "kernel_name": launch["kernel_name"],
                "duration_ms": metrics.get(DURATION_METRIC),
                "metrics": metrics,
            }
        )
    summary = _summarize_launches(normalized_launches)
    per_kernel = {
        kernel_name: _summarize_launches(
            [
                launch
                for launch in normalized_launches
                if launch["kernel_name"] == kernel_name
            ]
        )
        for kernel_name in sorted(
            {launch["kernel_name"] for launch in normalized_launches}
        )
    }

    serialized_raw = {
        name: {
            **{key: value for key, value in record.items() if key != "units"},
            "units": sorted(record["units"]),
        }
        for name, record in sorted(raw_metrics.items())
    }
    return {
        "schema": "wiki10m-local-shard-ncu-v1",
        "method": method,
        "kernel_class": kernel_class,
        **summary,
        "raw_metrics": serialized_raw,
        "kernels": list(per_kernel),
        "per_kernel": per_kernel,
    }


def _write_tsv(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "kernel_class",
        "launch_count",
        "total_duration_ms",
        *STABLE_METRICS,
        "counter_signature",
        "missing_fields",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "method": report["method"],
                "kernel_class": report["kernel_class"],
                "launch_count": report["launch_count"],
                "total_duration_ms": f"{report['total_duration_ms']:.9f}",
                **{
                    name: "" if value is None else f"{value:.9f}"
                    for name, value in report["metrics"].items()
                },
                "counter_signature": report["counter_signature"]["classification"],
                "missing_fields": ",".join(report["missing_fields"]),
            }
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument(
        "--method",
        choices=("cagra", "tagore_cagra", "tagore_vamana"),
        required=True,
    )
    parser.add_argument(
        "--kernel-class",
        choices=(
            "dense",
            "irregular",
            "mixed",
            "other_gpu",
            "irregular_cfs",
            "mixed_cfs",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = [row for path in args.csv for row in read_ncu_csv(path)]
    report = aggregate_ncu(rows, method=args.method, kernel_class=args.kernel_class)
    report["source_csv"] = [str(path.resolve()) for path in args.csv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_tsv(args.output_tsv, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
