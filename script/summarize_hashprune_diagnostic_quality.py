#!/usr/bin/env python3
"""Summarize paired Plain, shadow-Plain, and SoftTier diagnostic searches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterable


NOT_APPLICABLE = "not_applicable"
METHODS = ("plain", "shadow_plain", "soft")
PAIRS = (
    ("plain-soft", "plain", "soft"),
    ("shadow_plain-soft", "shadow_plain", "soft"),
    ("plain-shadow_plain", "plain", "shadow_plain"),
)


def safe_ratio(numerator: float, denominator: float) -> float | str:
    return NOT_APPLICABLE if denominator == 0 else numerator / denominator


def read_fbin_u32(path: Path, *, allow_trailing_distances: bool = False) -> list[list[int]]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: fbin is shorter than its header")
    rows, width = struct.unpack_from("<II", raw)
    expected = 8 + rows * width * 4
    accepted_lengths = {expected}
    if allow_trailing_distances:
        accepted_lengths.add(8 + rows * width * 8)
    if len(raw) not in accepted_lengths:
        expected_text = " or ".join(str(length) for length in sorted(accepted_lengths))
        raise ValueError(f"{path}: fbin length {len(raw)} != expected {expected_text}")
    values = struct.unpack_from(f"<{rows * width}I", raw, 8)
    return [list(values[offset : offset + width]) for offset in range(0, len(values), width)]


def _nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, int((len(ordered) * percentile) + 0.999999999))
    return ordered[min(rank, len(ordered)) - 1]


def pairwise_summary(
    gt: list[list[int]], left: list[list[int]], right: list[list[int]], k: int
) -> dict[str, Any]:
    if not (len(gt) == len(left) == len(right)):
        raise ValueError("paired result query counts differ")
    wins = ties = losses = recovered = lost = equal = 0
    intersections = unions = 0
    jaccards: list[float] = []
    recall_deltas: list[float] = []
    for query_gt, left_row, right_row in zip(gt, left, right, strict=True):
        if len(query_gt) < k or len(left_row) < k or len(right_row) < k:
            raise ValueError("ground truth or result width is smaller than K")
        gt_set = set(query_gt[:k])
        left_set = set(left_row[:k])
        right_set = set(right_row[:k])
        left_hits = left_set & gt_set
        right_hits = right_set & gt_set
        delta_hits = len(right_hits) - len(left_hits)
        wins += int(delta_hits > 0)
        ties += int(delta_hits == 0)
        losses += int(delta_hits < 0)
        recovered += len(right_hits - left_hits)
        lost += len(left_hits - right_hits)
        equal += int(left_set == right_set)
        intersection = len(left_set & right_set)
        union = len(left_set | right_set)
        intersections += intersection
        unions += union
        jaccards.append(1.0 if union == 0 else intersection / union)
        recall_deltas.append(delta_hits / k)
    return {
        "queries": len(gt),
        "exact_result_set_equal_queries": equal,
        "micro_result_set_jaccard": safe_ratio(intersections, unions),
        "macro_result_set_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "softtier_wins_canceled_by_losses": min(wins, losses),
        "recovered_gt": recovered,
        "lost_gt": lost,
        "net_recovered_gt": recovered - lost,
        "recall_delta_mean": sum(recall_deltas) / len(recall_deltas) if recall_deltas else 0.0,
        "recall_delta_p50": _nearest_rank(recall_deltas, 0.50),
        "recall_delta_p95": _nearest_rank(recall_deltas, 0.95),
        "recall_delta_min": min(recall_deltas, default=0.0),
        "recall_delta_max": max(recall_deltas, default=0.0),
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: TSV has no data rows")
    return rows


def _read_query_metrics(path: Path, expected_queries: int, expected_l: int) -> list[dict[str, int | None]]:
    rows = read_tsv(path)
    if len(rows) != expected_queries:
        raise ValueError(f"{path}: query row count mismatch")
    counters = (
        "gt_hits",
        "visited_nodes",
        "expanded_nodes",
        "one_bit_evaluations",
        "adaptive_exact_refinements",
        "total_exact_evaluations",
        "logical_exact_vector_bytes",
        "backfill_edges_scanned",
        "backfill_first_discoveries",
        "backfill_beam_admissions",
        "backfill_nodes_expanded",
        "backfill_exact_refinements",
        "backfill_lower_bound_rejects",
        "backfill_final_top10",
        "backfill_final_gt_hits",
    )
    parsed: list[dict[str, int | None]] = []
    for query_id, row in enumerate(rows):
        if int(row["query_id"]) != query_id or int(row["L"]) != expected_l:
            raise ValueError(f"{path}: non-sequential query ID or mismatched L")
        item: dict[str, int | None] = {}
        for key in counters:
            value = int(row[key])
            if value < 0:
                raise ValueError(f"{path}: negative counter {key}")
            item[key] = value
        ordinal = row["first_backfill_admission_expansion_ordinal"]
        item["first_backfill_admission_expansion_ordinal"] = None if ordinal == "NA" else int(ordinal)
        parsed.append(item)
    return parsed


def _counter_deltas(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "visited_nodes",
        "expanded_nodes",
        "one_bit_evaluations",
        "adaptive_exact_refinements",
        "total_exact_evaluations",
        "logical_exact_vector_bytes",
    )
    return {
        f"{key}_delta": sum(int(row[key]) for row in right) - sum(int(row[key]) for row in left)
        for key in keys
    }


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _tsv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = ["\t".join(rows[0])]
    output.extend("\t".join(str(row[key]) for key in rows[0]) for row in rows)
    return "\n".join(output) + "\n"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.experiment_root
    gt = read_fbin_u32(args.gt_file, allow_trailing_distances=True)
    if any(len(row) < args.k for row in gt):
        raise ValueError("ground-truth width is smaller than K")
    construction = json.loads(args.construction_metrics.read_text(encoding="utf-8"))
    dataset_rows = int(construction["rows"])
    primary_edges = int(construction["primary_edges"])
    backfill_edges = int(construction["backfill_edges"])
    if any(value >= dataset_rows for row in gt for value in row):
        raise ValueError("ground truth contains an out-of-range ID")
    pairwise_rows: list[dict[str, Any]] = []
    utilization_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"l_values": {}}

    for l_value in args.l_values:
        results: dict[str, list[list[int]]] = {}
        query_metrics: dict[str, list[dict[str, Any]]] = {}
        performance: dict[str, Any] = {}
        for method in METHODS:
            measured_paths = [
                root / "search" / method / f"L{l_value}" / f"measured{repeat}" / "results.fbin"
                for repeat in (1, 2)
            ]
            if measured_paths[0].read_bytes() != measured_paths[1].read_bytes():
                raise ValueError(f"{method} L={l_value}: measured result files differ")
            results[method] = read_fbin_u32(measured_paths[0])
            if len(results[method]) != len(gt):
                raise ValueError(f"{method} L={l_value}: result query count mismatch")
            if any(value >= dataset_rows for row in results[method] for value in row):
                raise ValueError(f"{method} L={l_value}: result contains an out-of-range ID")
            measured_query_metrics = [
                _read_query_metrics(
                    root / "search" / method / f"L{l_value}" / f"measured{repeat}" / "per_query_metrics.tsv",
                    len(gt),
                    l_value,
                )
                for repeat in (1, 2)
            ]
            query_metrics[method] = measured_query_metrics[0]
            metric_rows = [
                read_tsv(root / "search" / method / f"L{l_value}" / f"measured{repeat}" / "metrics.tsv")[0]
                for repeat in (1, 2)
            ]
            recalls = [float(row["recall"]) for row in metric_rows]
            if recalls[0] != recalls[1]:
                raise ValueError(f"{method} L={l_value}: measured recall differs")
            qps = [float(row["qps"]) for row in metric_rows]
            performance[method] = {
                "recall": recalls[0],
                "qps_values": qps,
                "qps_mean": sum(qps) / 2,
                "qps_min": min(qps),
                "qps_max": max(qps),
            }

        diagnostic_result_path = root / "search" / "soft" / f"L{l_value}" / "diagnostic" / "results.fbin"
        if diagnostic_result_path.read_bytes() != (
            root / "search" / "soft" / f"L{l_value}" / "measured1" / "results.fbin"
        ).read_bytes():
            raise ValueError(f"soft L={l_value}: diagnostic result differs from standard result")
        diagnostic_metrics = _read_query_metrics(
            root / "search" / "soft" / f"L{l_value}" / "diagnostic" / "per_query_metrics.tsv",
            len(gt),
            l_value,
        )

        per_l_pairs: dict[str, Any] = {}
        for pair_name, left_name, right_name in PAIRS:
            pair = pairwise_summary(gt, results[left_name], results[right_name], args.k)
            pair.update(_counter_deltas(query_metrics[left_name], query_metrics[right_name]))
            pairwise_rows.append({"L": l_value, "comparison": pair_name, **pair})
            per_l_pairs[pair_name] = pair

        sums = {
            key: sum(int(row[key]) for row in diagnostic_metrics)
            for key in (
                "backfill_edges_scanned",
                "backfill_first_discoveries",
                "backfill_beam_admissions",
                "backfill_nodes_expanded",
                "backfill_exact_refinements",
                "backfill_lower_bound_rejects",
                "backfill_final_top10",
                "backfill_final_gt_hits",
            )
        }
        causal = per_l_pairs["shadow_plain-soft"]
        degree_inflation = safe_ratio(backfill_edges, primary_edges)
        utilization = {
            "L": l_value,
            "primary_edges": primary_edges,
            "backfill_edges": backfill_edges,
            **sums,
            "queries_using_backfill": sum(
                int(row["backfill_first_discoveries"]) > 0 for row in diagnostic_metrics
            ),
            "degree_inflation": degree_inflation,
            "quality_per_density": (
                NOT_APPLICABLE
                if degree_inflation == NOT_APPLICABLE or degree_inflation == 0
                else causal["recall_delta_mean"] / degree_inflation
            ),
            "backfill_beam_yield": safe_ratio(
                sums["backfill_beam_admissions"], sums["backfill_first_discoveries"]
            ),
            "backfill_expand_yield": safe_ratio(
                sums["backfill_nodes_expanded"], sums["backfill_beam_admissions"]
            ),
            "backfill_gt_yield": safe_ratio(
                sums["backfill_final_gt_hits"], sums["backfill_nodes_expanded"]
            ),
            "net_recovered_gt": causal["net_recovered_gt"],
            "first_admission_ordinals": [
                row["first_backfill_admission_expansion_ordinal"]
                for row in diagnostic_metrics
                if row["first_backfill_admission_expansion_ordinal"] is not None
            ],
        }
        utilization_rows.append(utilization)
        summary["l_values"][str(l_value)] = {
            "canonical_performance": performance,
            "paired": per_l_pairs,
            "diagnostic_backfill": utilization,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(args.output_dir / "pairwise_query_summary.tsv", _tsv(pairwise_rows))
    _atomic_text(args.output_dir / "backfill_utilization.tsv", _tsv(utilization_rows))
    _atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = """# HashPrune SoftTier diagnostic summary

Canonical Recall@10 and QPS below come only from the two standard measured searches. Construction replay and provenance-search counters are diagnostic-only and are excluded from QPS. Ordinary Plain versus SoftTier includes independent-build variation; shadow Plain versus SoftTier is the causal suffix-removal comparison.

`backfill_final_gt_hits` attributes a result to its first-discovery path. It does not prove that one edge caused the result; only the shadow-Plain comparison estimates the causal outcome of removing the suffix.

Interpretation order: low sampled policy divergence points to weighting, quota, or candidate supply; divergence without discovery points to graph placement; discovery and beam admission without GT yield points to selection utility; simultaneous recovered and lost GT IDs point to beam displacement; preserved quality with higher exact work points to search cost. Sampled selection divergence alone is not evidence of search superiority.
"""
    _atomic_text(args.output_dir / "summary.md", markdown)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    parser.add_argument("--construction-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l-values", default="25,50,100,200,400")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    args.l_values = [int(value) for value in args.l_values.split(",") if value]
    if args.k <= 0 or not args.l_values:
        parser.error("K and every L value must be positive")
    return args


if __name__ == "__main__":
    summarize(parse_args())
