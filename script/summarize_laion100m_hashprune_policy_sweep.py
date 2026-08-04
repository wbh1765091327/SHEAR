#!/usr/bin/env python3
"""Deterministic aggregation for the frozen LAION100M HashPrune policy sweep."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


EXPECTED_LABELS = [
    "hashslot_q1",
    "distance_first_q2",
    "distance_first_q3",
    "hard_tier_q2",
    "hard_tier_q3",
    "soft_tier_q2_a1p05",
    "soft_tier_q2_a1p1",
    "soft_tier_q2_a1p2",
    "soft_tier_q2_a1p4",
    "soft_tier_q2_a2p0",
    "soft_tier_q3_a1p05",
    "soft_tier_q3_a1p1",
    "soft_tier_q3_a1p2",
    "soft_tier_q3_a1p4",
    "soft_tier_q3_a2p0",
]
EXPECTED_L = (25, 50, 100, 200, 400)
DEGREE_TOLERANCE = 0.25
BYTE_RELATIVE_TOLERANCE = 0.01
QPS_SPREAD_LIMIT = 0.05
RECALL_ADVANTAGE = 0.001
QPS_ADVANTAGE = 0.02


def method_family(label: str) -> str:
    if label.startswith("hashslot"):
        return "HashSlot"
    if label.startswith("distance_first"):
        return "DistanceFirst"
    if label.startswith("hard_tier"):
        return "HardTier"
    if label.startswith("soft_tier"):
        return "SoftTier"
    raise ValueError(f"unknown configuration label: {label}")


def budgets_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    degree_match = abs(float(left["avg_out_degree"]) - float(right["avg_out_degree"])) <= DEGREE_TOLERANCE
    left_bytes, right_bytes = float(left["graph_bytes"]), float(right["graph_bytes"])
    midpoint = (left_bytes + right_bytes) / 2.0
    byte_match = midpoint > 0 and abs(left_bytes - right_bytes) / midpoint <= BYTE_RELATIVE_TOLERANCE
    return degree_match and byte_match


def stable(point: dict[str, Any]) -> bool:
    return float(point["qps_relative_spread"]) <= QPS_SPREAD_LIMIT


def pareto_frontier(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    points = list(points)
    frontier = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            no_worse = float(other["recall"]) >= float(candidate["recall"]) and float(other["mean_qps"]) >= float(candidate["mean_qps"])
            strictly_better = float(other["recall"]) > float(candidate["recall"]) or float(other["mean_qps"]) > float(candidate["mean_qps"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def pairwise_decision(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    if not budgets_match(left, right) or not stable(left) or not stable(right):
        return None
    recall_delta = float(left["recall"]) - float(right["recall"])
    left_qps, right_qps = float(left["mean_qps"]), float(right["mean_qps"])
    if recall_delta >= RECALL_ADVANTAGE and left_qps >= right_qps:
        return str(left["label"])
    if recall_delta <= -RECALL_ADVANTAGE and right_qps >= left_qps:
        return str(right["label"])
    if abs(recall_delta) < RECALL_ADVANTAGE:
        if right_qps > 0 and left_qps / right_qps >= 1.0 + QPS_ADVANTAGE:
            return str(left["label"])
        if left_qps > 0 and right_qps / left_qps >= 1.0 + QPS_ADVANTAGE:
            return str(right["label"])
    return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_batch(batch_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transactions = sorted((batch_root / "transactions").glob("[0-9][0-9][0-9]_*"))
    if len(transactions) != 15:
        raise ValueError(f"expected 15 transactions, found {len(transactions)}")
    raw_rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    observed_labels = []
    for transaction in transactions:
        status = (transaction / "STATUS").read_text().strip()
        if status != "accepted":
            raise ValueError(f"transaction is not accepted: {transaction} ({status})")
        passport = json.loads((transaction / "parameter_passport.json").read_text())
        accepted = json.loads((transaction / "accepted_metrics.json").read_text())
        label = passport["label"]
        observed_labels.append(label)
        graph = accepted["graph"]
        quality = accepted.get("graph_quality", {})
        backfill = accepted.get("hashprune_backfill", {})
        resource = accepted.get("resource", {})
        with (transaction / "search_measurements.tsv").open(newline="") as stream:
            measured = list(csv.DictReader(stream, delimiter="\t"))
        if len(measured) != 10:
            raise ValueError(f"expected 10 measured search rows for {label}, found {len(measured)}")
        common = {
            "ordinal": int(passport["ordinal"]),
            "label": label,
            "method": method_family(label),
            "policy_mode": passport["policy_mode"],
            "hash_quota": int(passport["requested_hash_quota"]),
            "priority_alpha": passport["priority_alpha"],
            "graph_bytes": int(graph["graph_bytes"]),
            "avg_out_degree": float(graph["avg_out_degree"]),
            "peak_process_rss_bytes": resource.get("peak_process_rss_bytes"),
            "peak_hbm_mib": resource.get("peak_hbm_mib"),
            "build_elapsed_seconds": resource.get("monitor_elapsed_seconds"),
            "directed_reachable_ratio": quality.get("directed_reachable_from_start_ratio"),
            "backfill_edges": backfill.get("backfill_edges"),
            "rows_with_backfill": backfill.get("rows_with_backfill"),
        }
        by_l: dict[int, list[dict[str, str]]] = {value: [] for value in EXPECTED_L}
        for measured_row in measured:
            l_value = int(measured_row["L"])
            if l_value not in by_l:
                raise ValueError(f"unexpected L={l_value} for {label}")
            by_l[l_value].append(measured_row)
            raw_rows.append({**common, **measured_row})
        for l_value, pair in by_l.items():
            if len(pair) != 2 or {row["pass"] for row in pair} != {"pass1", "pass2"}:
                raise ValueError(f"expected pass1/pass2 pair for {label}, L={l_value}")
            recalls = [float(row["recall"]) for row in pair]
            if recalls[0] != recalls[1]:
                raise ValueError(f"recall mismatch for {label}, L={l_value}")
            digests = [row.get("result_sha256", "") for row in pair]
            if not all(digests) or digests[0] != digests[1]:
                raise ValueError(f"result SHA-256 mismatch for {label}, L={l_value}")
            qps = [float(row["qps"]) for row in pair]
            qps_mean = mean(qps)
            points.append({
                **common,
                "L": l_value,
                "recall": recalls[0],
                "mean_qps": qps_mean,
                "qps_min": min(qps),
                "qps_max": max(qps),
                "qps_relative_spread": (max(qps) - min(qps)) / qps_mean if qps_mean else math.inf,
                "stable_qps": qps_mean > 0 and (max(qps) - min(qps)) / qps_mean <= QPS_SPREAD_LIMIT,
                "result_sha256": digests[0],
            })
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(f"configuration schedule mismatch: {observed_labels}")
    if len(raw_rows) != 150 or len(points) != 75:
        raise ValueError(f"unexpected aggregation counts: raw={len(raw_rows)}, points={len(points)}")
    return raw_rows, points


def matched_comparisons(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for l_value in EXPECTED_L:
        group = [point for point in points if point["L"] == l_value]
        for left, right in itertools.combinations(group, 2):
            if not budgets_match(left, right):
                continue
            winner = pairwise_decision(left, right)
            rows.append({
                "L": l_value,
                "left_label": left["label"],
                "right_label": right["label"],
                "degree_difference": abs(left["avg_out_degree"] - right["avg_out_degree"]),
                "byte_relative_difference": abs(left["graph_bytes"] - right["graph_bytes"]) / ((left["graph_bytes"] + right["graph_bytes"]) / 2),
                "left_recall": left["recall"],
                "right_recall": right["recall"],
                "left_mean_qps": left["mean_qps"],
                "right_mean_qps": right["mean_qps"],
                "left_qps_spread": left["qps_relative_spread"],
                "right_qps_spread": right["qps_relative_spread"],
                "winner": winner or "",
                "decision": "advantage-gate" if winner else "unresolved",
            })
    return rows


def recommendations(points: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for l_value in EXPECTED_L:
        group = [point for point in points if point["L"] == l_value]
        scores = {point["label"]: {"wins": 0, "losses": 0} for point in group}
        for comparison in comparisons:
            if comparison["L"] != l_value or not comparison["winner"]:
                continue
            winner = comparison["winner"]
            loser = comparison["right_label"] if winner == comparison["left_label"] else comparison["left_label"]
            scores[winner]["wins"] += 1
            scores[loser]["losses"] += 1
        eligible = [
            (label, value) for label, value in scores.items()
            if value["wins"] > 0 and value["losses"] == 0
        ]
        eligible.sort(key=lambda item: (-item[1]["wins"], item[0]))
        if len(eligible) == 1 or (len(eligible) > 1 and eligible[0][1]["wins"] > eligible[1][1]["wins"]):
            winner, score = eligible[0]
            results.append({"L": l_value, "winner": winner, "wins": score["wins"], "status": "supported"})
        else:
            results.append({"L": l_value, "winner": "unresolved", "wins": 0, "status": "unresolved"})
    return results


def build_report(batch_root: Path) -> dict[str, Any]:
    raw_rows, points = load_batch(batch_root)
    report = batch_root / "report"
    report.mkdir(parents=True, exist_ok=True)

    raw_prefix = [
        "ordinal", "label", "method", "policy_mode", "hash_quota", "priority_alpha",
        "graph_bytes", "avg_out_degree", "peak_process_rss_bytes", "peak_hbm_mib",
        "build_elapsed_seconds", "directed_reachable_ratio", "backfill_edges", "rows_with_backfill",
        "pass", "L", "recall", "qps", "result_sha256",
    ]
    raw_extras = sorted(set().union(*(row.keys() for row in raw_rows)) - set(raw_prefix))
    atomic_write_tsv(report / "all_configurations.tsv", raw_rows, raw_prefix + raw_extras)

    frontier_rows = []
    for l_value in EXPECTED_L:
        frontier_rows.extend(pareto_frontier(point for point in points if point["L"] == l_value))
    point_fields = [
        "ordinal", "label", "method", "policy_mode", "hash_quota", "priority_alpha", "L",
        "recall", "mean_qps", "qps_min", "qps_max", "qps_relative_spread", "stable_qps",
        "graph_bytes", "avg_out_degree", "peak_process_rss_bytes", "peak_hbm_mib",
        "build_elapsed_seconds", "result_sha256",
    ]
    atomic_write_tsv(report / "recall_qps_pareto.tsv", frontier_rows, point_fields)

    comparisons = matched_comparisons(points)
    comparison_fields = [
        "L", "left_label", "right_label", "degree_difference", "byte_relative_difference",
        "left_recall", "right_recall", "left_mean_qps", "right_mean_qps", "left_qps_spread",
        "right_qps_spread", "winner", "decision",
    ]
    atomic_write_tsv(report / "matched_budget.tsv", comparisons, comparison_fields)

    method_winners = []
    for l_value in EXPECTED_L:
        for method in ("HashSlot", "DistanceFirst", "SoftTier", "HardTier"):
            candidates = [point for point in points if point["L"] == l_value and point["method"] == method]
            candidates.sort(key=lambda point: (-point["recall"], -point["mean_qps"], point["label"]))
            winner = candidates[0]
            method_winners.append({**winner, "selection_rule": "max-recall-then-qps"})
    atomic_write_tsv(report / "per_method_winners.tsv", method_winners, point_fields + ["selection_rule"])

    recommendation_rows = recommendations(points, comparisons)
    summary = {
        "schema": "laion100m_hashprune_policy_sweep_summary_v1",
        "accepted_configurations": 15,
        "measured_rows": len(raw_rows),
        "aggregated_points": len(points),
        "pareto_points": len(frontier_rows),
        "matched_budget_pairs": len(comparisons),
        "thresholds": {
            "degree_difference_max": DEGREE_TOLERANCE,
            "byte_relative_difference_max": BYTE_RELATIVE_TOLERANCE,
            "qps_relative_spread_max": QPS_SPREAD_LIMIT,
            "recall_advantage_min": RECALL_ADVANTAGE,
            "qps_advantage_min": QPS_ADVANTAGE,
        },
        "recommendations": recommendation_rows,
    }
    atomic_write_text(report / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# LAION100M HashPrune policy parameter sweep",
        "",
        f"Accepted configurations: {summary['accepted_configurations']}; measured search rows: {summary['measured_rows']}.",
        "",
        "Recommendations are emitted only for matched graph budgets, QPS spread at most 5%, and either "
        "Recall advantage at least 0.001 without lower QPS or QPS advantage at least 2% at matched Recall.",
        "",
        "| L | Winner | Status | Pairwise wins |",
        "|---:|---|---|---:|",
    ]
    for row in recommendation_rows:
        lines.append(f"| {row['L']} | {row['winner']} | {row['status']} | {row['wins']} |")
    lines.extend(["", "See the TSV files for raw, Pareto, matched-budget, and per-method evidence.", ""])
    atomic_write_text(report / "summary.md", "\n".join(lines))
    descriptor = os.open(report, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build_report(arguments.batch_root.resolve())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
