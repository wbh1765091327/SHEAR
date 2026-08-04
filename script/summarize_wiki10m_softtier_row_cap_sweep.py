#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path


EXPECTED_L = (10, 16, 25, 32, 50, 100, 200)
LOW_L = (10, 16, 25, 32, 50)
EXPECTED_GRAPHS = 18
DEGREE_TOLERANCE = 0.25
BYTE_RELATIVE_TOLERANCE = 0.01
RECALL_MATCH_TOLERANCE = 0.001
QPS_MATCH_TOLERANCE = 0.02
RECALL_ADVANTAGE = 0.003
QPS_ADVANTAGE = 0.02
REQUIRED_LOW_L_WINS = 3
COMPETITOR_METHODS = ("hashslot", "distance_first", "hard_tier")


def pareto_frontier(points):
    frontier = []
    for candidate in points:
        dominated = any(
            other["recall"] >= candidate["recall"]
            and other["qps"] >= candidate["qps"]
            and (
                other["recall"] > candidate["recall"]
                or other["qps"] > candidate["qps"]
            )
            for other in points
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda point: point["label"])


def resources_match(left, right):
    if abs(left["avg_out_degree"] - right["avg_out_degree"]) > DEGREE_TOLERANCE:
        return False
    denominator = min(left["graph_bytes"], right["graph_bytes"])
    if denominator <= 0:
        return False
    return abs(left["graph_bytes"] - right["graph_bytes"]) / denominator <= (
        BYTE_RELATIVE_TOLERANCE + 1e-15
    )


def point_win_forms(soft, competitor):
    qps_win = (
        soft["recall"] + RECALL_MATCH_TOLERANCE >= competitor["recall"]
        and soft["qps"] + 1e-12 >= competitor["qps"] * (1.0 + QPS_ADVANTAGE)
    )
    recall_win = (
        soft["qps"] + 1e-12 >= competitor["qps"] * (1.0 - QPS_MATCH_TOLERANCE)
        and soft["recall"] + 1e-15
        >= competitor["recall"] + RECALL_ADVANTAGE
    )
    forms = []
    if qps_win:
        forms.append("qps")
    if recall_win:
        forms.append("recall")
    return forms


def joint_regression(soft, competitor):
    return (
        soft["recall"] + RECALL_MATCH_TOLERANCE < competitor["recall"]
        and soft["qps"] < competitor["qps"] * (1.0 - QPS_MATCH_TOLERANCE)
    )


def evaluate_soft_candidate(soft_curve, competitors):
    labels = {point["label"] for point in soft_curve}
    if len(labels) != 1:
        raise ValueError("a SoftTier candidate must be one fixed graph")
    by_l = {int(point["L"]): point for point in soft_curve if int(point["L"]) in LOW_L}
    if set(by_l) != set(LOW_L):
        raise ValueError("a SoftTier candidate must contain all five low-L points")
    label = next(iter(labels))
    method_results = {}
    for method in COMPETITOR_METHODS:
        method_points = competitors.get(method, [])
        decisions = []
        wins = 0
        veto = False
        missing = []
        for l_value in LOW_L:
            soft = by_l[l_value]
            matched = [
                point
                for point in method_points
                if int(point["L"]) == l_value and resources_match(soft, point)
            ]
            envelope = pareto_frontier(matched)
            if not envelope:
                missing.append(l_value)
                decisions.append(
                    {
                        "L": l_value,
                        "status": "no_resource_match",
                        "competitor_labels": [],
                        "win_forms": [],
                    }
                )
                continue
            forms_by_point = [point_win_forms(soft, point) for point in envelope]
            is_win = all(forms for forms in forms_by_point)
            is_regression = any(joint_regression(soft, point) for point in envelope)
            if is_win:
                wins += 1
                status = "win"
            elif is_regression:
                veto = True
                status = "joint_regression"
            else:
                status = "within_gate_without_win"
            decisions.append(
                {
                    "L": l_value,
                    "status": status,
                    "competitor_labels": [point["label"] for point in envelope],
                    "win_forms": sorted({form for forms in forms_by_point for form in forms}),
                    "soft_recall": soft["recall"],
                    "soft_qps": soft["qps"],
                    "competitor_frontier": [
                        {
                            "label": point["label"],
                            "recall": point["recall"],
                            "qps": point["qps"],
                        }
                        for point in envelope
                    ],
                }
            )
        passes = wins >= REQUIRED_LOW_L_WINS and not veto and not missing
        method_results[method] = {
            "passes": passes,
            "wins": wins,
            "joint_regression_veto": veto,
            "missing_L": missing,
            "decisions": decisions,
        }
    first = by_l[LOW_L[0]]
    alpha = first.get("priority_alpha")
    return {
        "label": label,
        "avg_out_degree": first["avg_out_degree"],
        "graph_bytes": first["graph_bytes"],
        "priority_alpha": alpha,
        "row_cap": first.get("row_cap"),
        "methods": method_results,
        "passes": all(result["passes"] for result in method_results.values()),
    }


def select_passing_softtier(verdicts):
    passing = [verdict for verdict in verdicts if verdict["passes"]]
    if not passing:
        return None

    def key(verdict):
        alpha = verdict.get("priority_alpha")
        alpha_distance = (
            abs(float(alpha) - 1.10)
            if alpha not in (None, "", "NA")
            else math.inf
        )
        return (
            float(verdict["avg_out_degree"]),
            alpha_distance,
            verdict["label"],
        )

    return min(passing, key=key)


def read_tsv(path):
    with path.open(newline="") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        return list(csv.DictReader(lines, delimiter="\t"))


def numeric(value, *, integer=False, default=0):
    if value in (None, ""):
        return default
    return int(value) if integer else float(value)


def load_batch(batch_root):
    transactions = batch_root / "transactions"
    if not transactions.is_dir():
        raise ValueError(f"missing transactions directory: {transactions}")
    configurations = []
    points = []
    for case in sorted(path for path in transactions.iterdir() if path.is_dir()):
        status_path = case / "STATUS"
        if not status_path.is_file() or status_path.read_text().strip() != "accepted":
            raise ValueError(f"transaction is not accepted: {case}")
        passport = json.loads((case / "parameter_passport.json").read_text())
        accepted = json.loads((case / "accepted_metrics.json").read_text())
        graph = accepted["graph"]
        graph_hash = (case / "graph.sha256").read_text().split()[0]
        alpha = passport.get("priority_alpha")
        alpha_value = None if alpha in (None, "", "NA") else float(alpha)
        config = {
            "ordinal": int(passport["ordinal"]),
            "label": passport["label"],
            "method": passport["method"],
            "policy_mode": passport["policy_mode"],
            "row_cap": int(passport["row_cap"]),
            "priority_alpha": alpha_value,
            "diagnostics": int(passport.get("diagnostics", 0)),
            "avg_out_degree": float(graph["avg_out_degree"]),
            "graph_bytes": int(graph["graph_bytes"]),
            "max_degree": int(graph.get("max_degree", 64)),
            "graph_sha256": graph_hash,
            "backfill_edges": int(
                accepted.get("hashprune_backfill", {}).get("backfill_edges", 0)
            ),
            "rows_with_backfill": int(
                accepted.get("hashprune_backfill", {}).get("rows_with_backfill", 0)
            ),
            "peak_process_rss_bytes": accepted.get("resource", {}).get(
                "peak_process_rss_bytes"
            ),
            "peak_hbm_mib": accepted.get("resource", {}).get("peak_hbm_mib"),
            "transaction": case.name,
        }
        measurements = read_tsv(case / "search_measurements.tsv")
        observed_l = [int(row["L"]) for row in measurements]
        if observed_l != list(EXPECTED_L):
            raise ValueError(f"unexpected measured L sequence for {case}: {observed_l}")
        for row in measurements:
            point = {
                **config,
                "L": int(row["L"]),
                "recall": float(row["recall"]),
                "qps": float(row["qps"]),
                "result_sha256": row.get("result_sha256", ""),
            }
            for field in (
                "avg_visited_nodes",
                "p95_visited_nodes",
                "avg_expanded_nodes",
                "p95_expanded_nodes",
                "total_one_bit_evaluations",
                "total_lower_bound_rejects",
                "total_navigation_exact_evaluations",
                "total_adaptive_exact_refinements",
                "total_post_rerank_exact_evaluations",
                "total_exact_evaluations",
                "logical_exact_vector_bytes",
                "one_bit_estimate_seconds",
                "exact_l2_seconds",
                "navigation_seconds",
            ):
                if row.get(field) not in (None, ""):
                    point[field] = float(row[field])
            points.append(point)
        configurations.append(config)
    if len(configurations) != EXPECTED_GRAPHS:
        raise ValueError(
            f"expected {EXPECTED_GRAPHS} accepted graphs, found {len(configurations)}"
        )
    if len(points) != EXPECTED_GRAPHS * len(EXPECTED_L):
        raise ValueError(f"expected 126 measured rows, found {len(points)}")
    return configurations, points


def build_global_envelope(points):
    rows = []
    for method in sorted({point["method"] for point in points}):
        for l_value in EXPECTED_L:
            candidates = [
                point
                for point in points
                if point["method"] == method and point["L"] == l_value
            ]
            rows.extend(pareto_frontier(candidates))
    return rows


def parse_diagnostics(batch_root, configurations, verdicts):
    verdict_by_label = {verdict["label"]: verdict for verdict in verdicts}
    rows = []
    for config in configurations:
        if not config["diagnostics"]:
            continue
        root = batch_root / "transactions" / config["transaction"] / "diagnostics"
        metrics = json.loads((root / "construction_metrics.json").read_text())
        pairs = {
            row["pair"]: row
            for row in read_tsv(root / "counterfactual_policy_summary.tsv")
        }
        alpha_rows = {
            row["alpha"]: row for row in read_tsv(root / "alpha_sensitivity.tsv")
        }

        def divergence(pair_name):
            pair = pairs.get(pair_name, {})
            compared = int(pair.get("rows_compared", 0))
            changed = int(pair.get("rows_different_selected_sets", 0))
            return changed / compared if compared else None

        distance_divergence = divergence("distance_first__soft_110")
        hard_divergence = divergence("soft_110__hard_tier")
        binding = metrics.get("capacity_binding_rate")
        verdict = verdict_by_label.get(config["label"], {"passes": False})
        if binding is not None and float(binding) < 0.10:
            classification = "capacity_binding_remains_low"
        elif (
            distance_divergence is not None
            and hard_divergence is not None
            and min(distance_divergence, hard_divergence) < 0.05
        ):
            classification = "endpoint_divergence_below_5pct"
        elif verdict["passes"]:
            classification = "active_policy_and_primary_gate_passed"
        else:
            classification = "active_policy_without_verified_low_l_gate"
        rows.append(
            {
                "label": config["label"],
                "row_cap": config["row_cap"],
                "requested_max_backfill_per_row": metrics.get(
                    "requested_max_backfill_per_row"
                ),
                "effective_max_backfill_per_row": metrics.get(
                    "effective_max_backfill_per_row"
                ),
                "rows_capacity_bound": metrics.get("rows_capacity_bound"),
                "capacity_binding_rate": binding,
                "distance_first_divergence": distance_divergence,
                "hard_tier_divergence": hard_divergence,
                "alpha_102_changed_rows": alpha_rows.get("1.02", {}).get(
                    "changed_rows"
                ),
                "alpha_105_changed_rows": alpha_rows.get("1.05", {}).get(
                    "changed_rows"
                ),
                "classification": classification,
            }
        )
    return sorted(rows, key=lambda row: row["row_cap"])


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_tsv(path, rows, fieldnames):
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def summarize(batch_root):
    configurations, points = load_batch(batch_root)
    points_by_method = {
        method: [point for point in points if point["method"] == method]
        for method in COMPETITOR_METHODS
    }
    soft_labels = sorted(
        {point["label"] for point in points if point["method"] == "soft_tier"}
    )
    verdicts = [
        evaluate_soft_candidate(
            [point for point in points if point["label"] == label],
            points_by_method,
        )
        for label in soft_labels
    ]
    selected = select_passing_softtier(verdicts)
    global_envelope = build_global_envelope(points)
    comparisons = []
    gate_rows = []
    for verdict in verdicts:
        for method, result in verdict["methods"].items():
            gate_rows.append(
                {
                    "softtier_label": verdict["label"],
                    "competitor_method": method,
                    "wins": result["wins"],
                    "required_wins": REQUIRED_LOW_L_WINS,
                    "joint_regression_veto": int(result["joint_regression_veto"]),
                    "missing_L": ",".join(map(str, result["missing_L"])),
                    "method_pass": int(result["passes"]),
                    "fixed_graph_overall_pass": int(verdict["passes"]),
                }
            )
            for decision in result["decisions"]:
                comparisons.append(
                    {
                        "softtier_label": verdict["label"],
                        "competitor_method": method,
                        "L": decision["L"],
                        "status": decision["status"],
                        "competitor_labels": ",".join(
                            decision["competitor_labels"]
                        ),
                        "win_forms": ",".join(decision["win_forms"]),
                        "soft_recall": decision.get("soft_recall"),
                        "soft_qps": decision.get("soft_qps"),
                        "competitor_frontier_json": json.dumps(
                            decision.get("competitor_frontier", []),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
    mechanism = parse_diagnostics(batch_root, configurations, verdicts)
    report = batch_root / "report"
    config_fields = [
        "ordinal",
        "label",
        "method",
        "policy_mode",
        "row_cap",
        "priority_alpha",
        "diagnostics",
        "avg_out_degree",
        "graph_bytes",
        "max_degree",
        "graph_sha256",
        "backfill_edges",
        "rows_with_backfill",
        "peak_process_rss_bytes",
        "peak_hbm_mib",
        "transaction",
    ]
    point_fields = config_fields + [
        "L",
        "recall",
        "qps",
        "avg_visited_nodes",
        "p95_visited_nodes",
        "avg_expanded_nodes",
        "p95_expanded_nodes",
        "total_one_bit_evaluations",
        "total_lower_bound_rejects",
        "total_navigation_exact_evaluations",
        "total_adaptive_exact_refinements",
        "total_post_rerank_exact_evaluations",
        "total_exact_evaluations",
        "logical_exact_vector_bytes",
        "one_bit_estimate_seconds",
        "exact_l2_seconds",
        "navigation_seconds",
        "result_sha256",
    ]
    atomic_write_tsv(
        report / "all_configurations.tsv",
        sorted(configurations, key=lambda row: row["ordinal"]),
        config_fields,
    )
    atomic_write_tsv(
        report / "per_graph_curves.tsv",
        sorted(points, key=lambda row: (row["ordinal"], row["L"])),
        point_fields,
    )
    atomic_write_tsv(
        report / "method_tuning_envelope.tsv",
        sorted(
            global_envelope,
            key=lambda row: (row["method"], row["L"], row["label"]),
        ),
        point_fields,
    )
    atomic_write_tsv(
        report / "resource_matched_comparisons.tsv",
        comparisons,
        [
            "softtier_label",
            "competitor_method",
            "L",
            "status",
            "competitor_labels",
            "win_forms",
            "soft_recall",
            "soft_qps",
            "competitor_frontier_json",
        ],
    )
    atomic_write_tsv(
        report / "softtier_gate.tsv",
        gate_rows,
        [
            "softtier_label",
            "competitor_method",
            "wins",
            "required_wins",
            "joint_regression_veto",
            "missing_L",
            "method_pass",
            "fixed_graph_overall_pass",
        ],
    )
    atomic_write_tsv(
        report / "mechanism_diagnostics.tsv",
        mechanism,
        [
            "label",
            "row_cap",
            "requested_max_backfill_per_row",
            "effective_max_backfill_per_row",
            "rows_capacity_bound",
            "capacity_binding_rate",
            "distance_first_divergence",
            "hard_tier_divergence",
            "alpha_102_changed_rows",
            "alpha_105_changed_rows",
            "classification",
        ],
    )
    summary = {
        "schema": "wiki10m_softtier_row_cap_summary_v1",
        "accepted_graphs": len(configurations),
        "measured_rows": len(points),
        "low_L": list(LOW_L),
        "single_run_qps": True,
        "primary_gate_passed": selected is not None,
        "recommended_softtier": selected["label"] if selected else None,
        "mechanism_caps": [row["row_cap"] for row in mechanism],
        "softtier_verdicts": verdicts,
        "thresholds": {
            "degree_tolerance": DEGREE_TOLERANCE,
            "byte_relative_tolerance": BYTE_RELATIVE_TOLERANCE,
            "recall_match_tolerance": RECALL_MATCH_TOLERANCE,
            "qps_match_tolerance": QPS_MATCH_TOLERANCE,
            "recall_advantage": RECALL_ADVANTAGE,
            "qps_advantage": QPS_ADVANTAGE,
            "required_low_L_wins": REQUIRED_LOW_L_WINS,
        },
    }
    atomic_write_json(report / "summary.json", summary)
    verdict_text = (
        f"PASS；固定图配置 `{selected['label']}` 同时通过三类对照。"
        if selected
        else "FAIL；没有任何一个固定 SoftTier 图配置同时通过三类对照。"
    )
    markdown = f"""# Wiki10M SoftTier row-cap sweep

{verdict_text}

- 完成图配置：{len(configurations)}
- 测量搜索点：{len(points)}
- 低预算判定范围：L = {", ".join(map(str, LOW_L))}
- QPS 口径：single-run QPS estimate（单次测量估计），不表示统计显著性或跨次稳定性。

所有胜负均在平均度差不超过 {DEGREE_TOLERANCE}、索引字节差不超过
{BYTE_RELATIVE_TOLERANCE:.0%} 的图之间判定；未插值、未外推，也未把不同
SoftTier 图的搜索点拼接成一条通过曲线。若本阶段失败，不扩展冻结网格；
后续 utility gate 必须另行设计。
"""
    atomic_write_text(report / "summary.md", markdown)
    descriptor = os.open(report, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.batch_root.resolve())
    print(
        json.dumps(
            {
                "accepted_graphs": summary["accepted_graphs"],
                "measured_rows": summary["measured_rows"],
                "primary_gate_passed": summary["primary_gate_passed"],
                "recommended_softtier": summary["recommended_softtier"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
