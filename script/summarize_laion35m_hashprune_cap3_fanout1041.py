#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path


EXPECTED_L = (10, 16, 25, 32, 50, 100, 200)
LOW_L = (10, 16, 25, 32, 50)
LARGE_L = (100, 200)
EXPECTED_MATRIX = (
    ("001", "plain", "plain", "off", None, None),
    ("002", "c6_hashslot_q1", "hashslot", "rhalf-hash-slot-1", 6, None),
    (
        "003",
        "c6_distance_first_q3",
        "distance_first",
        "rhalf-distance-first-q3",
        6,
        None,
    ),
    (
        "004",
        "c6_soft_tier_q3_a1p02",
        "soft_tier",
        "rhalf-soft-tier-q3",
        6,
        1.02,
    ),
    ("005", "c6_hard_tier_q3", "hard_tier", "rhalf-hard-tier-q3", 6, None),
)
COMPETITOR_METHODS = ("hashslot", "distance_first", "hard_tier")
DEGREE_TOLERANCE = 0.25
GRAPH_BYTE_PERCENT_TOLERANCE = 1.0


def read_tsv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def integer(value, default=0):
    if value in (None, ""):
        return default
    return int(value)


def number(value, default=0.0):
    if value in (None, ""):
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value}")
    return result


def optional_number(value):
    if value in (None, "", "NA"):
        return None
    return number(value)


def display(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: display(row.get(field)) for field in fields})
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_matrix(batch_root):
    rows = read_tsv(batch_root / "frozen_matrix.tsv")
    actual = []
    for row in rows:
        actual.append(
            (
                row["ordinal"],
                row["label"],
                row["method"],
                row["policy_mode"],
                integer(row["row_cap"], None) if row["row_cap"] != "NA" else None,
                optional_number(row["priority_alpha"]),
            )
        )
    if tuple(actual) != EXPECTED_MATRIX:
        raise ValueError(f"frozen matrix mismatch: {actual}")


def validate_manifest(batch_root):
    manifest = json.loads((batch_root / "experiment_manifest.json").read_text())
    allowed = {
        "laion35m_hashprune_cap6_fanout421_smoke_v1",
        "laion35m_hashprune_cap6_fanout421_v1",
    }
    if manifest.get("schema") not in allowed:
        raise ValueError(f"unexpected experiment schema: {manifest.get('schema')}")
    expected = {"graphs": 5, "warmups": 5, "measured_rows": 35}
    checks = {key: manifest.get(key) == value for key, value in expected.items()}
    if not all(checks.values()):
        raise ValueError(f"experiment manifest mismatch: {checks}")
    return manifest


def validate_search_calls(case):
    rows = read_tsv(case / "search_calls.tsv")
    if len(rows) != 8:
        raise ValueError(f"{case.name}: expected 8 search calls, found {len(rows)}")
    warmups = [
        row
        for row in rows
        if row.get("kind") == "warmup" and integer(row.get("L")) == 200
    ]
    measured = [
        integer(row.get("L"))
        for row in rows
        if row.get("kind") == "measured"
    ]
    if len(warmups) != 1 or tuple(measured) != EXPECTED_L:
        raise ValueError(f"{case.name}: search schedule mismatch")


def resource_values(resource):
    wall = resource.get("build_wall_seconds")
    if wall is None:
        wall = resource.get("monitor_elapsed_seconds")
    return {
        "build_wall_seconds": optional_number(wall),
        "peak_process_rss_bytes": integer(resource.get("peak_process_rss_bytes"), None),
        "peak_hbm_mib": optional_number(resource.get("peak_hbm_mib")),
    }


def load_configuration(batch_root, expected):
    ordinal, label, method, policy, row_cap, priority_alpha = expected
    case = batch_root / "transactions" / f"{ordinal}_{label}"
    if not case.is_dir():
        raise ValueError(f"missing transaction: {case}")
    if (case / "STATUS").read_text().strip() != "accepted":
        raise ValueError(f"transaction is not accepted: {case.name}")
    if (case / "attempt_count").read_text().strip() != "1":
        raise ValueError(f"transaction was not executed exactly once: {case.name}")
    validate_search_calls(case)
    passport = json.loads((case / "parameter_passport.json").read_text())
    passport_tuple = (
        f"{integer(passport.get('ordinal')):03d}",
        passport.get("label"),
        passport.get("method"),
        passport.get("policy_mode"),
        passport.get("row_cap"),
        optional_number(passport.get("priority_alpha")),
    )
    if passport_tuple != expected:
        raise ValueError(f"parameter passport mismatch for {case.name}: {passport_tuple}")
    accepted = json.loads((case / "accepted_metrics.json").read_text())
    graph = accepted.get("graph", {})
    quality = accepted.get("graph_quality", {})
    backfill = accepted.get("hashprune_backfill", {})
    resource = accepted.get("resource", {})
    if backfill.get("mode") != policy:
        raise ValueError(f"policy profile mismatch for {case.name}")
    requested_cap = backfill.get("requested_max_backfill_per_row")
    effective_cap = backfill.get("effective_max_backfill_per_row")
    if row_cap is None:
        if requested_cap is not None or integer(effective_cap) != 0:
            raise ValueError(f"Plain unexpectedly reports a row cap: {case.name}")
    elif integer(requested_cap, None) != row_cap or integer(effective_cap, None) != row_cap:
        raise ValueError(f"row-cap profile mismatch for {case.name}")
    if method == "soft_tier":
        requested_alpha = optional_number(backfill.get("requested_priority_alpha"))
        effective_alpha = optional_number(backfill.get("effective_priority_alpha"))
        if not math.isclose(requested_alpha, 1.02) or not math.isclose(
            effective_alpha, 1.02
        ):
            raise ValueError(f"SoftTier alpha mismatch for {case.name}")

    graph_bytes = integer(graph.get("graph_bytes"), None)
    avg_degree = number(graph.get("avg_out_degree"))
    max_degree = integer(graph.get("max_degree"), None)
    resources = resource_values(resource)
    configuration = {
        "ordinal": ordinal,
        "label": label,
        "method": method,
        "policy_mode": policy,
        "row_cap": row_cap,
        "priority_alpha": priority_alpha,
        "status": "accepted",
        **resources,
        "graph_bytes": graph_bytes,
        "avg_out_degree": avg_degree,
        "max_out_degree": max_degree,
        "reciprocal_directed_ratio": optional_number(
            quality.get("reciprocal_directed_ratio")
        ),
        "directed_reachable_from_start_ratio": optional_number(
            quality.get("directed_reachable_from_start_ratio")
        ),
        "edge_distance_p50": optional_number(quality.get("edge_distance_p50")),
        "edge_distance_p95": optional_number(quality.get("edge_distance_p95")),
        "edge_distance_p99": optional_number(quality.get("edge_distance_p99")),
        "primary_edges": integer(backfill.get("primary_edges")),
        "backfill_edges": integer(backfill.get("backfill_edges")),
        "rows_underfull_before": integer(backfill.get("rows_underfull_before")),
        "rows_with_backfill": integer(backfill.get("rows_with_backfill")),
        "mean_degree_before": optional_number(backfill.get("mean_degree_before")),
        "mean_degree_after": optional_number(backfill.get("mean_degree_after")),
        "shortlist_admissions": integer(backfill.get("shortlist_admissions")),
        "shortlist_replacements": integer(backfill.get("shortlist_replacements")),
        "shortlist_capacity_rejections": integer(
            backfill.get("shortlist_capacity_rejections")
        ),
        "emitted_layer_1": integer(backfill.get("emitted_layer_1")),
        "emitted_layer_2": integer(backfill.get("emitted_layer_2")),
        "emitted_layer_3": integer(backfill.get("emitted_layer_3")),
        "rows_with_deeper_backfill": integer(
            backfill.get("rows_with_deeper_backfill")
        ),
    }

    measured = read_tsv(case / "search_measurements.tsv")
    if len(measured) != len(EXPECTED_L):
        raise ValueError(f"{case.name}: expected seven measured search rows")
    curves = []
    for expected_l, measurement in zip(EXPECTED_L, measured):
        l_value = integer(measurement.get("L"))
        if l_value != expected_l:
            raise ValueError(f"{case.name}: measured L order mismatch")
        backend = measurement.get("search_backend")
        if backend != "rabitq1-adaptive-exact":
            raise ValueError(f"{case.name}: unexpected search backend {backend}")
        curves.append(
            {
                "ordinal": ordinal,
                "label": label,
                "method": method,
                "policy_mode": policy,
                "row_cap": row_cap,
                "priority_alpha": priority_alpha,
                "L": l_value,
                "recall": number(measurement.get("recall")),
                "qps": number(measurement.get("qps")),
                "search_seconds": optional_number(measurement.get("search_seconds")),
                "avg_visited_nodes": optional_number(
                    measurement.get("avg_visited_nodes")
                ),
                "avg_expanded_nodes": optional_number(
                    measurement.get("avg_expanded_nodes")
                ),
                "avg_distance_comparisons": optional_number(
                    measurement.get("avg_distance_comparisons")
                ),
                "total_one_bit_evaluations": integer(
                    measurement.get("total_one_bit_evaluations"), None
                ),
                "total_navigation_exact_evaluations": integer(
                    measurement.get("total_navigation_exact_evaluations"), None
                ),
                "total_adaptive_exact_refinements": integer(
                    measurement.get("total_adaptive_exact_refinements"), None
                ),
                "total_post_rerank_exact_evaluations": integer(
                    measurement.get("total_post_rerank_exact_evaluations"), None
                ),
                "total_exact_evaluations": integer(
                    measurement.get("total_exact_evaluations"), None
                ),
                "logical_exact_vector_bytes": integer(
                    measurement.get("logical_exact_vector_bytes"), None
                ),
                "search_backend": backend,
                "graph_bytes": graph_bytes,
                "avg_out_degree": avg_degree,
            }
        )
    return configuration, curves, backfill


def resources_match(soft, competitor):
    degree_delta = soft["avg_out_degree"] - competitor["avg_out_degree"]
    if competitor["graph_bytes"] <= 0:
        byte_delta_pct = math.inf
    else:
        byte_delta_pct = (
            (soft["graph_bytes"] - competitor["graph_bytes"])
            / competitor["graph_bytes"]
            * 100.0
        )
    matched = (
        abs(degree_delta) <= DEGREE_TOLERANCE
        and abs(byte_delta_pct) <= GRAPH_BYTE_PERCENT_TOLERANCE
    )
    return matched, degree_delta, byte_delta_pct


def pareto_relation(soft, competitor):
    soft_better_or_equal = (
        soft["recall"] >= competitor["recall"] and soft["qps"] >= competitor["qps"]
    )
    competitor_better_or_equal = (
        competitor["recall"] >= soft["recall"] and competitor["qps"] >= soft["qps"]
    )
    if soft_better_or_equal and (
        soft["recall"] > competitor["recall"] or soft["qps"] > competitor["qps"]
    ):
        return "soft_dominates"
    if competitor_better_or_equal and (
        competitor["recall"] > soft["recall"] or competitor["qps"] > soft["qps"]
    ):
        return "competitor_dominates"
    if soft["recall"] == competitor["recall"] and soft["qps"] == competitor["qps"]:
        return "tie"
    return "tradeoff"


def candidate_vs_plain_relation(candidate, plain):
    candidate_better_or_equal = (
        candidate["recall"] >= plain["recall"]
        and candidate["qps"] >= plain["qps"]
    )
    plain_better_or_equal = (
        plain["recall"] >= candidate["recall"]
        and plain["qps"] >= candidate["qps"]
    )
    if candidate_better_or_equal and (
        candidate["recall"] > plain["recall"]
        or candidate["qps"] > plain["qps"]
    ):
        return "candidate_dominates"
    if plain_better_or_equal and (
        plain["recall"] > candidate["recall"]
        or plain["qps"] > candidate["qps"]
    ):
        return "plain_dominates"
    if (
        candidate["recall"] == plain["recall"]
        and candidate["qps"] == plain["qps"]
    ):
        return "tie"
    return "tradeoff"


def build_plain_uplift(configurations, curves):
    config_by_method = {row["method"]: row for row in configurations}
    curve_by_method_l = {(row["method"], row["L"]): row for row in curves}
    plain_config = config_by_method["plain"]
    rows = []
    for candidate_method in ("hashslot", "distance_first", "soft_tier", "hard_tier"):
        candidate_config = config_by_method[candidate_method]
        graph_bytes_delta_pct = (
            (
                candidate_config["graph_bytes"] - plain_config["graph_bytes"]
            )
            / plain_config["graph_bytes"]
            * 100.0
            if plain_config["graph_bytes"] > 0
            else math.inf
        )
        for l_value in EXPECTED_L:
            plain = curve_by_method_l[("plain", l_value)]
            candidate = curve_by_method_l[(candidate_method, l_value)]
            qps_delta = candidate["qps"] - plain["qps"]
            rows.append(
                {
                    "label": candidate_config["label"],
                    "method": candidate_method,
                    "L": l_value,
                    "plain_recall": plain["recall"],
                    "method_recall": candidate["recall"],
                    "recall_delta": candidate["recall"] - plain["recall"],
                    "plain_qps": plain["qps"],
                    "method_qps": candidate["qps"],
                    "qps_delta": qps_delta,
                    "qps_delta_pct": (
                        qps_delta / plain["qps"] * 100.0
                        if plain["qps"] > 0
                        else math.inf
                    ),
                    "pareto_relation": candidate_vs_plain_relation(
                        candidate, plain
                    ),
                    "avg_out_degree_delta": (
                        candidate_config["avg_out_degree"]
                        - plain_config["avg_out_degree"]
                    ),
                    "graph_bytes_delta_pct": graph_bytes_delta_pct,
                    "backfill_edges": candidate_config["backfill_edges"],
                }
            )
    return rows


def summarize_plain_uplift(rows):
    result = {}
    for method in ("hashslot", "distance_first", "soft_tier", "hard_tier"):
        relations = [
            row["pareto_relation"]
            for row in rows
            if row["method"] == method and row["L"] in LOW_L
        ]
        result[method] = {
            "candidate_dominates": relations.count("candidate_dominates"),
            "plain_dominates": relations.count("plain_dominates"),
            "tradeoffs": relations.count("tradeoff"),
            "ties": relations.count("tie"),
        }
    return result


def build_comparisons(configurations, curves):
    config_by_method = {row["method"]: row for row in configurations}
    curve_by_method_l = {
        (row["method"], row["L"]): row
        for row in curves
    }
    soft_config = config_by_method["soft_tier"]
    rows = []
    for competitor_method in COMPETITOR_METHODS:
        competitor_config = config_by_method[competitor_method]
        matched, degree_delta, byte_delta_pct = resources_match(
            soft_config, competitor_config
        )
        for l_value in EXPECTED_L:
            soft = curve_by_method_l[("soft_tier", l_value)]
            competitor = curve_by_method_l[(competitor_method, l_value)]
            qps_delta_pct = (
                (soft["qps"] - competitor["qps"]) / competitor["qps"] * 100.0
                if competitor["qps"] > 0
                else math.inf
            )
            rows.append(
                {
                    "competitor_label": competitor_config["label"],
                    "competitor_method": competitor_method,
                    "L": l_value,
                    "soft_recall": soft["recall"],
                    "competitor_recall": competitor["recall"],
                    "recall_delta": soft["recall"] - competitor["recall"],
                    "soft_qps": soft["qps"],
                    "competitor_qps": competitor["qps"],
                    "qps_delta": soft["qps"] - competitor["qps"],
                    "qps_delta_pct": qps_delta_pct,
                    "pareto_relation": pareto_relation(soft, competitor),
                    "avg_out_degree_delta": degree_delta,
                    "graph_bytes_delta_pct": byte_delta_pct,
                    "resources_match": matched,
                }
            )
    return rows


def softtier_verdict(comparisons):
    matched_set = {
        row["competitor_method"]
        for row in comparisons
        if row["resources_match"]
    }
    matched = [method for method in COMPETITOR_METHODS if method in matched_set]
    per_method = {}
    for method in COMPETITOR_METHODS:
        low_rows = [
            row
            for row in comparisons
            if row["competitor_method"] == method
            and row["L"] in LOW_L
            and row["resources_match"]
        ]
        relations = [row["pareto_relation"] for row in low_rows]
        per_method[method] = {
            "resource_matched": len(low_rows) == len(LOW_L),
            "soft_dominates": relations.count("soft_dominates"),
            "competitor_dominates": relations.count("competitor_dominates"),
            "tradeoffs": relations.count("tradeoff"),
            "ties": relations.count("tie"),
            "passes": (
                len(low_rows) == len(LOW_L)
                and relations.count("soft_dominates") >= 3
                and relations.count("competitor_dominates") == 0
            ),
        }
    supported = set(matched) == set(COMPETITOR_METHODS) and all(
        per_method[method]["passes"] for method in COMPETITOR_METHODS
    )
    return matched, per_method, supported


def markdown_table(curves, l_values):
    selected = [row for row in curves if row["L"] in l_values]
    lines = [
        "| 方法 | L | Recall@10 | QPS |",
        "|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['label']} | {row['L']} | {row['recall']:.6f} | {row['qps']:.2f} |"
        )
    return "\n".join(lines)


def markdown_uplift_table(rows):
    lines = [
        "| 方法 | L | Recall 增量 | QPS 增量 | QPS 增量 (%) | Pareto 关系 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['L']} | {row['recall_delta']:+.6f} | "
            f"{row['qps_delta']:+.2f} | {row['qps_delta_pct']:+.2f} | "
            f"{row['pareto_relation']} |"
        )
    return "\n".join(lines)


def summarize(batch_root):
    batch_root = batch_root.resolve()
    validate_matrix(batch_root)
    manifest = validate_manifest(batch_root)
    configurations = []
    curves = []
    profiles = {}
    for expected in EXPECTED_MATRIX:
        configuration, points, backfill = load_configuration(batch_root, expected)
        configurations.append(configuration)
        curves.extend(points)
        profiles[configuration["method"]] = backfill

    pressure = []
    for configuration in configurations:
        if configuration["method"] == "plain":
            continue
        backfill = profiles[configuration["method"]]
        requested_cap = integer(backfill.get("requested_max_backfill_per_row"), None)
        effective_cap = integer(backfill.get("effective_max_backfill_per_row"), None)
        replacements = integer(backfill.get("shortlist_replacements"))
        rejections = integer(backfill.get("shortlist_capacity_rejections"))
        pressure.append(
            {
                "label": configuration["label"],
                "method": configuration["method"],
                "policy_mode": configuration["policy_mode"],
                "requested_row_cap": requested_cap,
                "effective_row_cap": effective_cap,
                "requested_priority_alpha": optional_number(
                    backfill.get("requested_priority_alpha")
                ),
                "effective_priority_alpha": optional_number(
                    backfill.get("effective_priority_alpha")
                ),
                "shortlist_admissions": integer(
                    backfill.get("shortlist_admissions")
                ),
                "shortlist_replacements": replacements,
                "shortlist_capacity_rejections": rejections,
                "emitted_layer_1": integer(backfill.get("emitted_layer_1")),
                "emitted_layer_2": integer(backfill.get("emitted_layer_2")),
                "emitted_layer_3": integer(backfill.get("emitted_layer_3")),
                "rows_with_deeper_backfill": integer(
                    backfill.get("rows_with_deeper_backfill")
                ),
                "cap_pressure_observed": (
                    requested_cap == 6
                    and effective_cap == 6
                    and (replacements > 0 or rejections > 0)
                ),
            }
        )

    plain_uplift = build_plain_uplift(configurations, curves)
    plain_uplift_low_l = summarize_plain_uplift(plain_uplift)
    comparisons = build_comparisons(configurations, curves)
    resource_matched = [row for row in comparisons if row["resources_match"]]
    matched_methods, per_method, supported = softtier_verdict(comparisons)
    cap_pressure_all = len(pressure) == 4 and all(
        row["cap_pressure_observed"] for row in pressure
    )

    report = batch_root / "report"
    config_fields = [
        "ordinal",
        "label",
        "method",
        "policy_mode",
        "row_cap",
        "priority_alpha",
        "status",
        "build_wall_seconds",
        "peak_process_rss_bytes",
        "peak_hbm_mib",
        "graph_bytes",
        "avg_out_degree",
        "max_out_degree",
        "reciprocal_directed_ratio",
        "directed_reachable_from_start_ratio",
        "edge_distance_p50",
        "edge_distance_p95",
        "edge_distance_p99",
        "primary_edges",
        "backfill_edges",
        "rows_underfull_before",
        "rows_with_backfill",
        "mean_degree_before",
        "mean_degree_after",
        "shortlist_admissions",
        "shortlist_replacements",
        "shortlist_capacity_rejections",
        "emitted_layer_1",
        "emitted_layer_2",
        "emitted_layer_3",
        "rows_with_deeper_backfill",
    ]
    curve_fields = [
        "ordinal",
        "label",
        "method",
        "policy_mode",
        "row_cap",
        "priority_alpha",
        "L",
        "recall",
        "qps",
        "search_seconds",
        "avg_visited_nodes",
        "avg_expanded_nodes",
        "avg_distance_comparisons",
        "total_one_bit_evaluations",
        "total_navigation_exact_evaluations",
        "total_adaptive_exact_refinements",
        "total_post_rerank_exact_evaluations",
        "total_exact_evaluations",
        "logical_exact_vector_bytes",
        "search_backend",
        "graph_bytes",
        "avg_out_degree",
    ]
    pressure_fields = [
        "label",
        "method",
        "policy_mode",
        "requested_row_cap",
        "effective_row_cap",
        "requested_priority_alpha",
        "effective_priority_alpha",
        "shortlist_admissions",
        "shortlist_replacements",
        "shortlist_capacity_rejections",
        "emitted_layer_1",
        "emitted_layer_2",
        "emitted_layer_3",
        "rows_with_deeper_backfill",
        "cap_pressure_observed",
    ]
    plain_uplift_fields = [
        "label",
        "method",
        "L",
        "plain_recall",
        "method_recall",
        "recall_delta",
        "plain_qps",
        "method_qps",
        "qps_delta",
        "qps_delta_pct",
        "pareto_relation",
        "avg_out_degree_delta",
        "graph_bytes_delta_pct",
        "backfill_edges",
    ]
    comparison_fields = [
        "competitor_label",
        "competitor_method",
        "L",
        "soft_recall",
        "competitor_recall",
        "recall_delta",
        "soft_qps",
        "competitor_qps",
        "qps_delta",
        "qps_delta_pct",
        "pareto_relation",
        "avg_out_degree_delta",
        "graph_bytes_delta_pct",
        "resources_match",
    ]
    atomic_write_tsv(report / "all_configurations.tsv", configurations, config_fields)
    atomic_write_tsv(report / "per_graph_curves.tsv", curves, curve_fields)
    atomic_write_tsv(report / "candidate_pressure.tsv", pressure, pressure_fields)
    atomic_write_tsv(
        report / "plain_uplift.tsv", plain_uplift, plain_uplift_fields
    )
    atomic_write_tsv(
        report / "same_cap_comparisons.tsv", comparisons, comparison_fields
    )
    atomic_write_tsv(
        report / "resource_matched_comparisons.tsv",
        resource_matched,
        comparison_fields,
    )

    summary = {
        "schema": "laion35m_hashprune_cap6_fanout421_summary_v1",
        "experiment_schema": manifest["schema"],
        "graphs": len(configurations),
        "measured_rows": len(curves),
        "cap_pressure_all_enabled": cap_pressure_all,
        "resource_match_gate": {
            "max_abs_avg_out_degree_delta": DEGREE_TOLERANCE,
            "max_abs_graph_bytes_delta_pct": GRAPH_BYTE_PERCENT_TOLERANCE,
        },
        "resource_matched_competitors": matched_methods,
        "softtier_low_l_verdicts": per_method,
        "softtier_superiority_supported": supported and cap_pressure_all,
        "plain_uplift_low_l": plain_uplift_low_l,
        "qps_measurement": "single-run estimate",
    }
    atomic_write_json(report / "summary.json", summary)

    if summary["softtier_superiority_supported"]:
        conclusion = (
            "在预先规定的资源匹配门槛和低 L 判据下，本次固定 SoftTier 图支持优于"
            "三个替代策略的 Recall@10--QPS 权衡。"
        )
    elif set(matched_methods) != set(COMPETITOR_METHODS):
        conclusion = (
            "至少一个替代策略未通过资源匹配门槛，因此本次结果不足以支持 SoftTier "
            "对全部替代策略的优越性结论。"
        )
    else:
        conclusion = (
            "资源匹配比较呈现混合证据或权衡关系，本次结果不支持 SoftTier 对全部"
            "替代策略的统一优越性结论。"
        )
    pressure_text = (
        "四种启用策略均观察到 shortlist replacement 或 capacity rejection，"
        "说明 cap=6 确实触发了候选选择压力。"
        if cap_pressure_all
        else "并非所有启用策略都观察到 cap=6 的候选选择压力。"
    )
    markdown = f"""# LAION35M HashPrune cap=6、fanout 4/2/1 报告

## 实验口径

固定 Plain、HashSlot-1、DistanceFirst-Q3、SoftTier-Q3 和 HardTier-Q3；
四种补边策略均使用 cap=6，SoftTier priority alpha=1.02。搜索采用
`rabitq1-adaptive-exact`，每张图只在 L=200 预热一次，随后按升序测量
L=10,16,25,32,50,100,200。

QPS 是单次运行估计，不能据此声称统计显著性。

## 候选压力

{pressure_text}

## 相对 Plain 的提升

以下增量均以相同 L 的 Plain 为基准；正 Recall 增量表示质量提高，正 QPS
增量表示吞吐提高。该表用于描述补边收益，不作为实验继续或验收门槛。

{markdown_uplift_table(plain_uplift)}

## 低 L：L=10, 16, 25, 32, 50

{markdown_table(curves, LOW_L)}

## 大 L：L=100, 200

{markdown_table(curves, LARGE_L)}

## 资源匹配与结论

资源匹配要求平均出度差不超过 {DEGREE_TOLERANCE:.2f}，且图字节差不超过
{GRAPH_BYTE_PERCENT_TOLERANCE:.1f}%。通过门槛的替代策略：
{', '.join(matched_methods) if matched_methods else '无'}。

{conclusion}
"""
    atomic_write_text(report / "summary.md", markdown)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.batch_root)


if __name__ == "__main__":
    main()
