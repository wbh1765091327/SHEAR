#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


EXPECTED_L = (10, 16, 25, 32, 50, 100, 200)
EXPECTED_EXTENSION = (
    ("plain", "plain", "off", None, None),
    ("c1_soft_tier_q3_a1p02", "soft_tier", "rhalf-soft-tier-q3", 1, "1.02"),
    ("c5_soft_tier_q3_a1p02", "soft_tier", "rhalf-soft-tier-q3", 5, "1.02"),
    ("c6_soft_tier_q3_a1p02", "soft_tier", "rhalf-soft-tier-q3", 6, "1.02"),
    ("c8_soft_tier_q3_a1p02", "soft_tier", "rhalf-soft-tier-q3", 8, "1.02"),
    ("c10_soft_tier_q3_a1p02", "soft_tier", "rhalf-soft-tier-q3", 10, "1.02"),
)
PARENT_LABELS = {
    cap: f"c{cap}_soft_tier_q3_a1p02" for cap in (2, 3, 4)
}
DEFAULT_BASELINE_ROOT = Path(
    "/data_staging/wbh/hashprune_softtier_row_cap/wiki10m/20260723T045146Z"
)
CONFIG_FIELDS = (
    "source_batch",
    "ordinal",
    "label",
    "method",
    "policy_mode",
    "row_cap",
    "priority_alpha",
    "avg_out_degree",
    "graph_bytes",
    "max_degree",
    "graph_sha256",
    "backfill_edges",
    "rows_with_backfill",
    "peak_process_rss_bytes",
    "peak_hbm_mib",
    "transaction",
)
CURVE_FIELDS = CONFIG_FIELDS + (
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
)


def read_tsv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


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


def atomic_write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_extension(batch_root):
    transactions = batch_root / "transactions"
    cases = sorted(path for path in transactions.iterdir() if path.is_dir())
    if len(cases) != len(EXPECTED_EXTENSION):
        raise ValueError(
            f"expected {len(EXPECTED_EXTENSION)} extension transactions, found {len(cases)}"
        )
    configurations = []
    curves = []
    for ordinal, (case, expected) in enumerate(
        zip(cases, EXPECTED_EXTENSION), start=1
    ):
        if (case / "STATUS").read_text().strip() != "accepted":
            raise ValueError(f"extension transaction is not accepted: {case}")
        passport = json.loads((case / "parameter_passport.json").read_text())
        observed = (
            passport["label"],
            passport["method"],
            passport["policy_mode"],
            passport.get("row_cap"),
            passport.get("priority_alpha"),
        )
        if observed != expected or int(passport["ordinal"]) != ordinal:
            raise ValueError(
                f"extension passport mismatch for {case}: {observed} != {expected}"
            )
        accepted = json.loads((case / "accepted_metrics.json").read_text())
        graph = accepted["graph"]
        backfill = accepted.get("hashprune_backfill", {})
        resource = accepted.get("resource", {})
        config = {
            "source_batch": "extension",
            "ordinal": ordinal,
            "label": passport["label"],
            "method": passport["method"],
            "policy_mode": passport["policy_mode"],
            "row_cap": passport.get("row_cap"),
            "priority_alpha": passport.get("priority_alpha"),
            "avg_out_degree": float(graph["avg_out_degree"]),
            "graph_bytes": int(graph["graph_bytes"]),
            "max_degree": int(graph.get("max_degree", 64)),
            "graph_sha256": (case / "graph.sha256").read_text().split()[0],
            "backfill_edges": int(backfill.get("backfill_edges", 0)),
            "rows_with_backfill": int(backfill.get("rows_with_backfill", 0)),
            "peak_process_rss_bytes": resource.get("peak_process_rss_bytes"),
            "peak_hbm_mib": resource.get("peak_hbm_mib"),
            "transaction": case.name,
        }
        measurements = read_tsv(case / "search_measurements.tsv")
        if [int(row["L"]) for row in measurements] != list(EXPECTED_L):
            raise ValueError(f"unexpected measured L sequence for {case}")
        for measurement in measurements:
            curves.append({**config, **measurement, "L": int(measurement["L"])})
        configurations.append(config)
    return configurations, curves


def load_parent(baseline_root):
    path = baseline_root / "report" / "per_graph_curves.tsv"
    rows = read_tsv(path)
    selected = [row for row in rows if row["label"] in PARENT_LABELS.values()]
    by_label = {}
    for row in selected:
        by_label.setdefault(row["label"], []).append(row)
    if set(by_label) != set(PARENT_LABELS.values()):
        raise ValueError("parent report is missing a required alpha-1.02 cap")
    configurations = []
    curves = []
    for ordinal, cap in enumerate((2, 3, 4), start=1):
        label = PARENT_LABELS[cap]
        points = sorted(by_label[label], key=lambda row: int(row["L"]))
        if [int(row["L"]) for row in points] != list(EXPECTED_L):
            raise ValueError(f"unexpected parent L sequence for {label}")
        first = points[0]
        config = {
            field: first.get(field, "")
            for field in CONFIG_FIELDS
            if field not in ("source_batch", "ordinal")
        }
        config.update(
            {
                "source_batch": "parent",
                "ordinal": ordinal,
                "row_cap": cap,
                "priority_alpha": "1.02",
            }
        )
        configurations.append(config)
        for point in points:
            curves.append(
                {
                    **config,
                    **{field: point.get(field, "") for field in CURVE_FIELDS},
                    "source_batch": "parent",
                    "ordinal": ordinal,
                    "row_cap": cap,
                    "priority_alpha": "1.02",
                    "L": int(point["L"]),
                }
            )
    return configurations, curves


def cap_sort_key(row):
    cap = row.get("row_cap")
    return (-1 if cap in (None, "") else int(cap), int(row.get("L", 0)))


def pareto_caps(points):
    frontier = []
    for candidate in points:
        recall = float(candidate["recall"])
        qps = float(candidate["qps"])
        dominated = any(
            float(other["recall"]) >= recall
            and float(other["qps"]) >= qps
            and (
                float(other["recall"]) > recall
                or float(other["qps"]) > qps
            )
            for other in points
            if other is not candidate
        )
        if not dominated:
            frontier.append(int(candidate["row_cap"]))
    return sorted(frontier)


def summarize(batch_root, baseline_root):
    extension_configs, extension_curves = load_extension(batch_root)
    parent_configs, parent_curves = load_parent(baseline_root)
    plain_configs = [row for row in extension_configs if row["method"] == "plain"]
    plain_curves = [row for row in extension_curves if row["method"] == "plain"]
    soft_extension_configs = [
        row for row in extension_configs if row["method"] == "soft_tier"
    ]
    soft_extension_curves = [
        row for row in extension_curves if row["method"] == "soft_tier"
    ]
    combined_configs = plain_configs + parent_configs + soft_extension_configs
    combined_curves = plain_curves + parent_curves + soft_extension_curves
    combined_configs.sort(key=cap_sort_key)
    combined_curves.sort(key=cap_sort_key)

    low_l_analysis = []
    for l_value in EXPECTED_L:
        points = [
            row
            for row in combined_curves
            if row["method"] == "soft_tier" and int(row["L"]) == l_value
        ]
        best_recall = max(points, key=lambda row: float(row["recall"]))
        best_qps = max(points, key=lambda row: float(row["qps"]))
        low_l_analysis.append(
            {
                "L": l_value,
                "best_recall_cap": int(best_recall["row_cap"]),
                "best_recall": float(best_recall["recall"]),
                "best_qps_cap": int(best_qps["row_cap"]),
                "best_qps": float(best_qps["qps"]),
                "pareto_caps": pareto_caps(points),
            }
        )

    report = batch_root / "report"
    atomic_write_tsv(
        report / "extension_configurations.tsv",
        extension_configs,
        CONFIG_FIELDS,
    )
    atomic_write_tsv(
        report / "extension_curves.tsv",
        extension_curves,
        CURVE_FIELDS,
    )
    atomic_write_tsv(
        report / "combined_cap_curves.tsv",
        combined_curves,
        CURVE_FIELDS,
    )
    summary = {
        "schema": "wiki10m_softtier_wide_row_cap_extension_v1",
        "baseline_root": str(baseline_root),
        "extension_graphs": len(extension_configs),
        "extension_measured_rows": len(extension_curves),
        "combined_graphs": len(combined_configs),
        "combined_measured_rows": len(combined_curves),
        "combined_caps": sorted(
            int(row["row_cap"])
            for row in combined_configs
            if row["method"] == "soft_tier"
        ),
        "single_run_qps": True,
        "per_L": low_l_analysis,
    }
    atomic_write_json(report / "summary.json", summary)
    atomic_write_text(
        report / "summary.md",
        "# Wiki10M SoftTier wide row-cap extension\n\n"
        f"- 新增图：{summary['extension_graphs']}\n"
        f"- 合并图：{summary['combined_graphs']}\n"
        f"- 合并搜索点：{summary['combined_measured_rows']}\n"
        f"- SoftTier caps：{', '.join(map(str, summary['combined_caps']))}\n"
        "- QPS 为 single-run estimate；所有图使用一次 L=200 全局预热。\n",
    )
    descriptor = os.open(report, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT
    )
    args = parser.parse_args()
    summary = summarize(args.batch_root.resolve(), args.baseline_root.resolve())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
