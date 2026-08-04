#!/usr/bin/env python3
"""Aggregate the focused LAION100M four-policy HashPrune+ cap-3 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean


EXPECTED_LABELS = (
    "plain",
    "distance_first_q3_cap3",
    "hard_tier_q3_cap3",
    "soft_tier_q3_cap3_a1p02",
)
EXPECTED_L = (10, 16, 25, 32, 50, 64, 80, 100, 128, 160, 200, 300, 400)


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(text)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args()
    batch_root = args.batch_root
    transactions = sorted((batch_root / "transactions").glob("[0-9][0-9][0-9]_*"))
    if len(transactions) != len(EXPECTED_LABELS):
        raise SystemExit(f"expected four transactions, found {len(transactions)}")

    raw_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    observed_labels: list[str] = []
    build_rows: list[dict[str, object]] = []
    for transaction in transactions:
        if (transaction / "STATUS").read_text().strip() != "accepted":
            raise SystemExit(f"transaction is not accepted: {transaction}")
        passport = json.loads((transaction / "parameter_passport.json").read_text())
        accepted = json.loads((transaction / "accepted_metrics.json").read_text())
        label = passport["label"]
        observed_labels.append(label)
        graph = accepted["graph"]
        profile = accepted["hashprune_backfill"]
        resource = accepted["resource"]
        build_rows.append(
            {
                "label": label,
                "build_seconds": resource.get("monitor_elapsed_seconds"),
                "peak_rss_bytes": resource.get("peak_process_rss_bytes"),
                "peak_hbm_mib": resource.get("peak_hbm_mib"),
                "graph_bytes": graph["graph_bytes"],
                "avg_out_degree": graph["avg_out_degree"],
                "backfill_edges": profile.get("backfill_edges"),
                "rows_with_backfill": profile.get("rows_with_backfill"),
            }
        )
        with (transaction / "search_measurements.tsv").open(newline="") as stream:
            measured = list(csv.DictReader(stream, delimiter="\t"))
        by_l = {value: [] for value in EXPECTED_L}
        for row in measured:
            l_value = int(row["L"])
            if l_value not in by_l:
                raise SystemExit(f"unexpected L={l_value} for {label}")
            by_l[l_value].append(row)
            raw_rows.append({"label": label, **row})
        for l_value, pair in by_l.items():
            if len(pair) != 1 or {row["pass"] for row in pair} != {"pass1"}:
                raise SystemExit(f"expected one pass1 measurement for {label}, L={l_value}")
            recalls = {row["recall"] for row in pair}
            digests = {row["result_sha256"] for row in pair}
            if len(recalls) != 1 or len(digests) != 1:
                raise SystemExit(f"nondeterministic result for {label}, L={l_value}")
            qps = [float(row["qps"]) for row in pair]
            point_rows.append(
                {
                    "label": label,
                    "L": l_value,
                    "recall": float(pair[0]["recall"]),
                    "mean_qps": mean(qps),
                    "min_qps": min(qps),
                    "max_qps": max(qps),
                    "qps_relative_spread": 0.0,
                }
            )
    if tuple(observed_labels) != EXPECTED_LABELS:
        raise SystemExit(f"configuration order mismatch: {observed_labels}")

    report = batch_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("build_summary.tsv", build_rows),
        ("search_raw.tsv", raw_rows),
        ("recall_qps.tsv", point_rows),
    ):
        fields = list(rows[0])
        lines = ["\t".join(fields)]
        lines.extend("\t".join(str(row.get(field, "")) for field in fields) for row in rows)
        atomic_write(report / name, "\n".join(lines) + "\n")

    markdown = [
        "# LAION100M HashPrune four-policy cap-3 comparison",
        "",
        "All Q3 policies use an added-edge cap of 3. SoftTier-Q3 uses alpha=1.02.",
        "Search uses RaBitQ-1bit adaptive exact reranking with 8 CPU threads.",
        "",
        "| Policy | Build (s) | Avg. degree | Index (GiB) | Backfill edges |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in build_rows:
        markdown.append(
            f"| {row['label']} | {float(row['build_seconds']):.2f} | "
            f"{float(row['avg_out_degree']):.4f} | {int(row['graph_bytes']) / 2**30:.3f} | "
            f"{int(row['backfill_edges'] or 0):,} |"
        )
    markdown.extend(
        [
            "",
            "The complete Recall--QPS measurements are in `recall_qps.tsv`.",
            "",
        ]
    )
    atomic_write(report / "summary.md", "\n".join(markdown))


if __name__ == "__main__":
    main()
