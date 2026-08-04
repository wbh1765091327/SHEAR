#!/usr/bin/env python3
"""Plot and summarize the accepted four-dataset, five-method comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from paper_figure_style import (
    GRID_COLOR_HEX,
    GRID_LINE_WIDTH_PT,
    MATPLOTLIB_RCPARAMS,
    PANEL_TITLE_PAD_PT,
    SAVEFIG_KWARGS,
)


REPORT_ROOT = Path(
    "/data_staging/wbh/five_method_four_dataset/runs/"
    "formal-monitor-fix-20260723T155751Z"
)
CONSTRUCTION_SUMMARY = REPORT_ROOT / "construction_summary.tsv"
SEARCH_SUMMARY = REPORT_ROOT / "search_summary.tsv"
DATASETS = ("deep100m", "laion100m", "msmarco70m", "wiki35m")
METHODS = (
    "shear",
    "cagra_nn_descent",
    "tagore_cagra",
    "tagore_vamana",
    "ivf_rabitq",
)
GRAPH_METHODS = METHODS[:-1]
MIN_RECALL = 0.85
TARGET_RECALL = 0.95
QPS_DISPLAY_SCALE = 1000.0
PANEL_TITLE_Y = 1.02
DOUBLE_COLUMN_WIDTH_IN = 6.9
FIGURE_HEIGHT_IN = 17 / 12
PANEL_COLUMNS = 4
FIGURE_BOTTOM_MARGIN_IN = 37 / 120
FIGURE_TOP_MARGIN_IN = 277 / 1200
PANEL_HEIGHT_IN = (
    FIGURE_HEIGHT_IN - FIGURE_BOTTOM_MARGIN_IN - FIGURE_TOP_MARGIN_IN
)
OUTER_TEXT_OFFSET_IN = 1 / 60
PANEL_WIDTH_REFERENCE_LEFT = 0.05
PANEL_WIDTH_REFERENCE_WSPACE = 0.36
FOUR_PANEL_AXIS_LABEL_SIZE_PT = 7.5
FOUR_PANEL_TITLE_SIZE_PT = 8.0
FOUR_PANEL_TICK_SIZE_PT = 7.0
FOUR_PANEL_LEGEND_SIZE_PT = 7.0
GIB = 1024**3

DATASET_LABELS = {
    "deep100m": "DEEP100M",
    "laion100m": "LAION100M",
    "msmarco70m": "MSMARCO70M",
    "wiki35m": "Wiki35M",
}
METHOD_LABELS = {
    "shear": "SHEAR",
    "cagra_nn_descent": "CAGRA-NND",
    "tagore_cagra": "Tagore-CAGRA",
    "tagore_vamana": "Tagore-Vamana",
    "ivf_rabitq": "IVF-RaBitQ",
}
STYLES = {
    "shear": dict(
        color="#D55E00", marker="o", linestyle="-", markerfacecolor="#D55E00"
    ),
    "cagra_nn_descent": dict(color="#0072B2", marker="s", linestyle="--"),
    "tagore_cagra": dict(color="#009E73", marker="^", linestyle="-."),
    "tagore_vamana": dict(color="#CC79A7", marker="D", linestyle=":"),
    "ivf_rabitq": dict(color="#4D4D4D", marker="X", linestyle=(0, (4, 1.5))),
}


@dataclass(frozen=True)
class SearchPoint:
    parameter_kind: str
    parameter_value: int
    recall: float
    qps: float
    qps_stdev: float


@dataclass(frozen=True)
class BuildPoint:
    prep_s: float
    wall_s: float
    rss_gib: float
    index_gib: float


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_elapsed_seconds(path: Path) -> float:
    match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"cannot parse elapsed wall time from {path}")
    parts = [float(part) for part in match.group(1).split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unexpected elapsed wall-time format in {path}")


def graph_payload(dataset: str, method: str) -> Path:
    return REPORT_ROOT / "hot_path" / dataset / method / "run_1" / "payload"


def load_ivf_build_rows(source_root: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(source_root / "build_summary.tsv")
    accepted = {row["dataset"]: row for row in rows if row["status"] == "accepted"}
    if set(accepted) != set(DATASETS):
        raise ValueError("reused IVF source does not contain four accepted build rows")
    return accepted


def prep_seconds(dataset: str, method: str, ivf_row: dict[str, str] | None) -> float:
    if method == "ivf_rabitq":
        if ivf_row is None:
            raise ValueError(f"missing IVF build row for {dataset}")
        return float(ivf_row["gpu_build_seconds"])

    payload = graph_payload(dataset, method)
    if method == "shear":
        profile = read_json(payload / "profile.json")
        gpu_subtree = profile["gpu_subtree"]
        if not isinstance(gpu_subtree, dict):
            raise ValueError(f"invalid SHEAR profile for {dataset}")
        return float(gpu_subtree["d0_wall_ms"]) / 1000

    if method in {"cagra_nn_descent", "tagore_cagra"}:
        profile = read_json(payload / "profile.json")
        fields = (
            "partition_generate_ms",
            "partition_read_materialize_ms",
            "partition_pack_ms",
            "upload_h2d_ms",
        )
        return sum(float(profile[field]) for field in fields) / 1000

    if method == "tagore_vamana":
        return parse_elapsed_seconds(
            payload / "partition.time.log"
        ) + parse_elapsed_seconds(payload / "conversion.time.log")

    raise ValueError(f"unknown method: {method}")


def load_builds() -> dict[tuple[str, str], BuildPoint]:
    rows = read_tsv(CONSTRUCTION_SUMMARY)
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    actual = {(row["dataset"], row["method"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("construction summary does not contain the expected 4 x 5 matrix")
    if any(row["status"] != "accepted" for row in rows):
        raise ValueError("construction summary contains a non-accepted row")

    ivf_source_roots = {
        Path(row["source_root"])
        for row in rows
        if row["method"] == "ivf_rabitq"
    }
    if len(ivf_source_roots) != 1:
        raise ValueError("IVF construction rows do not share one provenance-bound source")
    ivf_build_rows = load_ivf_build_rows(ivf_source_roots.pop())

    builds: dict[tuple[str, str], BuildPoint] = {}
    for row in rows:
        dataset, method = row["dataset"], row["method"]
        builds[(dataset, method)] = BuildPoint(
            prep_s=prep_seconds(
                dataset,
                method,
                ivf_build_rows[dataset] if method == "ivf_rabitq" else None,
            ),
            wall_s=float(row["external_wall_seconds"]),
            rss_gib=int(row["peak_dram_bytes"]) / GIB,
            index_gib=int(row["index_bytes"]) / GIB,
        )
    return builds


def load_searches() -> dict[tuple[str, str], list[SearchPoint]]:
    searches: dict[tuple[str, str], list[SearchPoint]] = {}
    for row in read_tsv(SEARCH_SUMMARY):
        dataset, method = row["dataset"], row["method"]
        if dataset not in DATASETS or method not in METHODS:
            raise ValueError(f"unexpected search row: {dataset}/{method}")
        if int(row["repeat_count"]) != 2:
            raise ValueError(f"unexpected repeat count: {dataset}/{method}")
        expected_kind = "nprobe" if method == "ivf_rabitq" else "L"
        if row["parameter_kind"] != expected_kind:
            raise ValueError(
                f"unexpected parameter kind: {dataset}/{method}/{row['parameter_kind']}"
            )
        searches.setdefault((dataset, method), []).append(
            SearchPoint(
                parameter_kind=row["parameter_kind"],
                parameter_value=int(row["parameter_value"]),
                recall=float(row["recall_at_10"]),
                qps=float(row["qps_mean"]),
                qps_stdev=float(row["qps_sample_std"]),
            )
        )

    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if set(searches) != expected:
        raise ValueError("search summary does not contain the expected 4 x 5 matrix")
    for key, points in searches.items():
        points.sort(key=lambda point: point.parameter_value)
        if any(right.recall <= left.recall for left, right in zip(points, points[1:])):
            raise ValueError(f"recall is not strictly increasing: {key}")
    return searches


def interpolate_qps(
    points: list[SearchPoint], target: float = TARGET_RECALL
) -> tuple[float, SearchPoint, SearchPoint] | None:
    for left, right in zip(points, points[1:]):
        if left.recall <= target <= right.recall:
            fraction = (target - left.recall) / (right.recall - left.recall)
            qps = left.qps + fraction * (right.qps - left.qps)
            return qps, left, right
    return None


def format_qps_thousands(value: float, _: int) -> str:
    return f"{value / QPS_DISPLAY_SCALE:g}"


def print_table(
    builds: dict[tuple[str, str], BuildPoint],
    searches: dict[tuple[str, str], list[SearchPoint]],
) -> None:
    print(
        "dataset\tmethod\tprep_min\te2e_min\tpeak_dram_gib\tindex_gib\t"
        "qps_at_recall_0.95\tbracket"
    )
    for dataset in DATASETS:
        for method in METHODS:
            build = builds[(dataset, method)]
            interpolation = interpolate_qps(searches[(dataset, method)])
            if interpolation is None:
                qps_text = "--"
                bracket = f"not-covered(max={searches[(dataset, method)][-1].recall:.5f})"
            else:
                qps, left, right = interpolation
                qps_text = f"{qps:.6f}"
                bracket = (
                    f"{left.parameter_kind}="
                    f"{left.parameter_value}-{right.parameter_value}"
                )
            print(
                f"{DATASET_LABELS[dataset]}\t{METHOD_LABELS[method]}\t"
                f"{build.prep_s / 60:.6f}\t{build.wall_s / 60:.6f}\t"
                f"{build.rss_gib:.6f}\t{build.index_gib:.6f}\t"
                f"{qps_text}\t{bracket}"
            )


def plot_tradeoff(
    searches: dict[tuple[str, str], list[SearchPoint]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt.rcParams.update(MATPLOTLIB_RCPARAMS)
    plt.rcParams.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.titlesize": FOUR_PANEL_TITLE_SIZE_PT,
            "xtick.color": "black",
            "xtick.labelsize": FOUR_PANEL_TICK_SIZE_PT,
            "ytick.color": "black",
            "ytick.labelsize": FOUR_PANEL_TICK_SIZE_PT,
        }
    )
    fig, axes = plt.subplots(
        1,
        PANEL_COLUMNS,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
    )
    legend_handles = []
    panel_names = ("(a)", "(b)", "(c)", "(d)")

    for axis, dataset, panel_name in zip(axes, DATASETS, panel_names):
        for spine in axis.spines.values():
            spine.set_visible(True)
        for method in METHODS:
            points = searches[(dataset, method)]
            style = STYLES[method]
            handle = axis.errorbar(
                [point.recall * 100 for point in points],
                [point.qps for point in points],
                yerr=[point.qps_stdev for point in points],
                label=METHOD_LABELS[method],
                linewidth=1.05,
                markersize=3.2,
                markeredgewidth=0.65,
                capsize=1.0,
                elinewidth=0.55,
                alpha=0.96,
                **style,
            )
            interpolation = interpolate_qps(points)
            if interpolation is not None:
                qps_at_target, _, _ = interpolation
                axis.scatter(
                    [TARGET_RECALL * 100],
                    [qps_at_target],
                    s=22,
                    marker=style["marker"],
                    facecolors="white",
                    edgecolors=style["color"],
                    linewidths=0.9,
                    zorder=5,
                )
            if dataset == DATASETS[0]:
                legend_handles.append(handle.lines[0])

        axis.set_xlim(MIN_RECALL * 100, 100.0)
        axis.set_xticks((85, 90, 95, 100))
        axis.set_ylim(bottom=0)
        axis.axvline(
            TARGET_RECALL * 100,
            color="#555555",
            linestyle="--",
            linewidth=0.65,
            alpha=0.8,
        )
        axis.grid(
            True,
            color=GRID_COLOR_HEX,
            linestyle=":",
            linewidth=GRID_LINE_WIDTH_PT,
        )
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", pad=1.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axis.yaxis.set_major_formatter(FuncFormatter(format_qps_thousands))
        axis.set_title(
            f"{panel_name} {DATASET_LABELS[dataset]}",
            y=PANEL_TITLE_Y,
            pad=PANEL_TITLE_PAD_PT,
        )

    layout_left = 0.043
    layout_right = 0.985
    reference_panel_width = (layout_right - PANEL_WIDTH_REFERENCE_LEFT) / (
        PANEL_COLUMNS + (PANEL_COLUMNS - 1) * PANEL_WIDTH_REFERENCE_WSPACE
    )
    panel_wspace = (
        (layout_right - layout_left) / reference_panel_width - PANEL_COLUMNS
    ) / (PANEL_COLUMNS - 1)
    layout_bottom = FIGURE_BOTTOM_MARGIN_IN / FIGURE_HEIGHT_IN
    layout_top = 1 - FIGURE_TOP_MARGIN_IN / FIGURE_HEIGHT_IN
    figure_x_center = (layout_left + layout_right) / 2
    legend_anchor_y = 1 + OUTER_TEXT_OFFSET_IN / FIGURE_HEIGHT_IN
    xlabel_y = OUTER_TEXT_OFFSET_IN / FIGURE_HEIGHT_IN
    fig.legend(
        legend_handles,
        [METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(figure_x_center, legend_anchor_y),
        columnspacing=0.7,
        handlelength=1.45,
        handletextpad=0.3,
        borderaxespad=0.0,
        fontsize=FOUR_PANEL_LEGEND_SIZE_PT,
    )
    fig.supxlabel(
        "Recall@10 (%)",
        x=figure_x_center,
        fontsize=FOUR_PANEL_AXIS_LABEL_SIZE_PT,
        y=xlabel_y,
    )
    fig.supylabel(
        r"QPS ($\times 10^3$)",
        fontsize=FOUR_PANEL_AXIS_LABEL_SIZE_PT,
        x=0.003,
    )
    fig.subplots_adjust(
        left=layout_left,
        right=layout_right,
        bottom=layout_bottom,
        top=layout_top,
        wspace=panel_wspace,
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, **SAVEFIG_KWARGS)
    fig.savefig(png_path, dpi=300, **SAVEFIG_KWARGS)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=default_dir / "four_dataset_recall_qps_tradeoff.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=default_dir / "four_dataset_recall_qps_tradeoff.png",
    )
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builds = load_builds()
    searches = load_searches()
    if args.print_table:
        print_table(builds, searches)
    plot_tradeoff(searches, args.pdf, args.png)


if __name__ == "__main__":
    main()
    MATPLOTLIB_RCPARAMS,
