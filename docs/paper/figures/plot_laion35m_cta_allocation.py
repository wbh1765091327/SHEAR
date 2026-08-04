#!/usr/bin/env python3
"""Plot LAION35M initial persistent-CTA allocation timing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import (
    GRID_COLOR_HEX,
    GRID_LINE_WIDTH_PT,
    MATPLOTLIB_RCPARAMS,
    SAVEFIG_KWARGS,
    SINGLE_COLUMN_WIDTH_IN,
)


OUTPUT_BASENAME = "laion35m_cta_allocation"
ALLOCATION_ORDER = ("1:127", "4:124", "8:120", "16:112", "24:104", "32:96", "64:64", "104:24")
FIGURE_HEIGHT_IN = 2.60
BEST_ANNOTATION_OFFSET_PT = (-30, -15)
Y_MIN_S = 120


@dataclass(frozen=True)
class TimingPoint:
    allocation: str
    n_valid: int
    wall_min_s: float
    wall_median_s: float
    wall_max_s: float
    producer_min_s: float
    producer_median_s: float
    producer_max_s: float
    persistent_min_s: float
    persistent_median_s: float
    persistent_max_s: float
    recommended: bool


@dataclass(frozen=True)
class TimingSeries:
    label: str
    minimum: str
    median: str
    maximum: str
    color: str
    marker: str
    linestyle: str


SERIES = (
    TimingSeries("E2E build", "wall_min_s", "wall_median_s", "wall_max_s", "#4D4D4D", "o", "-"),
    TimingSeries(
        "D1+ producer",
        "producer_min_s",
        "producer_median_s",
        "producer_max_s",
        "#0072B2",
        "s",
        "--",
    ),
    TimingSeries(
        "Persistent kernel",
        "persistent_min_s",
        "persistent_median_s",
        "persistent_max_s",
        "#D55E00",
        "D",
        "-.",
    ),
)


def load_points(path: Path) -> list[TimingPoint]:
    points: list[TimingPoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            point = TimingPoint(
                allocation=row["allocation"],
                n_valid=int(row["n_valid"]),
                wall_min_s=float(row["wall_min_s"]),
                wall_median_s=float(row["wall_median_s"]),
                wall_max_s=float(row["wall_max_s"]),
                producer_min_s=float(row["d1plus_producer_min_s"]),
                producer_median_s=float(row["d1plus_producer_median_s"]),
                producer_max_s=float(row["d1plus_producer_max_s"]),
                persistent_min_s=float(row["persistent_event_min_s"]),
                persistent_median_s=float(row["persistent_event_median_s"]),
                persistent_max_s=float(row["persistent_event_max_s"]),
                recommended=row["recommended"] == "true",
            )
            points.append(point)

    if tuple(point.allocation for point in points) != ALLOCATION_ORDER:
        raise ValueError("unexpected CTA-allocation order")
    if any(point.n_valid != 3 for point in points):
        raise ValueError("every allocation must contain three valid repetitions")
    for point in points:
        for minimum, median, maximum in (
            (point.wall_min_s, point.wall_median_s, point.wall_max_s),
            (point.producer_min_s, point.producer_median_s, point.producer_max_s),
            (point.persistent_min_s, point.persistent_median_s, point.persistent_max_s),
        ):
            if not 0 < minimum <= median <= maximum:
                raise ValueError(f"invalid timing range for {point.allocation}")
    if [point.allocation for point in points if point.recommended] != ["32:96"]:
        raise ValueError("unexpected recommended allocation")
    return points


def plot_timings(points: list[TimingPoint], pdf_path: Path, png_path: Path) -> None:
    plt.rcParams.update(MATPLOTLIB_RCPARAMS)
    plt.rcParams.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN))
    x_values = list(range(len(points)))

    for series in SERIES:
        medians = [getattr(point, series.median) for point in points]
        lower = [median - getattr(point, series.minimum) for point, median in zip(points, medians, strict=True)]
        upper = [getattr(point, series.maximum) - median for point, median in zip(points, medians, strict=True)]
        axis.errorbar(
            x_values,
            medians,
            yerr=[lower, upper],
            label=series.label,
            color=series.color,
            marker=series.marker,
            linestyle=series.linestyle,
            linewidth=1.0,
            markersize=3.4,
            markeredgewidth=0.6,
            capsize=1.8,
            capthick=0.55,
            elinewidth=0.55,
            zorder=3,
        )

    persistent = SERIES[-1]
    default_point = points[0]
    best_index = next(index for index, point in enumerate(points) if point.recommended)
    best_point = points[best_index]
    axis.scatter(
        [best_index],
        [best_point.persistent_median_s],
        s=38,
        facecolors="none",
        edgecolors=persistent.color,
        linewidths=0.85,
        zorder=5,
    )
    axis.annotate(
        "default",
        (0, default_point.persistent_median_s),
        xytext=(4, 13),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#6A6A6A", "linewidth": 0.55},
        fontsize=7.2,
        color="black",
    )
    axis.annotate(
        "min. persistent",
        (best_index, best_point.persistent_median_s),
        xytext=BEST_ANNOTATION_OFFSET_PT,
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": persistent.color, "linewidth": 0.55},
        fontsize=7.2,
        color="black",
    )

    axis.set_xlabel("Initial CTA allocation (Partition:Top-k)")
    axis.set_ylabel("Time (s)")
    axis.set_xticks(x_values, [point.allocation for point in points], rotation=34, ha="right", rotation_mode="anchor")
    axis.set_xlim(-0.35, len(points) - 0.65)
    axis.set_ylim(Y_MIN_S, 250)
    axis.set_yticks((120, 140, 160, 180, 200, 220, 240))
    axis.grid(True, axis="y", color=GRID_COLOR_HEX, linestyle=":", linewidth=GRID_LINE_WIDTH_PT)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.30, top=0.81)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, **SAVEFIG_KWARGS)
    fig.savefig(png_path, dpi=300, **SAVEFIG_KWARGS)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=output_dir / f"{OUTPUT_BASENAME}.tsv")
    parser.add_argument("--pdf", type=Path, default=output_dir / f"{OUTPUT_BASENAME}.pdf")
    parser.add_argument("--png", type=Path, default=output_dir / f"{OUTPUT_BASENAME}.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_timings(load_points(args.data), args.pdf, args.png)


if __name__ == "__main__":
    main()
