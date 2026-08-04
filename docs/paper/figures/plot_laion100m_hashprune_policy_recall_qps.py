#!/usr/bin/env python3
"""Plot the LAION100M HashPrune policy Recall@10--QPS sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

from paper_figure_style import (
    GRID_COLOR_HEX,
    GRID_LINE_WIDTH_PT,
    MATPLOTLIB_RCPARAMS,
    PANEL_TITLE_PAD_PT,
    SAVEFIG_KWARGS,
)


POLICY_ORDER = ("Plain", "DistanceFirst-Q3", "HardTier-Q3", "SoftTier-Q3")
LAION_EXPECTED_L = (10, 16, 25, 32, 50, 64, 80, 100, 128, 160, 200, 300, 400)
STYLES = {
    "Plain": dict(color="#4D4D4D", marker="o", linestyle=":"),
    "DistanceFirst-Q3": dict(color="#0072B2", marker="s", linestyle="--"),
    "HardTier-Q3": dict(color="#009E73", marker="^", linestyle="-."),
    "SoftTier-Q3": dict(color="#D55E00", marker="D", linestyle="-"),
}
TARGET_PANEL_WSPACE = 0.34
REFERENCE_PANEL_WIDTH_IN = 6.9 * (0.985 - 0.05) / (4 + 3 * 0.36)
REFERENCE_PANEL_HEIGHT_PT = (17 / 12 - 37 / 120 - 277 / 1200) * 72.0
FIGURE_LEFT_MARGIN_PT = 30.0
FIGURE_RIGHT_MARGIN_PT = 5.0
FIGURE_TOP_MARGIN_PT = 23.5
FIGURE_BOTTOM_MARGIN_PT = 32.0
FIGURE_WIDTH_PT = (
    FIGURE_LEFT_MARGIN_PT
    + REFERENCE_PANEL_WIDTH_IN * (2 + TARGET_PANEL_WSPACE) * 72.0
    + FIGURE_RIGHT_MARGIN_PT
)
FIGURE_HEIGHT_PT = 108.2
FIGURE_WIDTH_IN = FIGURE_WIDTH_PT / 72.0
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_PT / 72.0
PANEL_TITLE_Y = 1.02
HASHPRUNE_AXIS_LABEL_SIZE_PT = 9.2
HASHPRUNE_PANEL_TITLE_SIZE_PT = 9.4
HASHPRUNE_TICK_SIZE_PT = 8.6
HASHPRUNE_LEGEND_SIZE_PT = 7.8
HASHPRUNE_Y_LABEL_PAD_PT = 0.55
HASHPRUNE_X_TICK_PAD_PT = 0.0
PLOT_LEFT_PT = FIGURE_LEFT_MARGIN_PT + 5.5
PLOT_RIGHT_PT = FIGURE_WIDTH_PT - FIGURE_RIGHT_MARGIN_PT
PLOT_TOP_PT = FIGURE_TOP_MARGIN_PT
PLOT_BOTTOM_PT = FIGURE_HEIGHT_PT - FIGURE_BOTTOM_MARGIN_PT
X_LABEL_BOTTOM_OFFSET_PT = 6.0
QPS_DISPLAY_SCALE = 1000.0


@dataclass(frozen=True)
class SearchPoint:
    policy: str
    l_value: int
    recall: float
    qps: float


def load_points(
    path: Path,
    policy_order: tuple[str, ...],
    expected_l: tuple[int, ...],
) -> dict[str, list[SearchPoint]]:
    grouped = {policy: [] for policy in policy_order}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            point = SearchPoint(
                policy=row["policy"],
                l_value=int(row["L"]),
                recall=float(row["recall"]),
                qps=float(row["qps"]),
            )
            if point.policy not in grouped:
                raise ValueError(f"unexpected policy: {point.policy}")
            grouped[point.policy].append(point)

    for policy, points in grouped.items():
        if tuple(point.l_value for point in points) != expected_l:
            raise ValueError(f"unexpected L sweep for {policy}")
        if any(right.recall <= left.recall for left, right in zip(points, points[1:])):
            raise ValueError(f"Recall must increase with L for {policy}")
        if any(right.qps >= left.qps for left, right in zip(points, points[1:])):
            raise ValueError(f"QPS must decrease with L for {policy}")
    return grouped


def draw_policy(axis: plt.Axes, policy: str, points: list[SearchPoint]) -> None:
    style = STYLES[policy]
    emphasized = policy == "SoftTier-Q3"
    axis.plot(
        [point.recall * 100 for point in points],
        [point.qps for point in points],
        label="TierLift" if policy == "SoftTier-Q3" else policy.removesuffix("-Q3"),
        linewidth=1.15 if emphasized else 0.75,
        markersize=2.9 if emphasized else 2.5,
        markerfacecolor=style["color"] if emphasized else "white",
        markeredgewidth=0.55,
        zorder=4 if emphasized else 3,
        **style,
    )


def format_qps_thousands(value: float, _: int) -> str:
    return f"{value / QPS_DISPLAY_SCALE:g}"


def plot_tradeoff(
    laion_points: dict[str, list[SearchPoint]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt.rcParams.update(MATPLOTLIB_RCPARAMS)
    plt.rcParams.update(
        {
            "font.size": HASHPRUNE_TICK_SIZE_PT,
            "axes.labelsize": HASHPRUNE_AXIS_LABEL_SIZE_PT,
            "axes.titlesize": HASHPRUNE_PANEL_TITLE_SIZE_PT,
            "legend.fontsize": HASHPRUNE_LEGEND_SIZE_PT,
            "xtick.labelsize": HASHPRUNE_TICK_SIZE_PT,
            "ytick.labelsize": HASHPRUNE_TICK_SIZE_PT,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    fig, (full_axis, high_axis) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
    )

    for policy in POLICY_ORDER:
        draw_policy(high_axis, policy, laion_points[policy])
        draw_policy(full_axis, policy, laion_points[policy])

    for axis in (high_axis, full_axis):
        for spine in axis.spines.values():
            spine.set_visible(True)
        axis.grid(
            True,
            color=GRID_COLOR_HEX,
            linestyle=":",
            linewidth=GRID_LINE_WIDTH_PT,
        )
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", pad=HASHPRUNE_X_TICK_PAD_PT)
        axis.yaxis.set_major_formatter(FuncFormatter(format_qps_thousands))
    full_axis.set_ylabel(
        r"QPS ($\times 10^3$)",
        labelpad=HASHPRUNE_Y_LABEL_PAD_PT,
        fontsize=HASHPRUNE_AXIS_LABEL_SIZE_PT,
    )

    high_axis.set_xlim(90.0, 97.2)
    high_axis.set_ylim(2000, 3800)
    high_axis.set_xticks((90, 92, 94, 96))
    high_axis.set_yticks((2000, 2400, 2800, 3200, 3600))
    high_axis.set_title(
        "(b) High recall",
        y=PANEL_TITLE_Y,
        pad=PANEL_TITLE_PAD_PT,
        fontsize=HASHPRUNE_PANEL_TITLE_SIZE_PT,
    )

    full_axis.set_xlim(72.0, 97.2)
    full_axis.set_ylim(2000, 4600)
    full_axis.set_xticks((75, 85, 95))
    full_axis.set_yticks((2000, 2800, 3600, 4400))
    full_axis.set_title(
        "(a) Full range",
        y=PANEL_TITLE_Y,
        pad=PANEL_TITLE_PAD_PT,
        fontsize=HASHPRUNE_PANEL_TITLE_SIZE_PT,
    )

    layout_left = PLOT_LEFT_PT / FIGURE_WIDTH_PT
    layout_right = PLOT_RIGHT_PT / FIGURE_WIDTH_PT
    layout_bottom = (FIGURE_HEIGHT_PT - PLOT_BOTTOM_PT) / FIGURE_HEIGHT_PT
    layout_top = 1.0 - PLOT_TOP_PT / FIGURE_HEIGHT_PT
    figure_x_center = (layout_left + layout_right) / 2
    xlabel_y = X_LABEL_BOTTOM_OFFSET_PT / FIGURE_HEIGHT_PT
    legend_anchor_y = 1.0 - 0.5 / FIGURE_HEIGHT_PT
    handles, labels = high_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(POLICY_ORDER),
        frameon=False,
        bbox_to_anchor=(figure_x_center, legend_anchor_y),
        columnspacing=0.6,
        handlelength=1.15,
        handletextpad=0.25,
        borderaxespad=0.0,
        fontsize=HASHPRUNE_LEGEND_SIZE_PT,
    )
    fig.supxlabel(
        "Recall@10 (%)",
        x=figure_x_center,
        y=xlabel_y,
        fontsize=HASHPRUNE_AXIS_LABEL_SIZE_PT,
    )
    fig.subplots_adjust(
        left=layout_left,
        right=layout_right,
        bottom=layout_bottom,
        top=layout_top,
        wspace=TARGET_PANEL_WSPACE,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    output_bbox = Bbox.from_bounds(
        0.0,
        0.0,
        FIGURE_WIDTH_IN,
        FIGURE_HEIGHT_IN,
    )
    savefig_kwargs = dict(SAVEFIG_KWARGS)
    savefig_kwargs["bbox_inches"] = output_bbox
    fig.savefig(pdf_path, **savefig_kwargs)
    fig.savefig(png_path, dpi=300, **savefig_kwargs)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=output_dir / "laion100m_hashprune_policy_recall_qps.tsv",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=output_dir / "laion100m_hashprune_policy_recall_qps.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=output_dir / "laion100m_hashprune_policy_recall_qps.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_tradeoff(
        load_points(args.data, POLICY_ORDER, LAION_EXPECTED_L),
        args.pdf,
        args.png,
    )


if __name__ == "__main__":
    main()
    MATPLOTLIB_RCPARAMS,
