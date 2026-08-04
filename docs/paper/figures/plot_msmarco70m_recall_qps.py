#!/usr/bin/env python3
"""Plot the MSMARCO70M Recall@10--QPS trade-off from the reported search sweeps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TARGET_RECALL = 0.95


@dataclass(frozen=True)
class SearchPoint:
    search_parameter: int
    recall: float
    qps: float


METHOD_POINTS = {
    "SHEAR": (
        (25, 0.85351, 7463.38),
        (50, 0.91366, 6625.97),
        (100, 0.94743, 5334.78),
        (200, 0.96812, 4139.46),
        (400, 0.98110, 2766.20),
    ),
    "CAGRA NN-Descent": (
        (25, 0.71955, 6540.71),
        (50, 0.81622, 5636.92),
        (100, 0.87758, 4384.31),
        (200, 0.91531, 3119.29),
        (400, 0.94015, 2062.23),
    ),
    "Tagore-CAGRA": (
        (25, 0.82072, 6873.80),
        (50, 0.89462, 6131.85),
        (100, 0.93681, 5116.73),
        (200, 0.96222, 3891.39),
        (400, 0.97565, 2659.55),
    ),
    "Tagore-Vamana": (
        (25, 0.85764, 7334.37),
        (50, 0.92255, 6588.23),
        (100, 0.95422, 5557.24),
        (200, 0.97203, 4349.64),
        (400, 0.98241, 3006.18),
    ),
    "IVF-RaBitQ": (
        (12, 0.83950, 3158.93),
        (18, 0.87010, 2903.76),
        (28, 0.89861, 2678.64),
        (42, 0.91991, 2397.61),
        (64, 0.93825, 2078.01),
        (128, 0.96233, 1465.50),
        (256, 0.97708, 919.25),
        (512, 0.98481, 555.79),
    ),
}

PARAMETER_SWEEPS = {
    "SHEAR": (25, 50, 100, 200, 400),
    "CAGRA NN-Descent": (25, 50, 100, 200, 400),
    "Tagore-CAGRA": (25, 50, 100, 200, 400),
    "Tagore-Vamana": (25, 50, 100, 200, 400),
    "IVF-RaBitQ": (12, 18, 28, 42, 64, 128, 256, 512),
}

STYLES = {
    "SHEAR": dict(
        color="#D55E00", marker="o", linestyle="-", markerfacecolor="#D55E00"
    ),
    "CAGRA NN-Descent": dict(color="#0072B2", marker="s", linestyle="--"),
    "Tagore-CAGRA": dict(color="#009E73", marker="^", linestyle="-."),
    "Tagore-Vamana": dict(color="#CC79A7", marker="D", linestyle=":"),
    "IVF-RaBitQ": dict(
        color="#4D4D4D", marker="X", linestyle=(0, (3, 1, 1, 1))
    ),
}


def load_points() -> dict[str, tuple[SearchPoint, ...]]:
    results = {
        method: tuple(SearchPoint(*values) for values in rows)
        for method, rows in METHOD_POINTS.items()
    }
    if set(results) != set(STYLES) or set(results) != set(PARAMETER_SWEEPS):
        raise ValueError("the data, style, and parameter-sweep method sets do not match")
    for method, points in results.items():
        if (
            tuple(point.search_parameter for point in points)
            != PARAMETER_SWEEPS[method]
        ):
            raise ValueError(f"incomplete search-parameter sweep: {method}")
        if any(right.recall <= left.recall for left, right in zip(points, points[1:])):
            raise ValueError(f"recall is not strictly increasing: {method}")
        if any(right.qps >= left.qps for left, right in zip(points, points[1:])):
            raise ValueError(f"QPS is not strictly decreasing: {method}")
    return results


def interpolate_qps(
    points: tuple[SearchPoint, ...], target: float = TARGET_RECALL
) -> float | None:
    for left, right in zip(points, points[1:]):
        if left.recall <= target <= right.recall:
            fraction = (target - left.recall) / (right.recall - left.recall)
            return left.qps + fraction * (right.qps - left.qps)
    return None


def plot_tradeoff(
    points_by_method: dict[str, tuple[SearchPoint, ...]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(3.35, 3.35))

    for method, points in points_by_method.items():
        style = STYLES[method]
        axis.plot(
            [point.recall * 100 for point in points],
            [point.qps for point in points],
            label=method,
            linewidth=1.2,
            markersize=4.0,
            markeredgewidth=0.7,
            alpha=0.96,
            **style,
        )
        qps_at_target = interpolate_qps(points)
        if qps_at_target is not None:
            axis.scatter(
                [TARGET_RECALL * 100],
                [qps_at_target],
                s=30,
                marker=style["marker"],
                facecolors="white",
                edgecolors=style["color"],
                linewidths=1.0,
                zorder=5,
            )

    axis.set_xlim(70, 99.5)
    axis.set_xticks((70, 75, 80, 85, 90, 95))
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.axvline(
        TARGET_RECALL * 100,
        color="#555555",
        linestyle="--",
        linewidth=0.7,
        alpha=0.8,
    )
    axis.grid(True, color="#B8B8B8", linestyle=":", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.set_xlabel("Recall@10 (%)")
    axis.set_ylabel("QPS")
    axis.set_title("MSMARCO70M", pad=3)

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0,
        borderpad=0,
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.80)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs = {"bbox_inches": "tight", "pad_inches": 0.01}
    fig.savefig(pdf_path, **savefig_kwargs)
    fig.savefig(png_path, dpi=300, **savefig_kwargs)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=output_dir / "msmarco70m_recall_qps_tradeoff.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=output_dir / "msmarco70m_recall_qps_tradeoff.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_tradeoff(load_points(), args.pdf, args.png)


if __name__ == "__main__":
    main()
