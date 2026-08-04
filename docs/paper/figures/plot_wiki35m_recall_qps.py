#!/usr/bin/env python3
"""Plot the Wiki35M Recall@10--QPS trade-off from the reported search sweeps."""

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
        (25, 0.78328, 6838),
        (50, 0.85719, 6002),
        (100, 0.91082, 4992),
        (200, 0.94351, 3705),
        (400, 0.96354, 2609),
    ),
    "CAGRA NN-Descent": (
        (25, 0.77213, 6152),
        (50, 0.86508, 5249),
        (100, 0.92387, 4210),
        (200, 0.95529, 3096),
        (400, 0.97203, 2053),
    ),
    "Tagore-CAGRA": (
        (25, 0.79109, 6158),
        (50, 0.87265, 5363),
        (100, 0.92399, 4377),
        (200, 0.95603, 3286),
        (400, 0.97245, 2232),
    ),
    "Tagore-Vamana": (
        (25, 0.75250, 5700),
        (50, 0.84204, 5024),
        (100, 0.90103, 4201),
        (200, 0.93843, 3279),
        (400, 0.96099, 2333),
    ),
    "IVF-RaBitQ": (
        (28, 0.84682, 2270),
        (42, 0.87734, 2059),
        (64, 0.90402, 1761),
        (128, 0.93685, 1284),
        (256, 0.95843, 822),
        (512, 0.97213, 468),
    ),
}

PARAMETER_SWEEPS = {
    "SHEAR": (25, 50, 100, 200, 400),
    "CAGRA NN-Descent": (25, 50, 100, 200, 400),
    "Tagore-CAGRA": (25, 50, 100, 200, 400),
    "Tagore-Vamana": (25, 50, 100, 200, 400),
    "IVF-RaBitQ": (28, 42, 64, 128, 256, 512),
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
) -> float:
    for left, right in zip(points, points[1:]):
        if left.recall <= target <= right.recall:
            fraction = (target - left.recall) / (right.recall - left.recall)
            return left.qps + fraction * (right.qps - left.qps)
    raise ValueError(f"Recall@10={target:.2f} is outside the measured range")


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
        axis.scatter(
            [TARGET_RECALL * 100],
            [interpolate_qps(points)],
            s=30,
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.0,
            zorder=5,
        )

    axis.set_xlim(74, 98.5)
    axis.set_xticks((75, 80, 85, 90, 95))
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
    axis.set_title("Wiki35M", pad=3)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.72)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=output_dir / "wiki35m_recall_qps_tradeoff.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=output_dir / "wiki35m_recall_qps_tradeoff.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_tradeoff(load_points(), args.pdf, args.png)


if __name__ == "__main__":
    main()
