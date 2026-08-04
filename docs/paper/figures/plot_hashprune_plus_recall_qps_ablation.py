#!/usr/bin/env python3
"""Plot the two-dataset HashPrune+ Recall@10--QPS ablation.

Each QPS value is the mean of three measured searches. The source table does
not report per-run dispersion, so the figure intentionally omits error bars.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


POLICY_ORDER = (
    "Plain",
    "HashSlot-1",
    "DistanceFirst-Q3",
    "HashPrune+",
    "HardTier-Q3",
)

POLICY_VALUES = {
    "LAION100M": {
        "Plain": (0.90567, 2384.06, 0.94786, 1286.57),
        "HashSlot-1": (0.91324, 2335.52, 0.95685, 1260.65),
        "DistanceFirst-Q3": (0.92221, 2297.35, 0.96408, 1250.99),
        "HashPrune+": (0.92024, 2359.02, 0.96413, 1259.72),
        "HardTier-Q3": (0.92132, 2343.59, 0.96385, 1264.38),
    },
    "Wiki35M": {
        "Plain": (0.85985, 2573.36, 0.94835, 1076.94),
        "HashSlot-1": (0.86504, 2394.86, 0.95017, 987.97),
        "DistanceFirst-Q3": (0.86497, 2364.53, 0.95109, 978.45),
        "HashPrune+": (0.86481, 2376.11, 0.94963, 979.56),
        "HardTier-Q3": (0.86366, 2367.13, 0.95083, 976.20),
    },
}

STYLES = {
    "Plain": dict(color="#4D4D4D", linestyle="-"),
    "HashSlot-1": dict(color="#0072B2", linestyle="--"),
    "DistanceFirst-Q3": dict(color="#009E73", linestyle="-."),
    "HashPrune+": dict(color="#D55E00", linestyle="-"),
    "HardTier-Q3": dict(color="#CC79A7", linestyle=":"),
}

PANEL_SPECS = {
    "LAION100M": dict(title="(a) LAION100M", xlim=(90.2, 96.7), xticks=(91, 92, 93, 94, 95, 96)),
    "Wiki35M": dict(title="(b) Wiki35M", xlim=(85.5, 95.5), xticks=(86, 88, 90, 92, 94)),
}


@dataclass(frozen=True)
class PolicyResult:
    l50_recall: float
    l50_qps: float
    l200_recall: float
    l200_qps: float


def load_results() -> dict[str, dict[str, PolicyResult]]:
    results = {
        dataset: {
            policy: PolicyResult(*values)
            for policy, values in policy_values.items()
        }
        for dataset, policy_values in POLICY_VALUES.items()
    }
    if tuple(results) != tuple(PANEL_SPECS):
        raise ValueError("dataset order does not match panel order")
    if set(STYLES) != set(POLICY_ORDER):
        raise ValueError("policy styles do not match the expected policy set")
    for dataset, policies in results.items():
        if tuple(policies) != POLICY_ORDER:
            raise ValueError(f"unexpected policy order: {dataset}")
        for policy, result in policies.items():
            if result.l200_recall <= result.l50_recall:
                raise ValueError(f"Recall must increase from L=50 to L=200: {dataset}/{policy}")
            if result.l200_qps >= result.l50_qps:
                raise ValueError(f"QPS must decrease from L=50 to L=200: {dataset}/{policy}")
    return results


def plot_tradeoff(
    results: dict[str, dict[str, PolicyResult]],
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
            "axes.titlesize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=True)

    for axis, (dataset, policies) in zip(axes, results.items(), strict=True):
        for policy in POLICY_ORDER:
            result = policies[policy]
            style = STYLES[policy]
            emphasized = policy == "HashPrune+"
            linewidth = 1.8 if emphasized else 1.05
            marker_size = 34 if emphasized else 26
            zorder = 4 if emphasized else 3
            axis.plot(
                [result.l50_recall * 100, result.l200_recall * 100],
                [result.l50_qps, result.l200_qps],
                linewidth=linewidth,
                alpha=0.98,
                zorder=zorder,
                **style,
            )
            axis.scatter(
                [result.l50_recall * 100],
                [result.l50_qps],
                s=marker_size,
                marker="o",
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.55,
                zorder=zorder + 1,
            )
            axis.scatter(
                [result.l200_recall * 100],
                [result.l200_qps],
                s=marker_size,
                marker="s",
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.55,
                zorder=zorder + 1,
            )

        spec = PANEL_SPECS[dataset]
        axis.set_xlim(*spec["xlim"])
        axis.set_xticks(spec["xticks"])
        axis.set_ylim(0, 2700)
        axis.set_yticks((0, 500, 1000, 1500, 2000, 2500))
        axis.grid(True, color="#B8B8B8", linestyle=":", linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_title(spec["title"], pad=2)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[policy]["color"],
            linestyle=STYLES[policy]["linestyle"],
            linewidth=1.8 if policy == "HashPrune+" else 1.05,
            label=policy,
        )
        for policy in POLICY_ORDER
    ]
    effort_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markeredgewidth=0.55,
            linestyle="None",
            markersize=5,
            label="$L=50$",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="s",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markeredgewidth=0.55,
            linestyle="None",
            markersize=5,
            label="$L=200$",
        ),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.05,
        handlelength=2.1,
        handletextpad=0.4,
        borderaxespad=0.0,
        borderpad=0.0,
    )
    fig.legend(
        handles=effort_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
        columnspacing=1.1,
        handletextpad=0.3,
        borderaxespad=0.0,
        borderpad=0.0,
    )
    fig.supxlabel("Recall@10 (%)", fontsize=8.5, y=0.02)
    fig.supylabel("QPS", fontsize=8.5, x=0.025)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.73, wspace=0.16)

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
        default=output_dir / "hashprune_plus_recall_qps_ablation.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=output_dir / "hashprune_plus_recall_qps_ablation.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_tradeoff(load_results(), args.pdf, args.png)


if __name__ == "__main__":
    main()
