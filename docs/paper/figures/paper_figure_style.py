#!/usr/bin/env python3
"""Shared print style for single-column experiment-analysis figures."""

SINGLE_COLUMN_WIDTH_IN = 3.35
SINGLE_COLUMN_WIDTH_PT = SINGLE_COLUMN_WIDTH_IN * 72.0

# Standalone evaluation PDFs retain this source height.
ROW_HEIGHT_PT = 132.0

# The compact three-figure row shares one logical 122 pt canvas and one 80 pt
# plot frame. Every renderer removes the same safe 5 pt strip from the top, so
# the final PDFs remain aligned without changing their source widths.
THREE_FIGURE_ROW_HEIGHT_PT = 127.0
THREE_FIGURE_CROP_TOP_PT = 5.0
THREE_FIGURE_PLOT_TOP_PT = 19.0
THREE_FIGURE_PLOT_BOTTOM_PT = 99.0

# The TierLift/residency row needs a taller header band for the two-panel
# trade-off figure's shared legend and panel titles.
TRADEOFF_RESIDENCY_ROW_WIDTH_PT = SINGLE_COLUMN_WIDTH_PT
TRADEOFF_RESIDENCY_ROW_HEIGHT_PT = 107.0
REUSE_RESIDENCY_ROW_HEIGHT_PT = 107.2
TRADEOFF_RESIDENCY_PLOT_LEFT_PT = 31.25
TRADEOFF_RESIDENCY_PLOT_RIGHT_PT = TRADEOFF_RESIDENCY_ROW_WIDTH_PT - 5.0
TRADEOFF_RESIDENCY_PLOT_TOP_PT = 27.0
TRADEOFF_RESIDENCY_PLOT_BOTTOM_PT = 75.38
TRADEOFF_RESIDENCY_HEADER_TOP_PT = 3.5
TRADEOFF_RESIDENCY_LEGEND_PLOT_TOP_PT = 12.87
REUSE_RESIDENCY_PLOT_TOP_PT = 12.88
REUSE_RESIDENCY_PLOT_BOTTOM_PT = 75.5

# Compact bar charts that are paired across the two paper columns use one
# physical height.  The narrower canvas is inserted at 0.933\linewidth so its
# final scale matches a 3.35-inch canvas inserted at \linewidth.
SINGLE_COLUMN_BAR_HEIGHT_PT = 126.0
COMPACT_SINGLE_COLUMN_WIDTH_PT = 190.0

FONT_FAMILY = "Arial"
FONT_FALLBACKS = ("Arial",)
BASE_SIZE_PT = 8.0
AXIS_LABEL_SIZE_PT = 8.5
PANEL_TITLE_SIZE_PT = 8.0
TICK_SIZE_PT = 8.0
VALUE_SIZE_PT = 8.0
LEGEND_SIZE_PT = 7.2

# Slightly larger source typography for the compact three-figure evaluation
# row.  The PDFs are inserted at about 70% of their natural width.
THREE_FIGURE_TICK_SIZE_PT = 10.0
THREE_FIGURE_AXIS_LABEL_SIZE_PT = 10.7
THREE_FIGURE_PANEL_TITLE_SIZE_PT = 11.4
THREE_FIGURE_VALUE_SIZE_PT = 10.0
THREE_FIGURE_LEGEND_SIZE_PT = 10.0

# These source sizes compensate for the common ~0.70 LaTeX scale used by the
# compact three-figure evaluation row.
ROW_AXIS_LABEL_SIZE_PT = 10.7
ROW_TICK_SIZE_PT = 10.0
ROW_LEGEND_SIZE_PT = 10.0

AXIS_LABEL_GAP_PT = 2.5
TICK_LABEL_PAD_PT = 3.0
AXIS_LABEL_PAD_PT = AXIS_LABEL_GAP_PT
PANEL_TITLE_PAD_PT = 1.0
LEGEND_GAP_PT = 2.0
SAFE_OUTER_MARGIN_PT = 6.0

AXIS_LINE_WIDTH_PT = 0.65
GRID_LINE_WIDTH_PT = 0.45
GRID_COLOR_HEX = "#E0E0E0"
GRID_COLOR_RGB = (0.88, 0.88, 0.88)

MATPLOTLIB_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": list(FONT_FALLBACKS),
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "font.size": BASE_SIZE_PT,
    "axes.labelsize": AXIS_LABEL_SIZE_PT,
    "axes.titlesize": PANEL_TITLE_SIZE_PT,
    "axes.labelpad": AXIS_LABEL_PAD_PT,
    "axes.titlepad": PANEL_TITLE_PAD_PT,
    "axes.linewidth": AXIS_LINE_WIDTH_PT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": LEGEND_SIZE_PT,
    "legend.frameon": False,
    "xtick.labelsize": TICK_SIZE_PT,
    "ytick.labelsize": TICK_SIZE_PT,
    "xtick.major.pad": TICK_LABEL_PAD_PT,
    "ytick.major.pad": TICK_LABEL_PAD_PT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SAVEFIG_KWARGS = {"bbox_inches": None, "pad_inches": 0}
