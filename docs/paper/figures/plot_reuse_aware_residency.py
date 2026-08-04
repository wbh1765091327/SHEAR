#!/usr/bin/env python3
"""Render the Wiki10M reuse-aware residency value comparison."""

from pathlib import Path

import cairo

from cairo_paper_figure import text_extents, y_axis_title_anchor_x
from paper_figure_style import (
    COMPACT_SINGLE_COLUMN_WIDTH_PT,
    FONT_FAMILY,
    REUSE_RESIDENCY_PLOT_BOTTOM_PT,
    REUSE_RESIDENCY_PLOT_TOP_PT,
    REUSE_RESIDENCY_ROW_HEIGHT_PT,
    TICK_LABEL_PAD_PT,
    TICK_SIZE_PT,
    THREE_FIGURE_AXIS_LABEL_SIZE_PT,
    THREE_FIGURE_TICK_SIZE_PT,
)


WIDTH_PT = COMPACT_SINGLE_COLUMN_WIDTH_PT
MAX_VALUE = 160.0
PLOT_LEFT_PT = 28.25
PLOT_RIGHT_PT = WIDTH_PT - 1.0
PLOT_TOP_PT = REUSE_RESIDENCY_PLOT_TOP_PT
PLOT_BOTTOM_PT = REUSE_RESIDENCY_PLOT_BOTTOM_PT
BAR_LABEL_BASELINE_Y_PT = PLOT_BOTTOM_PT + 12.0
GROUP_LABEL_BASELINE_Y_PT = PLOT_BOTTOM_PT + 25.5
Y_AXIS_TITLE_SHIFT_PT = -1.0
REUSE_AXIS_LABEL_SIZE_PT = 10.2
REUSE_CATEGORY_LABEL_SIZE_PT = 9.4
REUSE_RATIO_LABEL_SIZE_PT = 8.6

OUTPUT_HEIGHT_PT = REUSE_RESIDENCY_ROW_HEIGHT_PT

GROUP_CENTER_OFFSET_PT = 38.0
X_LABEL_COLOR = (0.0, 0.0, 0.0)
RATIO_ARROW_HEAD_LENGTH_PT = 3.0
RATIO_ARROW_HEAD_HALF_WIDTH_PT = 2.0
RATIO_LABEL_CLEARANCE_PT = 2.0
RATIO_STROKE_WIDTH_PT = 1.0

COLORS = {
    "SQ": (0.000, 0.447, 0.698),
    "Reservoir": (0.835, 0.369, 0.000),
    "ink": (0.0, 0.0, 0.0),
    "muted": (0.38, 0.38, 0.38),
    "grid": (0.88, 0.88, 0.88),
    "bracket": (0.32, 0.32, 0.32),
}

# Source configurations retained here to keep the plotted comparisons auditable.
SOURCE_CONFIGS = (
    "Dual (100/100)",
    "SQ-first (100/0)",
    "Reservoir-first (0/100)",
    "Half SQ (50/0)",
    "Proportional (50/57.5)",
)

# Values are raw-adjusted wall time avoided per resident GiB.
COMPARISONS = (
    ("Full-state", 146.99, 7.18, 20.48),
    ("Partial-state", 142.32, 7.22, 19.72),
)


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float = THREE_FIGURE_TICK_SIZE_PT,
    *,
    align: str = "left",
    bold: bool = False,
    color: tuple[float, float, float] = COLORS["ink"],
    rotation: float = 0.0,
) -> None:
    if size < TICK_SIZE_PT:
        raise ValueError("Figure text must remain at least 8 pt at the rendered size")
    ctx.save()
    ctx.translate(x, y)
    if rotation:
        ctx.rotate(rotation)
    ctx.select_font_face(
        FONT_FAMILY,
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    extents = ctx.text_extents(value)
    if align == "center":
        offset = -(extents.x_bearing + extents.width / 2)
    elif align == "right":
        offset = -(extents.x_bearing + extents.width)
    else:
        offset = 0.0
    set_color(ctx, color)
    ctx.move_to(offset, 0)
    ctx.show_text(value)
    ctx.restore()


def draw_line(
    ctx: cairo.Context,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: tuple[float, float, float] = COLORS["ink"],
    width: float = 0.6,
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(width)
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.stroke()
    ctx.restore()


def fill_rect(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[float, float, float],
) -> None:
    ctx.save()
    ctx.rectangle(x, y, width, height)
    set_color(ctx, color)
    ctx.fill()
    ctx.restore()


def draw_ratio_label(
    ctx: cairo.Context,
    bar_center: float,
    reference_bar_top: float,
    bar_top: float,
    bar_width: float,
    ratio: float,
) -> None:
    ratio_text = f"{ratio:.2f}×"
    top_cap_y = reference_bar_top
    arrow_bottom_y = bar_top
    label_center_y = (top_cap_y + arrow_bottom_y) / 2
    ratio_extents = text_extents(ctx, ratio_text, REUSE_RATIO_LABEL_SIZE_PT, bold=False)
    label_half_height = ratio_extents.width / 2 + RATIO_LABEL_CLEARANCE_PT
    upper_label_edge = label_center_y - label_half_height
    lower_label_edge = label_center_y + label_half_height
    arrow_color = COLORS["bracket"]

    draw_line(ctx, bar_center - bar_width / 2, top_cap_y,
              bar_center + bar_width / 2, top_cap_y,
              color=arrow_color, width=RATIO_STROKE_WIDTH_PT)
    draw_line(ctx, bar_center - bar_width / 2, arrow_bottom_y,
              bar_center + bar_width / 2, arrow_bottom_y,
              color=arrow_color, width=RATIO_STROKE_WIDTH_PT)
    draw_line(ctx, bar_center, top_cap_y, bar_center, upper_label_edge,
              color=arrow_color, width=RATIO_STROKE_WIDTH_PT)
    draw_line(ctx, bar_center, lower_label_edge, bar_center, arrow_bottom_y,
              color=arrow_color, width=RATIO_STROKE_WIDTH_PT)
    draw_line(
        ctx,
        bar_center,
        top_cap_y,
        bar_center - RATIO_ARROW_HEAD_HALF_WIDTH_PT,
        top_cap_y + RATIO_ARROW_HEAD_LENGTH_PT,
        color=arrow_color,
        width=RATIO_STROKE_WIDTH_PT,
    )
    draw_line(
        ctx,
        bar_center,
        top_cap_y,
        bar_center + RATIO_ARROW_HEAD_HALF_WIDTH_PT,
        top_cap_y + RATIO_ARROW_HEAD_LENGTH_PT,
        color=arrow_color,
        width=RATIO_STROKE_WIDTH_PT,
    )
    draw_line(
        ctx,
        bar_center,
        arrow_bottom_y,
        bar_center - RATIO_ARROW_HEAD_HALF_WIDTH_PT,
        arrow_bottom_y - RATIO_ARROW_HEAD_LENGTH_PT,
        color=arrow_color,
        width=RATIO_STROKE_WIDTH_PT,
    )
    draw_line(
        ctx,
        bar_center,
        arrow_bottom_y,
        bar_center + RATIO_ARROW_HEAD_HALF_WIDTH_PT,
        arrow_bottom_y - RATIO_ARROW_HEAD_LENGTH_PT,
        color=arrow_color,
        width=RATIO_STROKE_WIDTH_PT,
    )
    draw_text(
        ctx,
        bar_center,
        label_center_y,
        ratio_text,
        REUSE_RATIO_LABEL_SIZE_PT,
        align="center",
        bold=False,
        rotation=-1.57079632679,
    )


def draw_figure(ctx: cairo.Context) -> None:
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    set_color(ctx, (1.0, 1.0, 1.0))
    ctx.paint()

    plot_left, plot_right = PLOT_LEFT_PT, PLOT_RIGHT_PT
    plot_top, plot_bottom = PLOT_TOP_PT, PLOT_BOTTOM_PT
    plot_height = plot_bottom - plot_top
    plot_center = (plot_left + plot_right) / 2
    group_centers = (
        plot_center - GROUP_CENTER_OFFSET_PT,
        plot_center + GROUP_CENTER_OFFSET_PT,
    )
    bar_offset = 15.5
    bar_width = 18.0
    ticks = (0, 80, 160)
    tick_labels = ("0", "80", "")
    y_axis_title = "Time saved (s/GiB)"

    for tick, tick_label in zip(ticks, tick_labels):
        y = plot_bottom - tick / MAX_VALUE * plot_height
        draw_line(ctx, plot_left, y, plot_right, y, color=COLORS["grid"], width=0.45)
        if tick_label:
            draw_text(
                ctx,
                plot_left - TICK_LABEL_PAD_PT,
                y + 2.7,
                tick_label,
                align="right",
                color=COLORS["ink"],
            )

    draw_line(ctx, plot_left, plot_top, plot_right, plot_top, width=0.65)
    draw_line(ctx, plot_right, plot_top, plot_right, plot_bottom, width=0.65)
    draw_line(ctx, plot_left, plot_top, plot_left, plot_bottom, width=0.65)
    draw_line(ctx, plot_left, plot_bottom, plot_right, plot_bottom, width=0.65)
    draw_text(
        ctx,
        y_axis_title_anchor_x(
            ctx,
            plot_left,
            tick_labels,
            y_axis_title,
            tick_size=THREE_FIGURE_TICK_SIZE_PT,
            title_size=REUSE_AXIS_LABEL_SIZE_PT,
        )
        + Y_AXIS_TITLE_SHIFT_PT,
        (plot_top + plot_bottom) / 2,
        y_axis_title,
        REUSE_AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    for center_x, (group_label, sq_value, reservoir_value, ratio) in zip(group_centers, COMPARISONS):
        left_center = center_x - bar_offset
        right_center = center_x + bar_offset
        for bar_center, label, value in (
            (left_center, "SQ", sq_value),
            (right_center, "Reservoir", reservoir_value),
        ):
            height = value / MAX_VALUE * plot_height
            y = plot_bottom - height
            fill_rect(ctx, bar_center - bar_width / 2, y, bar_width, height, COLORS[label])
            draw_text(
                ctx,
                bar_center,
                BAR_LABEL_BASELINE_Y_PT,
                label,
                REUSE_CATEGORY_LABEL_SIZE_PT,
                align="center",
                bold=False,
                color=X_LABEL_COLOR,
            )

        left_bar_top = plot_bottom - sq_value / MAX_VALUE * plot_height
        right_bar_top = plot_bottom - reservoir_value / MAX_VALUE * plot_height
        draw_ratio_label(ctx, right_center, left_bar_top, right_bar_top, bar_width, ratio)
        draw_text(
            ctx,
            center_x,
            GROUP_LABEL_BASELINE_Y_PT,
            group_label,
            THREE_FIGURE_TICK_SIZE_PT,
            align="center",
            bold=False,
            color=X_LABEL_COLOR,
        )


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, OUTPUT_HEIGHT_PT)
    ctx = cairo.Context(surface)
    draw_figure(ctx)
    surface.finish()


def render_png(path: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32,
        round(WIDTH_PT * scale),
        round(OUTPUT_HEIGHT_PT * scale),
    )
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "reuse_aware_residency.pdf")
    render_png(output_dir / "reuse_aware_residency.png")


if __name__ == "__main__":
    main()
