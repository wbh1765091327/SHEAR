#!/usr/bin/env python3
"""Render the Wiki10M fused-persistent overlap ablation."""

from pathlib import Path

import cairo

from cairo_paper_figure import y_axis_title_anchor_x
from paper_figure_style import (
    FONT_FAMILY,
    ROW_AXIS_LABEL_SIZE_PT,
    ROW_LEGEND_SIZE_PT,
    ROW_TICK_SIZE_PT,
    SINGLE_COLUMN_WIDTH_PT,
    THREE_FIGURE_CROP_TOP_PT,
    THREE_FIGURE_PLOT_BOTTOM_PT,
    THREE_FIGURE_PLOT_TOP_PT,
    THREE_FIGURE_ROW_HEIGHT_PT,
    TICK_LABEL_PAD_PT,
)


OUTPUT_WIDTH_PT = 200.0
WIDTH_PT = OUTPUT_WIDTH_PT
OUTPUT_HEIGHT_PT = (
    (THREE_FIGURE_ROW_HEIGHT_PT - THREE_FIGURE_CROP_TOP_PT)
    * (0.34 / SINGLE_COLUMN_WIDTH_PT)
    / (0.28 / OUTPUT_WIDTH_PT)
)
Y_MAX_S = 130.0
X_LABEL_BASELINE_Y_PT = THREE_FIGURE_PLOT_BOTTOM_PT + 10.0
Y_AXIS_TITLE_SHIFT_PT = 0.0

PLOT_LEFT_PT = 33.75
PLOT_RIGHT_PT = OUTPUT_WIDTH_PT - 1.0
GROUP_CENTER_OFFSET_PT = 40.0

INK = (0.0, 0.0, 0.0)
X_LABEL_COLOR = (0.0, 0.0, 0.0)
MUTED = (0.38, 0.38, 0.38)
GRID = (0.88, 0.88, 0.88)
PERSISTENT_COLOR = (0.000, 0.447, 0.698)
PARTITION_COLOR = (0.72, 0.73, 0.74)
TOPK_COLOR = (0.835, 0.369, 0.000)
CANDIDATION_LABEL = "Candidation"

# Persistent is the complete GPU event for each execution mode. For fused
# execution, Partition and Top-k are average active times obtained from
# busy-clock counters normalized by worker count and measured SM clock. For
# two-stage execution, they are the isolated partition-only and top-k-only CUDA
# event walls.
VARIANTS_S = {
    "Fused": {
        "Persistent": 48.702,
        "Partition": 44.288727,
        "Top-k": 24.007744,
    },
    "Two-stage": {
        "Persistent": 119.744,
        "Partition": 31.751,
        "Top-k": 87.992,
    },
}
OVERLAP_LOWER_BOUND_S = (
    VARIANTS_S["Fused"]["Partition"]
    + VARIANTS_S["Fused"]["Top-k"]
    - VARIANTS_S["Fused"]["Persistent"]
)


def validate_data() -> None:
    if tuple(VARIANTS_S) != ("Fused", "Two-stage"):
        raise ValueError("Unexpected variant order")
    for timings in VARIANTS_S.values():
        if tuple(timings) != ("Persistent", "Partition", "Top-k"):
            raise ValueError("Unexpected timing order")
        if any(value <= 0.0 for value in timings.values()):
            raise ValueError("Timings must be positive")
    if abs(OVERLAP_LOWER_BOUND_S - 19.594471) > 1.0e-6:
        raise ValueError("Unexpected effective-overlap lower bound")
    two_stage = VARIANTS_S["Two-stage"]
    if abs(
        two_stage["Partition"] + two_stage["Top-k"] - two_stage["Persistent"]
    ) > 0.002:
        raise ValueError("Two-stage component events must sum to its total event")


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float = ROW_TICK_SIZE_PT,
    *,
    align: str = "left",
    bold: bool = False,
    color: tuple[float, float, float] = INK,
    rotation: float = 0.0,
) -> None:
    if size < ROW_LEGEND_SIZE_PT:
        raise ValueError("Figure text must remain at least the evaluation-row legend size")
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
    color: tuple[float, float, float] = INK,
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


def draw_legend_item(
    ctx: cairo.Context,
    x: float,
    y: float,
    label: str,
    color: tuple[float, float, float],
) -> None:
    fill_rect(ctx, x, y - 6.0, 7.0, 7.0, color)
    draw_text(ctx, x + 10.0, y, label, ROW_LEGEND_SIZE_PT)


def draw_legend_column(
    ctx: cairo.Context,
    x: float,
    first_baseline_y: float,
    items: tuple[tuple[str, tuple[float, float, float]], ...],
    *,
    row_gap: float = 10.0,
) -> None:
    for row, (label, color) in enumerate(items):
        draw_legend_item(ctx, x, first_baseline_y + row * row_gap, label, color)


def draw_figure(ctx: cairo.Context) -> None:
    validate_data()
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    set_color(ctx, (1.0, 1.0, 1.0))
    ctx.paint()

    plot_left, plot_right = PLOT_LEFT_PT, PLOT_RIGHT_PT
    plot_top, plot_bottom = (
        THREE_FIGURE_PLOT_TOP_PT,
        THREE_FIGURE_PLOT_BOTTOM_PT,
    )
    plot_height = plot_bottom - plot_top
    plot_center = (plot_left + plot_right) / 2
    group_centers = (
        plot_center - GROUP_CENTER_OFFSET_PT,
        plot_center + GROUP_CENTER_OFFSET_PT,
    )
    bar_offsets = (-21.0, 0.0, 21.0)
    bar_width = 20.0
    series_colors = {
        "Persistent": PERSISTENT_COLOR,
        "Partition": PARTITION_COLOR,
        "Top-k": TOPK_COLOR,
    }
    ticks = (0.0, 30.0, 60.0, 90.0, 120.0)
    tick_labels = tuple(f"{tick:.0f}" for tick in ticks)
    y_axis_title = "GPU time (s)"

    def y_for(value_s: float) -> float:
        return plot_bottom - value_s / Y_MAX_S * plot_height

    for tick, tick_label in zip(ticks, tick_labels):
        y = y_for(tick)
        draw_line(ctx, plot_left, y, plot_right, y, color=GRID, width=0.45)
        draw_text(
            ctx,
            plot_left - TICK_LABEL_PAD_PT,
            y + 2.7,
            tick_label,
            align="right",
            color=INK,
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
            tick_size=ROW_TICK_SIZE_PT,
            title_size=ROW_AXIS_LABEL_SIZE_PT,
        )
        + Y_AXIS_TITLE_SHIFT_PT,
        (plot_top + plot_bottom) / 2,
        y_axis_title,
        ROW_AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    for group_center, (variant, timings) in zip(group_centers, VARIANTS_S.items()):
        for offset, (name, value_s) in zip(bar_offsets, timings.items()):
            center_x = group_center + offset
            y = y_for(value_s)
            fill_rect(
                ctx,
                center_x - bar_width / 2,
                y,
                bar_width,
                plot_bottom - y,
                series_colors[name],
            )
        draw_text(
            ctx,
            group_center,
            X_LABEL_BASELINE_Y_PT,
            variant,
            ROW_TICK_SIZE_PT,
            align="center",
            bold=False,
            color=X_LABEL_COLOR,
        )

    legend_items = (
        ("Persistent", PERSISTENT_COLOR),
        ("Partition", PARTITION_COLOR),
        (CANDIDATION_LABEL, TOPK_COLOR),
    )
    draw_legend_column(
        ctx,
        x=plot_left + 3.0,
        first_baseline_y=plot_top + 10.0,
        items=legend_items,
        row_gap=11.0,
    )


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), OUTPUT_WIDTH_PT, OUTPUT_HEIGHT_PT)
    ctx = cairo.Context(surface)
    ctx.translate(0.0, -THREE_FIGURE_CROP_TOP_PT)
    draw_figure(ctx)
    surface.finish()


def render_png(path: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32,
        round(OUTPUT_WIDTH_PT * scale),
        round(OUTPUT_HEIGHT_PT * scale),
    )
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    ctx.translate(0.0, -THREE_FIGURE_CROP_TOP_PT)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "persistent_two_stage.pdf")
    render_png(output_dir / "persistent_two_stage.png")


if __name__ == "__main__":
    main()
