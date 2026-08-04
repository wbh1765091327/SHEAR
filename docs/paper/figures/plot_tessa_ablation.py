#!/usr/bin/env python3
"""Render the Wiki10M TESSA stage-wise operator ablation."""

from dataclasses import dataclass
from pathlib import Path

import cairo

from cairo_paper_figure import text_extents, y_axis_title_anchor_x
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


WIDTH_PT = SINGLE_COLUMN_WIDTH_PT
OUTPUT_HEIGHT_PT = THREE_FIGURE_ROW_HEIGHT_PT - THREE_FIGURE_CROP_TOP_PT
Y_MAX_S = 200.0
X_LABEL_Y_PT = THREE_FIGURE_PLOT_BOTTOM_PT + 12.5
Y_AXIS_TITLE_SHIFT_PT = 0.0

PLOT_LEFT_PT = 34.0
PLOT_RIGHT_PT = WIDTH_PT - 1.0

INK = (0.0, 0.0, 0.0)
MUTED = (0.38, 0.38, 0.38)
GRID = (0.88, 0.88, 0.88)
GATHER_COLOR = (1.0, 1.0, 1.0)
ASSIGNMENT_COLOR = (0.72, 0.73, 0.74)
CANDIDATE_COLOR = (0.000, 0.447, 0.698)


@dataclass(frozen=True)
class Variant:
    name: str
    gather: float
    gather_min: float
    gather_max: float
    assignment: float
    assignment_min: float
    assignment_max: float
    candidate: float
    candidate_min: float
    candidate_max: float


# Gather is the full-build CPU work that assembles the per-subtree payload before
# H2D: raw vector load/gather for A0/A1 and resident SQ8 sidecar gather otherwise.
# Gather values are medians and min--max ranges over three from-scratch builds.
# A0/A1/T1/T8 assignment and candidate values use the same aggregation. D1/D8
# use the packed-DP4A measurements from the matched first rerun; the completed
# second D8 rerun supplies its lower bound. All times are in seconds.
# Assignment and candidate values convert the persistent kernel's internal
# busy-clock counters to average active time per worker using the worker count
# and the SM clock measured during the same build. The two worker groups execute
# concurrently, so their active times are not additive.
# Sources:
#   /data_staging/wbh/tessa_ablation/wiki10m/20260721_011618
#   /data_staging/wbh/tessa_ablation/wiki10m/20260724_185453_leaf_dp4a_fix
# Conversion: busy_clocks / (worker_blocks * mean_SM_clock_hz), where the SM
# clock is averaged over >=90% GPU-utilization samples in the D1+ interval.
VARIANTS = (
    Variant("A0", 130.491, 130.125, 131.789, 166.880, 166.725, 167.172, 111.941, 111.839, 112.092),
    Variant("A1", 130.580, 130.557, 131.342, 163.777, 163.765, 163.798, 48.728, 48.718, 48.742),
    Variant("D1", 28.740, 28.559, 28.818, 69.755, 69.755, 69.755, 42.372, 42.372, 42.372),
    Variant("T1", 28.932, 28.788, 29.323, 94.949, 94.320, 95.212, 23.674, 23.533, 23.785),
    Variant("D8", 28.885, 28.734, 29.006, 77.034, 75.676, 77.034, 42.980, 42.391, 42.980),
    Variant("T8", 28.198, 27.944, 28.872, 44.546, 44.519, 44.853, 24.138, 24.124, 24.281),
)


def validate_data() -> None:
    if tuple(item.name for item in VARIANTS) != ("A0", "A1", "D1", "T1", "D8", "T8"):
        raise ValueError("Unexpected TESSA variant order")
    for item in VARIANTS:
        for stage in ("gather", "assignment", "candidate"):
            value = getattr(item, stage)
            minimum = getattr(item, f"{stage}_min")
            maximum = getattr(item, f"{stage}_max")
            if not (0 < minimum <= value <= maximum):
                raise ValueError(f"Invalid {stage} range for {item.name}")


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


def stroke_rect(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: tuple[float, float, float] = INK,
    line_width: float = 0.8,
) -> None:
    ctx.save()
    ctx.rectangle(x, y, width, height)
    set_color(ctx, color)
    ctx.set_line_width(line_width)
    ctx.stroke()
    ctx.restore()


def draw_bar(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[float, float, float],
    *,
    hatched: bool = False,
) -> None:
    fill_rect(ctx, x, y, width, height, color)
    if hatched:
        ctx.save()
        ctx.rectangle(x, y, width, height)
        ctx.clip()
        set_color(ctx, MUTED)
        ctx.set_line_width(0.65)
        offset = -height
        while offset < width:
            ctx.move_to(x + offset, y + height)
            ctx.line_to(x + offset + height, y)
            offset += 5.0
        ctx.stroke()
        ctx.restore()
    stroke_rect(ctx, x, y, width, height, color=MUTED, line_width=0.65)


def draw_legend_item(
    ctx: cairo.Context,
    x: float,
    baseline_y: float,
    label: str,
    color: tuple[float, float, float],
    *,
    hatched: bool = False,
) -> None:
    draw_bar(ctx, x, baseline_y - 6.0, 8.0, 6.0, color, hatched=hatched)
    draw_text(ctx, x + 9.0, baseline_y, label, ROW_LEGEND_SIZE_PT)


def draw_legend_column(
    ctx: cairo.Context,
    right_x: float,
    first_baseline_y: float,
    items: tuple[tuple[str, tuple[float, float, float], bool], ...],
    *,
    row_gap: float = 9.5,
) -> None:
    item_widths = tuple(
        9.0 + text_extents(ctx, label, ROW_LEGEND_SIZE_PT).width
        for label, _, _ in items
    )
    x = right_x - max(item_widths)
    for row, (label, color, hatched) in enumerate(items):
        draw_legend_item(
            ctx,
            x,
            first_baseline_y + row * row_gap,
            label,
            color,
            hatched=hatched,
        )


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
    centers = tuple(plot_center + offset for offset in (-88.5, -52.5, -16.5, 19.5, 55.5, 88.5))
    bar_width = 6.5
    offsets = (-7.5, 0.0, 7.5)
    ticks = (0.0, 50.0, 100.0, 150.0, 200.0)
    tick_labels = ("0", "50", "100", "150", "")
    y_axis_title = "Active time (s)"

    def y_for(value_s: float) -> float:
        return plot_bottom - value_s / Y_MAX_S * plot_height

    legend_items = (
        ("Input gather", GATHER_COLOR, True),
        ("Partition assignment", ASSIGNMENT_COLOR, False),
        ("Candidate scoring", CANDIDATE_COLOR, False),
    )
    draw_legend_column(
        ctx,
        right_x=plot_right - 2.0,
        first_baseline_y=plot_top + 9.0,
        items=legend_items,
        row_gap=10.5,
    )

    for tick, tick_label in zip(ticks, tick_labels):
        y = y_for(tick)
        draw_line(ctx, plot_left, y, plot_right, y, color=GRID, width=0.45)
        if tick_label:
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

    for center_x, item in zip(centers, VARIANTS):
        for offset, stage, color, hatched in (
            (offsets[0], "gather", GATHER_COLOR, True),
            (offsets[1], "assignment", ASSIGNMENT_COLOR, False),
            (offsets[2], "candidate", CANDIDATE_COLOR, False),
        ):
            value_s = getattr(item, stage)
            x = center_x + offset - bar_width / 2
            y = y_for(value_s)
            draw_bar(ctx, x, y, bar_width, plot_bottom - y, color, hatched=hatched)

        draw_text(
            ctx,
            center_x,
            X_LABEL_Y_PT,
            item.name,
            ROW_TICK_SIZE_PT,
            align="center",
            bold=False,
            color=INK,
        )


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, OUTPUT_HEIGHT_PT)
    ctx = cairo.Context(surface)
    ctx.translate(0.0, -THREE_FIGURE_CROP_TOP_PT)
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
    ctx.translate(0.0, -THREE_FIGURE_CROP_TOP_PT)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "tessa_ablation.pdf")
    render_png(output_dir / "tessa_ablation.png")


if __name__ == "__main__":
    main()
