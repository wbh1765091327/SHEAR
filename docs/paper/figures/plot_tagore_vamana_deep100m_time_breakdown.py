#!/usr/bin/env python3
"""Render the DEEP100M Tagore-Vamana largeIndex wall-time breakdown.

The values are copied from the accepted wait-probe run:
  /data_staging/wbh/tagore_vamana_four_dataset/runs/
  deep100m-wait-probe-20260729T090258Z

The three bars are selected wall-time components, not an exhaustive additive
decomposition.
"""

from pathlib import Path

import cairo

from cairo_paper_figure import y_axis_title_anchor_x
from paper_figure_style import (
    AXIS_LABEL_SIZE_PT,
    AXIS_LINE_WIDTH_PT,
    FONT_FAMILY,
    GRID_COLOR_RGB,
    GRID_LINE_WIDTH_PT,
    SINGLE_COLUMN_WIDTH_PT,
    TICK_LABEL_PAD_PT,
    TICK_SIZE_PT,
    VALUE_SIZE_PT,
)


WIDTH_PT = 0.8 * SINGLE_COLUMN_WIDTH_PT
HEIGHT_PT = 143.0

LARGE_INDEX_WALL_S = 1844.870634835
GPU_LOCAL_BUILD_S = 567.213248154
BACKPRESSURE_WAIT_S = 254.826961
POST_GPU_TAIL_S = 762.302178718
OTHER_WALL_S = (
    LARGE_INDEX_WALL_S
    - GPU_LOCAL_BUILD_S
    - BACKPRESSURE_WAIT_S
    - POST_GPU_TAIL_S
)

PLOT_LEFT_PT = 28.5
PLOT_RIGHT_PT = WIDTH_PT - 3.0
PLOT_TOP_PT = 3.0
PLOT_BOTTOM_PT = 125.0
X_LABEL_BASELINE_PT = 137.0
Y_MAX_S = 830.0
BAR_CENTER_FRACTIONS = (0.16, 0.50, 0.84)

INK = (0.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)
COLORS = {
    "gpu": (0.00, 0.60, 0.53),
    "wait": (0.00, 0.47, 0.73),
    "tail": (0.93, 0.47, 0.20),
}

BARS = (
    ("Local build", GPU_LOCAL_BUILD_S, COLORS["gpu"]),
    ("Backpressure", BACKPRESSURE_WAIT_S, COLORS["wait"]),
    ("Post-GPU tail", POST_GPU_TAIL_S, COLORS["tail"]),
)


def validate_data() -> None:
    if LARGE_INDEX_WALL_S <= 0.0:
        raise ValueError("largeIndex wall time must be positive")
    if any(value <= 0.0 for _, value, _ in BARS):
        raise ValueError("all displayed wall-time components must be positive")
    if OTHER_WALL_S < 0.0:
        raise ValueError("displayed components exceed largeIndex wall time")
    accounted = sum(value for _, value, _ in BARS) + OTHER_WALL_S
    if abs(accounted - LARGE_INDEX_WALL_S) > 1.0e-9:
        raise ValueError("wall-time components do not reconcile")


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float,
    *,
    align: str = "left",
    bold: bool = False,
    color: tuple[float, float, float] = INK,
) -> None:
    ctx.save()
    ctx.select_font_face(
        FONT_FAMILY,
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    extents = ctx.text_extents(value)
    if align == "center":
        x -= extents.x_bearing + extents.width / 2.0
    elif align == "right":
        x -= extents.x_bearing + extents.width
    set_color(ctx, color)
    ctx.move_to(x, y)
    ctx.show_text(value)
    ctx.restore()


def draw_rotated_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float,
) -> None:
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(-1.57079632679)
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    extents = ctx.text_extents(value)
    ctx.move_to(-(extents.x_bearing + extents.width / 2.0), 0.0)
    set_color(ctx, INK)
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
    width: float = AXIS_LINE_WIDTH_PT,
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(width)
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.stroke()
    ctx.restore()


def fill_bar(
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
    ctx.fill_preserve()
    set_color(ctx, (0.20, 0.20, 0.20))
    ctx.set_line_width(0.45)
    ctx.stroke()
    ctx.restore()


def draw_figure(ctx: cairo.Context) -> None:
    validate_data()
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    set_color(ctx, WHITE)
    ctx.paint()

    plot_height = PLOT_BOTTOM_PT - PLOT_TOP_PT
    y_ticks = (0, 200, 400, 600, 800)
    y_tick_labels = tuple(str(value) for value in y_ticks)
    y_axis_title = "Time (s)"

    for tick, label in zip(y_ticks, y_tick_labels, strict=True):
        y = PLOT_BOTTOM_PT - tick / Y_MAX_S * plot_height
        draw_line(
            ctx,
            PLOT_LEFT_PT,
            y,
            PLOT_RIGHT_PT,
            y,
            color=GRID_COLOR_RGB,
            width=GRID_LINE_WIDTH_PT,
        )
        draw_text(
            ctx,
            PLOT_LEFT_PT - TICK_LABEL_PAD_PT,
            y + 2.6,
            label,
            TICK_SIZE_PT,
            align="right",
        )

    draw_line(ctx, PLOT_LEFT_PT, PLOT_TOP_PT, PLOT_LEFT_PT, PLOT_BOTTOM_PT)
    draw_line(ctx, PLOT_LEFT_PT, PLOT_BOTTOM_PT, PLOT_RIGHT_PT, PLOT_BOTTOM_PT)
    draw_rotated_text(
        ctx,
        y_axis_title_anchor_x(ctx, PLOT_LEFT_PT, y_tick_labels, y_axis_title),
        (PLOT_TOP_PT + PLOT_BOTTOM_PT) / 2.0,
        y_axis_title,
        AXIS_LABEL_SIZE_PT,
    )

    plot_width = PLOT_RIGHT_PT - PLOT_LEFT_PT
    centers = tuple(PLOT_LEFT_PT + plot_width * fraction for fraction in BAR_CENTER_FRACTIONS)
    bar_width = 35.0

    for center, (label, value, color) in zip(centers, BARS, strict=True):
        bar_height = value / Y_MAX_S * plot_height
        bar_top = PLOT_BOTTOM_PT - bar_height
        fill_bar(ctx, center - bar_width / 2.0, bar_top, bar_width, bar_height, color)

        if label == "Post-GPU tail":
            draw_text(
                ctx,
                center,
                bar_top + bar_height / 2.0 + 2.7,
                f"{100.0 * value / LARGE_INDEX_WALL_S:.1f}%",
                VALUE_SIZE_PT,
                align="center",
                bold=True,
                color=WHITE,
            )

        draw_text(
            ctx,
            center,
            X_LABEL_BASELINE_PT,
            label,
            TICK_SIZE_PT,
            align="center",
        )


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, HEIGHT_PT)
    ctx = cairo.Context(surface)
    draw_figure(ctx)
    surface.finish()


def render_png(path: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32,
        round(WIDTH_PT * scale),
        round(HEIGHT_PT * scale),
    )
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "tagore_vamana_deep100m_time_breakdown.pdf")
    render_png(output_dir / "tagore_vamana_deep100m_time_breakdown.png")


if __name__ == "__main__":
    main()
