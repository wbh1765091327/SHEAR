#!/usr/bin/env python3
"""Render the LAION100M external-reduction timing comparison."""

from pathlib import Path

import cairo

from cairo_paper_figure import y_axis_title_anchor_x
from paper_figure_style import (
    FONT_FAMILY,
    TICK_LABEL_PAD_PT,
    THREE_FIGURE_AXIS_LABEL_SIZE_PT,
    THREE_FIGURE_LEGEND_SIZE_PT,
    THREE_FIGURE_TICK_SIZE_PT,
    TRADEOFF_RESIDENCY_HEADER_TOP_PT,
    TRADEOFF_RESIDENCY_LEGEND_PLOT_TOP_PT,
    TRADEOFF_RESIDENCY_PLOT_BOTTOM_PT,
    TRADEOFF_RESIDENCY_PLOT_LEFT_PT,
    TRADEOFF_RESIDENCY_PLOT_RIGHT_PT,
    TRADEOFF_RESIDENCY_ROW_HEIGHT_PT,
    TRADEOFF_RESIDENCY_ROW_WIDTH_PT,
)


WIDTH_PT = TRADEOFF_RESIDENCY_ROW_WIDTH_PT
HEIGHT_PT = TRADEOFF_RESIDENCY_ROW_HEIGHT_PT
PLOT_LEFT_PT = TRADEOFF_RESIDENCY_PLOT_LEFT_PT + 2.5
PLOT_RIGHT_PT = TRADEOFF_RESIDENCY_PLOT_RIGHT_PT
PLOT_TOP_PT = TRADEOFF_RESIDENCY_LEGEND_PLOT_TOP_PT
PLOT_BOTTOM_PT = TRADEOFF_RESIDENCY_PLOT_BOTTOM_PT
LEGEND_TOP_PT = TRADEOFF_RESIDENCY_HEADER_TOP_PT
LEGEND_TEXT_BASELINE_PT = LEGEND_TOP_PT + 6.5
X_TICK_BASELINE_OFFSET_PT = 12.0
X_LABEL_BASELINE_OFFSET_PT = 25.65
Y_AXIS_TITLE_SHIFT_PT = 0.0
MAX_TIME_S = 800.0
BAR_WIDTH_PT = 22.0
SERIES = ("Pre", "GPU", "Post-GPU")

BAR_COLORS = {
    "Pre": (0.720, 0.730, 0.740),
    "GPU": (0.835, 0.369, 0.000),
    "Post-GPU": (0.000, 0.620, 0.451),
}

COLORS = {
    "ink": (0.0, 0.0, 0.0),
    "muted": (0.38, 0.38, 0.38),
    "grid": (0.88, 0.88, 0.88),
}

# Values are seconds. The 0% point adjusts the established 700.8 s in-memory E2E
# by the GPU correction from 489.6 s to 455.0 s, yielding 666.2 s; its Post-GPU
# interval remains 31.8 s. Pre is the residual E2E - GPU - Post-GPU under the
# paper's timing convention. The remaining points use the compact-fused
# external-reduction experiment completed on 2026-07-31.
GROUPS = (
    ("0%", 666.2, 179.4, 455.0, 31.8),
    ("25%", 691.01, 191.629, 454.951, 44.43),
    ("50%", 709.15, 194.048, 454.753, 60.349),
    ("75%", 727.35, 198.025, 455.008, 74.317),
    ("100%", 747.29, 201.937, 454.947, 90.406),
)


def validate_data() -> None:
    for group_label, e2e, pre, gpu, post_gpu in GROUPS:
        if abs(pre + gpu + post_gpu - e2e) > 0.001:
            raise ValueError(f"Stage breakdown does not reconstruct E2E for {group_label}")


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


def draw_figure(ctx: cairo.Context) -> None:
    validate_data()
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    set_color(ctx, (1.0, 1.0, 1.0))
    ctx.paint()

    plot_left, plot_right = PLOT_LEFT_PT, PLOT_RIGHT_PT
    plot_top, plot_bottom = PLOT_TOP_PT, PLOT_BOTTOM_PT
    plot_height = plot_bottom - plot_top
    group_width = (plot_right - plot_left) / len(GROUPS)
    group_centers = tuple(plot_left + (index + 0.5) * group_width for index in range(len(GROUPS)))
    ticks = (0, 200, 400, 600, 800)
    tick_labels = ("0", "200", "400", "600", "")
    y_axis_title = "Time (s)"

    for tick, tick_label in zip(ticks, tick_labels):
        y = plot_bottom - tick / MAX_TIME_S * plot_height
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
            title_size=THREE_FIGURE_AXIS_LABEL_SIZE_PT,
        )
        + Y_AXIS_TITLE_SHIFT_PT,
        (plot_top + plot_bottom) / 2,
        y_axis_title,
        THREE_FIGURE_AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    legend_labels = SERIES
    legend_center_x = (plot_left + plot_right) / 2
    legend_swatch_width = 7.0
    legend_text_gap = 3.0
    legend_item_gap = 12.0
    ctx.save()
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(THREE_FIGURE_LEGEND_SIZE_PT)
    legend_text_widths = tuple(ctx.text_extents(label).x_advance for label in legend_labels)
    ctx.restore()
    legend_width = (
        sum(legend_swatch_width + legend_text_gap + width for width in legend_text_widths)
        + legend_item_gap * (len(legend_labels) - 1)
    )
    legend_x = legend_center_x - legend_width / 2
    for label, text_width in zip(legend_labels, legend_text_widths):
        fill_rect(ctx, legend_x, LEGEND_TOP_PT, 7.0, 7.0, BAR_COLORS[label])
        draw_text(
            ctx,
            legend_x + legend_swatch_width + legend_text_gap,
            LEGEND_TEXT_BASELINE_PT,
            label,
            THREE_FIGURE_LEGEND_SIZE_PT,
        )
        legend_x += legend_swatch_width + legend_text_gap + text_width + legend_item_gap

    def y_for(value_s: float) -> float:
        return plot_bottom - value_s / MAX_TIME_S * plot_height

    for center_x, (group_label, e2e, pre, gpu, post_gpu) in zip(group_centers, GROUPS):
        stack_bottom_s = 0.0
        for label, value in zip(SERIES, (pre, gpu, post_gpu)):
            stack_top_s = stack_bottom_s + value
            y = y_for(stack_top_s)
            segment_bottom_y = y_for(stack_bottom_s)
            fill_rect(
                ctx,
                center_x - BAR_WIDTH_PT / 2,
                y,
                BAR_WIDTH_PT,
                segment_bottom_y - y,
                BAR_COLORS[label],
            )
            stack_bottom_s = stack_top_s
        if abs(stack_bottom_s - e2e) > 0.001:
            raise ValueError(f"Stacked total does not match E2E for {group_label}")
        draw_text(
            ctx,
            center_x,
            PLOT_BOTTOM_PT + X_TICK_BASELINE_OFFSET_PT,
            group_label,
            THREE_FIGURE_TICK_SIZE_PT,
            align="center",
        )
    draw_text(
        ctx,
        (plot_left + plot_right) / 2,
        PLOT_BOTTOM_PT + X_LABEL_BASELINE_OFFSET_PT,
        "Out of core reduction (%)",
        THREE_FIGURE_AXIS_LABEL_SIZE_PT,
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
    render_pdf(output_dir / "laion100m_reduction_residency.pdf")
    render_png(output_dir / "laion100m_reduction_residency.png")


if __name__ == "__main__":
    main()
