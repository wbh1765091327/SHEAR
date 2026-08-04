#!/usr/bin/env python3
"""Render the LAION35M D0-capacity timing screen.

Source record:
  /data_staging/wbh/laion35m_gpu_subtree_measurement/runs/
  20260721_uniform_capacity_warm_r040_r006/experiment_summary.tsv
"""

from dataclasses import dataclass
from pathlib import Path

import cairo

from cairo_paper_figure import x_axis_title_baseline_y, y_axis_title_anchor_x
from paper_figure_style import (
    FONT_FAMILY,
    GRID_LINE_WIDTH_PT,
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


CONTENT_WIDTH_PT = SINGLE_COLUMN_WIDTH_PT
WIDTH_PT = SINGLE_COLUMN_WIDTH_PT
HEIGHT_PT = THREE_FIGURE_ROW_HEIGHT_PT - THREE_FIGURE_CROP_TOP_PT
PLOT_LEFT_PT = 34.0
PLOT_RIGHT_PT = WIDTH_PT - 1.0
Y_MIN = 60.0
Y_MAX = 360.0
X_AXIS_TITLE_SHIFT_PT = 2.5
D0_LEGEND_SIZE_PT = ROW_LEGEND_SIZE_PT

INK = (0.0, 0.0, 0.0)
MUTED = (0.38, 0.38, 0.38)
GRID = (0.88, 0.88, 0.88)
WALL_COLOR = (0.000, 0.447, 0.698)
GATHER_COLOR = (0.000, 0.620, 0.451)
PERSISTENT_COLOR = (0.580, 0.404, 0.741)
BEST_COLOR = (0.835, 0.369, 0.000)


@dataclass(frozen=True)
class Point:
    ratio: int
    wall: float
    gather: float
    persistent: float
    fs_inputs: int


POINTS = (
    Point(40, 349.34, 72.392, 192.198, 26960),
    Point(30, 343.13, 72.200, 189.375, 0),
    Point(20, 334.68, 72.995, 181.659, 0),
    Point(10, 314.34, 74.551, 167.002, 0),
    Point(8, 307.74, 75.187, 162.984, 0),
    Point(6, 303.31, 75.789, 159.251, 0),
    Point(4, 299.98, 76.372, 155.541, 0),
    Point(2, 303.17, 80.016, 156.667, 0),
    Point(1, 319.60, 82.235, 168.682, 0),
)


def validate_data() -> None:
    ratios = tuple(point.ratio for point in POINTS)
    if ratios != (40, 30, 20, 10, 8, 6, 4, 2, 1):
        raise ValueError(f"Unexpected D0-capacity ordering: {ratios}")
    if min(POINTS, key=lambda point: point.wall).ratio != 4:
        raise ValueError("The lowest observed external Wall must occur at 4%")
    if POINTS[0].fs_inputs <= 0 or any(point.fs_inputs != 0 for point in POINTS[1:]):
        raise ValueError("Only the 40% point may retain filesystem input")
    for point in POINTS:
        if not (Y_MIN <= point.gather <= point.persistent <= point.wall <= Y_MAX):
            raise ValueError(f"Unexpected timing relationship at {point.ratio}%")


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


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
    if size < D0_LEGEND_SIZE_PT:
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


def draw_marker(
    ctx: cairo.Context,
    x: float,
    y: float,
    shape: str,
    color: tuple[float, float, float],
    *,
    hollow: bool = False,
    radius: float = 2.25,
    line_width: float = 0.9,
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(line_width)
    if shape == "circle":
        ctx.arc(x, y, radius, 0.0, 6.28318530718)
    elif shape == "square":
        ctx.rectangle(x - radius, y - radius, radius * 2, radius * 2)
    elif shape == "triangle":
        ctx.move_to(x, y - radius - 0.4)
        ctx.line_to(x + radius + 0.4, y + radius)
        ctx.line_to(x - radius - 0.4, y + radius)
        ctx.close_path()
    else:
        raise ValueError(f"Unknown marker shape: {shape}")
    if hollow:
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.fill_preserve()
        set_color(ctx, color)
        ctx.stroke()
    else:
        ctx.fill()
    ctx.restore()


def draw_polyline(
    ctx: cairo.Context,
    coordinates: list[tuple[float, float]],
    *,
    color: tuple[float, float, float],
    dash: tuple[float, ...] = (),
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(1.25)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_dash(dash)
    ctx.move_to(*coordinates[0])
    for coordinate in coordinates[1:]:
        ctx.line_to(*coordinate)
    ctx.stroke()
    ctx.restore()


def draw_legend_item(
    ctx: cairo.Context,
    x: float,
    y: float,
    label: str,
    *,
    color: tuple[float, float, float],
    shape: str,
    dash: tuple[float, ...] = (),
) -> float:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(1.25)
    ctx.set_dash(dash)
    ctx.move_to(x, y - 2.4)
    ctx.line_to(x + 11.0, y - 2.4)
    ctx.stroke()
    ctx.restore()
    draw_marker(ctx, x + 5.5, y - 2.4, shape, color, radius=1.9)
    draw_text(ctx, x + 13.5, y, label, D0_LEGEND_SIZE_PT, color=INK)
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(D0_LEGEND_SIZE_PT)
    return x + 13.5 + ctx.text_extents(label).width + 2.0


def legend_item_width(ctx: cairo.Context, label: str) -> float:
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(D0_LEGEND_SIZE_PT)
    return 13.5 + ctx.text_extents(label).width + 2.0


def draw_figure(ctx: cairo.Context) -> None:
    validate_data()
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    plot_left, plot_right = PLOT_LEFT_PT, PLOT_RIGHT_PT
    plot_top, plot_bottom = (
        THREE_FIGURE_PLOT_TOP_PT,
        THREE_FIGURE_PLOT_BOTTOM_PT,
    )
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    plot_center = (plot_left + plot_right) / 2
    y_ticks = (75, 150, 225, 300, 350)
    y_tick_labels = ("75", "150", "225", "300", "")
    y_axis_title = "Time (s)"
    x_tick_labels = tuple(str(point.ratio) for point in POINTS)
    x_axis_title = "Subtree-task admission-budget ratio (%)"
    x_tick_baseline_y = THREE_FIGURE_PLOT_BOTTOM_PT + 10.0
    data_left = plot_left + 2.5
    data_right = plot_right - 2.5

    def x_for(index: int) -> float:
        return data_left + index * (data_right - data_left) / (len(POINTS) - 1)

    def y_for(value: float) -> float:
        return plot_bottom - (value - Y_MIN) / (Y_MAX - Y_MIN) * plot_height

    legend_items = (
        ("E2E build", WALL_COLOR, "circle", ()),
        ("Input gather", GATHER_COLOR, "square", (4.0, 2.2)),
        ("Persistent kernel", PERSISTENT_COLOR, "triangle", (1.2, 2.0)),
    )
    legend_gap = 2.0
    legend_width = sum(legend_item_width(ctx, item[0]) for item in legend_items)
    legend_width += legend_gap * (len(legend_items) - 1)
    legend_x = plot_center - legend_width / 2
    legend_x = min(max(legend_x, 2.0), WIDTH_PT - legend_width - 2.0)
    for label, color, shape, dash in legend_items:
        legend_x = draw_legend_item(
            ctx,
            legend_x,
            THREE_FIGURE_PLOT_TOP_PT - 4.0,
            label,
            color=color,
            shape=shape,
            dash=dash,
        )
        legend_x += legend_gap

    for tick, tick_label in zip(y_ticks, y_tick_labels):
        y = y_for(float(tick))
        ctx.save()
        set_color(ctx, GRID)
        ctx.set_line_width(GRID_LINE_WIDTH_PT)
        ctx.move_to(plot_left, y)
        ctx.line_to(plot_right, y)
        ctx.stroke()
        ctx.restore()
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
            y_tick_labels,
            y_axis_title,
            tick_size=ROW_TICK_SIZE_PT,
            title_size=ROW_AXIS_LABEL_SIZE_PT,
        ),
        (plot_top + plot_bottom) / 2,
        y_axis_title,
        ROW_AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    series = (
        ("wall", WALL_COLOR, "circle", ()),
        ("gather", GATHER_COLOR, "square", (4.0, 2.2)),
        ("persistent", PERSISTENT_COLOR, "triangle", (1.2, 2.0)),
    )
    for attribute, color, shape, dash in series:
        coordinates = [(x_for(index), y_for(getattr(point, attribute))) for index, point in enumerate(POINTS)]
        draw_polyline(ctx, coordinates, color=color, dash=dash)
        for x, y in coordinates:
            draw_marker(ctx, x, y, shape, color)

    best_index = next(index for index, point in enumerate(POINTS) if point.ratio == 4)
    best_x = x_for(best_index)
    best_y = y_for(POINTS[best_index].wall)
    draw_marker(ctx, best_x, best_y, "circle", WALL_COLOR, radius=3.2)
    draw_text(ctx, best_x, best_y - 7.0, "min. E2E", align="center", bold=True, color=BEST_COLOR)

    for index, (point, tick_label) in enumerate(zip(POINTS, x_tick_labels)):
        x = x_for(index)
        ctx.save()
        set_color(ctx, INK)
        ctx.set_line_width(0.55)
        ctx.move_to(x, plot_bottom)
        ctx.line_to(x, plot_bottom + 2.2)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, x, x_tick_baseline_y, tick_label, align="center")
    draw_text(
        ctx,
        plot_center,
        x_axis_title_baseline_y(
            ctx,
            x_tick_baseline_y,
            x_tick_labels,
            x_axis_title,
            tick_size=ROW_TICK_SIZE_PT,
            title_size=ROW_AXIS_LABEL_SIZE_PT,
        )
        + X_AXIS_TITLE_SHIFT_PT,
        x_axis_title,
        ROW_AXIS_LABEL_SIZE_PT,
        align="center",
    )


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, HEIGHT_PT)
    ctx = cairo.Context(surface)
    ctx.translate(0.0, -THREE_FIGURE_CROP_TOP_PT)
    draw_figure(ctx)
    surface.finish()


def render_png(path: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, round(WIDTH_PT * scale), round(HEIGHT_PT * scale))
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    ctx.translate(0.0, -THREE_FIGURE_CROP_TOP_PT)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "d0_capacity_budget.pdf")
    render_png(output_dir / "d0_capacity_budget.png")


if __name__ == "__main__":
    main()
