#!/usr/bin/env python3
"""Render the LAION35M root-path ablation as paired comparison panels.

Source records:
  /data_staging/wbh/root_partition_ablation/laion35m/
  laion35m-root-kmeans-rbc-20260723-204337/{rbc,kmeans}/
  {time.log,profile.json}
"""

from dataclasses import dataclass
from pathlib import Path

import cairo

from paper_figure_style import (
    AXIS_LABEL_SIZE_PT,
    FONT_FAMILY,
    LEGEND_SIZE_PT,
    PANEL_TITLE_SIZE_PT,
    TICK_SIZE_PT,
    VALUE_SIZE_PT,
)


WIDTH_PT = 6.9 * 72
HEIGHT_PT = 2.35 * 72
PANEL_TITLE_BASELINE_Y = 20.0

INK = (0.0, 0.0, 0.0)
MUTED = (0.38, 0.38, 0.38)
GRID = (0.88, 0.88, 0.88)
RBC_COLOR = (0.38, 0.38, 0.38)
KMEANS_COLOR = (0.000, 0.447, 0.698)


@dataclass(frozen=True)
class Metric:
    name: str
    rbc: float
    kmeans: float


@dataclass(frozen=True)
class TailProfile:
    name: str
    p50: float
    p95: float
    maximum: float
    color: tuple[float, float, float]


TIME_METRICS = (
    Metric("E2E", 383.590, 278.470),
    Metric("D0 stage", 13.509, 35.400),
    Metric("D1+ producer", 206.363, 198.511),
    Metric("Persistent", 143.629, 141.718),
)

TAIL_PROFILES = (
    TailProfile("RBC", 1734640.0, 9909697.0, 13264457.0, RBC_COLOR),
    TailProfile("K-Means", 2834066.0, 6360038.0, 10121300.0, KMEANS_COLOR),
)


def validate_data() -> None:
    if tuple(metric.name for metric in TIME_METRICS) != (
        "E2E",
        "D0 stage",
        "D1+ producer",
        "Persistent",
    ):
        raise ValueError("Unexpected timing metric order")
    if tuple(profile.name for profile in TAIL_PROFILES) != ("RBC", "K-Means"):
        raise ValueError("Unexpected tail-profile order")
    if min(TIME_METRICS[0].rbc, TIME_METRICS[0].kmeans) != TIME_METRICS[0].kmeans:
        raise ValueError("K-Means must have the lower E2E time")
    for profile in TAIL_PROFILES:
        if not (0 < profile.p50 <= profile.p95 <= profile.maximum):
            raise ValueError(f"Invalid child-size quantiles for {profile.name}")
    if TAIL_PROFILES[0].maximum / TAIL_PROFILES[0].p50 <= TAIL_PROFILES[1].maximum / TAIL_PROFILES[1].p50:
        raise ValueError("RBC must have the longer normalized upper tail")


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float = TICK_SIZE_PT,
    *,
    align: str = "left",
    bold: bool = False,
    color: tuple[float, float, float] = INK,
    rotation: float = 0.0,
) -> None:
    if size < LEGEND_SIZE_PT:
        raise ValueError("Figure text must remain at least the shared legend size")
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
    ctx.move_to(offset, 0.0)
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
    width: float = 0.7,
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(width)
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.stroke()
    ctx.restore()


def draw_marker(
    ctx: cairo.Context,
    x: float,
    y: float,
    *,
    shape: str,
    color: tuple[float, float, float],
    radius: float = 3.4,
) -> None:
    ctx.save()
    set_color(ctx, color)
    if shape == "circle":
        ctx.arc(x, y, radius, 0.0, 6.28318530718)
    elif shape == "square":
        ctx.rectangle(x - radius, y - radius, radius * 2, radius * 2)
    elif shape == "diamond":
        ctx.move_to(x, y - radius - 0.4)
        ctx.line_to(x + radius + 0.4, y)
        ctx.line_to(x, y + radius + 0.4)
        ctx.line_to(x - radius - 0.4, y)
        ctx.close_path()
    else:
        raise ValueError(f"Unknown marker shape: {shape}")
    ctx.fill()
    ctx.restore()


def draw_legend(ctx: cairo.Context, center_x: float) -> None:
    labels = ("RBC", "K-Means")
    marker_radius = 3.1
    marker_text_gap = 4.9
    item_gap = 18.0
    ctx.save()
    ctx.select_font_face(FONT_FAMILY, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(LEGEND_SIZE_PT)
    text_widths = tuple(ctx.text_extents(label).x_advance for label in labels)
    ctx.restore()
    item_widths = tuple(marker_radius * 2 + marker_text_gap + width for width in text_widths)
    legend_width = sum(item_widths) + item_gap
    legend_x = center_x - legend_width / 2
    legend_y = 38.0
    for label, shape, color, text_color, item_width in zip(
        labels,
        ("circle", "square"),
        (RBC_COLOR, KMEANS_COLOR),
        (INK, KMEANS_COLOR),
        item_widths,
        strict=True,
    ):
        marker_x = legend_x + marker_radius
        draw_marker(ctx, marker_x, legend_y - 3.0, shape=shape, color=color, radius=marker_radius)
        draw_text(
            ctx,
            legend_x + marker_radius * 2 + marker_text_gap,
            legend_y,
            label,
            LEGEND_SIZE_PT,
            color=text_color,
        )
        legend_x += item_width + item_gap


def fill_rect(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[float, float, float],
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.rectangle(x, y, width, height)
    ctx.fill()
    ctx.restore()


def draw_time_panel(ctx: cairo.Context) -> None:
    plot_left, plot_right = 54.0, 284.0
    plot_top, plot_bottom = 27.0, 137.0
    y_max = 420.0

    def y_for(value: float) -> float:
        return plot_bottom - value / y_max * (plot_bottom - plot_top)

    draw_text(
        ctx,
        (plot_left + plot_right) / 2,
        PANEL_TITLE_BASELINE_Y,
        "(a) Construction time",
        PANEL_TITLE_SIZE_PT,
        align="center",
        bold=True,
    )
    for tick in (0.0, 100.0, 200.0, 300.0, 400.0):
        y = y_for(tick)
        draw_line(ctx, plot_left, y, plot_right, y, color=GRID, width=0.45)
        draw_text(
            ctx,
            plot_left - 5.0,
            y + 2.7,
            f"{tick:.0f}",
            TICK_SIZE_PT,
            align="right",
            color=INK,
        )
    draw_line(ctx, plot_left, plot_top, plot_left, plot_bottom, width=0.65)
    draw_line(ctx, plot_left, plot_bottom, plot_right, plot_bottom, width=0.65)
    draw_text(
        ctx,
        25.0,
        (plot_top + plot_bottom) / 2,
        "Time (s)",
        AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    centers = (82.0, 140.0, 202.0, 260.0)
    bar_width = 15.0
    for center, metric in zip(centers, TIME_METRICS, strict=True):
        rbc_y = y_for(metric.rbc)
        kmeans_y = y_for(metric.kmeans)
        fill_rect(ctx, center - 16.0, rbc_y, bar_width, plot_bottom - rbc_y, RBC_COLOR)
        fill_rect(ctx, center + 1.0, kmeans_y, bar_width, plot_bottom - kmeans_y, KMEANS_COLOR)
        draw_text(
            ctx,
            center - 15.0,
            rbc_y - 4.0,
            f"{metric.rbc:.1f}",
            VALUE_SIZE_PT,
            align="center",
        )
        draw_text(
            ctx,
            center + 15.0,
            kmeans_y - 4.0,
            f"{metric.kmeans:.1f}",
            VALUE_SIZE_PT,
            align="center",
        )
        draw_text(ctx, center, 147.0, metric.name, TICK_SIZE_PT, align="center")


def draw_tail_panel(ctx: cairo.Context) -> None:
    plot_left, plot_right = 333.0, 478.0
    plot_top, plot_bottom = 27.0, 137.0
    y_max = 14.0

    def y_for(value: float) -> float:
        return plot_bottom - value / y_max * (plot_bottom - plot_top)

    draw_text(
        ctx,
        (plot_left + plot_right) / 2,
        PANEL_TITLE_BASELINE_Y,
        "(b) Root-task upper tail",
        PANEL_TITLE_SIZE_PT,
        align="center",
        bold=True,
    )
    for tick in (0.0, 3.0, 6.0, 9.0, 12.0):
        y = y_for(tick)
        draw_line(ctx, plot_left, y, plot_right, y, color=GRID, width=0.45)
        draw_text(
            ctx,
            plot_left - 5.0,
            y + 2.7,
            f"{tick:.0f}",
            TICK_SIZE_PT,
            align="right",
            color=INK,
        )
    draw_line(ctx, plot_left, plot_top, plot_left, plot_bottom, width=0.65)
    draw_line(ctx, plot_left, plot_bottom, plot_right, plot_bottom, width=0.65)
    draw_text(
        ctx,
        304.0,
        (plot_top + plot_bottom) / 2,
        "Root task size (M rows)",
        AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    x_positions = (350.0, 405.5, 461.0)
    for x, label in zip(x_positions, ("P50", "P95", "Max"), strict=True):
        draw_text(ctx, x, 147.0, label, TICK_SIZE_PT, align="center")

    for profile, shape in zip(TAIL_PROFILES, ("circle", "square"), strict=True):
        sizes = (profile.p50 / 1e6, profile.p95 / 1e6, profile.maximum / 1e6)
        coordinates = [(x, y_for(value)) for x, value in zip(x_positions, sizes, strict=True)]
        for first, second in zip(coordinates, coordinates[1:]):
            draw_line(ctx, first[0], first[1], second[0], second[1], color=profile.color, width=1.5)
        for x, y in coordinates:
            draw_marker(ctx, x, y, shape=shape, color=profile.color, radius=3.4)
        end_x, end_y = coordinates[-1]
        label_y = end_y + 11.0 if profile.name == "RBC" else end_y - 5.0
        label_color = INK if profile.name == "RBC" else profile.color
        draw_text(
            ctx,
            end_x,
            label_y,
            f"{sizes[-1]:.2f}M",
            VALUE_SIZE_PT,
            align="center",
            color=label_color,
        )

    draw_legend(ctx, (plot_left + plot_right) / 2)


def draw_figure(ctx: cairo.Context) -> None:
    validate_data()
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    draw_time_panel(ctx)
    draw_tail_panel(ctx)


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, HEIGHT_PT)
    ctx = cairo.Context(surface)
    draw_figure(ctx)
    surface.finish()


def render_png(path: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, round(WIDTH_PT * scale), round(HEIGHT_PT * scale))
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "root_partition_ablation.pdf")
    render_png(output_dir / "root_partition_ablation.png")


if __name__ == "__main__":
    main()
