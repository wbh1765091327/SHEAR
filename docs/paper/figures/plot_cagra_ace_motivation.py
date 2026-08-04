#!/usr/bin/env python3
"""Render the LAION100M CAGRA/ACE motivation figure.

The values below are copied from:
  /data_staging/wbh/laion100m_cagra_ace_motivation/comparison_metrics.json
and the remapping-time cgroup memory.stat snapshot in the same experiment.
"""

from pathlib import Path

import cairo

from cairo_paper_figure import text_extents
from paper_figure_style import (
    AXIS_LABEL_GAP_PT,
    AXIS_LABEL_SIZE_PT,
    FONT_FAMILY,
    LEGEND_SIZE_PT,
    PANEL_TITLE_SIZE_PT,
    TICK_LABEL_PAD_PT,
    TICK_SIZE_PT,
    VALUE_SIZE_PT,
)


WIDTH_PT = 6.9 * 72.0
HEIGHT_PT = 190.0
MOTIVATION_FONT_SCALE = 1.5
MOTIVATION_AXIS_LABEL_SIZE_PT = AXIS_LABEL_SIZE_PT * MOTIVATION_FONT_SCALE
MOTIVATION_LEGEND_SIZE_PT = LEGEND_SIZE_PT * MOTIVATION_FONT_SCALE
MOTIVATION_PANEL_TITLE_SIZE_PT = PANEL_TITLE_SIZE_PT * MOTIVATION_FONT_SCALE
MOTIVATION_TICK_SIZE_PT = TICK_SIZE_PT * MOTIVATION_FONT_SCALE
MOTIVATION_VALUE_SIZE_PT = VALUE_SIZE_PT * MOTIVATION_FONT_SCALE

CANVAS_OFFSET_X_PT = 5.0
CANVAS_OFFSET_Y_PT = 22.5

PLOT_TOP = 54.0
PLOT_BOTTOM = 175.0
PANEL_TITLE_BASELINE_Y = 38.0
PANEL_TITLE_SAFE_MARGIN_PT = 1.0
X_TICK_BASELINE_Y = 194.0

TIME_PLOT_WIDTH_PT = 195.0
DIAGNOSTIC_PLOT_WIDTH_PT = 170.0
TIME_PLOT_LEFT_PT = 54.0
TIME_PLOT_RIGHT_PT = TIME_PLOT_LEFT_PT + TIME_PLOT_WIDTH_PT
DIAGNOSTIC_PLOT_RIGHT_PT = 489.0
DIAGNOSTIC_PLOT_LEFT_PT = DIAGNOSTIC_PLOT_RIGHT_PT - DIAGNOSTIC_PLOT_WIDTH_PT
DIAGNOSTIC_LABEL_RIGHT_PT = DIAGNOSTIC_PLOT_LEFT_PT - 4.0
LEGEND_X_PT = 193.0

TIME_PANEL_TITLE = "(a) End-to-end time amplification"
DIAGNOSTIC_PANEL_TITLE = "(b) Host-memory-pressure diagnostics"

COLORS = {
    "data_prep": (0.00, 0.47, 0.73),
    "gpu_build": (0.00, 0.60, 0.53),
    "id_remap": (0.93, 0.47, 0.20),
    "other_host": (0.73, 0.73, 0.73),
    "diagnostic_mid": (0.20, 0.73, 0.93),
    "diagnostic_dark": (0.80, 0.20, 0.07),
    "black": (0.08, 0.08, 0.08),
    "grid": (0.87, 0.87, 0.87),
}

BASELINE = {
    "e2e": 1723.500,
    "data_prep": 754.106,
    "gpu_build": 557.395,
    "id_remap": 234.634,
}

LIMITED = {
    "e2e": 4642.000,
    "data_prep": 985.043,
    "gpu_build": 568.239,
    "id_remap": 2849.852,
}

RESOURCE_RATIOS = [
    ("FS read", 1.5223781298, COLORS["data_prep"]),
    ("FS write", 1.2074419846, COLORS["diagnostic_mid"]),
    ("System CPU time", 3.3147324759, COLORS["id_remap"]),
    ("Vol. ctx switch", 7.8284618493, COLORS["diagnostic_dark"]),
]


def validate_layout() -> None:
    time_width = TIME_PLOT_RIGHT_PT - TIME_PLOT_LEFT_PT
    diagnostic_width = DIAGNOSTIC_PLOT_RIGHT_PT - DIAGNOSTIC_PLOT_LEFT_PT
    if abs(time_width - TIME_PLOT_WIDTH_PT) > 1.0e-9:
        raise ValueError("Unexpected time-panel plot width")
    if abs(diagnostic_width - DIAGNOSTIC_PLOT_WIDTH_PT) > 1.0e-9:
        raise ValueError("Unexpected diagnostic-panel plot width")
    if not (TIME_PLOT_LEFT_PT <= LEGEND_X_PT < TIME_PLOT_RIGHT_PT):
        raise ValueError("The time-panel legend must remain inside panel (a)")


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


def motivation_y_axis_title_anchor_x(
    ctx: cairo.Context,
    plot_left: float,
    tick_labels: tuple[str, ...],
    title: str,
) -> float:
    widest_tick = max(
        text_extents(ctx, label, MOTIVATION_TICK_SIZE_PT).width for label in tick_labels
    )
    tick_ink_left = plot_left - TICK_LABEL_PAD_PT - widest_tick
    title_extents = text_extents(ctx, title, MOTIVATION_AXIS_LABEL_SIZE_PT)
    rotated_title_right_offset = title_extents.y_bearing + title_extents.height
    return tick_ink_left - AXIS_LABEL_GAP_PT - rotated_title_right_offset


def motivation_x_axis_title_baseline_y(
    ctx: cairo.Context,
    tick_baseline_y: float,
    tick_labels: tuple[str, ...],
    title: str,
) -> float:
    tick_ink_bottom = tick_baseline_y + max(
        text_extents(ctx, label, MOTIVATION_TICK_SIZE_PT).y_bearing
        + text_extents(ctx, label, MOTIVATION_TICK_SIZE_PT).height
        for label in tick_labels
    )
    title_extents = text_extents(ctx, title, MOTIVATION_AXIS_LABEL_SIZE_PT)
    return tick_ink_bottom + AXIS_LABEL_GAP_PT - title_extents.y_bearing


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float = MOTIVATION_TICK_SIZE_PT,
    *,
    align: str = "left",
    bold: bool = False,
    color: tuple[float, float, float] = COLORS["black"],
    rotation: float = 0.0,
) -> None:
    if size < MOTIVATION_LEGEND_SIZE_PT:
        raise ValueError("Figure text must remain at least the motivation legend size")
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
        offset = 0
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
    color: tuple[float, float, float] = COLORS["black"],
    width: float = 0.6,
    dash: tuple[float, ...] = (),
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(width)
    ctx.set_dash(dash)
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
    *,
    stroke: bool = True,
) -> None:
    ctx.save()
    ctx.rectangle(x, y, width, height)
    set_color(ctx, color)
    ctx.fill_preserve()
    if stroke:
        set_color(ctx, (0.20, 0.20, 0.20))
        ctx.set_line_width(0.45)
        ctx.stroke()
    else:
        ctx.new_path()
    ctx.restore()


def panel_title_center_x(
    ctx: cairo.Context,
    preferred_center_x: float,
    title: str,
) -> float:
    title_width = text_extents(
        ctx,
        title,
        MOTIVATION_PANEL_TITLE_SIZE_PT,
        bold=True,
    ).width
    half_width = title_width / 2.0
    visible_left = CANVAS_OFFSET_X_PT + PANEL_TITLE_SAFE_MARGIN_PT
    visible_right = CANVAS_OFFSET_X_PT + WIDTH_PT - PANEL_TITLE_SAFE_MARGIN_PT
    if visible_right - visible_left < title_width:
        raise ValueError(f"Panel title does not fit the canvas: {title}")
    return min(
        max(preferred_center_x, visible_left + half_width),
        visible_right - half_width,
    )


def draw_panel_title(
    ctx: cairo.Context,
    center_x: float,
    title: str,
) -> None:
    draw_text(
        ctx,
        panel_title_center_x(ctx, center_x, title),
        PANEL_TITLE_BASELINE_Y,
        title,
        MOTIVATION_PANEL_TITLE_SIZE_PT,
        align="center",
        bold=True,
    )


def draw_diagnostic_label(
    ctx: cairo.Context,
    label_right: float,
    row_y: float,
    label: str,
) -> None:
    lines = {
        "System CPU time": ("System CPU", "time"),
        "Vol. ctx switch": ("Vol. ctx", "switch"),
    }.get(label, (label,))
    if len(lines) == 1:
        draw_text(
            ctx,
            label_right,
            row_y + 11.0,
            lines[0],
            MOTIVATION_TICK_SIZE_PT,
            align="right",
        )
        return
    for baseline_y, line in zip((row_y + 4.0, row_y + 16.0), lines, strict=True):
        draw_text(
            ctx,
            label_right,
            baseline_y,
            line,
            MOTIVATION_TICK_SIZE_PT,
            align="right",
        )


def draw_time_panel(ctx: cairo.Context) -> None:
    plot_left, plot_right = TIME_PLOT_LEFT_PT, TIME_PLOT_RIGHT_PT
    plot_top, plot_bottom = PLOT_TOP, PLOT_BOTTOM
    plot_height = plot_bottom - plot_top
    max_time = 5200.0
    y_ticks = tuple(range(0, 5001, 1000))
    y_tick_labels = tuple(f"{tick:,}" for tick in y_ticks)
    y_axis_title = "End-to-end time (s)"
    x_tick_labels = ("Unlimited", "50%")
    x_axis_title = "Host-memory budget"

    time_panel_center = (plot_left + plot_right) / 2
    draw_panel_title(ctx, time_panel_center, TIME_PANEL_TITLE)

    for tick, tick_label in zip(y_ticks, y_tick_labels):
        y = plot_bottom - tick / max_time * plot_height
        draw_line(ctx, plot_left, y, plot_right, y, color=COLORS["grid"], width=0.45)
        draw_text(
            ctx,
            plot_left - TICK_LABEL_PAD_PT,
            y + 2.7,
            tick_label,
            MOTIVATION_TICK_SIZE_PT,
            align="right",
        )

    draw_line(ctx, plot_left, plot_top, plot_left, plot_bottom, width=0.65)
    draw_line(ctx, plot_left, plot_bottom, plot_right, plot_bottom, width=0.65)
    draw_text(
        ctx,
        motivation_y_axis_title_anchor_x(ctx, plot_left, y_tick_labels, y_axis_title),
        (plot_top + plot_bottom) / 2,
        y_axis_title,
        MOTIVATION_AXIS_LABEL_SIZE_PT,
        align="center",
        rotation=-1.57079632679,
    )

    bar_centers = [97.0, 165.0]
    bar_width = 38.0
    runs = [BASELINE, LIMITED]
    stage_order = ["data_prep", "gpu_build", "id_remap", "other_host"]

    for center, run, tick_label in zip(bar_centers, runs, x_tick_labels):
        stages = dict(run)
        stages["other_host"] = run["e2e"] - run["data_prep"] - run["gpu_build"] - run["id_remap"]
        bottom_value = 0.0
        for stage in stage_order:
            value = stages[stage]
            height = value / max_time * plot_height
            y = plot_bottom - (bottom_value + value) / max_time * plot_height
            fill_rect(ctx, center - bar_width / 2, y, bar_width, height, COLORS[stage])

            bottom_value += value

        top_y = plot_bottom - run["e2e"] / max_time * plot_height
        total_label = "2.69×" if run is LIMITED else None
        if total_label is not None:
            draw_text(
                ctx,
                center,
                top_y - 4.0,
                total_label,
                MOTIVATION_VALUE_SIZE_PT,
                align="center",
                bold=True,
            )

        draw_text(
            ctx,
            center,
            X_TICK_BASELINE_Y,
            tick_label,
            MOTIVATION_TICK_SIZE_PT,
            align="center",
        )

    draw_text(
        ctx,
        (plot_left + plot_right) / 2,
        motivation_x_axis_title_baseline_y(
            ctx,
            X_TICK_BASELINE_Y,
            x_tick_labels,
            x_axis_title,
        ),
        x_axis_title,
        MOTIVATION_AXIS_LABEL_SIZE_PT,
        align="center",
    )

    legend = (
        (LEGEND_X_PT, 77.0, "Data prep", COLORS["data_prep"]),
        (LEGEND_X_PT, 96.0, "GPU build", COLORS["gpu_build"]),
        (LEGEND_X_PT, 115.0, "ID remap", COLORS["id_remap"]),
        (LEGEND_X_PT, 134.0, "Other host", COLORS["other_host"]),
    )
    for x, baseline_y, label, color in legend:
        fill_rect(ctx, x, baseline_y - 10.5, 10.5, 10.5, color)
        draw_text(ctx, x + 14.0, baseline_y, label, MOTIVATION_LEGEND_SIZE_PT)


def draw_diagnostic_panel(ctx: cairo.Context) -> None:
    label_right = DIAGNOSTIC_LABEL_RIGHT_PT
    plot_left, plot_right = DIAGNOSTIC_PLOT_LEFT_PT, DIAGNOSTIC_PLOT_RIGHT_PT
    plot_top, plot_bottom = PLOT_TOP, PLOT_BOTTOM
    max_ratio = 8.7
    x_ticks = (0, 2, 4, 6, 8)
    x_tick_labels = tuple(str(tick) for tick in x_ticks)
    x_axis_title = "Relative to unlimited"

    diagnostic_panel_center = (plot_left + plot_right) / 2
    draw_panel_title(ctx, diagnostic_panel_center, DIAGNOSTIC_PANEL_TITLE)

    for tick, tick_label in zip(x_ticks, x_tick_labels):
        x = plot_left + tick / max_ratio * (plot_right - plot_left)
        draw_line(ctx, x, plot_top, x, plot_bottom, color=COLORS["grid"], width=0.45)
        draw_text(
            ctx,
            x,
            X_TICK_BASELINE_Y,
            tick_label,
            MOTIVATION_TICK_SIZE_PT,
            align="center",
        )

    baseline_x = plot_left + 1.0 / max_ratio * (plot_right - plot_left)
    draw_line(ctx, baseline_x, plot_top, baseline_x, plot_bottom, width=0.8, dash=(2.3, 2.0))
    draw_line(ctx, plot_left, plot_top, plot_left, plot_bottom, width=0.65)

    row_y = (63.0, 91.0, 119.0, 147.0)
    bar_height = 14.0
    for y, (label, ratio, color) in zip(row_y, RESOURCE_RATIOS):
        draw_diagnostic_label(ctx, label_right, y, label)
        width = ratio / max_ratio * (plot_right - plot_left)
        fill_rect(ctx, plot_left, y, width, bar_height, color)
        if plot_left + width + 27.0 <= plot_right:
            draw_text(
                ctx,
                plot_left + width + 3.0,
                y + 9.0,
                f"{ratio:.2f}×",
                MOTIVATION_VALUE_SIZE_PT,
                bold=True,
            )
        else:
            draw_text(
                ctx,
                plot_left + width - 2.0,
                y + 9.0,
                f"{ratio:.2f}×",
                MOTIVATION_VALUE_SIZE_PT,
                align="right",
                bold=True,
                color=(1.0, 1.0, 1.0),
            )

    draw_line(ctx, plot_left, plot_bottom, plot_right, plot_bottom, width=0.65)
    title_baseline = motivation_x_axis_title_baseline_y(
        ctx,
        X_TICK_BASELINE_Y,
        x_tick_labels,
        x_axis_title,
    )
    draw_text(
        ctx,
        (plot_left + plot_right) / 2,
        title_baseline,
        x_axis_title,
        MOTIVATION_AXIS_LABEL_SIZE_PT,
        align="center",
    )


def draw_figure(ctx: cairo.Context) -> None:
    validate_layout()
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    set_color(ctx, (1.0, 1.0, 1.0))
    ctx.paint()
    draw_time_panel(ctx)
    draw_diagnostic_panel(ctx)


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, HEIGHT_PT)
    ctx = cairo.Context(surface)
    ctx.translate(-CANVAS_OFFSET_X_PT, -CANVAS_OFFSET_Y_PT)
    draw_figure(ctx)
    surface.finish()


def render_png(path: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, round(WIDTH_PT * scale), round(HEIGHT_PT * scale))
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    ctx.translate(-CANVAS_OFFSET_X_PT, -CANVAS_OFFSET_Y_PT)
    draw_figure(ctx)
    surface.write_to_png(str(path))


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    render_pdf(output_dir / "cagra_ace_motivation.pdf")
    render_png(output_dir / "cagra_ace_motivation.png")


if __name__ == "__main__":
    main()
