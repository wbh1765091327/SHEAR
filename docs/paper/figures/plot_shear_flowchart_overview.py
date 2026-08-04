#!/usr/bin/env python3
"""Render the paper-ready SHEAR numbered flowchart overview."""

from __future__ import annotations

import math
from pathlib import Path

import cairo

from paper_figure_style import FONT_FAMILY


WIDTH_PT = 500.0
HEIGHT_PT = 216.0
DPI = 300
MIN_FONT_SIZE_PT = 8.0

OUTPUT_BASENAME = "shear_flowchart_overview"
OUTPUT_DIR = Path(__file__).resolve().parent

RAW_VECTOR_LABEL = "Raw vectors on CPU"
GPU_PREPROCESS_LABEL = "GPU preprocessing"
SQ_ROOT_LABEL = "SQ8 + root assignment"
PREPARED_VECTOR_LABEL = "Prepared vectors + sidecars"
ALL_VECTORS_READY_LABEL = "all vectors ready"
HOST_GATHER_LABEL = "Host gather / preload"
GPU_RESIDENT_SUBTREE_LABEL = "GPU-resident subtree"
TESSA_LABEL = "Persistent TESSA"
TESSA_DETAIL_LABEL = "Subtree recursion + leaf scoring"
COMPACT_LABEL = "Compact records"
COMPACT_SCHEMA_LABEL = "{src, nbr, dist}"
NOT_LOCAL_GRAPH_LABEL = "not a local graph"
D2H_LABEL = "Subtree-level D2H"
HASHPRUNE_LABEL = "CPU HashPrune"
DRAM_LABEL = "DRAM reservoir"
SSD_LABEL = "SSD: spill + external reduce"
FINAL_GRAPH_LABEL = "Bounded-degree graph G"
NEXT_SUBTREE_LABEL = "Next subtree"

INK = (0.04, 0.04, 0.04)
MID_GRAY = (0.38, 0.40, 0.41)
LIGHT_GRAY = (0.72, 0.73, 0.73)
WHITE = (1.0, 1.0, 1.0)
GPU_BG = (1.0, 0.978, 0.91)
HOST_BG = (0.92, 0.98, 0.99)
BLUE = (0.31, 0.57, 0.81)
BLUE_LIGHT = (0.78, 0.89, 0.98)
CYAN_LIGHT = (0.74, 0.91, 0.94)
GREEN = (0.43, 0.66, 0.31)
GREEN_LIGHT = (0.80, 0.91, 0.70)
AMBER = (0.89, 0.57, 0.24)
AMBER_LIGHT = (0.98, 0.79, 0.59)
VIOLET = (0.48, 0.32, 0.67)
VIOLET_LIGHT = (0.89, 0.83, 0.96)
RED = (0.78, 0.13, 0.14)

LANE_SPLIT_Y = 152.0
CONTENT_RIGHT = 464.0
STEP_WIDTH = 137.0
STEP_GAP = 16.0
STEP_LEFTS = tuple(7.0 + index * (STEP_WIDTH + STEP_GAP) for index in range(3))


def set_color(ctx: cairo.Context, color: tuple[float, float, float]) -> None:
    ctx.set_source_rgb(*color)


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    value: str,
    size: float = MIN_FONT_SIZE_PT,
    *,
    align: str = "left",
    bold: bool = False,
    italic: bool = False,
    color: tuple[float, float, float] = INK,
    family: str = FONT_FAMILY,
    rotation: float = 0.0,
) -> None:
    if size < MIN_FONT_SIZE_PT:
        raise ValueError(f"Figure text must remain at least {MIN_FONT_SIZE_PT:.1f} pt")
    ctx.save()
    ctx.translate(x, y)
    if rotation:
        ctx.rotate(rotation)
    ctx.select_font_face(
        family,
        cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    extents = ctx.text_extents(value)
    if align == "center":
        offset_x = -(extents.x_bearing + extents.width / 2)
    elif align == "right":
        offset_x = -(extents.x_bearing + extents.width)
    else:
        offset_x = 0.0
    set_color(ctx, color)
    ctx.move_to(offset_x, 0.0)
    ctx.show_text(value)
    ctx.restore()


def draw_multiline_text(
    ctx: cairo.Context,
    x: float,
    first_baseline_y: float,
    lines: tuple[str, ...],
    size: float,
    *,
    line_gap: float,
    align: str = "center",
    bold: bool = False,
    italic: bool = False,
    color: tuple[float, float, float] = INK,
    family: str = FONT_FAMILY,
) -> None:
    for index, line in enumerate(lines):
        draw_text(
            ctx,
            x,
            first_baseline_y + index * line_gap,
            line,
            size,
            align=align,
            bold=bold,
            italic=italic,
            color=color,
            family=family,
        )


def rounded_rect_path(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
) -> None:
    radius = min(radius, width / 2, height / 2)
    ctx.new_sub_path()
    ctx.arc(x + width - radius, y + radius, radius, -math.pi / 2, 0)
    ctx.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2)
    ctx.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
    ctx.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def draw_box(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: tuple[float, float, float],
    *,
    stroke: tuple[float, float, float] = INK,
    line_width: float = 1.0,
    radius: float = 2.0,
) -> None:
    ctx.save()
    rounded_rect_path(ctx, x, y, width, height, radius)
    set_color(ctx, fill)
    ctx.fill_preserve()
    set_color(ctx, stroke)
    ctx.set_line_width(line_width)
    ctx.stroke()
    ctx.restore()


def draw_line(
    ctx: cairo.Context,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: tuple[float, float, float] = INK,
    width: float = 1.0,
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


def draw_outline_arrow(
    ctx: cairo.Context,
    x1: float,
    x2: float,
    center_y: float,
) -> None:
    body_half_height = 3.2
    head_half_height = 7.2
    head_width = 7.5
    ctx.save()
    ctx.move_to(x1, center_y - body_half_height)
    ctx.line_to(x2 - head_width, center_y - body_half_height)
    ctx.line_to(x2 - head_width, center_y - head_half_height)
    ctx.line_to(x2, center_y)
    ctx.line_to(x2 - head_width, center_y + head_half_height)
    ctx.line_to(x2 - head_width, center_y + body_half_height)
    ctx.line_to(x1, center_y + body_half_height)
    ctx.close_path()
    set_color(ctx, WHITE)
    ctx.fill_preserve()
    set_color(ctx, INK)
    ctx.set_line_width(1.2)
    ctx.stroke()
    ctx.restore()


def draw_vertical_arrow(
    ctx: cairo.Context,
    x: float,
    y1: float,
    y2: float,
    *,
    upward: bool = False,
    color: tuple[float, float, float] = INK,
    width: float = 1.2,
) -> None:
    ctx.save()
    set_color(ctx, color)
    ctx.set_line_width(width)
    if upward:
        ctx.move_to(x, y1)
        ctx.line_to(x, y2 + 4.5)
        ctx.stroke()
        ctx.move_to(x, y2)
        ctx.line_to(x - 3.5, y2 + 6.0)
        ctx.line_to(x + 3.5, y2 + 6.0)
    else:
        ctx.move_to(x, y1)
        ctx.line_to(x, y2 - 4.5)
        ctx.stroke()
        ctx.move_to(x, y2)
        ctx.line_to(x - 3.5, y2 - 6.0)
        ctx.line_to(x + 3.5, y2 - 6.0)
    ctx.close_path()
    ctx.fill()
    ctx.restore()


def draw_dataset_grid(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    columns = 4
    rows = 4
    cell_width = width / columns
    cell_height = height / rows
    for row in range(rows):
        for column in range(columns):
            shade = 0.92 - 0.08 * ((row + column) % 3)
            ctx.save()
            ctx.rectangle(
                x + column * cell_width,
                y + row * cell_height,
                cell_width,
                cell_height,
            )
            ctx.set_source_rgb(0.55 * shade, 0.74 * shade + 0.16, 0.95)
            ctx.fill_preserve()
            set_color(ctx, INK)
            ctx.set_line_width(0.55)
            ctx.stroke()
            ctx.restore()


def draw_sidecar_stack(ctx: cairo.Context, x: float, y: float) -> None:
    for index, fill in enumerate((BLUE_LIGHT, CYAN_LIGHT, BLUE_LIGHT)):
        draw_box(ctx, x + index * 3.0, y + index * 2.0, 18.0, 22.0, fill, line_width=0.7, radius=0.8)
        for row in range(4):
            draw_line(
                ctx,
                x + index * 3.0 + 4.0,
                y + index * 2.0 + 5.0 + row * 4.0,
                x + index * 3.0 + 14.0,
                y + index * 2.0 + 5.0 + row * 4.0,
                color=MID_GRAY,
                width=0.45,
            )


def draw_sketch_ring(ctx: cairo.Context, center_x: float, center_y: float) -> None:
    radius = 11.0
    ctx.save()
    set_color(ctx, WHITE)
    ctx.arc(center_x, center_y, radius, 0, 2 * math.pi)
    ctx.fill_preserve()
    set_color(ctx, INK)
    ctx.set_line_width(1.0)
    ctx.stroke()
    for index in range(10):
        angle = index * 2 * math.pi / 10
        inner = 6.0
        draw_line(
            ctx,
            center_x + math.cos(angle) * inner,
            center_y + math.sin(angle) * inner,
            center_x + math.cos(angle) * radius,
            center_y + math.sin(angle) * radius,
            color=INK,
            width=0.65,
        )
    ctx.arc(center_x, center_y, 5.7, 0, 2 * math.pi)
    set_color(ctx, GPU_BG)
    ctx.fill_preserve()
    set_color(ctx, INK)
    ctx.set_line_width(0.8)
    ctx.stroke()
    ctx.restore()


def draw_tree_task(
    ctx: cairo.Context,
    center_x: float,
    top_y: float,
    *,
    scale: float = 1.0,
) -> None:
    nodes = (
        (center_x, top_y),
        (center_x - 18.0 * scale, top_y + 18.0 * scale),
        (center_x + 18.0 * scale, top_y + 18.0 * scale),
        (center_x - 27.0 * scale, top_y + 37.0 * scale),
        (center_x - 9.0 * scale, top_y + 37.0 * scale),
        (center_x + 9.0 * scale, top_y + 37.0 * scale),
        (center_x + 27.0 * scale, top_y + 37.0 * scale),
    )
    for source, target in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        draw_line(ctx, *nodes[source], *nodes[target], color=MID_GRAY, width=0.85)
    for index, (x, y) in enumerate(nodes):
        ctx.save()
        ctx.arc(x, y, (3.4 if index else 4.1) * scale, 0, 2 * math.pi)
        set_color(ctx, BLUE if index else AMBER)
        ctx.fill_preserve()
        set_color(ctx, INK)
        ctx.set_line_width(0.7)
        ctx.stroke()
        ctx.restore()
    ctx.save()
    set_color(ctx, BLUE)
    ctx.set_line_width(0.8)
    ctx.set_dash((2.0, 2.0))
    ctx.arc(center_x, top_y + 21.0 * scale, 36.0 * scale, 0, 2 * math.pi)
    ctx.stroke()
    ctx.restore()


def draw_matrix_tile(
    ctx: cairo.Context,
    x: float,
    y: float,
    size: float,
) -> None:
    cells = 4
    cell = size / cells
    for row in range(cells):
        for column in range(cells):
            ctx.save()
            ctx.rectangle(x + column * cell, y + row * cell, cell, cell)
            set_color(ctx, CYAN_LIGHT if (row + column) % 3 else BLUE_LIGHT)
            ctx.fill_preserve()
            set_color(ctx, MID_GRAY)
            ctx.set_line_width(0.45)
            ctx.stroke()
            ctx.restore()


def draw_graph_icon(
    ctx: cairo.Context,
    center_x: float,
    center_y: float,
    *,
    scale: float = 1.0,
) -> None:
    offsets = (
        (-18.0, -4.0),
        (-6.0, -14.0),
        (8.0, -5.0),
        (19.0, -13.0),
        (-1.0, 10.0),
        (14.0, 13.0),
    )
    points = tuple((center_x + dx * scale, center_y + dy * scale) for dx, dy in offsets)
    edges = ((0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (2, 5), (4, 5), (3, 5))
    for source, target in edges:
        draw_line(ctx, *points[source], *points[target], color=MID_GRAY, width=0.8)
    for index, (x, y) in enumerate(points):
        ctx.save()
        ctx.arc(x, y, 3.5 * scale, 0, 2 * math.pi)
        set_color(ctx, BLUE_LIGHT if index != 4 else AMBER_LIGHT)
        ctx.fill_preserve()
        set_color(ctx, MID_GRAY)
        ctx.set_line_width(0.7)
        ctx.stroke()
        ctx.restore()


def draw_compact_records(
    ctx: cairo.Context,
    x: float,
    y: float,
    width: float,
    *,
    row_height: float = 8.0,
) -> None:
    fills = (GREEN_LIGHT, AMBER_LIGHT, BLUE_LIGHT)
    for row, fill in enumerate(fills):
        draw_box(
            ctx,
            x + (row % 2) * 2.0,
            y + row * (row_height + 2.2),
            width - (row % 2) * 2.0,
            row_height,
            fill,
            line_width=0.8,
            radius=0.4,
        )
        first_separator = x + width * 0.33
        second_separator = x + width * 0.68
        draw_line(
            ctx,
            first_separator,
            y + row * (row_height + 2.2),
            first_separator,
            y + row * (row_height + 2.2) + row_height,
            color=MID_GRAY,
            width=0.45,
        )
        draw_line(
            ctx,
            second_separator,
            y + row * (row_height + 2.2),
            second_separator,
            y + row * (row_height + 2.2) + row_height,
            color=MID_GRAY,
            width=0.45,
        )


def draw_funnel(ctx: cairo.Context, center_x: float, top_y: float) -> None:
    ctx.save()
    ctx.move_to(center_x - 23.0, top_y)
    ctx.line_to(center_x + 23.0, top_y)
    ctx.line_to(center_x + 7.0, top_y + 16.0)
    ctx.line_to(center_x + 3.0, top_y + 16.0)
    ctx.line_to(center_x + 3.0, top_y + 25.0)
    ctx.line_to(center_x - 3.0, top_y + 25.0)
    ctx.line_to(center_x - 3.0, top_y + 16.0)
    ctx.line_to(center_x - 7.0, top_y + 16.0)
    ctx.close_path()
    set_color(ctx, VIOLET_LIGHT)
    ctx.fill_preserve()
    set_color(ctx, INK)
    ctx.set_line_width(1.0)
    ctx.stroke()
    ctx.restore()


def draw_step_one(ctx: cairo.Context, left: float) -> None:
    center_x = left + STEP_WIDTH / 2
    raw_center_x = left + 31.0
    prepared_center_x = left + 105.0

    draw_box(ctx, left + 15.0, 38.0, 106.0, 52.0, BLUE_LIGHT, line_width=0.9)
    draw_multiline_text(
        ctx,
        center_x,
        56.0,
        (GPU_PREPROCESS_LABEL, "SQ8 + root", "assignment"),
        8.0,
        line_gap=12.0,
        bold=True,
    )

    draw_vertical_arrow(ctx, raw_center_x, 161.0, 93.0, upward=True)
    draw_text(ctx, raw_center_x + 6.0, 136.0, "H2D", 8.0, bold=True, family="Arial")
    draw_vertical_arrow(ctx, prepared_center_x, 93.0, 160.0)
    draw_text(ctx, prepared_center_x + 4.0, 136.0, "D2H", 8.0, bold=True, family="Arial")

    draw_dataset_grid(ctx, left + 8.0, 164.0, 46.0, 24.0)
    draw_multiline_text(
        ctx,
        raw_center_x,
        199.0,
        ("Raw vectors", "on CPU"),
        8.0,
        line_gap=9.0,
        bold=True,
    )

    draw_box(ctx, left + 81.0, 161.0, 48.0, 31.0, WHITE, line_width=0.8)
    draw_sidecar_stack(ctx, left + 92.0, 164.0)
    draw_multiline_text(
        ctx,
        prepared_center_x,
        201.0,
        ("Prepared vectors", "+ sidecars"),
        8.0,
        line_gap=9.0,
        bold=True,
    )


def draw_step_two(ctx: cairo.Context, left: float) -> None:
    center_x = left + 85.0
    draw_tree_task(ctx, center_x, 43.0, scale=0.78)
    draw_text(ctx, center_x, 99.0, GPU_RESIDENT_SUBTREE_LABEL, 8.2, align="center", bold=True)
    draw_vertical_arrow(ctx, center_x, 158.0, 104.0, upward=True)
    draw_text(ctx, center_x + 7.0, 137.0, "H2D", 8.0, bold=True, family="Arial")

    draw_box(ctx, left + 16.0, 159.0, 105.0, 37.0, WHITE, line_width=0.9)
    draw_multiline_text(
        ctx,
        center_x,
        175.0,
        ("Host gather /", "preload"),
        8.0,
        line_gap=11.0,
        bold=True,
    )


def draw_step_three(ctx: cairo.Context, left: float) -> None:
    center_x = left + STEP_WIDTH / 2
    tessa_center_x = left + 20.0
    records_center_x = left + 100.0

    draw_matrix_tile(ctx, tessa_center_x - 18.0, 47.0, 36.0)
    draw_text(ctx, tessa_center_x, 92.0, TESSA_LABEL, 8.2, align="center", bold=True)
    draw_multiline_text(
        ctx,
        tessa_center_x,
        103.0,
        ("recursion +", "leaf scoring"),
        8.0,
        line_gap=10.0,
    )

    draw_compact_records(ctx, records_center_x - 27.0, 48.0, 54.0, row_height=6.0)
    draw_text(ctx, records_center_x, 86.0, COMPACT_LABEL, 8.0, align="center", bold=True)
    draw_text(
        ctx,
        records_center_x,
        97.0,
        COMPACT_SCHEMA_LABEL,
        8.0,
        align="center",
        family="Courier New",
    )
    draw_text(
        ctx,
        records_center_x,
        108.0,
        NOT_LOCAL_GRAPH_LABEL,
        8.0,
        align="center",
        bold=True,
        color=RED,
    )

    draw_box(ctx, left + 13.0, 157.0, 122.0, 23.0, GREEN_LIGHT, line_width=1.0)
    draw_text(ctx, center_x + 5.0, 172.0, HASHPRUNE_LABEL, 8.4, align="center", bold=True)

    draw_line(ctx, center_x + 5.0, 180.0, center_x + 5.0, 184.0, width=0.9)
    draw_line(ctx, left + 43.0, 184.0, left + 105.0, 184.0, width=0.9)
    draw_line(ctx, left + 43.0, 184.0, left + 43.0, 186.0, width=0.9)
    draw_line(ctx, left + 105.0, 184.0, left + 105.0, 186.0, width=0.9)

    draw_box(ctx, left + 13.0, 186.0, 59.0, 12.0, GREEN_LIGHT, line_width=0.8)
    draw_text(ctx, left + 42.5, 195.0, "DRAM", 8.0, align="center", bold=True)
    draw_box(ctx, left + 76.0, 186.0, 59.0, 12.0, VIOLET_LIGHT, line_width=0.8)
    draw_text(ctx, left + 105.5, 195.0, "SSD reduce", 8.0, align="center", bold=True)

    draw_box(ctx, left + 13.0, 201.0, 122.0, 13.0, GREEN_LIGHT, line_width=0.9)
    draw_text(ctx, center_x + 5.0, 211.0, "Bounded-degree G", 8.0, align="center", bold=True)


def draw_preprocessing_barrier(ctx: cairo.Context) -> None:
    source_x = STEP_LEFTS[0] + STEP_WIDTH + 1.0
    target_x = STEP_LEFTS[1] + 16.0
    draw_outline_arrow(ctx, source_x, target_x, 178.0)
    draw_text(
        ctx,
        (source_x + target_x) / 2,
        149.0,
        ALL_VECTORS_READY_LABEL,
        8.0,
        align="center",
        italic=True,
        color=MID_GRAY,
        family="Arial",
    )


def draw_subtree_handoff(ctx: cairo.Context) -> None:
    draw_outline_arrow(
        ctx,
        STEP_LEFTS[1] + 115.0,
        STEP_LEFTS[2] + 1.0,
        64.0,
    )


def draw_tessa_records_handoff(ctx: cairo.Context) -> None:
    draw_outline_arrow(
        ctx,
        STEP_LEFTS[2] + 41.0,
        STEP_LEFTS[2] + 70.0,
        64.0,
    )


def draw_compact_d2h_handoff(ctx: cairo.Context) -> None:
    arrow_x = STEP_LEFTS[2] + STEP_WIDTH - 8.0
    draw_vertical_arrow(ctx, arrow_x, 74.0, 156.0)
    draw_text(
        ctx,
        arrow_x - 4.0,
        146.0,
        "D2H",
        8.0,
        align="right",
        bold=True,
        family="Arial",
    )


def draw_overlap_loop(ctx: cairo.Context) -> None:
    source_x = STEP_LEFTS[2] + 13.0
    target_x = STEP_LEFTS[1] + STEP_WIDTH / 2
    path_y = 213.0
    ctx.save()
    set_color(ctx, INK)
    ctx.set_line_width(0.9)
    ctx.set_dash((3.0, 2.5))
    ctx.move_to(source_x, 177.0)
    ctx.curve_to(source_x - 10.0, 190.0, source_x - 10.0, path_y, source_x - 20.0, path_y)
    ctx.line_to(target_x + 7.0, path_y)
    ctx.curve_to(target_x, path_y, target_x, path_y - 5.0, target_x, path_y - 10.0)
    ctx.line_to(target_x, 201.0)
    ctx.stroke()
    ctx.set_dash(())
    ctx.move_to(target_x, 196.0)
    ctx.line_to(target_x - 3.5, 203.0)
    ctx.line_to(target_x + 3.5, 203.0)
    ctx.close_path()
    ctx.fill()
    ctx.restore()

    label_center_x = (source_x + target_x) / 2
    draw_box(ctx, label_center_x - 27.0, 203.0, 54.0, 12.0, HOST_BG, stroke=HOST_BG, line_width=0.0)
    draw_text(ctx, label_center_x, 212.0, NEXT_SUBTREE_LABEL, 8.0, align="center", italic=True, family="Arial")


def draw_lane_labels(ctx: cairo.Context) -> None:
    draw_line(ctx, CONTENT_RIGHT + 1.0, 0.0, CONTENT_RIGHT + 1.0, HEIGHT_PT, color=LIGHT_GRAY, width=0.6)
    draw_text(ctx, 485.0, 96.0, "GPU", 10.5, align="center", bold=True, family="Arial")
    draw_text(ctx, 485.0, 171.0, "HOST", 10.0, align="center", bold=True, family="Arial")
    draw_text(ctx, 485.0, 184.0, "CPU", 8.0, align="center", bold=True, family="Arial")
    draw_text(ctx, 485.0, 195.0, "DRAM", 8.0, align="center", bold=True, family="Arial")
    draw_text(ctx, 485.0, 206.0, "SSD", 8.0, align="center", bold=True, family="Arial")


def draw_flowchart(ctx: cairo.Context) -> None:
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.save()
    ctx.rectangle(0.0, 0.0, WIDTH_PT, LANE_SPLIT_Y)
    set_color(ctx, GPU_BG)
    ctx.fill()
    ctx.rectangle(0.0, LANE_SPLIT_Y, WIDTH_PT, HEIGHT_PT - LANE_SPLIT_Y)
    set_color(ctx, HOST_BG)
    ctx.fill()
    ctx.restore()

    draw_overlap_loop(ctx)
    draw_step_one(ctx, STEP_LEFTS[0])
    draw_step_two(ctx, STEP_LEFTS[1])
    draw_step_three(ctx, STEP_LEFTS[2])
    draw_preprocessing_barrier(ctx)
    draw_subtree_handoff(ctx)
    draw_tessa_records_handoff(ctx)
    draw_compact_d2h_handoff(ctx)
    draw_lane_labels(ctx)

    draw_line(ctx, 0.0, LANE_SPLIT_Y, WIDTH_PT, LANE_SPLIT_Y, color=WHITE, width=1.2)
    ctx.save()
    ctx.rectangle(0.3, 0.3, WIDTH_PT - 0.6, HEIGHT_PT - 0.6)
    set_color(ctx, (0.86, 0.86, 0.84))
    ctx.set_line_width(0.6)
    ctx.stroke()
    ctx.restore()


def render_pdf(path: Path) -> None:
    surface = cairo.PDFSurface(str(path), WIDTH_PT, HEIGHT_PT)
    surface.set_fallback_resolution(DPI, DPI)
    ctx = cairo.Context(surface)
    draw_flowchart(ctx)
    surface.finish()


def render_svg(path: Path) -> None:
    surface = cairo.SVGSurface(str(path), WIDTH_PT, HEIGHT_PT)
    ctx = cairo.Context(surface)
    draw_flowchart(ctx)
    surface.finish()


def render_png(path: Path) -> None:
    scale = DPI / 72.0
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32,
        round(WIDTH_PT * scale),
        round(HEIGHT_PT * scale),
    )
    ctx = cairo.Context(surface)
    ctx.scale(scale, scale)
    draw_flowchart(ctx)
    surface.write_to_png(str(path))
    surface.finish()


def main() -> None:
    render_pdf(OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf")
    render_svg(OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg")
    render_png(OUTPUT_DIR / f"{OUTPUT_BASENAME}.png")


if __name__ == "__main__":
    main()
