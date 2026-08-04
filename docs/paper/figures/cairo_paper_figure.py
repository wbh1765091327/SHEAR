#!/usr/bin/env python3
"""Measured text geometry shared by Cairo paper figures."""

from collections.abc import Sequence

import cairo

from paper_figure_style import (
    AXIS_LABEL_GAP_PT,
    AXIS_LABEL_SIZE_PT,
    FONT_FAMILY,
    TICK_LABEL_PAD_PT,
    TICK_SIZE_PT,
)


def text_extents(
    ctx: cairo.Context,
    value: str,
    size: float,
    *,
    bold: bool = False,
) -> cairo.TextExtents:
    """Return paper-font text bounds without changing the caller's Cairo state."""
    ctx.save()
    ctx.select_font_face(
        FONT_FAMILY,
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    extents = ctx.text_extents(value)
    ctx.restore()
    return extents


def y_axis_title_anchor_x(
    ctx: cairo.Context,
    plot_left: float,
    tick_labels: Sequence[str],
    title: str,
    *,
    tick_size: float = TICK_SIZE_PT,
    title_size: float = AXIS_LABEL_SIZE_PT,
) -> float:
    """Place a -90-degree title at a fixed visible gap from y tick labels."""
    if not tick_labels:
        raise ValueError("tick_labels must not be empty")
    widest_tick = max(text_extents(ctx, label, tick_size).width for label in tick_labels)
    tick_ink_left = plot_left - TICK_LABEL_PAD_PT - widest_tick
    title_extents = text_extents(ctx, title, title_size)
    rotated_title_right_offset = title_extents.y_bearing + title_extents.height
    return tick_ink_left - AXIS_LABEL_GAP_PT - rotated_title_right_offset


def x_axis_title_baseline_y(
    ctx: cairo.Context,
    tick_baseline_y: float,
    tick_labels: Sequence[str],
    title: str,
    *,
    tick_size: float = TICK_SIZE_PT,
    title_size: float = AXIS_LABEL_SIZE_PT,
) -> float:
    """Place a horizontal title at a fixed visible gap below x tick labels."""
    if not tick_labels:
        raise ValueError("tick_labels must not be empty")
    tick_ink_bottom = tick_baseline_y + max(
        text_extents(ctx, label, tick_size).y_bearing
        + text_extents(ctx, label, tick_size).height
        for label in tick_labels
    )
    title_extents = text_extents(ctx, title, title_size)
    return tick_ink_bottom + AXIS_LABEL_GAP_PT - title_extents.y_bearing
