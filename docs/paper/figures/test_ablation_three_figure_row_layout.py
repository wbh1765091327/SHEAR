#!/usr/bin/env python3
"""Regression checks for the D0/TESSA/persistent three-figure row."""

import runpy
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURE_DIR = Path(__file__).resolve().parent


def script_values(name: str) -> dict[str, object]:
    return runpy.run_path(FIGURE_DIR / name)


def pdf_text_bounds(name: str) -> tuple[float, float, float, float]:
    with tempfile.NamedTemporaryFile(suffix=".html") as output:
        subprocess.run(
            ["pdftotext", "-bbox", str(FIGURE_DIR / name), output.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        root = ET.parse(output.name).getroot()
    boxes = [
        tuple(float(item.get(key)) for key in ("xMin", "yMin", "xMax", "yMax"))
        for item in root.iter()
        if item.tag.endswith("word")
    ]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def pdf_word_box(name: str, word: str) -> tuple[float, float, float, float]:
    with tempfile.NamedTemporaryFile(suffix=".html") as output:
        subprocess.run(
            ["pdftotext", "-bbox", str(FIGURE_DIR / name), output.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        root = ET.parse(output.name).getroot()
    for item in root.iter():
        if item.tag.endswith("word") and item.text == word:
            return tuple(float(item.get(key)) for key in ("xMin", "yMin", "xMax", "yMax"))
    raise AssertionError(f"{word!r} not found in {name}")


class AblationThreeFigureRowLayoutTest(unittest.TestCase):
    def test_all_three_pdfs_have_matching_displayed_heights(self):
        d0 = script_values("plot_d0_capacity_budget.py")
        tessa = script_values("plot_tessa_ablation.py")
        persistent = script_values("plot_persistent_two_stage.py")
        heights = (
            d0["HEIGHT_PT"] * 0.34 / d0["WIDTH_PT"],
            tessa["OUTPUT_HEIGHT_PT"] * 0.34 / tessa["WIDTH_PT"],
            persistent["OUTPUT_HEIGHT_PT"] * 0.28 / persistent["WIDTH_PT"],
        )

        self.assertLess(max(heights) / min(heights) - 1.0, 0.001)
        self.assertGreater(min(heights), 0.170)

    def test_d0_legend_uses_the_shared_row_size(self):
        values = script_values("plot_d0_capacity_budget.py")
        self.assertEqual(values["D0_LEGEND_SIZE_PT"], values["ROW_LEGEND_SIZE_PT"])

    def test_plot_frame_side_margins_match_after_latex_scaling(self):
        d0 = script_values("plot_d0_capacity_budget.py")
        tessa = script_values("plot_tessa_ablation.py")
        persistent = script_values("plot_persistent_two_stage.py")
        items = (
            (d0, 0.34),
            (tessa, 0.34),
            (persistent, 0.28),
        )
        left_margins = tuple(
            item.get("PLOT_LEFT_PT", 39.0) * latex_width / item["WIDTH_PT"]
            for item, latex_width in items
        )
        right_margins = tuple(
            (item["WIDTH_PT"] - item.get("PLOT_RIGHT_PT", item["WIDTH_PT"] - 5.0))
            * latex_width
            / item["WIDTH_PT"]
            for item, latex_width in items
        )

        self.assertLess(max(left_margins) - min(left_margins), 0.001)
        self.assertLess(max(right_margins) - min(right_margins), 0.0001)

    def test_y_axis_titles_are_shifted_inside_the_pdf(self):
        for script_name in ("plot_tessa_ablation.py", "plot_persistent_two_stage.py"):
            with self.subTest(script_name=script_name):
                values = script_values(script_name)
                self.assertGreaterEqual(values["Y_AXIS_TITLE_SHIFT_PT"], 0.0)

    def test_rendered_text_has_safe_outer_clearance(self):
        d0 = script_values("plot_d0_capacity_budget.py")
        bounds = {
            name: pdf_text_bounds(name)
            for name in ("d0_capacity_budget.pdf", "tessa_ablation.pdf", "persistent_two_stage.pdf")
        }
        self.assertGreaterEqual(bounds["d0_capacity_budget.pdf"][1], 0.0)
        self.assertGreaterEqual(d0["HEIGHT_PT"] - bounds["d0_capacity_budget.pdf"][3], 2.0)
        self.assertGreaterEqual(bounds["tessa_ablation.pdf"][0], 1.0)
        self.assertGreaterEqual(bounds["persistent_two_stage.pdf"][0], 1.0)

    def test_d0_tick_labels_clear_the_x_axis(self):
        values = script_values("plot_d0_capacity_budget.py")
        plot_bottom = values["THREE_FIGURE_PLOT_BOTTOM_PT"] - values["THREE_FIGURE_CROP_TOP_PT"]
        tick_top = pdf_word_box("d0_capacity_budget.pdf", "40")[1]

        self.assertGreaterEqual(tick_top - plot_bottom, 0.5)


if __name__ == "__main__":
    unittest.main()
