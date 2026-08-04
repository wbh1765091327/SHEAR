#!/usr/bin/env python3
"""Regression checks for the aligned compact three-figure row."""

import ast
import runpy
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURE_DIR = Path(__file__).resolve().parent
SCRIPT_NAMES = (
    "plot_laion100m_hashprune_policy_recall_qps.py",
    "plot_reuse_aware_residency.py",
    "plot_laion100m_reduction_residency.py",
)


def numeric_assignments(script_name: str) -> dict[str, float]:
    tree = ast.parse((FIGURE_DIR / script_name).read_text(encoding="utf-8"))
    values = runpy.run_path(FIGURE_DIR / "paper_figure_style.py")
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                value = eval(
                    compile(ast.Expression(node.value), script_name, "eval"),
                    {"__builtins__": {}},
                    values,
                )
            except (NameError, TypeError):
                continue
            if isinstance(value, (int, float)):
                values[target.id] = float(value)
    return values


def word_box(pdf_name: str, word: str) -> tuple[float, float, float, float]:
    with tempfile.NamedTemporaryFile(suffix=".html") as output:
        subprocess.run(
            ["pdftotext", "-bbox", str(FIGURE_DIR / pdf_name), output.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        root = ET.parse(output.name).getroot()
    for item in root.iter():
        if item.tag.endswith("word") and item.text == word:
            return tuple(float(item.get(name)) for name in ("xMin", "yMin", "xMax", "yMax"))
    raise AssertionError(f"{word!r} not found in {pdf_name}")


def pdf_text_bounds(pdf_name: str) -> tuple[float, float, float, float]:
    with tempfile.NamedTemporaryFile(suffix=".html") as output:
        subprocess.run(
            ["pdftotext", "-bbox", str(FIGURE_DIR / pdf_name), output.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        root = ET.parse(output.name).getroot()
    boxes = [
        tuple(float(item.get(name)) for name in ("xMin", "yMin", "xMax", "yMax"))
        for item in root.iter()
        if item.tag.endswith("word")
    ]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def displayed_y_center(pdf_name: str, word: str, latex_width: float, source_width: float) -> float:
    _, y_min, _, y_max = word_box(pdf_name, word)
    return (y_min + y_max) / 2 * latex_width / source_width


def displayed_y_min(pdf_name: str, word: str, latex_width: float, source_width: float) -> float:
    _, y_min, _, _ = word_box(pdf_name, word)
    return y_min * latex_width / source_width


class ThreeFigureRowLayoutTest(unittest.TestCase):
    def test_sparse_residency_source_uses_the_compact_canvas_width(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)

        self.assertLess(constants[1]["WIDTH_PT"], constants[0]["FIGURE_WIDTH_PT"])
        self.assertLess(constants[1]["WIDTH_PT"], constants[2]["WIDTH_PT"])

    def test_all_three_pdfs_have_the_same_displayed_height(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        latex_widths = (0.349, 0.269, 0.342)
        source_widths = (
            constants[0]["FIGURE_WIDTH_PT"],
            constants[1]["WIDTH_PT"],
            constants[2]["WIDTH_PT"],
        )
        source_heights = (
            constants[0]["FIGURE_HEIGHT_PT"],
            constants[1]["OUTPUT_HEIGHT_PT"],
            constants[2]["HEIGHT_PT"],
        )
        displayed_heights = tuple(
            height * latex_width / width
            for height, latex_width, width in zip(source_heights, latex_widths, source_widths)
        )

        self.assertLess(max(displayed_heights) / min(displayed_heights) - 1.0, 0.001)
        self.assertGreater(min(displayed_heights), 0.151)

    def test_plot_frames_have_the_same_displayed_height(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        latex_widths = (0.349, 0.269, 0.342)
        source_widths = (
            constants[0]["FIGURE_WIDTH_PT"],
            constants[1]["WIDTH_PT"],
            constants[2]["WIDTH_PT"],
        )
        displayed_plot_heights = tuple(
            (item["PLOT_BOTTOM_PT"] - item["PLOT_TOP_PT"]) * latex_width / width
            for item, latex_width, width in zip(constants, latex_widths, source_widths)
        )

        self.assertLess(displayed_plot_heights[0], min(displayed_plot_heights[1:]))
        self.assertGreater(displayed_plot_heights[0] / min(displayed_plot_heights[1:]), 0.80)
        self.assertLess(
            max(displayed_plot_heights[1:]) / min(displayed_plot_heights[1:]) - 1.0,
            0.002,
        )

    def test_plot_frame_tops_are_aligned_after_latex_scaling(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        latex_widths = (0.349, 0.269, 0.342)
        source_widths = (
            constants[0]["FIGURE_WIDTH_PT"],
            constants[1]["WIDTH_PT"],
            constants[2]["WIDTH_PT"],
        )
        displayed_top_margins = tuple(
            item["PLOT_TOP_PT"] * latex_width / width
            for item, latex_width, width in zip(constants, latex_widths, source_widths)
        )

        self.assertGreater(displayed_top_margins[0], max(displayed_top_margins[1:]))
        self.assertLess(abs(displayed_top_margins[1] - displayed_top_margins[2]), 0.0002)

    def test_visible_header_content_is_aligned_after_latex_scaling(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        header_positions = (
            displayed_y_min(
                "laion100m_hashprune_policy_recall_qps.pdf",
                "Plain",
                0.349,
                constants[0]["FIGURE_WIDTH_PT"],
            ),
            displayed_y_min(
                "laion100m_reduction_residency.pdf",
                "Pre",
                0.342,
                constants[2]["WIDTH_PT"],
            ),
        )

        self.assertLess(max(header_positions) - min(header_positions), 0.003)

    def test_three_figure_row_uses_enlarged_typography(self):
        style = runpy.run_path(FIGURE_DIR / "paper_figure_style.py")
        minimum_sizes = (
            ("THREE_FIGURE_TICK_SIZE_PT", 9.2),
            ("THREE_FIGURE_AXIS_LABEL_SIZE_PT", 9.8),
            ("THREE_FIGURE_PANEL_TITLE_SIZE_PT", 9.2),
            ("THREE_FIGURE_VALUE_SIZE_PT", 9.2),
            ("THREE_FIGURE_LEGEND_SIZE_PT", 8.4),
        )
        for row_name, minimum_size in minimum_sizes:
            with self.subTest(row_name=row_name):
                self.assertGreaterEqual(style.get(row_name, 0.0), minimum_size)

    def test_enlarged_text_is_not_clipped_by_pdf_edges(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        pdf_names = (
            "laion100m_hashprune_policy_recall_qps.pdf",
            "reuse_aware_residency.pdf",
            "laion100m_reduction_residency.pdf",
        )
        source_heights = (
            constants[0]["FIGURE_HEIGHT_PT"],
            constants[1]["OUTPUT_HEIGHT_PT"],
            constants[2]["HEIGHT_PT"],
        )
        maximum_bottom_margins = (6.5, 5.0, 5.0)
        for pdf_name, source_height, maximum_bottom_margin in zip(
            pdf_names, source_heights, maximum_bottom_margins
        ):
            with self.subTest(pdf_name=pdf_name):
                x_min, y_min, _, y_max = pdf_text_bounds(pdf_name)
                self.assertGreaterEqual(x_min, -0.25)
                self.assertGreaterEqual(y_min, 0.0)
                self.assertGreaterEqual(source_height - y_max, 3.5)
                self.assertLessEqual(source_height - y_max, maximum_bottom_margin)

    def test_plot_bottoms_are_aligned_after_latex_scaling(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        latex_widths = (0.349, 0.269, 0.342)
        source_widths = (
            constants[0]["FIGURE_WIDTH_PT"],
            constants[1]["WIDTH_PT"],
            constants[2]["WIDTH_PT"],
        )
        plot_bottoms = tuple(
            item["PLOT_BOTTOM_PT"] * latex_width / source_width
            for item, latex_width, source_width in zip(constants, latex_widths, source_widths)
        )

        self.assertLess(max(plot_bottoms) - min(plot_bottoms), 0.0002)

    def test_bottom_axis_text_is_aligned_after_latex_scaling(self):
        constants = tuple(numeric_assignments(name) for name in SCRIPT_NAMES)
        title_centers = (
            displayed_y_center(
                "laion100m_hashprune_policy_recall_qps.pdf",
                "Recall@10",
                0.349,
                constants[0]["FIGURE_WIDTH_PT"],
            ),
            displayed_y_center(
                "reuse_aware_residency.pdf",
                "Full-state",
                0.269,
                constants[1]["WIDTH_PT"],
            ),
            displayed_y_center(
                "laion100m_reduction_residency.pdf",
                "Out",
                0.342,
                constants[2]["WIDTH_PT"],
            ),
        )

        self.assertLess(abs(title_centers[1] - title_centers[2]), 0.0008)
        hashprune_raise = min(title_centers[1:]) - title_centers[0]
        self.assertGreater(hashprune_raise, 0.0015)
        self.assertLess(hashprune_raise, 0.0030)

    def test_recall_title_has_visible_clearance_below_tick_labels(self):
        tick_bottom = max(
            word_box("laion100m_hashprune_policy_recall_qps.pdf", label)[3]
            for label in ("75", "85", "95", "90", "92", "94", "96")
        )
        title_top = word_box("laion100m_hashprune_policy_recall_qps.pdf", "Recall@10")[1]

        self.assertGreaterEqual(title_top - tick_bottom, 1.25)

    def test_residency_comparison_uses_reservoir_label(self):
        script = (FIGURE_DIR / "plot_reuse_aware_residency.py").read_text(encoding="utf-8")
        self.assertIn('"Reservoir"', script)
        self.assertNotIn('"Resizable"', script)
        word_box("reuse_aware_residency.pdf", "Reservoir")

    def test_latex_uses_the_natural_width_matched_minipages(self):
        evaluation = (FIGURE_DIR.parent / "08_evaluation.md").read_text(encoding="utf-8")

        self.assertEqual(evaluation.count(r"\begin{minipage}[t]{0.349\textwidth}"), 1)
        self.assertEqual(evaluation.count(r"\begin{minipage}[t]{0.269\textwidth}"), 1)
        self.assertEqual(evaluation.count(r"\begin{minipage}[t]{0.342\textwidth}"), 1)
        self.assertEqual(evaluation.count(r"\includegraphics[width=\linewidth]{figures/reuse_aware_residency.pdf}"), 1)


if __name__ == "__main__":
    unittest.main()
