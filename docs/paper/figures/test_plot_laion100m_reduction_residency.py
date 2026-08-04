#!/usr/bin/env python3
"""Regression checks for the LAION100M reduction-residency figure."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("plot_laion100m_reduction_residency.py")
PDF_PATH = SCRIPT_PATH.with_name("laion100m_reduction_residency.pdf")


def load_plot_module():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location("plot_laion100m_reduction_residency", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReductionResidencyFigureTest(unittest.TestCase):
    def test_uses_shared_arial_font_family(self):
        module = load_plot_module()

        self.assertEqual(module.FONT_FAMILY, "Arial")

    def test_rendered_pdf_embeds_shared_arial(self):
        result = subprocess.run(
            ["pdffonts", str(PDF_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Arial", result.stdout)
        self.assertNotIn("TimesNewRoman", result.stdout)

    def test_uses_five_external_reduction_levels_with_pre_breakdown(self):
        module = load_plot_module()

        self.assertEqual(
            module.GROUPS,
            (
                ("0%", 666.2, 179.4, 455.0, 31.8),
                ("25%", 691.01, 191.629, 454.951, 44.43),
                ("50%", 709.15, 194.048, 454.753, 60.349),
                ("75%", 727.35, 198.025, 455.008, 74.317),
                ("100%", 747.29, 201.937, 454.947, 90.406),
            ),
        )

    def test_uses_one_stacked_bar_per_group(self):
        module = load_plot_module()

        self.assertEqual(module.SERIES, ("Pre", "GPU", "Post-GPU"))
        for _, e2e, pre, gpu, post_gpu in module.GROUPS:
            self.assertAlmostEqual(pre + gpu + post_gpu, e2e, places=3)

    def test_uses_shared_solid_bar_palette(self):
        module = load_plot_module()

        self.assertEqual(
            module.BAR_COLORS,
            {
                "Pre": (0.720, 0.730, 0.740),
                "GPU": (0.835, 0.369, 0.000),
                "Post-GPU": (0.000, 0.620, 0.451),
            },
        )


if __name__ == "__main__":
    unittest.main()
