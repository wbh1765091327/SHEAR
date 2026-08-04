#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO / "script" / "summarize_tessa_progressive_ablation.py"
RUNNER = REPO / "script" / "run_tessa_progressive_ablation.sh"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("tessa_summary", SUMMARY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ContractTests(unittest.TestCase):
    def test_print_contract_has_exact_matrix_and_common_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "contract_test"
            result = subprocess.run(
                ["bash", str(RUNNER), "print-contract", str(run_root)],
                cwd=REPO,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertFalse(run_root.exists())
            text = result.stdout

        expected_by_mode = {
            "A0": [
                "--gpu-subtree-sq-staging-raw-d1-diagnostic",
                "--gpu-subtree-sq-wmma-mode off",
                "--gpu-subtree-assignment-backend persistent-tiled-gemm-fp32",
                "--gpu-subtree-leaf-topk-backend persistent-tiled-gemm-fp32",
                "--gpu-subtree-persistent-large-ticket-batch 1",
            ],
            "A1": [
                "--gpu-subtree-sq-staging-raw-d1-diagnostic",
                "--gpu-subtree-sq-wmma-mode off",
                "--gpu-subtree-assignment-backend persistent-wmma-tf32",
                "--gpu-subtree-leaf-topk-backend persistent-wmma-tf32",
                "--gpu-subtree-persistent-large-ticket-batch 1",
            ],
            "A2": [
                "--gpu-subtree-sq-wmma-mode build",
                "--gpu-subtree-sq8-assignment-kernel warp-dp4a",
                "--gpu-subtree-sq8-leaf-executor warp-dp4a",
                "--gpu-subtree-sq8-leaf-mode staged",
                "--gpu-subtree-persistent-large-ticket-batch 1",
            ],
            "A3": [
                "--gpu-subtree-sq-wmma-mode build",
                "--gpu-subtree-sq8-assignment-kernel wmma",
                "--gpu-subtree-sq8-leaf-executor wmma",
                "--gpu-subtree-sq8-leaf-mode staged",
                "--gpu-subtree-persistent-large-ticket-batch 8",
            ],
        }
        sections = {}
        for mode in expected_by_mode:
            marker = f"### {mode} screen command"
            self.assertIn(marker, text)
            sections[mode] = text.split(marker, 1)[1].split("###", 1)[0]
        for mode, flags in expected_by_mode.items():
            for flag in flags:
                self.assertIn(flag, sections[mode])

        common = [
            "--gpu-subtree-sq8-d0-assignment-kernel wmma-batch",
            "--fanout-top 6",
            "--fanout-second 3",
            "--seed 42",
            "--pipnn-gpu-workspace-budget-gb 16",
            "--oom-io-mode buffered",
            "--m-hash-bits 12",
            "--hashprune-backfill-mode rhalf-soft-tier-q3",
            "--hashprune-backfill-ablation none",
            "--hashprune-backfill-budget 0",
            "--hashprune-backfill-alpha 1.0",
        ]
        environment = [
            "SHEAR_PIPNN_PROFILE=1",
            "SHEAR_PIPNN_PHASE_MONITOR=1",
            "SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1",
            "SHEAR_PIPNN_WMMA_CTA_SHARED_B=0",
            "SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0",
            "SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0",
            "SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0",
            "SHEAR_PIPNN_PRUNE_BACKEND=hashprune",
            "SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0",
            "OPENBLAS_NUM_THREADS=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
        ]
        for section in sections.values():
            for token in common + environment:
                self.assertIn(token, section)

    def test_print_2x2_contract_has_only_missing_cells(self):
        parent = "/data_staging/wbh/tessa_ablation/wiki10m/20260721_011618"
        result = subprocess.run(
            ["bash", str(RUNNER), "print-2x2-contract", parent],
            cwd=REPO,
            check=True,
            text=True,
            capture_output=True,
        )
        text = result.stdout
        self.assertIn("### W1 screen command", text)
        self.assertIn("### D8 screen command", text)
        self.assertIn("full_build_2x2/full_builds/W1/run1", text)
        self.assertIn("full_build_2x2/full_builds/D8/run1", text)
        w1 = text.split("### W1 screen command", 1)[1].split("###", 1)[0]
        d8 = text.split("### D8 screen command", 1)[1].split("###", 1)[0]
        for flag in (
            "--gpu-subtree-sq8-assignment-kernel wmma",
            "--gpu-subtree-sq8-leaf-executor wmma",
            "--gpu-subtree-persistent-large-ticket-batch 1",
        ):
            self.assertIn(flag, w1)
        for flag in (
            "--gpu-subtree-sq8-assignment-kernel warp-dp4a",
            "--gpu-subtree-sq8-leaf-executor warp-dp4a",
            "--gpu-subtree-persistent-large-ticket-batch 8",
        ):
            self.assertIn(flag, d8)

    def test_print_2x2_contract_rejects_other_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["bash", str(RUNNER), "print-2x2-contract", str(Path(temp_dir) / "run")],
                cwd=REPO,
                check=False,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires the accepted parent run", result.stderr)

    def test_original_contract_does_not_gain_extension_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["bash", str(RUNNER), "print-contract", str(Path(temp_dir) / "run")],
                cwd=REPO,
                check=True,
                text=True,
                capture_output=True,
            )
        self.assertNotIn("### W1 screen command", result.stdout)
        self.assertNotIn("### D8 screen command", result.stdout)

    def test_print_fresh_2x2_contract_has_all_four_sq8_cells(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "fresh_2x2"
            result = subprocess.run(
                ["bash", str(RUNNER), "print-fresh-2x2-contract", str(run_root)],
                cwd=REPO,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertFalse(run_root.exists())
            text = result.stdout

        expected = {
            "A2": ("warp-dp4a", "warp-dp4a", "1"),
            "W1": ("wmma", "wmma", "1"),
            "D8": ("warp-dp4a", "warp-dp4a", "8"),
            "A3": ("wmma", "wmma", "8"),
        }
        for mode, (assignment, leaf, batch) in expected.items():
            marker = f"### {mode} fresh 2x2 screen command"
            self.assertIn(marker, text)
            section = text.split(marker, 1)[1].split("###", 1)[0]
            self.assertIn(f"full_builds/{mode}/run1", section)
            self.assertIn(f"--gpu-subtree-sq8-assignment-kernel {assignment}", section)
            self.assertIn(f"--gpu-subtree-sq8-leaf-executor {leaf}", section)
            self.assertIn(f"--gpu-subtree-persistent-large-ticket-batch {batch}", section)
        self.assertNotIn("full_build_2x2", text)
        self.assertIn("preflight-fresh-2x2", text)
        self.assertIn("cleanup-fresh-2x2", text)

    def test_runner_declares_extension_lifecycle_and_cleanup_boundary(self):
        text = RUNNER.read_text(encoding="utf-8")
        for function_name in (
            "validate_parent_2x2_inputs()",
            "validate_2x2_build_matrix()",
            "run_2x2_preflight()",
            "run_2x2_screen()",
            "run_2x2_repeat()",
            "run_2x2_search()",
            "run_2x2_summarize()",
            "run_2x2_cleanup()",
        ):
            self.assertIn(function_name, text)
        self.assertIn('sha256sum --check --status "${run_root}/binary_sha256.txt"', text)
        self.assertIn('"${extension_root}/script_sha256.txt"', text)
        self.assertIn('case "${target}" in "${extension_root}"/*)', text)

    def test_runner_declares_fresh_2x2_lifecycle(self):
        text = RUNNER.read_text(encoding="utf-8")
        for function_name in (
            "validate_fresh_2x2_build_matrix()",
            "run_fresh_2x2_preflight()",
            "run_fresh_2x2_screen()",
            "run_fresh_2x2_repeat()",
            "run_fresh_2x2_search()",
            "run_fresh_2x2_summarize()",
            "run_fresh_2x2_cleanup()",
        ):
            self.assertIn(function_name, text)
        self.assertIn('python3 "${summary_script}" summarize-fresh-2x2 "${run_root}"', text)

    def test_print_fresh_dp4a_contract_has_only_d1_and_d8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "fresh_dp4a"
            result = subprocess.run(
                ["bash", str(RUNNER), "print-fresh-dp4a-contract", str(run_root)],
                cwd=REPO,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertFalse(run_root.exists())
            text = result.stdout

        self.assertIn("### D1 fresh DP4A screen command", text)
        self.assertIn("### D8 fresh DP4A screen command", text)
        self.assertNotIn("### W1", text)
        self.assertNotIn("### A3", text)
        d1 = text.split("### D1 fresh DP4A screen command", 1)[1].split("###", 1)[0]
        d8 = text.split("### D8 fresh DP4A screen command", 1)[1].split("###", 1)[0]
        for section, batch in ((d1, "1"), (d8, "8")):
            self.assertIn("--gpu-subtree-sq8-assignment-kernel warp-dp4a", section)
            self.assertIn("--gpu-subtree-sq8-leaf-executor warp-dp4a", section)
            self.assertIn(f"--gpu-subtree-persistent-large-ticket-batch {batch}", section)
        self.assertIn("preflight-fresh-dp4a", text)
        self.assertIn("cleanup-fresh-dp4a", text)

    def test_runner_declares_fresh_dp4a_lifecycle(self):
        text = RUNNER.read_text(encoding="utf-8")
        for function_name in (
            "validate_fresh_dp4a_build_matrix()",
            "run_fresh_dp4a_preflight()",
            "run_fresh_dp4a_screen()",
            "run_fresh_dp4a_repeat()",
            "run_fresh_dp4a_search()",
            "run_fresh_dp4a_summarize()",
            "run_fresh_dp4a_cleanup()",
        ):
            self.assertIn(function_name, text)
        self.assertIn('python3 "${summary_script}" summarize-fresh-dp4a "${run_root}"', text)


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary_module()

    @staticmethod
    def build_records():
        rows = []
        for mode in ("A0", "A1", "A2", "A3"):
            for repeat in (1, 2, 3):
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "wall_s": 100.0,
                        "persistent_event_ms": 50.0,
                        "root_leader_hash": "root",
                        "d0_partition_hash": "partition",
                        "d0_task_list_hash": "tasks",
                        "fallback_count": 0,
                        "overflow_count": 0,
                        "watchdog_count": 0,
                    }
                )
        return rows

    @staticmethod
    def extension_build_records():
        rows = []
        for mode in ("A2", "A3", "W1", "D8"):
            for repeat in (1, 2, 3):
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "wall_s": 100.0,
                        "persistent_event_ms": 50.0,
                        "root_leader_hash": "root",
                        "d0_partition_hash": "partition",
                        "d0_task_list_hash": "tasks",
                        "fallback_count": 0,
                        "overflow_count": 0,
                        "watchdog_count": 0,
                        "assignment_dp4a_tiles": 1 if mode in ("A2", "D8") else 0,
                        "assignment_wmma_tiles": 1 if mode in ("A3", "W1") else 0,
                        "leaf_dp4a_tiles": 1 if mode in ("A2", "D8") else 0,
                        "leaf_wmma_tiles": 1 if mode in ("A3", "W1") else 0,
                    }
                )
        return rows

    def test_extension_mode_contracts(self):
        self.assertEqual(
            self.summary._expected_mode_contract("W1"),
            {
                "assignment_backend": "persistent-wmma-tf32",
                "leaf_topk_backend": "persistent-wmma-tf32",
                "batch": 1,
                "sq_assignment": "wmma",
                "sq_leaf": "wmma",
            },
        )
        self.assertEqual(
            self.summary._expected_mode_contract("D8"),
            {
                "assignment_backend": "persistent-wmma-tf32",
                "leaf_topk_backend": "persistent-wmma-tf32",
                "batch": 8,
                "sq_assignment": "warp-dp4a",
                "sq_leaf": "warp-dp4a",
            },
        )

    def test_extension_build_validation_requires_complete_2x2(self):
        rows = self.extension_build_records()
        self.summary.validate_extension_build_records(rows)
        with self.assertRaisesRegex(ValueError, "W1 must have exactly three repetitions"):
            self.summary.validate_extension_build_records(rows[:-4] + rows[-3:])
        rows = self.extension_build_records()
        rows[-1]["d0_partition_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "D0 hash mismatch"):
            self.summary.validate_extension_build_records(rows)

    def test_extension_build_validation_rejects_wrong_executor_counters(self):
        rows = self.extension_build_records()
        w1 = next(row for row in rows if row["mode"] == "W1")
        w1["assignment_dp4a_tiles"] = 1
        with self.assertRaisesRegex(ValueError, "W1 executor tile counters"):
            self.summary.validate_extension_build_records(rows)

    def test_extension_speedup_ratios(self):
        ratios = self.summary.extension_speedup_ratios(
            {"D1": 100.0, "W1": 125.0, "D8": 80.0, "W8": 40.0}
        )
        self.assertAlmostEqual(ratios["wmma_at_batch1"], 0.8)
        self.assertAlmostEqual(ratios["wmma_at_batch8"], 2.0)
        self.assertAlmostEqual(ratios["batch8_under_dp4a"], 1.25)
        self.assertAlmostEqual(ratios["batch8_under_wmma"], 3.125)
        self.assertAlmostEqual(ratios["executor_batch_interaction"], 2.5)

    def test_extension_speedup_ratios_reject_nonpositive_values(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.summary.extension_speedup_ratios(
                {"D1": 100.0, "W1": 0.0, "D8": 80.0, "W8": 40.0}
            )

    @staticmethod
    def extension_search_records():
        recall_by_l = {25: 0.90, 50: 0.93, 100: 0.95, 200: 0.97, 400: 0.98}
        qps_by_mode = {"A0": 1000.0, "A2": 1010.0, "A3": 1020.0, "W1": 1030.0, "D8": 1040.0}
        rows = []
        for mode in ("A0", "A2", "A3", "W1", "D8"):
            for search_l, recall in recall_by_l.items():
                for repeat in (1, 2, 3):
                    rows.append(
                        {
                            "mode": mode,
                            "l": search_l,
                            "repeat": repeat,
                            "queries": 10_000,
                            "wall_s": 1.0,
                            "qps": qps_by_mode[mode],
                            "recall": recall,
                            "result_bytes": 1,
                        }
                    )
        return rows

    def test_load_2x2_build_records_uses_exact_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            for row in self.extension_build_records():
                if row["mode"] in ("A2", "A3"):
                    base = run_root / "full_builds" / row["mode"]
                else:
                    base = run_root / "full_build_2x2" / "full_builds" / row["mode"]
                path = base / f"run{row['repeat']}" / "build_record.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            extra = run_root / "full_build_2x2" / "full_builds" / "W1" / "run1" / "validation.json"
            extra.write_text('{"ignored": true}\n', encoding="utf-8")

            records = self.summary.load_2x2_build_records(run_root)

        self.assertEqual(len(records), 12)
        self.assertEqual({row["mode"] for row in records}, {"A2", "A3", "W1", "D8"})

    def test_load_fresh_2x2_records_uses_one_run_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            for row in self.extension_build_records():
                path = (
                    run_root
                    / "full_builds"
                    / row["mode"]
                    / f"run{row['repeat']}"
                    / "build_record.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            for row in self.fresh_2x2_search_records():
                path = (
                    run_root
                    / "search"
                    / row["mode"]
                    / f"L{row['l']}"
                    / f"rep{row['repeat']}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            builds = self.summary.load_fresh_2x2_build_records(run_root)
            searches = self.summary.load_fresh_2x2_search_records(run_root)

        self.assertEqual(len(builds), 12)
        self.assertEqual(len(searches), 60)
        self.assertEqual({row["mode"] for row in builds}, {"A2", "A3", "W1", "D8"})
        self.assertEqual({row["mode"] for row in searches}, {"A2", "A3", "W1", "D8"})

    def test_validate_2x2_search_records_requires_complete_grid(self):
        rows = self.extension_search_records()
        self.summary.validate_2x2_search_records(rows)
        with self.assertRaisesRegex(ValueError, "search matrix is incomplete"):
            self.summary.validate_2x2_search_records(rows[:-1])
        rows = self.extension_search_records()
        rows[0]["queries"] = 9_999
        with self.assertRaisesRegex(ValueError, "must contain 10000 queries"):
            self.summary.validate_2x2_search_records(rows)

    def test_summarize_2x2_writes_separate_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            parent_summary = run_root / "summary.md"
            parent_summary.write_text("parent-summary\n", encoding="utf-8")
            wall_by_mode = {"A2": 100.0, "W1": 125.0, "D8": 80.0, "A3": 40.0}
            for row in self.extension_build_records():
                row["wall_s"] = wall_by_mode[row["mode"]]
                row["persistent_event_ms"] = wall_by_mode[row["mode"]] / 2.0
                if row["mode"] in ("A2", "A3"):
                    base = run_root / "full_builds" / row["mode"]
                else:
                    base = run_root / "full_build_2x2" / "full_builds" / row["mode"]
                path = base / f"run{row['repeat']}" / "build_record.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            for row in self.extension_search_records():
                if row["mode"] in ("A0", "A2", "A3"):
                    base = run_root / "search" / row["mode"]
                else:
                    base = run_root / "full_build_2x2" / "search" / row["mode"]
                path = base / f"L{row['l']}" / f"rep{row['repeat']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            self.summary.summarize_2x2(run_root)

            extension_summary = json.loads(
                (run_root / "full_build_2x2" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parent_summary.read_text(encoding="utf-8"), "parent-summary\n")
            self.assertTrue((run_root / "full_build_2x2" / "results.tsv").is_file())
            self.assertTrue((run_root / "full_build_2x2" / "summary.md").is_file())
        self.assertAlmostEqual(extension_summary["wall_speedup_ratios"]["wmma_at_batch1"], 0.8)
        self.assertAlmostEqual(extension_summary["wall_speedup_ratios"]["wmma_at_batch8"], 2.0)
        self.assertAlmostEqual(extension_summary["wall_speedup_ratios"]["executor_batch_interaction"], 2.5)

    @staticmethod
    def fresh_2x2_search_records():
        return [
            row
            for row in AggregationTests.extension_search_records()
            if row["mode"] in ("A2", "A3", "W1", "D8")
        ]

    def test_summarize_fresh_2x2_writes_non_post_hoc_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            wall_by_mode = {"A2": 100.0, "W1": 80.0, "D8": 90.0, "A3": 50.0}
            for row in self.extension_build_records():
                row["wall_s"] = wall_by_mode[row["mode"]]
                row["persistent_event_ms"] = wall_by_mode[row["mode"]] / 2.0
                path = (
                    run_root
                    / "full_builds"
                    / row["mode"]
                    / f"run{row['repeat']}"
                    / "build_record.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            for row in self.fresh_2x2_search_records():
                path = (
                    run_root
                    / "search"
                    / row["mode"]
                    / f"L{row['l']}"
                    / f"rep{row['repeat']}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            self.summary.summarize_fresh_2x2(run_root)
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertFalse(summary["post_hoc"])
            self.assertEqual(summary["schema"], "tessa-full-build-2x2-fresh-v1")
            self.assertTrue((run_root / "results.tsv").is_file())
            self.assertTrue((run_root / "summary.md").is_file())
            self.assertIn("same release binary", (run_root / "summary.md").read_text(encoding="utf-8"))

        self.assertAlmostEqual(summary["wall_speedup_ratios"]["wmma_at_batch1"], 1.25)
        self.assertAlmostEqual(summary["wall_speedup_ratios"]["wmma_at_batch8"], 1.8)

    def test_summarize_fresh_dp4a_writes_batch_comparison(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            wall_by_mode = {"A2": 120.0, "D8": 80.0}
            for row in self.extension_build_records():
                if row["mode"] not in wall_by_mode:
                    continue
                row["wall_s"] = wall_by_mode[row["mode"]]
                row["persistent_event_ms"] = wall_by_mode[row["mode"]] / 2.0
                path = (
                    run_root
                    / "full_builds"
                    / row["mode"]
                    / f"run{row['repeat']}"
                    / "build_record.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            for row in self.fresh_2x2_search_records():
                if row["mode"] not in wall_by_mode:
                    continue
                path = (
                    run_root
                    / "search"
                    / row["mode"]
                    / f"L{row['l']}"
                    / f"rep{row['repeat']}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            self.summary.summarize_fresh_dp4a(run_root)
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["schema"], "tessa-full-build-dp4a-fresh-v1")
            self.assertEqual(set(summary["cells"]), {"D1", "D8"})
            self.assertTrue((run_root / "results.tsv").is_file())
            self.assertTrue((run_root / "summary.md").is_file())

        self.assertAlmostEqual(summary["batch8_speedup"]["wall"], 1.5)
        self.assertAlmostEqual(summary["batch8_speedup"]["persistent_event"], 1.5)

    def test_median_and_range(self):
        self.assertEqual(
            self.summary.median_range([3.0, 1.0, 2.0]),
            {"median": 2.0, "min": 1.0, "max": 3.0},
        )

    def test_build_validation_requires_three_runs(self):
        rows = self.build_records()
        self.summary.validate_build_records(rows)
        with self.assertRaisesRegex(ValueError, "three repetitions"):
            self.summary.validate_build_records(rows[:-1])

    def test_build_validation_rejects_hash_mismatch_and_fallback(self):
        rows = self.build_records()
        rows[5]["d0_task_list_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "D0 hash mismatch"):
            self.summary.validate_build_records(rows)

        rows = self.build_records()
        rows[0]["fallback_count"] = 1
        with self.assertRaisesRegex(ValueError, "safety counter"):
            self.summary.validate_build_records(rows)

    def test_load_records_ignores_search_validation_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            search_dir = run_root / "search" / "A0" / "L25"
            search_dir.mkdir(parents=True)
            (search_dir / "rep1.json").write_text('{"mode": "A0"}\n')
            (search_dir / "rep1.validation.json").write_text('{"validated": true}\n')

            builds, searches = self.summary.load_records(run_root)

        self.assertEqual(builds, [])
        self.assertEqual(searches, [{"mode": "A0"}])

    def test_stable_gain_threshold_and_range(self):
        stable = self.summary.paired_gain_decision(
            [100.0, 100.0, 100.0], [95.0, 96.0, 94.0]
        )
        self.assertTrue(stable["stable"])
        weak = self.summary.paired_gain_decision(
            [100.0, 100.0, 100.0], [98.0, 98.0, 98.0]
        )
        self.assertFalse(weak["stable"])
        crosses_zero = self.summary.paired_gain_decision(
            [100.0, 100.0, 100.0], [94.0, 101.0, 94.0]
        )
        self.assertFalse(crosses_zero["stable"])

    def test_quality_loss_above_point_zero_zero_five_fails(self):
        rows = []
        for mode, recall in (("A0", 0.960), ("A1", 0.958), ("A2", 0.955), ("A3", 0.954)):
            for repeat in (1, 2, 3):
                rows.append({"mode": mode, "l": 100, "repeat": repeat, "recall": recall, "qps": 1.0})
        decision = self.summary.quality_decision(rows)
        self.assertFalse(decision["preserved"])
        self.assertAlmostEqual(decision["max_loss"], 0.006)

    def test_matched_recall_interpolates_only_when_bracketed(self):
        points = [(0.90, 1000.0), (0.95, 800.0), (0.98, 500.0)]
        self.assertEqual(self.summary.matched_recall_qps(points, 0.95), 800.0)
        self.assertAlmostEqual(
            self.summary.matched_recall_qps([(0.94, 900.0), (0.96, 700.0)], 0.95),
            800.0,
        )
        self.assertIsNone(
            self.summary.matched_recall_qps([(0.80, 1000.0), (0.90, 800.0)], 0.95)
        )


if __name__ == "__main__":
    unittest.main()
