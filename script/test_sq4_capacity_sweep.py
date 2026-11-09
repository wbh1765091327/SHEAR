"""Protocol, telemetry, failure and destructive-cleanup regression tests."""
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import types
from unittest.mock import patch

import run_sq4_capacity_sweep as sweep
import run_sq4_plus as recipe
import sq4_sweep_metrics as metrics
import sq4_sweep_runtime as runtime


def row(config, seconds, key=None, margin=True, status="success"):
    return dict(run=config, config=config, budget=int(config[1:3]) / 100, high=int(config[5:7]) / 100,
                status=status, metrics=dict(wall_s=seconds, task_equivalence_key=key, memory_margin_ok=margin),
                quality=[dict(L=L, recall=r, qps=q) for L, r, q in
                         [(64, .93, 2500), (128, .955, 1800), (256, .967, 1200), (512, .973, 750)]])


class Protocol(unittest.TestCase):
    def test_matrix_is_deterministic_and_contains_only_thirteen_valid_cells(self):
        cells = sweep.matrix()
        self.assertEqual(cells, sweep.matrix())
        self.assertEqual(len(cells), 13)
        self.assertEqual(len({c["config"] for c in cells}), 13)
        self.assertEqual(cells[0]["config"], metrics.BASELINE)
        self.assertTrue(all(0 < c["budget"] <= c["high"] <= .5 for c in cells))

    def test_interpolation_is_bounded_and_handles_exact_and_flat_recalls(self):
        points = row("b08-h50", 100)["quality"]
        self.assertIsNone(metrics.interpolate(points, .98))
        self.assertIsNone(metrics.interpolate(points, .90))
        self.assertEqual(metrics.interpolate(points, .955), 1800)
        self.assertAlmostEqual(metrics.interpolate(points, .95), 1940)
        points[1]["recall"] = points[0]["recall"]
        self.assertTrue(metrics.interpolate(points, .95) > 0)

    def test_finalists_exclude_failures_low_margin_and_prefer_distinct_tasks(self):
        rows = [row("b08-h50", 100, "base"), row("b08-h32", 80, "base"),
                row("b16-h32", 90, "x"), row("b16-h50", 85, "x"),
                row("b32-h50", 95, "y"), row("b04-h08", 1, "z", status="failed"),
                row("b04-h16", 2, "w", margin=False)]
        self.assertEqual(metrics.select_finalists(rows), ["b16-h32", "b32-h50"])

    def test_split_tasks_with_equal_sizes_are_not_grouped_without_id_hashes(self):
        rows = [row("b08-h50", 100), row("b16-h32", 90), row("b16-h50", 85)]
        self.assertEqual(metrics.select_finalists(rows), ["b16-h50", "b16-h32"])

    def test_noise_rule_uses_both_ranges_and_two_percent_floor(self):
        self.assertTrue(metrics.near_noise([100, 100, 100], [99, 99, 99]))
        self.assertTrue(metrics.near_noise([95, 100, 105], [94, 95, 96]))
        self.assertFalse(metrics.near_noise([99, 100, 101], [90, 91, 92]))

    def test_summary_keeps_failures_and_reports_recall_in_percentage_points(self):
        a, b = row("b08-h50", 100), row("b16-h32", 90)
        b["quality"][0]["recall"] -= .001
        failure = row("b16-h32", 0, status="failed")
        s = metrics.summarize([a, b, failure])
        self.assertEqual(s["b16-h32"]["successful"], 1)
        self.assertEqual(len(s["b16-h32"]["failures"]), 1)
        self.assertAlmostEqual(s["b16-h32"]["quality"]["64"]["recall_delta_pp"], -.1)
        self.assertAlmostEqual(s["b16-h32"]["wall_delta_percent"], -10)

    def test_gpu_oom_and_admission_denial_are_distinct(self):
        self.assertEqual(runtime.failure_kind(RuntimeError("failed"), "CUDA_ERROR_OUT_OF_MEMORY"), "gpu_oom")
        self.assertEqual(runtime.failure_kind(RuntimeError("admission exceeds budget"), ""), "admission_rejected")
        self.assertEqual(runtime.failure_kind(RuntimeError("No space left on device"), ""), "storage_exhausted")

    def test_controller_runs_13_screenings_six_confirmations_and_separate_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = sweep.Experiment.__new__(sweep.Experiment)
            experiment.root = Path(tmp)
            experiment.results = []
            experiment.args = types.SimpleNamespace(sq8_reference_run=Path("reference"))
            experiment.status = lambda *args, **kwargs: None
            calls = []
            def fake_run(run, cell, **kwargs):
                calls.append((run, cell["config"]))
                config = cell["config"]
                rank = [c["config"] for c in sweep.matrix()].index(config) if config != "sq8-reference" else 0
                seconds = 100 if config == metrics.BASELINE else 50 + rank * 3
                result = dict(run=run, config=config, budget=cell["budget"], high=cell["high"],
                              status="success", quality=row("b08-h50", 100)["quality"],
                              metrics=dict(wall_s=seconds, task_equivalence_key=config, memory_margin_ok=True))
                experiment.results.append(result)
                return result
            experiment.run = fake_run
            with patch.object(sweep, "write_report"):
                experiment.sweep()
            self.assertEqual(len(calls), 20)
            self.assertEqual(sum(run.startswith("screen-") for run, _ in calls), 13)
            self.assertEqual([run.split("-")[-1] for run, _ in calls if run.startswith("confirm-")],
                             ["A", "B", "C", "C", "B", "A"])
            self.assertEqual(calls[-1], ("sq8-reference", "sq8-reference"))


class Cleanup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.owned = self.root / "owned"
        self.owned.mkdir()
        self.scratch = self.owned / "run1"
        (self.scratch / "index").mkdir(parents=True)
        (self.scratch / "index/_mem.index").write_bytes(b"graph")
        metrics.save(self.scratch / ".sweep-owner.json", dict(token="ours", run="run1"))
        self.record = self.root / "records"
        (self.record / "quality").mkdir(parents=True)
        self.quality = [dict(L=L, recall=.95, qps=100) for L in metrics.LIST_SIZES]
        metrics.save(self.record / "quality/result.json", self.quality)
        (self.record / "quality/64.fbin").write_bytes(b"neighbors")
        (self.record / "quality/64.tsv").write_text("raw metrics")
        metrics.save(self.record / "config-audit.json", dict(accepted=True))
        self.result = dict(status="success", run="run1", quality=self.quality, graph_audit="passed",
                           graph_sha256=recipe.digest(self.scratch / "index/_mem.index"))
        metrics.save(self.record / "result.json", self.result)

    def clean(self):
        runtime.cleanup_owned(self.scratch, self.owned, "ours", self.record)

    def test_success_removes_only_owned_graph_and_neighbor_arrays(self):
        historical = self.root / "historical.index"
        historical.write_bytes(b"preserve")
        self.clean()
        self.assertFalse(self.scratch.exists())
        self.assertFalse((self.record / "quality/64.fbin").exists())
        self.assertTrue((self.record / "quality/64.tsv").exists())
        self.assertEqual(historical.read_bytes(), b"preserve")
        self.assertEqual(json.loads((self.record / "cleanup.json").read_text())["state"], "complete")

    def test_missing_quality_never_deletes_graph(self):
        metrics.save(self.record / "quality/result.json", self.quality[:-1])
        with self.assertRaises(ValueError):
            self.clean()
        self.assertTrue(self.scratch.exists())

    def test_wrong_owner_never_deletes_graph(self):
        with self.assertRaises(ValueError):
            runtime.cleanup_owned(self.scratch, self.owned, "someone-else", self.record)
        self.assertTrue(self.scratch.exists())

    def test_changed_graph_is_retained(self):
        (self.scratch / "index/_mem.index").write_bytes(b"modified")
        with self.assertRaises(ValueError):
            self.clean()
        self.assertTrue(self.scratch.exists())

    def test_symlink_to_external_graph_is_rejected(self):
        outside = self.root / "external"
        outside.write_bytes(b"graph")
        graph = self.scratch / "index/_mem.index"
        graph.unlink()
        graph.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.clean()
        self.assertTrue(outside.exists())


class Entry(unittest.TestCase):
    def test_default_prepares_all_commands_without_starting_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = {}
            for key, rows, width, stride in [("base", 100_000_000, 512, 4),
                                             ("query", 10000, 512, 4), ("gt100m", 10000, 100, 8)]:
                path = root / key
                with path.open("wb") as f:
                    f.write(struct.pack("<II", rows, width))
                    f.truncate(8 + rows * width * stride)
                entries[key] = dict(path=str(path), sha256="0" * 64)
            passport = root / "inputs.json"
            metrics.save(passport, entries)
            with patch.object(sweep, "Experiment", side_effect=AssertionError("must not execute")), \
                    patch.object(sweep.subprocess, "Popen", side_effect=AssertionError("must not spawn")):
                sweep.main(["--run-dir", str(root / "plan"), "--input-manifest", str(passport),
                            "--binary", "/usr/bin/true"])
            plan = json.loads((root / "plan/plan.json").read_text())
            self.assertFalse(plan["execution_started"])
            self.assertEqual(len(plan["screening"]), 13)
            for record in plan["screening"]:
                c = record["command"]
                for flag, value in {"--fanout-top": "6", "--root-risk-candidate-rs": "1,2,4,6",
                                    "--gpu-subtree-d0-tagore-sample-fraction": "0.002"}.items():
                    self.assertEqual(c[c.index(flag) + 1], value)
                self.assertNotIn("--gpu-subtree-d0-capacity-root-leader-cap", c)

    def test_invalid_10m_gate_blocks_100m(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.json"
            metrics.save(path, dict(accepted=True, rows=100_000_000))
            with self.assertRaises(ValueError):
                sweep.verify_gate(path, "a", "b")

    def test_prepared_plan_allows_shell_setup_but_not_parameter_drift(self):
        first = {"fixed": {"fanout": [6, 3, 1]}, "screening": [
            {"command": ["build", "--fanout-top", "6"], "environment": {"PATH": "/old"},
             "replaced_environment": {}}]}
        second = copy.deepcopy(first)
        second["screening"][0]["environment"]["PATH"] = "/new/toolchain"
        self.assertEqual(sweep.plan_contract(first), sweep.plan_contract(second))
        second["screening"][0]["command"][-1] = "8"
        self.assertNotEqual(sweep.plan_contract(first), sweep.plan_contract(second))

    def test_10m_gate_has_six_explicit_jobs_and_forces_real_split(self):
        jobs = sweep.validation_jobs()
        self.assertEqual(len(jobs), 6)
        self.assertEqual([old for _, _, old in jobs], [True, False, False, True, True, False])
        self.assertTrue(all(c["routing_leaders"] == 6 and c["high"] == .5 for _, c, _ in jobs[-2:]))


class SavedTelemetry(unittest.TestCase):
    def test_reference_replay_redirects_all_historical_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            original.mkdir()
            binary = original / "build"
            binary.write_bytes(b"frozen reference")
            command = [str(binary)]
            for flag in ("--data-path", "--index-path-prefix", "--oom-temp-dir", "--oom-profile-json",
                         "--gpu-subtree-sq-wmma-sidecar-prefix"):
                command.extend([flag, str(original / flag[2:])])
            metrics.save(original / "cmd.json", command)
            metrics.save(original / "binary.json", dict(path=str(binary), sha256=recipe.digest(binary)))
            metrics.save(original / "env.json", {"SHEAR_PIPNN_PERSISTENT_LEAF_HISTOGRAM_OUT": str(original / "leaves.jsonl")})
            experiment = sweep.Experiment.__new__(sweep.Experiment)
            experiment.root = root / "new"
            (experiment.root / "bin").mkdir(parents=True)
            experiment.inputs = {"base": {"path": "/verified/base.fbin"}}
            record, env = experiment.reference_manifest(original, "reference", root / "scratch")
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["RAYON_NUM_THREADS"], "8")
            self.assertEqual(env["SHEAR_PIPNN_PERSISTENT_LEAF_HISTOGRAM_OUT"],
                             str(experiment.root / "reference/leaves.jsonl"))
            self.assertTrue(all(str(original) not in arg for arg in record["command"]))
            self.assertFalse((original / "leaves.jsonl").exists())

    def test_real_split_and_nonsplit_profiles_have_complete_subtree_accounting(self):
        for label, count in [("standard-A1", 13), ("standard-B1", 0)]:
            d = sweep.FROZEN / label
            if not d.exists():
                self.skipTest("historical measurements are not shipped in the source checkout")
            profile = json.loads((d / "profile.json").read_text())
            resources = [dict(gpu=dict(process_used_bytes=15 << 30, used_bytes=16 << 30, free_bytes=8 << 30))]
            m, tasks, splits, roots = metrics.parse_build(profile, (d / "build.log").read_text(),
                                                         [40, 12000000, 0, 0, 0, 100], resources)
            self.assertEqual(len(splits), count)
            self.assertEqual(sum(roots.values()), 60_000_000)
            self.assertEqual(sum(t["rows"] for t in tasks), 60_000_000)
            self.assertEqual(m["task_equivalence_key"] is None, count > 0)
            self.assertEqual(sum(b["tasks"] for b in m["subtree_buckets"]), len(tasks))
            self.assertEqual(m["uploads_above_target"] > 0, count == 0)
            with self.assertRaises(ValueError):
                metrics.parse_build(profile, (d / "build.log").read_text(), [40] * 6, [])

    def test_ordinary_sq8_reference_uses_existing_nested_staging_fields(self):
        path = sweep.ROOT / "runs/sq4_sq8_laion100m_incremental_20261107/S2"
        if not path.exists():
            self.skipTest("historical reference is not shipped")
        p = json.loads((path / "profile.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "resources.jsonl").write_text(json.dumps(dict(gpu=dict(process_used_bytes=15 << 30, free_bytes=8 << 30))) + "\n")
            audit, m = sweep.Experiment.reference_metrics(None, p, "", [644, 145000000, 0, 0, 0, 0], d)
            self.assertTrue(audit["accepted"])
            self.assertEqual(m["wall_s"], 644)
            p["gpu_subtree"]["sq_gpu_staging"]["d0_assign_rows"] = 10_000_000
            with self.assertRaises(ValueError):
                sweep.Experiment.reference_metrics(None, p, "", [644] * 6, d)


if __name__ == "__main__":
    unittest.main()
