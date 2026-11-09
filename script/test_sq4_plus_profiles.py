#!/usr/bin/env python3
"""Guard scale-specific recipes, paper agreement, and effective capacity checks."""
import copy
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

import run_sq4_plus as runner


class Profiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, rows, **kwargs):
        path = self.root / f"{rows}.fbin"
        with path.open("wb") as f:
            f.write(struct.pack("<II", rows, 512))
            f.truncate(8 + rows * 512 * 4)
        return runner.make_manifest(path, "/usr/bin/true", self.root / "run", **kwargs)

    def test_scale_selects_distinct_recipes_and_removes_fixed_routing(self):
        for rows, name, fanout, sample, memory in [
            (10_000_000, "laion10m", "6", "0.002", "256"),
            (100_000_000, "laion100m", "8", "0.01", "0"),
        ]:
            m, _ = self.manifest(rows)
            c = m["command"]
            self.assertEqual(m["profile"], name)
            for flag, value in {"--fanout-top": fanout,
                                "--gpu-subtree-d0-tagore-sample-fraction": sample,
                                "--oom-memory-budget-gb": memory}.items():
                self.assertEqual(c[c.index(flag) + 1], value)
            self.assertNotIn("--gpu-subtree-d0-partition-mode", c)
            self.assertNotIn("--gpu-subtree-d0-capacity-root-leader-cap", c)
            self.assertNotIn("--gpu-subtree-d0-tagore-partitions", c)
            self.assertNotIn("--gpu-subtree-d0-tagore-k", c)
            for kind, ratio in [("high", "0.5"), ("target", "0.08"), ("low", "0.08")]:
                flag = "--gpu-subtree-d0-capacity-" + kind + "-ratio"
                self.assertEqual(c[c.index(flag) + 1], ratio)

    def test_100m_common_parameters_match_executable_paper_runbook(self):
        source = (runner.ROOT / "docs/runbooks/p1_p2_shear_four_dataset.md").read_text()
        command = source.split("run_shear_build() {", 1)[1].split("validate_shear_profile", 1)[0]
        m, env = self.manifest(100_000_000)
        c = m["command"]
        for flag in ["--data-type", "--dist-fn", "--oom-memory-budget-gb", "--oom-io-mode",
                     "--max-degree", "--lbuild", "--num-threads", "--leaf-size-min",
                     "--leaf-size-max", "--leaf-knn", "--psamp-fraction", "--fanout-top",
                     "--fanout-second", "--root-risk-candidate-rs", "--seed", "--partitioner",
                     "--gpu-subtree-assignment-backend", "--gpu-subtree-leaf-topk-backend",
                     "--gpu-subtree-execution-mode",
                     "--gpu-subtree-d0-capacity-target-ratio", "--gpu-subtree-d0-capacity-low-ratio",
                     "--gpu-subtree-persistent-large-ticket-batch", "--gpu-subtree-sq8-leaf-mode",
                     "--gpu-subtree-persistent-scheduler-mode"]:
            match = re.search(re.escape(flag) + r"\s+(\S+)", command)
            self.assertIsNotNone(match, flag)
            self.assertEqual(c[c.index(flag) + 1], match[1], flag)
        # The user's upload safety threshold intentionally differs from the paper's 8%.
        self.assertEqual(c[c.index("--gpu-subtree-d0-capacity-high-ratio") + 1], "0.5")
        for key in ["SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK", "SHEAR_PIPNN_WMMA_CTA_SHARED_B",
                    "SHEAR_PIPNN_COMPACT_LUNE_WITNESS", "SHEAR_PIPNN_COMPACT_LUNE_PRUNE",
                    "SHEAR_PIPNN_COMPACT_SPILL_DIGEST", "SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS"]:
            self.assertIn(f"{key}={env[key]}", command)

    def test_fixed_routing_count_is_explicit_and_does_not_change_encoding(self):
        m, _ = self.manifest(100_000_000, routing_leaders=528)
        c = m["command"]
        self.assertEqual(m["routing_leaders"], 528)
        for flag in ["--gpu-subtree-d0-capacity-root-leader-cap",
                     "--gpu-subtree-d0-capacity-quality-min-root-leaders"]:
            self.assertEqual(c[c.index(flag)+1], "528")
        self.assertIn("--gpu-subtree-sq4-primary-center", c)
        with self.assertRaises(ValueError):
            self.manifest(8193, profile="laion100m", validation_prefix=True, routing_leaders=10000)

    def test_inherited_experiments_cannot_silently_change_recipe(self):
        m, env = self.manifest(10_000_000, inherited={
            "SHEAR_PIPNN_SQ4_D0_BACKEND": "cpu", "SHEAR_PIPNN_SQ4_D1_BALANCED": "1",
            "SHEAR_PIPNN_SQ8_WMMA_OPT_MASK": "4785664", "SHEAR_UNRECOGNIZED_EXPERIMENT": "1",
            "OPENBLAS_NUM_THREADS": "64", "LD_LIBRARY_PATH": "/example"})
        self.assertEqual(env["SHEAR_PIPNN_SQ4_D0_BACKEND"], "gpu-exact")
        self.assertEqual(env["SHEAR_PIPNN_SQ4_D1_BALANCED"], "0")
        self.assertEqual(env["SHEAR_PIPNN_SQ8_WMMA_OPT_MASK"], "525824")
        self.assertNotIn("SHEAR_UNRECOGNIZED_EXPERIMENT", env)
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/example")

    def test_prefix_is_explicit_and_cannot_be_labeled_as_full_100m(self):
        with self.assertRaises(ValueError):
            self.manifest(10_000_000, profile="laion100m")
        m, _ = self.manifest(10_000_000, profile="laion100m", validation_prefix=True, host_gib=256)
        self.assertTrue(m["validation_prefix"])
        self.assertTrue(m["resource_override"])

    def test_sweep_overrides_are_explicit_and_do_not_change_default_recipes(self):
        before = copy.deepcopy(runner.PROFILES)
        m, env = self.manifest(100_000_000, fanout_top=6, sample_fraction=0.002,
                              planning_budget_ratio=0.16, admission_threshold_ratio=0.32, host_gib=256)
        c = m["command"]
        for flag, value in {"--fanout-top": "6", "--root-risk-candidate-rs": "1,2,4,6",
                            "--gpu-subtree-d0-tagore-sample-fraction": "0.002",
                            "--gpu-subtree-d0-capacity-target-ratio": "0.16",
                            "--gpu-subtree-d0-capacity-low-ratio": "0.16",
                            "--gpu-subtree-d0-capacity-high-ratio": "0.32"}.items():
            self.assertEqual(c[c.index(flag) + 1], value)
        self.assertEqual(runner.PROFILES, before)
        self.assertEqual(env["SHEAR_PIPNN_SQ4_STREAM_GATHER"], "1")
        self.assertNotIn("--gpu-subtree-d0-capacity-root-leader-cap", c)

    def test_invalid_ratios_samples_and_fanout_fail_before_execution(self):
        for kwargs in [
            dict(planning_budget_ratio=0), dict(planning_budget_ratio=0.6),
            dict(admission_threshold_ratio=float("nan")), dict(admission_threshold_ratio=1.01),
            dict(sample_fraction=0), dict(sample_fraction=float("inf")),
            dict(fanout_top=0), dict(fanout_top=33), dict(fanout_top=3.5),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.manifest(100_000_000, **kwargs)

    def test_saved_10m_profile_rejects_capacity_and_routing_drift(self):
        evidence = runner.ROOT / "runs/sq4_budget_paper8_20261108/paper8-B1"
        if not evidence.exists():
            self.skipTest("local full-build evidence is not included in the checkout")
        profile = json.loads((evidence / "profile.json").read_text())
        log = (evidence / "build.log").read_text()
        m, _ = self.manifest(10_000_000)
        with self.assertRaises(ValueError):
            runner.audit_profile(m, profile, log)
        # Historical 8% admission is valid only when explicitly recorded in its command.
        m["command"][m["command"].index("--gpu-subtree-d0-capacity-high-ratio") + 1] = "0.08"
        self.assertTrue(runner.audit_profile(m, profile, log)["accepted"])
        for key, value in [("c_target", 9_475_197), ("root_leader_cap", 64)]:
            changed = copy.deepcopy(profile)
            changed["gpu_subtree"]["d0_capacity"][key] = value
            with self.assertRaises(ValueError):
                runner.audit_profile(m, changed, log)
        changed = copy.deepcopy(profile)
        changed["gpu_subtree"]["d1plus_lazy_split_mode"] = "off"
        with self.assertRaises(ValueError):
            runner.audit_profile(m, changed, log)


if __name__ == "__main__":
    unittest.main()
