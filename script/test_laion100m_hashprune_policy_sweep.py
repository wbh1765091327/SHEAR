import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "script" / "run_laion100m_hashprune_policy_sweep.sh"

EXPECTED_MATRIX = [
    ("hashslot_q1", "rhalf-hash-slot-1", 1, "1.00"),
    ("distance_first_q2", "rhalf-distance-first-q3", 2, "1.00"),
    ("distance_first_q3", "rhalf-distance-first-q3", 3, "1.00"),
    ("hard_tier_q2", "rhalf-hard-tier-q3", 2, "strict"),
    ("hard_tier_q3", "rhalf-hard-tier-q3", 3, "strict"),
] + [
    (f"soft_tier_q{quota}_a{str(alpha).replace('.', 'p')}", "rhalf-soft-tier-q3", quota, f"{alpha:.2f}")
    for quota in (2, 3)
    for alpha in (1.05, 1.10, 1.20, 1.40, 2.00)
]


class RunnerContractTest(unittest.TestCase):
    def runner_text(self):
        return RUNNER.read_text()

    def test_matrix_is_exact_and_frozen(self):
        text = self.runner_text()
        match = re.search(r"# BEGIN FROZEN MATRIX\n(.*?)# END FROZEN MATRIX", text, re.S)
        self.assertIsNotNone(match)
        observed = []
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("MATRIX=") or line == ")":
                continue
            fields = line.strip("'").split("|")
            observed.append((fields[1], fields[2], int(fields[3]), fields[4]))
        self.assertEqual(observed, EXPECTED_MATRIX)

    def test_one_graph_level_warmup_and_opposing_passes(self):
        text = self.runner_text()
        self.assertIn("PASS1_L=(25 50 100 200 400)", text)
        self.assertIn("PASS2_L=(400 200 100 50 25)", text)
        sweep = re.search(r"run_search_sweep\(\) \{\n(.*?)\n\}\n\nfsync_evidence\(\)", text, re.S)
        self.assertIsNotNone(sweep)
        sweep_text = sweep.group(1)
        self.assertEqual(sweep_text.count('run_search_once "$case_dir" 400 warmup warmup'), 1)
        self.assertEqual(sweep_text.count('for l_value in "${PASS1_L[@]}"'), 1)
        self.assertEqual(sweep_text.count('for l_value in "${PASS2_L[@]}"'), 1)
        self.assertNotRegex(sweep_text, r"for l_value.*\n(?:.*\n){0,8}.*warmup")

    def test_prepare_and_preflight_require_cuda_feature(self):
        text = self.runner_text()
        self.assertIn("--features build_shear_index/cuda", text)
        self.assertIn("SHEAR was built without CUDA support", text)


class RunnerSmokeTest(unittest.TestCase):
    def run_smoke(self, output_base, run_id, extra_env=None):
        env = os.environ.copy()
        env.update({"OUTPUT_BASE": str(output_base), "RUN_ID": run_id})
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(RUNNER), "smoke"],
            cwd=REPO,
            env=env,
            text=True,
            capture_output=True,
        )

    def test_smoke_accepts_fifteen_transactions_and_records_call_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            completed = self.run_smoke(output, "accepted")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            root = output / "accepted"
            self.assertEqual((root / "STATUS").read_text().strip(), "accepted")
            transactions = sorted((root / "transactions").iterdir())
            self.assertEqual(len(transactions), 15)
            for transaction in transactions:
                self.assertEqual((transaction / "STATUS").read_text().strip(), "accepted")
                with (transaction / "search_calls.tsv").open() as stream:
                    calls = list(csv.DictReader(stream, delimiter="\t"))
                self.assertEqual(sum(row["kind"] == "warmup" for row in calls), 1)
                self.assertEqual(sum(row["kind"] == "measured" for row in calls), 10)
                self.assertEqual([int(row["L"]) for row in calls if row["kind"] == "measured"],
                                 [25, 50, 100, 200, 400, 400, 200, 100, 50, 25])

    def test_smoke_failure_is_preserved_without_retry_or_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            completed = self.run_smoke(
                output,
                "mismatch",
                {"SWEEP_SMOKE_MISMATCH_LABEL": "distance_first_q2"},
            )
            self.assertNotEqual(completed.returncode, 0)
            root = output / "mismatch"
            failing = root / "transactions" / "002_distance_first_q2"
            self.assertEqual((root / "STATUS").read_text().strip(), "failed")
            self.assertEqual((failing / "STATUS").read_text().strip(), "failed")
            self.assertTrue((failing / "index" / "_mem.index").exists())
            self.assertEqual((failing / "attempt_count").read_text().strip(), "1")
            self.assertFalse((root / "transactions" / "003_distance_first_q3").exists())

    def test_existing_run_root_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = self.run_smoke(output, "same")
            second = self.run_smoke(output, "same")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)


class SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        summary_path = REPO / "script" / "summarize_laion100m_hashprune_policy_sweep.py"
        import importlib.util
        specification = importlib.util.spec_from_file_location("hashprune_sweep_summary", summary_path)
        cls.summary = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = cls.summary
        specification.loader.exec_module(cls.summary)

    def point(self, label, recall, qps, degree=44.0, graph_bytes=18_000_000_000, spread=0.01):
        return {
            "label": label,
            "recall": recall,
            "mean_qps": qps,
            "avg_out_degree": degree,
            "graph_bytes": graph_bytes,
            "qps_relative_spread": spread,
        }

    def test_pareto_dominance_and_ties(self):
        points = [
            self.point("a", 0.95, 1000),
            self.point("b", 0.96, 1000),
            self.point("c", 0.96, 900),
            self.point("d", 0.94, 1100),
            self.point("e", 0.96, 1000),
        ]
        frontier = self.summary.pareto_frontier(points)
        self.assertEqual({point["label"] for point in frontier}, {"b", "d", "e"})

    def test_budget_and_recommendation_gates(self):
        base = self.point("base", 0.9500, 1000)
        recall_winner = self.point("recall", 0.9511, 1001, degree=44.2, graph_bytes=18_100_000_000)
        qps_winner = self.point("qps", 0.9505, 1021)
        unstable = self.point("unstable", 0.9520, 1050, spread=0.051)
        unmatched = self.point("unmatched", 0.9600, 1200, degree=44.26)
        self.assertTrue(self.summary.budgets_match(base, recall_winner))
        self.assertFalse(self.summary.budgets_match(base, unmatched))
        self.assertEqual(self.summary.pairwise_decision(recall_winner, base), "recall")
        self.assertEqual(self.summary.pairwise_decision(qps_winner, base), "qps")
        self.assertIsNone(self.summary.pairwise_decision(unstable, base))

    def write_synthetic_batch(self, root):
        transactions = root / "transactions"
        transactions.mkdir(parents=True)
        for ordinal, (label, policy, quota, alpha) in enumerate(EXPECTED_MATRIX, start=1):
            case = transactions / f"{ordinal:03d}_{label}"
            case.mkdir()
            (case / "STATUS").write_text("accepted\n")
            (case / "parameter_passport.json").write_text(json.dumps({
                "ordinal": int(ordinal), "label": label, "policy_mode": policy,
                "requested_hash_quota": quota, "priority_alpha": alpha,
            }) + "\n")
            (case / "accepted_metrics.json").write_text(json.dumps({
                "graph": {"graph_bytes": 18_000_000_000, "avg_out_degree": 44.0},
                "graph_quality": {"directed_reachable_from_start_ratio": "0.98"},
                "hashprune_backfill": {"backfill_edges": 10, "rows_with_backfill": 5},
                "resource": {"peak_process_rss_bytes": 100, "peak_hbm_mib": 200},
            }) + "\n")
            with (case / "search_measurements.tsv").open("w", newline="") as stream:
                fields = ["pass", "L", "recall", "qps", "result_sha256", "total_one_bit_evaluations",
                          "total_exact_evaluations", "logical_exact_vector_bytes"]
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for l_value in (25, 50, 100, 200, 400):
                    for pass_name in ("pass1", "pass2"):
                        recall = 0.90 + l_value / 10000
                        if label == "soft_tier_q2_a1p05":
                            recall += 0.002
                        writer.writerow({
                            "pass": pass_name, "L": l_value, "recall": f"{recall:.6f}",
                            "qps": "1000", "result_sha256": f"digest-{label}-{l_value}",
                            "total_one_bit_evaluations": "100",
                            "total_exact_evaluations": "10", "logical_exact_vector_bytes": "1000",
                        })

    def test_atomic_summary_outputs_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            self.write_synthetic_batch(root)
            summary_script = REPO / "script" / "summarize_laion100m_hashprune_policy_sweep.py"
            completed = subprocess.run(
                ["python3", str(summary_script), "--batch-root", str(root)],
                cwd=REPO,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = root / "report"
            for name in ("all_configurations.tsv", "recall_qps_pareto.tsv", "matched_budget.tsv",
                         "per_method_winners.tsv", "summary.json", "summary.md"):
                self.assertTrue((report / name).is_file(), name)
            with (report / "all_configurations.tsv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream, delimiter="\t"))), 150)
            summary = json.loads((report / "summary.json").read_text())
            self.assertEqual(summary["accepted_configurations"], 15)
            self.assertEqual(summary["measured_rows"], 150)
            self.assertEqual(summary["aggregated_points"], 75)
            self.assertTrue(all(row["winner"] == "soft_tier_q2_a1p05"
                                for row in summary["recommendations"]))


if __name__ == "__main__":
    unittest.main()
