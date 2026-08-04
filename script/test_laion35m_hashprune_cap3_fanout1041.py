#!/usr/bin/env python3
import re
import shlex
import subprocess
import tempfile
import unittest
import json
import os
import csv
import hashlib
import struct
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "script" / "run_laion35m_hashprune_cap3_fanout1041.sh"
SUMMARIZER = REPO / "script" / "summarize_laion35m_hashprune_cap3_fanout1041.py"
RUNBOOK = REPO / "docs" / "runbooks" / "laion35m_hashprune_cap3_fanout1041.md"

EXPECTED_MATRIX = [
    "001|plain|plain|off|NA|NA",
    "002|c6_hashslot_q1|hashslot|rhalf-hash-slot-1|6|NA",
    "003|c6_distance_first_q3|distance_first|rhalf-distance-first-q3|6|NA",
    "004|c6_soft_tier_q3_a1p02|soft_tier|rhalf-soft-tier-q3|6|1.02",
    "005|c6_hard_tier_q3|hard_tier|rhalf-hard-tier-q3|6|NA",
]


def runner_text() -> str:
    return RUNNER.read_text()


def frozen_matrix(text: str) -> list[str]:
    match = re.search(
        r"# BEGIN FROZEN MATRIX\nMATRIX=\(\n(?P<body>.*?)\n\)\n# END FROZEN MATRIX",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("runner is missing a parseable frozen matrix")
    rows = []
    for raw in match.group("body").splitlines():
        value = raw.strip()
        if not value:
            continue
        rows.append(shlex.split(value)[0])
    return rows


class RunnerContractTest(unittest.TestCase):
    def test_frozen_matrix_is_exact_and_ordered(self):
        self.assertEqual(frozen_matrix(runner_text()), EXPECTED_MATRIX)

    def test_frozen_dataset_and_output_paths(self):
        text = runner_text()
        expected = {
            "OUTPUT_BASE": "/data_staging/wbh/hashprune_softtier_cap6/laion35m_f421",
            "BASE": "/data_staging/wbh/laion-512-35M/base.fbin",
            "QUERY": "/data_nvme_kioxia/tyj/data/laion-512-100M/query.fbin",
            "GT": "/data_staging/wbh/laion-512-35M/gt100.bin",
            "RABITQ1_SIDECAR": (
                "/data_staging/wbh/unified_graph_rabitq1/sidecars/"
                "laion35m.rabitq1"
            ),
            "RABITQ1_MANIFEST": (
                "/data_staging/wbh/unified_graph_rabitq1/sidecars/"
                "laion35m.rabitq1.json"
            ),
            "ROWS": "35000000",
            "DIM": "512",
        }
        for name, value in expected.items():
            if name == "OUTPUT_BASE":
                needle = (
                    "OUTPUT_BASE=${OUTPUT_BASE:-"
                    "/data_staging/wbh/hashprune_softtier_cap6/laion35m_f421}"
                )
            else:
                needle = f"{name}={value}"
            self.assertIn(needle, text)

    def test_build_and_search_passport_is_frozen(self):
        text = runner_text()
        for needle in (
            "--max-degree 64",
            "--lbuild 100",
            "--num-threads 8",
            "--m-hash-bits 12",
            "--fanout-top 4",
            "--fanout-second 2",
            "--root-risk-candidate-rs 1,2,4",
            "--seed 42",
            "--hashprune-backfill-alpha 1.0",
            "--hashprune-rhalf-max-backfill-per-row",
            "--hashprune-soft-tier-priority-alpha",
            "--search-backend rabitq1-adaptive-exact",
            "--storage mmap",
            "--recall-at 10",
        ):
            self.assertIn(needle, text)
        for forbidden in (
            "--fanout-top 10",
            "--fanout-second 4",
            "--root-risk-candidate-rs 1,2,4,8,10",
        ):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("--hashprune-rhalf-hash-quota", text)

    def test_plain_is_not_a_continuation_gate(self):
        text = runner_text()
        self.assertNotIn("awaiting_user_continue", text)
        self.assertNotRegex(text, r"\bpilot\)")
        self.assertNotIn("avg_out_degree >=", text)
        self.assertNotIn("recall >= ", text)

    def test_search_schedule_has_one_warmup_and_one_measured_pass(self):
        text = runner_text()
        self.assertIn("MEASURED_L=(10 16 25 32 50 100 200)", text)
        self.assertEqual(text.count('run_search_once "$case_dir" 200 warmup warmup'), 1)
        self.assertEqual(
            text.count('run_search_once "$case_dir" "$l_value" measured measured'),
            1,
        )
        self.assertNotIn("reverse", text.lower())

    def test_public_modes_are_exact(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "help"],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        mode_lines = [
            line.strip().split()[0]
            for line in result.stdout.splitlines()
            if re.match(
                r"^\s+(help|prepare|build-sidecar|preflight|smoke|formal|summarize)\s",
                line,
            )
        ]
        self.assertEqual(
            mode_lines,
            [
                "help",
                "prepare",
                "build-sidecar",
                "preflight",
                "smoke",
                "formal",
                "summarize",
            ],
        )

    def test_help_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            result = subprocess.run(
                ["bash", str(RUNNER), "help"],
                cwd=REPO,
                env={"PATH": "/usr/bin:/bin", "OUTPUT_BASE": str(output)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())

    def test_zero_candidate_pressure_is_diagnostic_not_a_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profile.json"
            output = root / "policy_validation.json"
            profile.write_text(
                json.dumps(
                    {
                        "hashprune_backfill": {
                            "mode": "rhalf-distance-first-q3",
                            "requested_hash_quota": None,
                            "effective_hash_quota": 3,
                            "derived_budget": 32,
                            "requested_max_backfill_per_row": 3,
                            "effective_max_backfill_per_row": 3,
                            "capacity_rule": "min-derived-budget-row-cap-spare",
                            "requested_priority_alpha": None,
                            "effective_priority_alpha": 1.0,
                            "shortlist_replacements": 0,
                            "shortlist_capacity_rejections": 0,
                        }
                    }
                )
                + "\n"
            )
            script = f"""
set -- help
source {shlex.quote(str(RUNNER))}
validate_profile \
  {shlex.quote(str(profile))} \
  rhalf-distance-first-q3 \
  3 \
  NA \
  {shlex.quote(str(output))}
"""
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            validation = json.loads(output.read_text())
            self.assertFalse(validation["selection_pressure_observed"])


class RunnerSmokeTest(unittest.TestCase):
    def run_smoke(
        self,
        output: Path,
        run_id: str,
        *,
        fail_label: str | None = None,
    ) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment.update({"OUTPUT_BASE": str(output), "RUN_ID": run_id})
        if fail_label is not None:
            environment["LAION35M_SMOKE_FAIL_LABEL"] = fail_label
        return subprocess.run(
            ["bash", str(RUNNER), "smoke"],
            cwd=REPO,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_smoke_executes_frozen_matrix_and_search_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            result = self.run_smoke(output, "accepted-smoke")
            self.assertEqual(result.returncode, 0, result.stderr)
            batch = output / "accepted-smoke"
            self.assertEqual((batch / "STATUS").read_text().strip(), "accepted")
            transactions = sorted((batch / "transactions").iterdir())
            self.assertEqual(
                [path.name for path in transactions],
                [
                    "001_plain",
                    "002_c6_hashslot_q1",
                    "003_c6_distance_first_q3",
                    "004_c6_soft_tier_q3_a1p02",
                    "005_c6_hard_tier_q3",
                ],
            )
            cap_commands = 0
            alpha_commands = 0
            for case in transactions:
                self.assertEqual((case / "attempt_count").read_text().strip(), "1")
                self.assertEqual((case / "STATUS").read_text().strip(), "accepted")
                command = shlex.split((case / "build_command.sh").read_text())
                self.assertIn("--fanout-top", command)
                self.assertEqual(command[command.index("--fanout-top") + 1], "4")
                self.assertEqual(command[command.index("--fanout-second") + 1], "2")
                self.assertEqual(
                    command[command.index("--root-risk-candidate-rs") + 1],
                    "1,2,4",
                )
                if "--hashprune-rhalf-max-backfill-per-row" in command:
                    cap_commands += 1
                    self.assertEqual(
                        command[
                            command.index("--hashprune-rhalf-max-backfill-per-row") + 1
                        ],
                        "6",
                    )
                if "--hashprune-soft-tier-priority-alpha" in command:
                    alpha_commands += 1
                    self.assertEqual(
                        command[
                            command.index("--hashprune-soft-tier-priority-alpha") + 1
                        ],
                        "1.02",
                    )
                calls = [
                    line.split("\t")
                    for line in (case / "search_calls.tsv").read_text().splitlines()[1:]
                ]
                self.assertEqual(calls[0], ["warmup", "warmup", "200"])
                self.assertEqual(
                    calls[1:],
                    [
                        ["measured", "measured", str(value)]
                        for value in (10, 16, 25, 32, 50, 100, 200)
                    ],
                )
                for retained in (
                    "parameter_passport.json",
                    "profile.json",
                    "graph_validation.json",
                    "graph_quality.tsv",
                    "resource_summary.json",
                    "search_measurements.tsv",
                    "accepted_metrics.json",
                    "build_command.sh",
                    "search_calls.tsv",
                ):
                    self.assertTrue((case / retained).is_file(), f"missing {retained}")
                for removed in ("index", "oom", "sq_sidecar"):
                    self.assertFalse((case / removed).exists())
            self.assertEqual(cap_commands, 4)
            self.assertEqual(alpha_commands, 1)
            manifest = json.loads((batch / "experiment_manifest.json").read_text())
            self.assertEqual(manifest["graphs"], 5)
            self.assertEqual(manifest["warmups"], 5)
            self.assertEqual(manifest["measured_rows"], 35)
            self.assertTrue((batch / "report" / "summary.json").is_file())
            self.assertTrue((batch / "report" / "summary.md").is_file())

    def test_smoke_failure_preserves_payload_and_stops_later_transactions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            result = self.run_smoke(
                output,
                "failed-smoke",
                fail_label="c6_distance_first_q3",
            )
            self.assertNotEqual(result.returncode, 0)
            batch = output / "failed-smoke"
            self.assertEqual((batch / "STATUS").read_text().strip(), "failed")
            transactions = batch / "transactions"
            self.assertTrue((transactions / "001_plain").is_dir())
            self.assertTrue((transactions / "002_c6_hashslot_q1").is_dir())
            failed = transactions / "003_c6_distance_first_q3"
            self.assertTrue(failed.is_dir())
            self.assertEqual((failed / "STATUS").read_text().strip(), "failed")
            self.assertEqual((failed / "attempt_count").read_text().strip(), "1")
            self.assertTrue((failed / "index" / "_mem.index").is_file())
            self.assertTrue((failed / "oom").is_dir())
            self.assertTrue((failed / "sq_sidecar").is_dir())
            self.assertFalse((transactions / "004_c6_soft_tier_q3_a1p02").exists())
            self.assertFalse((transactions / "005_c6_hard_tier_q3").exists())


class RunnerFormalOrchestrationTest(unittest.TestCase):
    def run_synthetic_preflight(
        self,
        root: Path,
        run_id: str,
        *,
        create_existing: bool = False,
    ) -> subprocess.CompletedProcess:
        output = root / "output"
        if create_existing:
            (output / run_id).mkdir(parents=True)
        script = f"""
set -- help
source {shlex.quote(str(RUNNER))}
OUTPUT_BASE={shlex.quote(str(output))}
RUN_ID={shlex.quote(run_id)}
acquire_locks() {{ :; }}
validate_inputs() {{ :; }}
run_preflight
"""
        return subprocess.run(
            ["bash", "-c", script],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_preflight_requires_a_safe_new_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self.run_synthetic_preflight(root, "")
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("RUN_ID is required", missing.stderr)
            unsafe = self.run_synthetic_preflight(root, "../escape")
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertIn("unsafe RUN_ID", unsafe.stderr)
            existing = self.run_synthetic_preflight(
                root,
                "existing",
                create_existing=True,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn("run root already exists", existing.stderr)
            accepted = self.run_synthetic_preflight(root, "fresh-formal")
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertFalse((root / "output" / "fresh-formal").exists())

    def run_synthetic_formal(
        self,
        root: Path,
        *,
        fail_label: str = "",
    ) -> tuple[subprocess.CompletedProcess, Path]:
        output = root / "output"
        fake_summarizer = root / "fake_summarizer.py"
        fake_summarizer.write_text("raise SystemExit(0)\n")
        batch = output / "formal-orchestration"
        script = f"""
set -- help
source {shlex.quote(str(RUNNER))}
OUTPUT_BASE={shlex.quote(str(output))}
RUN_ID=formal-orchestration
BATCH_ROOT=
SUMMARIZER={shlex.quote(str(fake_summarizer))}
acquire_locks() {{ :; }}
validate_inputs() {{ :; }}
write_provenance() {{
  write_matrix "$BATCH_ROOT/frozen_matrix.tsv"
  printf '{{"schema":"laion35m_hashprune_cap6_fanout421_v1","graphs":5,"warmups":5,"measured_rows":35}}\\n' >"$BATCH_ROOT/experiment_manifest.json"
}}
run_formal_case() {{
  local row=$1 ordinal label
  IFS='|' read -r ordinal label _ <<<"$row"
  local case_dir="$BATCH_ROOT/transactions/${{ordinal}}_${{label}}"
  mkdir -p "$case_dir"
  atomic_status "$case_dir/STATUS" running
  printf '1\\n' >"$case_dir/attempt_count"
  printf '%s\\n' "$label" >>"$BATCH_ROOT/formal_order.log"
  if [[ "$label" == {shlex.quote(fail_label)} ]]; then
    return 7
  fi
  atomic_status "$case_dir/STATUS" accepted
}}
run_formal
"""
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result, batch

    def test_formal_orchestrator_runs_five_rows_once_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, batch = self.run_synthetic_formal(Path(temporary))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((batch / "STATUS").read_text().strip(), "accepted")
            self.assertEqual(
                (batch / "formal_order.log").read_text().splitlines(),
                [
                    "plain",
                    "c6_hashslot_q1",
                    "c6_distance_first_q3",
                    "c6_soft_tier_q3_a1p02",
                    "c6_hard_tier_q3",
                ],
            )
            attempts = sorted((batch / "transactions").glob("*/attempt_count"))
            self.assertEqual(len(attempts), 5)
            self.assertTrue(all(path.read_text().strip() == "1" for path in attempts))

    def test_formal_orchestrator_does_not_retry_or_continue_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, batch = self.run_synthetic_formal(
                Path(temporary),
                fail_label="c6_distance_first_q3",
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual((batch / "STATUS").read_text().strip(), "failed")
            self.assertEqual(
                (batch / "formal_order.log").read_text().splitlines(),
                ["plain", "c6_hashslot_q1", "c6_distance_first_q3"],
            )
            failed = batch / "transactions" / "003_c6_distance_first_q3"
            self.assertEqual((failed / "STATUS").read_text().strip(), "failed")
            self.assertEqual((failed / "attempt_count").read_text().strip(), "1")
            self.assertFalse(
                (batch / "transactions" / "004_c6_soft_tier_q3_a1p02").exists()
            )


class SidecarLifecycleTest(unittest.TestCase):
    def write_fbin(self, path: Path, rows: int, dim: int):
        path.write_bytes(struct.pack("<II", rows, dim) + bytes(rows * dim * 4))

    def create_fixture(self, root: Path) -> dict[str, Path | int]:
        rows, dim = 4, 2
        base = root / "base.fbin"
        query = root / "query.fbin"
        gt = root / "gt.bin"
        self.write_fbin(base, rows, dim)
        self.write_fbin(query, 10_000, dim)
        gt.write_bytes(struct.pack("<II", 10_000, 10) + bytes(10_000 * 10 * 4))
        builder = root / "fake_builder.py"
        builder.write_text(
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

output = Path(value("--output"))
data = Path(value("--data-path"))
log = Path(os.environ["FAKE_BUILDER_LOG"])
with log.open("a") as stream:
    stream.write(("validate" if "--validate-only" in args else "build") + "\\n")
if "--validate-only" in args:
    if not output.is_file() or not output.with_name(output.name + ".json").is_file():
        raise SystemExit(9)
    print('{"schema":"graph_rabitq1_validation_v1"}')
    raise SystemExit(0)
rows = int.from_bytes(data.read_bytes()[0:4], "little")
dim = int.from_bytes(data.read_bytes()[4:8], "little")
padded = ((dim + 63) // 64) * 64
stride = padded // 8 + 12
size = 256 + padded // 2 + 16 * padded * 4 + rows * (1 + stride)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(bytes(size))
manifest = {
    "schema": "graph_rabitq1_manifest_v1",
    "rows": rows,
    "dim": dim,
    "padded_dim": padded,
    "center_count": 16,
    "code_bytes": padded // 8,
    "bin_stride": stride,
    "ex_data_bytes": 0,
    "sidecar_bytes": size,
    "base_full_crc64": 1,
    "base_window_fingerprint": 2,
    "output_crc64": 3,
    "elapsed_seconds": 0.1,
}
output.with_name(output.name + ".json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n"
)
print(json.dumps(manifest))
"""
        )
        builder.chmod(0o755)
        return {
            "rows": rows,
            "dim": dim,
            "base": base,
            "query": query,
            "gt": gt,
            "builder": builder,
            "log": root / "builder.log",
            "sidecar": root / "sidecars" / "laion35m.rabitq1",
            "output": root / "output",
        }

    def run_build_sidecar(
        self,
        fixture: dict[str, Path | int],
        *,
        rebuild: bool = False,
    ) -> subprocess.CompletedProcess:
        sidecar = fixture["sidecar"]
        assert isinstance(sidecar, Path)
        script = f"""
set -- help
source {shlex.quote(str(RUNNER))}
ROWS={fixture["rows"]}
DIM={fixture["dim"]}
BASE={shlex.quote(str(fixture["base"]))}
QUERY={shlex.quote(str(fixture["query"]))}
GT={shlex.quote(str(fixture["gt"]))}
RABITQ_BUILDER={shlex.quote(str(fixture["builder"]))}
RABITQ1_SIDECAR={shlex.quote(str(sidecar))}
RABITQ1_MANIFEST={shlex.quote(str(sidecar) + ".json")}
OUTPUT_BASE={shlex.quote(str(fixture["output"]))}
REBUILD_SIDECAR={"1" if rebuild else "0"}
acquire_locks() {{ :; }}
run_build_sidecar
"""
        environment = os.environ.copy()
        environment["FAKE_BUILDER_LOG"] = str(fixture["log"])
        return subprocess.run(
            ["bash", "-c", script],
            cwd=REPO,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_build_sidecar_publishes_valid_pair_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.create_fixture(Path(temporary))
            first = self.run_build_sidecar(fixture)
            self.assertEqual(first.returncode, 0, first.stderr)
            sidecar = fixture["sidecar"]
            assert isinstance(sidecar, Path)
            manifest = Path(str(sidecar) + ".json")
            self.assertTrue(sidecar.is_file())
            self.assertTrue(manifest.is_file())
            padded = 64
            expected = 256 + padded // 2 + 16 * padded * 4 + 4 * (1 + 20)
            self.assertEqual(sidecar.stat().st_size, expected)
            self.assertEqual(
                Path(fixture["log"]).read_text().splitlines(),
                ["build", "validate", "validate"],
            )
            self.assertEqual(
                (Path(fixture["output"]) / "sidecar_build" / "STATUS")
                .read_text()
                .strip(),
                "accepted",
            )
            shutil.rmtree(Path(fixture["output"]) / "sidecar_build")
            second = self.run_build_sidecar(fixture)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                Path(fixture["log"]).read_text().splitlines(),
                ["build", "validate", "validate", "validate"],
            )
            audit = Path(fixture["output"]) / "sidecar_build"
            self.assertEqual((audit / "STATUS").read_text().strip(), "accepted")
            self.assertTrue((audit / "validation.json").is_file())
            self.assertTrue((audit / "sha256.txt").is_file())

    def test_partial_existing_sidecar_is_rejected_without_rebuild_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.create_fixture(Path(temporary))
            sidecar = fixture["sidecar"]
            assert isinstance(sidecar, Path)
            sidecar.parent.mkdir(parents=True)
            sidecar.write_bytes(b"partial")
            result = self.run_build_sidecar(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("partial or invalid sidecar exists", result.stderr)
            self.assertEqual(sidecar.read_bytes(), b"partial")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class SummaryTest(unittest.TestCase):
    def create_batch(self, root: Path) -> Path:
        output = root / "output"
        result = RunnerSmokeTest().run_smoke(output, "summary-fixture")
        self.assertEqual(result.returncode, 0, result.stderr)
        batch = output / "summary-fixture"

        # Make HardTier deliberately fail the resource-match gate.
        hard = batch / "transactions" / "005_c6_hard_tier_q3"
        accepted_path = hard / "accepted_metrics.json"
        accepted = json.loads(accepted_path.read_text())
        accepted["graph"]["avg_out_degree"] = 43.5
        accepted["graph"]["graph_bytes"] = int(accepted["graph"]["graph_bytes"] * 1.05)
        accepted["graph_quality"]["avg_out_degree"] = 43.5
        accepted_path.write_text(json.dumps(accepted, indent=2, sort_keys=True) + "\n")
        return batch

    def run_summarizer(self, batch: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["python3", str(SUMMARIZER), "--batch-root", str(batch)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_summary_emits_frozen_tables_and_resource_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = self.create_batch(Path(temporary))
            result = self.run_summarizer(batch)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = batch / "report"
            expected_files = (
                "all_configurations.tsv",
                "per_graph_curves.tsv",
                "candidate_pressure.tsv",
                "plain_uplift.tsv",
                "same_cap_comparisons.tsv",
                "resource_matched_comparisons.tsv",
                "summary.json",
                "summary.md",
            )
            for name in expected_files:
                self.assertTrue((report / name).is_file(), f"missing {name}")

            configurations = read_tsv(report / "all_configurations.tsv")
            self.assertEqual(len(configurations), 5)
            self.assertEqual(
                [row["label"] for row in configurations],
                [
                    "plain",
                    "c6_hashslot_q1",
                    "c6_distance_first_q3",
                    "c6_soft_tier_q3_a1p02",
                    "c6_hard_tier_q3",
                ],
            )
            self.assertEqual(configurations[0]["row_cap"], "")
            self.assertEqual(
                [row["row_cap"] for row in configurations[1:]],
                ["6", "6", "6", "6"],
            )

            curves = read_tsv(report / "per_graph_curves.tsv")
            self.assertEqual(len(curves), 35)
            self.assertEqual(
                sorted({int(row["L"]) for row in curves}),
                [10, 16, 25, 32, 50, 100, 200],
            )
            self.assertTrue(
                all(row["search_backend"] == "rabitq1-adaptive-exact" for row in curves)
            )

            pressure = read_tsv(report / "candidate_pressure.tsv")
            self.assertEqual(len(pressure), 4)
            self.assertTrue(
                all(row["requested_row_cap"] == "6" for row in pressure)
            )
            self.assertTrue(
                all(row["effective_row_cap"] == "6" for row in pressure)
            )
            self.assertTrue(
                all(row["cap_pressure_observed"] == "true" for row in pressure)
            )

            plain_uplift = read_tsv(report / "plain_uplift.tsv")
            self.assertEqual(len(plain_uplift), 28)
            self.assertEqual(
                {(row["method"], int(row["L"])) for row in plain_uplift},
                {
                    (method, l_value)
                    for method in (
                        "hashslot",
                        "distance_first",
                        "soft_tier",
                        "hard_tier",
                    )
                    for l_value in (10, 16, 25, 32, 50, 100, 200)
                },
            )
            for row in plain_uplift:
                self.assertAlmostEqual(
                    float(row["recall_delta"]),
                    float(row["method_recall"]) - float(row["plain_recall"]),
                )
                self.assertAlmostEqual(
                    float(row["qps_delta"]),
                    float(row["method_qps"]) - float(row["plain_qps"]),
                )

            same_cap = read_tsv(report / "same_cap_comparisons.tsv")
            self.assertEqual(len(same_cap), 21)
            self.assertEqual(
                {row["competitor_method"] for row in same_cap},
                {"hashslot", "distance_first", "hard_tier"},
            )
            self.assertEqual(
                {row["resources_match"] for row in same_cap},
                {"true", "false"},
            )

            matched = read_tsv(report / "resource_matched_comparisons.tsv")
            self.assertEqual(len(matched), 14)
            self.assertEqual(
                {row["competitor_method"] for row in matched},
                {"hashslot", "distance_first"},
            )

            summary = json.loads((report / "summary.json").read_text())
            self.assertEqual(summary["graphs"], 5)
            self.assertEqual(summary["measured_rows"], 35)
            self.assertTrue(summary["cap_pressure_all_enabled"])
            self.assertEqual(
                summary["resource_matched_competitors"],
                ["hashslot", "distance_first"],
            )
            self.assertFalse(summary["softtier_superiority_supported"])
            self.assertEqual(summary["qps_measurement"], "single-run estimate")
            self.assertEqual(
                set(summary["plain_uplift_low_l"]),
                {"hashslot", "distance_first", "soft_tier", "hard_tier"},
            )
            markdown = (report / "summary.md").read_text()
            self.assertIn("单次运行估计", markdown)
            self.assertIn("L=100, 200", markdown)
            self.assertIn("相对 Plain 的提升", markdown)

    def test_summary_regeneration_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = self.create_batch(Path(temporary))
            first = self.run_summarizer(batch)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = batch / "report"
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(report.iterdir())
                if path.is_file()
            }
            second = self.run_summarizer(batch)
            self.assertEqual(second.returncode, 0, second.stderr)
            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(report.iterdir())
                if path.is_file()
            }
            self.assertEqual(after, before)


class RunbookContractTest(unittest.TestCase):
    def test_runbook_contains_frozen_contract_and_all_modes(self):
        text = RUNBOOK.read_text()
        for needle in (
            "fanout 4/2/1",
            "cap=6",
            "Plain",
            "HashSlot-1",
            "DistanceFirst-Q3",
            "SoftTier-Q3",
            "HardTier-Q3",
            "priority alpha=1.02",
            "rabitq1-adaptive-exact",
            "L=10,16,25,32,50,100,200",
            "单次运行估计",
            "不自动重试",
            "约 30 秒",
            "Plain 不作为门控",
            "plain_uplift.tsv",
        ):
            self.assertIn(needle, text)
        for mode in (
            "prepare",
            "build-sidecar",
            "preflight",
            "smoke",
            "formal",
            "summarize",
        ):
            self.assertIn(
                f"run_laion35m_hashprune_cap3_fanout1041.sh {mode}",
                text,
            )

    def test_runbook_requires_distinct_smoke_and_formal_ids_and_command_gate(self):
        text = RUNBOOK.read_text()
        self.assertIn("smoke 与 formal 必须使用不同的 `RUN_ID`", text)
        self.assertIn("展示唯一的 formal 命令并再次确认", text)
        self.assertIn("不得把 8/24/40 当作实测候选数增幅", text)
        self.assertIn("平均出度差不超过 0.25", text)
        self.assertIn("图字节差不超过 1%", text)


if __name__ == "__main__":
    unittest.main()
