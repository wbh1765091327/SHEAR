#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/runbooks/hnsw_flash_four_dataset_build_search.md"
PATCH = ROOT / "script/patches/hnsw_flash_four_dataset_benchmark.patch"
SOURCE = Path("/data_staging/wbh/wiki35m_baseline_repos_20260630/HNSW-Flash")
SOURCE_COMMIT = "716e002596c9002810c6000072100a7ed0c64e33"
BEGIN = "<!-- BEGIN RUNBOOK_SCRIPT -->"
END = "<!-- END RUNBOOK_SCRIPT -->"

DATASETS = (
    ("wiki35m", 35_000_000, 1024, "wiki-1024-35M/en_base.fbin"),
    ("laion100m", 100_000_000, 512, "laion-512-100M/base.fbin"),
    ("msmarco70m", 70_000_000, 768, "msmarco-768-70M/base.fbin"),
    ("deep100m", 100_000_000, 96, "deep-96-100M/base.100M.fbin"),
)
SEARCH_EFS = (25, 50, 100, 200, 400)


def extract_script(text: str) -> str:
    start = text.index(BEGIN) + len(BEGIN)
    stop = text.index(END, start)
    match = re.search(r"```bash\n(?P<body>.*?)\n```", text[start:stop], re.S)
    assert match, "missing fenced Bash payload"
    return match.group("body") + "\n"


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def materialize_snapshot(destination: Path) -> Path:
    if destination.exists():
        raise AssertionError(f"destination already exists: {destination}")
    destination.mkdir(parents=True)
    archive = destination.parent / f"{destination.name}.tar"
    with archive.open("wb") as stream:
        result = subprocess.run(
            ["git", "-C", str(SOURCE), "archive", "--format=tar", SOURCE_COMMIT],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        raise AssertionError(result.stderr.decode())
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    archive.unlink()
    applied = run(["patch", "-p1", "-i", str(PATCH)], cwd=destination)
    if applied.returncode:
        raise AssertionError(applied.stdout + applied.stderr)
    return destination


def apply_patch_to_snapshot(root: Path) -> Path:
    return materialize_snapshot(root / "source")


class RunbookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNBOOK.read_text()
        cls.script = extract_script(cls.source)

    def test_fixed_contract(self) -> None:
        self.assertIn(
            "FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)",
            self.script,
        )
        self.assertIn("SEARCH_EFS=(25 50 100 200 400)", self.script)
        self.assertIn("BUILD_THREADS=64", self.script)
        self.assertIn("SEARCH_THREADS=8", self.script)
        self.assertIn("SEED=42", self.script)
        self.assertIn("M=32", self.script)
        self.assertIn("EF_CONSTRUCTION=1024", self.script)
        self.assertIn("SEARCH_REPETITIONS=2", self.script)
        self.assertIn("SEARCH_WARMUPS=1", self.script)

    def test_dataset_paths_and_shapes(self) -> None:
        for dataset, rows, dim, suffix in DATASETS:
            match = re.search(rf"^[ \t]*{dataset}\).*?;;", self.script, re.S | re.M)
            self.assertIsNotNone(match, dataset)
            branch = match.group(0)
            self.assertIn(f"ROWS={rows}", branch)
            self.assertIn(f"DIM={dim}", branch)
            self.assertIn(suffix, branch)

    def test_timing_schema(self) -> None:
        for field in (
            "base_load_seconds",
            "pca_fit_seconds",
            "pca_encode_base_seconds",
            "pq_train_seconds",
            "hnsw_insert_seconds",
            "artifact_serialize_seconds",
            "build_internal_e2e_seconds",
            "build_external_wall_seconds",
            "build_prewarm_seconds",
            "peak_process_tree_rss_bytes",
            "physical_read_bytes",
            "physical_write_bytes",
        ):
            self.assertIn(field, self.script)

    def test_modes_and_safe_cleanup(self) -> None:
        for mode in ("help", "prepare", "preflight", "smoke", "formal", "summarize"):
            self.assertIn(f"{mode})", self.script)
        cleanup = re.search(
            r"^cleanup_accepted_index\(\) \{.*?^\}",
            self.script,
            re.S | re.M,
        )
        self.assertIsNotNone(cleanup)
        body = cleanup.group(0)
        self.assertIn('[[ "$resolved" == "$artifacts"/* ]]', body)
        self.assertIn('rm -f -- "$resolved"', body)
        self.assertNotIn("rm -rf", body)

    def test_shell_syntax(self) -> None:
        result = run(["bash", "-n"], input_text=self.script)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_patch_applies_to_frozen_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = apply_patch_to_snapshot(Path(directory))
            self.assertTrue((snapshot / "benchmark_main.cc").is_file())
            self.assertTrue((snapshot / "include/benchmark/flat_binary.h").is_file())
            self.assertTrue((snapshot / "include/benchmark/flash_benchmark.h").is_file())
            cmake = (snapshot / "CMakeLists.txt").read_text()
            self.assertIn("hnsw_flash_benchmark", cmake)


def audit_batch(path: Path) -> None:
    required = (
        "status.json",
        "formal_schedule.tsv",
        "build_summary.tsv",
        "search_summary.tsv",
        "summary.md",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise AssertionError(f"missing batch files: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--apply-check", action="store_true")
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--audit-batch", type=Path)
    args, unittest_args = parser.parse_known_args()

    if args.extract:
        print(extract_script(RUNBOOK.read_text()), end="")
        return 0
    if args.apply_check:
        with tempfile.TemporaryDirectory() as directory:
            apply_patch_to_snapshot(Path(directory))
        print("patch application: PASS")
        return 0
    if args.materialize is not None:
        materialize_snapshot(args.materialize)
        print(args.materialize)
        return 0
    if args.audit_batch is not None:
        audit_batch(args.audit_batch)
        print("batch audit: PASS")
        return 0

    program = unittest.main(argv=[__file__, *unittest_args], exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
