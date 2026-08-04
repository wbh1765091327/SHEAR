import json
import struct
import tempfile
import unittest
from pathlib import Path

from script.run_laion35m_ace_disk_motivation import (
    FORMAL_CONTRACT,
    parse_ace_log,
    read_fbin_shape,
    summarize_pair,
)


class FrozenContractTest(unittest.TestCase):
    def test_formal_contract_is_the_approved_laion35m_matrix(self):
        self.assertEqual(FORMAL_CONTRACT["rows"], 35_000_000)
        self.assertEqual(FORMAL_CONTRACT["dim"], 512)
        self.assertEqual(FORMAL_CONTRACT["graph_degree"], 64)
        self.assertEqual(FORMAL_CONTRACT["intermediate_graph_degree"], 128)
        self.assertEqual(FORMAL_CONTRACT["ef_construction"], 176)
        self.assertEqual(FORMAL_CONTRACT["npartitions"], 40)
        self.assertEqual(FORMAL_CONTRACT["local_builder"], "cagra")
        self.assertEqual(FORMAL_CONTRACT["cagra_local_solver"], "nn_descent")
        self.assertEqual(FORMAL_CONTRACT["execution_mode"], "pipelined")

    def test_shell_runner_contains_required_safety_and_measurement_gates(self):
        runner = (
            Path(__file__).with_name("run_laion35m_ace_disk_motivation.sh").read_text()
        )
        required = (
            "/data_staging/wbh/laion-512-35M/base.fbin",
            "cmp --ignore-initial=8:8 --bytes=71680000000",
            "swapon --show",
            "--query-compute-apps=pid",
            "nsys profile",
            "--trace=cuda,nvtx",
            "OMP_NUM_THREADS=8",
            "OPENBLAS_NUM_THREADS=1",
            "ace_non_disk",
            "ace_disk",
            "generated_artifacts.txt",
            "assignment_create",
            "assignment_replay",
            "partition_assignment.bin",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, runner)


class InputValidationTest(unittest.TestCase):
    def test_fbin_header_and_length_are_both_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.fbin"
            path.write_bytes(struct.pack("<ii", 2, 3) + b"\0" * (2 * 3 * 4))
            self.assertEqual(read_fbin_shape(path), (2, 3))
            path.write_bytes(struct.pack("<ii", 2, 3) + b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "byte length"):
                read_fbin_shape(path)


class PairAcceptanceTest(unittest.TestCase):
    NON_DISK_LOG = """
ACE: Graph fits in host and GPU memory. Using in-memory mode.
ACE motivation: partition_assignment_digest_v1=0123456789abcdef0123456789abcdef partitions=2 partition_assignment_digest_ms=1.0
ACE motivation: partition_assignment_mode=create partition_assignment_io_ms=1.0 partition_stage_wall_ms=11
ACE motivation: deterministic partition mapping enabled for frozen assignment
ACE pipeline: partition    0 (     100 +       20), read_active_ms=1.0 local_build_active_ms=2.0 local_validate_ms=0.1 adjust_ms=0.2 write_ms=0.0
ACE pipeline: partition    1 (      90 +       30), read_active_ms=1.1 local_build_active_ms=2.1 local_validate_ms=0.1 adjust_ms=0.2 write_ms=0.0
ACE pipeline: workspace_init_ms=0.1 read_active_ms=2.1 local_build_active_ms=4.1 local_validate_ms=0.2 adjust_ms=0.4 write_ms=0.0 pipeline_wall_ms=5.0
ACE: Partition labeling completed in 10 ms (min_partition_size: 1000)
ACE: Vector list creation completed in 2 ms
ACE: All partition processing completed in 6 ms (2 partitions)
ACE: Final index creation completed in 3 ms
ACE: Partitioned CAGRA build completed in 25 ms total
"""
    DISK_LOG = NON_DISK_LOG.replace(
        "ACE: Graph fits in host and GPU memory. Using in-memory mode.",
        "ACE: Graph fits in host and GPU memory but disk mode is forced. Using disk-mode with temporary storage /tmp/x",
    ).replace("write_ms=0.0", "write_ms=0.5").replace(
        "partition_assignment_mode=create partition_assignment_io_ms=1.0",
        "partition_assignment_mode=replay partition_assignment_io_ms=2.0",
    )

    def test_log_parser_extracts_mode_digest_counts_and_outer_timers(self):
        parsed = parse_ace_log(self.NON_DISK_LOG)
        self.assertEqual(parsed["actual_mode"], "in_memory")
        self.assertEqual(
            parsed["partition_assignment_digest"],
            "0123456789abcdef0123456789abcdef",
        )
        self.assertEqual(parsed["partition_rows"], [[0, 100, 20], [1, 90, 30]])
        self.assertEqual(parsed["outer_ms"]["partition_labeling"], 10.0)
        self.assertEqual(parsed["outer_ms"]["pipeline_wall"], 5.0)
        self.assertEqual(parsed["outer_ms"]["total"], 25.0)

    def test_pair_rejects_digest_or_partition_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root, "ace_non_disk", self.NON_DISK_LOG, False)
            bad_disk = self.DISK_LOG.replace(
                "0123456789abcdef0123456789abcdef",
                "ffffffffffffffffffffffffffffffff",
            )
            self._write_case(root, "ace_disk", bad_disk, True)
            with self.assertRaisesRegex(ValueError, "digest"):
                summarize_pair(root)

    def test_pair_accepts_matched_storage_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_case(root, "ace_non_disk", self.NON_DISK_LOG, False)
            self._write_case(root, "ace_disk", self.DISK_LOG, True)
            summary = summarize_pair(root)
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["cases"]["ace_non_disk"]["actual_mode"], "in_memory")
        self.assertEqual(summary["cases"]["ace_disk"]["actual_mode"], "forced_disk")

    @staticmethod
    def _write_case(root: Path, name: str, log: str, use_disk: bool) -> None:
        case = root / name
        case.mkdir()
        (case / "stderr.log").write_text(log)
        (case / "driver_profile.json").write_text(
            json.dumps(
                {
                    "rows": 35_000_000,
                    "dim": 512,
                    "graph_degree": 64,
                    "use_disk": use_disk,
                    "ace_build_wall_ms": 25.0,
                }
            )
        )
        (case / "timeline.json").write_text(
            json.dumps(
                {
                    "aggregate": {
                        "partition_count": 2,
                        "local_build_wall_ms": 4.0,
                        "critical_path_closure_error_ms": 0.0,
                    },
                    "partitions": [],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
