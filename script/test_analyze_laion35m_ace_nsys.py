import sqlite3
import tempfile
import unittest
from pathlib import Path

from script.analyze_laion35m_ace_nsys import (
    analyze_sqlite,
    duration_ns,
    intersect_intervals,
    normalize_intervals,
    subtract_intervals,
)


class IntervalArithmeticTest(unittest.TestCase):
    def test_normalize_merges_overlapping_and_touching_intervals(self):
        self.assertEqual(
            normalize_intervals([(10, 20), (5, 8), (8, 12), (30, 30)]),
            [(5, 20)],
        )

    def test_intersection_and_subtraction(self):
        left = [(0, 20), (30, 50)]
        right = [(10, 40)]
        self.assertEqual(intersect_intervals(left, right), [(10, 20), (30, 40)])
        self.assertEqual(subtract_intervals(left, right), [(0, 10), (40, 50)])
        self.assertEqual(duration_ns(left), 40)


class SyntheticTraceTest(unittest.TestCase):
    def test_analyzer_reports_raw_overlap_and_exclusive_critical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(trace)
            connection.executescript(
                """
                CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE NVTX_EVENTS(
                    start INTEGER NOT NULL,
                    end INTEGER,
                    text TEXT,
                    textId INTEGER
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL
                );
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL
                );
                """
            )
            events = [
                (0, 100, "SHEAR_MOTIVATION/ACE_LOCAL_BUILD partition=0", None),
                (5, 80, "SHEAR_MOTIVATION/GNND_LOCAL_BUILD rows=10 dim=4", None),
                (20, 70, "SHEAR_MOTIVATION/CPU_GRAPH_UPDATE_SAMPLE", None),
                (0, 20, "SHEAR_MOTIVATION/COORD_H2D", None),
                (65, 80, "SHEAR_MOTIVATION/COORD_THREAD_JOIN", None),
                (80, 95, "SHEAR_MOTIVATION/CAGRA_POST_GNND_OPTIMIZE", None),
            ]
            connection.executemany(
                "INSERT INTO NVTX_EVENTS(start, end, text, textId) VALUES (?, ?, ?, ?)",
                events,
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, end) VALUES (?, ?)",
                [(10, 40), (30, 50)],
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY(start, end) VALUES (50, 60)"
            )
            connection.commit()
            connection.close()

            report = analyze_sqlite(trace)

        aggregate = report["aggregate"]
        self.assertEqual(aggregate["partition_count"], 1)
        self.assertAlmostEqual(aggregate["local_build_wall_ms"], 0.0001)
        self.assertAlmostEqual(aggregate["gpu_compute_active_ms"], 0.00004)
        self.assertAlmostEqual(aggregate["gpu_transfer_active_ms"], 0.00001)
        self.assertAlmostEqual(aggregate["cpu_graph_active_ms"], 0.00005)
        self.assertAlmostEqual(aggregate["coordination_active_ms"], 0.000035)
        self.assertAlmostEqual(aggregate["gpu_cpu_graph_overlap_ms"], 0.00003)
        self.assertAlmostEqual(aggregate["gpu_coordination_overlap_ms"], 0.00001)
        self.assertAlmostEqual(aggregate["cpu_graph_coordination_overlap_ms"], 0.000005)

        exclusive = aggregate["critical_path_exclusive_ms"]
        self.assertAlmostEqual(exclusive["gpu_compute"], 0.00004)
        self.assertAlmostEqual(exclusive["gpu_transfer"], 0.00001)
        self.assertAlmostEqual(exclusive["cpu_graph"], 0.00001)
        self.assertAlmostEqual(exclusive["coordination"], 0.00002)
        self.assertAlmostEqual(exclusive["other"], 0.00002)
        self.assertAlmostEqual(sum(exclusive.values()), aggregate["local_build_wall_ms"])
        self.assertAlmostEqual(aggregate["gnnd_wall_ms"], 0.000075)
        self.assertAlmostEqual(aggregate["post_gnnd_optimize_wall_ms"], 0.000015)
        self.assertAlmostEqual(
            report["marker_active_ms"][
                "SHEAR_MOTIVATION/CPU_GRAPH_UPDATE_SAMPLE"
            ],
            0.00005,
        )
        self.assertAlmostEqual(
            report["marker_active_ms"]["SHEAR_MOTIVATION/COORD_THREAD_JOIN"],
            0.000015,
        )


if __name__ == "__main__":
    unittest.main()
