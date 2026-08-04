#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "script/laion100m_shear_memory_scaling.sh"
RUNBOOK = ROOT / "docs/runbooks/laion100m_shear_memory_scaling.md"
EXPECTED_BASE_BYTES = 204_800_000_008
PAYLOAD_BYTES = 204_800_000_000
SIDECAR_BYTES = 52_000_000_000
CAPS = {
    "p50": 102_400_000_000,
    "p40": 81_920_000_000,
    "p30": 61_440_000_000,
    "p20": 40_960_000_000,
}


def function_body(script: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n", script, re.MULTILINE)
    assert match, f"missing function: {name}"
    next_match = re.search(
        r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{\n",
        script[match.end() :],
        re.MULTILINE,
    )
    stop = match.end() + next_match.start() if next_match else len(script)
    return script[match.start() : stop]


def heredoc_body(script: str, tag: str) -> str:
    match = re.search(
        rf"<<'{re.escape(tag)}'\n(?P<body>.*?)\n{re.escape(tag)}(?:\n|$)",
        script,
        re.DOTALL,
    )
    assert match, f"missing heredoc: {tag}"
    return match.group("body") + "\n"


def verify_static_contract(source: str) -> None:
    for label, value in CAPS.items():
        assert f"[{label}]={value}" in source
    assert "CAP_ORDER=(p50 p40 p30 p20)" in source
    assert f"readonly EXPECTED_BASE_BYTES={EXPECTED_BASE_BYTES}" in source
    assert f"readonly PAYLOAD_BYTES={PAYLOAD_BYTES}" in source
    assert f"readonly SIDECAR_BYTES={SIDECAR_BYTES}" in source
    assert "/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin" in source
    for mode in ("help", "preflight", "self-test", "formal", "summarize"):
        assert f"{mode})" in source, mode

    launcher = function_body(source, "launch_cgroup_service")
    for fragment in (
        '--property="MemoryMax=$cap_bytes"',
        '--property="MemorySwapMax=0"',
        '--property="OOMPolicy=stop"',
        'sample_cgroup "$unit" "$run_root" "$cap_bytes"',
    ):
        assert fragment in launcher, fragment

    sampler = function_body(source, "sample_cgroup")
    for fragment in (
        "memory.current",
        "memory.peak",
        "memory.stat",
        "memory.events",
        "io.stat",
        "memory.max",
        "memory.swap.max",
        "file_dirty",
        "file_writeback",
        "event_oom_kill",
        "rbytes",
        "wbytes",
        "rios",
        "wios",
    ):
        assert fragment in sampler, fragment
    assert 'if ! current="$(cat "$cgroup_dir/memory.current" 2>/dev/null)"' in sampler
    assert 'if ! io_totals="$(io_stat_totals "$cgroup_dir/io.stat" 2>/dev/null)"' in sampler

    preflight = function_body(source, "run_preflight")
    assert "/sys/fs/cgroup/memory.swap.max" not in preflight
    assert 'current_cgroup_path="$(current_cgroup_dir)"' in preflight
    assert '[[ -f "$current_cgroup_path/memory.swap.max" ]]' in preflight
    assert "verify_binary_cli_contract" in preflight
    cli_probe = function_body(source, "verify_binary_cli_contract")
    for fragment in (
        "--gpu-subtree-sq-staging-mode gpu-d0",
        "--gpu-subtree-sq-wmma-sidecar-mode rebuild",
        "--gpu-subtree-sq8-d0-assignment-kernel wmma-batch",
        "--hashprune-backfill-mode rhalf-soft-tier-q3",
        "--hashprune-rhalf-max-backfill-per-row 3",
        "--hashprune-soft-tier-priority-alpha 1.02",
        "--help",
    ):
        assert fragment in cli_probe

    wrapper = heredoc_body(source, "SHEAR_WRAPPER")
    for fragment in (
        "--oom-enable",
        "--oom-memory-budget-gb 0",
        "--oom-io-mode direct",
        "--max-degree 64",
        "--lbuild 100",
        "--num-threads 8",
        "--leaf-size-min 256",
        "--leaf-size-max 1280",
        "--leaf-knn 2",
        "--psamp-fraction 0.05",
        "--fanout-top 8",
        "--fanout-second 3",
        "--root-risk-candidate-rs 1,2,4,8",
        "--seed 42",
        "--hashprune-backfill-mode rhalf-soft-tier-q3",
        "--hashprune-rhalf-max-backfill-per-row 3",
        "--hashprune-soft-tier-priority-alpha 1.02",
        "--hashprune-backfill-alpha 1.0",
        "--gpu-subtree-root-assignment-backend cpu",
        "--gpu-subtree-d0-partition-mode gpu-kmeans-coarse",
        "--gpu-subtree-sq-staging-mode gpu-d0",
        "--gpu-subtree-sq-wmma-sidecar-mode rebuild",
        "--gpu-subtree-persistent-large-ticket-batch 8",
        "--gpu-subtree-sq8-assignment-kernel wmma",
        "--gpu-subtree-sq8-d0-assignment-kernel wmma-batch",
        "--pipnn-gpu-workspace-budget-gb 16",
    ):
        assert fragment in wrapper, fragment

    forbidden = (
        "search_shear_index",
        "run_search",
        "QUERY=",
        "GT=",
        "Recall",
        "QPS",
    )
    for fragment in forbidden:
        assert fragment not in source, fragment
        assert fragment not in wrapper, fragment

    formal = function_body(source, "run_formal")
    calls = [f'run_cap "{label}"' for label in CAPS]
    assert all(call in formal for call in calls)
    assert [formal.index(call) for call in calls] == sorted(
        formal.index(call) for call in calls
    )
    assert "drop_page_cache" not in formal
    run_cap = function_body(source, "run_cap")
    assert "drop_page_cache" in run_cap
    assert "( set -e; postprocess_completed_cap" in run_cap

    profile = function_body(source, "extract_shear_metrics")
    assert "candidate_reread_bytes" in profile
    assert "None" in profile
    assert "sq_wmma_sidecar_resident_rows" not in profile
    assert "sidecar_file_rows * 520" in profile

    self_test = function_body(source, "run_self_test")
    assert "time.sleep(2)" in self_test
    assert "time.sleep(1)" in self_test


def verify_service_classifier_uses_main_process_status(source: str) -> None:
    classifier = heredoc_body(source, "PY_CLASSIFY")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "cgroup_samples.tsv").write_text(
            "event_oom\tevent_oom_kill\n0\t0\n",
            encoding="utf-8",
        )
        (root / "cgroup_journal.log").write_text("", encoding="utf-8")
        (root / "systemd_show.txt").write_text(
            "Result=success\n"
            "ExecMainCode=0\n"
            "ExecMainStatus=0\n"
            "ActiveState=inactive\n"
            "SubState=dead\n",
            encoding="utf-8",
        )
        (root / "systemd_run.stdout.log").write_text(
            "Finished with result: success\n"
            "Main processes terminated with: code=killed/status=TERM\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["python3", "-", str(root), "0"],
            input=classifier,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        service = json.loads((root / "service_result.json").read_text())
        assert service["state"] == "failed"
        assert service["launcher_exit_code"] == 0
        assert service["exit_code"] == 143
        assert service["systemd_result"] == "signal"
        assert service["exec_main_code"] == "killed"
        assert service["exec_main_status"] == 15
        assert (root / "service_exit_code.txt").read_text().strip() == "143"

        success = root / "success"
        success.mkdir()
        (success / "cgroup_samples.tsv").write_text(
            "event_oom\tevent_oom_kill\n0\t0\n",
            encoding="utf-8",
        )
        (success / "cgroup_journal.log").write_text("", encoding="utf-8")
        (success / "systemd_show.txt").write_text(
            "Result=success\n"
            "ExecMainCode=0\n"
            "ExecMainStatus=0\n",
            encoding="utf-8",
        )
        (success / "systemd_run.stderr.log").write_text(
            "Finished with result: success\n"
            "Main processes terminated with: code=exited/status=0\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["python3", "-", str(success), "0"],
            input=classifier,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        service = json.loads((success / "service_result.json").read_text())
        assert service["state"] == "completed"
        assert service["exit_code"] == 0


def verify_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def verify_help() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "does not run search" in result.stdout.lower()
    for mode in ("preflight", "self-test", "formal", "summarize"):
        assert mode in result.stdout


def verify_embedded_wrapper_syntax(source: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        wrapper = Path(directory) / "shear_wrapper.sh"
        wrapper.write_text(heredoc_body(source, "SHEAR_WRAPPER"), encoding="utf-8")
        result = subprocess.run(
            ["bash", "-n", str(wrapper)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def write_synthetic_cap(root: Path, label: str, state: str) -> None:
    cap_root = root / label
    payload = cap_root / "payload"
    payload.mkdir(parents=True)
    exit_code = 0 if state in {"completed", "failed_validation"} else 137 if state == "oom" else 1
    service_state = "completed" if state in {"completed", "failed_validation"} else state.removeprefix("failed_")
    (cap_root / "service_result.json").write_text(
        json.dumps(
            {
                "schema": "laion100m_shear_cgroup_service_v1",
                "state": service_state,
                "exit_code": exit_code,
                "event_oom": 2 if state == "oom" else 0,
                "event_oom_kill": 1 if state == "oom" else 0,
                "journal_oom": state == "oom",
                "sample_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (cap_root / "status.json").write_text(
        json.dumps(
            {
                "schema": "laion100m_shear_memory_scaling_status_v1",
                "state": state,
                "exit_code": exit_code,
                "history": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (cap_root / "cgroup_samples.tsv").write_text(
        "epoch\tmemory_current\tmemory_peak\tanon\tfile\tfile_dirty\tfile_writeback\t"
        "event_high\tevent_max\tevent_oom\tevent_oom_kill\trbytes\twbytes\trios\twios\n"
        "1\t10\t20\t3\t4\t1\t2\t0\t0\t0\t0\t100\t200\t3\t4\n"
        f"2\t15\t25\t5\t6\t2\t3\t0\t1\t{2 if state == 'oom' else 0}\t"
        f"{1 if state == 'oom' else 0}\t1300\t2500\t17\t29\n",
        encoding="utf-8",
    )
    (payload / "time.log").write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n"
        "Maximum resident set size (kbytes): 12345\n",
        encoding="utf-8",
    )
    (payload / "resource_summary.json").write_text(
        json.dumps(
            {
                "peak_process_tree_rss_bytes": 12_641_280,
                "peak_hbm_bytes": 9_437_184,
                "peak_temp_allocated_bytes": 8_388_608,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if state != "completed":
        return

    (payload / "profile.json").write_text(
        json.dumps(
            {
                "memory_budget_source": "auto-host-cgroup",
                "memory_budget_bytes": CAPS[label] * 8 // 10,
                "resident_reservoir": {"points": 10, "bytes": 4096},
                "memory_plan": {
                    "sidecar_resident_kind": "sq_wmma",
                    "sidecar_resident_admitted_rows": 80_000_000,
                    "sidecar_resident_budget_bytes": 41_600_000_000,
                },
                "gpu_subtree": {
                    "sq_wmma_sidecar_resident_bytes": 41_600_000_000,
                    "sq_wmma_sidecar_file_rows": 20_000_000,
                    "sq_wmma_sidecar_file_gather_ms": 1234,
                },
                "gpu_subtree_io": {
                    "candidate_spill_bytes": 987_654,
                    "candidate_spill_files": 4,
                    "reduce_segments_scanned": 8,
                    "reduce_part_files_opened": 4,
                },
                "serial_timing": {"rows": {"external_reduce_ms": 3456}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (payload / "shear_metrics.json").write_text(
        json.dumps(
            {
                "actual_bh_bytes": CAPS[label] * 8 // 10,
                "reservoir_resident_bytes": 4096,
                "sidecar_theoretical_bytes": SIDECAR_BYTES,
                "sidecar_resident_bytes": 41_600_000_000,
                "sidecar_resident_fraction": 0.8,
                "sidecar_admitted_rows": 80_000_000,
                "sidecar_file_rows": 20_000_000,
                "sidecar_file_logical_bytes": 10_400_000_000,
                "sidecar_file_gather_ms": 1234,
                "candidate_spill_bytes": 987_654,
                "candidate_spill_files": 4,
                "candidate_reread_bytes": None,
                "reduce_segments_scanned": 8,
                "reduce_part_files_opened": 4,
                "external_reduce_ms": 3456,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (payload / "graph_validation.json").write_text(
        json.dumps(
            {
                "valid": True,
                "nodes": 100_000_000,
                "edges": 4_200_000_000,
                "observed_min_degree": 32,
                "mean_degree": 42.0,
                "observed_max_degree": 64,
                "invalid_edges": 0,
                "self_edges": 0,
                "duplicate_edges": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (payload / "graph_quality.tsv").write_text(
        "rows\tedges\tavg_out_degree\tmax_degree\n"
        "100000000\t4200000000\t42.0\t64\n",
        encoding="utf-8",
    )
    (payload / "index_size_before_cleanup.json").write_text(
        json.dumps({"bytes": 17_200_000_000}) + "\n",
        encoding="utf-8",
    )
    (payload / "cleanup_summary.txt").write_text(
        "verified=true\n",
        encoding="utf-8",
    )


def verify_synthetic_summary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for label, state in zip(
            CAPS,
            ("completed", "oom", "failed_validation", "failed_runtime"),
            strict=True,
        ):
            write_synthetic_cap(root, label, state)
        result = subprocess.run(
            ["bash", str(RUNNER), "summarize", str(root)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        with (root / "summary.tsv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert [row["cap_label"] for row in rows] == list(CAPS)
        assert [row["status"] for row in rows] == [
            "completed",
            "oom",
            "failed_validation",
            "failed_runtime",
        ]
        row = rows[0]
        assert row["cgroup_cap_bytes"] == str(CAPS["p50"])
        assert row["actual_bh_bytes"] == str(CAPS["p50"] * 8 // 10)
        assert row["sidecar_theoretical_bytes"] == str(SIDECAR_BYTES)
        assert row["sidecar_resident_bytes"] == "41600000000"
        assert row["sidecar_resident_fraction"] == "0.800000"
        assert row["sidecar_file_rows"] == "20000000"
        assert row["sidecar_file_logical_bytes"] == "10400000000"
        assert row["candidate_spill_bytes"] == "987654"
        assert row["candidate_reread_bytes"] == "N/A"
        assert row["external_reduce_ms"] == "3456"
        assert row["cgroup_physical_read_bytes"] == "1200"
        assert row["cgroup_physical_write_bytes"] == "2300"
        assert row["cgroup_read_ios"] == "14"
        assert row["cgroup_write_ios"] == "25"
        assert row["degree_min"] == "32"
        assert row["degree_mean"] == "42.000000"
        assert row["degree_max"] == "64"
        assert row["cleanup_verified"] == "true"
        markdown = (root / "summary.md").read_text(encoding="utf-8")
        assert "file_rows × 520 B" in markdown
        assert "not physical NVMe traffic" in markdown
        assert "not independently exposed" in markdown
        assert "No search" in markdown


def verify_runbook() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for fragment in (
        "204,800,000,000",
        "52,000,000,000",
        "102,400,000,000",
        "81,920,000,000",
        "61,440,000,000",
        "40,960,000,000",
        "SoftTier-Q3",
        "cap=3",
        "alpha=1.02",
        "formal",
        "summarize",
        "No search",
        "explicit approval",
    ):
        assert fragment in text, fragment
    formal_command = "bash script/laion100m_shear_memory_scaling.sh formal"
    assert text.count(formal_command) == 1


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    verify_static_contract(source)
    verify_service_classifier_uses_main_process_status(source)
    verify_shell_syntax()
    verify_help()
    verify_embedded_wrapper_syntax(source)
    verify_synthetic_summary()
    verify_runbook()
    print("laion100m SHEAR memory-scaling contract tests passed")


if __name__ == "__main__":
    main()
