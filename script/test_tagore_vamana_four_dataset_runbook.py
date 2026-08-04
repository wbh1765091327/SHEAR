#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/runbooks/tagore_vamana_four_dataset.md"
BEGIN = "<!-- BEGIN RUNBOOK_SCRIPT -->"
END = "<!-- END RUNBOOK_SCRIPT -->"
DATASETS = {
    "wiki35m": (35_000_000, 1024, 1_200_000, "wiki-1024-35M/en_base.fbin"),
    "laion100m": (100_000_000, 512, 3_000_000, "laion-512-100M/base.fbin"),
    "msmarco70m": (70_000_000, 768, 2_500_000, "msmarco-768-70M/base.fbin"),
    "deep100m": (100_000_000, 96, 5_000_000, "deep-96-100M/base.100M.fbin"),
}


def extract_script(source: str) -> str:
    start = source.index(BEGIN) + len(BEGIN)
    stop = source.index(END, start)
    match = re.search(r"```bash\n(?P<body>.*?)\n```", source[start:stop], re.S)
    assert match, "missing fenced Bash payload"
    return match.group("body") + "\n"


def function_body(script: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n", script, re.M)
    assert match, f"missing function: {name}"
    following = re.search(
        r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{\n", script[match.end() :], re.M
    )
    stop = match.end() + following.start() if following else len(script)
    return script[match.start() : stop]


def embedded_python(script: str) -> dict[str, str]:
    pattern = re.compile(
        r"<<'(?P<tag>PY_[A-Z0-9_]+)'\n(?P<body>.*?)\n(?P=tag)(?:\n|$)", re.S
    )
    matches = list(pattern.finditer(script))
    programs = {match.group("tag"): match.group("body") + "\n" for match in matches}
    assert len(programs) == len(matches), "duplicate Python heredoc tag"
    return programs


def script_library(script: str) -> str:
    terminal = script.rfind('\ncase "$MODE" in\n')
    assert terminal >= 0, "missing terminal mode dispatch"
    return script[: terminal + 1]


def run(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
        env=env,
    )


def verify_static_contract(source: str, script: str) -> None:
    assert "independent" in source.lower()
    assert "FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)" in script
    assert "TAGORE_K=96" in script
    assert "TAGORE_FINAL_DEGREE=64" in script
    assert "TAGORE_BUFFER_SIZE=16" in script
    assert "PARTITION_REPLICATION=2" in script
    for mode in (
        "help",
        "prepare",
        "preflight",
        "smoke",
        "discover",
        "formal",
        "summarize",
    ):
        assert f"{mode})" in script, mode
    assert "wiki35m|laion100m|msmarco70m|deep100m)" in script

    loader = function_body(script, "load_dataset")
    for dataset, (rows, dim, max_points, base_suffix) in DATASETS.items():
        branch = re.search(rf"{dataset}\).*?;;", loader, re.S)
        assert branch, dataset
        body = branch.group(0)
        assert f"ROWS={rows}" in body
        assert f"DIM={dim}" in body
        assert f"MAX_POINTS={max_points}" in body
        assert base_suffix in body


def verify_syntax(script: str) -> None:
    result = run(["bash", "-n"], input_text=script)
    assert result.returncode == 0, result.stderr
    for tag, program in embedded_python(script).items():
        result = run(
            [
                "python3",
                "-c",
                "import sys; compile(sys.stdin.read(), sys.argv[1], 'exec')",
                tag,
            ],
            input_text=program,
        )
        assert result.returncode == 0, f"{tag}: {result.stderr}"


def write_sparse_bin(path: Path, rows: int, dim: int, element_bytes: int = 4) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", rows, dim))
        stream.truncate(8 + rows * dim * element_bytes)


def write_partition_fixture(
    root: Path,
    *,
    counts: list[int],
    dim: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / "dataset"
    write_sparse_bin(Path(f"{prefix}_centroids.bin"), len(counts), dim)
    for cluster, count in enumerate(counts):
        write_sparse_bin(Path(f"{prefix}_subshard-{cluster}.bin"), count, dim)
        write_sparse_bin(
            Path(f"{prefix}_subshard-{cluster}_ids_uint32.bin"), count, 1
        )
    return prefix


def run_embedded_program(
    program: str,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "program.py"
        path.write_text(program, encoding="utf-8")
        return run(["python3", str(path), *arguments])


def verify_partition_policy_and_inspector(script: str) -> None:
    programs = embedded_python(script)
    assert "PY_PARTITION_POLICY" in programs
    policy = programs["PY_PARTITION_POLICY"]
    cases = (
        (
            [
                "initial",
                "--rows",
                "35000000",
                "--max-points",
                "1200000",
                "--buffer-size",
                "16",
            ],
            "64",
        ),
        (
            [
                "initial",
                "--rows",
                "100000000",
                "--max-points",
                "3000000",
                "--buffer-size",
                "16",
            ],
            "72",
        ),
        (
            [
                "initial",
                "--rows",
                "70000000",
                "--max-points",
                "2500000",
                "--buffer-size",
                "16",
            ],
            "56",
        ),
        (
            [
                "initial",
                "--rows",
                "100000000",
                "--max-points",
                "5000000",
                "--buffer-size",
                "16",
            ],
            "40",
        ),
        (
            [
                "next",
                "--current",
                "64",
                "--observed-max",
                "1800000",
                "--max-points",
                "1200000",
            ],
            "112",
        ),
    )
    for arguments, expected in cases:
        result = run_embedded_program(policy, arguments)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

    assert "PY_INSPECT_PARTITION" in programs
    inspector = programs["PY_INSPECT_PARTITION"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prefix = write_partition_fixture(root, counts=[6, 5, 5, 4], dim=3)
        output = root / "inspection.json"
        counts_path = root / "counts.tsv"
        arguments = [
            "--prefix",
            str(prefix),
            "--clusters",
            "4",
            "--rows",
            "10",
            "--dim",
            "3",
            "--max-points",
            "6",
            "--json",
            str(output),
            "--counts",
            str(counts_path),
        ]
        result = run_embedded_program(inspector, arguments)
        assert result.returncode == 0, result.stderr
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["fits_capacity"] is True
        assert value["min_shard_points"] == 4
        assert value["max_shard_points"] == 6
        assert value["total_shard_points"] == 20
        assert len(value["centroids_sha256"]) == 64
        assert counts_path.read_text(encoding="utf-8").splitlines()[0] == (
            "cluster\tcount\tdata_bytes\tid_bytes"
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prefix = write_partition_fixture(root, counts=[7, 5, 4, 4], dim=3)
        output = root / "inspection.json"
        counts_path = root / "counts.tsv"
        result = run_embedded_program(
            inspector,
            [
                "--prefix",
                str(prefix),
                "--clusters",
                "4",
                "--rows",
                "10",
                "--dim",
                "3",
                "--max-points",
                "6",
                "--json",
                str(output),
                "--counts",
                str(counts_path),
            ],
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text(encoding="utf-8"))["fits_capacity"] is False

    invalid_cases = (
        ("zero", [6, 5, 9, 0]),
        ("bad_total", [6, 5, 5, 5]),
    )
    for name, invalid_counts in invalid_cases:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = write_partition_fixture(root, counts=invalid_counts, dim=3)
            result = run_embedded_program(
                inspector,
                [
                    "--prefix",
                    str(prefix),
                    "--clusters",
                    "4",
                    "--rows",
                    "10",
                    "--dim",
                    "3",
                    "--max-points",
                    "9",
                    "--json",
                    str(root / "inspection.json"),
                    "--counts",
                    str(root / "counts.tsv"),
                ],
            )
            assert result.returncode != 0, name

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prefix = write_partition_fixture(root, counts=[6, 5, 5, 4], dim=3)
        bad_ids = Path(f"{prefix}_subshard-2_ids_uint32.bin")
        write_sparse_bin(bad_ids, 4, 1)
        result = run_embedded_program(
            inspector,
            [
                "--prefix",
                str(prefix),
                "--clusters",
                "4",
                "--rows",
                "10",
                "--dim",
                "3",
                "--max-points",
                "6",
                "--json",
                str(root / "inspection.json"),
                "--counts",
                str(root / "counts.tsv"),
            ],
        )
        assert result.returncode != 0
        assert "header mismatch" in result.stderr

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prefix = write_partition_fixture(root, counts=[6, 5, 5, 4], dim=3)
        bad_data = Path(f"{prefix}_subshard-1.bin")
        with bad_data.open("r+b") as stream:
            stream.truncate(bad_data.stat().st_size - 4)
        result = run_embedded_program(
            inspector,
            [
                "--prefix",
                str(prefix),
                "--clusters",
                "4",
                "--rows",
                "10",
                "--dim",
                "3",
                "--max-points",
                "6",
                "--json",
                str(root / "inspection.json"),
                "--counts",
                str(root / "counts.tsv"),
            ],
        )
        assert result.returncode != 0
        assert "payload mismatch" in result.stderr


def write_fake_partition_binary(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            from pathlib import Path
            import struct
            import sys

            _, kind, base, prefix_raw, sample_rate, clusters_raw, replication_raw = sys.argv
            if (kind, sample_rate, replication_raw) != ("float", "0.01", "2"):
                raise SystemExit(91)
            prefix = Path(prefix_raw)
            clusters = int(clusters_raw)
            counter_path = Path(os.environ["FAKE_PARTITION_COUNTER"])
            attempt = int(counter_path.read_text()) + 1 if counter_path.exists() else 1
            counter_path.write_text(str(attempt))
            if int(os.environ.get("FAKE_PARTITION_FAIL_AT", "0")) == attempt:
                raise SystemExit(7)
            rows = int(os.environ["FAKE_ROWS"])
            dim = int(os.environ["FAKE_DIM"])
            total = 2 * rows
            if attempt == 1:
                largest = 40
            elif attempt == 2:
                largest = 35
            else:
                largest = (total + clusters - 1) // clusters
            counts = [largest]
            remaining = total - largest
            for slot in range(clusters - 1):
                slots_left = clusters - 1 - slot
                count = remaining // slots_left
                counts.append(count)
                remaining -= count
            if min(counts) <= 0 or sum(counts) != total:
                raise SystemExit(92)

            def sparse(path, count, width):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    stream.write(struct.pack("<II", count, width))
                    stream.truncate(8 + count * width * 4)

            sparse(Path(f"{prefix}_centroids.bin"), clusters, dim)
            for cid, count in enumerate(counts):
                sparse(Path(f"{prefix}_subshard-{cid}.bin"), count, dim)
                sparse(Path(f"{prefix}_subshard-{cid}_ids_uint32.bin"), count, 1)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)


def write_fixture_source_runbook(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            # Fixture runbook

            <!-- BEGIN RUNBOOK_SCRIPT -->
            ```bash
            #!/usr/bin/env bash
            write_converter() {
              printf '#!/usr/bin/env python3\\n' >"$1"
              chmod 700 "$1"
            }
            case "$MODE" in
              help) : ;;
            esac
            ```
            <!-- END RUNBOOK_SCRIPT -->
            """
        ),
        encoding="utf-8",
    )


def discovery_fixture_shell(library: str) -> str:
    return (
        library
        + textwrap.dedent(
            """\
            load_dataset() {
              DATASET=fixture
              ROWS=100
              DIM=4
              MAX_POINTS=30
              BASE="$FIXTURE_ROOT/base.fbin"
              QUERY="$FIXTURE_ROOT/query.fbin"
              GT="$FIXTURE_ROOT/gt.ibin"
              SOURCE_RUNBOOK="$FIXTURE_ROOT/source_runbook.md"
            }
            initialize_batch
            discover_dataset fixture
            """
        )
    )


def verify_discovery_fixtures(script: str) -> None:
    library = script_library(script)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fake = root / "partition_data"
        write_fake_partition_binary(fake)
        for name in ("base.fbin", "query.fbin", "gt.ibin"):
            write_sparse_bin(root / name, 1, 4)
        write_fixture_source_runbook(root / "source_runbook.md")
        batch = root / "batch"
        counter = root / "counter.txt"
        env = os.environ.copy()
        env.update(
            {
                "BATCH_ROOT": str(batch),
                "PARTITION_BIN": str(fake),
                "CUVS_PYTHON": sys.executable,
                "FIXTURE_ROOT": str(root),
                "FAKE_PARTITION_COUNTER": str(counter),
                "FAKE_ROWS": "100",
                "FAKE_DIM": "4",
                "SESSION_ID": "fixture-session",
            }
        )
        result = run(["bash"], env=env, input_text=discovery_fixture_shell(library))
        assert result.returncode == 0, result.stderr
        assert counter.read_text(encoding="utf-8") == "3"
        assert (batch / "provenance/orchestrator_sources.sha256").is_file()

        attempts = sorted((batch / "fixture/discovery/attempts").iterdir())
        assert len(attempts) == 3
        records = [
            json.loads((attempt / "attempt.json").read_text(encoding="utf-8"))
            for attempt in attempts
        ]
        assert [record["status"] for record in records] == [
            "rejected_capacity",
            "rejected_capacity",
            "accepted",
        ]
        assert [record["clusters"] for record in records] == [16, 24, 32]
        assert not (attempts[0] / "payload").exists()
        assert not (attempts[1] / "payload").exists()
        assert (attempts[0] / "cleanup_manifest.tsv").is_file()
        assert (attempts[1] / "cleanup_manifest.tsv").is_file()

        frozen = batch / "fixture/frozen"
        manifest = json.loads(
            (frozen / "partition_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "accepted"
        assert manifest["clusters"] == 32
        assert manifest["session_id"] == "fixture-session"
        assert (frozen / "partition/dataset_centroids.bin").is_file()
        assert counter.read_text(encoding="utf-8") == "3"

        summary = json.loads(
            (batch / "fixture/discovery/discovery_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["attempt_count"] == 3
        assert summary["accepted_clusters"] == 32
        assert summary["accepted_partition_seconds"] >= 0
        assert summary["rejected_partition_seconds"] >= 0
        assert summary["all_attempt_partition_seconds"] >= (
            summary["accepted_partition_seconds"]
            + summary["rejected_partition_seconds"]
            - 1e-9
        )
        assert summary["discovery_wall_seconds"] >= (
            summary["all_attempt_partition_seconds"]
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fake = root / "partition_data"
        write_fake_partition_binary(fake)
        for name in ("base.fbin", "query.fbin", "gt.ibin"):
            write_sparse_bin(root / name, 1, 4)
        write_fixture_source_runbook(root / "source_runbook.md")
        batch = root / "batch"
        env = os.environ.copy()
        env.update(
            {
                "BATCH_ROOT": str(batch),
                "PARTITION_BIN": str(fake),
                "CUVS_PYTHON": sys.executable,
                "FIXTURE_ROOT": str(root),
                "FAKE_PARTITION_COUNTER": str(root / "counter.txt"),
                "FAKE_PARTITION_FAIL_AT": "1",
                "FAKE_ROWS": "100",
                "FAKE_DIM": "4",
                "SESSION_ID": "fixture-failure-session",
            }
        )
        result = run(["bash"], env=env, input_text=discovery_fixture_shell(library))
        assert result.returncode == 7
        first_attempt = next((batch / "fixture/discovery/attempts").iterdir())
        assert (first_attempt / "payload").is_dir()
        assert not (batch / "fixture/frozen/partition_manifest.json").exists()


def verify_event_capture_and_timeline(script: str) -> None:
    programs = embedded_python(script)
    assert "PY_TAGORE_TIMED_RUNNER" in programs
    assert "PY_ANALYZE_TAGORE_TIMELINE" in programs

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner = root / "tagore_timed_runner.py"
        runner.write_text(programs["PY_TAGORE_TIMED_RUNNER"], encoding="utf-8")
        child = root / "child.py"
        child.write_text(
            textwrap.dedent(
                """\
                import sys
                import time

                print("stdout-one", flush=True)
                print("stderr-one", file=sys.stderr, flush=True)
                time.sleep(0.01)
                print("stdout-two", flush=True)
                print("stderr-two", file=sys.stderr, flush=True)
                raise SystemExit(9)
                """
            ),
            encoding="utf-8",
        )
        stdout = root / "stdout.log"
        stderr = root / "stderr.log"
        events = root / "events.jsonl"
        result = run(
            [
                sys.executable,
                str(runner),
                "capture",
                "--stdout-log",
                str(stdout),
                "--stderr-log",
                str(stderr),
                "--events",
                str(events),
                "--",
                sys.executable,
                str(child),
            ]
        )
        assert result.returncode == 9
        assert stdout.read_text(encoding="utf-8") == "stdout-one\nstdout-two\n"
        assert stderr.read_text(encoding="utf-8") == "stderr-one\nstderr-two\n"
        event_rows = [
            json.loads(line)
            for line in events.read_text(encoding="utf-8").splitlines()
        ]
        assert {row["channel"] for row in event_rows} == {"stdout", "stderr"}
        assert {row["text"] for row in event_rows} == {
            "stdout-one",
            "stdout-two",
            "stderr-one",
            "stderr-two",
        }
        assert all(isinstance(row["monotonic_ns"], int) for row in event_rows)
        assert all(row["utc"].endswith("+00:00") for row in event_rows)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        analyzer = root / "analyze.py"
        analyzer.write_text(
            programs["PY_ANALYZE_TAGORE_TIMELINE"], encoding="utf-8"
        )
        events = []

        def add(seconds: float, text: str) -> None:
            events.append(
                {
                    "monotonic_ns": int(seconds * 1_000_000_000),
                    "utc": "2026-07-28T00:00:00+00:00",
                    "channel": "stdout",
                    "text": text,
                }
            )

        merge_ends = []
        for ordinal in range(20):
            moved = ordinal * 10.0
            ready = moved + 8.0
            transferred = ready + 1.0
            mapped = transferred + 1.0
            if ordinal <= 15:
                merge_end = ordinal * 10.0 + 20.0
            else:
                merge_end = ordinal * 10.0 + 15.0
            merge_begin = merge_end - 4.0
            merge_wait = merge_begin - 1.0
            merge_ends.append(merge_end)
            add(moved, f"Device:0; Points num:{100 + ordinal} has been move to GPU")
            add(ready, f"Device:0; Waiting for transfering No.{ordinal} to CPU")
            add(transferred, f"Device:0; No.{ordinal} has been transfered to CPU")
            add(mapped, f"Device:0; No.{ordinal} has been mapped")
            add(merge_wait, f"Waiting to merge No.{ordinal}")
            add(merge_begin, f"Begin to merge No.{ordinal}")
            add(merge_end, f"End to merge No.{ordinal}")
        add(merge_ends[-1] + 3.0, "Begin to merge")
        add(merge_ends[-1] + 10.0, "Index size:123456")
        events.sort(key=lambda row: row["monotonic_ns"])
        event_path = root / "events.jsonl"
        event_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
            encoding="utf-8",
        )
        summary_path = root / "summary.json"
        timeline_path = root / "timeline.tsv"
        result = run(
            [
                sys.executable,
                str(analyzer),
                "--events",
                str(event_path),
                "--clusters",
                "20",
                "--buffer-size",
                "16",
                "--largeindex-started-ns",
                "0",
                "--summary",
                str(summary_path),
                "--timeline",
                str(timeline_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["eligible_backpressure_events"] == 4
        assert summary["gpu_ready_before_previous_merge_end_count"] == 1
        assert summary["cpu_wait_count"] == 20
        assert summary["final_global_post_seconds"] == 7.0
        assert summary["drain_flush_seconds"] == 3.0
        rows = timeline_path.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 21
        assert "gpu_blocked_seconds" not in rows[0]
        assert "kernel_seconds" not in rows[0]
        assert "d2h_seconds" not in rows[0]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        analyzer = root / "analyze.py"
        analyzer.write_text(
            programs["PY_ANALYZE_TAGORE_TIMELINE"], encoding="utf-8"
        )
        events = root / "events.jsonl"
        events.write_text(
            json.dumps(
                {
                    "monotonic_ns": 1,
                    "utc": "2026-07-28T00:00:00+00:00",
                    "channel": "stdout",
                    "text": "Device:0; Points num:10 has been move to GPU",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary = root / "summary.json"
        result = run(
            [
                sys.executable,
                str(analyzer),
                "--events",
                str(events),
                "--clusters",
                "1",
                "--buffer-size",
                "16",
                "--largeindex-started-ns",
                "0",
                "--summary",
                str(summary),
                "--timeline",
                str(root / "timeline.tsv"),
            ]
        )
        assert result.returncode != 0
        assert not summary.exists()


def verify_formal_adapter_fixture(script: str) -> None:
    programs = embedded_python(script)
    assert "PY_VALIDATE_FROZEN_PARTITION" in programs
    for function in (
        "extract_runbook_library",
        "run_conversion",
        "run_largeindex",
        "analyze_largeindex",
        "validate_no_internal_split",
        "validate_final_graph",
        "run_formal_dataset",
    ):
        function_body(script, function)

    formal = function_body(script, "run_formal_dataset")
    assert "partition_data" not in formal
    assert "PARTITION_BIN" not in formal
    ordered_calls = (
        "run_conversion",
        "run_largeindex",
        "analyze_largeindex",
        "validate_no_internal_split",
        "validate_final_graph",
        "run_graph_search",
        "write_dataset_summary",
        "cleanup_successful_dataset",
    )
    positions = [formal.index(call) for call in ordered_calls]
    assert positions == sorted(positions)

    conversion = function_body(script, "run_conversion")
    assert "--consume-source" in conversion
    assert 'partition/dataset"' in formal
    largeindex = function_body(script, "run_largeindex")
    assert "monitor_gpu_build.sh" in largeindex
    assert "--interval 0.1" in largeindex
    assert "--gpu 0" in largeindex
    assert "tagore_timed_runner.py" in largeindex
    graph_validation = function_body(script, "validate_final_graph")
    assert '[[ -s "$graph" ]]' in graph_validation

    revalidator = programs["PY_VALIDATE_FROZEN_PARTITION"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.fbin"
        write_sparse_bin(base, 10, 3)
        prefix = write_partition_fixture(
            root / "frozen/partition", counts=[6, 5, 5, 4], dim=3
        )
        inspector_path = root / "inspect.py"
        inspector_path.write_text(
            programs["PY_INSPECT_PARTITION"], encoding="utf-8"
        )
        inspection_path = root / "inspection.json"
        result = run(
            [
                sys.executable,
                str(inspector_path),
                "--prefix",
                str(prefix),
                "--clusters",
                "4",
                "--rows",
                "10",
                "--dim",
                "3",
                "--max-points",
                "6",
                "--json",
                str(inspection_path),
                "--counts",
                str(root / "counts.tsv"),
            ]
        )
        assert result.returncode == 0, result.stderr
        manifest_value = json.loads(inspection_path.read_text(encoding="utf-8"))
        base_stat = base.stat()
        manifest_value.update(
            {
                "schema": "tagore_vamana_frozen_partition_v1",
                "status": "accepted",
                "dataset": "fixture",
                "base": str(base.resolve()),
                "source_runbook": str((root / "source.md").resolve()),
                "session_id": "fixture-session",
                "attempt": 1,
                "accepted_command_started_ns": 1,
                "frozen_ns": 2,
                "base_identity": {
                    "path": str(base.resolve()),
                    "bytes": base_stat.st_size,
                    "device": base_stat.st_dev,
                    "inode": base_stat.st_ino,
                    "mtime_ns": base_stat.st_mtime_ns,
                },
            }
        )
        manifest = root / "frozen/partition_manifest.json"
        manifest.write_text(
            json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validator_path = root / "validate.py"
        validator_path.write_text(revalidator, encoding="utf-8")
        arguments = [
            "--manifest",
            str(manifest),
            "--dataset",
            "fixture",
            "--base",
            str(base),
            "--rows",
            "10",
            "--dim",
            "3",
            "--max-points",
            "6",
            "--output",
            str(root / "revalidated.json"),
        ]
        result = run([sys.executable, str(validator_path), *arguments])
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "4"
        assert json.loads(
            (root / "revalidated.json").read_text(encoding="utf-8")
        )["valid"] is True

        changed_base = root / "changed.fbin"
        write_sparse_bin(changed_base, 10, 3)
        changed_arguments = list(arguments)
        changed_arguments[changed_arguments.index(str(base))] = str(changed_base)
        result = run([sys.executable, str(validator_path), *changed_arguments])
        assert result.returncode != 0

        shard = Path(f"{prefix}_subshard-0.bin")
        with shard.open("r+b") as stream:
            stream.truncate(shard.stat().st_size - 4)
        result = run([sys.executable, str(validator_path), *arguments])
        assert result.returncode != 0

    library = script_library(script)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        batch = root / "batch"
        base = root / "base.fbin"
        write_sparse_bin(base, 10, 3)
        for name in ("query.fbin", "gt.ibin"):
            write_sparse_bin(root / name, 1, 3)
        prefix = write_partition_fixture(
            batch / "fixture/frozen/partition",
            counts=[6, 5, 5, 4],
            dim=3,
        )
        inspector = root / "inspect.py"
        inspector.write_text(programs["PY_INSPECT_PARTITION"], encoding="utf-8")
        inspection = root / "inspection.json"
        result = run(
            [
                sys.executable,
                str(inspector),
                "--prefix",
                str(prefix),
                "--clusters",
                "4",
                "--rows",
                "10",
                "--dim",
                "3",
                "--max-points",
                "6",
                "--json",
                str(inspection),
                "--counts",
                str(root / "counts.tsv"),
            ]
        )
        assert result.returncode == 0, result.stderr
        manifest_value = json.loads(inspection.read_text(encoding="utf-8"))
        base_stat = base.stat()
        manifest_value.update(
            {
                "schema": "tagore_vamana_frozen_partition_v1",
                "status": "accepted",
                "dataset": "fixture",
                "base": str(base.resolve()),
                "source_runbook": str((root / "source.md").resolve()),
                "session_id": "fixture-session",
                "attempt": 1,
                "accepted_command_started_ns": 1,
                "frozen_ns": 2,
                "base_identity": {
                    "path": str(base.resolve()),
                    "bytes": base_stat.st_size,
                    "device": base_stat.st_dev,
                    "inode": base_stat.st_ino,
                    "mtime_ns": base_stat.st_mtime_ns,
                },
            }
        )
        manifest = batch / "fixture/frozen/partition_manifest.json"
        manifest.write_text(
            json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        counter = root / "partition_counter.txt"
        counter.write_text("3", encoding="utf-8")
        fixture_shell = library + textwrap.dedent(
            """\
            load_dataset() {
              DATASET=fixture
              ROWS=10
              DIM=3
              MAX_POINTS=6
              BASE="$FIXTURE_ROOT/base.fbin"
              QUERY="$FIXTURE_ROOT/query.fbin"
              GT="$FIXTURE_ROOT/gt.ibin"
              SOURCE_RUNBOOK="$FIXTURE_ROOT/source.md"
            }
            run_conversion() {
              [[ "$(cat "$FIXTURE_COUNTER")" == 3 ]]
              mkdir -p "$2/tagore_input"
              printf '{}\\n' >"$2/tagore_input/convert_metadata.json"
            }
            run_largeindex() {
              local formal_root="$2" clusters="$3"
              mkdir -p "$formal_root/tagore_run/index" "$formal_root/index"
              printf 'Cluster num after being adjusted:%s\\n' "$clusters" \
                >"$formal_root/largeindex.stdout.log"
              printf 'graph\\n' \
                >"$formal_root/tagore_run/index/final.tagore.vamana"
              ln -s ../tagore_run/index/final.tagore.vamana \
                "$formal_root/index/_mem.index"
            }
            analyze_largeindex() {
              printf '{}\\n' >"$1/overlap_summary.json"
            }
            validate_final_graph() {
              [[ -s "$2/index/_mem.index" ]]
              printf '{"valid":true}\\n' >"$2/graph_validation.json"
            }
            run_graph_search() {
              printf 'searched\\n' >"$2/search_fixture.txt"
            }
            write_dataset_summary() {
              printf 'summarized\\n' >"$1/summary_fixture.txt"
            }
            cleanup_successful_dataset() {
              printf 'complete\\n' >"$1/formal/adapter_complete.txt"
            }
            write_tools
            run_formal_dataset fixture
            [[ "$(cat "$FIXTURE_COUNTER")" == 3 ]]
            """
        )
        env = os.environ.copy()
        env.update(
            {
                "BATCH_ROOT": str(batch),
                "CUVS_PYTHON": sys.executable,
                "FIXTURE_ROOT": str(root),
                "FIXTURE_COUNTER": str(counter),
                "SESSION_ID": "fixture-session",
            }
        )
        result = run(["bash"], env=env, input_text=fixture_shell)
        assert result.returncode == 0, result.stderr
        assert (batch / "fixture/formal/adapter_complete.txt").is_file()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = root / "largeindex.stdout.log"
        log.write_text(
            "Cluster num after being adjusted:5\n", encoding="utf-8"
        )
        result = run(
            ["bash"],
            input_text=(
                library
                + f'validate_no_internal_split "{root}" 4\n'
            ),
        )
        assert result.returncode != 0


def verify_search_summary_and_cleanup_fixtures(script: str) -> None:
    programs = embedded_python(script)
    for tag in (
        "PY_SELECT_SIDECAR",
        "PY_SUMMARIZE_SEARCH",
        "PY_GRAPH_PASSPORT",
        "PY_SUMMARIZE_DATASET",
        "PY_SUMMARIZE_BATCH",
    ):
        assert tag in programs
    for function in (
        "select_sidecar",
        "run_graph_search",
        "write_dataset_summary",
        "cleanup_successful_dataset",
        "run_summarize",
    ):
        function_body(script, function)

    search = function_body(script, "run_graph_search")
    for fragment in (
        "--recall-at 10",
        "--num-threads 8",
        "--storage mmap",
        "--search-backend rabitq1-adaptive-exact",
        "--rabitq1-sidecar",
        "for l in 25 50 100 200 400",
        "for rep in 1 2",
    ):
        assert fragment in search
    assert search.count("warmup") >= 1
    cleanup = function_body(script, "cleanup_successful_dataset")
    assert 'resolved_batch="$(realpath -m -- "$BATCH_ROOT")"' in cleanup
    assert '"$resolved_root" == "$resolved_batch"/*' in cleanup

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.fbin"
        write_sparse_bin(base, 10, 3)
        sidecar = root / "sidecar.bin"
        sidecar.write_bytes(b"accepted-sidecar")
        import hashlib

        sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        manifest = root / "sidecars.tsv"
        manifest.write_text(
            "\t".join(
                (
                    "dataset",
                    "base",
                    "sidecar",
                    "sidecar_bytes",
                    "sha256",
                    "format_version",
                    "rows",
                    "dim",
                    "padded_dim",
                    "validation_status",
                )
            )
            + "\n"
            + "\t".join(
                (
                    "fixture",
                    str(base),
                    str(sidecar),
                    str(sidecar.stat().st_size),
                    sidecar_sha,
                    "graph_rabitq1_v1",
                    "10",
                    "3",
                    "8",
                    "accepted",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        selector = root / "select.py"
        selector.write_text(programs["PY_SELECT_SIDECAR"], encoding="utf-8")
        passport = root / "passport.json"
        arguments = [
            "--manifest",
            str(manifest),
            "--dataset",
            "fixture",
            "--base",
            str(base),
            "--rows",
            "10",
            "--dim",
            "3",
            "--output",
            str(passport),
        ]
        result = run([sys.executable, str(selector), *arguments])
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(sidecar.resolve())
        assert json.loads(passport.read_text(encoding="utf-8"))[
            "validation_status"
        ] == "accepted"
        sidecar.write_bytes(b"changed")
        result = run([sys.executable, str(selector), *arguments])
        assert result.returncode != 0

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "search_sweep.tsv"
        fields = (
            "dataset",
            "parameter_value",
            "repeat",
            "recall_at_10",
            "qps",
            "result_sha256",
        )
        lines = ["\t".join(fields)]
        for offset, value in enumerate((25, 50, 100, 200, 400)):
            recall = 0.5 + offset * 0.05
            for repeat in (1, 2):
                lines.append(
                    f"fixture\t{value}\t{repeat}\t{recall}\t"
                    f"{1000.0 / repeat}\thash-{value}"
                )
        raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summarizer = root / "summarize_search.py"
        summarizer.write_text(
            programs["PY_SUMMARIZE_SEARCH"], encoding="utf-8"
        )
        output = root / "search_summary.tsv"
        result = run(
            [
                sys.executable,
                str(summarizer),
                "--input",
                str(raw),
                "--output",
                str(output),
            ]
        )
        assert result.returncode == 0, result.stderr
        assert len(output.read_text(encoding="utf-8").splitlines()) == 6

        changed = raw.read_text(encoding="utf-8").replace(
            "fixture\t100\t2\t0.6\t500.0\thash-100",
            "fixture\t100\t2\t0.6\t500.0\tchanged-hash",
        )
        raw.write_text(changed, encoding="utf-8")
        result = run(
            [
                sys.executable,
                str(summarizer),
                "--input",
                str(raw),
                "--output",
                str(root / "rejected.tsv"),
            ]
        )
        assert result.returncode != 0

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_root = root / "fixture"
        formal = dataset_root / "formal"
        discovery = dataset_root / "discovery"
        formal.mkdir(parents=True)
        discovery.mkdir(parents=True)
        (discovery / "discovery_summary.json").write_text(
            json.dumps(
                {
                    "schema": "tagore_vamana_partition_discovery_v1",
                    "session_id": "same-session",
                    "attempt_count": 2,
                    "accepted_clusters": 32,
                    "discovery_started_ns": 100,
                    "accepted_command_started_ns": 200,
                    "frozen_ns": 300,
                    "discovery_wall_seconds": 2.0,
                    "all_attempt_partition_seconds": 1.5,
                    "rejected_partition_seconds": 0.5,
                    "accepted_partition_seconds": 1.0,
                    "discovery_overhead_seconds": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (formal / "session_id.txt").write_text(
            "same-session\n", encoding="utf-8"
        )
        for name, value in (
            ("conversion_started_ns.txt", 300),
            ("conversion_completed_ns.txt", 1_000_000_300),
            ("largeindex_started_ns.txt", 1_000_000_300),
            ("largeindex_completed_ns.txt", 3_000_000_300),
        ):
            (formal / name).write_text(f"{value}\n", encoding="utf-8")
        (formal / "overlap_summary.json").write_text(
            '{"schema":"tagore_vamana_overlap_summary_v1"}\n',
            encoding="utf-8",
        )
        (formal / "graph_passport.json").write_text(
            '{"valid":true,"bytes":123}\n', encoding="utf-8"
        )
        search_dir = formal / "search"
        search_dir.mkdir()
        (search_dir / "search_summary.tsv").write_text(
            "parameter_value\trepeat_count\n"
            "25\t2\n50\t2\n100\t2\n200\t2\n400\t2\n",
            encoding="utf-8",
        )
        (dataset_root / "status.json").write_text(
            '{"state":"accepted"}\n', encoding="utf-8"
        )
        gpu = formal / "gpu_monitor"
        gpu.mkdir()
        (gpu / "gpu_samples.csv").write_text(
            "sample_epoch,sample_time,gpu_timestamp,gpu_index,gpu_name,"
            "gpu_util_percent,mem_util_percent,memory_used_mib,"
            "memory_total_mib,power_draw_w,temp_c,sm_clock_mhz,pstate\n"
            "1,t,t,0,GPU,80,20,1000,24564,100,50,1000,P0\n",
            encoding="utf-8",
        )
        summarizer = root / "summarize_dataset.py"
        summarizer.write_text(
            programs["PY_SUMMARIZE_DATASET"], encoding="utf-8"
        )
        output = dataset_root / "summary.json"
        result = run(
            [
                sys.executable,
                str(summarizer),
                "--dataset-root",
                str(dataset_root),
                "--dataset",
                "fixture",
                "--output",
                str(output),
            ]
        )
        assert result.returncode == 0, result.stderr
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["accepted_pipeline_seconds"] == 4.0
        assert value["discovery_inclusive_seconds"] == 5.0
        assert isinstance(value["accepted_configuration_wall_seconds"], float)
        assert isinstance(value["full_discovery_execution_wall_seconds"], float)
        assert value["gpu_cpu_pipeline_seconds"] == 2.0

        (formal / "session_id.txt").write_text(
            "later-session\n", encoding="utf-8"
        )
        result = run(
            [
                sys.executable,
                str(summarizer),
                "--dataset-root",
                str(dataset_root),
                "--dataset",
                "fixture",
                "--output",
                str(output),
            ]
        )
        assert result.returncode == 0, result.stderr
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["accepted_configuration_wall_seconds"] == "not_applicable"
        assert (
            value["full_discovery_execution_wall_seconds"] == "not_applicable"
        )

    library = script_library(script)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_root = root / "fixture"
        formal = dataset_root / "formal"
        targets = (
            dataset_root / "frozen/partition",
            formal / "tagore_input/clusters",
            formal / "tagore_run/local_index",
            formal / "tagore_run/index",
            formal / "index",
        )
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)
            (target / "payload").write_bytes(b"x")
        (formal / "tagore_input/centers.bin").write_bytes(b"centers")
        (dataset_root / "status.json").write_text(
            '{"state":"accepted"}\n', encoding="utf-8"
        )
        for relative, content in (
            ("formal/timing_summary.json", "{}\n"),
            ("formal/overlap_summary.json", "{}\n"),
            ("formal/graph_passport.json", '{"valid":true}\n'),
            ("formal/graph_validation.json", '{"valid":true}\n'),
            ("formal/graph_quality.tsv", "ok\n"),
            (
                "formal/search/search_summary.tsv",
                "parameter_value\trepeat_count\n"
                "25\t2\n50\t2\n100\t2\n200\t2\n400\t2\n",
            ),
        ):
            path = dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        result = run(
            ["bash"],
            env={
                **os.environ,
                "KEEP_LARGE_ARTIFACTS": "0",
                "CUVS_PYTHON": sys.executable,
                "BATCH_ROOT": str(root),
            },
            input_text=library
            + f'cleanup_successful_dataset "{dataset_root}"\n',
        )
        assert result.returncode == 0, result.stderr
        assert all(not target.exists() for target in targets)
        assert not (formal / "tagore_input/centers.bin").exists()
        assert (dataset_root / "cleanup_manifest.tsv").is_file()
        assert (formal / "timing_summary.json").is_file()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_root = root / "fixture"
        formal = dataset_root / "formal"
        outside = root / "outside"
        outside.mkdir()
        (outside / "keep").write_bytes(b"safe")
        for target in (
            dataset_root / "frozen/partition",
            formal / "tagore_input/clusters",
            formal / "tagore_run/local_index",
            formal / "tagore_run/index",
        ):
            target.mkdir(parents=True, exist_ok=True)
        (formal / "tagore_input/centers.bin").write_bytes(b"centers")
        (formal / "index").parent.mkdir(parents=True, exist_ok=True)
        (formal / "index").symlink_to(outside, target_is_directory=True)
        (dataset_root / "status.json").write_text(
            '{"state":"accepted"}\n', encoding="utf-8"
        )
        for relative, content in (
            ("formal/timing_summary.json", "{}\n"),
            ("formal/overlap_summary.json", "{}\n"),
            ("formal/graph_passport.json", '{"valid":true}\n'),
            ("formal/graph_validation.json", '{"valid":true}\n'),
            ("formal/graph_quality.tsv", "ok\n"),
            (
                "formal/search/search_summary.tsv",
                "parameter_value\trepeat_count\n"
                "25\t2\n50\t2\n100\t2\n200\t2\n400\t2\n",
            ),
        ):
            path = dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        result = run(
            ["bash"],
            env={
                **os.environ,
                "CUVS_PYTHON": sys.executable,
                "BATCH_ROOT": str(root),
            },
            input_text=library
            + f'cleanup_successful_dataset "{dataset_root}"\n',
        )
        assert result.returncode != 0
        assert (outside / "keep").is_file()


def verify_mode_safety_and_preflight(script: str) -> None:
    programs = embedded_python(script)
    assert "PY_PREFLIGHT_DATASET" in programs
    for function in (
        "run_prepare",
        "run_preflight",
        "run_gpu_preflight",
        "run_discovery_for_active_datasets",
        "run_formal_for_active_datasets",
    ):
        function_body(script, function)
    assert "systemd-run" not in script
    assert "memory.max" in script
    assert "memory.high" in script
    assert "memory.swap.max" in script
    assert 'REFERENCE_GPU_MEMORY_MIB=24564' in script
    gpu = function_body(script, "run_gpu_preflight")
    assert "--query-gpu=index,name,memory.total,memory.free,driver_version" in gpu
    assert "--query-compute-apps=pid" in gpu
    assert "REFERENCE_GPU_MEMORY_MIB" in gpu

    preflight = function_body(script, "run_preflight")
    assert "partition_data" not in preflight
    assert "largeIndex" not in preflight
    assert "Tagore.largeIndex" not in preflight
    formal = function_body(script, "run_formal_for_active_datasets")
    assert "run_gpu_preflight" in formal
    assert 'partition_manifest.json" ]]' in formal
    assert "discover_dataset" in formal

    dispatch = script[script.rfind('\ncase "$MODE" in\n') :]
    assert "smoke) python3 \"$TEST_SCRIPT\" --fixture-only ;;" in dispatch
    assert "preflight) run_preflight ;;" in dispatch
    assert "discover) run_discovery_for_active_datasets ;;" in dispatch
    assert "formal) run_formal_for_active_datasets ;;" in dispatch
    assert "summarize) run_summarize ;;" in dispatch
    assert "not_implemented" not in dispatch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.fbin"
        query = root / "query.fbin"
        gt = root / "gt.bin"
        write_sparse_bin(base, 10, 3)
        write_sparse_bin(query, 2, 3)
        write_sparse_bin(gt, 2, 10)
        tool = root / "preflight.py"
        tool.write_text(programs["PY_PREFLIGHT_DATASET"], encoding="utf-8")
        output = root / "preflight.json"
        arguments = [
            "--dataset",
            "fixture",
            "--base",
            str(base),
            "--query",
            str(query),
            "--gt",
            str(gt),
            "--rows",
            "10",
            "--dim",
            "3",
            "--max-points",
            "6",
            "--staging-path",
            str(root),
            "--available-bytes",
            str(10**15),
            "--output",
            str(output),
        ]
        result = run([sys.executable, str(tool), *arguments])
        assert result.returncode == 0, result.stderr
        value = json.loads(output.read_text(encoding="utf-8"))
        expected = int(
            (
                2 * 10 * (4 * 3 + 4)
                + 8 * 10
                + 2 * 10 * (4 + 64 * 4)
                + 24
                + 10 * (4 + 64 * 4)
            )
            * 1.10
        ) + 10 * 1024**3
        assert value["required_free_bytes"] == expected
        assert value["available_free_bytes"] == 10**15
        assert value["accepted"] is True

        low = list(arguments)
        low[low.index(str(10**15))] = str(expected - 1)
        low_output = root / "low.json"
        low[low.index(str(output))] = str(low_output)
        result = run([sys.executable, str(tool), *low])
        assert result.returncode != 0
        rejected = json.loads(low_output.read_text(encoding="utf-8"))
        assert rejected["accepted"] is False
        assert rejected["required_free_bytes"] == expected
        assert rejected["available_free_bytes"] == expected - 1

        resumed = list(arguments)
        resumed.extend(["--already-allocated-bytes", "100"])
        resumed_output = root / "resumed.json"
        resumed[resumed.index(str(output))] = str(resumed_output)
        resumed[resumed.index(str(10**15))] = str(expected - 1)
        result = run([sys.executable, str(tool), *resumed])
        assert result.returncode == 0, result.stderr
        resumed_value = json.loads(
            resumed_output.read_text(encoding="utf-8")
        )
        assert resumed_value["required_free_bytes"] == expected
        assert resumed_value["already_allocated_bytes"] == 100
        assert resumed_value["required_additional_free_bytes"] == expected - 100
        assert resumed_value["accepted"] is True

    library = script_library(script)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.md"
        write_fixture_source_runbook(source)
        batch = root / "batch"
        shell = library + textwrap.dedent(
            """\
            load_dataset() {
              DATASET=fixture
              ROWS=10
              DIM=3
              MAX_POINTS=6
              BASE="$FIXTURE_ROOT/base.fbin"
              QUERY="$FIXTURE_ROOT/query.fbin"
              GT="$FIXTURE_ROOT/gt.bin"
              SOURCE_RUNBOOK="$FIXTURE_ROOT/source.md"
            }
            initialize_batch
            printf 'tampered\\n' \
              >>"$BATCH_ROOT/provenance/tools/partition_policy.py"
            initialize_batch
            """
        )
        for name in ("base.fbin", "query.fbin", "gt.bin"):
            write_sparse_bin(root / name, 1, 3)
        result = run(
            ["bash"],
            env={
                **os.environ,
                "BATCH_ROOT": str(batch),
                "FIXTURE_ROOT": str(root),
                "SESSION_ID": "immutable-fixture",
            },
            input_text=shell,
        )
        assert result.returncode != 0
        assert "immutable" in result.stderr or "changed" in result.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    source = RUNBOOK.read_text(encoding="utf-8")
    script = extract_script(source)
    if args.extract:
        sys.stdout.write(script)
        return
    if not args.fixture_only:
        verify_static_contract(source, script)
        verify_syntax(script)
    verify_partition_policy_and_inspector(script)
    verify_discovery_fixtures(script)
    verify_event_capture_and_timeline(script)
    verify_formal_adapter_fixture(script)
    verify_search_summary_and_cleanup_fixtures(script)
    verify_mode_safety_and_preflight(script)
    if args.fixture_only:
        print("tagore-vamana fixture tests passed")
    else:
        print("tagore-vamana four-dataset runbook contract tests passed")


if __name__ == "__main__":
    main()
