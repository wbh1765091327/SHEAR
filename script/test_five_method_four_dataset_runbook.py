#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import threading


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/runbooks/five_method_four_dataset_build_search.md"
BEGIN = "<!-- BEGIN RUNBOOK_SCRIPT -->"
END = "<!-- END RUNBOOK_SCRIPT -->"
DATASETS = (
    ("wiki35m", 35_000_000, 1024, "wiki-1024-35M/en_base.fbin"),
    ("laion100m", 100_000_000, 512, "laion-512-100M/base.fbin"),
    ("msmarco70m", 70_000_000, 768, "msmarco-768-70M/base.fbin"),
    ("deep100m", 100_000_000, 96, "deep-96-100M/base.100M.fbin"),
)
METHODS = ("shear", "cagra_nn_descent", "tagore_cagra", "tagore_vamana")


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


def heredoc_body(script: str, tag: str) -> str:
    match = re.search(
        rf"<<'{re.escape(tag)}'\n(?P<body>.*?)\n{re.escape(tag)}(?:\n|$)",
        script,
        re.S,
    )
    assert match, f"missing heredoc: {tag}"
    return match.group("body") + "\n"


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
    assert "five methods" in source.lower()
    assert "IVF-RaBitQ" in source
    assert "not executed" in source
    assert "FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)" in script
    assert "GRAPH_METHODS=(shear cagra_nn_descent tagore_cagra tagore_vamana)" in script
    assert "SEARCH_LISTS=(25 50 100 200 400)" in script
    assert (
        'IVF_REUSE_ROOT="${IVF_REUSE_ROOT:-/data_staging/wbh/ivf_rabitq_four_dataset/runs/20260721_235157}"'
        in script
    )
    for mode in (
        "help",
        "prepare",
        "preflight",
        "smoke",
        "formal",
        "wiki35m",
        "laion100m",
        "msmarco70m",
        "deep100m",
        "summarize",
    ):
        assert f"{mode})" in script, mode

    loader = function_body(script, "load_dataset")
    for dataset, rows, dim, base_suffix in DATASETS:
        branch = re.search(rf"{dataset}\).*?;;", loader, re.S)
        assert branch, dataset
        body = branch.group(0)
        assert f"ROWS={rows}" in body
        assert f"DIM={dim}" in body
        assert base_suffix in body

    run_dataset = function_body(script, "run_dataset")
    assert 'for method in "${GRAPH_METHODS[@]}"; do' in run_dataset
    run_formal = function_body(script, "run_formal")
    assert 'for dataset in "${FORMAL_DATASETS[@]}"; do' in run_formal
    assert "run_gpu_build" not in run_formal
    assert "run_formal_search" not in run_formal
    assert "ivf_rabitq_cpu_index_build" not in run_formal

    shear = heredoc_body(script, "SHEAR_WRAPPER")
    for fragment in (
        "SHEAR_PIPNN_HASHPRUNE_BACKFILL_MODE=rhalf-soft-tier-q3",
        "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ABLATION=none",
        "SHEAR_PIPNN_HASHPRUNE_BACKFILL_BUDGET=0",
        "SHEAR_PIPNN_HASHPRUNE_BACKFILL_ALPHA=1.0",
        "SHEAR_PIPNN_HASHPRUNE_SOFT_TIER_PRIORITY_ALPHA=1.02",
        "SHEAR_PIPNN_HASHPRUNE_RHALF_MAX_BACKFILL_PER_ROW=3",
        "run_shear_build",
        "validate_shear_profile",
        "validate_graph",
    ):
        assert fragment in shear, fragment
    soft_tail = heredoc_body(script, "PY_SOFT_TAIL_AUDIT")
    for fragment in (
        '"rhalf-soft-tier-q3"',
        '"soft-tier-q3"',
        '"priority_alpha"',
        '"effective_priority_alpha"',
        '"requested_priority_alpha"',
        "1.02",
        '"effective_max_backfill_per_row"',
        '"requested_max_backfill_per_row"',
        '"derived_budget"',
        "32",
        '"backup_hash_quota"',
        "3",
        '"uses_alpha_gate"',
    ):
        assert fragment in soft_tail, fragment

    search = function_body(script, "run_graph_search")
    for fragment in (
        "--recall-at 10",
        '--search-list-size "$l"',
        "--num-threads 8",
        "--storage mmap",
        "--search-backend rabitq1-adaptive-exact",
        '--rabitq1-sidecar "$SIDECAR"',
        "for l in 25 50 100 200 400; do",
        "for rep in 1 2; do",
    ):
        assert fragment in search, fragment
    assert "rabitq1-post-rerank" not in search

    importer = function_body(script, "import_ivf_results")
    assert "PY_IVF_IMPORT" in importer
    assert "build_shear_index" not in importer
    assert "search_shear_index" not in importer
    assert "ivf_rabitq_fbin_benchmark" not in importer
    cleanup = function_body(script, "cleanup_successful_method")
    assert "realpath -m" in cleanup
    assert '[[ "$resolved" == "$payload"/* ]]' in cleanup
    assert 'rm -rf -- "$target"' in cleanup

    for source_path in (
        "docs/runbooks/p1_p2_shear_four_dataset.md",
        "docs/runbooks/p1_p2_wiki35m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_laion100m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_msmarco70m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_shear_cagra_tagore.md",
        "docs/runbooks/unified_graph_rabitq1_search_four_dataset.md",
        "docs/runbooks/p1_p2_ivf_rabitq_four_dataset.md",
    ):
        assert source_path in script, source_path


def verify_syntax(script: str) -> None:
    syntax = run(["bash", "-n"], input_text=script)
    assert syntax.returncode == 0, syntax.stderr
    for tag in ("SHEAR_WRAPPER", "ACE_WRAPPER", "TAGORE_WRAPPER"):
        syntax = run(["bash", "-n"], input_text=heredoc_body(script, tag))
        assert syntax.returncode == 0, f"{tag}: {syntax.stderr}"
    for tag, program in embedded_python(script).items():
        compiled = run(
            [
                "python3",
                "-c",
                "import sys; compile(sys.stdin.read(), sys.argv[1], 'exec')",
                tag,
            ],
            input_text=program,
        )
        assert compiled.returncode == 0, f"{tag}: {compiled.stderr}"


def verify_dependency_extraction(script: str) -> None:
    extractor = heredoc_body(script, "PY_EXTRACT_RUNBOOK")
    dependencies = (
        "docs/runbooks/p1_p2_shear_four_dataset.md",
        "docs/runbooks/p1_p2_wiki35m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_laion100m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_msmarco70m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_shear_cagra_tagore.md",
        "docs/runbooks/unified_graph_rabitq1_search_four_dataset.md",
        "docs/runbooks/p1_p2_ivf_rabitq_four_dataset.md",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for ordinal, relative in enumerate(dependencies):
            output = root / f"dependency_{ordinal}.lib.sh"
            result = run(
                ["python3", "-", str(ROOT / relative), str(output)],
                input_text=extractor,
            )
            assert result.returncode == 0, f"{relative}: {result.stderr}"
            library = output.read_text(encoding="utf-8")
            assert library.startswith("#!/usr/bin/env bash\n"), relative
            assert 'case "$MODE" in' not in library, relative


def verify_tagore_converter_consumes_validated_shards(script: str) -> None:
    extractor = heredoc_body(script, "PY_EXTRACT_RUNBOOK")
    dependencies = (
        "docs/runbooks/p1_p2_wiki35m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_laion100m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_msmarco70m_shear_cagra_tagore.md",
        "docs/runbooks/p1_p2_shear_cagra_tagore.md",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        converter_source = None
        for ordinal, relative in enumerate(dependencies):
            output = root / f"dependency_{ordinal}.lib.sh"
            result = run(
                ["python3", "-", str(ROOT / relative), str(output)],
                input_text=extractor,
            )
            assert result.returncode == 0, f"{relative}: {result.stderr}"
            library = output.read_text(encoding="utf-8")
            tagore = function_body(library, "run_tagore_vamana")
            assert "--consume-source" in tagore, relative
            writer = function_body(library, "write_converter")
            match = re.search(r"<<'PY'\n(?P<body>.*?)\nPY\n", writer, re.S)
            assert match, relative
            if converter_source is None:
                converter_source = match.group("body") + "\n"

        converter = root / "convert_diskann_partition_to_tagore.py"
        converter.write_text(converter_source, encoding="utf-8")
        prefix = root / "dataset"
        for cluster in range(2):
            data = Path(f"{prefix}_subshard-{cluster}.bin")
            ids = Path(f"{prefix}_subshard-{cluster}_ids_uint32.bin")
            data.write_bytes(
                struct.pack("<II", 4, 2)
                + struct.pack("<8f", *(float(cluster * 10 + value) for value in range(8)))
            )
            ids.write_bytes(struct.pack("<II4I", 4, 1, 0, 1, 2, 3))
        output = root / "tagore_input"
        result = run(
            [
                "python3",
                str(converter),
                "--diskann-prefix",
                str(prefix),
                "--tagore-out",
                str(output),
                "--num-points",
                "4",
                "--dim",
                "2",
                "--clusters",
                "2",
                "--max-points",
                "4",
                "--consume-source",
            ]
        )
        assert result.returncode == 0, result.stderr
        for cluster in range(2):
            assert not Path(f"{prefix}_subshard-{cluster}.bin").exists()
            assert not Path(f"{prefix}_subshard-{cluster}_ids_uint32.bin").exists()
            assert (output / "clusters" / f"cluster_{cluster}").stat().st_size == 36
        metadata = json.loads((output / "convert_metadata.json").read_text(encoding="utf-8"))
        assert metadata["source_consumed"] is True
        assert metadata["consumed_source_files"] == 4


def verify_process_monitor_tolerates_proc_exit_race(script: str) -> None:
    monitor_source = heredoc_body(script, "PY_PROCESS_MONITOR")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        monitor = root / "process_tree_io_monitor.py"
        monitor.write_text(monitor_source, encoding="utf-8")
        sitecustomize = root / "sitecustomize.py"
        sitecustomize.write_text(
            """
import pathlib

_original_read_text = pathlib.PosixPath.read_text
_raised = False

def _flaky_read_text(self, *args, **kwargs):
    global _raised
    if not _raised and self.name == "stat" and self.parent.name.isdigit():
        _raised = True
        raise ProcessLookupError(3, "No such process", str(self))
    return _original_read_text(self, *args, **kwargs)

pathlib.PosixPath.read_text = _flaky_read_text
""".lstrip(),
            encoding="utf-8",
        )
        root_process = subprocess.Popen(["sleep", "1"])
        reaper = threading.Thread(target=root_process.wait)
        reaper.start()
        try:
            environment = os.environ.copy() | {"PYTHONPATH": str(root)}
            result = run(
                [
                    "python3",
                    str(monitor),
                    str(root_process.pid),
                    str(root / "samples.tsv"),
                    str(root / "summary.json"),
                ],
                env=environment,
            )
        finally:
            reaper.join()
        assert result.returncode == 0, result.stderr
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert summary["valid_samples"] > 0


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_ivf_fixture(root: Path) -> None:
    query_and_gt = {
        "wiki35m": ("wiki-1024-35M/query.fbin", "wiki-1024-35M/gt100.bin"),
        "laion100m": ("laion-512-100M/query.fbin", "laion-512-100M/gt100.bin"),
        "msmarco70m": ("msmarco-768-114M/query.fbin", "msmarco-768-70M/gt100.bin"),
        "deep100m": (
            "deep-96-100M/query.public.10K.fbin",
            "deep-96-100M/groundtruth.public.10K.ibin",
        ),
    }
    nlists = {
        "wiki35m": 32768,
        "laion100m": 65536,
        "msmarco70m": 65536,
        "deep100m": 131072,
    }
    build_fields = (
        "dataset",
        "rows",
        "dim",
        "nlist",
        "bits_per_dim",
        "fast_quantize",
        "gpu_build_seconds",
        "end_to_end_wall_seconds",
        "external_wall_seconds",
        "peak_process_tree_rss_bytes",
        "peak_hbm_bytes",
        "main_index_bytes",
        "hnsw_index_bytes",
        "status",
    )
    build_rows = []
    for ordinal, (dataset, rows, dim, base_suffix) in enumerate(DATASETS, 1):
        query_suffix, gt_suffix = query_and_gt[dataset]
        run_root = root / "hot_path" / dataset / "ivf_rabitq_exact" / "run_1"
        run_root.mkdir(parents=True)
        (run_root / "status.json").write_text(
            json.dumps({"phase": "accepted", "exit_code": 0}) + "\n", encoding="utf-8"
        )
        (run_root / "cleanup_summary.txt").write_text(
            f"dataset={dataset}\nverified=true\n", encoding="utf-8"
        )
        (run_root / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "ivf_rabitq_dataset_manifest_v1",
                    "base": {
                        "path": f"/data_nvme_kioxia/tyj/data/{base_suffix}",
                        "rows": rows,
                        "dim": dim,
                    },
                    "query": {
                        "path": f"/data_nvme_kioxia/tyj/data/{query_suffix}",
                        "rows": 10_000,
                        "dim": dim,
                    },
                    "ground_truth": {
                        "path": f"/data_nvme_kioxia/tyj/data/{gt_suffix}",
                        "rows": 10_000,
                        "width": 100,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        search_rows = []
        for nprobe, recall in ((64, 0.9), (128, 0.95)):
            for repeat, qps in ((1, 1000.0 / ordinal), (2, 800.0 / ordinal), (3, 1.0)):
                search_rows.append(
                    {
                        "dataset": dataset,
                        "nprobe": nprobe,
                        "repeat": repeat,
                        "recall_at_10": recall,
                        "seconds": 10.0,
                        "qps": qps,
                        "query_workers": 8,
                        "rerank": "false",
                    }
                )
        write_tsv(
            run_root / "search_sweep.tsv",
            (
                "dataset",
                "nprobe",
                "repeat",
                "recall_at_10",
                "seconds",
                "qps",
                "query_workers",
                "rerank",
            ),
            search_rows,
        )
        build_rows.append(
            {
                "dataset": dataset,
                "rows": rows,
                "dim": dim,
                "nlist": nlists[dataset],
                "bits_per_dim": 7,
                "fast_quantize": "False",
                "gpu_build_seconds": 10.0 * ordinal,
                "end_to_end_wall_seconds": 20.0 * ordinal,
                "external_wall_seconds": 21.0 * ordinal,
                "peak_process_tree_rss_bytes": 1000 * ordinal,
                "peak_hbm_bytes": 2000 * ordinal,
                "main_index_bytes": 3000 * ordinal,
                "hnsw_index_bytes": 4000 * ordinal,
                "status": "accepted",
            }
        )
    write_tsv(root / "build_summary.tsv", build_fields, build_rows)
    (root / "summary.md").write_text("# accepted fixture\n", encoding="utf-8")


def run_library_function(
    script: str,
    expression: str,
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        library = Path(directory) / "library.sh"
        library.write_text(script_library(script), encoding="utf-8")
        return run(
            ["bash", "-c", 'source "$1"; shift; eval "$1"', "bash", str(library), expression],
            env=environment,
        )


def verify_ivf_import_and_report(script: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ivf = root / "ivf"
        batch = root / "batch"
        write_ivf_fixture(ivf)
        environment = os.environ.copy() | {
            "MODE": "help",
            "BATCH_ROOT": str(batch),
            "IVF_REUSE_ROOT": str(ivf),
            "SHEAR_ROOT": str(ROOT),
            "PATH": "/usr/bin:/bin",
        }
        result = run_library_function(script, "import_ivf_results", environment=environment)
        assert result.returncode == 0, result.stderr
        with (batch / "ivf_reused_search_sweep.tsv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert rows
        assert {row["repeat"] for row in rows} == {"1", "2"}
        assert all(row["execution_origin"] == "reused" for row in rows)
        with (batch / "ivf_reused_search_summary.tsv").open(encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream, delimiter="\t"))
        wiki64 = next(
            row for row in summary if row["dataset"] == "wiki35m" and row["parameter_value"] == "64"
        )
        assert wiki64["parameter_kind"] == "nprobe"
        assert wiki64["repeat_count"] == "2"
        assert abs(float(wiki64["qps_mean"]) - 900.0) < 1e-9

        graph_construction = [
            {
                "dataset": "wiki35m",
                "method": "shear",
                "execution_origin": "current",
                "status": "accepted",
                "external_wall_seconds": 1.0,
                "peak_dram_bytes": 2,
                "read_bytes": 3,
                "write_bytes": 4,
                "peak_temp_bytes": 5,
                "index_bytes": 6,
                "source_root": "",
            }
        ]
        write_tsv(
            batch / "graph_construction_summary.tsv",
            tuple(graph_construction[0]),
            graph_construction,
        )
        graph_search = [
            {
                "dataset": "wiki35m",
                "method": "shear",
                "execution_origin": "current",
                "parameter_kind": "L",
                "parameter_value": 25,
                "recall_at_10": 0.9,
                "qps_mean": 100.0,
                "qps_sample_std": 1.0,
                "qps_median": 100.0,
                "qps_min": 99.0,
                "qps_max": 101.0,
                "repeat_count": 2,
                "source_root": "",
            }
        ]
        write_tsv(batch / "graph_search_summary.tsv", tuple(graph_search[0]), graph_search)
        result = run_library_function(script, "generate_batch_report", environment=environment)
        assert result.returncode == 0, result.stderr
        report = (batch / "report.md").read_text(encoding="utf-8")
        assert "IVF-RaBitQ was not executed" in report
        assert "reused" in report
        with (batch / "search_summary.tsv").open(encoding="utf-8") as stream:
            combined = list(csv.DictReader(stream, delimiter="\t"))
        assert {row["parameter_kind"] for row in combined} == {"L", "nprobe"}


def write_fake_search(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import csv
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
def option(name):
    position = arguments.index(name)
    return arguments[position + 1]

result = Path(option("--result-path"))
metrics = Path(option("--metrics-path"))
search_list = int(option("--search-list-size"))
call_log = Path(os.environ["CALL_LOG"])
with call_log.open("a") as stream:
    stream.write(str(result) + "\\n")
payload = f"L={search_list}\\n".encode()
if os.environ.get("MISMATCH") == "1" and "L100/rep_2" in str(result):
    payload += b"changed\\n"
result.write_bytes(payload)
fields = (
    "recall", "qps", "total_one_bit_evaluations", "total_lower_bound_rejects",
    "total_navigation_exact_evaluations", "total_adaptive_exact_refinements",
    "total_exact_evaluations", "logical_exact_vector_bytes",
)
row = {
    "recall": (
        0.0
        if os.environ.get("ZERO_RECALL") == "1"
        else 0.80 + (25, 50, 100, 200, 400).index(search_list) * 0.02
    ),
    "qps": 2000.0 - search_list,
    "total_one_bit_evaluations": 100,
    "total_lower_bound_rejects": 50,
    "total_navigation_exact_evaluations": 10,
    "total_adaptive_exact_refinements": 5,
    "total_exact_evaluations": 15,
    "logical_exact_vector_bytes": 4096,
}
with metrics.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\\t", lineterminator="\\n")
    writer.writeheader()
    writer.writerow(row)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def verify_graph_search_fixture(script: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fake_search = root / "fake_search.py"
        write_fake_search(fake_search)

        def execute(
            name: str, mismatch: bool, zero_recall: bool = False
        ) -> tuple[subprocess.CompletedProcess[str], Path]:
            method_root = root / name
            (method_root / "payload/index").mkdir(parents=True)
            (method_root / "payload/index/_mem.index").write_bytes(b"graph")
            call_log = root / f"{name}.calls"
            environment = os.environ.copy() | {
                "MODE": "help",
                "SHEAR_ROOT": str(ROOT),
                "SEARCH_BIN": str(fake_search),
                "METHOD_ROOT": str(method_root),
                "FIXTURE_BASE": str(root / "base.fbin"),
                "FIXTURE_QUERY": str(root / "query.fbin"),
                "FIXTURE_GT": str(root / "gt.bin"),
                "FIXTURE_SIDECAR": str(root / "sidecar.bin"),
                "CALL_LOG": str(call_log),
                "MISMATCH": "1" if mismatch else "0",
                "ZERO_RECALL": "1" if zero_recall else "0",
                "PATH": "/usr/bin:/bin",
            }
            expression = (
                "prepare_graph_search_cache() { :; }; "
                "DATASET=wiki35m; BASE=\"$FIXTURE_BASE\"; QUERY=\"$FIXTURE_QUERY\"; "
                "GT=\"$FIXTURE_GT\"; SIDECAR=\"$FIXTURE_SIDECAR\"; "
                "run_graph_search shear \"$METHOD_ROOT\""
            )
            return run_library_function(script, expression, environment=environment), call_log

        accepted, accepted_calls = execute("accepted", mismatch=False)
        assert accepted.returncode == 0, accepted.stderr
        calls = accepted_calls.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 15
        assert sum("/warmup/" in call for call in calls) == 5
        assert sum("/rep_" in call for call in calls) == 10
        with (root / "accepted/graph_search_sweep.tsv").open(encoding="utf-8") as stream:
            sweep = list(csv.DictReader(stream, delimiter="\t"))
        assert len(sweep) == 10
        assert {row["repeat"] for row in sweep} == {"1", "2"}
        with (root / "accepted/graph_search_summary.tsv").open(encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream, delimiter="\t"))
        assert len(summary) == 5
        assert all(row["repeat_count"] == "2" for row in summary)

        rejected, rejected_calls = execute("rejected", mismatch=True)
        assert rejected.returncode != 0
        calls = rejected_calls.read_text(encoding="utf-8").splitlines()
        assert sum("/warmup/" in call for call in calls) == 5
        assert sum("/rep_" in call for call in calls) == 10

        zero_recall, zero_recall_calls = execute(
            "zero_recall", mismatch=False, zero_recall=True
        )
        assert zero_recall.returncode != 0
        calls = zero_recall_calls.read_text(encoding="utf-8").splitlines()
        assert sum("/warmup/" in call for call in calls) == 5
        assert sum("/rep_" in call for call in calls) == 10


def verify_graph_quality_gate_fixture(script: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        environment = os.environ.copy() | {
            "MODE": "help",
            "SHEAR_ROOT": str(ROOT),
            "PATH": "/usr/bin:/bin",
        }

        def execute(name: str, reachable_ratio: float) -> subprocess.CompletedProcess[str]:
            method_root = root / name
            payload = method_root / "payload"
            payload.mkdir(parents=True)
            write_tsv(
                payload / "graph_quality.tsv",
                (
                    "nodes",
                    "directed_reachable_from_start",
                    "directed_reachable_from_start_ratio",
                ),
                (
                    {
                        "nodes": 100,
                        "directed_reachable_from_start": int(100 * reachable_ratio),
                        "directed_reachable_from_start_ratio": reachable_ratio,
                    },
                ),
            )
            expression = f"validate_graph_quality_gate {method_root!s}"
            return run_library_function(script, expression, environment=environment)

        accepted = execute("accepted", 0.99)
        assert accepted.returncode == 0, accepted.stderr
        rejected = execute("rejected", 0.01)
        assert rejected.returncode != 0


def verify_cleanup_boundary(script: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        batch = root / "batch"
        method_root = batch / "hot_path/wiki35m/shear/run_1"
        payload = method_root / "payload"
        payload.mkdir(parents=True)
        outside = root / "outside"
        outside.mkdir()
        (outside / "keep").write_text("keep\n", encoding="utf-8")
        (payload / "index").symlink_to(outside, target_is_directory=True)
        environment = os.environ.copy() | {
            "MODE": "help",
            "BATCH_ROOT": str(batch),
            "SHEAR_ROOT": str(ROOT),
            "PATH": "/usr/bin:/bin",
        }
        expression = f'cleanup_successful_method shear {method_root!s}'
        result = run_library_function(script, expression, environment=environment)
        assert result.returncode != 0
        assert (outside / "keep").is_file()


def verify_help(script: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner = root / "runner.sh"
        runner.write_text(script, encoding="utf-8")
        runner.chmod(0o755)
        call_log = root / "calls.log"
        environment = os.environ.copy() | {
            "SHEAR_ROOT": str(ROOT),
            "OUTPUT_BASE": str(root / "output"),
            "CALL_LOG": str(call_log),
            "PATH": "/usr/bin:/bin",
        }
        result = run([str(runner), "help"], env=environment)
        assert result.returncode == 0, result.stderr
        assert "formal" in result.stdout
        assert "IVF-RaBitQ" in result.stdout
        assert not (root / "output").exists()
        assert not call_log.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    source = RUNBOOK.read_text(encoding="utf-8")
    script = extract_script(source)
    if args.extract:
        print(script, end="")
        return
    verify_static_contract(source, script)
    verify_syntax(script)
    verify_dependency_extraction(script)
    verify_tagore_converter_consumes_validated_shards(script)
    verify_process_monitor_tolerates_proc_exit_race(script)
    verify_ivf_import_and_report(script)
    verify_graph_search_fixture(script)
    verify_graph_quality_gate_fixture(script)
    verify_cleanup_boundary(script)
    verify_help(script)
    print("five-method four-dataset runbook tests passed")


if __name__ == "__main__":
    main()
