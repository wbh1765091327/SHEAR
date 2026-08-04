#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/runbooks/p1_p2_ivf_rabitq_four_dataset.md"
BEGIN = "<!-- BEGIN RUNBOOK_SCRIPT -->"
END = "<!-- END RUNBOOK_SCRIPT -->"
DATASETS = {
    "wiki35m": (35_000_000, 1024, 32_768),
    "laion100m": (100_000_000, 512, 65_536),
    "msmarco70m": (70_000_000, 768, 65_536),
    "deep100m": (100_000_000, 96, 131_072),
}


def extract_script(source: str) -> str:
    start = source.index(BEGIN) + len(BEGIN)
    stop = source.index(END, start)
    match = re.search(r"```bash\n(?P<body>.*?)\n```", source[start:stop], re.S)
    assert match, "missing fenced Bash payload"
    return match.group("body") + "\n"


def function_body(script: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n", script, re.M)
    assert match, f"missing function {name}"
    next_match = re.search(
        r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{\n", script[match.end() :], re.M
    )
    stop = match.end() + next_match.start() if next_match else len(script)
    return script[match.start() : stop]


def embedded_python(script: str) -> dict[str, str]:
    pattern = re.compile(
        r"<<'(?P<tag>PY_[A-Z0-9_]+)'\n(?P<body>.*?)\n(?P=tag)(?:\n|$)", re.S
    )
    matches = list(pattern.finditer(script))
    programs = {match.group("tag"): match.group("body") + "\n" for match in matches}
    assert len(programs) == len(matches), "duplicate embedded-Python tag"
    assert programs, "missing embedded Python"
    return programs


def run_program(program: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "-", *arguments],
        input=program,
        text=True,
        capture_output=True,
        check=False,
    )


def verify_documented_extractor(source: str, script: str) -> None:
    awk_program = (
        "/^[<]!-- BEGIN RUNBOOK_SCRIPT --[>]$/{on=1;next}"
        "/^[<]!-- END RUNBOOK_SCRIPT --[>]$/{on=0}on"
    )
    assert f"awk '{awk_program}'" in source
    result = subprocess.run(
        ["awk", awk_program],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "```bash" and lines[-1] == "```"
    assert "\n".join(lines[1:-1]) + "\n" == script


def make_search_json(
    nprobes: list[int],
    recalls: dict[int, float],
    repeats: list[float],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "params": {
            "topk": 10,
            "target_recall": 0.97,
            "warmup_queries": 100,
            "repeats": len(repeats),
            "use_hacc": True,
            "query_limit": 0,
            "nprobe_mode": "explicit_exact",
            "requested_nprobes": nprobes,
        },
        "index_metadata": {
            "num_vectors": 35_000_000,
            "dim": 1024,
            "nlist": 32_768,
            "bits_per_dim": 7,
        },
        "execution": {
            "query_parallelism": 8,
            "omp_threads": 8,
            "rerank": False,
            "num_queries": 10_000,
        },
        "results": [
            {
                "nprobe": nprobe,
                "recall_at_10": recalls[nprobe],
                "qps_median": sorted(repeats)[len(repeats) // 2],
                "qps_min": min(repeats),
                "qps_max": max(repeats),
                "repeat_seconds": [10_000.0 / value for value in repeats],
                "repeat_qps": repeats,
            }
            for nprobe in nprobes
        ],
        "target_reached": any(recalls[value] >= 0.97 for value in nprobes),
        "first_target_nprobe": next(
            (value for value in nprobes if recalls[value] >= 0.97), None
        ),
        "max_recall": max(recalls.values()),
    }


def verify_static_contract(source: str, script: str) -> None:
    assert script.count('MODE="${1:-help}"') == 1
    for mode in ("help", "prepare", "smoke", "formal", *DATASETS):
        assert f"{mode})" in script

    load_dataset = function_body(script, "load_dataset")
    for name, (rows, dim, nlist) in DATASETS.items():
        branch = re.search(rf"{name}\).*?;;", load_dataset, re.S)
        assert branch, name
        body = branch.group(0)
        assert f"ROWS={rows}" in body
        assert f"DIM={dim}" in body
        assert f"NLIST={nlist}" in body

    build = function_body(script, "run_gpu_build")
    for fragment in (
        '--bits 7',
        '--max-points-per-cluster 64',
        '--fast-quantize false',
        'local niter=20 batch_vectors=131072',
        'niter=5 batch_vectors=8192',
        '--niter "$niter"',
        '--batch-vectors "$batch_vectors"',
    ):
        assert build.count(fragment) == 1, fragment

    search = function_body(script, "run_search_json")
    for fragment in (
        '--topk 10',
        '--query-threads 8',
        '--use-hacc true',
        '--warmup-queries 100',
        '--nprobes-exact "$nprobes"',
    ):
        assert search.count(fragment) == 1, fragment
    assert "--query-limit" not in search

    discovery = function_body(script, "run_discovery")
    assert 'CORE_NPROBES="64,128,256,512"' in script
    assert 'run_search_json "$RUN_ROOT/discovery/core.json" 1 "$CORE_NPROBES"' in discovery
    assert "next=$((current * 2 / 3))" in discovery
    assert "next=$(((current * 3 + 1) / 2))" in discovery
    assert "current=4096" in discovery
    assert 'run_search_json "$output" 1 "$current"' in discovery

    formal = function_body(script, "run_formal_search")
    assert 'run_search_json "$RUN_ROOT/formal_search.json" 3 "$selected"' in formal

    build_cache = function_body(script, "prepare_build_cache")
    search_cache = function_body(script, "prepare_search_cache")
    assert build_cache.index("drop_page_cache") < build_cache.index(
        'prewarm_paths build "$RUN_ROOT" "$BASE"'
    )
    assert '"$INDEX_FILE" "$HNSW_FILE" "$QUERY" "$GT"' in search_cache
    assert '"$BASE"' not in search_cache

    pipeline = function_body(script, "run_dataset_configured")
    ordered = (
        "prepare_build_cache",
        "run_gpu_build",
        "validate_build",
        "prepare_search_cache discovery",
        "run_discovery",
        "prepare_search_cache formal_search",
        "run_formal_search",
        "summarize_search",
        "finalize_successful_dataset",
    )
    positions = [pipeline.index(value) for value in ordered]
    assert positions == sorted(positions)

    cleanup = function_body(script, "cleanup_successful_dataset")
    for fragment in (
        'sha256sum "$INDEX_FILE"',
        'sha256sum "$HNSW_FILE"',
        'rm -f -- "$INDEX_FILE"',
        'rm -f -- "$HNSW_FILE"',
    ):
        assert fragment in cleanup
    assert cleanup.index("sha256sum") < cleanup.index("rm -f --")

    formal_entry = function_body(script, "run_formal_entry")
    calls = [f'run_dataset "{dataset}"' for dataset in DATASETS]
    assert all(call in formal_entry for call in calls)
    assert [formal_entry.index(call) for call in calls] == sorted(
        formal_entry.index(call) for call in calls
    )

    prepare = function_body(script, "run_prepare")
    assert 'CUVS_CMAKE="${CUVS_CMAKE:-/path/to/cmake}"' in script
    assert '"$CUVS_CMAKE" -S "$CUVS_ROOT/cpp"' in prepare
    assert '"$CUVS_CMAKE" --build "$CUVS_ROOT/cpp/build"' in prepare

    for fragment in (
        "gpu_build_seconds",
        "end_to_end_wall_seconds",
        "external_wall_seconds",
        "centroid_hnsw_build_parallelism=8",
        "rerank=false",
        "quality_lower_bound_not_reached",
        "quality_upper_bound_not_reached",
        "paper_eligible=false",
    ):
        assert fragment in source


def verify_discovery_merge(program: str, directory: Path) -> list[int]:
    discovery = directory / "discovery"
    discovery.mkdir()
    core_recalls = {64: 0.88, 128: 0.91, 256: 0.94, 512: 0.96}
    (discovery / "core.json").write_text(
        json.dumps(make_search_json(list(core_recalls), core_recalls, [100.0])),
        encoding="utf-8",
    )
    for name, nprobe, recall in (("lower_0042", 42, 0.84), ("upper_0768", 768, 0.971)):
        (discovery / f"{name}.json").write_text(
            json.dumps(make_search_json([nprobe], {nprobe: recall}, [100.0])),
            encoding="utf-8",
        )

    selected = directory / "selected_nprobes.tsv"
    merged = discovery / "discovery_merged.tsv"
    result = run_program(
        program,
        str(discovery),
        str(selected),
        str(merged),
        "0.85",
        "0.97",
    )
    assert result.returncode == 0, result.stderr
    with selected.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    values = [int(row["nprobe"]) for row in rows]
    assert values == [42, 64, 128, 256, 512, 768]
    assert float(rows[0]["discovery_recall_at_10"]) == 0.84
    assert float(rows[-1]["discovery_recall_at_10"]) == 0.971
    return values


def verify_search_summary(program: str, directory: Path, selected: list[int]) -> None:
    selected_path = directory / "selected_nprobes.tsv"
    recalls = {
        42: 0.84,
        64: 0.88,
        128: 0.91,
        256: 0.94,
        512: 0.96,
        768: 0.971,
    }
    formal = directory / "formal.json"
    formal.write_text(
        json.dumps(make_search_json(selected, recalls, [100.0, 110.0, 90.0])),
        encoding="utf-8",
    )
    sweep = directory / "search_sweep.tsv"
    summary = directory / "search_summary.tsv"
    anomalies = directory / "search_anomalies.tsv"
    result = run_program(
        program,
        str(formal),
        str(selected_path),
        "wiki35m",
        str(sweep),
        str(summary),
        str(anomalies),
        "3",
        "true",
    )
    assert result.returncode == 0, result.stderr
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == len(selected)
    assert math.isclose(float(rows[0]["qps_mean"]), 100.0)
    assert math.isclose(float(rows[0]["qps_sample_std"]), 10.0)
    assert math.isclose(float(rows[0]["qps_median"]), 100.0)
    assert math.isclose(float(rows[0]["qps_min"]), 90.0)
    assert math.isclose(float(rows[0]["qps_max"]), 110.0)

    smoke_selected = directory / "smoke_selected.tsv"
    smoke_selected.write_text(
        "nprobe\tdiscovery_recall_at_10\tsource\n8\t0.5\tsmoke\n",
        encoding="utf-8",
    )
    smoke_json = directory / "smoke.json"
    smoke_json.write_text(
        json.dumps(make_search_json([8], {8: 0.5}, [125.0])), encoding="utf-8"
    )
    smoke_summary = directory / "smoke_summary.tsv"
    result = run_program(
        program,
        str(smoke_json),
        str(smoke_selected),
        "smoke",
        str(directory / "smoke_sweep.tsv"),
        str(smoke_summary),
        str(directory / "smoke_anomalies.tsv"),
        "1",
        "false",
    )
    assert result.returncode == 0, result.stderr
    with smoke_summary.open(newline="", encoding="utf-8") as handle:
        smoke_row = next(csv.DictReader(handle, delimiter="\t"))
    assert float(smoke_row["qps_sample_std"]) == 0.0


def verify_prepare_audit(program: str, directory: Path) -> None:
    cache = directory / "CMakeCache.txt"
    cache.write_text(
        "\n".join(
            (
                "CMAKE_BUILD_TYPE:STRING=Release",
                "BUILD_IVF_RABITQ_CPU_EXPORTER:BOOL=ON",
                "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=89-real",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_program(program, str(cache))
    assert result.returncode == 0, result.stderr


def verify_background_error_isolation(runner: Path, directory: Path) -> None:
    marker = directory / "error_handler_calls.txt"
    run_root = directory / "run"
    result = subprocess.run(
        [
            "bash",
            "-c",
            r'''
source "$1" help >/dev/null
marker="$2"
RUN_ROOT="$3"
mkdir -p "$RUN_ROOT"
BATCH_ROOT="$RUN_ROOT/batch"
DATASET=wiki35m
CURRENT_STAGE=monitoring
cleanup_monitors() { :; }
write_status() { printf 'status\n' >>"$marker"; }
update_schedule() { printf 'schedule\n' >>"$marker"; }
generate_batch_summary() { :; }
(false) &
monitor_pid=$!
wait "$monitor_pid" || true
[[ ! -s "$marker" ]]
''',
            "background-error-isolation",
            str(runner),
            str(marker),
            str(run_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        "a background monitor error invoked the runner-wide error handler\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
        f"calls:\n{marker.read_text() if marker.exists() else ''}"
    )


def main() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    script = extract_script(source)
    verify_documented_extractor(source, script)
    verify_static_contract(source, script)

    parsed = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    assert parsed.returncode == 0, parsed.stderr
    programs = embedded_python(script)
    for tag, program in programs.items():
        compile(program, f"{RUNBOOK.name}:{tag}", "exec")

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        runner = directory / "runner.sh"
        runner.write_text(script, encoding="utf-8")
        result = subprocess.run(
            ["bash", str(runner), "help"], text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
        for mode in ("prepare", "smoke", "formal", "wiki35m"):
            assert mode in result.stdout

        verify_background_error_isolation(runner, directory)
        selected = verify_discovery_merge(programs["PY_DISCOVERY_MERGE"], directory)
        verify_search_summary(programs["PY_SEARCH_SUMMARY"], directory, selected)
        verify_prepare_audit(programs["PY_PREPARE_AUDIT"], directory)


if __name__ == "__main__":
    main()
