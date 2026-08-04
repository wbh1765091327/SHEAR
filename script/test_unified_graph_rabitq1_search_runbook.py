#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/runbooks/unified_graph_rabitq1_search_four_dataset.md"
BEGIN = "<!-- BEGIN RUNBOOK_SCRIPT -->"
END = "<!-- END RUNBOOK_SCRIPT -->"


def extract_script(source: str) -> str:
    start = source.index(BEGIN) + len(BEGIN)
    stop = source.index(END, start)
    match = re.search(r"```bash\n(?P<body>.*?)\n```", source[start:stop], re.S)
    assert match, "missing fenced Bash payload"
    return match.group("body") + "\n"


def function_body(script: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n", script, re.M)
    assert match, f"missing function {name}"
    following = re.search(
        r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{\n", script[match.end() :], re.M
    )
    stop = match.end() + following.start() if following else len(script)
    return script[match.start() : stop]


def verify_static_contract(source: str, script: str) -> None:
    for mode in (
        "help",
        "prepare",
        "smoke",
        "build-sidecars",
        "formal",
        "wiki35m",
        "laion100m",
        "msmarco70m",
        "deep100m",
    ):
        assert f"{mode})" in script, mode

    contracts = {
        "wiki35m": ("wiki-1024-35M/en_base.fbin", "wiki-1024-35M/query.fbin", "wiki-1024-35M/gt100.bin", 35_000_000, 1024),
        "laion100m": ("laion-512-100M/base.fbin", "laion-512-100M/query.fbin", "laion-512-100M/gt100.bin", 100_000_000, 512),
        "msmarco70m": ("msmarco-768-70M/base.fbin", "msmarco-768-114M/query.fbin", "msmarco-768-70M/gt100.bin", 70_000_000, 768),
        "deep100m": ("deep-96-100M/base.100M.fbin", "deep-96-100M/query.public.10K.fbin", "deep-96-100M/groundtruth.public.10K.ibin", 100_000_000, 96),
    }
    loader = function_body(script, "load_dataset")
    for dataset, values in contracts.items():
        branch = re.search(rf"{dataset}\).*?;;", loader, re.S)
        assert branch, dataset
        body = branch.group(0)
        for value in values[:3]:
            assert value in body
        assert f"ROWS={values[3]}" in body
        assert f"DIM={values[4]}" in body

    formal = function_body(script, "run_formal_dataset")
    assert "for method in shear cagra_nn_descent tagore_cagra tagore_vamana; do" in formal
    assert "for backend in rabitq1-post-rerank rabitq1-adaptive-exact; do" in formal
    assert "for l in 25 50 100 200 400; do" in formal
    assert formal.count('run_search_once "$method" "$backend" "$l" warmup 0') == 1
    assert "for rep in 1 2; do" in formal
    assert formal.count('run_search_once "$method" "$backend" "$l" measured "$rep"') == 1

    for fragment in (
        "--recall-at 10",
        "--num-threads 8",
        "--storage mmap",
        "--rabitq1-sidecar",
        "--search-backend",
        "/data_staging/wbh/unified_graph_rabitq1/sidecars",
        "/data_staging/wbh/unified_graph_rabitq1/runs",
        "dataset\\tmethod\\tindex_path_prefix",
        "prior_result_audit",
        "provenance",
        "prepare_search_cache",
        "validate_repetitions",
        "accepted",
        "flock",
        "trap 'on_error",
        "sha256sum",
    ):
        assert fragment in script, fragment
    assert "build_shear" not in script
    assert "rm -f -- \"$INDEX_FILE\"" not in script


def write_fake_search(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >>"$FAKE_LOG"
while (($#)); do
  case "$1" in
    --result-path) result="$2"; shift 2 ;;
    --metrics-path) metrics="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ "${FAKE_FAIL:-0}" == 0 ]] || exit 9
mkdir -p "$(dirname "$result")" "$(dirname "$metrics")"
payload=stable
[[ "${FAKE_MISMATCH:-0}" == 0 || "$result" != *rep_2* ]] || payload=different
printf '%s\n' "$payload" >"$result"
printf 'storage\tqueries\trecall_at\tsearch_list_size\trecall\tsearch_seconds\tqps\tavg_visited_nodes\tp95_visited_nodes\tavg_expanded_nodes\tp95_expanded_nodes\tavg_distance_comparisons\tp95_distance_comparisons\tearly_stopped_queries\trecall_per_1k_distance_comparisons\tsearch_backend\tsidecar_load_seconds\tsidecar_resident_bytes\ttotal_one_bit_evaluations\ttotal_lower_bound_rejects\ttotal_navigation_exact_evaluations\ttotal_adaptive_exact_refinements\ttotal_post_rerank_exact_evaluations\ttotal_exact_evaluations\tlogical_exact_vector_bytes\tquery_preprocess_seconds\tone_bit_estimate_seconds\texact_l2_seconds\tnavigation_seconds\tgraph_encoding\tgraph_bytes\n' >"$metrics"
printf 'mmap\t2\t10\t25\t0.9\t1\t2\t1\t1\t1\t1\t1\t1\t0\t1\trabitq1-post-rerank\t0\t1\t2\t0\t0\t0\t50\t50\t12800\t0\t0\t0\t0\tfake\t32\n' >>"$metrics"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_fake_fixture(root: Path) -> tuple[Path, Path, Path]:
    data = root / "data"
    staging = root / "staging"
    graphs = root / "graphs"
    data.mkdir()
    staging.mkdir()
    graphs.mkdir()
    paths = (
        "wiki-1024-35M/en_base.fbin", "wiki-1024-35M/query.fbin", "wiki-1024-35M/gt100.bin",
        "laion-512-100M/base.fbin", "laion-512-100M/query.fbin", "laion-512-100M/gt100.bin",
        "msmarco-768-70M/base.fbin", "msmarco-768-114M/query.fbin", "msmarco-768-70M/gt100.bin",
        "deep-96-100M/base.100M.fbin", "deep-96-100M/query.public.10K.fbin", "deep-96-100M/groundtruth.public.10K.ibin",
    )
    for relative in paths:
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    manifest = root / "graphs.tsv"
    rows = ["dataset\tmethod\tindex_path_prefix"]
    for dataset in ("wiki35m", "laion100m", "msmarco70m", "deep100m"):
        for method in ("shear", "cagra_nn_descent", "tagore_cagra", "tagore_vamana"):
            prefix = graphs / dataset / method
            prefix.mkdir(parents=True)
            (prefix / "_mem.index").write_bytes(b"fake graph")
            rows.append(f"{dataset}\t{method}\t{prefix}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    sidecars = staging / "sidecars"
    sidecars.mkdir(parents=True)
    for dataset in ("wiki35m", "laion100m", "msmarco70m", "deep100m"):
        (sidecars / f"{dataset}.rabitq1").write_bytes(b"fake sidecar")
    return data, staging, manifest


def verify_behavior(script: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runner = root / "runner.sh"
        runner.write_text(script, encoding="utf-8")
        runner.chmod(0o755)
        data, staging, manifest = write_fake_fixture(root)
        fake_search = root / "search_shear_index"
        write_fake_search(fake_search)
        fake_builder = root / "build_graph_rabitq_1bit"
        fake_builder.write_text("#!/usr/bin/env bash\nexit \"${FAKE_BUILDER_STATUS:-0}\"\n")
        fake_builder.chmod(0o755)
        log = root / "calls.log"
        environment = {
            "DATA_ROOT": str(data),
            "EVAL_ROOT": str(staging),
            "GRAPH_MANIFEST": str(manifest),
            "SEARCH_BIN": str(fake_search),
            "SIDECAR_BUILDER": str(fake_builder),
            "FAKE_LOG": str(log),
            "SKIP_GRAPH_BINARY_VALIDATION": "1",
            "RUN_ID": "contract",
            "PATH": "/usr/bin:/bin",
        }

        result = subprocess.run(
            [str(runner), "wiki35m"], env=environment, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        calls = log.read_text().splitlines()
        assert sum("rep_1" in call or "rep_2" in call for call in calls) == 80
        assert sum("warmup" in call for call in calls) == 40

        mismatch_env = environment | {"RUN_ID": "mismatch", "FAKE_MISMATCH": "1"}
        result = subprocess.run(
            [str(runner), "wiki35m"], env=mismatch_env, text=True, capture_output=True
        )
        assert result.returncode != 0
        assert "result hash" in result.stderr.lower()

        missing = root / "missing.tsv"
        missing.write_text("\n".join(manifest.read_text().splitlines()[:-1]) + "\n")
        before = len(log.read_text().splitlines())
        result = subprocess.run(
            [str(runner), "wiki35m"],
            env=environment | {"RUN_ID": "missing", "GRAPH_MANIFEST": str(missing)},
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert len(log.read_text().splitlines()) == before

        duplicate = root / "duplicate.tsv"
        duplicate.write_text(manifest.read_text() + manifest.read_text().splitlines()[1] + "\n")
        result = subprocess.run(
            [str(runner), "wiki35m"],
            env=environment | {"RUN_ID": "duplicate", "GRAPH_MANIFEST": str(duplicate)},
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0

        failed_env = environment | {"RUN_ID": "failed", "FAKE_FAIL": "1"}
        result = subprocess.run(
            [str(runner), "wiki35m"], env=failed_env, text=True, capture_output=True
        )
        assert result.returncode != 0
        assert not (staging / "runs" / "failed" / "accepted.tsv").exists()

        accepted = staging / "runs" / "prior"
        accepted.mkdir(parents=True)
        (accepted / "status").write_text("accepted\n")
        result = subprocess.run(
            [str(runner), "wiki35m"],
            env=environment | {"RUN_ID": "prior"},
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "prior accepted" in result.stderr.lower()

        result = subprocess.run(
            [str(runner), "build-sidecars"],
            env=environment | {"RUN_ID": "invalid-sidecar", "FAKE_BUILDER_STATUS": "7"},
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert (staging / "sidecars" / "wiki35m.rabitq1").read_bytes() == b"fake sidecar"


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
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    verify_behavior(script)
    print("unified graph RaBitQ1 runbook tests passed")


if __name__ == "__main__":
    main()
