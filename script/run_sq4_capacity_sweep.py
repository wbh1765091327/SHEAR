#!/usr/bin/env python3
"""Prepare (default), validate on 10M, or explicitly execute the SQ4 100M sweep."""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import run_sq4_plus as recipe
from sq4_sweep_metrics import (BASELINE, BUDGETS, THRESHOLDS, near_noise, parse_build,
                               save, select_finalists, summarize, write_report)
from sq4_sweep_runtime import (Nvml, audit_graph, check_inputs, cleanup_owned, cleanup_search_intermediates, failure_kind,
                               identity, prewarm, quality, run_measured)

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILES = ("run_sq4_plus.py", "run_sq4_capacity_sweep.py", "sq4_sweep_metrics.py", "sq4_sweep_runtime.py")
FROZEN = ROOT / "runs/sq4_budget_threshold_20261108"


def version():
    return {name: recipe.digest(ROOT / "script" / name) for name in VERSION_FILES}


def cache_identity(directory):
    return {str(p.relative_to(directory)): recipe.digest(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def config_id(budget, high):
    return f"b{budget:02}-h{high:02}"


def matrix():
    cells = [dict(config=config_id(b, h), budget=b / 100, high=h / 100)
             for b in BUDGETS for h in THRESHOLDS if b <= h]
    first = next(c for c in cells if c["config"] == BASELINE)
    remaining = [c for c in cells if c != first]
    random.Random(42).shuffle(remaining)
    return [first, *remaining]


def passport_for(args):
    if args.input_manifest:
        return json.loads(args.input_manifest.read_text())
    if args.validate_10m:
        base = json.loads((FROZEN / "standard-B1/cache.json").read_text())
        q = json.loads((FROZEN / "quality/standard-B1/inputs.json").read_text())
        return dict(base=dict(identity=base["identity"], sha256=base["sha256"]),
                    query=q["query"], gt10m=q["gt10m"])
    return json.loads((ROOT / "runs/sq4_random_leaders_100m_20261108/inputs.json").read_text())


def validation_jobs():
    baseline = matrix()[0]
    # top6 of six leaders forces six 10M-row parents above the 50% limit.
    trigger = dict(config="trigger-fixed6", budget=0.08, high=0.50, routing_leaders=6)
    return [(f"validation-{name}", baseline, name.startswith("A")) for name in ("A1", "B1", "B2", "A2")] + [
        (f"trigger-{name}", trigger, name.startswith("A")) for name in ("A1", "B1")]


def load_legacy(path):
    spec = importlib.util.spec_from_file_location("sq4_legacy_entry", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return legacy


def manifest_for(args, cell, run, base, binary, scratch, legacy=None):
    profile = "laion10m" if args.validate_10m else "laion100m"
    if legacy:
        record, env = legacy.make_manifest(base, binary, scratch, profile=profile, host_gib=256,
                                           routing_leaders=cell.get("routing_leaders"))
        for kind, value in (("target", cell["budget"]), ("low", cell["budget"]), ("high", cell["high"])):
            flag = "--gpu-subtree-d0-capacity-" + kind + "-ratio"
            record["command"][record["command"].index(flag) + 1] = str(value)
    else:
        record, env = recipe.make_manifest(base, binary, scratch, profile=profile, host_gib=256,
            fanout_top=6, sample_fraction=0.002, planning_budget_ratio=cell["budget"],
            admission_threshold_ratio=cell["high"], routing_leaders=cell.get("routing_leaders"))
    cmd = record["command"]
    cmd[cmd.index("--oom-profile-json") + 1] = str(args.run_dir / run / "profile.json")
    env.update(CUDA_CACHE_PATH=str(scratch / "cuda-cache"), TMPDIR=str(scratch / "tmp"))
    # Host-side diagnostics expose existing CUDA-event timers, not new device instrumentation.
    env["SHEAR_PIPNN_D1_CHILD_DIAGNOSTICS"] = "1"
    record["environment"] = {k: v for k, v in env.items()
                             if k.startswith(("SHEAR_", "CUDA_")) or k.endswith("NUM_THREADS")
                             or k in ("PATH", "LD_LIBRARY_PATH", "TMPDIR")}
    record["experiment"] = dict(run=run, **cell, seed=42, source="capacity sweep override",
                                legacy_entry=legacy is not None, source_version=version())
    return record, env


def make_plan(args, inputs):
    commands = []
    jobs = validation_jobs() if args.validate_10m else [
        (f"screen-{i:02}-{c['config']}", c, False) for i, c in enumerate(matrix())]
    legacy = load_legacy(args.legacy_runner) if args.validate_10m else None
    for run, cell, old in jobs:
        record, _ = manifest_for(args, cell, run, inputs["base"]["path"], args.binary,
                                 args.scratch_root / args.run_dir.name / run, legacy if old else None)
        commands.append(record)
    return dict(schema_version=1, mode="10m-validation" if args.validate_10m else "100m-sweep",
                execution_started=False, inputs=inputs, source_version=version(),
                fixed=dict(fanout=[6, 3, 1], sample=0.002, seed=42, primary_centers=64,
                           host_gib=256, workspace_gib=16, threads=8, blas_threads=1,
                           low_follows_target=True, memory_sampling_ms=100),
                screening=commands, confirmation_order=[] if args.validate_10m else ["A", "B", "C", "C", "B", "A"],
                planned_parameter_builds=6 if args.validate_10m else 19,
                optional_noise_confirmation=None if args.validate_10m else "one ABBA per close finalist",
                reference=None if args.validate_10m else "one separately accounted frozen SQ8 build and quality evaluation",
                cleanup="all successful graphs and intermediates after audit and Recall-QPS; retain evidence",
                default_changes=False)


def plan_contract(plan):
    """Preparation-time shell paths are not an executed experiment environment."""
    contract = copy.deepcopy(plan)
    for record in contract["screening"]:
        record.pop("environment", None)
        record.pop("replaced_environment", None)
    return contract


def verify_gate(path, binary_sha256, search_sha256):
    gate = json.loads(Path(path).read_text())
    if (not gate.get("accepted") or gate.get("rows") != 10_000_000
            or gate.get("build_sha256") != binary_sha256 or gate.get("search_sha256") != search_sha256
            or gate.get("source_version") != version()):
        raise ValueError("10M gate is missing, failed, or belongs to different scripts/binaries")
    evidence = Path(path).parent
    for run in gate["runs"]:
        result = json.loads((evidence / run / "result.json").read_text())
        clean = json.loads((evidence / run / "cleanup.json").read_text())
        if result["status"] != "success" or clean["state"] != "complete":
            raise ValueError("10M gate references incomplete measurements or cleanup")


class Experiment:
    def __init__(self, args, inputs):
        self.args, self.inputs, self.root = args, inputs, args.run_dir
        if os.environ.get("CUDA_VISIBLE_DEVICES", "0") not in ("0", ""):
            raise ValueError("this protocol measures physical GPU 0; conflicting CUDA_VISIBLE_DEVICES")
        self.root.mkdir(parents=True, exist_ok=True)
        self.results = []
        requested_binaries = dict(build=args.binary, search=args.search_binary, audit=args.audit_binary)
        library_paths = {key: os.environ.get(key) for key in ("LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_ROOT")}
        state_file = self.root / "execution.json"
        if state_file.exists():
            if not args.resume:
                raise ValueError("execution already exists; use --resume to preserve completed runs")
            state = json.loads(state_file.read_text())
            if (state["source_version"] != version() or state["inputs"] != inputs
                    or state["library_paths"] != library_paths):
                raise ValueError("resume cannot change source version or input identities")
            self.token, self.work = state["token"], Path(state["scratch"])
            self.binaries = {k: Path(v["path"]) for k, v in state["binaries"].items()}
            for k, v in state["binaries"].items():
                if recipe.digest(v["path"]) != v["sha256"] or recipe.digest(requested_binaries[k]) != v["sha256"]:
                    raise ValueError(f"frozen binary changed: {k}")
            if cache_identity(self.root / "cuda-seed") != json.loads((self.root / "cache-seed-hashes.json").read_text()):
                raise ValueError("frozen CUDA cache changed")
            for path in sorted(self.root.glob("*/result.json")):
                result = json.loads(path.read_text())
                if result.get("run"):
                    self.results.append(result)
        else:
            self.token = uuid.uuid4().hex
            args.scratch_root.mkdir(parents=True, exist_ok=True)
            self.work = Path(tempfile.mkdtemp(prefix="sq4-capacity-", dir=args.scratch_root))
            (self.root / "bin").mkdir()
            self.binaries = {}
            for key, source in requested_binaries.items():
                destination = self.root / "bin" / key
                shutil.copy2(source, destination)
                if recipe.digest(source) != recipe.digest(destination):
                    raise ValueError("binary freeze mismatch")
                self.binaries[key] = destination
            (self.root / "source").mkdir()
            for name in VERSION_FILES:
                shutil.copy2(ROOT / "script" / name, self.root / "source" / name)
            git = shutil.which("git") or "/data/profwang/toolchains/sysroot/usr/bin/git"
            with (self.root / "workspace.patch").open("w") as patch:
                subprocess.run([git, "-C", str(ROOT), "diff", "--binary", "HEAD"], stdout=patch, check=True)
            head = subprocess.check_output([git, "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            shutil.copytree(args.cuda_seed, self.root / "cuda-seed")
            save(self.root / "cache-seed-hashes.json", cache_identity(self.root / "cuda-seed"))
            save(self.root / "environment.json", dict(
                uname=list(os.uname()), gpu=subprocess.check_output(["nvidia-smi", "-q"], text=True),
                meminfo=Path("/proc/meminfo").read_text(), cpuinfo=Path("/proc/cpuinfo").read_text(),
                source_head=head, note="Frozen executable hashes identify measured code; source snapshot includes existing local changes."))
            state = dict(token=self.token, scratch=str(self.work), source_version=version(), inputs=inputs,
                         library_paths=library_paths,
                         binaries={k: dict(path=str(v), sha256=recipe.digest(v)) for k, v in self.binaries.items()})
            save(state_file, state)
        self.work.mkdir(exist_ok=True)
        self.nvml = Nvml()
        self.nvml.require_idle()

    def status(self, phase, **extra):
        record = dict(phase=phase, utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), **extra)
        save(self.root / "status.json", record)
        print(json.dumps(record), flush=True)

    def existing(self, run):
        found = next((r for r in self.results if r["run"] == run), None)
        if found and found["status"] == "success":
            cleanup = self.root / run / "cleanup.json"
            if not cleanup.exists() or json.loads(cleanup.read_text())["state"] != "complete":
                scratch = self.work / run
                if scratch.exists():
                    cleanup_owned(scratch, self.work, self.token, self.root / run)
                else:
                    # A crash may occur after rmtree but before committing the cleanup record.
                    if not cleanup.exists() or json.loads(cleanup.read_text())["state"] != "verified":
                        raise ValueError("missing graph without a verified cleanup record")
                    cleanup_search_intermediates(self.root / run)
                    save(cleanup, dict(state="complete", scratch=str(scratch), recovered_after_removal=True,
                                       graph_sha256=found["graph_sha256"], graphs_retained=False))
        return found

    def run(self, run, cell, legacy=None, reference=None):
        found = self.existing(run)
        if found:
            return found
        d, scratch = self.root / run, self.work / run
        if d.exists() or scratch.exists():
            raise ValueError(f"interrupted run retained at {d}; inspect it before resuming, never overwrite it")
        self.nvml.require_idle()
        if shutil.disk_usage(self.work).free < 8 * self.inputs["base"]["shape"][0] * 64:
            raise RuntimeError("scratch filesystem lacks conservative output/temp headroom; retained failures were not deleted")
        for entry in self.inputs.values():
            if identity(entry["path"]) != entry["identity"]:
                raise ValueError("input changed after checksum verification")
        d.mkdir()
        scratch.mkdir()
        (scratch / "tmp").mkdir()
        save(scratch / ".sweep-owner.json", dict(token=self.token, run=run))
        shutil.copytree(self.root / "cuda-seed", scratch / "cuda-cache")
        result = dict(run=run, config=cell["config"], budget=cell["budget"], high=cell["high"],
                      routing_leaders=cell.get("routing_leaders"))
        phase = "prewarm"
        try:
            self.status(phase, run=run)
            prewarm([Path(self.inputs["base"]["path"])], d / "cache.json")
            if reference:
                record, env = self.reference_manifest(reference, run, scratch)
            else:
                record, env = manifest_for(self.args, cell, run, self.inputs["base"]["path"],
                                           self.binaries["build"], scratch, legacy)
            save(d / "manifest.json", record)
            save(d / "env.json", record["environment"])
            phase = "build"
            self.status(phase, run=run)
            timing = run_measured(record["command"], env, d, self.nvml, ROOT)
            log = (d / "build.log").read_text()
            p = json.loads((d / "profile.json").read_text())
            phase = "config_audit"
            if reference:
                audit, metrics = self.reference_metrics(p, log, timing, d)
            else:
                audit = recipe.audit_profile(record, p, log)
                g = p["gpu_subtree"]
                if (g["sidecar_preload_workers"] != 1 or g["sidecar_queue_max_children"] != 1
                        or "per_subtree_gather=true" not in log or g["persistent_async_d2h_enabled"]
                        or p["gpu_subtree_io"]["host_point_read_bytes"] != 0
                        or g["workspace_peak_bytes"] > g["workspace_budget_bytes"]):
                    raise ValueError("streaming, raw read, or workspace execution drift")
                resources = [json.loads(line) for line in (d / "resources.jsonl").read_text().splitlines()]
                metrics, tasks, splits, roots = parse_build(p, log, timing, resources)
                save(d / "subtrees.json", tasks)
                save(d / "splits.json", splits)
                save(d / "root-sizes.json", roots)
                self.check_d0(cell, metrics)
            save(d / "config-audit.json", audit)
            save(d / "metrics.json", metrics)
            phase = "graph_audit"
            self.status(phase, run=run)
            graph = scratch / "index/_mem.index"
            result.update(metrics=metrics, graph_audit=audit_graph(self.binaries["audit"], graph,
                self.inputs["base"]["shape"][0], d / "graph-audit.txt"),
                graph_sha256=recipe.digest(graph), graph_bytes=graph.stat().st_size)
            phase = "quality"
            self.status(phase, run=run)
            result["quality"] = quality(self.binaries["search"], self.inputs, graph, d / "quality", env)
            result["status"] = "success"
            save(d / "result.json", result)
            phase = "cleanup"
            cleanup_owned(scratch, self.work, self.token, d)
        except Exception as error:
            log = (d / "build.log").read_text() if (d / "build.log").exists() else ""
            failure = dict(status="failed", phase=phase, error=str(error), kind=failure_kind(error, log),
                           scratch_retained=str(scratch), last_log_lines=log.splitlines()[-30:])
            # Do not overwrite a complete measurement if only cleanup failed.
            if phase == "cleanup":
                save(d / "cleanup-error.json", failure)
                raise
            result.update(failure)
            save(d / "result.json", result)
        self.results.append(result)
        write_report(self.root, self.results)
        self.status(result["status"], run=run, kind=result.get("kind"))
        # No concurrent process is allowed to silently take over GPU 0.
        self.nvml.require_idle()
        return result

    def check_d0(self, cell, metrics):
        for previous in self.results:
            if previous["status"] != "success" or previous["config"] == "sq8-reference":
                continue
            p = previous["metrics"]
            if p["primary_hash"] != metrics["primary_hash"]:
                raise ValueError("encoding centers changed across the parameter matrix")
            if (previous["budget"] == cell["budget"] and previous.get("routing_leaders") == cell.get("routing_leaders")
                    and any(p[k] != metrics[k] for k in
                    ("d0_partition_hash", "d0_task_hash", "leaders"))):
                raise ValueError("fixed-budget D0 changed with threshold or repetition")

    def reference_manifest(self, reference, run, scratch):
        original = json.loads((reference / "cmd.json").read_text())
        binary = json.loads((reference / "binary.json").read_text())
        if recipe.digest(binary["path"]) != binary["sha256"]:
            raise ValueError("ordinary SQ8 frozen binary hash mismatch")
        destination = self.root / "bin/sq8-reference"
        if not destination.exists():
            shutil.copy2(binary["path"], destination)
        if recipe.digest(destination) != binary["sha256"]:
            raise ValueError("ordinary SQ8 archived binary changed")
        command = [str(destination), *original[original.index(binary["path"]) + 1:]]
        replace = {"--data-path": self.inputs["base"]["path"], "--index-path-prefix": scratch / "index",
                   "--oom-temp-dir": scratch / "tmp", "--oom-profile-json": self.root / run / "profile.json",
                   "--gpu-subtree-sq-wmma-sidecar-prefix": scratch / "sidecar/sq"}
        for flag, value in replace.items():
            command[command.index(flag) + 1] = str(value)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SHEAR_")}
        env.update(json.loads((reference / "env.json").read_text()))
        env.update(CUDA_CACHE_PATH=str(scratch / "cuda-cache"), TMPDIR=str(scratch / "tmp"))
        env.update(RAYON_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if "SHEAR_PIPNN_PERSISTENT_LEAF_HISTOGRAM_OUT" in env:
            env["SHEAR_PIPNN_PERSISTENT_LEAF_HISTOGRAM_OUT"] = str(self.root / run / "leaves.jsonl")
        record = dict(command=command, environment={k: v for k, v in env.items()
                      if k.startswith(("SHEAR_", "CUDA_")) or k.endswith("NUM_THREADS")
                      or k in ("PATH", "LD_LIBRARY_PATH", "TMPDIR")}, binary=binary,
                      source=str(reference), role="separate ordinary SQ8 system reference")
        return record, env

    def reference_metrics(self, p, log, timing, directory):
        g = p["gpu_subtree"]
        s = g["sq_gpu_staging"]
        if (s["mode"] != "gpu-d0" or s["rows"] != 100_000_000
                or not s.get("sidecar_source") == "gpu-d0-resident"
                or s["d0_assign_rows"] != 100_000_000 or not s["gpu_sq_quantize_ms"]
                or g["sq_wmma_sidecar_file_rows"] != 0 or g["sq_wmma_sidecar_preload_ms"] != 0
                or g["cpu_fallbacks"] or g["persistent_fallback_count"] or g["persistent_queue_overflow_flags"]):
            raise ValueError("ordinary SQ8 GPU-D0/resident/no-fallback baseline drift")
        resources = [json.loads(line)["gpu"] for line in (directory / "resources.jsonl").read_text().splitlines()]
        metrics = dict(wall_s=timing[0], rss_gib=timing[1] / 1048576,
                       physical_read_bytes=int(timing[2] * 512), d0_s=g["d0_wall_ms"] / 1000,
                       d1_s=g["d1plus_producer_ms"] / 1000, gpu_s=g["persistent_kernel_event_ms"] / 1000,
                       gpu_process_peak_gib=max(x["process_used_bytes"] for x in resources) / (1 << 30),
                       gpu_min_free_gib=min(x["free_bytes"] for x in resources) / (1 << 30),
                       memory_margin_ok=min(x["free_bytes"] for x in resources) >= 1 << 30)
        return dict(accepted=True, role="ordinary SQ8 system reference", staging=s), metrics

    def validate(self):
        legacy = load_legacy(self.args.legacy_runner)
        shutil.copy2(self.args.legacy_runner, self.root / "source/legacy_run_sq4_plus.py")
        runs = []
        for run, cell, old in validation_jobs():
            r = self.run(run, cell, legacy if old else None)
            runs.append(r)
            if r["status"] != "success":
                raise ValueError("10M validation failed; 100M remains gated")
            if run.startswith("trigger-") and not r["metrics"]["split_count"]:
                raise ValueError("10M trigger failed to validate actual lazy splitting")
        gate = dict(accepted=True, rows=10_000_000, source_version=version(),
                    build_sha256=recipe.digest(self.binaries["build"]),
                    search_sha256=recipe.digest(self.binaries["search"]), runs=[r["run"] for r in runs],
                    comparisons="legacy/new A-B-B-A at 8/50; matched fixed-six-leader split trigger A-B",
                    algorithm_changed=False)
        save(self.root / "validation.json", gate)
        write_report(self.root, self.results, plot=True)
        self.status("10m_validated_100m_not_started")

    def sweep(self):
        cells = {c["config"]: c for c in matrix()}
        for i, cell in enumerate(matrix()):
            r = self.run(f"screen-{i:02}-{cell['config']}", cell)
            if i == 0 and r["status"] != "success":
                raise ValueError("baseline failed; cannot rank the remaining matrix")
            if r.get("kind") in ("storage_exhausted", "host_allocation_failed", "killed_unknown"):
                raise ValueError("infrastructure failure retained; stop before another expensive build")
        selected_path = self.root / "finalists.json"
        if selected_path.exists():
            selected = json.loads(selected_path.read_text())
        else:
            selected = select_finalists([r for r in self.results if r["run"].startswith("screen-")])
            save(selected_path, selected)
        if len(selected) < 2:
            self.status("insufficient_valid_finalists", finalists=selected)
            write_report(self.root, self.results, plot=True)
            return
        arms = {"A": BASELINE, "B": selected[0], "C": selected[1]}
        for i, arm in enumerate(("A", "B", "C", "C", "B", "A")):
            self.run(f"confirm-{i:02}-{arm}", cells[arms[arm]])
        noise_path = self.root / "noise-confirmations.json"
        if noise_path.exists():
            noise = json.loads(noise_path.read_text())
        else:
            summary = summarize(self.results)
            a = summary[BASELINE]["metrics"]["wall_s"]["raw"]
            noise = [c for c in selected if not summary[c]["failures"] and summary[c]["memory_margin_ok"]
                     and near_noise(a, summary[c]["metrics"]["wall_s"]["raw"])]
            save(noise_path, noise)
        for c in noise:
            for i, config in enumerate((BASELINE, c, c, BASELINE)):
                self.run(f"noise-{c}-{i:02}", cells[config])
        self.run("sq8-reference", dict(config="sq8-reference", budget=None, high=None),
                 reference=self.args.sq8_reference_run)
        write_report(self.root, self.results, plot=True)
        self.status("complete_with_failures" if any(r["status"] != "success" for r in self.results) else "complete",
                    successful=sum(r["status"] == "success" for r in self.results),
                    failed=sum(r["status"] != "success" for r in self.results),
                    defaults_changed=False, successful_graphs_retained=False,
                    failure_scratch=[r["scratch_retained"] for r in self.results if r["status"] != "success"])


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="explicitly execute the 100M protocol, after a matching 10M gate")
    modes.add_argument("--validate-10m", action="store_true", help="execute only the six 10M gate builds and quality checks")
    modes.add_argument("--report-only", action="store_true", help="regenerate tables/plots without launching builds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--skip-10m-gate-reason", help="record an explicit user-authorized waiver of the new 10M gate")
    parser.add_argument("--binary", type=Path, default=FROZEN / "bin/B")
    parser.add_argument("--search-binary", type=Path, default=FROZEN / "bin/search_shear_index")
    parser.add_argument("--audit-binary", type=Path, default=FROZEN / "bin/audit")
    parser.add_argument("--cuda-seed", type=Path, default=FROZEN / "cuda-seed")
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--legacy-runner", type=Path, default=FROZEN / "source/script/run_sq4_plus.py")
    parser.add_argument("--sq8-reference-run", type=Path,
                        default=ROOT / "runs/sq4_sq8_laion100m_incremental_20261107/S2")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    return args


def main(argv=None):
    args = arguments(argv)
    if args.report_only:
        results = [json.loads(p.read_text()) for p in sorted(args.run_dir.glob("*/result.json"))]
        write_report(args.run_dir, results, plot=True)
        return
    rows = 10_000_000 if args.validate_10m else 100_000_000
    inputs = check_inputs(passport_for(args), rows, hash_files=False)
    plan = make_plan(args, inputs)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.run_dir / "plan.json"
    if plan_path.exists() and plan_contract(json.loads(plan_path.read_text())) != plan_contract(plan):
        raise ValueError("existing plan differs; use a new run directory")
    save(plan_path, plan)
    if not (args.execute or args.validate_10m):
        print(json.dumps(dict(plan=str(plan_path), configurations=13, planned_parameter_builds=19,
                              execution_started=False), indent=2))
        return
    # Validate all dependencies and the gate before hashing/prewarming 204.8 GB.
    import matplotlib
    if not args.cuda_seed.is_dir():
        raise ValueError("frozen CUDA cache seed missing")
    for path in (args.binary, args.search_binary, args.audit_binary):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"executable missing: {path}")
    if args.execute:
        if not args.validation_report and not args.skip_10m_gate_reason:
            raise ValueError("--execute requires a matching --validation-report from --validate-10m")
        if args.skip_10m_gate_reason:
            save(args.run_dir / "10m-gate-waiver.json", dict(reason=args.skip_10m_gate_reason,
                 prior_evidence=str(ROOT / "runs/sq4_capacity_sweep_implementation_20261109/fixed-check.json"),
                 full_10m_gate_executed=False))
        else:
            verify_gate(args.validation_report, recipe.digest(args.binary), recipe.digest(args.search_binary))
        for name in ("cmd.json", "env.json", "binary.json"):
            if not (args.sq8_reference_run / name).is_file():
                raise ValueError("frozen ordinary SQ8 reference is incomplete")
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    with (args.scratch_root / ".sq4-capacity-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check = Nvml()
        try:
            check.require_idle()
        finally:
            check.close()
        inputs = check_inputs(passport_for(args), rows, hash_files=True)
        save(args.run_dir / "inputs.json", inputs)
        experiment = Experiment(args, inputs)
        try:
            if args.validate_10m:
                experiment.validate()
            else:
                experiment.sweep()
        except BaseException as error:
            experiment.status("interrupted_or_failed", error=str(error), scratch_retained=str(experiment.work))
            raise
        finally:
            experiment.nvml.close()


if __name__ == "__main__":
    main()
