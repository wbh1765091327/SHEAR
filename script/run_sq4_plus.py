#!/usr/bin/env python3
"""Run SQ4+ with an explicit, audited LAION configuration for each scale."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "laion10m": dict(rows=10_000_000, fanout=6, sample=0.002, host_gib=256,
                     candidates="1,2,4,6", leaf_wmma="0",
                     source="docs/runbooks/sq8_gpu_d0_baseline.md + SQ4+ 8% planning / 50% admission"),
    "laion100m": dict(rows=100_000_000, fanout=6, sample=0.002, host_gib=256,
                      candidates="1,2,4,6", leaf_wmma="1",
                      source="2026-11-09 SQ4+ verified optimizations; fanout6/3/1; sample0.002; target8%/high50%"),
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def effective_recipe(profile, fanout_top=None, sample_fraction=None):
    cfg = dict(PROFILES[profile])
    if fanout_top is not None:
        if isinstance(fanout_top, bool) or not isinstance(fanout_top, int) or not 1 <= fanout_top <= 32:
            raise ValueError("GPU-exact root fanout must be an integer in [1, 32]")
        cfg["fanout"] = fanout_top
        cfg["candidates"] = ",".join(map(str, sorted({x for x in (1, 2, 4, fanout_top) if x <= fanout_top})))
    if sample_fraction is not None:
        if not math.isfinite(sample_fraction) or not 0 < sample_fraction <= 1:
            raise ValueError("D0 sample fraction must be finite and in (0, 1]")
        cfg["sample"] = sample_fraction
    return cfg


def validate_ratios(planning_budget_ratio, admission_threshold_ratio):
    if not (math.isfinite(planning_budget_ratio) and math.isfinite(admission_threshold_ratio)
            and 0 < planning_budget_ratio <= admission_threshold_ratio <= 1):
        raise ValueError("capacity ratios require 0 < planning budget <= admission threshold <= 1")


def make_manifest(data, binary, run_dir, profile="auto", validation_prefix=False,
                  host_gib=None, inherited=None, routing_leaders=None, *,
                  fanout_top=None, sample_fraction=None, planning_budget_ratio=0.08,
                  admission_threshold_ratio=0.50):
    validate_ratios(planning_budget_ratio, admission_threshold_ratio)
    data, binary, run_dir = (Path(p).resolve() for p in (data, binary, run_dir))
    with data.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if dim != 512 or rows == 0 or data.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("expected a complete LAION 512D FP32 fbin")
    if profile == "auto":
        profile = next((k for k, v in PROFILES.items() if v["rows"] == rows), None)
        if profile is None:
            raise ValueError("auto requires exactly 10M or 100M rows; select a profile for prefix validation")
    cfg = effective_recipe(profile, fanout_top, sample_fraction)
    if rows != cfg["rows"] and not (validation_prefix and rows < cfg["rows"]):
        raise ValueError("dataset size does not match the selected profile")
    memory = cfg["host_gib"] if host_gib is None else host_gib
    if not math.isfinite(memory) or memory < 0:
        raise ValueError("host memory budget must be finite and nonnegative")
    values = {
        "--data-type": "float", "--dist-fn": "l2", "--data-path": data,
        "--index-path-prefix": run_dir / "index", "--oom-temp-dir": run_dir / "tmp",
        "--oom-profile-json": run_dir / "profile.json", "--oom-io-mode": "buffered",
        "--oom-memory-budget-gb": memory, "--max-degree": 64, "--lbuild": 100,
        "--num-threads": 8, "--m-hash-bits": 12, "--leaf-size-min": 256,
        "--leaf-size-max": 1280, "--leaf-knn": 2, "--psamp-fraction": 0.05,
        "--fanout-top": cfg["fanout"], "--fanout-second": 3,
        "--root-risk-candidate-rs": cfg["candidates"], "--seed": 42,
        "--kmeans-iterations": 4, "--partitioner": "gpu-subtree",
        "--gpu-subtree-root-assignment-backend": "cpu",
        "--gpu-subtree-assignment-backend": "persistent-wmma-tf32",
        "--gpu-subtree-leaf-topk-backend": "persistent-wmma-tf32",
        "--gpu-subtree-execution-mode": "fused-persistent",
        "--gpu-subtree-d0-capacity-high-ratio": admission_threshold_ratio,
        "--gpu-subtree-d0-capacity-target-ratio": planning_budget_ratio,
        "--gpu-subtree-d0-capacity-low-ratio": planning_budget_ratio,
        "--gpu-subtree-d0-tagore-sample-fraction": cfg["sample"],
        "--gpu-subtree-sq-wmma-mode": "build", "--gpu-subtree-sq-bits": 4,
        "--gpu-subtree-persistent-large-ticket-batch": 8,
        "--gpu-subtree-persistent-leaf-tile-rows": 128,
        "--gpu-subtree-sq8-leaf-mode": "auto",
        "--gpu-subtree-persistent-scheduler-mode": "auto",
        "--pipnn-backend": "cuda", "--pipnn-gpu-device": 0,
        "--pipnn-gpu-workspace-budget-gb": 16,
        "--hashprune-backfill-mode": "rhalf-soft-tier-q3",
        "--hashprune-soft-tier-priority-alpha": 1.02,
        "--hashprune-rhalf-max-backfill-per-row": 3,
    }
    command = [str(binary)]
    if routing_leaders is not None:
        if not 1 <= routing_leaders <= rows:
            raise ValueError("fixed routing leader count must be within the dataset")
        values["--gpu-subtree-d0-capacity-quality-min-root-leaders"] = routing_leaders
        values["--gpu-subtree-d0-capacity-root-leader-cap"] = routing_leaders
    for key, value in values.items():
        command.extend((key, str(value)))
    command += ["--oom-enable", "--gpu-subtree-d0-capacity-enable",
                "--gpu-subtree-sq-native-int4", "--gpu-subtree-sq4-plus",
                "--gpu-subtree-sq4-primary-center"]
    inherited = os.environ if inherited is None else inherited
    env = {k: v for k, v in inherited.items() if not k.startswith("SHEAR_")}
    env.update({
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "8", "SHEAR_PIPNN_PROFILE": "1",
        "SHEAR_PIPNN_PRUNE_BACKEND": "hashprune",
        "SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK": cfg["leaf_wmma"],
        "SHEAR_PIPNN_WMMA_CTA_SHARED_B": "0",
        "SHEAR_PIPNN_COMPACT_LUNE_WITNESS": "0",
        "SHEAR_PIPNN_COMPACT_LUNE_PRUNE": "0",
        "SHEAR_PIPNN_COMPACT_SPILL_DIGEST": "0",
        "SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS": "0",
        "SHEAR_PIPNN_SQ4_D0_BACKEND": "gpu-exact",
        "SHEAR_PIPNN_SQ4_D0_FUSED_SKETCH": "1",
        "SHEAR_PIPNN_SQ4_D0_SKETCH_BACKEND": "cpu",
        "SHEAR_PIPNN_SQ8_WMMA_OPT_MASK": "525824",
        "SHEAR_PIPNN_SQ4_PRIMARY_TRIANGULAR": "0",
        "SHEAR_PIPNN_SQ4_D1_BALANCED": "0",
        "SHEAR_PIPNN_SQ4_STREAM_GATHER": "1",
        "SHEAR_PIPNN_SQ4_D0_TRAIN": "cpu",
        "SHEAR_PIPNN_PERSISTENT_ASYNC_D2H": "0",
        "SHEAR_PIPNN_ASYNC_RESULT_SCATTER": "1",
    })
    record = dict(profile=profile, source=cfg["source"], validation_prefix=rows != cfg["rows"],
                  input=dict(path=str(data), rows=rows, dim=dim, bytes=data.stat().st_size),
                  binary=dict(path=str(binary), sha256=digest(binary)), command=command,
                  host_budget_gib=memory, resource_override=host_gib is not None,
                  routing_leaders=routing_leaders,
                  recipe_overrides=dict(fanout_top=fanout_top, sample_fraction=sample_fraction,
                                        planning_budget_ratio=planning_budget_ratio,
                                        admission_threshold_ratio=admission_threshold_ratio),
                  replaced_environment={k: v for k, v in inherited.items()
                                        if k.startswith("SHEAR_") and env.get(k) != v},
                  environment={k: v for k, v in env.items() if k.startswith(("SHEAR_", "CUDA_"))
                               or k.endswith("NUM_THREADS")})
    return record, env


def audit_profile(record, profile, log):
    command = record["command"]
    cfg = effective_recipe(record["profile"], int(command[command.index("--fanout-top") + 1]),
                           float(command[command.index("--gpu-subtree-d0-tagore-sample-fraction") + 1]))
    rows = record["input"]["rows"]
    g = profile["gpu_subtree"]
    c = g["d0_capacity"]
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require((profile["num_points"], profile["dim"]) == (rows, 512), "input shape drift")
    require(profile["pool_threads"] == 8, "thread drift")
    require(g["workspace_budget_bytes"] == 16 << 30, "GPU workspace drift")
    if record["host_budget_gib"]:
        require(profile["memory_budget_bytes"] == int(record["host_budget_gib"] * (1 << 30)), "host budget drift")
    require(c["enabled"] and g["d0_partition_mode"] == "budget-kmeans-topk", "fixed/disabled D0 planner")
    require(g["d1plus_lazy_split_mode"] == "on", "upload threshold was bypassed by disabling lazy admission")
    target_ratio = float(command[command.index("--gpu-subtree-d0-capacity-target-ratio") + 1])
    high_ratio = float(command[command.index("--gpu-subtree-d0-capacity-high-ratio") + 1])
    validate_ratios(target_ratio, high_ratio)
    require(float(command[command.index("--gpu-subtree-d0-capacity-low-ratio") + 1]) == target_ratio,
            "low ratio must follow planning target")
    require(command[command.index("--root-risk-candidate-rs") + 1] == cfg["candidates"],
            "root candidate set does not match fanout")
    threshold = max(1, int(c["c_gpu"] * target_ratio))
    high_threshold = max(1, int(c["c_gpu"] * high_ratio))
    if c.get("threshold_policy") == "byte-admission":
        require(0 < c["c_target"] <= threshold and c["c_low"] == c["c_target"]
                and c["c_target"] <= c["c_high"] <= high_threshold,
                "invalid byte-admission row thresholds")
        target_budget = int((16 << 30) * target_ratio)
        high_budget = int((16 << 30) * high_ratio)
        require(c["target_budget_bytes"] == target_budget and c["high_budget_bytes"] == high_budget,
                "planning budget or upload threshold drift")
        require(c["high_required_bytes"] <= high_budget and c["target_required_bytes"] <= target_budget,
                "row threshold exceeds byte budget")
        if high_ratio > target_ratio:
            require(c["c_high"] > c["c_target"] and c["high_required_bytes"] > target_budget,
                    "upload admission still uses the planning target")
        threshold = c["c_target"]
    else:
        require(c["c_high"] == high_threshold and c["c_target"] == c["c_low"] == threshold,
                "planning/admission capacity ratios drift")
    expected = (rows * cfg["fanout"] + threshold - 1) // threshold
    fixed = record.get("routing_leaders")
    floor, cap = (fixed, fixed) if fixed is not None else (64, 10000)
    leaders = min(rows, cap, max(floor, expected))
    require(c["k_capacity"] == expected and c["root_leader_cap"] == min(rows, cap), "leader formula/cap drift")
    require(c["quality_min_root_leaders"] == floor and c["root_leaders_initial"] == c["root_leaders_final"] == leaders,
            "leader floor/count drift")
    require(g["d0_tagore_k"] == cfg["fanout"], "root fanout drift")
    require(g["d0_tagore_sample_points"] == min(rows, max(leaders, math.ceil(rows * cfg["sample"]))), "D0 sampling drift")
    require(g["sq_bits"] == 4 and g["sq_native_int4"], "SQ4 quantization disabled")
    require(g["persistent_sq8_assignment_kernel_actual"] == "int4-wmma"
            and g["persistent_large_sq8_dp4a_tiles"] == 0, "INT4 WMMA assignment drift")
    require(g["persistent_large_ticket_batch_actual"] == 8, "ticket batch drift")
    require(g["persistent_scheduler_mode_actual"] == "baseline" and g["persistent_sq8_leaf_mode_actual"] == "staged",
            "scheduler/leaf executor drift")
    require(g["cpu_fallbacks"] == g["persistent_fallback_count"] == g["persistent_queue_overflow_flags"] == 0,
            "unexpected fallback/overflow")
    skipped = [int(v) for v in re.findall(r"gpu-subtree D1\+ child skipped: idx=\d+/\d+ rows=(\d+) split_depth=\d+ reason=singleton mode=sq-wmma-sidecar", log)]
    require(all(v <= 1 for v in skipped), "invalid singleton skip")
    uploaded_rows = rows * cfg["fanout"] - sum(skipped)
    require(g["sq_wmma_sidecar_rows"] == uploaded_rows and g["sq_wmma_sidecar_file_gather_ms"] == 0,
            "membership count or sidecar reload drift")
    require(g["h2d_bytes"] == uploaded_rows * 272, "unexpected D1 encoded upload volume")
    require(profile["stage_ms"]["sketch_persist"] == 0 and g["sq_staging_fused_sketch_enabled"],
            "separate sketch pass was used")
    pruning = profile["hashprune_backfill"]
    require(pruning["mode"] == "rhalf-soft-tier-q3" and pruning["effective_max_backfill_per_row"] == 3
            and pruning["effective_priority_alpha"] == 1.02, "HashPrune policy drift")
    require(g["persistent_success_count"] == g["persistent_attempt_count"] > 0, "GPU tasks failed")
    admissions = [tuple(map(int, re.search(r"admission_required_bytes=(\d+) admission_budget_bytes=(\d+)", line).groups()))
                  for line in log.splitlines() if "D1+ child backend start:" in line
                  and "mode=sq-wmma-sidecar" in line]
    upload_limit = int((16 << 30) * high_ratio)
    require(len(admissions) == g["persistent_success_count"]
            and all(required <= budget == upload_limit for required, budget in admissions),
            "per-subtree upload admission did not enforce the configured threshold")
    if g["d1plus_admission_split_count"]:
        require(g["d1plus_lazy_split_source"] == "sq4-primary-residual", "CPU split distance-source drift")
    line = re.search(r"SQ4\+ GPU D0: (.+)", log)
    require(line is not None and "mode=gpu-exact" in line[1], "GPU-exact D0 missing")
    stats = {k: float(v) for k, v in re.findall(r"(\w+): ([0-9.eE+\-]+)", line[1])}
    require(stats["exact_gpu_rows"] == rows and stats["checked_cpu_rows"] == 0,
            "D0 did not use GPU-exact for every point")
    require(stats["raw_h2d_bytes"] == rows * 512 * 4, "extra raw-vector D0 uploads")
    require("SQ4+ large INT4 WMMA projection reuse enabled" in log, "projection reuse disabled")
    require(f"SQ4+ primary table: rows={rows} centers=64" in log, "primary codebook drift")
    require("SQ4+ primary large INT4 point x leader shape: 8x32" in log, "INT4 assignment optimization disabled")
    require("SQ4 PUBLISH cooperative member copies enabled" in log, "PUBLISH optimization disabled")
    return dict(accepted=True, profile=record["profile"], rows=rows, leaders=leaders,
                fanout=cfg["fanout"], sample_fraction=cfg["sample"], capacity_ratio=target_ratio, primary_centers=64,
                planning_budget_ratio=target_ratio, admission_threshold_ratio=high_ratio,
                admission_splits=g["d1plus_admission_split_count"], gpu_tasks=g["persistent_attempt_count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release/build_shear_index")
    parser.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    parser.add_argument("--validation-prefix", action="store_true", help="validate a named recipe on a smaller prefix; never labels it a full-scale run")
    parser.add_argument("--memory-budget-gb", type=float, help="explicit, recorded host-budget override; 0 means automatic")
    parser.add_argument("--routing-leaders", type=int, help="explicit fixed routing count; encoding codebook stays at 64")
    parser.add_argument("--fanout-top", type=int, help="recorded override; also derives root risk candidates")
    parser.add_argument("--d0-sample-fraction", type=float, help="recorded override for both routing and encoding center training")
    parser.add_argument("--planning-budget-ratio", type=float, default=0.08, help="D0 planning and split target / GPU workspace")
    parser.add_argument("--admission-threshold-ratio", type=float, default=0.50, help="per-subtree upload limit / GPU workspace")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    record, env = make_manifest(args.data_path, args.binary, args.run_dir, args.profile,
                                args.validation_prefix, args.memory_budget_gb,
                                routing_leaders=args.routing_leaders, fanout_top=args.fanout_top,
                                sample_fraction=args.d0_sample_fraction,
                                planning_budget_ratio=args.planning_budget_ratio,
                                admission_threshold_ratio=args.admission_threshold_ratio)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.run_dir / "cmd.json").write_text(json.dumps(record["command"], indent=2) + "\n")
    (args.run_dir / "env.json").write_text(json.dumps(record["environment"], indent=2) + "\n")
    print(f"SQ4+ profile={record['profile']} rows={record['input']['rows']} source={record['source']}", flush=True)
    with (args.run_dir / "build.log").open("x") as output:
        subprocess.run(record["command"], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    p = json.loads((args.run_dir / "profile.json").read_text())
    audit = audit_profile(record, p, (args.run_dir / "build.log").read_text())
    (args.run_dir / "config-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
