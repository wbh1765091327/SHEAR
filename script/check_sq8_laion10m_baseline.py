#!/usr/bin/env python3
"""Check saved build evidence against the versioned ordinary SQ8 10M baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

BASE_SHA256 = "89632b353305bf33fb3733ef4b7f2d2c15a2485e49fe86c3b20d851ffc0ed0c6"
BINARY_SHA256 = "d46d7e33a57642b635d646fe5385582eb7ce2c3abd8656b73c8e9f9b2735e5ff"
D0_HASHES = (17602056864155019802, 5716008263347199385)
ARGS = {
    "--data-type": "float", "--dist-fn": "l2", "--oom-io-mode": "buffered",
    "--partitioner": "gpu-subtree", "--num-threads": "8", "--seed": "42",
    "--oom-memory-budget-gb": "256", "--pipnn-gpu-workspace-budget-gb": "16",
    "--pipnn-backend": "cuda", "--pipnn-gpu-device": "0",
    "--gpu-subtree-d0-partition-mode": "gpu-kmeans-coarse",
    "--gpu-subtree-sq-staging-mode": "gpu-d0", "--gpu-subtree-sq-bits": "8",
    "--gpu-subtree-sq-wmma-mode": "build", "--gpu-subtree-sq-wmma-sidecar-mode": "rebuild",
    "--gpu-subtree-execution-mode": "fused-persistent",
    "--gpu-subtree-sq-staging-upload-shard-rows": "262144",
    "--gpu-subtree-d0-capacity-quality-min-root-leaders": "64",
    "--gpu-subtree-d0-capacity-root-leader-cap": "64",
    "--gpu-subtree-d0-tagore-sample-fraction": "0.002",
    "--max-degree": "64", "--lbuild": "100", "--m-hash-bits": "12",
    "--fanout-top": "6", "--fanout-second": "3", "--psamp-fraction": "0.05",
    "--leaf-size-min": "256", "--leaf-size-max": "1280", "--leaf-knn": "2",
    "--gpu-subtree-persistent-large-ticket-batch": "8",
    "--gpu-subtree-persistent-leaf-tile-rows": "128",
}
ENV = {
    "SHEAR_PIPNN_SQ8_WMMA_OPT_MASK": "0", "SHEAR_PIPNN_PERSISTENT_ASYNC_D2H": "0",
    "SHEAR_PIPNN_PRUNE_BACKEND": "hashprune", "SHEAR_PIPNN_PROFILE": "1",
    "OPENBLAS_NUM_THREADS": "1", "RAYON_NUM_THREADS": "8",
}


def check(directory, expected_binary):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    def read(name):
        return json.loads((directory / name).read_text())
    cmd, env, cache, binary, p = [read(n) for n in (
        "cmd.json", "env.json", "cache.json", "binary.json", "profile.json")]
    g = p["gpu_subtree"]
    for key, value in ARGS.items():
        require(cmd.count(key) == 1 and cmd.index(key) + 1 < len(cmd)
                and cmd[cmd.index(key) + 1] == value, f"argument drift: {key} must be {value}")
    require("--oom-enable" in cmd, "missing --oom-enable")
    require(not any(x.startswith(("--gpu-subtree-sq4", "--gpu-subtree-sq8-plus",
                "--gpu-subtree-sq8-rotated-residual", "--gpu-subtree-sq-staging-raw"))
                or x == "--gpu-subtree-sq-native-int4" for x in cmd), "not ordinary SQ8")
    for key, value in ENV.items():
        require(env.get(key) == value, f"environment drift: {key} must be {value}")
    require(binary["sha256"] == expected_binary, "baseline binary version differs")
    require(hashlib.sha256(Path(binary["path"]).read_bytes()).hexdigest() == binary["sha256"],
            "saved binary hash does not match the executable")
    require(cmd[:3] == ["/usr/bin/time", "-f", "%e %M %I %O %F %R"]
            and cmd[5] == binary["path"], "timing command/executable differs")
    require(cache["sha256"] == BASE_SHA256, "LAION10M input digest differs")
    require(cache.get("excluded_from_build") is True, "prewarm must be outside build timing")
    require(cache["after"]["pages"] == cache["after"]["resident_pages"] == 5_000_001,
            "input was not fully resident before the timed build")
    require((p["num_points"], p["dim"], p["pool_threads"], p["memory_budget_bytes"],
             g["workspace_budget_bytes"]) == (10_000_000, 512, 8, 256 << 30, 16 << 30),
            "profile input/thread/budget mismatch")
    st = g.get("sq_gpu_staging", {})
    require(st.get("enabled") is True and st.get("mode") == "gpu-d0"
            and st.get("bits") == 8 and st.get("d0_assign_kernel_mode") == "cuda-wmma-batch",
            "GPU quantization/D0 assignment was not verified")
    require(st.get("gpu_sq_quantize_ms", 0) > 0 and st.get("gpu_sq_d0_assign_ms", 0) > 0
            and st.get("d0_assign_rows") == 10_000_000, "GPU D0 counters are incomplete")
    require(st.get("sidecar_source") == "gpu-d0-resident", "unexpected disk sidecar staging")
    require((g["d0_tagore_sample_points"], g["d0_child_run_count"], g["d0_child_points"])
            == (20_000, 64, 60_000_000), "D0 sample/center/fanout counts differ")
    require((g["d0_partition_hash"], g["d0_task_list_hash"]) == D0_HASHES,
            "D0 memberships/tasks differ from this baseline version")
    require(g["persistent_success_count"] == g["persistent_attempt_count"] > 0
            and g["persistent_large_sq8_wmma_tiles"] > 0, "GPU build path was not verified")
    require(all(g[k] == 0 for k in ("cpu_fallbacks", "persistent_fallback_count",
            "persistent_queue_overflow_flags", "persistent_large_sq8_dp4a_tiles")),
            "unexpected fallback/overflow/large-node DP4A path")
    log = (directory / "build.log").read_text()
    require("index build completed successfully" in log, "build did not complete")
    require("resident sidecar reused from just-created runtime cache:" in log
            and "resident sidecar preloaded:" not in log, "sidecar was reloaded or reuse unverified")
    audit = (directory / "audit.txt").read_text()
    require(re.search(r"nodes=10000000 edges=\d+ max_degree=64 duplicates=0 self_loops=0 invalid=0", audit),
            "full graph structural audit missing/failed")
    require(len((directory / "time.txt").read_text().split()) == 6
            and (directory / "resources.jsonl").stat().st_size > 0, "timing/I/O evidence missing")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-binary-sha256", default=BINARY_SHA256,
                        help="Explicitly version a newly validated binary; other protocol checks still apply")
    args = parser.parse_args()
    try:
        errors = check(args.run_dir, args.expected_binary_sha256)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        errors = [f"missing or malformed evidence: {exc}"]
    print(json.dumps({"protocol": "sq8-laion10m-gpu-d0-20261107", "run": str(args.run_dir),
                      "status": "PASS" if not errors else "NONCANONICAL", "errors": errors}, indent=2))
    sys.exit(bool(errors))
