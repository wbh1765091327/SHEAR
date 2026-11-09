"""Parse existing SQ4 telemetry; keep overlapping timers and work counts separate."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

BASELINE = "b08-h50"
BUDGETS = (4, 8, 16, 32)
THRESHOLDS = (8, 16, 32, 50)
LIST_SIZES = (64, 128, 256, 512)


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def fields(line):
    result = {}
    for key, value in re.findall(r"(\w+)=([^\s,]+)", line):
        try:
            result[key] = float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            result[key] = value
    return result


def stats(values):
    values = list(values)
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), raw=values) if values else None


def interpolate(points, target):
    """No extrapolation, no sorting a nonmonotonic recall curve into fake brackets."""
    points = sorted(points, key=lambda p: p["L"])
    for p in points:
        if p["recall"] == target:
            return p["qps"]
    for a, b in zip(points, points[1:]):
        if a["recall"] < target < b["recall"]:
            return a["qps"] + (b["qps"] - a["qps"]) * (target - a["recall"]) / (b["recall"] - a["recall"])
    return None


def parse_build(profile, log, timing, resources):
    g, stage = profile["gpu_subtree"], profile["stage_ms"]
    capacity = g["d0_capacity"]
    preparation_match = re.search(r"SQ4\+ D0 leader preparation: (.+)", log)
    d0_match = re.search(r"SQ4\+ GPU D0: (.+)", log)
    if not preparation_match or not d0_match:
        raise ValueError("missing SQ4 center preparation or GPU D0 telemetry")
    preparation = fields(preparation_match[1])
    d0 = {k: float(v) for k, v in re.findall(r"(\w+): ([0-9.eE+\-]+)", d0_match[1])}
    tasks, pending, splits, roots = [], [], [], {}
    for line in log.splitlines():
        if "gpu-subtree D1+ child skipped:" in line and "reason=singleton mode=sq-wmma-sidecar" in line:
            skipped = fields(line)
            if skipped["split_depth"] == 0:
                roots[int(skipped["idx"].split("/")[0])] = skipped["rows"]
        elif "SQ-only lazy split before admission:" in line:
            s = fields(line)
            s["parent"] = int(re.search(r"child (\d+) SQ-only", line)[1])
            splits.append(s)
            if s["depth"] == 0:
                roots[s["parent"]] = s["rows"]
        elif "D1+ child backend start:" in line and "mode=sq-wmma-sidecar" in line:
            task = fields(line)
            task["parent"] = int(task["idx"].split("/")[0])
            task["ordinal"] = len(tasks)
            tasks.append(task)
            pending.append(task)
            if task["split_depth"] == 0:
                roots[task["parent"]] = task["rows"]
        elif "gpu-subtree D1+ child done:" in line and "mode=sq-wmma-sidecar" in line:
            done = fields(line)
            if not pending:
                raise ValueError("completion without an upload")
            task = pending.pop(0)
            if done["idx"] != task["idx"] or done["run_len"] != task["rows"]:
                raise ValueError("upload/completion order mismatch")
            task["completion"] = done
    if pending or len(tasks) != g["persistent_success_count"] or len(splits) != g["d1plus_admission_split_count"]:
        raise ValueError("incomplete subtree/split telemetry")
    skipped_rows = sum(int(v) for v in re.findall(r"gpu-subtree D1\+ child skipped: idx=\d+/\d+ rows=(\d+) split_depth=\d+ reason=singleton mode=sq-wmma-sidecar", log))
    if len(roots) != g["d0_tagore_partitions"] or sum(roots.values()) != g["sq_wmma_sidecar_rows"] + skipped_rows:
        raise ValueError("cannot reconstruct all D0 root sizes")
    gpu_samples = [r["gpu"] for r in resources if "gpu" in r]
    if not gpu_samples:
        raise ValueError("missing actual GPU memory observations")
    gpu_ms = g["persistent_kernel_event_ms"]
    result = dict(
        wall_s=timing[0], rss_gib=timing[1] / 1048576,
        physical_read_bytes=int(timing[2] * 512), physical_write_bytes=int(timing[3] * 512),
        major_faults=int(timing[4]), minor_faults=int(timing[5]),
        d0_s=g["d0_wall_ms"] / 1000, d1_s=g["d1plus_producer_ms"] / 1000,
        emit_s=stage["emit"] / 1000, gpu_s=gpu_ms / 1000,
        kernel_prepare_s=g["persistent_kernel_prepare_ms"] / 1000,
        split_s=g["d1plus_lazy_split_ms"] / 1000, split_count=len(splits),
        split_rows=sum(s["rows"] for s in splits),
        split_distance_pairs=sum(s["rows"] * s["split_count"] for s in splits),
        max_split_depth=max((t["split_depth"] for t in tasks), default=0),
        gather_s=g["sq_wmma_sidecar_gather_ms"] / 1000,
        wait_s=g["gpu_wait_for_sidecar_ms"] / 1000,
        queue_peak_bytes=g["sidecar_queue_max_bytes"],
        input_h2d_with_projection_s=g["sq_wmma_h2d_ms"] / 1000,
        projection_s=sum(int(x) for x in re.findall(r"SQ4\+ GPU projection:.*?device_us=(\d+)", log)) / 1e6,
        h2d_bytes=g["h2d_bytes"], d2h_bytes=g["d2h_bytes"],
        tasks=len(tasks), leaders=g["d0_tagore_partitions"],
        max_upload_rows=max(t["rows"] for t in tasks),
        max_upload_required_bytes=max(t["admission_required_bytes"] for t in tasks),
        uploads_above_target=sum(t["admission_required_bytes"] > capacity["target_budget_bytes"] for t in tasks),
        roots_above_target=sum(n > capacity["c_target"] for n in roots.values()),
        roots_above_high=sum(n > capacity["c_high"] for n in roots.values()),
        root_p50_rows=capacity["actual_child_p50_unique_rows"],
        root_p95_rows=capacity["actual_child_p95_unique_rows"],
        root_p99_rows=capacity["actual_child_p99_unique_rows"],
        root_max_rows=capacity["actual_child_max_unique_rows"],
        gpu_process_peak_gib=max(s["process_used_bytes"] for s in gpu_samples) / (1 << 30),
        gpu_device_peak_gib=max(s["used_bytes"] for s in gpu_samples) / (1 << 30),
        gpu_min_free_gib=min(s["free_bytes"] for s in gpu_samples) / (1 << 30),
        gpu_ms_per_million_members=gpu_ms * 1e6 / g["sq_wmma_sidecar_rows"],
        assignment_physical_tiles=g["persistent_large_physical_tiles"],
        leaf_count=g["persistent_leaf_size_count"], leaf_rows=g["persistent_leaf_size_sum"],
        leaf_min=g["persistent_leaf_size_min"], leaf_max=g["persistent_leaf_size_max"],
        leaf_mean=g["persistent_leaf_size_sum"] / max(1, g["persistent_leaf_size_count"]),
        leaf_pairs=g["leaf_dco_pair_count"],
        route_prepare_s=preparation["route_prepare_ms"] / 1000,
        primary_train_s=preparation["primary_train_ms"] / 1000,
        center_prepare_s=preparation["total_ms"] / 1000,
        sketch_worker_s=g["sq_staging_fused_sketch_compute_ms"] / 1000,
        sketch_join_wait_s=g["sq_staging_fused_sketch_join_wait_ms"] / 1000,
        capacity=capacity, preparation=preparation, gpu_d0=d0,
        d0_partition_hash=g["d0_partition_hash"], d0_task_hash=g["d0_task_list_hash"],
        primary_hash=preparation["primary_hash"],
    )
    for name in ("host_read", "reader_wait", "h2d", "assign", "encode", "d2h"):
        result["d0_" + name + "_s"] = d0[name + "_ms"] / 1000
    result["memory_margin_ok"] = result["gpu_min_free_gib"] >= 1.0
    # Only a no-split path can be certified from the existing D0 membership/task
    # hashes. Matching child sizes after splitting is NOT proof of equal IDs.
    result["task_equivalence_key"] = (
        f'{g["d0_partition_hash"]}:{g["d0_task_list_hash"]}:{preparation["primary_hash"]}'
        if not splits else None)
    buckets = []
    for lo, hi in zip((0, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000),
                      (500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, math.inf)):
        selected = [t for t in tasks if lo < t["rows"] <= hi]
        if selected:
            rows = sum(t["rows"] for t in selected)
            kernel_ms = sum(t["completion"]["topk_kernel_ms"] for t in selected)
            buckets.append(dict(lower_exclusive=lo, upper_inclusive=hi if math.isfinite(hi) else None,
                                tasks=len(selected), rows=rows, kernel_ms=kernel_ms,
                                gpu_ms_per_million_members=kernel_ms * 1e6 / rows))
    result["subtree_buckets"] = buckets
    return result, tasks, splits, roots


def select_finalists(results):
    candidates = [r for r in results if r["status"] == "success" and r["config"] != BASELINE
                  and r["metrics"]["memory_margin_ok"]]
    groups = {}
    baseline_keys = {r["metrics"]["task_equivalence_key"] for r in results
                     if r["config"] == BASELINE and r["status"] == "success"
                     and r["metrics"].get("task_equivalence_key")}
    for r in candidates:
        key = r["metrics"].get("task_equivalence_key") or r["config"]
        groups.setdefault(key, []).append(r)
    representatives = [min(items, key=lambda r: (r["high"], r["metrics"]["wall_s"])) for items in groups.values()]
    representatives.sort(key=lambda r: (r["metrics"].get("task_equivalence_key") in baseline_keys,
                                       r["metrics"]["wall_s"], r["high"], r["config"]))
    return [r["config"] for r in representatives[:2]]


def near_noise(a, b):
    """A practical range check, not a statistical significance test."""
    if not a or not b:
        return False
    spread = max(max(a) - min(a), max(b) - min(b), statistics.median(a) * 0.02)
    return abs(statistics.median(a) - statistics.median(b)) <= spread


def summarize(results):
    grouped = {}
    for r in results:
        grouped.setdefault(r["config"], []).append(r)
    summary = {}
    for config, runs in grouped.items():
        good = [r for r in runs if r["status"] == "success"]
        numeric = {k for r in good for k, v in r["metrics"].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith("_hash")}
        quality = {str(L): {k: stats(p[k] for r in good for p in r["quality"] if p["L"] == L)
                            for k in ("recall", "qps")} for L in LIST_SIZES}
        matched = {str(t): stats(v for r in good if (v := interpolate(r["quality"], t)) is not None)
                   for t in (0.95, 0.97)}
        summary[config] = dict(successful=len(good), failures=[r for r in runs if r["status"] != "success"],
                               metrics={k: stats(r["metrics"][k] for r in good if k in r["metrics"]) for k in sorted(numeric)},
                               quality=quality, matched_recall=matched,
                               memory_margin_ok=bool(good) and all(r["metrics"]["memory_margin_ok"] for r in good),
                               task_equivalence_keys=sorted({r["metrics"]["task_equivalence_key"] for r in good
                                                            if r["metrics"].get("task_equivalence_key")}))
    baseline = summary.get(BASELINE, {})
    for config, row in summary.items():
        if not row["successful"] or not baseline.get("successful"):
            continue
        row["wall_delta_percent"] = (row["metrics"]["wall_s"]["median"] /
                                     baseline["metrics"]["wall_s"]["median"] - 1) * 100
        row["within_timing_noise"] = near_noise(baseline["metrics"]["wall_s"]["raw"], row["metrics"]["wall_s"]["raw"])
        for L in map(str, LIST_SIZES):
            q, ref = row["quality"][L], baseline["quality"][L]
            if q["recall"] and ref["recall"]:
                q["recall_delta_pp"] = (q["recall"]["median"] - ref["recall"]["median"]) * 100
        for target, measured in row["matched_recall"].items():
            ref = baseline["matched_recall"][target]
            if measured and ref:
                measured["delta_percent"] = (measured["median"] / ref["median"] - 1) * 100
    return summary


def write_report(root, results, plot=False):
    root = Path(root)
    summary = summarize(results)
    save(root / "summary.json", summary)
    successful = [r for r in results if r["status"] == "success"]
    columns = sorted({k for r in successful for k, v in r["metrics"].items()
                      if isinstance(v, (int, float, bool))})
    with (root / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "config", "status", *columns])
        writer.writeheader()
        for r in results:
            writer.writerow(dict(run=r["run"], config=r["config"], status=r["status"],
                                 **{k: r.get("metrics", {}).get(k) for k in columns}))
    with (root / "quality.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "config", "L", "recall", "qps"])
        for r in successful:
            for q in r["quality"]:
                writer.writerow([r["run"], r["config"], q["L"], q["recall"], q["qps"]])
    lines = ["# SQ4+ capacity sweep", "", "fanout=6/3/1; sample=0.002; seed=42; workspace=16 GiB.",
             "Timers overlap. GPU throughput is workload dependent, not an SM-efficiency measurement.",
             "D0 code/member D2H has a combined timer; separate byte counts remain in each metrics.json.",
             "Memory peaks are 100 ms observations. No OOM is not a proof that the threshold boundary is safe.",
             "Graphs are removed only after structural audit and four validated Recall–QPS measurements.", "",
             "| Budget / threshold | Successful | Wall median [range] s | Delta % | GPU s | Split s | Wait s | GPU process GiB | Min free GiB |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for b in BUDGETS:
        for h in THRESHOLDS:
            if b > h:
                continue
            row = summary.get(f"b{b:02}-h{h:02}")
            if not row or not row["successful"]:
                lines.append(f"| {b}% / {h}% | 0 | pending/failed | — | — | — | — | — | — |")
                continue
            m, w = row["metrics"], row["metrics"]["wall_s"]
            lines.append(f"| {b}% / {h}% | {row['successful']} | {w['median']:.3f} [{w['minimum']:.3f}, {w['maximum']:.3f}] "
                         f"| {row.get('wall_delta_percent', 0):+.2f} | {m['gpu_s']['median']:.3f} | {m['split_s']['median']:.3f} "
                         f"| {m['wait_s']['median']:.3f} | {m['gpu_process_peak_gib']['maximum']:.3f} | {m['gpu_min_free_gib']['minimum']:.3f} |")
    lines += ["", "## Detailed metrics", "",
              "Each configuration includes medians, ranges and raw measurements; recall deltas are percentage points.",
              "matched_recall contains 95%/97% interpolations only where measured points bracket the target.",
              "All successful measurements, including quality regressions and low-memory-margin configurations, remain visible.",
              "summary.json, runs.csv, quality.csv and per-run subtrees.json contain the complete comparison.", ""]
    compared = [name for name, r in summary.items() if r["successful"] >= 3 and name.startswith("b")]
    if BASELINE in compared:
        compared.remove(BASELINE)
        compared.insert(0, BASELINE)
    if compared:
        lines += ["| Metric | " + " | ".join(compared) + " |",
                  "|---|" + "---:|" * len(compared)]
        names = ["wall_s", "d0_s", "route_prepare_s", "primary_train_s", "center_prepare_s",
                 "d0_host_read_s", "d0_reader_wait_s", "sketch_worker_s", "sketch_join_wait_s",
                 "d0_h2d_s", "d0_assign_s", "d0_encode_s", "d0_d2h_s", "d1_s",
                 "split_s", "split_count", "split_distance_pairs", "max_split_depth", "gather_s", "wait_s",
                 "gpu_s", "projection_s", "input_h2d_with_projection_s", "emit_s",
                 "rss_gib", "gpu_process_peak_gib", "gpu_min_free_gib", "physical_read_bytes",
                 "root_p50_rows", "root_p95_rows", "root_p99_rows", "root_max_rows",
                 "tasks", "max_upload_rows", "assignment_physical_tiles", "leaf_count", "leaf_mean",
                 "gpu_ms_per_million_members"]
        for key in names:
            values = [summary[name]["metrics"].get(key) for name in compared]
            lines.append("| " + key + " | " + " | ".join(
                f"{v['median']:.3f} [{v['minimum']:.3f}, {v['maximum']:.3f}]" if v else "—" for v in values) + " |")
    lines += ["", "## Recall–QPS", "", "| Configuration | L | Recall % [range] | Δ recall pp | QPS [range] |",
              "|---|---:|---:|---:|---:|"]
    for config, row in summary.items():
        for L, q in row["quality"].items():
            if not q["recall"]:
                continue
            r, speed = q["recall"], q["qps"]
            delta = q.get("recall_delta_pp", 0)
            lines.append(f"| {config} | {L} | {r['median']*100:.4f} [{r['minimum']*100:.4f}, {r['maximum']*100:.4f}] "
                         f"| {delta:+.4f} | {speed['median']:.1f} [{speed['minimum']:.1f}, {speed['maximum']:.1f}] |")
    lines += ["", "## Same-recall QPS (interpolated, no extrapolation)", "",
              "| Configuration | Recall target | QPS median [range] | Measured graphs reaching target |",
              "|---|---:|---:|---:|"]
    for config, row in summary.items():
        for target, q in row["matched_recall"].items():
            value = f"{q['median']:.1f} [{q['minimum']:.1f}, {q['maximum']:.1f}]" if q else "not bracketed"
            lines.append(f"| {config} | {float(target)*100:.0f}% | {value} | {len(q['raw']) if q else 0}/{row['successful']} |")
    lines += ["", "## Interpretation", "",
              "A larger budget changes D0 K and initial subtree sizes; a higher threshold moves split work from CPU to GPU.",
              "Use fixed-budget columns and fixed-threshold rows for attribution. Different D0 hashes are system comparisons.",
              "D0 host_read is accumulated read/copy work, not a disk-wait timer. Physical I/O is reported separately.",
              "Only no-split tasks with identical D0 and encoding hashes are certified as equivalent; matching row counts alone are insufficient.",
              "Ordinary SQ8 is a separately frozen system reference, not a cell in the parameter matrix.", ""]
    candidates = [(name, row) for name, row in summary.items()
                  if row["successful"] >= 3 and not row["failures"] and row["memory_margin_ok"]
                  and name.startswith("b")]
    if candidates:
        fastest, row = min(candidates, key=lambda x: x[1]["metrics"]["wall_s"]["median"])
        lines.append(f"Fastest validated median: {fastest}. Defaults are not changed.")
        lines.append("No stable timing gain established." if row.get("within_timing_noise") else
                     "The observed median difference exceeds the predeclared timing-noise rule.")
        regressions = [L for L, q in row["quality"].items() if q.get("recall_delta_pp", 0) < 0]
        lines.append(f"Recall decreases versus baseline at L: {', '.join(regressions) or 'none observed'}. See raw ranges and matched-recall QPS.")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, field, title in zip(axes, ("wall_s", "gpu_process_peak_gib"),
                                    ("Build wall time (s)", "Observed GPU process peak (GiB)")):
            values = np.full((len(BUDGETS), len(THRESHOLDS)), np.nan)
            for i, b in enumerate(BUDGETS):
                for j, h in enumerate(THRESHOLDS):
                    row = summary.get(f"b{b:02}-h{h:02}", {})
                    if row.get("successful"):
                        values[i, j] = row["metrics"][field]["median" if field == "wall_s" else "maximum"]
            heat = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis")
            for i in range(len(BUDGETS)):
                for j in range(len(THRESHOLDS)):
                    ax.text(j, i, f"{values[i,j]:.2f}" if np.isfinite(values[i,j]) else "—",
                            ha="center", va="center", color="red" if np.isfinite(values[i,j]) else "black")
            ax.set(xticks=range(4), xticklabels=[f"{h}%" for h in THRESHOLDS],
                   yticks=range(4), yticklabels=[f"{b}%" for b in BUDGETS],
                   xlabel="Admission threshold", ylabel="Planning budget", title=title)
            fig.colorbar(heat, ax=ax)
        fig.savefig(root / "heatmaps.png", dpi=180)
        fig.savefig(root / "heatmaps.pdf")
        plt.close(fig)
    return summary
