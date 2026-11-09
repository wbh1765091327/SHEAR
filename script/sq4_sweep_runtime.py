"""Linux measurement and lifecycle helpers for the SQ4 capacity experiment."""
import array
import csv
import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import subprocess
import time

from run_sq4_plus import digest
from sq4_sweep_metrics import LIST_SIZES, save


def identity(path):
    path = Path(path).resolve()
    st = path.stat()
    return dict(path=str(path), device=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def shape(path):
    with Path(path).open("rb") as f:
        return struct.unpack("<II", f.read(8))


def check_inputs(passport, expected_rows, hash_files=True):
    """Accept the historical passport layout, but validate against this scale."""
    result = {}
    truth_key = "gt100m" if expected_rows == 100_000_000 else "gt10m"
    for key, entry in (("base", passport["base"]), ("query", passport["query"]),
                       ("truth", passport.get("truth", passport.get(truth_key)))):
        if not entry:
            raise ValueError(f"missing {truth_key} identity; never substitute 100M truth for 10M")
        path = Path(entry.get("path", entry.get("identity", {}).get("path", ""))).resolve()
        before = identity(path)
        rows, dim = shape(path)
        expected_shape = (expected_rows, 512) if key == "base" else ((10000, 512) if key == "query" else (10000, 100))
        stride = 8 if key == "truth" else 4
        if (rows, dim) != expected_shape or before["bytes"] != 8 + rows * dim * stride:
            raise ValueError(f"wrong {key} shape or length: {path}")
        expected_hash = entry.get("sha256")
        if not expected_hash or len(expected_hash) != 64:
            raise ValueError(f"{key} needs an expected SHA256 tied to the chosen dataset")
        if hash_files and digest(path) != expected_hash:
            raise ValueError(f"{key} SHA256 mismatch: {path}")
        if identity(path) != before:
            raise ValueError(f"{key} changed during verification")
        result[key] = dict(path=str(path), shape=[rows, dim], identity=before, sha256=expected_hash,
                           sha256_verified=hash_files)
    if hash_files:
        with Path(result["truth"]["path"]).open("rb") as f:
            f.seek(8)
            ids = array.array("I")
            ids.frombytes(f.read(10000 * 100 * 4))
        if os.sys.byteorder != "little":
            ids.byteswap()
        if max(ids) >= expected_rows:
            raise ValueError("truth contains IDs outside this base")
        result["truth"].update(min_id=min(ids), max_id=max(ids))
    return result


def io_bytes():
    return {k: int(v) for k, v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}


def probe(path):
    """Actual PTE reads + smaps RSS, not permission-protected mincore bits.

    Reuses the protocol from runs/sq4_gpu_d0_exact_20261108/cache_probe.py.
    """
    path = Path(path).resolve()
    pages = (path.stat().st_size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    start = time.monotonic()
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        before_io, before_fault = io_bytes(), resource.getrusage(resource.RUSAGE_SELF)
        touched = mm[::mmap.PAGESIZE]
        after_fault, after_io = resource.getrusage(resource.RUSAGE_SELF), io_bytes()
        active, rss = False, 0
        for line in Path("/proc/self/smaps").read_text().splitlines():
            if line and "-" in line.split()[0]:
                active = line.split()[-1] == str(path)
            elif active and line.startswith("Rss:"):
                rss += int(line.split()[1]) * 1024
        if len(touched) != pages:
            raise ValueError("incomplete page touch")
        return dict(method="mmap-touch-every-page-and-smaps-rss", pages=pages,
                    resident_pages=rss // mmap.PAGESIZE,
                    read_bytes=after_io["read_bytes"] - before_io["read_bytes"],
                    major_faults=after_fault.ru_majflt - before_fault.ru_majflt,
                    seconds=time.monotonic() - start)


def is_hot(p):
    return p["pages"] == p["resident_pages"] and p["read_bytes"] == p["major_faults"] == 0


def prewarm(paths, output):
    start = time.monotonic()
    for path in paths:
        with Path(path).open("rb") as f:
            for _ in iter(lambda: f.read(32 << 20), b""):
                pass
    attempts = []
    for _ in range(3):
        current = {str(p): probe(p) for p in paths}
        attempts.append(current)
        save(output, dict(excluded_from_build=True, seconds=time.monotonic() - start, attempts=attempts))
        if all(is_hot(p) for p in current.values()):
            return
    raise ValueError("initial full input residency could not be established")


class Nvml:
    """Direct NVML calls avoid spawning nvidia-smi ten times per second."""
    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    class Process(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong),
                    ("gpu_instance", ctypes.c_uint), ("compute_instance", ctypes.c_uint)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.device = ctypes.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.device)))

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f"NVML measurement failed, status={code}")

    def sample(self):
        memory = self.Memory()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.device, ctypes.byref(memory)))
        count = ctypes.c_uint(256)
        processes = (self.Process * count.value)()
        self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.device, ctypes.byref(count), processes))
        values = [dict(pid=p.pid, used_bytes=p.used) for p in processes[:count.value]]
        if any(p["used_bytes"] == (1 << 64) - 1 for p in values):
            raise RuntimeError("NVML process memory unavailable")
        return dict(total_bytes=memory.total, free_bytes=memory.free, used_bytes=memory.used,
                    processes=values, process_used_bytes=sum(p["used_bytes"] for p in values))

    def require_idle(self):
        observed = self.sample()
        if observed["processes"]:
            raise RuntimeError(f"GPU busy: {observed['processes']}")
        return observed

    def close(self):
        self.check(self.lib.nvmlShutdown())


def process_tree(pid):
    result, pending = [], [pid]
    while pending:
        current = pending.pop()
        base = Path(f"/proc/{current}")
        try:
            status = dict(line.split(":", 1) for line in (base / "status").read_text().splitlines() if ":" in line)
            result.append(dict(pid=current, rss_bytes=int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                               swap_bytes=int(status.get("VmSwap", "0 kB").split()[0]) * 1024))
            pending.extend(map(int, (base / f"task/{current}/children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def read_timing(path):
    lines = Path(path).read_text().splitlines()
    values = list(map(float, lines[-1].split()))
    if len(values) != 6 or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("missing or invalid GNU time record")
    return values


def run_measured(command, env, directory, nvml=None, cwd=None):
    directory = Path(directory)
    cmd = ["/usr/bin/time", "-f", "%e %M %I %O %F %R", "-o", str(directory / "time.txt"), *command]
    save(directory / "cmd.json", cmd)
    start = time.monotonic()
    with (directory / "build.log").open("x") as output, (directory / "resources.jsonl").open("x") as samples:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                record = dict(seconds=time.monotonic() - start, processes=process_tree(proc.pid))
                if nvml:
                    record["gpu"] = nvml.sample()
                    for gpu_process in record["gpu"]["processes"]:
                        try:
                            if os.getpgid(gpu_process["pid"]) != proc.pid:
                                raise RuntimeError("another GPU process started during the measured build")
                        except ProcessLookupError:
                            pass
                samples.write(json.dumps(record) + "\n")
                samples.flush()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return read_timing(directory / "time.txt")


def quality(search, inputs, graph, directory, env):
    directory = Path(directory)
    directory.mkdir()
    base, query, truth = [Path(inputs[k]["path"]) for k in ("base", "query", "truth")]
    for entry in inputs.values():
        if identity(entry["path"]) != entry["identity"]:
            raise ValueError("input identity changed before search")
    prewarm([base, graph, query, truth], directory / "cache.json")
    save(directory / "inputs.json", dict(inputs=inputs, graph_sha256=digest(graph),
                                         search_sha256=digest(search), storage="mmap", backend="exact", threads=8))
    result = []
    for name, size in [("warm", 64), *[(str(size), size) for size in LIST_SIZES]]:
        command = [str(search), "--index-path-prefix", str(graph.parent), "--data-path", str(base),
                   "--query-file", str(query), "--gt-file", str(truth), "--recall-at", "10",
                   "--search-list-size", str(size), "--num-threads", "8", "--storage", "mmap",
                   "--search-backend", "exact", "--result-path", str(directory / f"{name}.fbin"),
                   "--metrics-path", str(directory / f"{name}.tsv")]
        cmd = ["/usr/bin/time", "-f", "%e %M %I %O %F %R", "-o", str(directory / f"{name}.time"), *command]
        save(directory / f"{name}-cmd.json", cmd)
        with (directory / f"{name}.log").open("x") as out:
            subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
        with (directory / f"{name}.tsv").open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if len(rows) != 1:
            raise ValueError("search did not produce one complete metrics row")
        r = rows[0]
        recall, qps = float(r["recall"]), float(r["qps"])
        if (r["storage"] != "mmap" or r["search_backend"] != "exact" or int(r["queries"]) != 10000
                or int(r["search_list_size"]) != size or int(r["recall_at"]) != 10
                or not math.isfinite(recall) or not 0 <= recall <= 1 or not math.isfinite(qps) or qps <= 0):
            raise ValueError("invalid search result or execution path")
        timing = read_timing(directory / f"{name}.time")
        if name != "warm":
            result.append(dict(L=size, recall=recall, qps=qps, wall_s=timing[0],
                               physical_read_bytes=int(timing[2] * 512), major_faults=int(timing[4])))
    save(directory / "result.json", result)
    return result


def audit_graph(binary, graph, rows, output):
    command = [str(binary), str(graph), str(rows)]
    save(output.with_suffix(".cmd.json"), command)
    with output.open("x") as out:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
    text = output.read_text()
    required = [f"nodes={rows}", "duplicates=0", "self_loops=0", "invalid=0"]
    if not all(x in text for x in required):
        raise ValueError("graph audit did not certify all structural invariants")
    import re
    maximum = re.search(r"max_degree=(\d+)", text)
    if not maximum or int(maximum[1]) > 64:
        raise ValueError("graph degree exceeds 64")
    return text.strip()


def cleanup_search_intermediates(run_dir):
    for path in (Path(run_dir) / "quality").glob("*.fbin"):
        if path.is_symlink():
            raise ValueError("unexpected search artifact symlink")
        path.unlink()


def cleanup_owned(scratch, owner_root, token, run_dir):
    """Never follow a supplied symlink or remove an unaudited/unevaluated graph."""
    scratch, owner_root, run_dir = Path(scratch), Path(owner_root), Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    quality_result = json.loads((run_dir / "quality/result.json").read_text())
    audit = json.loads((run_dir / "config-audit.json").read_text())
    if (result["status"] != "success" or not audit["accepted"] or not result["graph_audit"]
            or {q["L"] for q in quality_result} != set(LIST_SIZES) or len(quality_result) != 4
            or quality_result != result["quality"]):
        raise ValueError("cleanup requires a complete audited build and four quality results")
    if scratch.is_symlink() or scratch.resolve().parent != owner_root.resolve():
        raise ValueError("cleanup path is not a direct owned scratch directory")
    if json.loads((scratch / ".sweep-owner.json").read_text()) != {"token": token, "run": result["run"]}:
        raise ValueError("scratch ownership mismatch")
    graph = scratch / "index/_mem.index"
    if graph.is_symlink() or digest(graph) != result["graph_sha256"]:
        raise ValueError("graph changed after evaluation")
    save(run_dir / "cleanup.json", dict(state="verified", scratch=str(scratch), graph_sha256=result["graph_sha256"]))
    shutil.rmtree(scratch)
    # Search neighbor arrays are reproducible intermediates; retain raw TSV/logs.
    cleanup_search_intermediates(run_dir)
    save(run_dir / "cleanup.json", dict(state="complete", scratch=str(scratch),
                                       graph_sha256=result["graph_sha256"], graphs_retained=False))


def failure_kind(error, log):
    text = (str(error) + "\n" + log).lower()
    if "cuda_error_out_of_memory" in text or ("cuda" in str(error).lower() and "out of memory" in str(error).lower()):
        return "gpu_oom"
    if "admission" in text and any(x in text for x in ("unsupported", "exceeds", "exceeded", "no progress")):
        return "admission_rejected"
    if "fallback/overflow" in text or "queue overflow" in text:
        return "fallback_or_overflow"
    if "no space left" in text:
        return "storage_exhausted"
    if "cannot allocate memory" in text or "memory allocation of" in text:
        return "host_allocation_failed"
    if isinstance(error, subprocess.CalledProcessError) and error.returncode in (-9, 137):
        return "killed_unknown"
    return "validation_or_execution_failed"
