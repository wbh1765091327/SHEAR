#!/usr/bin/env python3
import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import types
from pathlib import Path
from typing import Iterator

import numpy as np


ACE_HEADER = struct.Struct("<8s7I4Q")
FORMAL_ROWS = 10_000_000
FORMAL_DIM = 1024
FORMAL_BYTES = 8 + FORMAL_ROWS * FORMAL_DIM * 4
METHODS = ("cagra", "tagore_cagra", "tagore_vamana")
DISKANN_HEADER = struct.Struct("<QIIQ")


@dataclasses.dataclass(frozen=True)
class AcePartitionPlan:
    rows: int
    dim: int
    partitions: int
    histogram: np.ndarray
    core_offsets: np.ndarray
    augmented_offsets: np.ndarray
    sha256: str


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_fbin_shape(path: Path) -> tuple[int, int]:
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"short fbin header: {path}")
    rows, dim = struct.unpack("<ii", header)
    if rows <= 0 or dim <= 0:
        raise ValueError(f"invalid fbin shape rows={rows} dim={dim}: {path}")
    expected = 8 + rows * dim * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"fbin byte length mismatch: got {actual}, expected {expected}")
    return rows, dim


def read_ace_partition_plan(path: Path) -> AcePartitionPlan:
    path = Path(path)
    with path.open("rb") as stream:
        header_bytes = stream.read(ACE_HEADER.size)
        if len(header_bytes) != ACE_HEADER.size:
            raise ValueError(f"short ACE plan header: {path}")
        (
            magic,
            version,
            header_size,
            endian,
            _data_type,
            index_type,
            _metric,
            _reserved,
            rows,
            dim,
            partitions,
            payload_bytes,
        ) = ACE_HEADER.unpack(header_bytes)
        if magic != b"ACEPLAN\0" or version != 1 or header_size != ACE_HEADER.size:
            raise ValueError(f"unsupported ACE plan header: {path}")
        if endian != 0x01020304 or index_type != 1:
            raise ValueError(f"unsupported ACE plan encoding: {path}")
        if rows <= 0 or dim <= 0 or partitions <= 0:
            raise ValueError(f"invalid ACE plan dimensions: {path}")
        expected_values = 3 * rows + 4 * partitions + 2
        if payload_bytes != expected_values * 4:
            raise ValueError(f"ACE plan payload size mismatch: {path}")
        values = np.fromfile(stream, dtype="<u4", count=expected_values)
    if values.size != expected_values or path.stat().st_size != ACE_HEADER.size + payload_bytes:
        raise ValueError(f"truncated ACE plan payload: {path}")

    cursor = 0
    histogram = values[cursor : cursor + 2 * partitions].reshape(partitions, 2)
    cursor += 2 * partitions + 3 * rows
    core_offsets = values[cursor : cursor + partitions + 1]
    cursor += partitions + 1
    augmented_offsets = values[cursor : cursor + partitions + 1]
    if int(core_offsets[-1]) != rows or int(augmented_offsets[-1]) != rows:
        raise ValueError("ACE plan offsets do not close to dataset rows")
    if not np.array_equal(np.diff(core_offsets), histogram[:, 0]):
        raise ValueError("ACE core offsets disagree with histogram")
    if not np.array_equal(np.diff(augmented_offsets), histogram[:, 1]):
        raise ValueError("ACE augmented offsets disagree with histogram")
    return AcePartitionPlan(
        rows=int(rows),
        dim=int(dim),
        partitions=int(partitions),
        histogram=histogram.copy(),
        core_offsets=core_offsets.copy(),
        augmented_offsets=augmented_offsets.copy(),
        sha256=sha256_file(path),
    )


def choose_partition(histogram: np.ndarray, max_rows: int) -> int:
    if max_rows <= 0:
        raise ValueError("row cap must be positive")
    histogram = np.asarray(histogram)
    if histogram.ndim != 2 or histogram.shape[1] != 2:
        raise ValueError("partition histogram must have shape (partitions, 2)")
    combined = histogram.astype(np.uint64, copy=False).sum(axis=1)
    eligible = np.flatnonzero((combined > 0) & (combined <= max_rows))
    if eligible.size == 0:
        raise ValueError(f"no partition fits row cap {max_rows}")
    best_rows = int(combined[eligible].max())
    return int(eligible[combined[eligible] == best_rows][0])


def _write_shard_formats(
    shard_fbin: Path,
    shard_fvecs: Path,
    chunks: Iterator[np.ndarray],
    rows: int,
    dim: int,
) -> None:
    shard_fbin.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with shard_fbin.open("wb") as fbin, shard_fvecs.open("wb") as fvecs:
        fbin.write(struct.pack("<ii", rows, dim))
        for chunk in chunks:
            matrix = np.asarray(chunk, dtype="<f4", order="C")
            if matrix.ndim != 2 or matrix.shape[1] != dim:
                raise ValueError("shard chunk has unexpected shape")
            matrix.tofile(fbin)
            encoded = np.empty((matrix.shape[0], dim + 1), dtype="<u4")
            encoded[:, 0] = dim
            encoded[:, 1:] = matrix.view("<u4")
            encoded.tofile(fvecs)
            written += matrix.shape[0]
    if written != rows:
        raise ValueError(f"shard writer emitted {written} rows; expected {rows}")


def extract_frozen_shard(
    source_fbin: Path,
    plan_path: Path,
    reordered_path: Path,
    augmented_path: Path,
    output_dir: Path,
    max_rows: int = 1_100_000,
    chunk_rows: int = 8192,
) -> dict:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    source_fbin = Path(source_fbin)
    output_dir = Path(output_dir)
    source_shape = read_fbin_shape(source_fbin)
    plan = read_ace_partition_plan(plan_path)
    if source_shape != (plan.rows, plan.dim):
        raise ValueError("source fbin and ACE plan shapes differ")
    core = np.load(reordered_path, mmap_mode="r", allow_pickle=False)
    augmented = np.load(augmented_path, mmap_mode="r", allow_pickle=False)
    if core.shape != source_shape or augmented.shape != source_shape:
        raise ValueError("ACE reordered arrays do not match source shape")

    partition = choose_partition(plan.histogram, max_rows)
    core_start, core_end = map(int, plan.core_offsets[partition : partition + 2])
    aug_start, aug_end = map(int, plan.augmented_offsets[partition : partition + 2])
    rows = (core_end - core_start) + (aug_end - aug_start)

    def chunks() -> Iterator[np.ndarray]:
        for matrix, start, end in (
            (core, core_start, core_end),
            (augmented, aug_start, aug_end),
        ):
            for offset in range(start, end, chunk_rows):
                yield matrix[offset : min(end, offset + chunk_rows)]

    output_dir.mkdir(parents=True, exist_ok=False)
    shard_fbin = output_dir / "shard.fbin"
    shard_fvecs = output_dir / "shard.fvecs"
    _write_shard_formats(shard_fbin, shard_fvecs, chunks(), rows, plan.dim)
    manifest = {
        "schema": "wiki10m-local-shard-v1",
        "source_fbin": str(source_fbin.resolve()),
        "source_shape": list(source_shape),
        "source_bytes": source_fbin.stat().st_size,
        "source_sha256": sha256_file(source_fbin),
        "partition_plan": str(Path(plan_path).resolve()),
        "partition_plan_sha256": plan.sha256,
        "partitions": plan.partitions,
        "partition_id": partition,
        "core_rows": core_end - core_start,
        "augmented_rows": aug_end - aug_start,
        "rows": rows,
        "dim": plan.dim,
        "row_order": "core_then_augmented",
        "max_rows": max_rows,
        "shard_fbin": str(shard_fbin.resolve()),
        "shard_fvecs": str(shard_fvecs.resolve()),
        "shard_sha256": sha256_file(shard_fbin),
        "shard_fvecs_sha256": sha256_file(shard_fvecs),
    }
    manifest_path = output_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_shard_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "wiki10m-local-shard-v1":
        raise ValueError("unsupported shard manifest schema")
    fbin = Path(manifest["shard_fbin"])
    fvecs = Path(manifest["shard_fvecs"])
    if read_fbin_shape(fbin) != (manifest["rows"], manifest["dim"]):
        raise ValueError("shard shape differs from manifest")
    if sha256_file(fbin) != manifest["shard_sha256"]:
        raise ValueError("shard fbin SHA-256 mismatch")
    if sha256_file(fvecs) != manifest["shard_fvecs_sha256"]:
        raise ValueError("shard fvecs SHA-256 mismatch")
    expected_fvecs = manifest["rows"] * (manifest["dim"] + 1) * 4
    if fvecs.stat().st_size != expected_fvecs:
        raise ValueError("shard fvecs byte length mismatch")
    return manifest


def create_deterministic_shard(
    output_dir: Path,
    rows: int = 8192,
    dim: int = 128,
    seed: int = 20260713,
    chunk_rows: int = 8192,
) -> dict:
    if rows <= 0 or dim <= 0 or chunk_rows <= 0:
        raise ValueError("deterministic shard dimensions must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shard_fbin = output_dir / "shard.fbin"
    shard_fvecs = output_dir / "shard.fvecs"

    def chunks() -> Iterator[np.ndarray]:
        columns = np.arange(dim, dtype=np.uint64)[None, :]
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            row_ids = np.arange(start, stop, dtype=np.uint64)[:, None]
            integers = (
                row_ids * np.uint64(1_315_423_911)
                + columns * np.uint64(2_654_435_761)
                + np.uint64(seed)
            ) % np.uint64(104_729)
            yield (integers.astype(np.float32) / np.float32(104_729.0)) - np.float32(0.5)

    _write_shard_formats(shard_fbin, shard_fvecs, chunks(), rows, dim)
    manifest = {
        "schema": "wiki10m-local-shard-v1",
        "source_kind": "deterministic_smoke",
        "seed": seed,
        "rows": rows,
        "dim": dim,
        "row_order": "generated_row_major",
        "shard_fbin": str(shard_fbin.resolve()),
        "shard_fvecs": str(shard_fvecs.resolve()),
        "shard_sha256": sha256_file(shard_fbin),
        "shard_fvecs_sha256": sha256_file(shard_fvecs),
    }
    (output_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def create_query_subset(
    source_fbin: Path,
    output_fbin: Path,
    count: int = 10_000,
    seed: int = 20260713,
) -> dict:
    rows, dim = read_fbin_shape(source_fbin)
    if count <= 0 or count > rows:
        raise ValueError(f"query count must be in [1, {rows}]")
    output_fbin = Path(output_fbin)
    if output_fbin.exists():
        raise FileExistsError(output_fbin)
    output_fbin.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    row_ids = np.sort(generator.choice(rows, size=count, replace=False))
    source = np.memmap(
        source_fbin,
        dtype="<f4",
        mode="r",
        offset=8,
        shape=(rows, dim),
    )
    with output_fbin.open("wb") as stream:
        stream.write(struct.pack("<ii", count, dim))
        np.asarray(source[row_ids], dtype="<f4", order="C").tofile(stream)
    manifest = {
        "schema": "wiki10m-local-shard-query-subset-v1",
        "source_kind": "shard_subset",
        "source_fbin": str(Path(source_fbin).resolve()),
        "output_fbin": str(output_fbin.resolve()),
        "rows": count,
        "dim": dim,
        "seed": seed,
        "row_ids_sha256": hashlib.sha256(
            row_ids.astype("<u8", copy=False).tobytes()
        ).hexdigest(),
        "query_sha256": sha256_file(output_fbin),
    }
    manifest_path = output_fbin.with_suffix(output_fbin.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def copy_query_file(source_query: Path, output_fbin: Path) -> dict:
    source_query = Path(source_query)
    output_fbin = Path(output_fbin)
    rows, dim = read_fbin_shape(source_query)
    if output_fbin.exists():
        raise FileExistsError(output_fbin)
    output_fbin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_query, output_fbin)
    source_sha256 = sha256_file(source_query)
    query_sha256 = sha256_file(output_fbin)
    if query_sha256 != source_sha256:
        raise ValueError("copied query SHA-256 does not match its source")
    manifest = {
        "schema": "wiki10m-local-shard-query-subset-v1",
        "source_kind": "external_query_copy",
        "source_query_fbin": str(source_query.resolve()),
        "source_query_sha256": source_sha256,
        "output_fbin": str(output_fbin.resolve()),
        "rows": rows,
        "dim": dim,
        "query_sha256": query_sha256,
    }
    manifest_path = output_fbin.with_suffix(output_fbin.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_quality_contract(
    manifest_path: Path,
    query_path: Path,
    groundtruth_path: Path,
    output_path: Path,
    expected_k: int = 10,
) -> dict:
    manifest_path = Path(manifest_path)
    query_path = Path(query_path)
    groundtruth_path = Path(groundtruth_path)
    output_path = Path(output_path)
    manifest = verify_shard_manifest(manifest_path)
    query_rows, query_dim = read_fbin_shape(query_path)
    query_manifest_path = query_path.with_suffix(query_path.suffix + ".manifest.json")
    try:
        query_manifest = json.loads(query_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid query manifest: {query_manifest_path}") from error
    shard_path = Path(manifest["shard_fbin"]).resolve()
    query_source_kind = query_manifest.get("source_kind", "shard_subset")
    if query_source_kind == "shard_subset":
        query_origin = Path(query_manifest.get("source_fbin", "")).resolve()
        if query_origin != shard_path:
            raise ValueError("query manifest source does not match the frozen shard")
        query_origin_sha256 = manifest["shard_sha256"]
    elif query_source_kind == "external_query_copy":
        query_origin = Path(query_manifest.get("source_query_fbin", "")).resolve()
        if not query_origin.is_file():
            raise ValueError("external query source is missing")
        query_origin_sha256 = sha256_file(query_origin)
        if query_manifest.get("source_query_sha256") != query_origin_sha256:
            raise ValueError("external query source SHA-256 mismatch")
    else:
        raise ValueError(f"unsupported query source kind: {query_source_kind!r}")
    if Path(query_manifest.get("output_fbin", "")).resolve() != query_path.resolve():
        raise ValueError("query manifest output path does not match the query")
    if query_manifest.get("rows") != query_rows or query_manifest.get("dim") != query_dim:
        raise ValueError("query manifest shape does not match the query")
    query_sha256 = sha256_file(query_path)
    if query_manifest.get("query_sha256") != query_sha256:
        raise ValueError("query manifest SHA-256 does not match the query")
    if query_dim != int(manifest["dim"]):
        raise ValueError("query dimension does not match the frozen shard")

    with groundtruth_path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"short ground-truth header: {groundtruth_path}")
    groundtruth_rows, groundtruth_k = struct.unpack("<II", header)
    if groundtruth_rows != query_rows or groundtruth_k != expected_k:
        raise ValueError(
            "ground-truth header does not match the query rows and expected K"
        )
    expected_bytes = 8 + groundtruth_rows * groundtruth_k * 8
    if groundtruth_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"ground-truth byte length mismatch: got "
            f"{groundtruth_path.stat().st_size}, expected {expected_bytes}"
        )

    contract = {
        "schema": "wiki10m-local-shard-quality-contract-v1",
        "construction_manifest": str(manifest_path.resolve()),
        "construction_manifest_sha256": sha256_file(manifest_path),
        "shard_fbin": str(shard_path),
        "shard_sha256": manifest["shard_sha256"],
        "shard_rows": int(manifest["rows"]),
        "shard_dim": int(manifest["dim"]),
        "query_fbin": str(query_path.resolve()),
        "query_sha256": query_sha256,
        "query_manifest": str(query_manifest_path.resolve()),
        "query_manifest_sha256": sha256_file(query_manifest_path),
        "query_source_kind": query_source_kind,
        "query_origin": str(query_origin),
        "query_origin_sha256": query_origin_sha256,
        "query_rows": query_rows,
        "query_dim": query_dim,
        "query_seed": query_manifest.get("seed"),
        "query_row_ids_sha256": query_manifest.get("row_ids_sha256"),
        "groundtruth": str(groundtruth_path.resolve()),
        "groundtruth_sha256": sha256_file(groundtruth_path),
        "groundtruth_rows": groundtruth_rows,
        "groundtruth_k": groundtruth_k,
        "metric": "sqeuclidean",
        "recall_at": expected_k,
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


class DiskannGraphReader:
    def __init__(self, path: Path, rows: int):
        self.path = Path(path)
        self.rows = rows
        if rows <= 0:
            raise ValueError("graph rows must be positive")
        self._stream = self.path.open("rb")
        try:
            header = self._stream.read(DISKANN_HEADER.size)
            if len(header) != DISKANN_HEADER.size:
                raise ValueError("short DiskANN graph header")
            size, self.max_degree, self.entry_point, self.frozen_points = (
                DISKANN_HEADER.unpack(header)
            )
            if (
                size != self.path.stat().st_size
                or self.max_degree <= 0
                or self.entry_point >= rows
                or self.frozen_points != 0
            ):
                raise ValueError("invalid DiskANN graph header")
            self._offsets = []
            for _ in range(rows):
                self._offsets.append(self._stream.tell())
                raw = self._stream.read(4)
                if len(raw) != 4:
                    raise ValueError("short DiskANN graph row")
                degree = struct.unpack("<I", raw)[0]
                if degree == 0 or degree > self.max_degree:
                    raise ValueError("invalid DiskANN graph degree")
                self._stream.seek(degree * 4, os.SEEK_CUR)
            if self._stream.tell() != size:
                raise ValueError("DiskANN graph has trailing or missing bytes")
        except Exception:
            self._stream.close()
            raise

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def neighbors(self, row: int) -> np.ndarray:
        if not 0 <= row < self.rows:
            raise IndexError(row)
        self._stream.seek(self._offsets[row])
        degree = struct.unpack("<I", self._stream.read(4))[0]
        return np.frombuffer(self._stream.read(degree * 4), dtype="<u4").copy()


def _validate_neighbors(row: int, neighbors: np.ndarray, rows: int) -> None:
    if neighbors.ndim != 1 or neighbors.size == 0:
        raise ValueError(f"row {row} has no neighbors")
    if np.any(neighbors >= rows):
        raise ValueError(f"row {row} contains out-of-range neighbor")
    if np.any(neighbors == row):
        raise ValueError(f"row {row} contains self edge")
    if np.unique(neighbors).size != neighbors.size:
        raise ValueError(f"row {row} contains duplicate neighbor")


def write_dense_diskann_graph(path: Path, graph: np.ndarray, entry_point: int) -> None:
    graph = np.asarray(graph, dtype="<u4", order="C")
    if graph.ndim != 2:
        raise ValueError("dense graph must be a two-dimensional array")
    rows, degree = graph.shape
    if rows <= 0 or degree <= 0:
        raise ValueError("dense graph must be non-empty")
    if not 0 <= entry_point < rows:
        raise ValueError("entry point is outside graph")
    for row, neighbors in enumerate(graph):
        _validate_neighbors(row, neighbors, rows)
    size = DISKANN_HEADER.size + rows * (4 + degree * 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(DISKANN_HEADER.pack(size, degree, entry_point, 0))
        for neighbors in graph:
            stream.write(struct.pack("<I", degree))
            neighbors.tofile(stream)


def write_tagore_graph_as_diskann(
    source: Path,
    output: Path,
    rows: int,
    entry_point: int,
    source_kind: str,
) -> None:
    source = Path(source)
    if rows <= 0 or not 0 <= entry_point < rows:
        raise ValueError("invalid graph dimensions or entry point")
    if source_kind not in {"tagore_cagra", "tagore_vamana"}:
        raise ValueError(f"unsupported Tagore graph kind: {source_kind}")
    with source.open("rb") as stream:
        source_max_degree = None
        if source_kind == "tagore_vamana":
            raw = stream.read(DISKANN_HEADER.size)
            if len(raw) != DISKANN_HEADER.size:
                raise ValueError("short Tagore-Vamana header")
            old_size, source_max_degree, _old_entry, frozen = DISKANN_HEADER.unpack(raw)
            if old_size != source.stat().st_size or frozen != 0:
                raise ValueError("invalid Tagore-Vamana header")
        records = []
        max_degree = 0
        for row in range(rows):
            raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError(f"short Tagore graph at row {row}")
            degree = struct.unpack("<I", raw)[0]
            neighbors = np.frombuffer(stream.read(degree * 4), dtype="<u4").copy()
            if neighbors.size != degree:
                raise ValueError(f"short Tagore neighbors at row {row}")
            _validate_neighbors(row, neighbors, rows)
            max_degree = max(max_degree, degree)
            records.append(neighbors)
        if stream.read(1):
            raise ValueError("Tagore graph has trailing bytes")
        if source_max_degree is not None and source_max_degree != max_degree:
            raise ValueError("Tagore-Vamana max degree disagrees with rows")
    size = DISKANN_HEADER.size + sum(4 + neighbors.nbytes for neighbors in records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(DISKANN_HEADER.pack(size, max_degree, entry_point, 0))
        for neighbors in records:
            stream.write(struct.pack("<I", neighbors.size))
            neighbors.tofile(stream)


def compute_common_entry_point(matrix: np.ndarray, chunk_rows: int = 8192) -> int:
    if matrix.ndim != 2 or matrix.shape[0] <= 0:
        raise ValueError("entry-point matrix must be non-empty and two-dimensional")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    rows = matrix.shape[0]
    center = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, rows, chunk_rows):
        center += np.asarray(matrix[start : start + chunk_rows], dtype=np.float64).sum(axis=0)
    center /= rows
    best_id, best_distance = 0, float("inf")
    for start in range(0, rows, chunk_rows):
        chunk = np.asarray(matrix[start : start + chunk_rows], dtype=np.float32)
        distances = np.square(chunk - center, dtype=np.float64).sum(axis=1)
        local = int(np.argmin(distances))
        if float(distances[local]) < best_distance:
            best_id, best_distance = start + local, float(distances[local])
    return best_id


def validate_diskann_graph(
    path: Path,
    rows: int,
    expected_max_degree: int = 64,
) -> dict:
    if expected_max_degree <= 0:
        raise ValueError("expected max degree must be positive")
    degree_values = []
    edges = 0
    with DiskannGraphReader(path, rows) as graph:
        if graph.max_degree > expected_max_degree:
            raise ValueError(
                f"graph max degree {graph.max_degree} exceeds {expected_max_degree}"
            )
        for row in range(rows):
            neighbors = graph.neighbors(row)
            _validate_neighbors(row, neighbors, rows)
            degree_values.append(int(neighbors.size))
            edges += int(neighbors.size)
        entry_point = graph.entry_point
        max_degree = graph.max_degree
    return {
        "schema": "wiki10m-local-shard-graph-validation-v1",
        "valid": True,
        "path": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "edges": edges,
        "entry_point": entry_point,
        "max_degree": max_degree,
        "min_row_degree": min(degree_values),
        "max_row_degree": max(degree_values),
        "mean_row_degree": edges / rows,
    }


class Nvtx:
    def __init__(self, backend=None) -> None:
        if backend is None:
            try:
                from cupy.cuda import nvtx as backend
            except (ImportError, RuntimeError) as error:
                raise RuntimeError("CuPy NVTX support is required for profiling") from error
        if not getattr(backend, "available", False):
            raise RuntimeError("CuPy NVTX support is unavailable")
        self._backend = backend

    @contextlib.contextmanager
    def range(self, name: str):
        self._backend.RangePush(name)
        try:
            yield
        finally:
            self._backend.RangePop()


def _set_cuda_device(device: int) -> None:
    from cuda.bindings import runtime as cudart

    (error,) = cudart.cudaSetDevice(device)
    if error.name != "cudaSuccess":
        raise RuntimeError(f"cudaSetDevice({device}) failed: {error.name}")


def _import_cagra(disable_mg_import: bool):
    if disable_mg_import:
        import cuvs

        mg_stub = types.ModuleType("cuvs.neighbors.mg")
        mg_stub.__path__ = []
        sys.modules.setdefault("cuvs.neighbors.mg", mg_stub)
    from cuvs.neighbors import cagra

    return cagra


def _import_tagore(module_dir: Path | None):
    if module_dir is None:
        default_module_dir = Path(__file__).resolve().parents[2] / "Tagore" / "build"
        module_dir = Path(os.environ.get("TAGORE_MODULE_DIR", default_module_dir))
    sys.path.insert(0, str(Path(module_dir).resolve()))
    import Tagore

    return Tagore


def _host_graph(graph) -> np.ndarray:
    if hasattr(graph, "copy_to_host"):
        graph = graph.copy_to_host()
    return np.asarray(graph, dtype="<u4", order="C")


def run_local_build(
    method: str,
    manifest_path: Path,
    output_dir: Path,
    profile_path: Path,
    device: int = 0,
    cagra_module=None,
    tagore_module=None,
    nvtx=None,
    tagore_module_dir: Path | None = None,
    cuvs_disable_mg_import: bool = False,
    profiled_timing: bool = False,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    injected_backend = cagra_module is not None or tagore_module is not None
    if not injected_backend:
        _set_cuda_device(device)
    if nvtx is None:
        nvtx = Nvtx()
    if method == "cagra" and cagra_module is None:
        cagra_module = _import_cagra(cuvs_disable_mg_import)
    if method != "cagra" and tagore_module is None:
        tagore_module = _import_tagore(tagore_module_dir)

    manifest = verify_shard_manifest(manifest_path)
    rows, dim = int(manifest["rows"]), int(manifest["dim"])
    shard_fbin = Path(manifest["shard_fbin"])
    shard_fvecs = Path(manifest["shard_fvecs"])
    matrix = np.memmap(shard_fbin, dtype="<f4", mode="r", offset=8, shape=(rows, dim))
    entry_point = compute_common_entry_point(matrix)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_graph = output_dir / "_mem.index"
    raw_graph = output_dir / f"{method}.raw.index"
    gnn_wall_ms = None
    prune_wall_ms = None
    export_wall_ms = None
    ptrs = None
    index = None

    local_start = time.perf_counter()
    try:
        with nvtx.range(f"WIKI10M_LOCAL_SHARD/{method}/LOCAL_BUILD"):
            if method == "cagra":
                params = cagra_module.IndexParams(
                    metric="sqeuclidean",
                    intermediate_graph_degree=128,
                    graph_degree=64,
                    build_algo="nn_descent",
                    nn_descent_niter=16,
                )
                start = time.perf_counter()
                with nvtx.range(f"WIKI10M_LOCAL_SHARD/{method}/GNN"):
                    index = cagra_module.build(params, matrix)
                gnn_wall_ms = (time.perf_counter() - start) * 1000.0

                start = time.perf_counter()
                with nvtx.range(f"WIKI10M_LOCAL_SHARD/{method}/EXPORT"):
                    graph = _host_graph(index.graph)
                    if graph.shape != (rows, 64):
                        raise ValueError(
                            f"unexpected CAGRA graph shape {graph.shape}; expected {(rows, 64)}"
                        )
                    write_dense_diskann_graph(output_graph, graph, entry_point)
                export_wall_ms = (time.perf_counter() - start) * 1000.0
            else:
                start = time.perf_counter()
                with nvtx.range(f"WIKI10M_LOCAL_SHARD/{method}/GNN"):
                    ptrs = tagore_module.GNN_descent(
                        128, rows, dim, 16, str(shard_fvecs)
                    )
                gnn_wall_ms = (time.perf_counter() - start) * 1000.0

                start = time.perf_counter()
                with nvtx.range(f"WIKI10M_LOCAL_SHARD/{method}/PRUNE"):
                    tagore_module.Pruning(
                        128,
                        rows,
                        dim,
                        64,
                        64,
                        ptrs,
                        "CAGRA" if method == "tagore_cagra" else "Vamana",
                        1.2,
                        str(raw_graph),
                    )
                prune_wall_ms = (time.perf_counter() - start) * 1000.0

                start = time.perf_counter()
                with nvtx.range(f"WIKI10M_LOCAL_SHARD/{method}/EXPORT"):
                    write_tagore_graph_as_diskann(
                        raw_graph,
                        output_graph,
                        rows=rows,
                        entry_point=entry_point,
                        source_kind=method,
                    )
                export_wall_ms = (time.perf_counter() - start) * 1000.0
        local_build_wall_ms = (time.perf_counter() - local_start) * 1000.0
    finally:
        ptrs = None
        index = None
        gc.collect()

    with DiskannGraphReader(output_graph, rows) as graph_reader:
        if graph_reader.entry_point != entry_point:
            raise ValueError("normalized graph entry point changed unexpectedly")

    analytical_hbm_bytes = {
        "vector_storage_bytes": rows * dim * 2,
        "graph_state_lower_bound_bytes": rows * (128 + 128) * 4,
        "candidate_distance_lower_bound_bytes": rows * (128 + 128) * 4,
        "output_graph_neighbor_bytes": rows * 64 * 4,
    }
    profile = {
        "schema": "wiki10m-local-shard-build-v1",
        "method": method,
        "profiled_timing": bool(profiled_timing),
        "device": device,
        "manifest": str(Path(manifest_path).resolve()),
        "shard_sha256": manifest["shard_sha256"],
        "rows": rows,
        "dim": dim,
        "metric": "sqeuclidean",
        "intermediate_graph_degree": 128,
        "graph_degree": 64,
        "iterations": 16,
        "topm": 64,
        "threshold": 1.2,
        "seed": 20260713,
        "entry_point": entry_point,
        "prune_semantics": (
            "included_in_cagra_build" if method == "cagra" else "explicit_tagore_pruning"
        ),
        "local_build_wall_ms": local_build_wall_ms,
        "gnn_wall_ms": gnn_wall_ms,
        "prune_wall_ms": prune_wall_ms,
        "prune_api_wall_ms": prune_wall_ms,
        "export_wall_ms": export_wall_ms,
        "analytical_hbm_bytes": analytical_hbm_bytes,
        "output_graph": str(output_graph.resolve()),
        "output_graph_bytes": output_graph.stat().st_size,
        "raw_graph": None if method == "cagra" else str(raw_graph.resolve()),
    }
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract a frozen ACE local shard")
    extract.add_argument("--source-fbin", type=Path, required=True)
    extract.add_argument("--plan", type=Path, required=True)
    extract.add_argument("--reordered", type=Path, required=True)
    extract.add_argument("--augmented", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--max-rows", type=int, default=1_100_000)
    extract.add_argument("--chunk-rows", type=int, default=8192)
    smoke = subparsers.add_parser(
        "make-smoke-shard", help="write a deterministic smoke-test shard"
    )
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--rows", type=int, default=8192)
    smoke.add_argument("--dim", type=int, default=128)
    smoke.add_argument("--seed", type=int, default=20260713)
    validate = subparsers.add_parser(
        "validate-graph", help="strictly validate a normalized graph"
    )
    validate.add_argument("--graph", type=Path, required=True)
    validate.add_argument("--rows", type=int, required=True)
    validate.add_argument("--expected-max-degree", type=int, default=64)
    validate.add_argument("--output", type=Path, required=True)
    queries = subparsers.add_parser(
        "make-query-subset", help="write a deterministic query subset"
    )
    queries.add_argument("--source-fbin", type=Path, required=True)
    queries.add_argument("--output-fbin", type=Path, required=True)
    queries.add_argument("--count", type=int, default=10_000)
    queries.add_argument("--seed", type=int, default=20260713)
    copy_query = subparsers.add_parser(
        "copy-query", help="copy and hash-bind an external canonical query file"
    )
    copy_query.add_argument("--source-fbin", type=Path, required=True)
    copy_query.add_argument("--output-fbin", type=Path, required=True)
    quality_contract = subparsers.add_parser(
        "make-quality-contract",
        help="bind the frozen shard, query subset, and exact ground truth",
    )
    quality_contract.add_argument("--manifest", type=Path, required=True)
    quality_contract.add_argument("--query", type=Path, required=True)
    quality_contract.add_argument("--groundtruth", type=Path, required=True)
    quality_contract.add_argument("--output", type=Path, required=True)
    quality_contract.add_argument("--expected-k", type=int, default=10)
    build = subparsers.add_parser("build", help="build one local graph")
    build.add_argument("--method", choices=METHODS, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--profile-json", type=Path, required=True)
    build.add_argument("--device", type=int, default=0)
    build.add_argument("--tagore-module-dir", type=Path)
    build.add_argument("--cuvs-disable-mg-import", action="store_true")
    build.add_argument("--profiled-timing", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "extract":
        extract_frozen_shard(
            source_fbin=args.source_fbin,
            plan_path=args.plan,
            reordered_path=args.reordered,
            augmented_path=args.augmented,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            chunk_rows=args.chunk_rows,
        )
        print(args.output_dir / "shard_manifest.json")
        return 0
    if args.command == "make-smoke-shard":
        create_deterministic_shard(
            output_dir=args.output_dir,
            rows=args.rows,
            dim=args.dim,
            seed=args.seed,
        )
        print(args.output_dir / "shard_manifest.json")
        return 0
    if args.command == "validate-graph":
        report = validate_diskann_graph(
            args.graph,
            rows=args.rows,
            expected_max_degree=args.expected_max_degree,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(args.output)
        return 0
    if args.command == "make-query-subset":
        create_query_subset(
            source_fbin=args.source_fbin,
            output_fbin=args.output_fbin,
            count=args.count,
            seed=args.seed,
        )
        print(args.output_fbin)
        return 0
    if args.command == "copy-query":
        copy_query_file(
            source_query=args.source_fbin,
            output_fbin=args.output_fbin,
        )
        print(args.output_fbin)
        return 0
    if args.command == "make-quality-contract":
        write_quality_contract(
            manifest_path=args.manifest,
            query_path=args.query,
            groundtruth_path=args.groundtruth,
            output_path=args.output,
            expected_k=args.expected_k,
        )
        print(args.output)
        return 0
    if args.command == "build":
        run_local_build(
            method=args.method,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            profile_path=args.profile_json,
            device=args.device,
            tagore_module_dir=args.tagore_module_dir,
            cuvs_disable_mg_import=args.cuvs_disable_mg_import,
            profiled_timing=args.profiled_timing,
        )
        print(args.profile_json)
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
