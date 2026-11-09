#!/usr/bin/env python3
"""Compare plain SQ8 and a rotated residual SQ8 sidecar on an fbin prefix.

Uses only the Python standard library. The plain SQ8 code is regenerated from
the original vectors; the rotated residual code is read from the sidecar.
"""

import argparse
import array
import json
import math
import struct
import sys
from pathlib import Path


MASK64 = (1 << 64) - 1
SIDECAR_HEADER_BYTES = 128
SIDECAR_META_ROW_BYTES = 12


def read_floats(path: Path, count: int, offset: int = 0) -> array.array:
    with path.open("rb") as handle:
        handle.seek(offset)
        values = array.array("f")
        values.fromfile(handle, count)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def rotation_signs(dim: int, seed: int) -> list[list[int]]:
    signs = []
    for round_id in range(4):
        row = []
        for axis in range(dim):
            value = (
                seed
                ^ 0x5261426974512B
                ^ ((round_id * 0x9E3779B97F4A7C15) & MASK64)
                ^ ((axis * 0xBF58476D1CE4E5B9) & MASK64)
            )
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
            row.append(1 if ((value ^ (value >> 31)) & 1) == 0 else -1)
        signs.append(row)
    return signs


def rotate(vector: list[float], signs: list[list[int]]) -> list[float]:
    values = vector.copy()
    dim = len(values)
    factor = 1.0 / math.sqrt(dim)
    for sign_row in signs:
        for axis in range(dim):
            values[axis] *= sign_row[axis]
        width = 1
        while width < dim:
            step = width * 2
            for base in range(0, dim, step):
                for offset in range(width):
                    left = values[base + offset]
                    right = values[base + width + offset]
                    values[base + offset] = left + right
                    values[base + width + offset] = left - right
            width = step
        for axis in range(dim):
            values[axis] *= factor
    return values


def plain_sq8(vector: list[float]) -> tuple[float, list[int]]:
    max_abs = max(map(abs, vector))
    if max_abs == 0.0:
        return 1.0, [0] * len(vector)
    scale = max_abs / 127.0
    codes = []
    for value in vector:
        scaled = value / scale
        rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
        codes.append(max(-127, min(127, rounded)))
    return scale, codes


def squared_l2(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_fbin", type=Path)
    parser.add_argument("sidecar_prefix", type=Path)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--anchors", type=int, default=20)
    args = parser.parse_args()

    with args.base_fbin.open("rb") as handle:
        base_rows, dim = struct.unpack("<II", handle.read(8))
    if args.rows < 2 or args.rows > base_rows or args.anchors < 1:
        parser.error("rows must be in [2, base_rows] and anchors must be positive")

    prefix = str(args.sidecar_prefix)
    meta = Path(prefix + ".meta").read_bytes()[: SIDECAR_HEADER_BYTES + args.rows * SIDECAR_META_ROW_BYTES]
    if len(meta) != SIDECAR_HEADER_BYTES + args.rows * SIDECAR_META_ROW_BYTES:
        parser.error("sidecar metadata is shorter than requested rows")
    if meta[:8] != b"PGSRR2\0\0" or struct.unpack_from("<I", meta, 8)[0] != 2:
        parser.error("expected a rotated residual SQ8 v2 sidecar")
    sidecar_rows, sidecar_dim, padded_dim, center_count = struct.unpack_from("<QIII", meta, 12)
    seed = struct.unpack_from("<Q", meta, 32)[0]
    if sidecar_rows != base_rows or sidecar_dim != dim or padded_dim != dim:
        parser.error("base and sidecar shapes differ, or padded dimension is not supported")
    centers = read_floats(Path(prefix + ".centers"), center_count * dim)
    with Path(prefix + ".codes").open("rb") as handle:
        code_bytes = handle.read(args.rows * padded_dim)
    if len(code_bytes) != args.rows * padded_dim:
        parser.error("sidecar code file is shorter than requested rows")
    raw = read_floats(args.base_fbin, args.rows * dim, offset=8)
    signs = rotation_signs(dim, seed)

    raw_vectors = []
    plain_codes = []
    plain_scales = []
    decoded_vectors = []
    plain_errors = []
    rotated_errors = []
    for row in range(args.rows):
        original = list(raw[row * dim : (row + 1) * dim])
        scale, code = plain_sq8(original)
        center_id, residual_scale, _ = struct.unpack_from(
            "<Iff", meta, SIDECAR_HEADER_BYTES + row * SIDECAR_META_ROW_BYTES
        )
        if center_id >= center_count:
            parser.error(f"invalid center ID at row {row}")
        residual_code = struct.unpack_from(f"<{dim}b", code_bytes, row * padded_dim)
        center = centers[center_id * dim : (center_id + 1) * dim]
        decoded = [center[axis] + residual_scale * residual_code[axis] for axis in range(dim)]
        rotated = rotate(original, signs)
        plain_errors.append(squared_l2(original, [scale * value for value in code]) / dim)
        rotated_errors.append(squared_l2(rotated, decoded) / dim)
        raw_vectors.append(original)
        plain_codes.append(code)
        plain_scales.append(scale)
        decoded_vectors.append(decoded)

    anchor_ids = sorted({row * args.rows // min(args.rows, args.anchors) for row in range(min(args.rows, args.anchors))})
    plain_hits = 0
    rotated_hits = 0
    topk = min(10, args.rows - 1)
    for anchor in anchor_ids:
        original = raw_vectors[anchor]
        code = plain_codes[anchor]
        decoded = decoded_vectors[anchor]
        original_norm = sum(value * value for value in original)
        candidate_distances = []
        plain_distances = []
        rotated_distances = []
        for row in range(args.rows):
            if row == anchor:
                continue
            candidate = raw_vectors[row]
            exact = squared_l2(original, candidate)
            candidate_norm = sum(value * value for value in candidate)
            code_dot = sum(a * b for a, b in zip(code, plain_codes[row]))
            approx = original_norm + candidate_norm - 2.0 * plain_scales[anchor] * plain_scales[row] * code_dot
            candidate_distances.append((exact, row))
            plain_distances.append((approx, row))
            rotated_distances.append((squared_l2(decoded, decoded_vectors[row]), row))
        truth = {row for _, row in sorted(candidate_distances)[:topk]}
        plain_hits += len(truth & {row for _, row in sorted(plain_distances)[:topk]})
        rotated_hits += len(truth & {row for _, row in sorted(rotated_distances)[:topk]})

    plain_mse = sum(plain_errors) / args.rows
    rotated_mse = sum(rotated_errors) / args.rows
    print(json.dumps({
        "rows": args.rows,
        "dim": dim,
        "centers": center_count,
        "rotation_seed": seed,
        "plain_sq8_mean_per_component_mse": plain_mse,
        "rotated_residual_sq8_mean_per_component_mse": rotated_mse,
        "mse_ratio_rotated_over_plain": rotated_mse / plain_mse,
        "plain_sq8_topk_hits": plain_hits,
        "rotated_residual_sq8_topk_hits": rotated_hits,
        "topk_total": len(anchor_ids) * topk,
        "anchor_count": len(anchor_ids),
        "candidate_count_per_anchor": args.rows - 1,
    }, indent=2))


if __name__ == "__main__":
    main()
