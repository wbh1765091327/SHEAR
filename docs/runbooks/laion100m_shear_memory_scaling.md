# LAION100M SHEAR Memory-Scaling Runbook

## Purpose

This runbook measures only SHEAR construction under progressively tighter
whole-process cgroup limits. It does not repeat the multi-method Motivation
comparison.

**No search** is performed. Query files, ground truth, graph-search
quantization, and search-quality measurements are outside this experiment.
Successful builds receive strict graph-structure validation and a sampled
`graph_quality` report only.

## Frozen data and memory contract

```text
base:          /data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin
header:        100,000,000 x 512 float32
file bytes:    204,800,000,008
payload bytes: 204,800,000,000
SQ sidecar:     52,000,000,000 bytes
```

The cgroup percentages use the 204,800,000,000-byte vector payload and are
rounded down to a 4096-byte boundary. The runner executes them in the fixed
order 50% → 40% → 30% → 20%.

| Label | cgroup `memory.max` | GiB | Ideal 80% \(B_h\) upper bound | Full sidecar |
|---|---:|---:|---:|---|
| p50 | 102,400,000,000 | 95.367 | 81,920,000,000 B | fits |
| p40 | 81,920,000,000 | 76.294 | 65,536,000,000 B | fits |
| p30 | 61,440,000,000 | 57.220 | 49,152,000,000 B | does not fit in \(B_h\) |
| p20 | 40,960,000,000 | 38.147 | 32,768,000,000 B | does not fit even in `memory.max` |

The table contains theoretical upper bounds. Report the actual
`memory_budget_bytes` selected in each `profile.json`; do not substitute the
80% values when current cgroup availability selects a smaller budget.

## Frozen SHEAR configuration

All four runs use:

```text
distance:                 squared L2
R / Lbuild:               64 / 100
threads:                  8
leaf min/max/KNN:         256 / 1280 / 2
sampling:                 0.05
fanout:                   8 / 3 / 1
root-risk candidate R:    1,2,4,8
seed:                     42
partitioner:              GPU subtree
D0 partition:             GPU K-Means coarse
SQ staging:               GPU-D0, SQ8 WMMA sidecar rebuild
persistent execution:     fused, WMMA, ticket batch 8
GPU workspace:            16 GiB
OOM I/O:                  direct
host budget:              automatic, --oom-memory-budget-gb 0
HashPrune:                SoftTier-Q3, cap=3, alpha=1.02
```

The runner explicitly passes the SoftTier mode, row cap, and priority alpha so
that later CLI-default changes cannot silently change the experiment.
Because the 204.8 GB raw payload exceeds every tested host budget, the runner
also explicitly enables Direct-I/O. GPU-D0 uses a bounded one-shard-ahead
pipeline, so raw reads overlap the current shard's GPU quantization/assignment
and SQ sidecar output. In OOM mode, non-resident D1+ SQ sidecar suffix reads
also use Direct-I/O.

## Cache, swap, and isolation

- Host swap must be disabled before preflight.
- Each cap is executed in a new systemd cgroup with the exact `memory.max`,
  `memory.swap.max=0`, and `OOMPolicy=stop`.
- Page cache is dropped immediately before every cap. There is no prewarm.
- No other GPU compute process or `build_shear_index` process may be active.
- Every cap records `memory.current`, `memory.peak`, `memory.stat`,
  `memory.events`, and `io.stat` once per second.
- A per-cap OOM or runtime failure does not suppress later caps. An
  environment-level preflight failure stops the batch before construction.

## Output

Formal results are written under:

```text
/data_staging/wbh/laion100m_shear_memory_scaling/runs/<run-id>/
```

The batch contains:

```text
run_manifest.json
schedule.tsv
summary.tsv
summary.md
provenance/
p50/
p40/
p30/
p20/
```

Each cap retains the command, environment, cgroup limits and samples, systemd
journal, host/GPU telemetry, GNU time output, profile, status, and postprocess
logs. After a successful profile and graph validation, only the generated
`index`, `oom_temp`, and `sq_sidecar` directories are deleted. Their sizes and
the validation evidence remain. OOM and failed runs retain their artifacts.

## Static checks

```bash
cd /path/to/SHEAR
python3 script/test_laion100m_shear_memory_scaling.py
bash script/laion100m_shear_memory_scaling.sh help
```

These commands do not create a formal result directory.

## Preflight

```bash
sudo -n true
bash script/laion100m_shear_memory_scaling.sh preflight
```

Preflight checks the dataset header and byte size, cgroup v2, disabled swap,
required executables, idle GPU, absence of another SHEAR builder, and at least
350 GB of free staging space. It does not start construction.

## Bounded cgroup self-test

```bash
bash script/laion100m_shear_memory_scaling.sh self-test
```

The self-test does not read LAION100M and does not use the GPU. It runs one
small allocation that must complete and one allocation that must be
OOM-classified, then removes its own temporary directory.

## Formal execution — separate approval required

Do not run the following command until the user gives explicit approval for
the long LAION100M task:

```bash
bash script/laion100m_shear_memory_scaling.sh formal
```

The runner creates a fresh timestamped batch. It refuses to overwrite an
existing batch root and uses an exclusive lock to prevent concurrent formal
runs.

## Rebuild a summary without rerunning construction

```bash
bash script/laion100m_shear_memory_scaling.sh summarize \
  /data_staging/wbh/laion100m_shear_memory_scaling/runs/<run-id>
```

`summary.tsv` always contains p50, p40, p30, and p20 in that order. A missing
cap is reported as `not_run`.

## Metric interpretation

- `cgroup_peak_dram_bytes` is the peak of live cgroup `memory.peak` samples.
- `actual_bh_bytes` is the measured automatic SHEAR budget.
- `sidecar_resident_fraction` uses measured resident bytes divided by
  52,000,000,000.
- `sidecar_file_logical_bytes` equals `sq_wmma_sidecar_file_rows × 520 B`.
  It is a logical file-backed gather volume, not physical NVMe traffic.
- Physical reads and writes are the last-minus-first cgroup `io.stat`
  counters and include every file accessed by the process tree.
- Candidate spill bytes are directly reported by the profile. The current
  profile does not expose candidate reread bytes independently, so that field
  remains `N/A`.
- A cap is `completed` only after the index passes the strict stream validator
  and `graph_quality` finishes. `oom`, `failed_validation`, and
  `failed_runtime` are distinct outcomes and are never silently retried.

This experiment supports conclusions about construction completion, memory,
critical-path I/O, and graph structural integrity. It does not independently
establish search quality.
