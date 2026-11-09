# Breaking the Memory Wall in GPU-Accelerated ANN Graph Construction

This repository contains **SHEAR**, a single-machine GPU--CPU research system for constructing graph-based approximate
nearest-neighbor indexes when the dataset and intermediate construction state exceed GPU memory.

[![SHEAR workflow](docs/pdf/Fig2-Workflow.png)](docs/pdf/Fig2-Workflow.pdf)

*SHEAR workflow. Click the figure to open the original PDF.*

SHEAR combines three mechanisms:

- **GPU-memory-aware task formation** creates overlapping subtree tasks that fit a fixed device workspace.
- **TESSA** persistently executes recursive assignment and SQ8/Tensor Core leaf evaluation on the GPU.
- **TierLift** reduces streamed candidate records in bounded host memory and spills source ranges to SSD when needed.

The public artifact contains the active construction/search implementation, focused tests, reproducibility runbooks,
paper figures, and a curated text-only experiment archive. Raw datasets, generated indexes, and retired implementations
are not included.

## Requirements

- Linux and the Rust nightly toolchain selected by `rust-toolchain.toml`
- LAPACK, LAPACKE, BLAS, a C++17 compiler, and OpenMP
- An AVX2/AVX-512-capable CPU
- An NVIDIA GPU with a CUDA 12-compatible driver for the CUDA path

On Debian or Ubuntu:

```bash
sudo apt-get install build-essential liblapack-dev liblapacke-dev libblas-dev
```

The required RaBitQ headers are pinned under `third_party/rabitq/`; no sibling checkout is required.

## Build

Build the CPU workspace:

```bash
cargo build --release
```

Build the SHEAR index constructor with CUDA support:

```bash
cargo build --release -p build_shear_index --features cuda
```

Build the search validator and TESSA replay tool:

```bash
cargo build --release -p search_shear_index
cargo build --release -p tessa_operator_replay --features cuda
```

Run the focused integration tests:

```bash
cargo test -p shear_core --test pipnn_oom_build_test
cargo test -p shear_core --test root_risk_budget_test
```

## Reproducing the Evaluation

Large datasets, generated indexes, and binary profiler artifacts stay outside the repository. Adapt the data/output
roots in these runbooks to the local machine:

- [Wiki10M GPU-subtree measurement](docs/runbooks/wiki10m_gpu_subtree_measurement.md)
- [Wiki35M GPU-subtree measurement](docs/runbooks/wiki35m_gpu_subtree_measurement.md)
- [TESSA progressive ablation](docs/runbooks/tessa_progressive_ablation.md)
- [Ordinary SQ8 GPU-D0 baseline and timing protocol](docs/runbooks/sq8_gpu_d0_baseline.md)
- [SQ4+ target leaf sizing experiment](docs/runbooks/sq4_target_leaf_policy.md)
- [SQ4+ shared full-distance leaf experiment](docs/runbooks/sq4_leaf_shared_distance.md)
- [SQ4+ scale-specific configurations and runtime audits](docs/runbooks/sq4_plus_profiles.md)

Use `python3 script/run_sq4_plus.py --binary /path/to/build_shear_index
--data-path /path/to/base.fbin --run-dir /path/to/new-run` for current SQ4+ builds.
The input row count selects the validated 10M recipe (fanout 6, D0 sample 0.002,
256 GiB host budget) or the 100M paper common parameters (fanout 8, D0 sample
0.01, automatic host budget). Both pin 8% capacity ratios and audit the actual
profile. Generic bare CLI defaults are not a paper reproduction recipe.
The documented SQ4+ quantization and D0 training differences still apply.

SQ4+ CLI builds default to one code per point against its nearest primary encoding center;
use `--gpu-subtree-sq4-primary-center=false` for the per-partition reference.
Native INT4 kernels default to projection reuse, 8×32 assignment, and cooperative
PUBLISH copies (`SHEAR_PIPNN_SQ8_WMMA_OPT_MASK=525824`). An explicit mask overrides
these defaults, including `0` to disable them. SQ8 retains a default mask of `0`.
Triangular leaves, microtile operand reuse, and CUTLASS remain opt-in experiments.
SQ4+ primary-center builds compute HashPrune sketches on the CPU from the same
4096-row raw batches used by D0. With a CUDA backend, 512 dimensions,
a 64-center primary codebook, top-k up to 32, and fused CPU sketch enabled, D0 assignment and
residual encoding now default to `gpu-exact` when `SHEAR_PIPNN_SQ4_D0_BACKEND`
is unset. It reproduces the pinned `ndarray 0.14` / `matrixmultiply 0.2.4` FMA
arithmetic, including 256-wide accumulation panels, with no CPU assignment rechecks.
Other configurations retain the CPU path; GPU errors or unsupported CPU
floating-point modes stop a selected GPU build instead of silently falling back.

SQ4+ now defaults to `budget-kmeans-topk` D0 routing, using the same
`plan_root_leaders` capacity planner as the paper path:
`K_capacity = ceil(N * fanout_top / C_target)`, bounded by the quality floor,
root leader cap, and dataset size. SQ4+ CLI defaults to the paper capacity
ratios: high, target, and low are all `0.08`. Explicit ratio options override
these defaults; other modes retain their existing defaults. Library callers
set the three corresponding `PiPNNParams` fields to `0.08`. The leader defaults
are a floor of 64 and a cap of 10000; a historical explicit cap of 64 still prevents growth and must be removed
from new budget-based commands. The existing FP32 center trainer is retained.
Routing leaders can grow independently of the 64-center primary codebook, keeping
the existing INT4 projection and D1 kernels. When routing uses 64 leaders, both
tables are shared. Otherwise, GPU routing and primary-center assignment reuse the
same uploaded raw batch; primary assignment adds computation and one 32-bit ID
per point downloaded, with no extra raw-vector upload. Explicit
`--gpu-subtree-d0-partition-mode tagore-kmeans-topk` preserves the fixed-count
reference; its `--gpu-subtree-d0-tagore-partitions` and `--gpu-subtree-d0-tagore-k`
do not determine the budget mode's routing count or fanout. Increasing the planned
count controls average partition size and does not guarantee every partition fits.
See the [paper 8% validation](runs/sq4_budget_paper8_20261108/report.md) and
[budget routing implementation tests](runs/sq4_budget_d0_20261108/report.md).

Explicit backend settings override the default: `cpu` restores CPU assignment and
encoding; `gpu-encode` isolates GPU encoding; `gpu` retains the earlier GPU
assignment with CPU compatibility checks; `gpu-exact` explicitly selects the new
path and rejects unsupported shapes. GPU D0 modes require fused CPU sketch and
produce one packed INT4 code per point. With separate routing and primary tables,
use `gpu-exact` or `cpu`; the older GPU diagnostic modes require a shared table.

The sketch backend still defaults to `cpu`.
`SHEAR_PIPNN_SQ4_D0_SKETCH_BACKEND=gpu` selects the separate GPU sketch experiment,
and `SHEAR_PIPNN_SQ4_D0_FUSED_SKETCH=0` restores the historical separate CPU pass;
either keeps CPU D0 when no D0 backend override is provided. Per-partition SQ4+
and ordinary SQ8 retain their existing paths. See the
[CPU fusion results](runs/sq4_d0_fused_sketch_20261107/report.md),
[GPU D0 validation](runs/sq4_gpu_d0_exact_20261108/report.md), and
[default-entry verification](runs/sq4_gpu_d0_default_20261108/report.md).
- [Four-dataset build and search workflow](docs/runbooks/five_method_four_dataset_build_search.md)

Curated commands, text logs, profiles, metrics, and validation outputs from the GPU experiments are available under
[`experiment/`](experiment/).

## Repository Layout

```text
.
├── shear-core/       SHEAR library and graph-construction implementation
├── apps/             Constructor, search validator, and TESSA replay binaries
├── docs/paper/       Paper figures and supporting material
├── docs/runbooks/    Reproducible measurement procedures
├── experiment/       Curated text-only experiment records
├── script/           Scripts directly required by the retained runbooks
├── local-sync/       Internal synchronization support crate
└── third_party/      Pinned header-only dependencies and licenses
```

## Paper Assets

Figures and supporting material for **Breaking the Memory Wall in GPU-Accelerated ANN Graph Construction** are under
[`docs/paper/`](docs/paper/). Draft sections `00` through `09` are intentionally excluded from this public artifact.
Citation metadata will be added when the paper is publicly available.

Upstream lineage and third-party licensing information is recorded in [NOTICE.md](NOTICE.md).

## Local Checkout Setup

Source the local environment before building or executing commands:

```bash
cd SHEAR
source script/local_env.sh
cargo --config net.git-fetch-with-cli=false build --release -p build_shear_index -p tessa_operator_replay --features cuda
bash script/run_local_gpu_smoke.sh
```

`local_env.sh` derives `SHEAR_ROOT` from its own location, defaults datasets to
`$SHEAR_ROOT/data` and outputs/locks to `$SHEAR_ROOT/runs`, and adds the locally
unpacked OpenBLAS libraries to the build/runtime paths. Override
`SHEAR_DATA_ROOT` and `SHEAR_OUTPUT_ROOT` before sourcing it to use other disks.
The shell experiment runners use these roots. Historical runbooks and archived
experiment records still describe the original machine; set their dataset,
output, sidecar, and external-tool variables explicitly before reproducing them.
No full evaluation datasets are included or downloaded by this setup.

The local smoke command creates deterministic synthetic data (96 vectors,
8 dimensions), exact ground truth for 8 queries, builds with the CUDA GPU-subtree
backend, verifies GPU success counters and graph structure, and searches the
result with the exact CPU search validator. Results are under `runs/local-smoke-*`.
It is an execution check, not a full-dataset quality or performance benchmark.
The C++ RaBitQ paths still require AVX-512; this smoke does not validate those paths.
