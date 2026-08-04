# Wiki10M True WMMA Post-Optimization Measurement

- Suite root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260615_192942_true_wmma_postopt`
- Baseline: `persistent-tiled-gemm-fp32`
- Candidate: `persistent-wmma-tf32`
- Common data: Wiki10M base, Wiki35M query, Wiki10M GT100
- Common build params: gpu-subtree, CUDA backend, 16 GiB GPU workspace, 200 GiB OOM budget, seed 42, leaf size 140..640, leaf-knn 2

## Results

| Backend | Build wall | GPU-subtree total | persistent_total | resident event | host_control | Max RSS KiB | Search s | QPS | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistent-tiled-gemm-fp32` | 9:32.55 | 446.240s | 317.224s | 317.223s | 120.582s | 15098480 | 3.96 | 2528.0 | 0.8428 |
| `persistent-wmma-tf32` | 9:43.38 | 456.393s | 328.467s | 328.466s | 120.895s | 15092788 | 3.69 | 2713.2 | 0.8422 |

## Delta vs Baseline

- GPU-subtree total: +10.153s (+2.28%)
- persistent_total: +11.243s (+3.54%)
- resident event: +11.243s (+3.54%)
- host_control: +0.313s (+0.26%)
- Build wall: candidate `9:43.38` vs baseline `9:32.55`
- Recall@10 delta: -0.0006
- QPS delta: +185.2

## Pass Checks

- `persistent-tiled-gemm-fp32`: build/search exit 0, queue overflow 0, claim/pop counters match, `_mem.index` bytes `902184536`, search result bytes 400008.
- `persistent-wmma-tf32`: build/search exit 0, queue overflow 0, claim/pop counters match, `_mem.index` bytes `902025936`, search result bytes 400008.

## Interpretation

The true fragment-level WMMA implementation is functionally admitted on Wiki10M, but this first WMMA tile version is slower than the tiled FP32 baseline on GPU-subtree total. The likely reason is that the current WMMA path packs A/B tiles into per-warp shared scratch and then does fused top-k per row, so Tensor Core dot throughput is offset by tile packing, shared-memory traffic, and per-tile synchronization overhead. The persistent-tiled-gemm-fp32 baseline remains faster for this leaf shape.
