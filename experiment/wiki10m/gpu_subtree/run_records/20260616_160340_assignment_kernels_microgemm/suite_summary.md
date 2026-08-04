# Wiki10M Assignment Kernel Suite Summary

- Suite root: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_160340_assignment_kernels_microgemm`

| Backend | Build Wall | GPU-subtree total_ms | persistent_total_ms | resident kernel event_ms | Recall | QPS | mem index bytes |
|---|---:|---:|---:|---:|---|---|---:|
| `persistent_cuvs_bruteforce_fp32` | `10:28.64` | `497639` | `364396` | `364396` | `Recall@10: 0.8396` | `Search: 10000 queries in 4.14s (2413.0 QPS)` | `901792860` |
| `persistent_tiled_gemm_fp32` | `6:45.27` | `273697` | `141106` | `141104` | `Recall@10: 0.8437` | `Search: 10000 queries in 3.77s (2654.5 QPS)` | `901515056` |
| `persistent_wmma_tf32` | `6:03.78` | `233526` | `100400` | `100399` | `Recall@10: 0.8416` | `Search: 10000 queries in 3.57s (2800.0 QPS)` | `901859948` |
