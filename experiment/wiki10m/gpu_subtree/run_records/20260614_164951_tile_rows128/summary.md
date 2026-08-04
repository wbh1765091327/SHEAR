# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_164951_tile_rows128
- Date: 2026-06-14
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: explicit persistent leaf top-k tile rows parameter
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: yes, added `--gpu-subtree-persistent-leaf-tile-rows 128`
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 18:38.63
- Max RSS: 15113252 kbytes
- `_mem.index` bytes: 881468220
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- `persistent_leaf_tile_rows`: 128
- `persistent_kernel_event_ms`: 874807
- `persistent_total_ms`: 874822
- `persistent_topk_rows`: 80000000
- `persistent_leaf_count`: 388461
- `persistent_topk_claims`: 803626
- `persistent_leaf_pushes`: 803626
- `persistent_partition_claims`: 10653
- `persistent_large_node_splits`: 4155
- `persistent_large_node_max_rows`: 1378181
- `persistent_leaf_size_max`: 1280
- `persistent_leaf_max`: 1280
- `persistent_queue_overflow_flags`: 0
- `persistent_success_count`: 64

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Base data: 10000000 points, dim=1024
- Graph: 10000000 nodes, start=6843085, max_degree=64
- Queries: 10000
- Search: 10000 queries in 3.92s (2550.8 QPS)
- Recall@10: 0.8575
- Result file: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_164951_tile_rows128/search_results.fbin
- Result bytes: 400008

## Verdict

- Build exit status: pass
- Search exit status: pass
- Runbook persistent queue invariants: mostly pass
- Runbook `persistent_leaf_size_max <= 1024` gate: fail, observed 1280 because `persistent_leaf_max` is 1280
- Compared baseline: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_123434_leaf_tile_topk`
- Baseline build wall: 18:31.99; current build wall: 18:38.63
- Baseline `persistent_kernel_event_ms`: 866292; current: 874807
- Baseline Recall@10: 0.8554; current: 0.8575
- Baseline QPS: 2753.8; current QPS: 2550.8
- Notes: current profile explicitly records `persistent_leaf_tile_rows=128`; previous baseline predates this field and therefore records it as absent. Do not treat this run as a full runbook pass until the leaf-size gate is reconciled with the current semantic leaf max policy.
