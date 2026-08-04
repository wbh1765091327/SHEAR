# Wiki10M GPU-subtree WMMA Measurement

- run_id: `20260616_185933_staged_async_wmma`
- date: `2026-06-16`
- git_commit: `3ff96972031a1396535e6174ae180eadf2b9e0c0`
- build_exit: `0`
- search_exit: `0`
- build_wall: `4:31.25`
- max_rss_kb: `15120892`
- mem_index_bytes_before_cleanup: `901620340`

## Inputs

- base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- gt: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`

## Explicit Build Backends

- `--partitioner gpu-subtree`
- `--gpu-subtree-assignment-backend persistent-wmma-tf32`
- `--gpu-subtree-leaf-topk-backend persistent-wmma-tf32`

## Profile Evidence

- root_assignment_backend: `cpu`
- assignment_backend: `persistent-wmma-tf32`
- subtree_assignment_backend: `persistent-wmma-tf32`
- leaf_topk_backend: `persistent-wmma-tf32`
- d0_wall_ms: `20884`
- d1plus_backend_ms: `197699`
- d1plus_producer_ms: `203273`
- async_flush_ms: `46114`
- persistent_kernel_event_ms: `55830`
- persistent_total_ms: `55831`
- topk_kernel_ms: `55830`
- leaf_count: `996294`
- compact_topk_batches: `996294`
- compact_topk_rows: `80000000`
- cpu_fallbacks: `0`
- compact_unsupported_count: `0`

## Search Validation

- search_l: `100`
- search_threads: `8`
- qps: `2548.5 QPS`
- recall_at_10: `0.8406`
- results: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_185933_staged_async_wmma/search_results.fbin`

## Cleanup

`_mem.index` cleanup is performed after this summary is written; see `cleanup_status.txt` and `metadata.txt`.
