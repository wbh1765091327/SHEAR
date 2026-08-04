# Wiki10M GPU-Subtree Measurement

- Run ID: 20260616_182818_staged_async
- Date: 2026-06-16T18:28:18+08:00
- Git commit: 3ff96972031a1396535e6174ae180eadf2b9e0c0
- Code changes under test: GPU-subtree D0/D1 split plus bounded ordered async compact top-k flush
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: yes; removed obsolete `--gpu-subtree-output compact-topk` because current CLI rejects it and `gpu-subtree` OOM path now directly uses compact top-k
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: 9:35.14
- Max RSS: 15117772 kbytes
- `_mem.index` bytes: 901735048
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Notable `profile.json` fields:
  - `gpu_subtree.d0_wall_ms`: 20930
  - `gpu_subtree.d0_child_run_count`: 64
  - `gpu_subtree.d0_child_points`: 40000000
  - `gpu_subtree.d0_child_max_len`: 1378181
  - `gpu_subtree.d0_child_p50_len`: 543772
  - `gpu_subtree.d0_child_p95_len`: 1221235
  - `gpu_subtree.d0_child_p99_len`: 1378181
  - `gpu_subtree.d1plus_producer_ms`: 507105
  - `gpu_subtree.d1plus_child_replay_ms`: 223
  - `gpu_subtree.d1plus_backend_ms`: 501661
  - `gpu_subtree.d1plus_push_wait_ms`: 23
  - `gpu_subtree.d1plus_queue_max_bytes`: 55059956
  - `gpu_subtree.d1plus_queue_max_batches`: 47875
  - `gpu_subtree.async_flush_ms`: 46026
  - `gpu_subtree.async_flush_worker_ms_sum`: 46026
  - `gpu_subtree.async_flush_batches`: 999011
  - `gpu_subtree.async_flush_rows`: 80000000
  - `gpu_subtree.persistent_success_count`: 64
  - `gpu_subtree.persistent_partition_claims`: 32684
  - `gpu_subtree.persistent_partition_pops`: 32684
  - `gpu_subtree.persistent_topk_claims`: 1286287
  - `gpu_subtree.persistent_topk_pops`: 1286287
  - `gpu_subtree.persistent_leaf_pushes`: 1286287
  - `gpu_subtree.persistent_queue_overflow_flags`: 0
  - `gpu_subtree.persistent_large_node_splits`: 32684
  - `gpu_subtree.persistent_large_node_max_rows`: 1378181
  - `gpu_subtree.persistent_leaf_size_max`: 640
  - `gpu_subtree.persistent_partition_worker_blocks`: 128
  - `gpu_subtree.persistent_topk_worker_blocks`: 128
  - `leaf_flow.hash_wall_ms`: 2597
  - `leaf_flow.edge_sink_wall_ms`: 35142
  - `leaf_flow.flush_wall_ms`: 35142

## Search

- Search L: 100
- Recall@: 10
- Query count: 10000
- Search wall time: 10000 queries in 3.68s (2714.9 QPS)
- QPS: 2714.9 QPS
- Recall: 0.8433
- Result file: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260616_182818_staged_async/search_results.fbin
- `search_results.fbin` bytes: 400008

## Verdict

- Pass/fail: pass
- Compared baseline: not selected in this run; runbook says to use previous accepted run with same base/query/GT/search/build parameters once established
- Regression or improvement: not classified without a selected baseline
- Generated index cleanup: cleaned; cleanup byte size recorded in `metadata.txt`
- Notes: First build attempt failed before execution because the runbook still included obsolete CLI flag `--gpu-subtree-output compact-topk`; the successful build removed only that obsolete flag. The failed attempt logs are preserved as `stderr_attempt1_obsolete_flag.log`, `stdout_attempt1_obsolete_flag.log`, and `time_attempt1_obsolete_flag.log`.
