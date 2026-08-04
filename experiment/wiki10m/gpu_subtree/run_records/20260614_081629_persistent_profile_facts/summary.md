# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_081629_persistent_profile_facts
- Date: 2026-06-14T08:27:46+08:00
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: current dirty workspace, persistent producer-consumer path
- Base: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin
- Build command changed from runbook: no
- Search command changed from runbook: no

## Build

- Exit code: 0
- Wall time: Elapsed (wall clock) time (h:mm:ss or m:ss): 9:23.41
- Max RSS: Maximum resident set size (kbytes): 15092652
- _mem.index bytes: 921276128
- Profile: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_081629_persistent_profile_facts/profile.json

## Profile Extract

```text
profile= /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_081629_persistent_profile_facts/profile.json
memory_budget_bytes= 214748364800
partitioner= {'kind': 'gpu-subtree', 'kmeans': {'points': 0, 'emitted_points': 0, 'leaves': 0, 'split_attempts': 0, 'assignment_passes': 0, 'empty_clusters': 0, 'fallback_splits': 0, 'max_leaf_size': 0, 'wall_ms': 0}}
persistent_attempt_count=64
persistent_success_count=64
persistent_queue_overflow_flags=0
persistent_partition_claims=107253
persistent_partition_pops=107253
persistent_topk_claims=1665502
persistent_topk_pops=1665502
persistent_leaf_pushes=1665502
persistent_large_node_splits=107253
persistent_large_node_max_rows=1378181
persistent_kernel_event_ms=319702
persistent_total_ms=319705
persistent_topk_rows=80000000
persistent_leaf_count=1665502
persistent_leaf_size_min=1
persistent_leaf_size_max=256
persistent_leaf_row_amplification_x1000=2000
workspace_peak_bytes=298206716
workspace_budget_bytes=17179869184
persistent_partition_worker_blocks=128
persistent_topk_worker_blocks=128
gpu_subtree_keys=
assignment_active_ids_h2d_ms
assignment_d2h_ms
assignment_d2h_submit_ms
assignment_kernel_event_ms
assignment_kernel_ms
assignment_leader_ids_h2d_ms
assignment_passes
assignment_sync_wait_ms
child_count
child_points
compact_topk_batches
compact_topk_rows
compact_unsupported_count
compact_unsupported_points
cpu_cluster_merge_ms
cpu_compact_flush_ms
cpu_fallbacks
cpu_leaf_global_to_local_ms
cpu_result_scatter_ms
d2h_bytes
enabled
h2d_bytes
host_control_ms
host_load_vectors_ms
host_normalize_ms
host_norms_ms
leaf_count
leaf_topk_backend
native_assignment_passes
native_attempt_count
native_attempt_points
native_leaf_count
native_max_points
native_success_count
native_success_points
native_unsupported_count
native_unsupported_points
native_wall_ms
overlap_probe_count
overlap_probe_effective_overlap_ms
overlap_probe_queue_kernel_ms
overlap_probe_topk_kernel_ms
overlap_probe_wall_ms
persistent_active_wait_cycles
persistent_attempt_count
persistent_done_cas_wins
persistent_effective_deeper_fanout
persistent_effective_root_fanout
persistent_fallback_count
persistent_fanout_assignment_slots
persistent_kernel_event_ms
persistent_large_node_claims
persistent_large_node_empty_polls
persistent_large_node_max_rows
persistent_large_node_rows
persistent_large_node_serialized_wait_cycles
persistent_large_node_splits
persistent_leaf_count
persistent_leaf_max
persistent_leaf_min
persistent_leaf_pushes
persistent_leaf_row_amplification_x1000
persistent_leaf_size_count
persistent_leaf_size_max
persistent_leaf_size_min
persistent_leaf_size_sum
persistent_leaf_target
persistent_partition_claims
persistent_partition_empty_polls
persistent_partition_pops
persistent_partition_worker_blocks
persistent_queue_overflow_flags
persistent_replicated_rows
persistent_success_count
persistent_topk_claims
persistent_topk_empty_polls
persistent_topk_pops
persistent_topk_rows
persistent_topk_worker_blocks
persistent_total_ms
persistent_unique_resident_rows
queue_admission_budget_bytes
queue_admission_required_bytes
queue_emitted_leaf_count
queue_emitted_leaf_rows
queue_fallback_reason_flags
queue_global_row_capacity
queue_kernel_ms
queue_large_node_count
queue_large_node_passes
queue_large_node_rows
queue_max_node_rows
queue_overflow_count
queue_overflow_flags
queue_passes
queue_peak_leaf_rows
queue_peak_nodes
queue_peak_rows
queue_small_node_count
queue_small_node_limit
queue_small_node_passes
queue_small_node_rows
queue_status_d2h_ms
resident_h2d_bytes
resident_topk_d2h_submit_ms
resident_topk_kernel_event_ms
resident_topk_metadata_h2d_bytes
resident_topk_metadata_h2d_ms
resident_topk_sync_wait_ms
root_assignment_backend
subtree_assignment_backend
subtree_global_ids_h2d_ms
subtree_norms_h2d_ms
subtree_vector_h2d_ms
topk_d2h_ms
topk_kernel_ms
total_ms
vector_upload_ms
workspace_budget_bytes
workspace_peak_bytes
```

## Search

```text
Base data: 10000000 points, dim=1024
Graph: 10000000 nodes, start=6843085, max_degree=64
Queries: 10000
Search: 10000 queries in 4.13s (2421.0 QPS)
Recall@10: 0.8201
search_results_bytes=400008
```

## Verdict

- Pass/fail: pass for Wiki10M measurement; recall is lower than old fallback-style baseline and should be tracked as a quality regression candidate.
