# Wiki10M GPU-Subtree Measurement

- Run root: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_032113_persistent_profile_facts
- Build exit code: 1
- profile.json exists: False
- _mem.index bytes: 24
- Elapsed (wall clock) time (h:mm:ss or m:ss): 1:06.64
- Maximum resident set size (kbytes): 9606624

## stderr tail
```text
Error: Index { err: "CUDA persistent subtree overflow: resident_rows=148098 overflow_flags=0x1000 leaf_rows=0 leaf_count=0 partition_pops=1 leaf_pushes=0 topk_pops=0 assignment_passes=0 max_active_points=0 persistent_kernel_event_ms=229.472 progress_epoch=4 watchdog_last_epoch=3 watchdog_stall_polls=31443 watchdog_trips=0 active_work_ticks=148098 active_work_trips=35 snapshot_partition_head=1 snapshot_partition_tail=1 snapshot_leaf_head=0 snapshot_leaf_pushes=0 snapshot_active_partition=1 snapshot_active_topk=0 snapshot_last_node_rows=148098 snapshot_last_node_depth=1 snapshot_last_leaf_rows=0 snapshot_large_lock=1 snapshot_last_node_index=0 snapshot_last_leaf_index=0" }
```

## stdout tail
```text
ng_s=28.995
    t=54.8s phase=build_graph threads=14 rss_gib=8.62 cpu_running_s=28.995
    t=55.0s phase=build_graph threads=14 rss_gib=8.66 cpu_running_s=28.995
    t=55.2s phase=build_graph threads=14 rss_gib=8.69 cpu_running_s=28.995
    t=55.4s phase=build_graph threads=14 rss_gib=8.70 cpu_running_s=28.995
    t=55.6s phase=build_graph threads=14 rss_gib=8.65 cpu_running_s=28.995
    t=55.8s phase=build_graph threads=14 rss_gib=8.62 cpu_running_s=28.995
    t=56.1s phase=build_graph threads=14 rss_gib=8.65 cpu_running_s=28.995
    t=56.3s phase=build_graph threads=14 rss_gib=8.68 cpu_running_s=28.995
    t=56.5s phase=build_graph threads=14 rss_gib=8.69 cpu_running_s=28.995
    t=56.7s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=56.9s phase=build_graph threads=14 rss_gib=8.62 cpu_running_s=28.995
    t=57.1s phase=build_graph threads=14 rss_gib=8.65 cpu_running_s=28.995
    t=57.3s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=57.5s phase=build_graph threads=14 rss_gib=8.69 cpu_running_s=28.995
    t=57.7s phase=build_graph threads=14 rss_gib=8.68 cpu_running_s=28.995
    t=57.9s phase=build_graph threads=14 rss_gib=8.60 cpu_running_s=28.995
    t=58.1s phase=build_graph threads=14 rss_gib=8.61 cpu_running_s=28.995
    t=58.3s phase=build_graph threads=14 rss_gib=8.62 cpu_running_s=28.995
    t=58.5s phase=build_graph threads=14 rss_gib=8.64 cpu_running_s=28.995
    t=58.7s phase=build_graph threads=14 rss_gib=8.66 cpu_running_s=28.995
    t=58.9s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=59.1s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=59.3s phase=build_graph threads=14 rss_gib=8.68 cpu_running_s=28.995
    t=59.5s phase=build_graph threads=14 rss_gib=8.68 cpu_running_s=28.995
    t=59.7s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=59.9s phase=build_graph threads=14 rss_gib=8.65 cpu_running_s=28.995
    t=60.1s phase=build_graph threads=14 rss_gib=8.63 cpu_running_s=28.995
    t=60.3s phase=build_graph threads=14 rss_gib=8.61 cpu_running_s=28.995
    t=60.5s phase=build_graph threads=14 rss_gib=8.63 cpu_running_s=28.995
    t=60.7s phase=build_graph threads=14 rss_gib=8.65 cpu_running_s=28.995
    t=60.9s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=61.1s phase=build_graph threads=14 rss_gib=8.66 cpu_running_s=28.995
    t=61.3s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=61.5s phase=build_graph threads=14 rss_gib=8.68 cpu_running_s=28.995
    t=61.7s phase=build_graph threads=14 rss_gib=8.70 cpu_running_s=28.995
    t=61.9s phase=build_graph threads=14 rss_gib=8.61 cpu_running_s=28.995
    t=62.1s phase=build_graph threads=14 rss_gib=8.63 cpu_running_s=28.995
    t=62.3s phase=build_graph threads=14 rss_gib=8.67 cpu_running_s=28.995
    t=62.5s phase=build_graph threads=14 rss_gib=8.69 cpu_running_s=28.995
    t=62.7s phase=build_graph threads=14 rss_gib=8.64 cpu_running_s=28.995
    t=62.9s phase=build_graph threads=14 rss_gib=8.63 cpu_running_s=28.995
    t=63.1s phase=build_graph threads=14 rss_gib=8.66 cpu_running_s=28.995
    t=63.3s phase=build_graph threads=14 rss_gib=8.68 cpu_running_s=28.995
    t=63.5s phase=build_graph threads=14 rss_gib=8.69 cpu_running_s=28.995
    t=63.7s phase=build_graph threads=14 rss_gib=8.70 cpu_running_s=28.995
    t=63.9s phase=build_graph threads=14 rss_gib=8.60 cpu_running_s=28.995
    t=64.1s phase=build_graph threads=14 rss_gib=8.61 cpu_running_s=28.995
    t=64.3s phase=build_graph threads=14 rss_gib=8.62 cpu_running_s=28.995
    t=64.5s phase=build_graph threads=13 rss_gib=8.56 cpu_running_s=28.995
    t=64.7s phase=build_graph threads=13 rss_gib=8.56 cpu_running_s=28.995
    t=64.9s phase=build_graph threads=13 rss_gib=8.56 cpu_running_s=28.995
    t=65.1s phase=build_graph threads=13 rss_gib=8.78 cpu_running_s=28.995
    t=65.3s phase=build_graph threads=13 rss_gib=9.11 cpu_running_s=28.995
    t=65.5s phase=build_graph threads=13 rss_gib=9.16 cpu_running_s=28.995
    t=65.7s phase=build_graph threads=13 rss_gib=9.04 cpu_running_s=28.995
    t=65.9s phase=build_graph threads=5 rss_gib=8.04 cpu_running_s=29.174
    t=66.1s phase=build_graph threads=4 rss_gib=8.04 cpu_running_s=29.183
===========================
PiPNN: error: Index { err: "CUDA persistent subtree overflow: resident_rows=148098 overflow_flags=0x1000 leaf_rows=0 leaf_count=0 partition_pops=1 leaf_pushes=0 topk_pops=0 assignment_passes=0 max_active_points=0 persistent_kernel_event_ms=229.472 progress_epoch=4 watchdog_last_epoch=3 watchdog_stall_polls=31443 watchdog_trips=0 active_work_ticks=148098 active_work_trips=35 snapshot_partition_head=1 snapshot_partition_tail=1 snapshot_leaf_head=0 snapshot_leaf_pushes=0 snapshot_active_partition=1 snapshot_active_topk=0 snapshot_last_node_rows=148098 snapshot_last_node_depth=1 snapshot_last_leaf_rows=0 snapshot_large_lock=1 snapshot_last_node_index=0 snapshot_last_leaf_index=0" }
```
