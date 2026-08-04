# Wiki35M GPU-Subtree Measurement

- Run root: /data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_032317_persistent_profile_facts
- Command shape: Wiki10M GPU-subtree runbook command shape, with canonical Wiki35M base/query/GT
- Build exit code: 1
- profile.json exists: False
- _mem.index bytes: 24
- Elapsed (wall clock) time (h:mm:ss or m:ss): 3:57.65
- Maximum resident set size (kbytes): 32659132

## stderr tail
```text
Error: Index { err: "CUDA persistent subtree overflow: resident_rows=546143 overflow_flags=0x1000 leaf_rows=0 leaf_count=0 partition_pops=1 leaf_pushes=0 topk_pops=0 assignment_passes=0 max_active_points=0 persistent_kernel_event_ms=234.919 progress_epoch=4 watchdog_last_epoch=3 watchdog_stall_polls=49944 watchdog_trips=0 active_work_ticks=546143 active_work_trips=132 snapshot_partition_head=1 snapshot_partition_tail=1 snapshot_leaf_head=0 snapshot_leaf_pushes=0 snapshot_active_partition=1 snapshot_active_topk=0 snapshot_last_node_rows=546143 snapshot_last_node_depth=1 snapshot_last_leaf_rows=0 snapshot_large_lock=1 snapshot_last_node_index=0 snapshot_last_leaf_index=0" }
```

## stdout tail
```text
_s=103.870
    t=225.4s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=225.6s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=225.8s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=226.0s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=226.2s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=226.4s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=226.6s phase=build_graph threads=14 rss_gib=29.26 cpu_running_s=103.870
    t=226.8s phase=build_graph threads=14 rss_gib=29.26 cpu_running_s=103.870
    t=227.0s phase=build_graph threads=14 rss_gib=29.25 cpu_running_s=103.870
    t=227.2s phase=build_graph threads=14 rss_gib=29.24 cpu_running_s=103.870
    t=227.4s phase=build_graph threads=14 rss_gib=29.23 cpu_running_s=103.870
    t=227.6s phase=build_graph threads=14 rss_gib=29.25 cpu_running_s=103.870
    t=227.8s phase=build_graph threads=14 rss_gib=29.26 cpu_running_s=103.870
    t=228.0s phase=build_graph threads=14 rss_gib=29.27 cpu_running_s=103.870
    t=228.2s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=228.4s phase=build_graph threads=14 rss_gib=29.28 cpu_running_s=103.870
    t=228.6s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=228.8s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=229.0s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=229.2s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=229.4s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=229.6s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=229.8s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=230.0s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=230.2s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=230.4s phase=build_graph threads=13 rss_gib=29.16 cpu_running_s=103.870
    t=230.6s phase=build_graph threads=13 rss_gib=29.22 cpu_running_s=103.870
    t=230.8s phase=build_graph threads=13 rss_gib=29.55 cpu_running_s=103.870
    t=231.0s phase=build_graph threads=13 rss_gib=29.72 cpu_running_s=103.870
    t=231.2s phase=build_graph threads=13 rss_gib=29.90 cpu_running_s=103.870
    t=231.4s phase=build_graph threads=13 rss_gib=30.11 cpu_running_s=103.870
    t=231.6s phase=build_graph threads=13 rss_gib=30.39 cpu_running_s=103.870
    t=231.8s phase=build_graph threads=13 rss_gib=30.69 cpu_running_s=103.870
    t=232.0s phase=build_graph threads=13 rss_gib=31.01 cpu_running_s=103.870
    t=232.2s phase=build_graph threads=13 rss_gib=31.11 cpu_running_s=103.870
    t=232.4s phase=build_graph threads=13 rss_gib=31.11 cpu_running_s=103.870
    t=232.6s phase=build_graph threads=13 rss_gib=31.11 cpu_running_s=103.870
    t=232.8s phase=build_graph threads=13 rss_gib=31.15 cpu_running_s=103.870
    t=233.0s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=103.897
    t=233.2s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=104.098
    t=233.4s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=104.298
    t=233.6s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=104.498
    t=233.8s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=104.698
    t=234.0s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=104.898
    t=234.2s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=105.099
    t=234.4s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=105.299
    t=234.6s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=105.499
    t=234.8s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=105.699
    t=235.0s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=105.899
    t=235.2s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=106.100
    t=235.4s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=106.300
    t=235.6s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=106.500
    t=235.8s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=106.700
    t=236.0s phase=build_graph threads=5 rss_gib=28.94 cpu_running_s=106.900
    t=236.2s phase=build_graph threads=4 rss_gib=27.11 cpu_running_s=107.095
===========================
PiPNN: error: Index { err: "CUDA persistent subtree overflow: resident_rows=546143 overflow_flags=0x1000 leaf_rows=0 leaf_count=0 partition_pops=1 leaf_pushes=0 topk_pops=0 assignment_passes=0 max_active_points=0 persistent_kernel_event_ms=234.919 progress_epoch=4 watchdog_last_epoch=3 watchdog_stall_polls=49944 watchdog_trips=0 active_work_ticks=546143 active_work_trips=132 snapshot_partition_head=1 snapshot_partition_tail=1 snapshot_leaf_head=0 snapshot_leaf_pushes=0 snapshot_active_partition=1 snapshot_active_topk=0 snapshot_last_node_rows=546143 snapshot_last_node_depth=1 snapshot_last_leaf_rows=0 snapshot_large_lock=1 snapshot_last_node_index=0 snapshot_last_leaf_index=0" }
```
