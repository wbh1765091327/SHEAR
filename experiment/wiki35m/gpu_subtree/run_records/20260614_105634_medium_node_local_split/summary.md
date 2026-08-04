# Wiki35M GPU-Subtree Measurement

- Run ID: 20260614_105634_medium_node_local_split
- Date: 2026-06-14
- Git commit: 90d4de0b5669adb0e5f9b50e670966d0f9c23194
- Code changes under test: medium nodes above leaf_max but within small_node_limit route through local persistent splitter
- Base: /data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
- Query: /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin
- GT: /data_nvme_kioxia/tyj/data/wiki-1024-35M/gt100.bin
- Build command changed from Wiki10M runbook shape: dataset paths only
- Search command: not run because build failed

## Build

- Exit code: 1
- Wall time: 33:14.39
- Max RSS: 42666008 KB
- `_mem.index` bytes: 24
- `profile.json`: missing because build failed before profile write
- OOM memory budget bytes: 214748364800
- GPU workspace GiB: 16
- Phase monitor:
  - build_graph wall: 1992.735s
  - build_graph CPU: 108.592s
  - avg CPU cores: 0.05
  - peak CPU cores: 1.00
  - peak threads: 14
  - peak RSS: 40.67 GiB

## Failure

The failure was a CUDA allocation OOM while preparing a persistent subtree workspace for a later subtree:

```text
Failed to allocate CUDA GPU persistent subtree partition row free stack:
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Persistent plan reported in the error:

```text
max_queue_nodes=5305648
max_queue_rows=21058072
max_leaf_count=658072
max_leaf_rows=5264518
large_scratch_rows=5264518
required_bytes=485819132
budget_bytes=17179869184
resident_device_bytes=10802790936
device_admission_required_bytes=11288610068
device_admission_budget_bytes=11045339672
fallback_reason_flags=0x0
overflow_flags=0x0
```

## Interpretation

- This is not the previous `GPU_PERSISTENT_OVERSIZED_TERMINAL_NODE` failure: `overflow_flags=0x0`.
- This is not an in-kernel queue/row/leaf overflow.
- The run reached a long GPU persistent build phase and then failed on a subsequent CUDA allocation.
- The device admission estimate exceeded the admission budget by about 243.3 MB:
  - required: 11,288,610,068 bytes
  - budget: 11,045,339,672 bytes
- Observed GPU memory during the run reached about 23.6 GB on a 24.6 GB GPU before failure.

## Verdict

- Pass/fail: fail
- Search validation: skipped
- Main performance fact: the medium-node-local-split route avoids the earlier terminal-node overflow class, but it is too slow and too memory-tight at Wiki35M scale on the current 24 GB GPU.
- Next optimization target: keep correctness for leaf-oversized terminal nodes while avoiding routing all `257..small_node_limit` nodes through the expensive local splitter path.
