# GPU Subtree Persistent Profile Facts - 2026-06-14

## Wiki10M

- Run: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_081629_persistent_profile_facts
- Build: exit 0, Elapsed (wall clock) time (h:mm:ss or m:ss): 9:23.41
- Max RSS: Maximum resident set size (kbytes): 15092652
- Search: Search: 10000 queries in 4.13s (2421.0 QPS)
- Recall: Recall@10: 0.8201
- Profile extract: /data_staging/wbh/wiki10m_gpu_subtree_measurement/runs/20260614_081629_persistent_profile_facts/profile_extract.txt

## Wiki35M

- Run: /data_staging/wbh/wiki35m_gpu_subtree_measurement/runs/20260614_082839_persistent_profile_facts
- Build: exit 1, Elapsed (wall clock) time (h:mm:ss or m:ss): 20:04.91
- Max RSS: Maximum resident set size (kbytes): 42621012
- Search: not run because build failed
- Failure: PiPNN: error: Index { err: "Failed to allocate CUDA GPU persistent subtree partition rows: IndexError: Failed to allocate CUDA GPU persistent subtree partition rows: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"); persistent_plan={max_queue_nodes=5305648 max_queue_rows=57909698 max_leaf_count=658072 max_leaf_rows=5264518 large_scratch_rows=5264518 required_bytes=569557860 budget_bytes=17179869184 resident_device_bytes=10802790936 device_admission_required_bytes=11372348796 device_admission_budget_bytes=11045339672 fallback_reason_flags=0x0 overflow_flags=0x0}" }

## Immediate Interpretation

- Wiki10M confirms the persistent producer-consumer path is active and internally consistent: partition/top-k claims match pops, leaf pushes match top-k claims, queue overflow flags are zero.
- Wiki35M does not pass at 16 GiB workspace on this 24 GB RTX 4090. The failure is device allocation pressure for a resident subtree with a plan requiring 569,557,860 scratch bytes on top of about 10.8 GB resident device bytes.
- Runtime nvidia-smi polling observed Wiki35M device memory reaching roughly 23.86 GiB before the final OOM, so the current design is operating with near-zero VRAM headroom at full scale.
