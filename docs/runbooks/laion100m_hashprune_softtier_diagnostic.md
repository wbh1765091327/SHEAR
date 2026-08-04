# HashPrune SoftTier diagnostic runbook

This transaction compares independently built Plain, same-trace `shadow_plain`, and SoftTier graphs with the current RaBitQ1 adaptive-exact search. Run Wiki10M first; LAION100M is allowed only after the Wiki10M transaction passes every graph, sidecar, result-hash, and metric-conservation gate.

```bash
bash script/run_hashprune_softtier_diagnostic.sh wiki10m /data_staging/wbh/NEW_RUN_ROOT
bash script/run_hashprune_softtier_diagnostic.sh laion100m /data_staging/wbh/NEW_RUN_ROOT
```

The accepted dataset selector is `wiki10m|laion100m`. The runner creates a new root, holds the SHEAR experiment lock, checks the GPU and cache-control preconditions, validates input headers and IDs, and records binary/input hashes before work begins. LAION uses `/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin` and `/data_staging/wbh/unified_graph_rabitq1/sidecars/laion100m.rabitq1`; the frozen sidecar SHA-256 is `46a7758e2008313f1d31893966db2afe10c46d12103780b4ee06b796c975fae0`.

Plain uses backfill off. Soft uses `rhalf-soft-tier-q3` with CLI alpha 1.0; its named-policy effective priority alpha remains 1.1 and is audited from the profile. Diagnostics are enabled only for Soft with `--hashprune-backfill-diagnostics-prefix`. The shadow builder retains the exact primary prefix recorded by the diagnostic sidecar.

For each graph and L=25,50,100,200,400, the frozen `rabitq1-adaptive-exact` search performs one warm-up and two measured repetitions. Every standard run writes `--query-metrics-path`; Soft additionally receives `--hashprune-primary-count-sidecar` in one untimed diagnostic pass. Both measured result files must match, and the diagnostic result SHA-256 must match standard Soft. Diagnostic elapsed time never contributes to canonical QPS.

There is no automatic retry. On any failure, preserve the failing transaction and all generated evidence. Cleanup is allowed only after manifests and summaries have been fsynced and status is accepted; it removes only the explicitly listed large graph/work directories below the new run root.
