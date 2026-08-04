## Material Passport

- Material ID: `laion35m-initial-cta-allocation-figure-source`
- Type: `experiment_figure_source`
- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `validate`
- Origin Date: `2026-08-04`
- Verification Status: `ANALYZED`; source-artifact integrity is `VERIFIED`
- Version Label: `figure-source-v1`

# LAION35M Initial CTA Allocation Figure Provenance

## Accepted source

- Formal run root: `/data_staging/wbh/laion35m_cta_allocation/20260803_primary_v3`
- Promoted source: `/data_staging/wbh/laion35m_cta_allocation/20260803_primary_v3/plot_ready.tsv`
- Raw observations: `/data_staging/wbh/laion35m_cta_allocation/20260803_primary_v3/raw_results.tsv`
- Aggregate results: `/data_staging/wbh/laion35m_cta_allocation/20260803_primary_v3/results.tsv`
- Accepted report: `/data_staging/wbh/laion35m_cta_allocation/20260803_primary_v3/summary.md`
- Source TSV SHA-256: `13c53eadb085fcfa86599dac307abaa89e9a00f9d614bb805640ae312edac0e4`
- Completion: 24/24 measured builds accepted and 8/8 block-2 L100 quality searches accepted.

The failed protocol attempts `20260803_primary_v1` and `20260803_primary_v2` are excluded. Their
diagnostic evidence remains in the experiment output root and was not substituted into this figure.

## Input and binary identity

| Material | SHA-256 |
|---|---|
| LAION35M base | `e85c99bb60498432afd28c5a29b367bdb3ac63b96922f91270bab3b94d03fb0d` |
| Query | `671720739208ba56731c679d2acf68458b50ed391290f1f6ebce6d7ec122444c` |
| LAION35M ground truth | `f69cab5989db3a12c054c8a3c239acb08b74a4866344c25f67aad51b244ea98c` |
| Frozen build binary | `6dc7f3137eab3cae02c590c1580c5090a26c9cd595b088060652eecac6e49a98` |
| Frozen search binary | `d3735e7caed3265ee874f2e7bc5bcfc293abf8757c3aa5c6fb2c52aacdfad1e0` |

## Matrix and cache protocol

- Allocation order: `1:127`, `4:124`, `8:120`, `16:112`, `24:104`, `32:96`, `64:64`, `104:24`.
- Fixed blocks: increasing; decreasing; `24,4,64,8,104,16,32,1` partition CTAs.
- Each allocation has n=3 accepted measurements. Curves show the median and min--max whiskers.
- The hot-cache protocol used one preload, one unmeasured `1:127` warmup, and no cache drops between
  formal transactions.
- All cases used the same frozen binary and 128-CTA grid. Only the initial Partition:Top-k split changed;
  elastic cross-role helping remained enabled.

## Measurement layers

- `E2E build`: external wall time from GNU `time`.
- `D1+ producer`: `serial_timing.diagnostics.d1plus_producer_ms`; overlapping sidecar-gather activity is
  not added again.
- `Persistent kernel`: `gpu_subtree.persistent_kernel_event_ms`, the persistent kernel event measured
  with CUDA events.
- Raw GPU busy/wait clocks and poll totals are not interpreted as wall time and are not plotted.

## Integrity, quality, and interpretation

- All 24 source rows were independently compared with their `time.log`, `profile.json`, GPU-monitor
  summary, parameter passport, and, where applicable, search metrics. All eight medians, minima, maxima,
  normalized scheduler ratios, quality deltas, and decision fields were recomputed independently.
- Re-running the summarizer twice produced byte-stable raw, aggregate, plot-ready, and Markdown outputs.
- Recall@10 ranged from `0.96449` to `0.96529`; every allocation stayed within the frozen `0.002` guard
  relative to `1:127`. The recommendation rule selects `32:96`.
- Accepted transaction-local `index`, `oom`, and `sq_wmma` payloads were removed after validation;
  logs, profiles, search results, hashes, and recorded sizes remain.
- `workspace_peak_bytes=0` is an unavailable merged-telemetry value in this async-D2H path, not evidence
  of zero workspace use. The acceptance gate verifies that the reported value does not exceed the 16-GiB
  budget; the figure does not use this field.
- Statistical interpretation is descriptive and environment-sensitive. With n=3, no statistical-
  significance or broad hardware-generalization claim is made.
- Statistical fallacy scan coverage: 11/11. No Simpson reversal appears across the three blocks; no
  missing/survivor observations exist; the matrix and decision rule were frozen before the accepted run;
  causal interpretation is limited to the controlled initial-allocation change on this host and binary.

## Reproducibility status

- Method: artifact-level independent recomputation; no separate full 24-build rerun.
- Verdict: `CANNOT_VERIFY` for independent environment-level reproducibility; `VERIFIED` for the accepted
  run's artifact integrity and deterministic aggregation.
