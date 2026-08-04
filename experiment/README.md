# Experiment Records

This directory contains a curated, text-only archive of GPU experiments run between June 11 and June 17, 2026. It
preserves commands, environment snippets, logs, timing records, exit codes, profile JSON/JSONL, summaries, tables, and
search-validation output needed to inspect the reported development experiments.

## Layout

| Directory | Contents |
| --- | --- |
| `wiki10m/gpu_subtree/` | Wiki10M GPU-subtree construction runs, diagnostics, and profiles |
| `wiki10m/tagore/` | Wiki10M Tagore comparison and search-validation records |
| `wiki35m/gpu_subtree/` | Wiki35M GPU-subtree construction runs and profiles |
| `wiki35m/tagore/` | Wiki35M Tagore comparison records |
| `validation/subtree_overlap/` | Focused overlap validation on a SIFT200K probe |
| `validation/resident_topk/` | Resident top-k workspace validation |
| `microbenchmarks/wmma_shape_sweep/` | WMMA shape sweep records |
| `microbenchmarks/wmma_multileaf_replay/` | Representative multi-leaf WMMA replay records |
| `summary/` | Cross-run profile summaries |

`MANIFEST.tsv` lists every archived file and its byte size.

## Scope

The archive intentionally excludes generated datasets, graph indexes, temporary OOM files, build directories, virtual
environments, Nsight binary reports, and other large binary artifacts. In particular, `_mem.index`, `.fbin`, `.bin`,
`.so`, `.a`, and `.ncu-rep` files are not included.

Commands and logs retain paths and executable names from the original measurement environment as provenance. Adapt
those paths to the local machine before rerunning an experiment. The maintained procedures are documented in
[`docs/runbooks/`](../docs/runbooks/).
