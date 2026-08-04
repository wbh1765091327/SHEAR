#!/usr/bin/env bash
set -euo pipefail
variant="$1"
export DATA_ROOT=/data_nvme_kioxia/tyj/data
export STAGING_ROOT=/data_staging/wbh
export WIKI35M_DIR="${DATA_ROOT}/wiki-1024-35M"
export WIKI10M_DATA_DIR="${STAGING_ROOT}/wiki10m_gpu_subtree_measurement/data"
export WIKI10M_BASE="${WIKI10M_DATA_DIR}/wiki_10m_1024.fbin"
export WIKI10M_QUERY_PATH="${WIKI35M_DIR}/query.fbin"
export WIKI10M_GT_PATH="${WIKI10M_DATA_DIR}/wiki_10m_gt100.bin"
export SUITE_ROOT="$(dirname "$0")"
export RUN_ROOT="${SUITE_ROOT}/${variant}"
export INDEX_DIR="${RUN_ROOT}/index"
export SEARCH_DIR="${RUN_ROOT}/search"
export SEARCH_SWEEP_TSV="${RUN_ROOT}/search_sweep.tsv"
mkdir -p "${SEARCH_DIR}"
export RECALL_AT=10
export SEARCH_THREADS=8
export SEARCH_L_VALUES="${SEARCH_L_VALUES:-25 50 100 200 400}"
printf "search_l\trecall_at\tqueries\twall_s\tqps\trecall\tresult_bytes\tlog\tresult\n" > "${SEARCH_SWEEP_TSV}"
for SEARCH_L in ${SEARCH_L_VALUES}; do
  export SEARCH_L
  export SEARCH_LOG="${SEARCH_DIR}/search_l${SEARCH_L}.log"
  export SEARCH_RESULTS="${SEARCH_DIR}/search_l${SEARCH_L}_results.fbin"
  target/release/search_mem_index \
    --index-path-prefix "${INDEX_DIR}" \
    --data-path "${WIKI10M_BASE}" \
    --query-file "${WIKI10M_QUERY_PATH}" \
    --gt-file "${WIKI10M_GT_PATH}" \
    --result-path "${SEARCH_RESULTS}" \
    --recall-at "${RECALL_AT}" \
    --search-list-size "${SEARCH_L}" \
    --num-threads "${SEARCH_THREADS}" \
    > "${SEARCH_LOG}" \
    2>&1
  export SEARCH_RESULT_BYTES
  SEARCH_RESULT_BYTES="$(stat -c "%s" "${SEARCH_RESULTS}")"
  python3 - <<'PY' >> "${SEARCH_SWEEP_TSV}"
import os
import re
log_path = os.environ["SEARCH_LOG"]
with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()
search = re.search(r"Search:\s+(\d+)\s+queries in\s+([0-9.]+)s\s+\(([0-9.]+)\s+QPS\)", text)
recall = re.search(r"Recall@(\d+):\s+([0-9.]+)", text)
if search is None or recall is None:
    raise SystemExit(f"failed to parse search metrics from {log_path}")
print("\t".join([
    os.environ["SEARCH_L"],
    recall.group(1),
    search.group(1),
    search.group(2),
    search.group(3),
    recall.group(2),
    os.environ["SEARCH_RESULT_BYTES"],
    log_path,
    os.environ["SEARCH_RESULTS"],
]))
PY
  echo "${variant} search_l=${SEARCH_L} done"
done
