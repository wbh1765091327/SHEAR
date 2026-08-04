#!/usr/bin/env bash
set -euo pipefail
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
