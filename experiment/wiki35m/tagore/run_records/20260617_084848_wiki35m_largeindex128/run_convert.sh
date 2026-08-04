#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/data_staging/wbh/tagore_wiki35m_measurement/runs/20260617_084848_wiki35m_largeindex128
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tagore_input"
date -Is > "$RUN_ROOT/logs/convert_start.txt"
set +e
/usr/bin/time -v /data_staging/wbh/tagore_wiki10m_measurement/scripts/convert_diskann_partition_to_tagore.py \
  --diskann-prefix "$RUN_ROOT/diskann_part/wiki35m" \
  --tagore-out "$RUN_ROOT/tagore_input" \
  --num-points 35000000 \
  --dim 1024 \
  --clusters 128 \
  > "$RUN_ROOT/logs/convert_stdout.log" \
  2> "$RUN_ROOT/logs/convert_stderr_time.log"
status=$?
echo "$status" > "$RUN_ROOT/logs/convert_exit_code.txt"
date -Is > "$RUN_ROOT/logs/convert_done.txt"
exit "$status"
