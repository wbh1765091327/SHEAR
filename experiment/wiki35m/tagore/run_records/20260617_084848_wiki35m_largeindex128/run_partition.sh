#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/data_staging/wbh/tagore_wiki35m_measurement/runs/20260617_084848_wiki35m_largeindex128
BASE=/data_nvme_kioxia/tyj/data/wiki-1024-35M/en_base.fbin
PREFIX=$RUN_ROOT/diskann_part/wiki35m
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/diskann_part"
date -Is > "$RUN_ROOT/logs/partition_start.txt"
set +e
/usr/bin/time -v /home/wbh/cppwork/DiskANN-OOM/build/apps/utils/partition_data float "$BASE" "$PREFIX" 0.01 128 2 \
  > "$RUN_ROOT/logs/partition_stdout.log" \
  2> "$RUN_ROOT/logs/partition_stderr_time.log"
status=$?
echo "$status" > "$RUN_ROOT/logs/partition_exit_code.txt"
date -Is > "$RUN_ROOT/logs/partition_done.txt"
exit "$status"
