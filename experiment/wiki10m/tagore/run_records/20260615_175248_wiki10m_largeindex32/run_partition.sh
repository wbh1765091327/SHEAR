#!/usr/bin/env bash
set -u
source "$(dirname "$0")/env.sh"
date -Is > "$RUN_ROOT/logs/partition_start.txt"
/usr/bin/time -v -o "$RUN_ROOT/logs/partition_time.log" \
  /home/wbh/cppwork/DiskANN-OOM/build/apps/utils/partition_data \
  float "$WIKI10M_BASE" "$DISKANN_PREFIX" 0.01 32 2 \
  > "$RUN_ROOT/logs/partition_stdout.log" \
  2> "$RUN_ROOT/logs/partition_stderr.log"
rc=$?
echo "$rc" > "$RUN_ROOT/logs/partition_exit_code.txt"
date -Is > "$RUN_ROOT/logs/partition_done.txt"
exit "$rc"
