#!/usr/bin/env bash
set -u
source "$(dirname "$0")/env.sh"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TAGORE_BUILD:${PYTHONPATH:-}"
date -Is > "$RUN_ROOT/logs/tagore_start.txt"
nvidia-smi > "$RUN_ROOT/logs/nvidia_smi_before_tagore.txt"
/usr/bin/time -v -o "$RUN_ROOT/logs/tagore_time.log" \
  "$TAGORE_VENV/bin/python" "$RUN_ROOT/tagore_run/run_tagore_largeindex.py" \
  > "$RUN_ROOT/logs/tagore_stdout.log" \
  2> "$RUN_ROOT/logs/tagore_stderr.log"
rc=$?
echo "$rc" > "$RUN_ROOT/logs/tagore_exit_code.txt"
nvidia-smi > "$RUN_ROOT/logs/nvidia_smi_after_tagore.txt"
date -Is > "$RUN_ROOT/logs/tagore_done.txt"
exit "$rc"
