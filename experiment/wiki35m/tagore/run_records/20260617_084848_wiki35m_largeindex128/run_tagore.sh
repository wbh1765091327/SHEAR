#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/data_staging/wbh/tagore_wiki35m_measurement/runs/20260617_084848_wiki35m_largeindex128
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tagore_run/local_index" "$RUN_ROOT/tagore_run/index"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvidia-smi > "$RUN_ROOT/logs/nvidia_smi_before_tagore.txt"
date -Is > "$RUN_ROOT/logs/tagore_start.txt"
set +e
/usr/bin/time -v /data_staging/wbh/tagore_wiki10m_measurement/venv/bin/python "$RUN_ROOT/tagore_run/run_tagore_largeindex.py" \
  > "$RUN_ROOT/logs/tagore_stdout.log" \
  2> "$RUN_ROOT/logs/tagore_stderr_time.log"
status=$?
echo "$status" > "$RUN_ROOT/logs/tagore_exit_code.txt"
nvidia-smi > "$RUN_ROOT/logs/nvidia_smi_after_tagore.txt" || true
date -Is > "$RUN_ROOT/logs/tagore_done.txt"
exit "$status"
