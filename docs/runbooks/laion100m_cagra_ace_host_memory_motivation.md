# LAION100M CAGRA ACE Host-Memory Motivation Runbook

本文档生成论文 motivation 所需的两组 CAGRA ACE 构图记录：

1. **Baseline**：不人为限制主机内存和 GPU 显存，测量 E2E、ACE 内部阶段占比、RSS、GPU 和磁盘行为。
2. **Cgroup stress**：根据 baseline peak RSS 预先确定一个主机内存上限，只执行一次受限构图，检验内存压力是否会放大 I/O、增加 partition、造成 GPU 供给不足或触发 OOM。

这不是 memory sweep。两组实验使用同一个数据文件和相同构图参数，不执行 search sweep。

## Fixed Contract

```text
dataset:                  LAION100M, 100,000,000 x 512 float32
graph degree:             64
intermediate degree:      128
ACE ef_construction:      120
ACE partitions:           auto, npartitions=0
ACE storage:              disk mode
GPU memory limit:         0, use all available GPU memory
GPU device:               0
```

`max_gpu_memory_gb=0` 表示不人为限制 ACE 可使用的显存，不保证每个时刻都占满 HBM。实际峰值由 `nvidia-smi` 采样确定。

## Paths And Preconditions

当前 cuVS native ACE 启动器位于独立 worktree。该路径可由环境变量覆盖：

```bash
export PROJECT_ROOT=/path/to/SHEAR
export ACE_ROOT="${ACE_ROOT:-/path/to/cagra-driver-checkout}"
export DATA_PATH=/data_nvme_kioxia/tyj/data/laion-512-100M/base.fbin
export STAGING_ROOT=/data_staging/wbh/laion100m_cagra_ace_motivation
export CAGRA_COMMAND="${ACE_ROOT}/tools/cagra_ace_build/cagra_ace_build.sh"
export ACE_BIN="${ACE_ROOT}/target/release/build_cagra_ace_index"
```

验证入口、版本和数据：

```bash
test -f "${ACE_ROOT}/apps/build_cagra_ace_index/src/main.rs"
test -x "${CAGRA_COMMAND}"
git -C "${ACE_ROOT}" merge-base --is-ancestor 2a146a2 HEAD
git -C "${ACE_ROOT}" rev-parse HEAD
git -C "${ACE_ROOT}" status --short

test -s "${DATA_PATH}"
od -An -tu4 -N 8 "${DATA_PATH}"
test "$(stat -c %s "${DATA_PATH}")" -eq 204800000008
```

期望 header 为 `100000000 512`。Base payload 为 204.8 GB，即约 190.73 GiB。原始数据小于当前机器 DRAM，不代表 ACE 的重排数据、augmented data、图、映射和 workspace 可以同时驻留内存。

检查机器和存储。`/data/staging` 至少保留 500 GB：

```bash
free -h
df -h /data_staging /data_nvme_kioxia
test "$(df --output=avail -B1 /data_staging | tail -n 1)" -ge 500000000000
nvidia-smi
test "$(stat -fc %T /sys/fs/cgroup)" = cgroup2fs
command -v systemd-run
command -v pidstat
command -v iostat
test -x "${PROJECT_ROOT}/script/monitor_gpu_build.sh"
```

正式运行期间不得有其他 GPU 构图任务或大规模磁盘任务。

## Build And Self-Test

```bash
cd "${ACE_ROOT}"
cargo build --release -p build_cagra_ace_index
"${CAGRA_COMMAND}" --self-test --device 0
"${ACE_BIN}" --help | tee "${STAGING_ROOT}/ace_help.log"
```

Help 中必须存在 `--npartitions`、`--use-disk`、`--max-host-memory-gb` 和 `--max-gpu-memory-gb`。正式调用必须进入 cuVS `cagra.IndexParams(build_algo="ace", ...)`，不得替换为 Tagore ACE-like wrapper。

## Shared Launch Function

以下函数启动一次 ACE build。参数依次为 run root、ACE 主机内存预算 GiB、标签。预算 `0` 表示 baseline；正数同时用于 `AceParams.max_host_memory_gb` 和 cgroup `MemoryMax`。

它记录：

- `/usr/bin/time -v`：E2E、peak RSS、major page faults 和文件系统 I/O。
- `pidstat`：ACE driver/Python child 的 RSS、CPU 和读写速率。
- `vmstat`、`iostat`：主机内存、I/O wait 和块设备吞吐。
- `monitor_gpu_build.sh`：GPU utilization、HBM、power 和进程采样。
- `systemd-run`：整棵进程树的 cgroup 结果和 memory peak。

两次正式运行都清空 page cache。该操作只能在独占实验机上执行。

```bash
run_ace() {
  local run_root="$1"
  local host_cap_gib="$2"
  local tag="$3"
  local index_dir="${run_root}/index"
  local host_monitor_dir="${run_root}/host_monitor"
  local gpu_monitor_dir="${run_root}/gpu_monitor"
  local unit="shear-cagra-ace-$(basename "${run_root}" | tr '_' '-')"
  local time_log="${run_root}/time.log"
  local stdout_log="${run_root}/stdout.log"
  local stderr_log="${run_root}/stderr.log"
  local command_file="${run_root}/build_command.sh"
  local systemd_props=()

  mkdir -p "${index_dir}" "${host_monitor_dir}" "${gpu_monitor_dir}"
  if [ "${host_cap_gib}" -gt 0 ]; then
    systemd_props+=(
      --property="MemoryMax=${host_cap_gib}G"
      --property="MemorySwapMax=0"
      --property="OOMPolicy=stop"
    )
  fi

  {
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "dataset=${DATA_PATH}"
    echo "dataset_bytes=$(stat -c %s "${DATA_PATH}")"
    echo "project_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
    echo "ace_commit=$(git -C "${ACE_ROOT}" rev-parse HEAD)"
    echo "ace_host_cap_gib=${host_cap_gib}"
    echo "ace_gpu_cap_gib=0"
    echo "ace_npartitions=0"
    free -b
    nvidia-smi
    df -B1 /data_staging /data_nvme_kioxia
  } > "${run_root}/metadata.txt" 2>&1

  cat > "${command_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/time -v -o "${time_log}" \
  "${ACE_BIN}" \
    --data-path "${DATA_PATH}" \
    --index-path-prefix "${index_dir}" \
    --run-root "${run_root}" \
    --cagra-command "${CAGRA_COMMAND}" \
    --device 0 \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --npartitions 0 \
    --ef-construction 120 \
    --max-host-memory-gb "${host_cap_gib}" \
    --max-gpu-memory-gb 0 \
    --use-disk \
    > "${stdout_log}" \
    2> "${stderr_log}"
EOF
  chmod +x "${command_file}"

  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  vmstat -w 1 > "${host_monitor_dir}/vmstat.log" & local vmstat_pid=$!
  iostat -dxz -y 1 > "${host_monitor_dir}/iostat.log" & local iostat_pid=$!
  pidstat -r -d -u -h -l -G 'build_cagra|cagra_ace_build|python' 1 \
    > "${host_monitor_dir}/pidstat.log" & local pidstat_pid=$!

  set +e
  bash "${PROJECT_ROOT}/script/monitor_gpu_build.sh" \
    --gpu 0 \
    --interval 1 \
    --out-dir "${gpu_monitor_dir}" \
    --tag "${tag}" \
    -- \
    sudo systemd-run \
      --unit="${unit}" \
      --wait \
      --pipe \
      --uid="$(id -un)" \
      --gid="$(id -gn)" \
      --working-directory="${ACE_ROOT}" \
      "${systemd_props[@]}" \
      "${command_file}"
  local build_exit=$?
  set -e

  kill "${vmstat_pid}" "${iostat_pid}" "${pidstat_pid}" 2>/dev/null || true
  wait "${vmstat_pid}" "${iostat_pid}" "${pidstat_pid}" 2>/dev/null || true
  printf '%s\n' "${build_exit}" > "${run_root}/build_exit_code.txt"
  sudo systemctl show "${unit}.service" \
    -p Result -p ExecMainCode -p ExecMainStatus \
    -p MemoryCurrent -p MemoryPeak -p MemoryMax -p MemorySwapCurrent \
    > "${run_root}/cgroup_summary.txt" || true
  sudo journalctl -u "${unit}.service" --no-pager \
    > "${run_root}/cgroup_journal.log" || true
  return "${build_exit}"
}
```

## Shared Metrics Parser

`optimize` 是 ACE 日志中最接近核心 GPU CAGRA 构建的可见代理，不等于精确的 kernel active time。论文中的核心构建占比固定为：

```text
sum(per-partition optimize) / external E2E wall
```

定义统一 parser：

```bash
extract_ace_metrics() {
  python3 - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
stderr_path = root / "stderr.log"
time_path = root / "time.log"
profile_path = root / "profile.json"
stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
time_text = time_path.read_text(errors="replace") if time_path.exists() else ""
profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}

def one(pattern):
    match = re.search(pattern, stderr)
    return int(match.group(1)) if match else None

def seconds(value):
    parts = [float(part) for part in value.split(":")]
    return sum(part * 60 ** i for i, part in enumerate(reversed(parts)))

elapsed = re.search(r"Elapsed \(wall clock\) time .*?: ([0-9:.]+)", time_text)
rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_text)
partitions = [
    tuple(map(int, match.groups()))
    for match in re.finditer(
        r"Partition\s+(\d+) \([^)]*\) completed in\s+(\d+) ms: "
        r"read\s+(\d+) ms .*? optimize\s+(\d+) ms, "
        r"adjust\s+(\d+) ms, write\s+(\d+) ms",
        stderr,
    )
]
e2e_s = seconds(elapsed.group(1)) if elapsed else None
optimize_ms = sum(row[3] for row in partitions)
metrics = {
    "build_exit_code": int((root / "build_exit_code.txt").read_text().strip()),
    "e2e_wall_s": e2e_s,
    "max_rss_kb": int(rss.group(1)) if rss else None,
    "partition_count": len(partitions),
    "partition_label_ms": one(r"Partition labeling completed in (\d+) ms"),
    "dataset_reorder_ms": one(r"reordering completed in (\d+) ms"),
    "partition_total_ms": sum(row[1] for row in partitions),
    "partition_read_ms": sum(row[2] for row in partitions),
    "partition_optimize_ms": optimize_ms,
    "partition_adjust_ms": sum(row[4] for row in partitions),
    "partition_write_ms": sum(row[5] for row in partitions),
    "ace_internal_total_ms": one(r"Partitioned CAGRA build completed in (\d+) ms total"),
    "final_pack_ms": profile.get("partition_pack_ms"),
    "mem_index_bytes": profile.get("mem_index_bytes"),
    "optimize_e2e_fraction": optimize_ms / (e2e_s * 1000) if e2e_s else None,
}
(root / "motivation_metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
}
```

## Experiment 1: Baseline

```bash
export BASELINE_ID="$(date +%Y%m%d_%H%M%S)"
export BASELINE_ROOT="${STAGING_ROOT}/runs/${BASELINE_ID}_baseline"

set +e
run_ace "${BASELINE_ROOT}" 0 baseline
BASELINE_EXIT=$?
set -e
extract_ace_metrics "${BASELINE_ROOT}"
```

Baseline 必须成功，否则不进入压力实验：

```bash
test "${BASELINE_EXIT}" -eq 0
test -s "${BASELINE_ROOT}/profile.json"
test -s "${BASELINE_ROOT}/ace_profile.json"
test -s "${BASELINE_ROOT}/index/_mem.index"
grep -q 'Partitioned CAGRA build completed' "${BASELINE_ROOT}/stderr.log"
grep -q 'Maximum resident set size' "${BASELINE_ROOT}/time.log"
```

## Select One Cgroup Limit

压力组 cap 固定为 baseline peak RSS 的 50%，向下取整到 8 GiB，最低 32 GiB：

```bash
export BASELINE_RSS_KB="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_rss_kb"])' "${BASELINE_ROOT}/motivation_metrics.json")"
export CAP_GIB="$(awk -v rss_kb="${BASELINE_RSS_KB}" 'BEGIN {
  cap = int(((rss_kb / 1024 / 1024) * 0.5) / 8) * 8;
  if (cap < 32) cap = 32;
  print cap;
}')"
printf 'baseline_rss_kb=%s\ncgroup_cap_gib=%s\n' "${BASELINE_RSS_KB}" "${CAP_GIB}"
```

这个规则只定义一个可复现的 extreme case。不得根据压力组结果继续尝试多个 cap 并挑选最慢结果。

## Experiment 2: Single Cgroup Stress

```bash
export STRESS_ID="$(date +%Y%m%d_%H%M%S)"
export STRESS_ROOT="${STAGING_ROOT}/runs/${STRESS_ID}_cgroup_${CAP_GIB}g"
printf 'baseline_root=%s\nbaseline_rss_kb=%s\ncgroup_cap_gib=%s\n' \
  "${BASELINE_ROOT}" "${BASELINE_RSS_KB}" "${CAP_GIB}" \
  > "${STAGING_ROOT}/stress_selection.txt"

set +e
run_ace "${STRESS_ROOT}" "${CAP_GIB}" cgroup_stress
STRESS_EXIT=$?
set -e
extract_ace_metrics "${STRESS_ROOT}"
```

压力组不自动重试。若失败，检查：

```bash
grep -Ei 'out of memory|oom|killed process|bad_alloc' \
  "${STRESS_ROOT}/stderr.log" \
  "${STRESS_ROOT}/cgroup_journal.log" \
  "${STRESS_ROOT}/gpu_monitor/command.log" || true
```

确认 cgroup OOM 后，将结果报告为“在预定义 cap 下无法完成”，不要换一个更宽松 cap 后只报告成功结果。

## Required Comparison

只比较以下字段：

```text
external E2E wall
ACE partition count
partition labeling and dataset reordering
partition read, optimize, adjust, and write
final pack/materialization
optimize / E2E
peak process RSS and cgroup MemoryPeak
major page faults
disk throughput, queue depth, and utilization
GPU utilization, HBM peak, and low-utilization intervals
build outcome and cgroup OOM evidence
```

Baseline 的低 `optimize / E2E` 说明大量时间位于核心 CAGRA optimize 之外。压力组若出现更多 partition、更高 I/O、major faults 或 GPU 低利用率区间，可支持“主机内存压力放大外围路径并造成 GPU 供给不足”的假设。单点压力实验只用于机制验证，不用于宣称任意内存预算下的普遍性能曲线。

## Cleanup

提取 metrics 并记录索引大小后，删除两次运行生成的大索引，保留命令、日志、profiles 和 telemetry：

```bash
for root in "${BASELINE_ROOT}" "${STRESS_ROOT}"; do
  if [ -f "${root}/index/_mem.index" ]; then
    stat -c '%s' "${root}/index/_mem.index" > "${root}/mem_index_bytes.txt"
    rm -f "${root}/index/_mem.index"
  fi
done
```

失败运行可能留下 `${STRESS_ROOT}/ace_export`。先记录大小，再只删除本次生成的中间目录：

```bash
du -sh "${STRESS_ROOT}/ace_export" 2>/dev/null || true
rm -rf "${STRESS_ROOT}/ace_export"
```

不得删除 `stdout.log`、`stderr.log`、`time.log`、`profile.json`、`ace_profile.json`、`motivation_metrics.json`、`metadata.txt`、`host_monitor/`、`gpu_monitor/`、`cgroup_summary.txt` 或 `cgroup_journal.log`。
