#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_env.sh"
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash script/monitor_gpu_build.sh [options] -- <build command> [args...]

Options:
  --out-dir DIR       Directory for monitor logs.
                      Default: ${SHEAR_OUTPUT_ROOT}/shear_gpu_monitor/<timestamp>_<tag>
  --interval SECONDS  Sampling interval for nvidia-smi. Default: 1
  --gpu IDS           Optional GPU id list for nvidia-smi -i, for example 0 or 0,1.
                      Default: all visible GPUs.
  --tag NAME          Tag used in the default output directory. Default: build
  --no-process-samples
                      Do not sample the nvidia-smi compute-process table.
  -h, --help          Show this help.

Examples:
  bash script/monitor_gpu_build.sh --gpu 0 --interval 1 -- \
    env SHEAR_PIPNN_PROFILE=1 target/release/build_shear_index ...

  bash script/monitor_gpu_build.sh --out-dir "${RUN_ROOT}/gpu_monitor" -- \
    bash -lc 'SHEAR_PIPNN_PROFILE=1 /usr/bin/time -v -o "${TIME_LOG}" target/release/build_shear_index ...'

Outputs:
  gpu_samples.csv       Timestamped GPU utilization, memory, power, temp, clock samples.
  gpu_processes.csv     Timestamped compute-process memory samples.
  command.log           Combined stdout/stderr of the wrapped command.
  command.sh            Re-runnable command record.
  metadata.txt          Host, git, nvidia-smi, and monitor configuration.
  summary.txt           Exit code plus simple aggregate GPU statistics.
USAGE
}

die() {
    echo "error: $*" >&2
    exit 2
}

interval="1"
gpu_ids=""
tag="build"
out_dir=""
process_samples=1

while (($# > 0)); do
    case "$1" in
        --out-dir)
            (($# >= 2)) || die "--out-dir requires a value"
            out_dir="$2"
            shift 2
            ;;
        --interval)
            (($# >= 2)) || die "--interval requires a value"
            interval="$2"
            shift 2
            ;;
        --gpu)
            (($# >= 2)) || die "--gpu requires a value"
            gpu_ids="$2"
            shift 2
            ;;
        --tag)
            (($# >= 2)) || die "--tag requires a value"
            tag="$2"
            shift 2
            ;;
        --no-process-samples)
            process_samples=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

(($# > 0)) || die "missing build command; pass it after --"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found in PATH"

if ! awk -v value="$interval" 'BEGIN { exit !(value + 0 > 0) }'; then
    die "--interval must be a positive number"
fi

safe_tag="$(printf '%s' "$tag" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_*//; s/_*$//')"
if [[ -z "$safe_tag" ]]; then
    safe_tag="build"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$out_dir" ]]; then
    monitor_root="${GPU_BUILD_MONITOR_ROOT:-${SHEAR_OUTPUT_ROOT}/shear_gpu_monitor}"
    out_dir="${monitor_root}/${timestamp}_${safe_tag}"
fi

mkdir -p "$out_dir"
out_dir="$(realpath "$out_dir")"

gpu_log="${out_dir}/gpu_samples.csv"
process_log="${out_dir}/gpu_processes.csv"
command_log="${out_dir}/command.log"
command_record="${out_dir}/command.sh"
metadata_log="${out_dir}/metadata.txt"
summary_log="${out_dir}/summary.txt"
monitor_error_log="${out_dir}/monitor_errors.log"

gpu_args=()
if [[ -n "$gpu_ids" ]]; then
    gpu_args=(-i "$gpu_ids")
fi

printf '%s\n' \
    "sample_epoch,sample_time,gpu_timestamp,gpu_index,gpu_name,gpu_util_percent,mem_util_percent,memory_used_mib,memory_total_mib,power_draw_w,temp_c,sm_clock_mhz,pstate" \
    > "$gpu_log"
printf '%s\n' \
    "sample_epoch,sample_time,gpu_uuid,pid,process_name,used_memory_mib" \
    > "$process_log"
: > "$monitor_error_log"

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$PWD"
    printf 'exec'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
} > "$command_record"
chmod +x "$command_record"

start_epoch="$(date +%s.%N)"
start_time="$(date --iso-8601=seconds)"

{
    echo "date_start=${start_time}"
    echo "host=$(hostname)"
    echo "cwd=$(pwd)"
    echo "out_dir=${out_dir}"
    echo "interval_seconds=${interval}"
    echo "gpu_filter=${gpu_ids:-all}"
    echo "process_samples=${process_samples}"
    echo "command_begin"
    printf '%q ' "$@"
    printf '\n'
    echo "command_end"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_begin"
    git status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo "nvidia_smi_list_begin"
    nvidia-smi -L 2>/dev/null || true
    echo "nvidia_smi_list_end"
    echo "nvidia_smi_snapshot_begin"
    nvidia-smi "${gpu_args[@]}" 2>/dev/null || true
    echo "nvidia_smi_snapshot_end"
} > "$metadata_log" 2>&1

sample_once() {
    local sample_epoch
    local sample_time
    local rows
    local process_rows

    sample_epoch="$(date +%s.%N)"
    sample_time="$(LC_ALL=C date -u '+%Y-%m-%dT%H:%M:%S.%NZ')"

    if rows="$(
        nvidia-smi "${gpu_args[@]}" \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,pstate \
            --format=csv,noheader,nounits 2>>"$monitor_error_log"
    )"; then
        while IFS= read -r row; do
            [[ -n "$row" ]] || continue
            printf '%s,%s,%s\n' "$sample_epoch" "$sample_time" "$row" >> "$gpu_log"
        done <<< "$rows"
    else
        printf '%s %s gpu query failed\n' "$sample_time" "$sample_epoch" >> "$monitor_error_log"
    fi

    if ((process_samples == 1)); then
        if process_rows="$(
            nvidia-smi "${gpu_args[@]}" \
                --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>>"$monitor_error_log" \
                | sed '/No running processes found/d'
        )"; then
            while IFS= read -r row; do
                [[ -n "$row" ]] || continue
                printf '%s,%s,%s\n' "$sample_epoch" "$sample_time" "$row" >> "$process_log"
            done <<< "$process_rows"
        fi
    fi
}

monitor_loop() {
    while true; do
        sample_once
        sleep "$interval" || break
    done
}

write_summary() {
    local command_rc="$1"
    local end_epoch
    local end_time
    local elapsed

    end_epoch="$(date +%s.%N)"
    end_time="$(date --iso-8601=seconds)"
    elapsed="$(awk -v start="$start_epoch" -v end="$end_epoch" 'BEGIN { printf "%.3f", end - start }')"

    {
        echo "command_exit_code=${command_rc}"
        echo "date_start=${start_time}"
        echo "date_end=${end_time}"
        echo "elapsed_seconds=${elapsed}"
        echo "out_dir=${out_dir}"
        echo "command_log=${command_log}"
        echo "gpu_samples_csv=${gpu_log}"
        echo "gpu_processes_csv=${process_log}"
        awk -F, '
            NR > 1 {
                n += 1
                gpu_util = $6 + 0
                mem_util = $7 + 0
                mem_used = $8 + 0
                power = $10 + 0
                temp = $11 + 0
                util_sum += gpu_util
                mem_util_sum += mem_util
                mem_used_sum += mem_used
                power_sum += power
                temp_sum += temp
                if (gpu_util > util_max) util_max = gpu_util
                if (mem_util > mem_util_max) mem_util_max = mem_util
                if (mem_used > mem_used_max) mem_used_max = mem_used
                if (power > power_max) power_max = power
                if (temp > temp_max) temp_max = temp
            }
            END {
                printf "gpu_sample_rows=%d\n", n
                if (n > 0) {
                    printf "gpu_util_avg_percent=%.2f\n", util_sum / n
                    printf "gpu_util_max_percent=%.2f\n", util_max
                    printf "mem_util_avg_percent=%.2f\n", mem_util_sum / n
                    printf "mem_util_max_percent=%.2f\n", mem_util_max
                    printf "memory_used_avg_mib=%.2f\n", mem_used_sum / n
                    printf "memory_used_max_mib=%.2f\n", mem_used_max
                    printf "power_draw_avg_w=%.2f\n", power_sum / n
                    printf "power_draw_max_w=%.2f\n", power_max
                    printf "temp_avg_c=%.2f\n", temp_sum / n
                    printf "temp_max_c=%.2f\n", temp_max
                }
            }
        ' "$gpu_log"
    } > "$summary_log"
}

monitor_pid=""
cleanup_monitor() {
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}

trap 'cleanup_monitor' EXIT
trap 'cleanup_monitor; exit 130' INT
trap 'cleanup_monitor; exit 143' TERM

echo "GPU monitor output: ${out_dir}"
monitor_loop &
monitor_pid="$!"

set +e
"$@" 2>&1 | tee "$command_log"
command_rc="${PIPESTATUS[0]}"
set -e

cleanup_monitor
trap - EXIT INT TERM
write_summary "$command_rc"

echo "command exit code: ${command_rc}"
echo "GPU monitor summary: ${summary_log}"
exit "$command_rc"
