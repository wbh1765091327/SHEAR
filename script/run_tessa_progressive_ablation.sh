#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_env.sh"
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root_base="${SHEAR_OUTPUT_ROOT}/tessa_ablation/wiki10m"
formal_parent_run="${run_root_base}/20260721_011618"
extension_subdir="full_build_2x2"
data_path="${SHEAR_DATA_ROOT}/wiki10m/wiki_10m_1024.fbin"
query_path="${SHEAR_DATA_ROOT}/wiki-1024-35M/query.fbin"
gt_path="${SHEAR_DATA_ROOT}/wiki10m/wiki_10m_gt100.bin"
summary_script="${repo}/script/summarize_tessa_progressive_ablation.py"
monitor_script="${repo}/script/monitor_gpu_build.sh"

die() {
    echo "error: $*" >&2
    exit 2
}

usage() {
    echo "Usage: bash script/run_tessa_progressive_ablation.sh PHASE RUN_ROOT" >&2
    echo "PHASE: print-contract|preflight|capture|replay|screen|repeat|search|summarize|cleanup" >&2
    echo "       print-2x2-contract|preflight-2x2|screen-2x2|repeat-2x2|search-2x2|summarize-2x2|cleanup-2x2" >&2
    echo "       print-fresh-2x2-contract|preflight-fresh-2x2|screen-fresh-2x2|repeat-fresh-2x2" >&2
    echo "       search-fresh-2x2|summarize-fresh-2x2|cleanup-fresh-2x2" >&2
    echo "       print-fresh-dp4a-contract|preflight-fresh-dp4a|screen-fresh-dp4a|repeat-fresh-dp4a" >&2
    echo "       search-fresh-dp4a|summarize-fresh-dp4a|cleanup-fresh-dp4a" >&2
}

[[ $# -eq 2 ]] || { usage; exit 2; }
phase="$1"
run_root="$(realpath -m "$2")"
case "${phase}" in
    print-contract|preflight|capture|replay|screen|repeat|search|summarize|cleanup|\
        print-2x2-contract|preflight-2x2|screen-2x2|repeat-2x2|search-2x2|summarize-2x2|cleanup-2x2|\
        print-fresh-2x2-contract|preflight-fresh-2x2|screen-fresh-2x2|repeat-fresh-2x2|\
        search-fresh-2x2|summarize-fresh-2x2|cleanup-fresh-2x2|\
        print-fresh-dp4a-contract|preflight-fresh-dp4a|screen-fresh-dp4a|repeat-fresh-dp4a|\
        search-fresh-dp4a|summarize-fresh-dp4a|cleanup-fresh-dp4a) ;;
    *) die "unsupported phase '${phase}'" ;;
esac

case "${phase}" in
    print-2x2-contract|preflight-2x2|screen-2x2|repeat-2x2|search-2x2|summarize-2x2|cleanup-2x2)
        [[ "${run_root}" == "${formal_parent_run}" ]] \
            || die "2x2 extension requires the accepted parent run: ${formal_parent_run}"
        ;;
esac
extension_root="${run_root}/${extension_subdir}"

common_environment=(
    SHEAR_PIPNN_PROFILE=1
    SHEAR_PIPNN_PHASE_MONITOR=1
    SHEAR_PIPNN_ENABLE_LEAF_WMMA_TOPK=1
    SHEAR_PIPNN_WMMA_CTA_SHARED_B=0
    SHEAR_PIPNN_COMPACT_LUNE_WITNESS=0
    SHEAR_PIPNN_COMPACT_LUNE_PRUNE=0
    SHEAR_PIPNN_COMPACT_SPILL_DIGEST=0
    SHEAR_PIPNN_PRUNE_BACKEND=hashprune
    SHEAR_PIPNN_LRPF_PRE_SPILL_COMPRESS=0
    OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
)

common_build_args=(
    --data-type float
    --dist-fn l2
    --data-path "${data_path}"
    --overwrite
    --oom-enable
    --oom-io-mode buffered
    --max-degree 64
    --lbuild 100
    --m-hash-bits 12
    --num-threads 8
    --leaf-size-min 256
    --leaf-size-max 1280
    --leaf-knn 2
    --psamp-fraction 0.05
    --fanout-top 6
    --fanout-second 3
    --root-risk-candidate-rs 1,2,4,6
    --seed 42
    --partitioner gpu-subtree
    --gpu-subtree-root-assignment-backend cpu
    --gpu-subtree-execution-mode fused-persistent
    --gpu-subtree-d0-capacity-enable
    --gpu-subtree-d0-partition-mode gpu-kmeans-coarse
    --gpu-subtree-sq-staging-mode gpu-d0
    --gpu-subtree-sq-staging-upload-shard-rows 262144
    --gpu-subtree-sq-wmma-sidecar-mode rebuild
    --gpu-subtree-sq8-d0-assignment-kernel wmma-batch
    --gpu-subtree-persistent-leaf-tile-rows 128
    --gpu-subtree-persistent-scheduler-mode auto
    --gpu-subtree-persistent-idle-backoff-ns 350
    --hashprune-backfill-mode rhalf-soft-tier-q3
    --hashprune-backfill-ablation none
    --hashprune-backfill-budget 0
    --hashprune-backfill-alpha 1.0
    --pipnn-backend cuda
    --pipnn-gpu-device 0
    --pipnn-gpu-workspace-budget-gb 16
)

variant_a0=(
    --gpu-subtree-sq-staging-raw-d1-diagnostic
    --gpu-subtree-sq-wmma-mode off
    --gpu-subtree-assignment-backend persistent-tiled-gemm-fp32
    --gpu-subtree-leaf-topk-backend persistent-tiled-gemm-fp32
    --gpu-subtree-persistent-large-ticket-batch 1
    --gpu-subtree-sq8-assignment-kernel auto
    --gpu-subtree-sq8-leaf-executor auto
    --gpu-subtree-sq8-leaf-mode auto
)
variant_a1=(
    --gpu-subtree-sq-staging-raw-d1-diagnostic
    --gpu-subtree-sq-wmma-mode off
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32
    --gpu-subtree-persistent-large-ticket-batch 1
    --gpu-subtree-sq8-assignment-kernel auto
    --gpu-subtree-sq8-leaf-executor auto
    --gpu-subtree-sq8-leaf-mode auto
)
variant_a2=(
    --gpu-subtree-sq-wmma-mode build
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32
    --gpu-subtree-persistent-large-ticket-batch 1
    --gpu-subtree-sq8-assignment-kernel warp-dp4a
    --gpu-subtree-sq8-leaf-executor warp-dp4a
    --gpu-subtree-sq8-leaf-mode staged
)
variant_a3=(
    --gpu-subtree-sq-wmma-mode build
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32
    --gpu-subtree-persistent-large-ticket-batch 8
    --gpu-subtree-sq8-assignment-kernel wmma
    --gpu-subtree-sq8-leaf-executor wmma
    --gpu-subtree-sq8-leaf-mode staged
)
variant_w1=(
    --gpu-subtree-sq-wmma-mode build
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32
    --gpu-subtree-persistent-large-ticket-batch 1
    --gpu-subtree-sq8-assignment-kernel wmma
    --gpu-subtree-sq8-leaf-executor wmma
    --gpu-subtree-sq8-leaf-mode staged
)
variant_d8=(
    --gpu-subtree-sq-wmma-mode build
    --gpu-subtree-assignment-backend persistent-wmma-tf32
    --gpu-subtree-leaf-topk-backend persistent-wmma-tf32
    --gpu-subtree-persistent-large-ticket-batch 8
    --gpu-subtree-sq8-assignment-kernel warp-dp4a
    --gpu-subtree-sq8-leaf-executor warp-dp4a
    --gpu-subtree-sq8-leaf-mode staged
)

variant_args() {
    case "$1" in
        A0) printf '%s\0' "${variant_a0[@]}" ;;
        A1) printf '%s\0' "${variant_a1[@]}" ;;
        A2) printf '%s\0' "${variant_a2[@]}" ;;
        A3) printf '%s\0' "${variant_a3[@]}" ;;
        W1) printf '%s\0' "${variant_w1[@]}" ;;
        D8) printf '%s\0' "${variant_d8[@]}" ;;
        *) die "unknown TESSA mode '$1'" ;;
    esac
}

resolve_variant_args() {
    local mode="$1"
    local -n output="$2"
    mapfile -d '' -t output < <(variant_args "${mode}")
}

build_command_array() {
    local mode="$1"
    local case_dir="$2"
    local capture_dir="${3:-}"
    local -n output="$4"
    local variant=()
    resolve_variant_args "${mode}" variant
    output=(
        env "${common_environment[@]}"
        /usr/bin/time -f $'wall_s\t%e\nmax_rss_kb\t%M' -o "${case_dir}/time.tsv"
        "${repo}/target/release/build_shear_index"
        "${common_build_args[@]}"
        "${variant[@]}"
        --index-path-prefix "${case_dir}/index"
        --oom-temp-dir "${case_dir}/oom"
        --oom-profile-json "${case_dir}/profile.json"
        --driver-timing-json "${case_dir}/driver_timing.json"
        --gpu-subtree-sq-wmma-sidecar-prefix "${case_dir}/sq_wmma/sidecar"
    )
    if [[ -n "${capture_dir}" ]]; then
        output+=(--gpu-subtree-tessa-capture-dir "${capture_dir}")
    fi
}

print_shell_command() {
    printf '%q ' "$@"
    printf '\n'
}

print_build_contract_at() {
    local mode="$1" case_dir="$2" heading="$3"
    local command=()
    build_command_array "${mode}" "${case_dir}" "" command
    echo "### ${heading}"
    print_shell_command timeout --signal=TERM --kill-after=30s 60m bash "${monitor_script}" \
        --out-dir "${case_dir}/gpu_monitor" --gpu 0 --interval 1 -- "${command[@]}"
}

print_build_contract() {
    local mode="$1"
    print_build_contract_at "${mode}" "${run_root}/full_builds/${mode}/run1" "${mode} screen command"
}

print_contract() {
    echo "# TESSA progressive ablation resolved contract"
    echo "RUN_ROOT=${run_root}"
    echo "# Formal replay: seed 42, 5 warmup sweeps, 20 measured sweeps"
    local mode
    for mode in A0 A1 A2 A3; do
        print_build_contract "${mode}"
    done
    echo "### Phase invocations"
    local contract_phase
    for contract_phase in preflight capture replay screen repeat search summarize cleanup; do
        print_shell_command bash "${repo}/script/run_tessa_progressive_ablation.sh" "${contract_phase}" "${run_root}"
    done
    echo "### Capture finalization"
    print_shell_command "${repo}/target/release/tessa_operator_replay" finalize-capture \
        --fragment-dir "${run_root}/capture/fragments" \
        --capture-profile "${run_root}/capture/build/profile.json" \
        --provenance-json "${run_root}/capture/provenance.json" \
        --output-dir "${run_root}/capture/final" --seed 42
    echo "### Operator replay"
    print_shell_command "${repo}/target/release/tessa_operator_replay" run \
        --manifest "${run_root}/capture/final/manifest.json" --base-path "${data_path}" \
        --output-dir "${run_root}/operator_replay" --gpu-device 0 --seed 42 \
        --warmup-sweeps 5 --measured-sweeps 20 --formal true
    echo "### Search command template"
    print_shell_command "${repo}/target/release/search_shear_index" \
        --index-path-prefix "${run_root}/full_builds/A0/run1/index" --data-path "${data_path}" \
        --query-file "${query_path}" --gt-file "${gt_path}" --result-path "RESULT_PATH" \
        --recall-at 10 --search-list-size "SEARCH_L" --num-threads 8
}

print_2x2_contract() {
    echo "# TESSA Wiki10M full-build 2x2 extension resolved contract"
    echo "PARENT_RUN=${run_root}"
    echo "EXTENSION_ROOT=${extension_root}"
    print_build_contract_at W1 "${extension_root}/full_builds/W1/run1" "W1 screen command"
    print_build_contract_at D8 "${extension_root}/full_builds/D8/run1" "D8 screen command"
    echo "### Phase invocations"
    local contract_phase
    for contract_phase in preflight-2x2 screen-2x2 repeat-2x2 search-2x2 summarize-2x2 cleanup-2x2; do
        print_shell_command bash "${repo}/script/run_tessa_progressive_ablation.sh" "${contract_phase}" "${run_root}"
    done
    echo "### Search command template"
    print_shell_command "${repo}/target/release/search_shear_index" \
        --index-path-prefix "${extension_root}/full_builds/W1/run1/index" --data-path "${data_path}" \
        --query-file "${query_path}" --gt-file "${gt_path}" --result-path "RESULT_PATH" \
        --recall-at 10 --search-list-size "SEARCH_L" --num-threads 8
}

print_fresh_2x2_contract() {
    echo "# TESSA Wiki10M fresh full-build 2x2 resolved contract"
    echo "RUN_ROOT=${run_root}"
    echo "# Cells: D1=A2, T1=W1, D8=D8, T8=A3"
    echo "# Build order: run1 A2/W1/D8/A3; run2 A3/D8/W1/A2; run3 W1/A3/A2/D8"
    local mode
    for mode in A2 W1 D8 A3; do
        print_build_contract_at "${mode}" "${run_root}/full_builds/${mode}/run1" \
            "${mode} fresh 2x2 screen command"
    done
    echo "### Phase invocations"
    local contract_phase
    for contract_phase in preflight-fresh-2x2 screen-fresh-2x2 repeat-fresh-2x2 \
        search-fresh-2x2 summarize-fresh-2x2 cleanup-fresh-2x2; do
        print_shell_command bash "${repo}/script/run_tessa_progressive_ablation.sh" \
            "${contract_phase}" "${run_root}"
    done
    echo "### Search command template"
    print_shell_command "${repo}/target/release/search_shear_index" \
        --index-path-prefix "${run_root}/full_builds/A2/run1/index" --data-path "${data_path}" \
        --query-file "${query_path}" --gt-file "${gt_path}" --result-path "RESULT_PATH" \
        --recall-at 10 --search-list-size "SEARCH_L" --num-threads 8
}

print_fresh_dp4a_contract() {
    echo "# TESSA Wiki10M fresh DP4A resolved contract"
    echo "RUN_ROOT=${run_root}"
    echo "# Cells: D1=A2 (batch 1), D8=D8 (batch 8)"
    echo "# Build order: run1 A2/D8; run2 D8/A2; run3 A2/D8"
    print_build_contract_at A2 "${run_root}/full_builds/A2/run1" \
        "D1 fresh DP4A screen command"
    print_build_contract_at D8 "${run_root}/full_builds/D8/run1" \
        "D8 fresh DP4A screen command"
    echo "### Phase invocations"
    local contract_phase
    for contract_phase in preflight-fresh-dp4a screen-fresh-dp4a repeat-fresh-dp4a \
        search-fresh-dp4a summarize-fresh-dp4a cleanup-fresh-dp4a; do
        print_shell_command bash "${repo}/script/run_tessa_progressive_ablation.sh" \
            "${contract_phase}" "${run_root}"
    done
    echo "### Search command template"
    print_shell_command "${repo}/target/release/search_shear_index" \
        --index-path-prefix "${run_root}/full_builds/A2/run1/index" --data-path "${data_path}" \
        --query-file "${query_path}" --gt-file "${gt_path}" --result-path "RESULT_PATH" \
        --recall-at 10 --search-list-size "SEARCH_L" --num-threads 8
}

if [[ "${phase}" == "print-contract" ]]; then
    print_contract
    exit 0
fi
if [[ "${phase}" == "print-2x2-contract" ]]; then
    print_2x2_contract
    exit 0
fi
if [[ "${phase}" == "print-fresh-2x2-contract" ]]; then
    print_fresh_2x2_contract
    exit 0
fi
if [[ "${phase}" == "print-fresh-dp4a-contract" ]]; then
    print_fresh_dp4a_contract
    exit 0
fi

case "${run_root}" in
    "${run_root_base}"/*) ;;
    *) die "RUN_ROOT must resolve below ${run_root_base}: ${run_root}" ;;
esac
[[ "${run_root}" != "${run_root_base}" ]] || die "RUN_ROOT cannot be the experiment base directory"
cd "${repo}"

require_preflight() {
    [[ -s "${run_root}/material_passport.md" ]] || die "preflight has not completed for ${run_root}"
}

require_2x2_preflight() {
    [[ "${run_root}" == "${formal_parent_run}" ]] || die "2x2 extension requires ${formal_parent_run}"
    [[ -s "${extension_root}/material_passport.md" ]] || die "2x2 extension preflight has not completed"
}

require_fresh_2x2_preflight() {
    [[ -s "${run_root}/fresh_2x2_preflight_complete.txt" ]] \
        || die "fresh 2x2 preflight has not completed for ${run_root}"
}

require_fresh_dp4a_preflight() {
    [[ -s "${run_root}/fresh_dp4a_preflight_complete.txt" ]] \
        || die "fresh DP4A preflight has not completed for ${run_root}"
}

check_header() {
    local path="$1" expected_rows="$2" expected_dim="$3" label="$4"
    [[ -s "${path}" ]] || die "missing ${label}: ${path}"
    local rows dim
    read -r rows dim < <(od -An -tu4 -N 8 "${path}")
    [[ "${rows}" == "${expected_rows}" && "${dim}" == "${expected_dim}" ]] \
        || die "unexpected ${label} header: got ${rows}x${dim}, expected ${expected_rows}x${expected_dim}"
}

run_preflight() {
    [[ ! -e "${run_root}" ]] || die "fresh RUN_ROOT already exists: ${run_root}"
    check_header "${data_path}" 10000000 1024 "Wiki10M base"
    check_header "${query_path}" 10000 1024 "Wiki query"
    check_header "${gt_path}" 10000 100 "Wiki10M ground truth"
    local binary
    for binary in build_shear_index search_shear_index graph_quality tessa_operator_replay; do
        [[ -x "${repo}/target/release/${binary}" ]] || die "missing release binary: target/release/${binary}"
    done
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    nvidia-smi -i 0 --query-gpu=name,compute_cap,memory.total --format=csv,noheader >/dev/null
    local available_kib required_kib
    available_kib="$(df -Pk ${SHEAR_OUTPUT_ROOT} | awk 'NR==2 {print $4}')"
    required_kib="${SHEAR_TESSA_MIN_FREE_KIB:-104857600}"
    (( available_kib >= required_kib )) || die "insufficient free space: ${available_kib} KiB < ${required_kib} KiB"

    mkdir -p "${run_root}"
    print_contract > "${run_root}/commands.sh.txt"
    sha256sum target/release/build_shear_index target/release/search_shear_index \
        target/release/graph_quality target/release/tessa_operator_replay > "${run_root}/binary_sha256.txt"
    {
        echo "# TESSA material passport"
        echo
        echo "- schema: tessa-progressive-ablation-v1"
        echo "- created_at: $(date --iso-8601=seconds)"
        echo "- run_root: ${run_root}"
        echo "- git_commit: $(git rev-parse HEAD)"
        echo "- git_diff_sha256: $(git diff --binary | sha256sum | awk '{print $1}')"
        echo "- base: ${data_path} (10000000 x 1024)"
        echo "- query: ${query_path} (10000 x 1024)"
        echo "- ground_truth: ${gt_path} (10000 x 100)"
        echo "- fixed_D0: gpu-kmeans-coarse + gpu-d0 SQ staging + wmma-batch; fanout 6/3/1; seed 42"
        echo "- replay: 5 warmup + 20 measured sweeps"
        echo "- build_repeats: 3; search_repeats: 3"
        echo "- timeout: 60m, SIGTERM then SIGKILL after 30s; no automatic retry"
        echo
        echo "## Environment contract"
        printf -- '- %s\n' "${common_environment[@]}"
        echo
        echo "## Git status at preflight"
        echo '```text'
        git status --short
        echo '```'
    } > "${run_root}/material_passport.md"
    {
        echo "# Prior evidence (design context only; not reusable as formal cells)"
        echo
        echo "- ${SHEAR_OUTPUT_ROOT}/persistent_sq_wmma_experiments/wiki10m_20260712_010422: old binary and no SQ8-DP4A leaf variant."
        echo "- ${SHEAR_OUTPUT_ROOT}/wiki10m_gpu_subtree_measurement/microbench/20260715_highdim_sq8_ablation_screen: mismatched fanout/config and leaf path."
        echo "- ${SHEAR_OUTPUT_ROOT}/wiki35m_gpu_subtree_measurement/runs/20260715_highdim_sq8_ablation_formal: Wiki35M coupled comparison, not this four-mode matrix."
        echo "- ${SHEAR_OUTPUT_ROOT}/wiki35m_gpu_subtree_measurement/runs/20260718_162354_sq8_d0_wmma_ab: D0-only root-stop run."
    } > "${run_root}/prior_evidence.md"
    env | LC_ALL=C sort > "${run_root}/environment.txt"
    nvidia-smi > "${run_root}/nvidia_smi.txt"
}

run_build() {
    local mode="$1" repeat="$2" case_dir="$3" capture_dir="${4:-}"
    [[ ! -e "${case_dir}" ]] || die "build case already exists; no automatic overwrite/retry: ${case_dir}"
    mkdir -p "${case_dir}/index" "${case_dir}/oom" "${case_dir}/sq_wmma" "${case_dir}/gpu_monitor"
    local command=()
    build_command_array "${mode}" "${case_dir}" "${capture_dir}" command
    print_shell_command timeout --signal=TERM --kill-after=30s 60m bash "${monitor_script}" \
        --out-dir "${case_dir}/gpu_monitor" --gpu 0 --interval 1 -- "${command[@]}" > "${case_dir}/resolved_command.sh"

    set +e
    timeout --signal=TERM --kill-after=30s 60m bash "${monitor_script}" \
        --out-dir "${case_dir}/gpu_monitor" --gpu 0 --interval 1 -- "${command[@]}" \
        > "${case_dir}/stdout.log" 2> "${case_dir}/stderr.log" &
    local build_pid=$!
    local previous_size=0 idle_seconds=0
    while kill -0 "${build_pid}" 2>/dev/null; do
        sleep 30
        local current_size
        current_size="$(( $(stat -c %s "${case_dir}/stdout.log" 2>/dev/null || echo 0) + $(stat -c %s "${case_dir}/stderr.log" 2>/dev/null || echo 0) ))"
        if [[ "${current_size}" == "${previous_size}" ]]; then
            idle_seconds=$((idle_seconds + 30))
        else
            idle_seconds=0
            previous_size="${current_size}"
        fi
        echo "heartbeat mode=${mode} repeat=${repeat} pid=${build_pid} output_bytes=${current_size} idle_seconds=${idle_seconds}" | tee -a "${case_dir}/heartbeat.log"
        if (( idle_seconds >= 90 )); then
            echo "warning: no new stdout/stderr for ${idle_seconds}s; build continues until the 60m hard timeout" | tee -a "${case_dir}/heartbeat.log" >&2
        fi
    done
    wait "${build_pid}"
    local status=$?
    set -e
    printf '%s\n' "${status}" > "${case_dir}/exit_status"
    [[ "${status}" -eq 0 ]] || die "${mode} repeat ${repeat} failed with status ${status}; no retry performed"

    "${repo}/target/release/graph_quality" --index-path-prefix "${case_dir}/index" \
        --data-path "${data_path}" --output "${case_dir}/graph_quality.tsv"
    python3 "${summary_script}" record-build --case-dir "${case_dir}" --mode "${mode}" --repeat "${repeat}" \
        > "${case_dir}/validation.json"
}

validate_existing_build_matrix() {
    local repeats="$1" mode repeat
    for mode in A0 A1 A2 A3; do
        for repeat in $(seq 1 "${repeats}"); do
            [[ -s "${run_root}/full_builds/${mode}/run${repeat}/build_record.json" ]] \
                || die "missing accepted ${mode} run${repeat} build record"
        done
    done
    RUN_ROOT="${run_root}" EXPECTED_REPEATS="${repeats}" python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
repeats = int(os.environ["EXPECTED_REPEATS"])
records = [json.loads((root / "full_builds" / mode / f"run{repeat}" / "build_record.json").read_text()) for mode in ("A0", "A1", "A2", "A3") for repeat in range(1, repeats + 1)]
capture_record = root / "capture" / "build" / "build_record.json"
if capture_record.is_file():
    records.append(json.loads(capture_record.read_text()))
for field in ("root_leader_hash", "d0_partition_hash", "d0_task_list_hash"):
    values = {row[field] for row in records}
    if len(values) != 1:
        raise SystemExit(f"D0 hash mismatch in accepted build matrix for {field}: {values}")
PY
}

validate_parent_2x2_inputs() {
    [[ "${run_root}" == "${formal_parent_run}" ]] || die "unexpected 2x2 parent run: ${run_root}"
    [[ -s "${run_root}/material_passport.md" ]] || die "missing parent material passport"
    [[ -s "${run_root}/summary.json" ]] || die "missing parent summary"
    [[ -s "${run_root}/cleanup_complete.txt" ]] || die "parent cleanup is incomplete"
    sha256sum --check --status "${run_root}/binary_sha256.txt" \
        || die "current release binaries do not match the parent experiment"
    validate_existing_build_matrix 3

    local mode search_l repeat
    for mode in A0 A2 A3; do
        for search_l in 25 50 100 200 400; do
            for repeat in 1 2 3; do
                [[ -s "${run_root}/search/${mode}/L${search_l}/rep${repeat}.json" ]] \
                    || die "missing reusable parent search record: ${mode} L=${search_l} repeat=${repeat}"
            done
        done
    done
}

validate_2x2_build_matrix() {
    local repeats="$1" mode repeat
    for mode in A2 A3; do
        for repeat in $(seq 1 "${repeats}"); do
            [[ -s "${run_root}/full_builds/${mode}/run${repeat}/build_record.json" ]] \
                || die "missing accepted parent ${mode} run${repeat} build record"
        done
    done
    for mode in W1 D8; do
        for repeat in $(seq 1 "${repeats}"); do
            [[ -s "${extension_root}/full_builds/${mode}/run${repeat}/build_record.json" ]] \
                || die "missing accepted extension ${mode} run${repeat} build record"
        done
    done

    RUN_ROOT="${run_root}" EXTENSION_ROOT="${extension_root}" EXPECTED_REPEATS="${repeats}" \
        SUMMARY_SCRIPT="${summary_script}" python3 - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
extension_root = Path(os.environ["EXTENSION_ROOT"])
repeats = int(os.environ["EXPECTED_REPEATS"])
spec = importlib.util.spec_from_file_location("tessa_summary", os.environ["SUMMARY_SCRIPT"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
records = []
for mode in ("A2", "A3"):
    for repeat in range(1, repeats + 1):
        records.append(json.loads((run_root / "full_builds" / mode / f"run{repeat}" / "build_record.json").read_text()))
for mode in ("W1", "D8"):
    for repeat in range(1, repeats + 1):
        records.append(json.loads((extension_root / "full_builds" / mode / f"run{repeat}" / "build_record.json").read_text()))
module.validate_extension_build_records(records, tuple(range(1, repeats + 1)))
PY
}

validate_fresh_2x2_build_matrix() {
    local repeats="$1" mode repeat
    for mode in A2 W1 D8 A3; do
        for repeat in $(seq 1 "${repeats}"); do
            [[ -s "${run_root}/full_builds/${mode}/run${repeat}/build_record.json" ]] \
                || die "missing fresh 2x2 ${mode} run${repeat} build record"
        done
    done

    RUN_ROOT="${run_root}" EXPECTED_REPEATS="${repeats}" SUMMARY_SCRIPT="${summary_script}" \
        python3 - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
repeats = int(os.environ["EXPECTED_REPEATS"])
spec = importlib.util.spec_from_file_location("tessa_summary", os.environ["SUMMARY_SCRIPT"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
records = [
    json.loads(
        (run_root / "full_builds" / mode / f"run{repeat}" / "build_record.json").read_text()
    )
    for mode in ("A2", "W1", "D8", "A3")
    for repeat in range(1, repeats + 1)
]
module.validate_extension_build_records(records, tuple(range(1, repeats + 1)))
PY
}

validate_fresh_dp4a_build_matrix() {
    local repeats="$1" mode repeat
    for mode in A2 D8; do
        for repeat in $(seq 1 "${repeats}"); do
            [[ -s "${run_root}/full_builds/${mode}/run${repeat}/build_record.json" ]] \
                || die "missing fresh DP4A ${mode} run${repeat} build record"
        done
    done

    RUN_ROOT="${run_root}" EXPECTED_REPEATS="${repeats}" SUMMARY_SCRIPT="${summary_script}" \
        python3 - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
repeats = int(os.environ["EXPECTED_REPEATS"])
spec = importlib.util.spec_from_file_location("tessa_summary", os.environ["SUMMARY_SCRIPT"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
records = [
    json.loads(
        (run_root / "full_builds" / mode / f"run{repeat}" / "build_record.json").read_text()
    )
    for mode in ("A2", "D8")
    for repeat in range(1, repeats + 1)
]
module.validate_dp4a_build_records(records, tuple(range(1, repeats + 1)))
PY
}

run_fresh_2x2_preflight() {
    local scope="${1:-2x2}"
    local contract_fn schema title cells marker selected_description
    case "${scope}" in
        2x2)
            contract_fn="print_fresh_2x2_contract"
            schema="tessa-full-build-2x2-fresh-v1"
            title="TESSA fresh full-build 2x2"
            cells="D1=A2,D8=D8,T1=W1,T8=A3"
            marker="fresh_2x2_preflight_complete.txt"
            selected_description="All four cells"
            ;;
        dp4a)
            contract_fn="print_fresh_dp4a_contract"
            schema="tessa-full-build-dp4a-fresh-v1"
            title="TESSA fresh DP4A"
            cells="D1=A2,D8=D8"
            marker="fresh_dp4a_preflight_complete.txt"
            selected_description="Both DP4A cells"
            ;;
        *) die "unknown fresh preflight scope: ${scope}" ;;
    esac
    [[ ! -e "${run_root}" ]] || die "fresh RUN_ROOT already exists: ${run_root}"
    check_header "${data_path}" 10000000 1024 "Wiki10M base"
    check_header "${query_path}" 10000 1024 "Wiki query"
    check_header "${gt_path}" 10000 100 "Wiki10M ground truth"
    python3 - "${gt_path}" <<'PY'
import array
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as handle:
    rows, width = struct.unpack("<II", handle.read(8))
    values = array.array("I")
    values.fromfile(handle, rows * width)
if sys.byteorder != "little":
    values.byteswap()
if len(values) != rows * width or max(values, default=0) >= 10_000_000:
    raise SystemExit("Wiki10M ground truth contains an invalid or out-of-range ID")
PY
    local binary
    for binary in build_shear_index search_shear_index graph_quality; do
        [[ -x "${repo}/target/release/${binary}" ]] \
            || die "missing release binary: target/release/${binary}"
    done
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    nvidia-smi -i 0 --query-gpu=name,compute_cap,memory.total --format=csv,noheader >/dev/null
    local available_kib required_kib
    available_kib="$(df -Pk ${SHEAR_OUTPUT_ROOT} | awk 'NR==2 {print $4}')"
    required_kib="${SHEAR_TESSA_MIN_FREE_KIB:-104857600}"
    (( available_kib >= required_kib )) \
        || die "insufficient free space: ${available_kib} KiB < ${required_kib} KiB"

    mkdir -p "${run_root}"
    "${contract_fn}" > "${run_root}/commands.sh.txt"
    sha256sum target/release/build_shear_index target/release/search_shear_index \
        target/release/graph_quality > "${run_root}/binary_sha256.txt"
    sha256sum \
        "${repo}/script/run_tessa_progressive_ablation.sh" \
        "${repo}/script/summarize_tessa_progressive_ablation.py" \
        "${repo}/script/test_tessa_progressive_ablation.py" \
        "${repo}/docs/runbooks/tessa_progressive_ablation.md" \
        > "${run_root}/script_sha256.txt"
    {
        echo "# ${title} material passport"
        echo
        echo "- schema: ${schema}"
        echo "- created_at: $(date --iso-8601=seconds)"
        echo "- run_root: ${run_root}"
        echo "- git_commit: $(git rev-parse HEAD)"
        echo "- git_diff_sha256: $(git diff --binary | sha256sum | awk '{print $1}')"
        echo "- base: ${data_path} (10000000 x 1024)"
        echo "- query: ${query_path} (10000 x 1024)"
        echo "- ground_truth: ${gt_path} (10000 x 100; IDs range-checked)"
        echo "- cells: ${cells}"
        echo "- fixed_D0: gpu-kmeans-coarse + gpu-d0 SQ staging + wmma-batch; fanout 6/3/1; seed 42"
        echo "- build_repeats: 3; search_repeats: 3"
        echo "- search_l: 25,50,100,200,400"
        echo "- timeout: 60m, SIGTERM then SIGKILL after 30s; no automatic retry"
        echo
        echo "## Environment contract"
        printf -- '- %s\n' "${common_environment[@]}"
        echo
        echo "## Git status at preflight"
        echo '```text'
        git status --short
        echo '```'
    } > "${run_root}/material_passport.md"
    {
        echo "# Prior evidence (historical context only)"
        echo
        echo "- ${SHEAR_OUTPUT_ROOT}/tessa_ablation/wiki10m/20260721_011618 used the pre-fix leaf executor."
        echo "- Its forced warp-DP4A leaf cell executed scalar int8 products and is not reused."
        echo "- ${selected_description} in this run use the same newly built release binary."
    } > "${run_root}/prior_evidence.md"
    env | LC_ALL=C sort > "${run_root}/environment.txt"
    nvidia-smi > "${run_root}/nvidia_smi.txt"
    df -Pk ${SHEAR_OUTPUT_ROOT} > "${run_root}/disk_free.txt"
    date --iso-8601=seconds > "${run_root}/${marker}"
}

run_fresh_dp4a_preflight() {
    run_fresh_2x2_preflight dp4a
}

run_2x2_preflight() {
    [[ ! -e "${extension_root}" ]] || die "2x2 extension already exists: ${extension_root}"
    validate_parent_2x2_inputs
    check_header "${data_path}" 10000000 1024 "Wiki10M base"
    check_header "${query_path}" 10000 1024 "Wiki query"
    check_header "${gt_path}" 10000 100 "Wiki10M ground truth"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    nvidia-smi -i 0 --query-gpu=name,compute_cap,memory.total --format=csv,noheader >/dev/null

    local available_kib required_kib
    available_kib="$(df -Pk ${SHEAR_OUTPUT_ROOT} | awk 'NR==2 {print $4}')"
    required_kib="${SHEAR_TESSA_MIN_FREE_KIB:-104857600}"
    (( available_kib >= required_kib )) \
        || die "insufficient free space: ${available_kib} KiB < ${required_kib} KiB"

    mkdir -p "${extension_root}"
    print_2x2_contract > "${extension_root}/commands.sh.txt"
    cp -- "${run_root}/binary_sha256.txt" "${extension_root}/parent_binary_sha256.txt"
    sha256sum \
        "${repo}/script/run_tessa_progressive_ablation.sh" \
        "${repo}/script/summarize_tessa_progressive_ablation.py" \
        "${repo}/script/test_tessa_progressive_ablation.py" \
        "${repo}/docs/runbooks/tessa_progressive_ablation.md" \
        > "${extension_root}/script_sha256.txt"
    {
        echo "# TESSA full-build 2x2 extension material passport"
        echo
        echo "- schema: tessa-full-build-2x2-extension-v1"
        echo "- created_at: $(date --iso-8601=seconds)"
        echo "- parent_run: ${run_root}"
        echo "- extension_root: ${extension_root}"
        echo "- git_commit: $(git rev-parse HEAD)"
        echo "- git_diff_sha256: $(git diff --binary | sha256sum | awk '{print $1}')"
        echo "- new_modes: W1,D8"
        echo "- fixed_D0: gpu-kmeans-coarse + gpu-d0 SQ staging + wmma-batch; fanout 6/3/1; seed 42"
        echo "- build_repeats: 3"
        echo "- search_repeats: 3"
        echo "- search_l: 25,50,100,200,400"
        echo "- timeout: 60m, SIGTERM then SIGKILL after 30s; no automatic retry"
        echo
        echo "## Git status at extension preflight"
        echo '```text'
        git status --short
        echo '```'
    } > "${extension_root}/material_passport.md"
    env | LC_ALL=C sort > "${extension_root}/environment.txt"
    nvidia-smi > "${extension_root}/nvidia_smi.txt"
    df -Pk ${SHEAR_OUTPUT_ROOT} > "${extension_root}/disk_free.txt"
}

run_2x2_screen() {
    require_2x2_preflight
    run_build W1 1 "${extension_root}/full_builds/W1/run1"
    run_build D8 1 "${extension_root}/full_builds/D8/run1"
    validate_2x2_build_matrix 1
}

run_2x2_repeat() {
    require_2x2_preflight
    validate_2x2_build_matrix 1
    run_build D8 2 "${extension_root}/full_builds/D8/run2"
    remove_repeat_index "${extension_root}/full_builds/D8/run2"
    run_build W1 2 "${extension_root}/full_builds/W1/run2"
    remove_repeat_index "${extension_root}/full_builds/W1/run2"
    run_build W1 3 "${extension_root}/full_builds/W1/run3"
    remove_repeat_index "${extension_root}/full_builds/W1/run3"
    run_build D8 3 "${extension_root}/full_builds/D8/run3"
    remove_repeat_index "${extension_root}/full_builds/D8/run3"
    validate_2x2_build_matrix 3
}

run_fresh_2x2_screen() {
    require_fresh_2x2_preflight
    local mode
    for mode in A2 W1 D8 A3; do
        run_build "${mode}" 1 "${run_root}/full_builds/${mode}/run1"
    done
    validate_fresh_2x2_build_matrix 1
}

run_fresh_2x2_repeat() {
    require_fresh_2x2_preflight
    validate_fresh_2x2_build_matrix 1
    local mode
    for mode in A3 D8 W1 A2; do
        run_build "${mode}" 2 "${run_root}/full_builds/${mode}/run2"
        remove_repeat_index "${run_root}/full_builds/${mode}/run2"
    done
    for mode in W1 A3 A2 D8; do
        run_build "${mode}" 3 "${run_root}/full_builds/${mode}/run3"
        remove_repeat_index "${run_root}/full_builds/${mode}/run3"
    done
    validate_fresh_2x2_build_matrix 3
}

run_fresh_dp4a_screen() {
    require_fresh_dp4a_preflight
    run_build A2 1 "${run_root}/full_builds/A2/run1"
    run_build D8 1 "${run_root}/full_builds/D8/run1"
    validate_fresh_dp4a_build_matrix 1
}

run_fresh_dp4a_repeat() {
    require_fresh_dp4a_preflight
    validate_fresh_dp4a_build_matrix 1
    local mode
    for mode in D8 A2; do
        run_build "${mode}" 2 "${run_root}/full_builds/${mode}/run2"
        remove_repeat_index "${run_root}/full_builds/${mode}/run2"
    done
    for mode in A2 D8; do
        run_build "${mode}" 3 "${run_root}/full_builds/${mode}/run3"
        remove_repeat_index "${run_root}/full_builds/${mode}/run3"
    done
    validate_fresh_dp4a_build_matrix 3
}

run_capture() {
    require_preflight
    local capture_root="${run_root}/capture"
    [[ ! -e "${capture_root}" ]] || die "capture phase already started; no automatic retry"
    mkdir -p "${capture_root}"
    run_build A3 1 "${capture_root}/build" "${capture_root}/fragments"
    RUN_ROOT="${run_root}" DATA_PATH="${data_path}" REPO="${repo}" python3 - <<'PY'
import hashlib, json, os, struct
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
case = root / "capture" / "build"
profile = json.loads((case / "profile.json").read_text())
base = Path(os.environ["DATA_PATH"]).resolve()
with base.open("rb") as handle:
    header = handle.read(8)
rows, dim = struct.unpack("<II", header)
command_hash = hashlib.sha256((case / "resolved_command.sh").read_bytes()).hexdigest()
binary_hash = hashlib.sha256((Path(os.environ["REPO"]) / "target/release/build_shear_index").read_bytes()).hexdigest()
metadata = {
    "dataset": {"canonical_base_path": str(base), "rows": rows, "dimension": dim, "header_sha256": hashlib.sha256(header).hexdigest()},
    "provenance": {
        "seed": 42,
        "git_commit": os.popen(f"git -C {os.environ['REPO']} rev-parse HEAD").read().strip(),
        "binary_sha256": binary_hash,
        "command_sha256": command_hash,
        "root_leader_hash": profile["root_fanout"]["root_leader_hash"],
        "d0_partition_hash": profile["gpu_subtree"]["d0_partition_hash"],
        "d0_task_list_hash": profile["gpu_subtree"]["d0_task_list_hash"],
    },
}
(root / "capture" / "provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
    target/release/tessa_operator_replay finalize-capture \
        --fragment-dir "${capture_root}/fragments" --capture-profile "${capture_root}/build/profile.json" \
        --provenance-json "${capture_root}/provenance.json" --output-dir "${capture_root}/final" --seed 42 \
        > "${capture_root}/finalize.stdout.log" 2> "${capture_root}/finalize.stderr.log"
    [[ -s "${capture_root}/final/manifest.json" ]] || die "capture finalization did not produce manifest.json"
    rm -f -- "${capture_root}/build/index/_mem.index"
}

run_replay() {
    require_preflight
    [[ -s "${run_root}/capture/final/manifest.json" ]] || die "capture manifest is missing"
    [[ ! -e "${run_root}/operator_replay" ]] || die "replay phase already started; no automatic retry"
    target/release/tessa_operator_replay run --manifest "${run_root}/capture/final/manifest.json" \
        --base-path "${data_path}" --output-dir "${run_root}/operator_replay" --gpu-device 0 --seed 42 \
        --warmup-sweeps 5 --measured-sweeps 20 --formal true
    [[ -s "${run_root}/operator_replay/summary.json" ]] || die "operator replay summary is missing"
    python3 - "${run_root}/operator_replay/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x["formal"] is True and x["warmup_sweeps"] == 5 and x["measured_sweeps"] == 20
assert x["a2_a3_candidate_ids_and_distance_bits_exact"] is True
for mode in x["modes"]:
    safety = mode["safety_counters"]
    assert safety.get("fallback_count", 0) == 0
PY
}

run_screen() {
    require_preflight
    [[ -s "${run_root}/operator_replay/summary.json" ]] || die "successful operator replay is required before screening"
    local mode
    for mode in A0 A1 A2 A3; do
        run_build "${mode}" 1 "${run_root}/full_builds/${mode}/run1"
    done
    validate_existing_build_matrix 1
}

remove_repeat_index() {
    local case_dir="$1"
    [[ -s "${case_dir}/build_record.json" ]] || die "refusing repeat-index removal without durable build record"
    rm -f -- "${case_dir}/index/_mem.index"
}

run_repeat() {
    require_preflight
    validate_existing_build_matrix 1
    local mode
    for mode in A3 A2 A1 A0; do
        run_build "${mode}" 2 "${run_root}/full_builds/${mode}/run2"
        remove_repeat_index "${run_root}/full_builds/${mode}/run2"
    done
    for mode in A1 A3 A0 A2; do
        run_build "${mode}" 3 "${run_root}/full_builds/${mode}/run3"
        remove_repeat_index "${run_root}/full_builds/${mode}/run3"
    done
    validate_existing_build_matrix 3
}

run_one_search() {
    local mode="$1" search_l="$2" repeat="$3" warmup="$4"
    local index_dir="${run_root}/full_builds/${mode}/run1/index"
    [[ -s "${index_dir}/_mem.index" ]] || die "retained screening index is missing for ${mode}"
    local output_dir="${run_root}/search/${mode}/L${search_l}"
    mkdir -p "${output_dir}"
    local suffix="rep${repeat}"
    [[ "${warmup}" == "1" ]] && suffix="warmup"
    local result_path="${output_dir}/${suffix}.fbin"
    local log_path="${output_dir}/${suffix}.log"
    [[ ! -e "${log_path}" && ! -e "${result_path}" ]] || die "search cell already exists; no retry: ${mode} L=${search_l} ${suffix}"
    target/release/search_shear_index --index-path-prefix "${index_dir}" --data-path "${data_path}" \
        --query-file "${query_path}" --gt-file "${gt_path}" --result-path "${result_path}" \
        --recall-at 10 --search-list-size "${search_l}" --num-threads 8 > "${log_path}" 2>&1
    if [[ "${warmup}" == "0" ]]; then
        python3 "${summary_script}" record-search --mode "${mode}" --search-l "${search_l}" \
            --repeat "${repeat}" --log "${log_path}" --result "${result_path}" \
            --output "${output_dir}/rep${repeat}.json" > "${output_dir}/rep${repeat}.validation.json"
    fi
}

run_search() {
    require_preflight
    validate_existing_build_matrix 3
    local search_l mode repeat
    for search_l in 25 50 100 200 400; do
        for mode in A0 A1 A2 A3; do
            run_one_search "${mode}" "${search_l}" 0 1
        done
        repeat=1
        for mode in A0 A1 A2 A3; do run_one_search "${mode}" "${search_l}" "${repeat}" 0; done
        repeat=2
        for mode in A3 A2 A1 A0; do run_one_search "${mode}" "${search_l}" "${repeat}" 0; done
        repeat=3
        for mode in A1 A3 A0 A2; do run_one_search "${mode}" "${search_l}" "${repeat}" 0; done
    done
}

run_one_2x2_search() {
    local mode="$1" search_l="$2" repeat="$3" warmup="$4"
    local index_dir="${extension_root}/full_builds/${mode}/run1/index"
    [[ -s "${index_dir}/_mem.index" ]] || die "retained extension index is missing for ${mode}"
    local output_dir="${extension_root}/search/${mode}/L${search_l}"
    mkdir -p "${output_dir}"
    local suffix="rep${repeat}"
    [[ "${warmup}" == "1" ]] && suffix="warmup"
    local result_path="${output_dir}/${suffix}.fbin"
    local log_path="${output_dir}/${suffix}.log"
    [[ ! -e "${log_path}" && ! -e "${result_path}" ]] \
        || die "extension search cell already exists; no retry: ${mode} L=${search_l} ${suffix}"
    target/release/search_shear_index --index-path-prefix "${index_dir}" --data-path "${data_path}" \
        --query-file "${query_path}" --gt-file "${gt_path}" --result-path "${result_path}" \
        --recall-at 10 --search-list-size "${search_l}" --num-threads 8 > "${log_path}" 2>&1
    if [[ "${warmup}" == "0" ]]; then
        python3 "${summary_script}" record-search --mode "${mode}" --search-l "${search_l}" \
            --repeat "${repeat}" --log "${log_path}" --result "${result_path}" \
            --output "${output_dir}/rep${repeat}.json" > "${output_dir}/rep${repeat}.validation.json"
    fi
}

run_2x2_search() {
    require_2x2_preflight
    validate_2x2_build_matrix 3
    local search_l mode
    for search_l in 25 50 100 200 400; do
        for mode in W1 D8; do
            run_one_2x2_search "${mode}" "${search_l}" 0 1
        done
        for mode in W1 D8; do run_one_2x2_search "${mode}" "${search_l}" 1 0; done
        for mode in D8 W1; do run_one_2x2_search "${mode}" "${search_l}" 2 0; done
        for mode in W1 D8; do run_one_2x2_search "${mode}" "${search_l}" 3 0; done
    done
}

run_fresh_2x2_search() {
    require_fresh_2x2_preflight
    validate_fresh_2x2_build_matrix 3
    local search_l mode
    for search_l in 25 50 100 200 400; do
        for mode in A2 W1 D8 A3; do
            run_one_search "${mode}" "${search_l}" 0 1
        done
        for mode in A2 W1 D8 A3; do
            run_one_search "${mode}" "${search_l}" 1 0
        done
        for mode in A3 D8 W1 A2; do
            run_one_search "${mode}" "${search_l}" 2 0
        done
        for mode in W1 A3 A2 D8; do
            run_one_search "${mode}" "${search_l}" 3 0
        done
    done
}

run_fresh_dp4a_search() {
    require_fresh_dp4a_preflight
    validate_fresh_dp4a_build_matrix 3
    local search_l mode
    for search_l in 25 50 100 200 400; do
        for mode in A2 D8; do
            run_one_search "${mode}" "${search_l}" 0 1
        done
        for mode in A2 D8; do
            run_one_search "${mode}" "${search_l}" 1 0
        done
        for mode in D8 A2; do
            run_one_search "${mode}" "${search_l}" 2 0
        done
        for mode in A2 D8; do
            run_one_search "${mode}" "${search_l}" 3 0
        done
    done
}

run_summarize() {
    require_preflight
    python3 "${summary_script}" summarize "${run_root}"
}

run_2x2_summarize() {
    require_2x2_preflight
    validate_2x2_build_matrix 3
    python3 "${summary_script}" summarize-2x2 "${run_root}"
}

run_fresh_2x2_summarize() {
    require_fresh_2x2_preflight
    validate_fresh_2x2_build_matrix 3
    python3 "${summary_script}" summarize-fresh-2x2 "${run_root}"
}

run_fresh_dp4a_summarize() {
    require_fresh_dp4a_preflight
    validate_fresh_dp4a_build_matrix 3
    python3 "${summary_script}" summarize-fresh-dp4a "${run_root}"
}

run_cleanup() {
    require_preflight
    [[ -s "${run_root}/summary.md" && -s "${run_root}/summary.json" && -s "${run_root}/results.tsv" ]] \
        || die "refusing cleanup before completed summary"
    validate_existing_build_matrix 3
    local mode search_l repeat
    for mode in A0 A1 A2 A3; do
        python3 - "${run_root}/full_builds/${mode}/run1/build_record.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["index_bytes"] > 0
PY
        for search_l in 25 50 100 200 400; do
            for repeat in 1 2 3; do
                [[ -s "${run_root}/search/${mode}/L${search_l}/rep${repeat}.json" ]] \
                    || die "refusing cleanup: missing ${mode} L=${search_l} repeat=${repeat} search record"
            done
        done
    done

    local manifest="${run_root}/cleanup_manifest.txt"
    : > "${manifest}"
    find "${run_root}/full_builds" "${run_root}/capture" -type f -name '_mem.index' -print >> "${manifest}"
    find "${run_root}/full_builds" "${run_root}/capture" -type f -path '*/sq_wmma/*' -print >> "${manifest}"
    find "${run_root}/full_builds" "${run_root}/capture" -type f -path '*/oom/*' -print >> "${manifest}"
    while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        case "${target}" in "${run_root}"/*) ;; *) die "cleanup target escaped RUN_ROOT: ${target}" ;; esac
        rm -f -- "${target}"
    done < "${manifest}"
    date --iso-8601=seconds > "${run_root}/cleanup_complete.txt"
}

run_2x2_cleanup() {
    require_2x2_preflight
    [[ -s "${extension_root}/summary.md" && -s "${extension_root}/summary.json" \
        && -s "${extension_root}/results.tsv" ]] || die "refusing 2x2 cleanup before completed summary"
    validate_2x2_build_matrix 3

    local mode search_l repeat
    for mode in W1 D8; do
        python3 - "${extension_root}/full_builds/${mode}/run1/build_record.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["index_bytes"] > 0
PY
        for search_l in 25 50 100 200 400; do
            for repeat in 1 2 3; do
                [[ -s "${extension_root}/search/${mode}/L${search_l}/rep${repeat}.json" ]] \
                    || die "refusing 2x2 cleanup: missing ${mode} L=${search_l} repeat=${repeat} search record"
            done
        done
    done

    local manifest="${extension_root}/cleanup_manifest.txt"
    : > "${manifest}"
    find "${extension_root}/full_builds" -type f -name '_mem.index' -print >> "${manifest}"
    find "${extension_root}/full_builds" -type f -path '*/sq_wmma/*' -print >> "${manifest}"
    find "${extension_root}/full_builds" -type f -path '*/oom/*' -print >> "${manifest}"
    while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        case "${target}" in "${extension_root}"/*) ;; *) die "cleanup target escaped extension root: ${target}" ;; esac
        rm -f -- "${target}"
    done < "${manifest}"
    date --iso-8601=seconds > "${extension_root}/cleanup_complete.txt"
}

run_fresh_2x2_cleanup() {
    require_fresh_2x2_preflight
    [[ -s "${run_root}/summary.md" && -s "${run_root}/summary.json" \
        && -s "${run_root}/results.tsv" ]] \
        || die "refusing fresh 2x2 cleanup before completed summary"
    validate_fresh_2x2_build_matrix 3

    local mode search_l repeat
    for mode in A2 W1 D8 A3; do
        python3 - "${run_root}/full_builds/${mode}/run1/build_record.json" <<'PY'
import json
import sys

assert json.load(open(sys.argv[1]))["index_bytes"] > 0
PY
        for search_l in 25 50 100 200 400; do
            for repeat in 1 2 3; do
                [[ -s "${run_root}/search/${mode}/L${search_l}/rep${repeat}.json" ]] \
                    || die "refusing fresh 2x2 cleanup: missing ${mode} L=${search_l} repeat=${repeat}"
            done
        done
    done

    local manifest="${run_root}/cleanup_manifest.txt"
    : > "${manifest}"
    find "${run_root}/full_builds" -type f -name '_mem.index' -print >> "${manifest}"
    find "${run_root}/full_builds" -type f -path '*/sq_wmma/*' -print >> "${manifest}"
    find "${run_root}/full_builds" -type f -path '*/oom/*' -print >> "${manifest}"
    while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        case "${target}" in "${run_root}"/*) ;; *) die "cleanup target escaped RUN_ROOT: ${target}" ;; esac
        rm -f -- "${target}"
    done < "${manifest}"
    date --iso-8601=seconds > "${run_root}/cleanup_complete.txt"
}

run_fresh_dp4a_cleanup() {
    require_fresh_dp4a_preflight
    [[ -s "${run_root}/summary.md" && -s "${run_root}/summary.json" \
        && -s "${run_root}/results.tsv" ]] \
        || die "refusing fresh DP4A cleanup before completed summary"
    validate_fresh_dp4a_build_matrix 3

    local mode search_l repeat
    for mode in A2 D8; do
        python3 - "${run_root}/full_builds/${mode}/run1/build_record.json" <<'PY'
import json
import sys

assert json.load(open(sys.argv[1]))["index_bytes"] > 0
PY
        for search_l in 25 50 100 200 400; do
            for repeat in 1 2 3; do
                [[ -s "${run_root}/search/${mode}/L${search_l}/rep${repeat}.json" ]] \
                    || die "refusing fresh DP4A cleanup: missing ${mode} L=${search_l} repeat=${repeat}"
            done
        done
    done

    local manifest="${run_root}/cleanup_manifest.txt"
    : > "${manifest}"
    find "${run_root}/full_builds" -type f -name '_mem.index' -print >> "${manifest}"
    find "${run_root}/full_builds" -type f -path '*/sq_wmma/*' -print >> "${manifest}"
    find "${run_root}/full_builds" -type f -path '*/oom/*' -print >> "${manifest}"
    while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        case "${target}" in "${run_root}"/*) ;; *) die "cleanup target escaped RUN_ROOT: ${target}" ;; esac
        rm -f -- "${target}"
    done < "${manifest}"
    date --iso-8601=seconds > "${run_root}/cleanup_complete.txt"
}

case "${phase}" in
    preflight) run_preflight ;;
    capture) run_capture ;;
    replay) run_replay ;;
    screen) run_screen ;;
    repeat) run_repeat ;;
    search) run_search ;;
    summarize) run_summarize ;;
    cleanup) run_cleanup ;;
    preflight-2x2) run_2x2_preflight ;;
    screen-2x2) run_2x2_screen ;;
    repeat-2x2) run_2x2_repeat ;;
    search-2x2) run_2x2_search ;;
    summarize-2x2) run_2x2_summarize ;;
    cleanup-2x2) run_2x2_cleanup ;;
    preflight-fresh-2x2) run_fresh_2x2_preflight ;;
    screen-fresh-2x2) run_fresh_2x2_screen ;;
    repeat-fresh-2x2) run_fresh_2x2_repeat ;;
    search-fresh-2x2) run_fresh_2x2_search ;;
    summarize-fresh-2x2) run_fresh_2x2_summarize ;;
    cleanup-fresh-2x2) run_fresh_2x2_cleanup ;;
    preflight-fresh-dp4a) run_fresh_dp4a_preflight ;;
    screen-fresh-dp4a) run_fresh_dp4a_screen ;;
    repeat-fresh-dp4a) run_fresh_dp4a_repeat ;;
    search-fresh-dp4a) run_fresh_dp4a_search ;;
    summarize-fresh-dp4a) run_fresh_dp4a_summarize ;;
    cleanup-fresh-dp4a) run_fresh_dp4a_cleanup ;;
esac
