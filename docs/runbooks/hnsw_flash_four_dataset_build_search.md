# HNSW-Flash four-dataset build and search runbook

This runbook applies the repository-owned benchmark patch to an immutable archive of
HNSW-Flash commit `716e002596c9002810c6000072100a7ed0c64e33`. It never edits the
pre-existing dirty checkout. Formal inputs come from `/data_nvme_kioxia/tyj/data`;
all generated artifacts and retained evidence go under `/data_staging/wbh`.

The fixed build configuration is `K=10`, `M=32`, `efConstruction=1024`, PCA 32,
PQ 16x16 (4-bit), `PQLINK_STORE`, `PQLINK_CALC`, AVX, 64 build threads, and seed
42. Search uses 8 threads and `efSearch={25,50,100,200,400}`, with one warm-up and
two measured full-query repetitions.

Extract the fenced script with:

```bash
python3 script/test_hnsw_flash_four_dataset_runbook.py --extract > /tmp/hnsw_flash_runbook.sh
bash /tmp/hnsw_flash_runbook.sh help
```

<!-- BEGIN RUNBOOK_SCRIPT -->
```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/path/to/SHEAR
SOURCE_REPO=/data_staging/wbh/wiki35m_baseline_repos_20260630/HNSW-Flash
SOURCE_COMMIT=716e002596c9002810c6000072100a7ed0c64e33
PATCH_FILE="$REPO_ROOT/script/patches/hnsw_flash_four_dataset_benchmark.patch"
DATA_ROOT=/data_nvme_kioxia/tyj/data
OUTPUT_ROOT=/data_staging/wbh/hnsw_flash_four_dataset
SOURCE_TREE="$OUTPUT_ROOT/source-$SOURCE_COMMIT-benchmark-v2"
BIN="$SOURCE_TREE/bin/hnsw_flash_benchmark"
LOCK_FILE="$OUTPUT_ROOT/formal.lock"

FORMAL_DATASETS=(wiki35m laion100m msmarco70m deep100m)
SEARCH_EFS=(25 50 100 200 400)
BUILD_THREADS=64
SEARCH_THREADS=8
SEED=42
M=32
EF_CONSTRUCTION=1024
SEARCH_REPETITIONS=2
SEARCH_WARMUPS=1

dataset_config() {
    case "$1" in
        wiki35m)
            ROWS=35000000; DIM=1024
            BASE="$DATA_ROOT/wiki-1024-35M/en_base.fbin"
            QUERY="$DATA_ROOT/wiki-1024-35M/query.fbin"
            GT="$DATA_ROOT/wiki-1024-35M/gt100.bin"
            ;;
        laion100m)
            ROWS=100000000; DIM=512
            BASE="$DATA_ROOT/laion-512-100M/base.fbin"
            QUERY="$DATA_ROOT/laion-512-100M/query.fbin"
            GT="$DATA_ROOT/laion-512-100M/gt100.bin"
            ;;
        msmarco70m)
            ROWS=70000000; DIM=768
            BASE="$DATA_ROOT/msmarco-768-70M/base.fbin"
            QUERY="$DATA_ROOT/msmarco-768-114M/query.fbin"
            GT="$DATA_ROOT/msmarco-768-70M/gt100.bin"
            ;;
        deep100m)
            ROWS=100000000; DIM=96
            BASE="$DATA_ROOT/deep-96-100M/base.100M.fbin"
            QUERY="$DATA_ROOT/deep-96-100M/query.public.10K.fbin"
            GT="$DATA_ROOT/deep-96-100M/groundtruth.public.10K.ibin"
            ;;
        *) echo "unknown dataset: $1" >&2; return 2 ;;
    esac
}

metric() {
    local key=$1 log=$2
    awk -v key="$key" '$1 == "METRIC" && $2 == key {value=$3} END {if (value != "") print value}' "$log"
}

time_value() {
    local key=$1 log=$2
    awk -F: -v key="$key" 'index($1,key) {sub(/^[ \t]+/,"",$2); print $2}' "$log" | tail -1
}

elapsed_to_seconds() {
    awk -v value="$1" 'BEGIN {
        n=split(value,a,":");
        if (n==3) print a[1]*3600+a[2]*60+a[3];
        else if (n==2) print a[1]*60+a[2];
        else print value;
    }'
}

prepare() {
    mkdir -p "$OUTPUT_ROOT"
    if [[ ! -x "$BIN" ]]; then
        local temporary="$OUTPUT_ROOT/source.tmp.$$"
        [[ ! -e "$temporary" ]]
        mkdir -p "$temporary"
        git -C "$SOURCE_REPO" archive "$SOURCE_COMMIT" | tar -xf - -C "$temporary"
        patch -d "$temporary" -p1 -i "$PATCH_FILE"
        cmake -S "$temporary" -B "$temporary/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$temporary/build" -j 8 --target hnsw_flash_benchmark
        mv "$temporary" "$SOURCE_TREE"
    fi
    "$BIN" 2>&1 | grep -q 'usage: hnsw_flash_benchmark' || true
    echo "$BIN"
}

preflight() {
    prepare >/dev/null
    local available_kb free_kb
    available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    free_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
    (( available_kb >= 225 * 1024 * 1024 )) || {
        echo "need at least 225 GiB MemAvailable; found $((available_kb/1024/1024)) GiB" >&2
        return 1
    }
    (( free_kb >= 350 * 1024 * 1024 )) || {
        echo "need at least 350 GiB free output space; found $((free_kb/1024/1024)) GiB" >&2
        return 1
    }
    sudo -n true
    for dataset in "${FORMAL_DATASETS[@]}"; do
        dataset_config "$dataset"
        [[ -r "$BASE" && -r "$QUERY" && -r "$GT" ]]
    done
    echo "preflight: PASS"
}

drop_caches() {
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

cleanup_accepted_index() {
    local candidate=$1 artifacts=$2 resolved
    resolved=$(realpath -e "$candidate")
    artifacts=$(realpath -e "$artifacts")
    [[ "$resolved" == "$artifacts"/* ]]
    [[ "$resolved" == *.index ]]
    rm -f -- "$resolved"
}

run_dataset() {
    local dataset=$1 run_dir=$2
    dataset_config "$dataset"
    local ds="$run_dir/$dataset" artifacts="$run_dir/$dataset/artifacts"
    mkdir -p "$artifacts" "$ds/logs" "$ds/search"
    local index="$artifacts/hnsw_flash.index"
    local codebook="$artifacts/hnsw_flash.codebook"
    local build_log="$ds/logs/build.log" time_log="$ds/logs/build.time"
    local build_metrics="$ds/build_internal.tsv"
    local index_bytes codebook_bytes index_sha codebook_sha

    if [[ -s "$index" && -s "$codebook" && -s "$ds/build_summary.tsv" ]]; then
        echo "resume: reuse validated build for $dataset" >&2
        "$BIN" validate --base "$BASE" --codebook "$codebook" --index "$index" \
            --expected-rows "$ROWS" --metrics "$ds/validate.tsv"
        index_bytes=$(stat -c %s "$index")
        index_sha=$(sha256sum "$index" | awk '{print $1}')
        tail -n +2 "$ds/build_summary.tsv" >>"$run_dir/build_summary.tsv"
    else

        drop_caches
        local prewarm_start prewarm_end build_prewarm_seconds
        prewarm_start=$(date +%s.%N)
        dd if="$BASE" of=/dev/null bs=64M status=none
        prewarm_end=$(date +%s.%N)
        build_prewarm_seconds=$(awk -v a="$prewarm_start" -v b="$prewarm_end" 'BEGIN {print b-a}')

        drop_caches
        /usr/bin/time -v "$BIN" build \
            --base "$BASE" --codebook "$codebook" --index "$index" \
            --metrics "$build_metrics" --threads "$BUILD_THREADS" --seed "$SEED" \
            >"$build_log" 2>"$time_log"
        "$BIN" validate --base "$BASE" --codebook "$codebook" --index "$index" \
            --expected-rows "$ROWS" --metrics "$ds/validate.tsv"

    local elapsed build_external_wall_seconds maxrss input_blocks output_blocks
        elapsed=$(awk '/Elapsed \(wall clock\)/ {print $NF}' "$time_log")
        build_external_wall_seconds=$(elapsed_to_seconds "$elapsed")
    maxrss=$(time_value "Maximum resident set size" "$time_log")
    input_blocks=$(time_value "File system inputs" "$time_log")
    output_blocks=$(time_value "File system outputs" "$time_log")
    local peak_process_tree_rss_bytes=$((maxrss * 1024))
    local physical_read_bytes=$((input_blocks * 512))
    local physical_write_bytes=$((output_blocks * 512))
    local base_load_seconds pca_fit_seconds pca_encode_base_seconds pq_train_seconds
    local hnsw_insert_seconds artifact_serialize_seconds build_internal_e2e_seconds
    base_load_seconds=$(metric base_load_seconds "$build_log")
    pca_fit_seconds=$(metric pca_fit_seconds "$build_log")
    pca_encode_base_seconds=$(metric pca_encode_base_seconds "$build_log")
    pq_train_seconds=$(metric pq_train_seconds "$build_log")
    hnsw_insert_seconds=$(metric hnsw_insert_seconds "$build_log")
    artifact_serialize_seconds=$(metric artifact_serialize_seconds "$build_log")
    build_internal_e2e_seconds=$(awk -F'\t' '$1=="internal_e2e_seconds"{print $2}' "$build_metrics")
        index_bytes=$(stat -c %s "$index"); codebook_bytes=$(stat -c %s "$codebook")
        index_sha=$(sha256sum "$index" | awk '{print $1}')
        codebook_sha=$(sha256sum "$codebook" | awk '{print $1}')
        {
            echo -e "dataset\trows\tdim\tbase_load_seconds\tpca_fit_seconds\tpca_encode_base_seconds\tpq_train_seconds\thnsw_insert_seconds\tartifact_serialize_seconds\tbuild_internal_e2e_seconds\tbuild_external_wall_seconds\tbuild_prewarm_seconds\tpeak_process_tree_rss_bytes\tphysical_read_bytes\tphysical_write_bytes\tindex_bytes\tcodebook_bytes\tindex_sha256\tcodebook_sha256"
            echo -e "$dataset\t$ROWS\t$DIM\t$base_load_seconds\t$pca_fit_seconds\t$pca_encode_base_seconds\t$pq_train_seconds\t$hnsw_insert_seconds\t$artifact_serialize_seconds\t$build_internal_e2e_seconds\t$build_external_wall_seconds\t$build_prewarm_seconds\t$peak_process_tree_rss_bytes\t$physical_read_bytes\t$physical_write_bytes\t$index_bytes\t$codebook_bytes\t$index_sha\t$codebook_sha"
        } >"$ds/build_summary.tsv"
        tail -n +2 "$ds/build_summary.tsv" >>"$run_dir/build_summary.tsv"
    fi

    for ef in "${SEARCH_EFS[@]}"; do
        for ((warmup=1; warmup<=SEARCH_WARMUPS; warmup++)); do
            "$BIN" search --base "$BASE" --query "$QUERY" --ground-truth "$GT" \
                --codebook "$codebook" --index "$index" \
                --output "$ds/search/ef${ef}.warmup${warmup}.ivecs" \
                --metrics "$ds/search/ef${ef}.warmup${warmup}.tsv" \
                --threads "$SEARCH_THREADS" --seed "$SEED" --ef-search "$ef" \
                >"$ds/logs/search.ef${ef}.warmup${warmup}.log" 2>&1
        done
        local expected_sha=
        for ((rep=1; rep<=SEARCH_REPETITIONS; rep++)); do
            local result="$ds/search/ef${ef}.rep${rep}.ivecs"
            local search_log="$ds/logs/search.ef${ef}.rep${rep}.log"
            local search_time="$ds/logs/search.ef${ef}.rep${rep}.time"
            local search_metrics="$ds/search/ef${ef}.rep${rep}.tsv"
            /usr/bin/time -v "$BIN" search --base "$BASE" --query "$QUERY" --ground-truth "$GT" \
                --codebook "$codebook" --index "$index" --output "$result" \
                --metrics "$search_metrics" --threads "$SEARCH_THREADS" --seed "$SEED" \
                --ef-search "$ef" >"$search_log" 2>"$search_time"
            local recall seconds qps result_sha query_rows
            recall=$(awk -F'\t' '$1=="recall_at_10"{print $2}' "$search_metrics")
            seconds=$(metric search_seconds "$search_log")
            query_rows=$(python3 - "$QUERY" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    print(struct.unpack("<I", f.read(4))[0])
PY
)
            qps=$(awk -v n="$query_rows" -v s="$seconds" 'BEGIN {if(s>0) print n/s; else print 0}')
            result_sha=$(sha256sum "$result" | awk '{print $1}')
            [[ -z "$expected_sha" || "$expected_sha" == "$result_sha" ]]
            expected_sha=$result_sha
            echo -e "$dataset\t$ef\t$rep\t$query_rows\t$seconds\t$qps\t$recall\t$result_sha" \
                >>"$run_dir/search_summary.tsv"
        done
    done

    printf '{"dataset":"%s","status":"accepted","index_bytes":%s,"index_sha256":"%s"}\n' \
        "$dataset" "$index_bytes" "$index_sha" >"$ds/status.json"
    cleanup_accepted_index "$index" "$artifacts"
}

smoke() {
    prepare >/dev/null
    local run="$OUTPUT_ROOT/smoke/$(date +%Y%m%d-%H%M%S)"
    local data="$SOURCE_TREE/data/cohere10k"
    mkdir -p "$run"
    "$BIN" build --base "$data/cohere10k_base.fvecs" --codebook "$run/codebook.bin" \
        --index "$run/index.bin" --metrics "$run/build.tsv" --threads 8 --seed "$SEED"
    "$BIN" validate --base "$data/cohere10k_base.fvecs" --codebook "$run/codebook.bin" \
        --index "$run/index.bin" --expected-rows 10000 --metrics "$run/validate.tsv"
    "$BIN" search --base "$data/cohere10k_base.fvecs" --query "$data/cohere10k_query.fvecs" \
        --ground-truth "$data/cohere10k_groundtruth.ivecs" --codebook "$run/codebook.bin" \
        --index "$run/index.bin" --output "$run/result.ivecs" --metrics "$run/search.tsv" \
        --threads 8 --seed "$SEED" --ef-search 25
    echo "$run"
}

summarize() {
    local run_dir=$1
    {
        echo "# HNSW-Flash four-dataset result"
        echo
        echo "- Source commit: \`$SOURCE_COMMIT\`"
        echo "- Build: K=10, M=$M, efConstruction=$EF_CONSTRUCTION, PCA=32, PQ=16x16, threads=$BUILD_THREADS"
        echo "- Search: efSearch=${SEARCH_EFS[*]}, threads=$SEARCH_THREADS, warmups=$SEARCH_WARMUPS, repetitions=$SEARCH_REPETITIONS"
        echo
        echo "Detailed machine-readable results are in \`build_summary.tsv\` and \`search_summary.tsv\`."
    } >"$run_dir/summary.md"
}

formal() {
    preflight
    exec 9>"$LOCK_FILE"
    flock -n 9 || { echo "another formal run owns $LOCK_FILE" >&2; return 1; }
    local run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
    local run_dir="$OUTPUT_ROOT/runs/$run_id"
    mkdir -p "$run_dir"
    printf '{"status":"running","run_id":"%s","source_commit":"%s"}\n' "$run_id" "$SOURCE_COMMIT" >"$run_dir/status.json"
    echo -e "ordinal\tdataset\trows\tdim" >"$run_dir/formal_schedule.tsv"
    echo -e "dataset\trows\tdim\tbase_load_seconds\tpca_fit_seconds\tpca_encode_base_seconds\tpq_train_seconds\thnsw_insert_seconds\tartifact_serialize_seconds\tbuild_internal_e2e_seconds\tbuild_external_wall_seconds\tbuild_prewarm_seconds\tpeak_process_tree_rss_bytes\tphysical_read_bytes\tphysical_write_bytes\tindex_bytes\tcodebook_bytes\tindex_sha256\tcodebook_sha256" >"$run_dir/build_summary.tsv"
    echo -e "dataset\tef_search\trepetition\tqueries\tsearch_seconds\tqps\trecall_at_10\tresult_sha256" >"$run_dir/search_summary.tsv"
    local ordinal=0
    for dataset in "${FORMAL_DATASETS[@]}"; do
        ordinal=$((ordinal+1)); dataset_config "$dataset"
        echo -e "$ordinal\t$dataset\t$ROWS\t$DIM" >>"$run_dir/formal_schedule.tsv"
        run_dataset "$dataset" "$run_dir"
    done
    summarize "$run_dir"
    printf '{"status":"accepted","run_id":"%s","source_commit":"%s"}\n' "$run_id" "$SOURCE_COMMIT" >"$run_dir/status.json"
    echo "$run_dir"
}

case "${1:-help}" in
    help) echo "usage: $0 {help|prepare|preflight|smoke|formal|summarize RUN_DIR}" ;;
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    formal) formal ;;
    summarize) summarize "${2:?run directory required}" ;;
    *) echo "unknown mode: $1" >&2; exit 2 ;;
esac
```
<!-- END RUNBOOK_SCRIPT -->

An accepted dataset retains logs, timing files, codebook, validation metadata,
search results, checksums, and summaries. Its large `.index` is deleted only after
both measured repetitions agree byte-for-byte for every `efSearch` value.
