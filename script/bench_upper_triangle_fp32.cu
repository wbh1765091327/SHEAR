// Standalone FP32 distance-tile experiment for symmetric leaf top-k.
// Build: nvcc -O3 -std=c++17 -arch=sm_70 script/bench_upper_triangle_fp32.cu -o /tmp/bench_upper_triangle_fp32
// Run:   /tmp/bench_upper_triangle_fp32 [rows] [leaves] [dim] [iterations]
// Benchmarks both dense-matrix selection and tile-local bidirectional top-k.
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

constexpr int TILE = 16;
constexpr int TOPK = 8;

__global__ void distances_fp32(const float* __restrict__ x,
                               float* __restrict__ dist,
                               int rows, int dim, int upper_only) {
    const int tile_count = (rows + TILE - 1) / TILE;
    const int tile_id = blockIdx.x;
    int tile_i = 0, tile_j = 0;
    if (upper_only) {
        // Enumerate (0,0), (0,1), ..., (0,T-1), (1,1), ... without
        // allocating square-grid blocks that immediately return.
        int remaining = tile_id;
        for (int i = 0; i < tile_count; ++i) {
            const int count = tile_count - i;
            if (remaining < count) {
                tile_i = i;
                tile_j = i + remaining;
                break;
            }
            remaining -= count;
        }
    } else {
        tile_i = tile_id / tile_count;
        tile_j = tile_id % tile_count;
    }

    const int i = tile_i * TILE + threadIdx.y;
    const int j = tile_j * TILE + threadIdx.x;
    if (i >= rows || j >= rows) return;
    const size_t leaf = blockIdx.y;
    const float* a = x + (leaf * rows + i) * dim;
    const float* b = x + (leaf * rows + j) * dim;
    float sum = 0.0f;
    for (int axis = 0; axis < dim; ++axis) {
        const float delta = a[axis] - b[axis];
        sum = fmaf(delta, delta, sum);
    }
    // Diagonal tiles still evaluate every valid pair. Only self-pairs are
    // masked; off-diagonal tiles reuse one result for both source rows.
    const float value = i == j ? CUDART_INF_F : sum;
    float* leaf_dist = dist + leaf * size_t(rows) * rows;
    leaf_dist[size_t(i) * rows + j] = value;
    if (upper_only && tile_i != tile_j) {
        leaf_dist[size_t(j) * rows + i] = value;
    }
}

__global__ void select_topk(const float* __restrict__ dist,
                            int* __restrict__ ids, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int leaf = blockIdx.y;
    if (row >= rows) return;
    float best_dist[TOPK];
    int best_id[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        best_dist[k] = CUDART_INF_F;
        best_id[k] = -1;
    }
    const float* source = dist + (size_t(leaf) * rows + row) * rows;
    for (int col = 0; col < rows; ++col) {
        const float value = source[col];
        if (value < best_dist[TOPK - 1]) {
            int slot = TOPK - 1;
            while (slot > 0 && value < best_dist[slot - 1]) {
                best_dist[slot] = best_dist[slot - 1];
                best_id[slot] = best_id[slot - 1];
                --slot;
            }
            best_dist[slot] = value;
            best_id[slot] = col;
        }
    }
    int* output = ids + (size_t(leaf) * rows + row) * TOPK;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) output[k] = best_id[k];
}

// Each tile writes a sorted local top-k list for every affected source row.
// For an off-diagonal upper tile, the same FP32 distances feed both its rows
// and its columns. Diagonal tiles update rows only and mask self-pairs.
__global__ void tile_topk_fp32(const float* __restrict__ x,
                              float* __restrict__ partial_dist,
                              int* __restrict__ partial_id,
                              int rows, int dim, int upper_only) {
    const int tile_count = (rows + TILE - 1) / TILE;
    const int tile_id = blockIdx.x;
    int tile_i = 0, tile_j = 0;
    if (upper_only) {
        int remaining = tile_id;
        for (int i = 0; i < tile_count; ++i) {
            const int count = tile_count - i;
            if (remaining < count) {
                tile_i = i;
                tile_j = i + remaining;
                break;
            }
            remaining -= count;
        }
    } else {
        tile_i = tile_id / tile_count;
        tile_j = tile_id % tile_count;
    }
    const int local_i = threadIdx.y;
    const int local_j = threadIdx.x;
    const int i = tile_i * TILE + local_i;
    const int j = tile_j * TILE + local_j;
    const size_t leaf = blockIdx.y;
    __shared__ float values[TILE][TILE];
    if (i < rows && j < rows) {
        const float* a = x + (leaf * rows + i) * dim;
        const float* b = x + (leaf * rows + j) * dim;
        float sum = 0.0f;
        for (int axis = 0; axis < dim; ++axis) {
            const float delta = a[axis] - b[axis];
            sum = fmaf(delta, delta, sum);
        }
        values[local_i][local_j] = i == j ? CUDART_INF_F : sum;
    }
    __syncthreads();

    const int lane = local_i * TILE + local_j;
    if (lane >= 2 * TILE || (lane >= TILE && (!upper_only || tile_i == tile_j))) return;
    const bool column = lane >= TILE;
    const int local_source = lane % TILE;
    const int source = (column ? tile_j : tile_i) * TILE + local_source;
    if (source >= rows) return;
    const int candidate_tile = column ? tile_i : tile_j;
    float best_dist[TOPK];
    int best_id[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        best_dist[k] = CUDART_INF_F;
        best_id[k] = -1;
    }
    for (int local = 0; local < TILE; ++local) {
        const int target = candidate_tile * TILE + local;
        if (target >= rows || source == target) continue;
        const float value = column ? values[local][local_source]
                                   : values[local_source][local];
        if (value < best_dist[TOPK - 1]) {
            int slot = TOPK - 1;
            while (slot > 0 && value < best_dist[slot - 1]) {
                best_dist[slot] = best_dist[slot - 1];
                best_id[slot] = best_id[slot - 1];
                --slot;
            }
            best_dist[slot] = value;
            best_id[slot] = target;
        }
    }
    const size_t offset = ((leaf * rows + source) * tile_count + candidate_tile) * TOPK;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        partial_dist[offset + k] = best_dist[k];
        partial_id[offset + k] = best_id[k];
    }
}

__global__ void merge_tile_topk(const float* __restrict__ partial_dist,
                                const int* __restrict__ partial_id,
                                int* __restrict__ ids, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int leaf = blockIdx.y;
    if (row >= rows) return;
    const int tile_count = (rows + TILE - 1) / TILE;
    const size_t offset = (size_t(leaf) * rows + row) * tile_count * TOPK;
    float best_dist[TOPK];
    int best_id[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        best_dist[k] = CUDART_INF_F;
        best_id[k] = -1;
    }
    for (int candidate = 0; candidate < tile_count * TOPK; ++candidate) {
        const float value = partial_dist[offset + candidate];
        if (value < best_dist[TOPK - 1]) {
            int slot = TOPK - 1;
            while (slot > 0 && value < best_dist[slot - 1]) {
                best_dist[slot] = best_dist[slot - 1];
                best_id[slot] = best_id[slot - 1];
                --slot;
            }
            best_dist[slot] = value;
            best_id[slot] = partial_id[offset + candidate];
        }
    }
    int* output = ids + (size_t(leaf) * rows + row) * TOPK;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) output[k] = best_id[k];
}

struct Times { float distance_ms, select_ms, total_ms; };

static Times run_once(const float* x, float* dist, int* ids,
                      int rows, int leaves, int dim, bool upper_only) {
    const int tiles = (rows + TILE - 1) / TILE;
    const int tile_blocks = upper_only ? tiles * (tiles + 1) / 2 : tiles * tiles;
    cudaEvent_t start, middle, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&middle));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    distances_fp32<<<dim3(tile_blocks, leaves), dim3(TILE, TILE)>>>(
        x, dist, rows, dim, upper_only);
    CUDA_CHECK(cudaEventRecord(middle));
    select_topk<<<dim3((rows + 127) / 128, leaves), 128>>>(dist, ids, rows);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    Times t{};
    CUDA_CHECK(cudaEventElapsedTime(&t.distance_ms, start, middle));
    CUDA_CHECK(cudaEventElapsedTime(&t.select_ms, middle, end));
    CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(middle));
    CUDA_CHECK(cudaEventDestroy(end));
    return t;
}

static Times run_direct_once(const float* x, float* partial_dist, int* partial_id,
                             int* ids, int rows, int leaves, int dim, bool upper_only) {
    const int tiles = (rows + TILE - 1) / TILE;
    const int tile_blocks = upper_only ? tiles * (tiles + 1) / 2 : tiles * tiles;
    cudaEvent_t start, middle, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&middle));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    tile_topk_fp32<<<dim3(tile_blocks, leaves), dim3(TILE, TILE)>>>(
        x, partial_dist, partial_id, rows, dim, upper_only);
    CUDA_CHECK(cudaEventRecord(middle));
    merge_tile_topk<<<dim3((rows + 127) / 128, leaves), 128>>>(
        partial_dist, partial_id, ids, rows);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    Times t{};
    CUDA_CHECK(cudaEventElapsedTime(&t.distance_ms, start, middle));
    CUDA_CHECK(cudaEventElapsedTime(&t.select_ms, middle, end));
    CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(middle));
    CUDA_CHECK(cudaEventDestroy(end));
    return t;
}

static float median(std::vector<float>& values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
                             : (values[middle - 1] + values[middle]) * 0.5f;
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 128;
    const int leaves = argc > 2 ? std::atoi(argv[2]) : 64;
    const int dim = argc > 3 ? std::atoi(argv[3]) : 96;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 21;
    if (rows < TOPK + 1 || leaves < 1 || dim < 1 || iterations < 3) {
        std::fprintf(stderr, "Usage: %s [rows>=9] [leaves>=1] [dim>=1] [iterations>=3]\n", argv[0]);
        return 2;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const size_t point_count = size_t(rows) * leaves;
    const size_t input_bytes = point_count * dim * sizeof(float);
    const size_t matrix_bytes = point_count * rows * sizeof(float);
    const size_t ids_bytes = point_count * TOPK * sizeof(int);
    std::vector<float> host_x(point_count * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float& value : host_x) value = distribution(rng);
    const int tile_count = (rows + TILE - 1) / TILE;
    const size_t partial_count = point_count * tile_count * TOPK;
    float *device_x, *device_dist, *partial_dist;
    int *device_ids, *partial_id;
    CUDA_CHECK(cudaMalloc(&device_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&device_dist, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&device_ids, ids_bytes));
    CUDA_CHECK(cudaMalloc(&partial_dist, partial_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&partial_id, partial_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_x, host_x.data(), input_bytes, cudaMemcpyHostToDevice));

    const int full_tiles = tile_count * tile_count;
    const int upper_tiles = tile_count * (tile_count + 1) / 2;
    const int last_tile_rows = rows - (tile_count - 1) * TILE;
    const size_t diagonal_tile_elements = size_t(tile_count - 1) * TILE * TILE
                                        + size_t(last_tile_rows) * last_tile_rows;
    // Diagonal tiles execute their full square, including both orientations.
    const size_t upper_evaluations = (size_t(rows) * rows + diagonal_tile_elements) / 2;
    std::vector<float> full_matrix(size_t(rows) * rows), upper_matrix(full_matrix.size());
    std::vector<int> full_ids(point_count * TOPK), upper_ids(full_ids.size());
    run_once(device_x, device_dist, device_ids, rows, leaves, dim, false);
    CUDA_CHECK(cudaMemcpy(full_matrix.data(), device_dist, full_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(full_ids.data(), device_ids, ids_bytes, cudaMemcpyDeviceToHost));
    run_once(device_x, device_dist, device_ids, rows, leaves, dim, true);
    CUDA_CHECK(cudaMemcpy(upper_matrix.data(), device_dist, upper_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper_ids.data(), device_ids, ids_bytes, cudaMemcpyDeviceToHost));

    const bool matrix_equal = std::memcmp(full_matrix.data(), upper_matrix.data(),
                                          full_matrix.size() * sizeof(float)) == 0;
    const bool topk_equal = full_ids == upper_ids;
    run_direct_once(device_x, partial_dist, partial_id, device_ids, rows, leaves, dim, false);
    std::vector<int> direct_full_ids(full_ids.size()), direct_upper_ids(full_ids.size());
    CUDA_CHECK(cudaMemcpy(direct_full_ids.data(), device_ids, ids_bytes, cudaMemcpyDeviceToHost));
    run_direct_once(device_x, partial_dist, partial_id, device_ids, rows, leaves, dim, true);
    CUDA_CHECK(cudaMemcpy(direct_upper_ids.data(), device_ids, ids_bytes, cudaMemcpyDeviceToHost));
    const bool direct_topk_equal = direct_full_ids == direct_upper_ids;
    const bool direct_matches_dense = direct_upper_ids == upper_ids;
    int cpu_rank_mismatches = 0, cpu_large_regrets = 0, self_neighbors = 0;
    double max_cpu_regret = 0.0;
    for (int i = 0; i < rows; ++i) {
        std::vector<std::pair<double, int>> reference;
        std::vector<double> reference_by_id(rows);
        reference.reserve(rows - 1);
        for (int j = 0; j < rows; ++j) {
            if (i == j) continue;
            double sum = 0.0;
            for (int axis = 0; axis < dim; ++axis) {
                const double delta = double(host_x[size_t(i) * dim + axis])
                                   - double(host_x[size_t(j) * dim + axis]);
                sum += delta * delta;
            }
            reference.emplace_back(sum, j);
            reference_by_id[j] = sum;
        }
        std::sort(reference.begin(), reference.end());
        for (int k = 0; k < TOPK; ++k) {
            const int actual = upper_ids[size_t(i) * TOPK + k];
            if (actual < 0 || actual >= rows) {
                ++cpu_large_regrets;
                continue;
            }
            if (actual != reference[k].second) {
                ++cpu_rank_mismatches;
                const double regret = reference_by_id[actual] - reference[k].first;
                max_cpu_regret = std::max(max_cpu_regret, std::abs(regret));
                // FP32 accumulation can reorder neighbors whose true
                // distances are virtually tied. Both GPU schedules must
                // still agree exactly; only the FP64 audit uses tolerance.
                const double tolerance = 1e-6 * std::max(1.0, reference[k].first);
                if (std::abs(regret) > tolerance) {
                    ++cpu_large_regrets;
                    std::fprintf(stderr, "CPU reference disagreement row=%d rank=%d expected=%d actual=%d fp64_dist_delta=%.9g tolerance=%.9g\n",
                                 i, k, reference[k].second, actual, regret, tolerance);
                }
            }
            self_neighbors += actual == i;
        }
    }
    if (!matrix_equal || !topk_equal || !direct_topk_equal || !direct_matches_dense
        || cpu_large_regrets || self_neighbors) {
        std::fprintf(stderr, "FAIL matrix_equal=%d topk_equal=%d direct_topk_equal=%d direct_matches_dense=%d cpu_rank_mismatches=%d cpu_large_regrets=%d self_neighbors=%d\n",
                     matrix_equal, topk_equal, direct_topk_equal, direct_matches_dense,
                     cpu_rank_mismatches, cpu_large_regrets, self_neighbors);
        return 1;
    }

    std::vector<float> full_distance, upper_distance, full_select, upper_select, full_total, upper_total;
    for (int warmup = 0; warmup < 3; ++warmup) {
        run_once(device_x, device_dist, device_ids, rows, leaves, dim, false);
        run_once(device_x, device_dist, device_ids, rows, leaves, dim, true);
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const bool reverse = (iteration & 1) != 0;
        for (int turn = 0; turn < 2; ++turn) {
            const bool upper = (turn == 0) == reverse;
            const Times t = run_once(device_x, device_dist, device_ids, rows, leaves, dim, upper);
            (upper ? upper_distance : full_distance).push_back(t.distance_ms);
            (upper ? upper_select : full_select).push_back(t.select_ms);
            (upper ? upper_total : full_total).push_back(t.total_ms);
        }
    }
    const float fd = median(full_distance), ud = median(upper_distance);
    const float fs = median(full_select), us = median(upper_select);
    const float ft = median(full_total), ut = median(upper_total);
    std::printf("gpu=%s sm=%d%d rows=%d leaves=%d dim=%d topk=%d iterations=%d\n",
                prop.name, prop.major, prop.minor, rows, leaves, dim, TOPK, iterations);
    std::printf("tiles full=%d upper=%d reduction=%.1f%% computed_elements full=%zu upper=%zu diagonal_tile_nonself=%zu\n",
                full_tiles, upper_tiles, 100.0 * (1.0 - double(upper_tiles) / full_tiles),
                point_count * rows, size_t(leaves) * upper_evaluations,
                diagonal_tile_elements - rows);
    std::printf("check matrix_bitwise=pass topk_all_leaves=pass direct_topk_all_leaves=pass topk_cpu_first_leaf=pass self_excluded=pass cpu_fp64_rank_mismatches=%d cpu_fp64_max_rank_delta=%.9g\n",
                cpu_rank_mismatches, max_cpu_regret);
    std::printf("median_ms distance full=%.4f upper=%.4f speedup=%.3fx\n", fd, ud, fd / ud);
    std::printf("median_ms selection full=%.4f upper=%.4f\n", fs, us);
    std::printf("median_ms total full=%.4f upper=%.4f speedup=%.3fx\n", ft, ut, ft / ut);
    full_distance.clear(); upper_distance.clear(); full_select.clear(); upper_select.clear();
    full_total.clear(); upper_total.clear();
    for (int warmup = 0; warmup < 3; ++warmup) {
        run_direct_once(device_x, partial_dist, partial_id, device_ids, rows, leaves, dim, false);
        run_direct_once(device_x, partial_dist, partial_id, device_ids, rows, leaves, dim, true);
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const bool reverse = (iteration & 1) != 0;
        for (int turn = 0; turn < 2; ++turn) {
            const bool upper = (turn == 0) == reverse;
            const Times t = run_direct_once(device_x, partial_dist, partial_id,
                                            device_ids, rows, leaves, dim, upper);
            (upper ? upper_distance : full_distance).push_back(t.distance_ms);
            (upper ? upper_select : full_select).push_back(t.select_ms);
            (upper ? upper_total : full_total).push_back(t.total_ms);
        }
    }
    const float dfd = median(full_distance), dud = median(upper_distance);
    const float dfs = median(full_select), dus = median(upper_select);
    const float dft = median(full_total), dut = median(upper_total);
    std::printf("direct_topk_median_ms tile_compute_select full=%.4f upper=%.4f speedup=%.3fx\n",
                dfd, dud, dfd / dud);
    std::printf("direct_topk_median_ms merge full=%.4f upper=%.4f\n", dfs, dus);
    std::printf("direct_topk_median_ms total full=%.4f upper=%.4f speedup=%.3fx\n",
                dft, dut, dft / dut);
    CUDA_CHECK(cudaFree(device_x));
    CUDA_CHECK(cudaFree(device_dist));
    CUDA_CHECK(cudaFree(device_ids));
    CUDA_CHECK(cudaFree(partial_dist));
    CUDA_CHECK(cudaFree(partial_id));
    return 0;
}
