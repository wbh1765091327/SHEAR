#include "rabitq_functions.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <cmath>
#include <queue>
#include <memory>
#include <mutex>
#include <fstream>
#include <atomic>
#include <limits>
#include <chrono>
#include <array>

#if __has_include(<rabitqlib/utils/fht_avx.hpp>)
#include <rabitqlib/utils/fht_avx.hpp>
#define SHEAR_HAS_RABITQ_FHT_AVX 1
#else
#define SHEAR_HAS_RABITQ_FHT_AVX 0
#endif

#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <rabitqlib/utils/space.hpp>

namespace {
thread_local std::string graph_rabitq1_error;

struct GraphRabitq1CodecImpl {
    size_t dim;
    size_t padded_dim;
    size_t center_count;
    size_t bin_stride;
    std::unique_ptr<rabitqlib::Rotator<float>> rotator;
    std::vector<float> centers;
    rabitqlib::quant::RabitqConfig query_config;
};

struct GraphRabitq1QueryImpl {
    const GraphRabitq1CodecImpl* codec;
    std::vector<float> rotated;
    std::unique_ptr<rabitqlib::SplitSingleQuery<float>> wrapper;
    std::vector<float> q_to_centroids;
};

uint64_t graph_rabitq1_splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

bool graph_rabitq1_valid_padded_dim(size_t value) {
    return value >= 64 && value < 4096 && value % 64 == 0;
}

float graph_rabitq1_squared_l2(const float* left, const float* right, size_t dim) {
    return rabitqlib::euclidean_sqr<float>(left, right, dim);
}

std::unique_ptr<rabitqlib::Rotator<float>> graph_rabitq1_make_rotator(
    size_t dim,
    const uint8_t* state,
    size_t state_bytes
) {
    const size_t padded = graph_rabitq1_padded_dim(dim);
    if (padded == 0 || state == nullptr || state_bytes != graph_rabitq1_rotator_bytes(padded)) {
        throw std::invalid_argument("invalid graph RaBitQ1 rotator state");
    }
    std::unique_ptr<rabitqlib::Rotator<float>> rotator(rabitqlib::choose_rotator<float>(
        dim, rabitqlib::RotatorType::FhtKacRotator, padded
    ));
    rotator->load(reinterpret_cast<const char*>(state));
    return rotator;
}

template <typename Function>
int graph_rabitq1_guard(const char* operation, Function&& function) {
    graph_rabitq1_error.clear();
    try {
        function();
        return 0;
    } catch (const std::exception& error) {
        graph_rabitq1_error = std::string(operation) + ": " + error.what();
    } catch (...) {
        graph_rabitq1_error = std::string(operation) + ": unknown C++ exception";
    }
    return 1;
}
}  // namespace

size_t graph_rabitq1_padded_dim(size_t dim) {
    if (dim < 64 || dim >= 4096 || dim > std::numeric_limits<size_t>::max() - 63) {
        return 0;
    }
    return ((dim + 63) / 64) * 64;
}

size_t graph_rabitq1_rotator_bytes(size_t padded_dim) {
    return graph_rabitq1_valid_padded_dim(padded_dim) ? padded_dim / 2 : 0;
}

size_t graph_rabitq1_bin_data_bytes(size_t padded_dim) {
    return graph_rabitq1_valid_padded_dim(padded_dim)
        ? rabitqlib::BinDataMap<float>::data_bytes(padded_dim)
        : 0;
}

int graph_rabitq1_generate_rotator(size_t dim,
                                   uint64_t seed,
                                   uint8_t* output,
                                   size_t output_bytes) {
    graph_rabitq1_error.clear();
    const size_t padded = graph_rabitq1_padded_dim(dim);
    const size_t required = graph_rabitq1_rotator_bytes(padded);
    if (required == 0 || output == nullptr || output_bytes != required) {
        graph_rabitq1_error = "invalid graph RaBitQ1 rotator output";
        return 1;
    }
    uint64_t state = seed;
    for (size_t index = 0; index < required; ++index) {
        output[index] = static_cast<uint8_t>(graph_rabitq1_splitmix64(state) >> 56);
    }
    return 0;
}

const char* graph_rabitq1_last_error() {
    return graph_rabitq1_error.c_str();
}

int graph_rabitq1_rotate_vectors(size_t dim,
                                 const uint8_t* rotator,
                                 size_t rotator_bytes,
                                 const float* input,
                                 size_t rows,
                                 float* output) {
    return graph_rabitq1_guard("rotate graph RaBitQ1 vectors", [&] {
        if (rows == 0 || input == nullptr || output == nullptr) {
            throw std::invalid_argument("invalid vector buffer");
        }
        const size_t padded = graph_rabitq1_padded_dim(dim);
        auto state = graph_rabitq1_make_rotator(dim, rotator, rotator_bytes);
        for (size_t row = 0; row < rows; ++row) {
            const float* source = input + row * dim;
            for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                if (!std::isfinite(source[coordinate])) {
                    throw std::invalid_argument("non-finite input vector");
                }
            }
            state->rotate(source, output + row * padded);
        }
    });
}

int graph_rabitq1_codec_create(size_t dim,
                               const uint8_t* rotator,
                               size_t rotator_bytes,
                               const float* rotated_centers,
                               size_t center_count,
                               void** output) {
    return graph_rabitq1_guard("create graph RaBitQ1 codec", [&] {
        if (output == nullptr) {
            throw std::invalid_argument("null codec output");
        }
        *output = nullptr;
        if (center_count == 0 || center_count > 256 || rotated_centers == nullptr) {
            throw std::invalid_argument("center_count must be in 1..=256");
        }
        const size_t padded = graph_rabitq1_padded_dim(dim);
        if (padded == 0 || center_count > std::numeric_limits<size_t>::max() / padded) {
            throw std::invalid_argument("invalid codec dimensions");
        }
        const size_t center_values = center_count * padded;
        for (size_t index = 0; index < center_values; ++index) {
            if (!std::isfinite(rotated_centers[index])) {
                throw std::invalid_argument("non-finite rotated center");
            }
        }
        auto codec = std::make_unique<GraphRabitq1CodecImpl>();
        codec->dim = dim;
        codec->padded_dim = padded;
        codec->center_count = center_count;
        codec->bin_stride = graph_rabitq1_bin_data_bytes(padded);
        codec->rotator = graph_rabitq1_make_rotator(dim, rotator, rotator_bytes);
        codec->centers.assign(rotated_centers, rotated_centers + center_values);
        codec->query_config = rabitqlib::quant::faster_config(
            padded, rabitqlib::SplitSingleQuery<float>::kNumBits
        );
        *output = codec.release();
    });
}

void graph_rabitq1_codec_destroy(void* codec) {
    delete static_cast<GraphRabitq1CodecImpl*>(codec);
}

int graph_rabitq1_quantize(void* codec_raw,
                           const float* input,
                           size_t rows,
                           size_t threads,
                           uint8_t* center_ids,
                           uint8_t* bin_data,
                           size_t bin_data_bytes) {
    return graph_rabitq1_guard("quantize graph RaBitQ1 vectors", [&] {
        auto* codec = static_cast<GraphRabitq1CodecImpl*>(codec_raw);
        if (codec == nullptr || input == nullptr || rows == 0 || threads == 0 ||
            center_ids == nullptr || bin_data == nullptr) {
            throw std::invalid_argument("invalid quantization arguments");
        }
        if (rows > std::numeric_limits<size_t>::max() / codec->bin_stride ||
            bin_data_bytes != rows * codec->bin_stride) {
            throw std::invalid_argument("invalid BinData output length");
        }
        if (threads > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("thread count is too large");
        }

        std::atomic<bool> failed(false);
        std::mutex failure_mutex;
        std::string failure_detail;
#pragma omp parallel for num_threads(static_cast<int>(threads)) schedule(static) if(rows > 1)
        for (size_t row = 0; row < rows; ++row) {
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                const float* source = input + row * codec->dim;
                for (size_t coordinate = 0; coordinate < codec->dim; ++coordinate) {
                    if (!std::isfinite(source[coordinate])) {
                        throw std::invalid_argument("non-finite input vector");
                    }
                }
                std::vector<float> rotated(codec->padded_dim);
                codec->rotator->rotate(source, rotated.data());
                size_t best_center = 0;
                float best_distance = graph_rabitq1_squared_l2(
                    rotated.data(), codec->centers.data(), codec->padded_dim
                );
                for (size_t center = 1; center < codec->center_count; ++center) {
                    const float distance = graph_rabitq1_squared_l2(
                        rotated.data(),
                        codec->centers.data() + center * codec->padded_dim,
                        codec->padded_dim
                    );
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_center = center;
                    }
                }
                center_ids[row] = static_cast<uint8_t>(best_center);
                char* output_row = reinterpret_cast<char*>(bin_data + row * codec->bin_stride);
                if (best_distance == 0.0F) {
                    std::memset(output_row, 0, codec->bin_stride);
                } else {
                    rabitqlib::quant::quantize_compact_one_bit(
                        rotated.data(),
                        codec->centers.data() + best_center * codec->padded_dim,
                        codec->padded_dim,
                        output_row,
                        rabitqlib::METRIC_L2
                    );
                }
            } catch (const std::exception& error) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(failure_mutex);
                failure_detail = error.what();
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(failure_mutex);
                failure_detail = "unknown row quantization exception";
            }
        }
        if (failed.load(std::memory_order_relaxed)) {
            throw std::runtime_error(failure_detail);
        }
    });
}

int graph_rabitq1_query_create(void* codec_raw,
                               const float* query,
                               size_t query_dim,
                               void** output) {
    return graph_rabitq1_guard("create graph RaBitQ1 query", [&] {
        auto* codec = static_cast<GraphRabitq1CodecImpl*>(codec_raw);
        if (codec == nullptr || query == nullptr || output == nullptr || query_dim != codec->dim) {
            throw std::invalid_argument("invalid query arguments");
        }
        *output = nullptr;
        for (size_t coordinate = 0; coordinate < query_dim; ++coordinate) {
            if (!std::isfinite(query[coordinate])) {
                throw std::invalid_argument("non-finite query vector");
            }
        }
        auto prepared = std::make_unique<GraphRabitq1QueryImpl>();
        prepared->codec = codec;
        prepared->rotated.resize(codec->padded_dim);
        codec->rotator->rotate(query, prepared->rotated.data());
        prepared->wrapper = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
            prepared->rotated.data(),
            codec->padded_dim,
            0,
            codec->query_config,
            rabitqlib::METRIC_L2
        );
        prepared->q_to_centroids.resize(codec->center_count);
        for (size_t center = 0; center < codec->center_count; ++center) {
            prepared->q_to_centroids[center] = std::sqrt(graph_rabitq1_squared_l2(
                prepared->rotated.data(),
                codec->centers.data() + center * codec->padded_dim,
                codec->padded_dim
            ));
        }
        *output = prepared.release();
    });
}

void graph_rabitq1_query_destroy(void* query) {
    delete static_cast<GraphRabitq1QueryImpl*>(query);
}

int graph_rabitq1_estimate_ids(void* query_raw,
                               const uint8_t* center_ids,
                               const uint8_t* bin_data,
                               size_t rows,
                               size_t bin_stride,
                               const uint32_t* ids,
                               size_t id_count,
                               GraphRabitq1Estimate* output) {
    return graph_rabitq1_guard("estimate graph RaBitQ1 IDs", [&] {
        auto* query = static_cast<GraphRabitq1QueryImpl*>(query_raw);
        if (query == nullptr || center_ids == nullptr || bin_data == nullptr || rows == 0 ||
            bin_stride != query->codec->bin_stride || ids == nullptr || id_count == 0 ||
            output == nullptr) {
            throw std::invalid_argument("invalid estimate arguments");
        }
        for (size_t index = 0; index < id_count; ++index) {
            const size_t id = ids[index];
            if (id >= rows) {
                throw std::out_of_range("estimate ID is outside sidecar rows");
            }
            const size_t center_id = center_ids[id];
            if (center_id >= query->codec->center_count) {
                throw std::out_of_range("center ID is outside codec centers");
            }
            float ip_x0_qr = 0;
            const float norm = query->q_to_centroids[center_id];
            rabitqlib::split_single_estdist(
                reinterpret_cast<const char*>(bin_data + id * bin_stride),
                *query->wrapper,
                query->codec->padded_dim,
                ip_x0_qr,
                output[index].est_dist,
                output[index].low_dist,
                norm * norm,
                norm
            );
        }
    });
}

int graph_rabitq1_reference_estimate(void* query_raw,
                                     uint8_t center_id,
                                     const uint8_t* bin_data,
                                     size_t bin_bytes,
                                     GraphRabitq1Estimate* output) {
    return graph_rabitq1_guard("reference graph RaBitQ1 estimate", [&] {
        auto* query = static_cast<GraphRabitq1QueryImpl*>(query_raw);
        if (query == nullptr || bin_data == nullptr || output == nullptr ||
            bin_bytes != query->codec->bin_stride || center_id >= query->codec->center_count) {
            throw std::invalid_argument("invalid reference estimate arguments");
        }
        float ip_x0_qr = 0;
        const float norm = query->q_to_centroids[center_id];
        rabitqlib::split_single_estdist(
            reinterpret_cast<const char*>(bin_data),
            *query->wrapper,
            query->codec->padded_dim,
            ip_x0_qr,
            output->est_dist,
            output->low_dist,
            norm * norm,
            norm
        );
    });
}

// Define FINTEGER type for LAPACK functions
typedef long long FINTEGER;

// LAPACK function declarations
extern "C" {
    void sgeqrf_(const FINTEGER* m, const FINTEGER* n, float* a, const FINTEGER* lda, float* tau, float* work, const FINTEGER* lwork, FINTEGER* info);
    void sorgqr_(const FINTEGER* m, const FINTEGER* n, const FINTEGER* k, float* a, const FINTEGER* lda, const float* tau, float* work, const FINTEGER* lwork, FINTEGER* info);
    int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

#define B_QUERY  4

static inline bool cpuHasAvx512vpopSupport()
{
    return __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vpopcntdq");
}

static bool Avx512vpopSupported = true;

inline uint32_t reverseBits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
    n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
    n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
    n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
    return n;
}

// ==============================================================
// compute the min and max value of the entries of q
// ==============================================================
void rangeSIMD(float* q, float* c, float& vl, float &vr, int B)
{
    vl = +1e20;
    vr = -1e20;

    __m256 min_m = _mm256_set1_ps(vl);
    __m256 max_m = _mm256_set1_ps(vr);
    for(int i = 0; i < B; i += 8) {
        __m256 v01 = _mm256_loadu_ps(q + i);
        __m256 v02 = _mm256_loadu_ps(c + i);
        __m256 t = _mm256_sub_ps(v01, v02);
        max_m = _mm256_max_ps(max_m, t);
        min_m = _mm256_min_ps(min_m, t);
    }
    std::vector<float> r(8);
    _mm256_storeu_ps(r.data(), max_m);
    vr = *std::max_element(r.begin(), r.end());

    _mm256_storeu_ps(r.data(), min_m);
    vl = *std::min_element(r.begin(), r.end());
}

void rangeCommon(float* q, float* c, float& vl, float &vr, int B)
{
    vl = +1e20;
    vr = -1e20;
    for(int i=0;i<B;i++){
        float tmp = (*q) - (*c);
        if(tmp < vl)vl = tmp;
        if(tmp > vr)vr = tmp;
        q ++;
        c ++;
    }
}

void range(float* q, float* c, float& vl, float &vr, int B)
{
    switch(B) {
        case 768:
            rangeSIMD(q,c,vl,vr,B);
            break;
        case 128:
            rangeCommon(q,c,vl,vr,B);
            break;
        default:
            rangeCommon(q, c, vl, vr, B);
            break;
    }
}

// ==============================================================
// quantize the query vector with uniform scalar quantization
// result:= output 这个就是得到q_u[i]
// q := rd_query
// c := centroid
// u := 全0.5
// vl:= vl
// width:= delta
// sum_q := output  q_u[i] 的sum
// ==============================================================
void quantizeSIMD(uint8_t *result, float *q, float* c, const float * u, float vl, float width, uint32_t* sum_q, int B){
   //float one_over_width = 1.0 / width;
    uint8_t *ptr_res = result; //这里uint8 但是实际只有4bit 即没有sq的q_u[i]
    // uint32_t sum = 0; //这里是qui的求和，对应公式20的第三项 (已注释未使用的变量)

    __m256 vl_256 = _mm256_set1_ps(vl);
    __m256 width_256 = _mm256_set1_ps(width);
    __m256 u_256 = _mm256_set1_ps(u[0]);
    __m256i sum_256 = _mm256_set1_epi8(0);
    const __mmask8 ff_mask = 0xff;
    for(int i=0; i<B; i+=8) {
        //对应公式18， uint8的转换是因为转换后不会超过uint8
        __m256 v01 = _mm256_loadu_ps(q + i);
        __m256 v02 = _mm256_loadu_ps(c + i);
        __m256 t = _mm256_sub_ps(v01, v02); //q-c
        t = _mm256_sub_ps(t, vl_256); // q-c -vl
        t = _mm256_div_ps(t, width_256); // (q - c - vl) / width
        t = _mm256_add_ps(t, u_256);

        __m256i ti = _mm256_cvtps_epi32(t); //float to int
        sum_256 = _mm256_add_epi32(sum_256, ti);
        _mm256_mask_cvtepi32_storeu_epi8((void*)(ptr_res+i), ff_mask, ti);
    }
    uint32_t u32_t[8];
    *sum_q = 0;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(u32_t), sum_256);
    for(int i = 0; i < 8; i++) {
        *sum_q += u32_t[i];
    }
}

void quantize(uint8_t *result, float *q, float* c, const float * u, float vl, float width, uint32_t* sum_q, int B){
    float one_over_width = 1.0 / width;
    uint8_t *ptr_res = result; //这里uint8 但是实际只有4bit 即没有sq的q_u[i]
    uint32_t sum = 0; //这里是qui的求和，对应公式20的第三项
    for(int i=0;i<B;i++) {
        //对应公式18， uint8的转换是因为转换后不会超过uint8
        (*ptr_res) = (uint8_t)((((*q) - (*c)) - vl) * one_over_width + (*u));
        sum += (*ptr_res);
        q ++;
        c ++;
        ptr_res ++;
        u ++;
    }
    *sum_q = sum;
}

//把_ndims维度，每维度B_QUERY位的数字，转成B_QUERY个_ndims位的数
void transpose_bin(uint8_t *q, uint32_t* tq, int B) {
    //32 -> m256 / uint8
    for(int i = 0; i < B; i += 32) {
        //256位，放32个8位
        __m256i v = _mm256_load_si256(reinterpret_cast<__m256i*>(q));
        //每个uint8 左移8-B_QUERY位
        v = _mm256_slli_epi32(v, (8-B_QUERY));
        for(int j=0;j<B_QUERY;j++){
            //取32个的第一位，一共32个组成uint，相当于qu^[j]
            uint32_t v1 = _mm256_movemask_epi8(v);
            // 不需要进行reverse，因为拿出来的bit就是按维度，从低到高进行排序的了
            // v1 = reverseBits(v1);
            tq[(B_QUERY - 1 - j) * (B / 32) + i/32] = v1;
            v = _mm256_slli_epi32(v, 1);
        }
        q += 32;
    }
}

// ==============================================================
// inner product between binary strings
// ==============================================================
uint32_t ip_bin_bin(uint32_t * q, uint32_t * d, int B){
    uint32_t ret = 0;
    for(int i = 0; i < (int)std::ceil(B * 1.0 / 32); i ++){
        ret += __builtin_popcountll((*d) & (*q));
        q ++;
        d ++;
    }
    return ret;
}

//768位，uint32在256一次处理8个，3个_mm256就好，
uint32_t ip_768bin_bin(uint32_t * q, uint32_t * d, int /*B*/){
    uint32_t ret = 0;

    __m256i v01 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(q));
    __m256i v02 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d));
    __m256i r0 = _mm256_and_si256(v01,v02);

    __m256i v11 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(q+8));
    __m256i v12 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d+8));
    __m256i r1 = _mm256_and_si256(v11,v12);

    __m256i v21 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(q+16));
    __m256i v22 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d+16));
    __m256i r2 = _mm256_and_si256(v21,v22);

    r0 = _mm256_popcnt_epi64(r0);
    r1 = _mm256_popcnt_epi64(r1);
    r2 = _mm256_popcnt_epi64(r2);
    __m256i sum = _mm256_add_epi64(r0, r1);
    sum = _mm256_add_epi64(sum, r2);
    uint64_t u64_t[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(u64_t), sum);
    ret = u64_t[0] + u64_t[1] + u64_t[2] + u64_t[3];
    return ret;
}

//128位，uint32在128一次处理4个，1个_mm128就好，
uint32_t ip_128bin_bin(uint32_t * q, uint32_t * d, int /*B*/) {
    uint32_t ret = 0;

    __m128i v01 = _mm_lddqu_si128(reinterpret_cast<__m128i*>(q));
    __m128i v02 = _mm_lddqu_si128(reinterpret_cast<__m128i*>(d));
    __m128i r0 = _mm_and_si128(v01,v02);

    r0 = _mm_popcnt_epi64(r0);
    uint64_t u64_t[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(u64_t), r0);
    ret += u64_t[0] + u64_t[1];
    return ret;
}

uint32_t ip_byte_bin_256x(uint32_t * q, uint32_t * d, int B) {
    uint32_t ret = 0;
    __m256i sum = _mm256_setzero_si256();
    //32*8=256位
    for (int i = 0; i < B/32; i+=8) {
        __m256i v01 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(q+i));
        __m256i v02 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(d+i));
        __m256i r0 = _mm256_and_si256(v01, v02);
        r0 = _mm256_popcnt_epi64(r0);
        sum = _mm256_add_epi64(sum, r0);
    }
    uint64_t u64_t[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(u64_t), sum);
    ret = u64_t[0] + u64_t[1] + u64_t[2] + u64_t[3];
    return ret;
}

uint32_t ip_byte_bin_128x(uint32_t * q, uint32_t * d, int B) {
    uint32_t ret = 0;
    __m128i sum = _mm_setzero_si128();
    //32*8=256位
    for (int i = 0; i < B/32; i+=4) {
        __m128i v01 = _mm_lddqu_si128(reinterpret_cast<__m128i*>(q+i));
        __m128i v02 = _mm_lddqu_si128(reinterpret_cast<__m128i*>(d+i));
        __m128i r0 = _mm_and_si128(v01,v02);
        r0 = _mm_popcnt_epi64(r0);
        sum = _mm_add_epi64(sum, r0);
    }
    uint64_t u64_t[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(u64_t), sum);
    ret = u64_t[0] + u64_t[1];
    return ret;
}

// ==============================================================
// inner product between a decomposed byte string q
// and a binary string d
// ==============================================================
uint32_t ip_byte_bin(uint32_t* q, uint32_t *d, int B, bool x_256, bool x_128) {
    uint32_t ret = 0;

    if (Avx512vpopSupported) {
        if (B == 768) {
            ret += (ip_768bin_bin(q + 0 * (B/32), d, B) << 0);
            ret += (ip_768bin_bin(q + 1 * (B/32), d, B) << 1);
            ret += (ip_768bin_bin(q + 2 * (B/32), d, B) << 2);
            ret += (ip_768bin_bin(q + 3 * (B/32), d, B) << 3);
        } else if (B == 128) {
            ret += (ip_128bin_bin(q + 0 * (B/32), d, B) << 0);
            ret += (ip_128bin_bin(q + 1 * (B/32), d, B) << 1);
            ret += (ip_128bin_bin(q + 2 * (B/32), d, B) << 2);
            ret += (ip_128bin_bin(q + 3 * (B/32), d, B) << 3);
        } else if (x_256) {
            ret += (ip_byte_bin_256x(q + 0 * (B/32), d, B) << 0);
            ret += (ip_byte_bin_256x(q + 1 * (B/32), d, B) << 1);
            ret += (ip_byte_bin_256x(q + 2 * (B/32), d, B) << 2);
            ret += (ip_byte_bin_256x(q + 3 * (B/32), d, B) << 3);
        } else if (x_128) {
            ret += (ip_byte_bin_128x(q + 0 * (B/32), d, B) << 0);
            ret += (ip_byte_bin_128x(q + 1 * (B/32), d, B) << 1);
            ret += (ip_byte_bin_128x(q + 2 * (B/32), d, B) << 2);
            ret += (ip_byte_bin_128x(q + 3 * (B/32), d, B) << 3);
        } else {
            for(int i = 0; i < B_QUERY; i++){
                //  = SUM(2^j * <XB,QU> 公式22
                ret += (ip_bin_bin(q + i * (B/32), d, B) << i);
            }
        }
    } else {
        for(int i = 0; i < B_QUERY; i++){
            //  = SUM(2^j * <XB,QU> 公式22
            ret += (ip_bin_bin(q + i * (B/32), d, B) << i);
        }
    }
    return ret;
}

// Fast quantization function
void fast_quantize(const float* o_prime, uint8_t* code, float& ip_norm, uint32_t max_bd)
{
    constexpr double eps = 1e-5;
    constexpr int n_enum = 10;
    double max_o = -1;

    for (size_t i = 0; i < max_bd; i++)
        if (std::abs(o_prime[i]) > max_o)
            max_o = std::abs(o_prime[i]);
    //(2^(B-1) -1) / 3
    double t_start = (double)(((1 << 4) - 1) / 3) / max_o;
    double t_end = (double)(((1 << 4) - 1) + n_enum) / max_o;

    int cur_o_bar[max_bd];
    double sqr_denominator = max_bd * 0.25;
    double numerator = 0;

    for (size_t i = 0; i < max_bd; ++i) {
        cur_o_bar[i] = int((double)t_start * std::abs(o_prime[i]) + eps);
        sqr_denominator += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
        numerator += (cur_o_bar[i] + 0.5) * std::abs(o_prime[i]);
    }

    std::priority_queue<
        std::pair<double, size_t>,
        std::vector<std::pair<double, size_t>>,
        std::greater<std::pair<double, size_t>>>
        next_t;

    for (size_t i = 0; i < max_bd; ++i) {
        next_t.emplace(std::make_pair((double)(cur_o_bar[i] + 1) / std::abs(o_prime[i]), i));
    }

    double max_ip = 0;
    double t = 0;

    size_t cnt = 0;
    while (next_t.empty() == false) {//
        double cur_t = next_t.top().first;
        size_t update_id = next_t.top().second;
        ++cnt;
        next_t.pop();

        cur_o_bar[update_id]++;
        int update_o_bar = cur_o_bar[update_id];
        sqr_denominator += 2 * update_o_bar;
        numerator += std::abs(o_prime[update_id]);
        //numerator = <y_cur, o'> ; sqr_denominator = |ycur|
        double cur_ip = numerator / std::sqrt(sqr_denominator);
        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < (1 << 4) - 1) {
            double t_next = (double)(update_o_bar + 1) / std::abs(o_prime[update_id]);
            if (t_next < t_end)
                next_t.emplace(std::make_pair(t_next, update_id));
        }
    }

    sqr_denominator = max_bd * 0.25;
    numerator = 0;
    int32_t o_bar[max_bd];
    for (size_t i = 0; i < max_bd; i++) {
        o_bar[i] = int((double)t * std::abs(o_prime[i]) + eps);
        if (o_bar[i] >= (1 << 4))
            o_bar[i] = (1 << 4) - 1;
        sqr_denominator += (int)o_bar[i] * o_bar[i] + o_bar[i];
        numerator += (o_bar[i] + 0.5) * std::abs(o_prime[i]);
    }
    // double norm = std::sqrt(sqr_denominator);
    // double ip = numerator / norm;
    ip_norm = 1 / numerator;  // 1/(ip*norm)
    if (!std::isfinite(ip_norm)) {
        ip_norm = 1.f;
    }

    /* Store code as uint8 */
    for (size_t i = 0; i < max_bd; ++i) {
        code[i] = static_cast<uint8_t>(o_bar[i]);
    }
}

static inline bool shear_rabitq_plus_sign(uint64_t seed, uint32_t round, size_t axis) {
    uint64_t value = seed ^ 0x005261426974512bULL ^
                     (static_cast<uint64_t>(round) * 0x9e3779b97f4a7c15ULL) ^
                     (static_cast<uint64_t>(axis) * 0xbf58476d1ce4e5b9ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return ((value ^ (value >> 31)) & 1ULL) == 0ULL;
}

static inline bool shear_run_rabitq_fht(float* data, size_t dim) {
#if SHEAR_HAS_RABITQ_FHT_AVX
    switch (dim) {
        case 64:
            helper_float_6(data);
            return true;
        case 128:
            helper_float_7(data);
            return true;
        case 256:
            helper_float_8(data);
            return true;
        case 512:
            helper_float_9(data);
            return true;
        case 1024:
            helper_float_10(data);
            return true;
        case 2048:
            helper_float_11(data);
            return true;
        default:
            break;
    }
#endif
    if (dim == 0 || ((dim & (dim - 1)) != 0)) {
        return false;
    }
    for (size_t width = 1; width < dim; width *= 2) {
        const size_t step = width * 2;
        for (size_t base = 0; base < dim; base += step) {
            for (size_t offset = 0; offset < width; ++offset) {
                const float left = data[base + offset];
                const float right = data[base + width + offset];
                data[base + offset] = left + right;
                data[base + width + offset] = left - right;
            }
        }
    }
    return true;
}

static inline float shear_sum_squares_f32(const float* data, size_t dim) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 v = _mm256_loadu_ps(data + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dim; ++i) {
        sum += data[i] * data[i];
    }
    return sum;
#else
    float sum = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        sum += data[i] * data[i];
    }
    return sum;
#endif
}

static inline void shear_apply_sign_mask(float* data, const uint32_t* sign_mask, size_t dim) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 v = _mm256_loadu_ps(data + i);
        const __m256 mask = _mm256_castsi256_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sign_mask + i))
        );
        _mm256_storeu_ps(data + i, _mm256_xor_ps(v, mask));
    }
    for (; i < dim; ++i) {
        if (sign_mask[i] != 0) {
            data[i] = -data[i];
        }
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        if (sign_mask[i] != 0) {
            data[i] = -data[i];
        }
    }
#endif
}

static inline bool shear_rotate_fht_kac_row(
    const float* input,
    float* rotated,
    size_t dim,
    const std::vector<uint32_t>& sign_masks
) {
    std::memcpy(rotated, input, dim * sizeof(float));
    const float norm_scale = 1.0F / std::sqrt(static_cast<float>(dim));
    for (uint32_t round = 0; round < 4; ++round) {
        shear_apply_sign_mask(
            rotated,
            sign_masks.data() + (static_cast<size_t>(round) * dim),
            dim
        );
        if (!shear_run_rabitq_fht(rotated, dim)) {
            return false;
        }
        for (size_t axis = 0; axis < dim; ++axis) {
            rotated[axis] *= norm_scale;
        }
    }
    return true;
}

static inline std::vector<uint32_t> shear_rabitq_sign_masks(uint64_t seed, size_t dim) {
    std::vector<uint32_t> sign_masks(4 * dim);
    for (uint32_t round = 0; round < 4; ++round) {
        uint32_t* round_masks = sign_masks.data() + (static_cast<size_t>(round) * dim);
        for (size_t axis = 0; axis < dim; ++axis) {
            round_masks[axis] = shear_rabitq_plus_sign(seed, round, axis)
                                    ? 0U
                                    : 0x80000000U;
        }
    }
    return sign_masks;
}

static inline bool shear_quantize_rabitq_row_dense_fast(
    const float* input,
    int8_t* output,
    size_t dim,
    int total_bits,
    const float* centroid,
    uint8_t* total_code,
    const rabitqlib::quant::RabitqConfig& config,
    float* delta,
    float* code_sum
) {
    if (total_bits < 1 || total_bits > 8) {
        return false;
    }
    for (size_t axis = 0; axis < dim; ++axis) {
        if (!std::isfinite(input[axis])) {
            return false;
        }
    }
    const float norm_sqr = shear_sum_squares_f32(input, dim);
    if (norm_sqr == 0.0F) {
        std::fill(output, output + dim, static_cast<int8_t>(0));
        *delta = 0.0F;
        *code_sum = 0.0F;
        return true;
    }

    float vl = 0.0F;
    rabitqlib::quant::quantize_scalar(
        input,
        centroid,
        dim,
        static_cast<size_t>(total_bits),
        total_code,
        *delta,
        vl,
        config
    );

    const int center = 1 << (total_bits - 1);
    float sum_code = 0.0F;
    for (size_t axis = 0; axis < dim; ++axis) {
        const int signed_code = static_cast<int>(total_code[axis]) - center;
        if (signed_code < std::numeric_limits<int8_t>::min() ||
            signed_code > std::numeric_limits<int8_t>::max()) {
            return false;
        }
        output[axis] = static_cast<int8_t>(signed_code);
        sum_code += static_cast<float>(signed_code);
    }
    *code_sum = sum_code;
    return true;
}

double rabitq_plus_fast_t_const(size_t dim, int total_bits) {
    if (dim == 0 || total_bits < 1 || total_bits > 8) {
        return -1.0;
    }
    if (total_bits <= 1) {
        return -1.0;
    }
    return rabitqlib::quant::faster_config(dim, static_cast<size_t>(total_bits)).t_const;
}

int rabitq_plus_quantize_dense_fht_kac_simd(const float* data,
                                           size_t rows,
                                           size_t dim,
                                           int total_bits,
                                           double t_const,
                                           uint64_t seed,
                                           int8_t* codes,
                                           float* deltas,
                                           float* code_sums) {
    if (data == nullptr || codes == nullptr || deltas == nullptr || code_sums == nullptr) {
        return 0;
    }
    if (dim == 0 || ((dim & (dim - 1)) != 0) || total_bits < 1 || total_bits > 8) {
        return 0;
    }

    std::vector<uint32_t> sign_masks = shear_rabitq_sign_masks(seed, dim);

    std::vector<float> centroid(dim, 0.0F);
    rabitqlib::quant::RabitqConfig config;
    if (total_bits > 1) {
        config.t_const = t_const > 0.0
                             ? t_const
                             : rabitqlib::quant::faster_config(dim, static_cast<size_t>(total_bits)).t_const;
    }
    std::atomic<int> ok(1);
    const long long rows_ll = static_cast<long long>(rows);

#pragma omp parallel
    {
        std::vector<float> rotated(dim);
        std::vector<uint8_t> total_code(dim, 0);
#pragma omp for schedule(static)
        for (long long row_ll = 0; row_ll < rows_ll; ++row_ll) {
            if (ok.load(std::memory_order_relaxed) == 0) {
                continue;
            }
            const size_t row = static_cast<size_t>(row_ll);
            const float* input = data + (row * dim);
            if (!shear_rotate_fht_kac_row(input, rotated.data(), dim, sign_masks)) {
                ok.store(0, std::memory_order_relaxed);
                continue;
            }
            if (!shear_quantize_rabitq_row_dense_fast(
                    rotated.data(),
                    codes + (row * dim),
                    dim,
                    total_bits,
                    centroid.data(),
                    total_code.data(),
                    config,
                    deltas + row,
                    code_sums + row
                )) {
                ok.store(0, std::memory_order_relaxed);
            }
        }
    }
    return ok.load(std::memory_order_relaxed);
}

size_t rabitq_split_batch_data_bytes(size_t padded_dim) {
    if (padded_dim == 0) {
        return 0;
    }
    return rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
}

size_t rabitq_split_ex_data_bytes(size_t padded_dim, int ex_bits) {
    if (padded_dim == 0 || ex_bits < 0 || ex_bits > 7) {
        return 0;
    }
    return rabitqlib::ExDataMap<float>::data_bytes(padded_dim, static_cast<size_t>(ex_bits));
}

int rabitq_split_quantize_fht_kac_simd(const float* data,
                                       size_t rows,
                                       size_t dim,
                                       int total_bits,
                                       double t_const,
                                       uint64_t seed,
                                       uint8_t* batch_data,
                                       uint8_t* ex_data) {
    if (data == nullptr || batch_data == nullptr) {
        return 0;
    }
    if (dim == 0 || ((dim & (dim - 1)) != 0) || total_bits < 1 || total_bits > 8) {
        return 0;
    }
    const size_t ex_bits = static_cast<size_t>(total_bits - 1);
    if (ex_bits > 0 && ex_data == nullptr) {
        return 0;
    }

    std::vector<uint32_t> sign_masks = shear_rabitq_sign_masks(seed, dim);
    std::vector<float> centroid(dim, 0.0F);
    rabitqlib::quant::RabitqConfig config;
    if (total_bits > 1) {
        config.t_const = t_const > 0.0
                             ? t_const
                             : rabitqlib::quant::faster_config(dim, static_cast<size_t>(total_bits)).t_const;
    }

    const size_t batch_bytes = rabitqlib::BatchDataMap<float>::data_bytes(dim);
    const size_t ex_row_bytes = rabitqlib::ExDataMap<float>::data_bytes(dim, ex_bits);
    const size_t batch_count = (rows + rabitqlib::fastscan::kBatchSize - 1) /
                               rabitqlib::fastscan::kBatchSize;
    std::atomic<int> ok(1);

#pragma omp parallel
    {
        std::vector<float> rotated(rabitqlib::fastscan::kBatchSize * dim);
#pragma omp for schedule(static)
        for (long long batch_ll = 0; batch_ll < static_cast<long long>(batch_count); ++batch_ll) {
            if (ok.load(std::memory_order_relaxed) == 0) {
                continue;
            }
            const size_t batch = static_cast<size_t>(batch_ll);
            const size_t row_start = batch * rabitqlib::fastscan::kBatchSize;
            const size_t row_count =
                std::min(rabitqlib::fastscan::kBatchSize, rows - row_start);
            for (size_t row = 0; row < row_count; ++row) {
                const float* input = data + ((row_start + row) * dim);
                float* output = rotated.data() + (row * dim);
                if (!shear_rotate_fht_kac_row(input, output, dim, sign_masks)) {
                    ok.store(0, std::memory_order_relaxed);
                    break;
                }
            }
            if (ok.load(std::memory_order_relaxed) == 0) {
                continue;
            }
            uint8_t* batch_out = batch_data + (batch * batch_bytes);
            uint8_t* ex_out = ex_bits > 0 ? ex_data + (row_start * ex_row_bytes) : nullptr;
            rabitqlib::quant::quantize_split_batch(
                rotated.data(),
                centroid.data(),
                row_count,
                dim,
                ex_bits,
                reinterpret_cast<char*>(batch_out),
                reinterpret_cast<char*>(ex_out),
                rabitqlib::METRIC_L2,
                config
            );
        }
    }

    return ok.load(std::memory_order_relaxed);
}

struct ShearRabitqTopCandidate {
    float distance;
    uint16_t leader;
    float ip_x0_qr;
};

static inline bool shear_candidate_better(float left_dist, uint16_t left_leader,
                                            float right_dist, uint16_t right_leader) {
    if (std::isfinite(left_dist) != std::isfinite(right_dist)) {
        return std::isfinite(left_dist);
    }
    if (left_dist < right_dist) {
        return true;
    }
    if (left_dist > right_dist) {
        return false;
    }
    return left_leader < right_leader;
}

static inline void shear_insert_top_candidate(ShearRabitqTopCandidate* top,
                                                size_t cap,
                                                float distance,
                                                uint16_t leader,
                                                float ip_x0_qr) {
    size_t worst = 0;
    for (size_t i = 1; i < cap; ++i) {
        if (shear_candidate_better(top[worst].distance, top[worst].leader,
                                     top[i].distance, top[i].leader)) {
            worst = i;
        }
    }
    if (!shear_candidate_better(distance, leader, top[worst].distance, top[worst].leader)) {
        return;
    }
    top[worst] = ShearRabitqTopCandidate{distance, leader, ip_x0_qr};
}

static inline void shear_sort_top_candidates(ShearRabitqTopCandidate* top, size_t len) {
    std::sort(top, top + len, [](const auto& left, const auto& right) {
        return shear_candidate_better(left.distance, left.leader, right.distance, right.leader);
    });
}

int rabitq_split_d0_assign_fht_kac_simd(const uint8_t* batch_data,
                                        const uint8_t* ex_data,
                                        size_t rows,
                                        size_t dim,
                                        int total_bits,
                                        const float* leaders,
                                        size_t leader_count,
                                        uint64_t seed,
                                        size_t fanout,
                                        size_t survivors,
                                        uint16_t* assignments,
                                        double* coarse_seconds,
                                        double* boost_seconds,
                                        uint64_t* boosted_pairs) {
    if (batch_data == nullptr || leaders == nullptr || assignments == nullptr) {
        return 0;
    }
    if (dim == 0 || ((dim & (dim - 1)) != 0) || total_bits < 1 || total_bits > 8 ||
        leader_count == 0 || fanout == 0 || survivors < fanout || survivors > leader_count ||
        leader_count > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
        return 0;
    }
    const size_t ex_bits = static_cast<size_t>(total_bits - 1);
    if (ex_bits > 0 && ex_data == nullptr) {
        return 0;
    }

    std::vector<uint32_t> sign_masks = shear_rabitq_sign_masks(seed, dim);
    std::vector<float> rotated_leaders(leader_count * dim);
    for (size_t leader = 0; leader < leader_count; ++leader) {
        if (!shear_rotate_fht_kac_row(
                leaders + (leader * dim),
                rotated_leaders.data() + (leader * dim),
                dim,
                sign_masks
            )) {
            return 0;
        }
    }

    std::vector<std::unique_ptr<rabitqlib::SplitBatchQuery<float>>> queries;
    queries.reserve(leader_count);
    for (size_t leader = 0; leader < leader_count; ++leader) {
        const float* rotated = rotated_leaders.data() + (leader * dim);
        auto query = std::make_unique<rabitqlib::SplitBatchQuery<float>>(
            rotated,
            dim,
            ex_bits,
            rabitqlib::METRIC_L2,
            true
        );
        query->set_g_add(std::sqrt(shear_sum_squares_f32(rotated, dim)));
        queries.emplace_back(std::move(query));
    }

    float (*ip_func)(const float*, const uint8_t*, size_t) =
        rabitqlib::select_excode_ipfunc(ex_bits);
    const size_t batch_bytes = rabitqlib::BatchDataMap<float>::data_bytes(dim);
    const size_t ex_row_bytes = rabitqlib::ExDataMap<float>::data_bytes(dim, ex_bits);
    const size_t batch_count = (rows + rabitqlib::fastscan::kBatchSize - 1) /
                               rabitqlib::fastscan::kBatchSize;
    const auto coarse_start = std::chrono::steady_clock::now();
    std::atomic<uint64_t> boosted(0);
    std::atomic<uint64_t> boost_nanos(0);

#pragma omp parallel
    {
        std::array<float, rabitqlib::fastscan::kBatchSize> est_distance;
        std::array<float, rabitqlib::fastscan::kBatchSize> low_distance;
        std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr;
        std::vector<ShearRabitqTopCandidate> coarse_top(
            rabitqlib::fastscan::kBatchSize * survivors
        );
        std::vector<ShearRabitqTopCandidate> final_top(
            rabitqlib::fastscan::kBatchSize * fanout
        );

#pragma omp for schedule(static)
        for (long long batch_ll = 0; batch_ll < static_cast<long long>(batch_count); ++batch_ll) {
            const size_t batch = static_cast<size_t>(batch_ll);
            const size_t row_start = batch * rabitqlib::fastscan::kBatchSize;
            const size_t row_count =
                std::min(rabitqlib::fastscan::kBatchSize, rows - row_start);
            for (size_t i = 0; i < row_count * survivors; ++i) {
                coarse_top[i] = ShearRabitqTopCandidate{
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<uint16_t>::max(),
                    0.0F,
                };
            }

            const char* batch_ptr =
                reinterpret_cast<const char*>(batch_data + (batch * batch_bytes));
            for (size_t leader = 0; leader < leader_count; ++leader) {
                rabitqlib::split_batch_estdist(
                    batch_ptr,
                    *queries[leader],
                    dim,
                    est_distance.data(),
                    low_distance.data(),
                    ip_x0_qr.data(),
                    true
                );
                const uint16_t leader_u16 = static_cast<uint16_t>(leader);
                for (size_t row = 0; row < row_count; ++row) {
                    shear_insert_top_candidate(
                        coarse_top.data() + (row * survivors),
                        survivors,
                        est_distance[row],
                        leader_u16,
                        ip_x0_qr[row]
                    );
                }
            }

            for (size_t row = 0; row < row_count; ++row) {
                for (size_t i = 0; i < fanout; ++i) {
                    final_top[(row * fanout) + i] = ShearRabitqTopCandidate{
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<uint16_t>::max(),
                        0.0F,
                    };
                }
                ShearRabitqTopCandidate* row_coarse =
                    coarse_top.data() + (row * survivors);
                for (size_t cand = 0; cand < survivors; ++cand) {
                    const uint16_t leader = row_coarse[cand].leader;
                    if (leader == std::numeric_limits<uint16_t>::max()) {
                        continue;
                    }
                    float dist = row_coarse[cand].distance;
                    if (ex_bits > 0) {
                        const auto boost_start = std::chrono::steady_clock::now();
                        const char* ex_ptr = reinterpret_cast<const char*>(
                            ex_data + ((row_start + row) * ex_row_bytes)
                        );
                        dist = rabitqlib::split_distance_boosting(
                            ex_ptr,
                            ip_func,
                            *queries[leader],
                            dim,
                            ex_bits,
                            row_coarse[cand].ip_x0_qr
                        );
                        const auto boost_end = std::chrono::steady_clock::now();
                        boost_nanos.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    boost_end - boost_start
                                ).count()
                            ),
                            std::memory_order_relaxed
                        );
                        boosted.fetch_add(1, std::memory_order_relaxed);
                    }
                    shear_insert_top_candidate(
                        final_top.data() + (row * fanout),
                        fanout,
                        dist,
                        leader,
                        0.0F
                    );
                }
                ShearRabitqTopCandidate* row_final = final_top.data() + (row * fanout);
                shear_sort_top_candidates(row_final, fanout);
                uint16_t* row_out = assignments + ((row_start + row) * fanout);
                for (size_t i = 0; i < fanout; ++i) {
                    row_out[i] = row_final[i].leader;
                }
            }
        }
    }

    const auto coarse_end = std::chrono::steady_clock::now();
    const double total_seconds =
        std::chrono::duration<double>(coarse_end - coarse_start).count();
    const double boost_s = static_cast<double>(boost_nanos.load(std::memory_order_relaxed)) / 1.0e9;
    if (coarse_seconds != nullptr) {
        *coarse_seconds = total_seconds;
    }
    if (boost_seconds != nullptr) {
        *boost_seconds = boost_s;
    }
    if (boosted_pairs != nullptr) {
        *boosted_pairs = boosted.load(std::memory_order_relaxed);
    }
    return 1;
}

// Store compacted code function
void store_compacted_code(uint8_t* o_raw, uint8_t* o_compact, size_t D) {
    for (size_t j = 0; j < D; j += 32) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 16));
        vec_16_to_31 = _mm_slli_epi16(vec_16_to_31, 4);

        __m128i compact = _mm_or_si128(vec_00_to_15, vec_16_to_31);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        o_raw += 32;
        o_compact += 16;
    }
}

// IP32_fxu4 function
float IP32_fxu4(const float* query, uint8_t* y, float /*c*/, float /*delta*/, size_t D) {
    __m128i mask = _mm_set1_epi8(0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 32) {
        __m128i a8 = _mm_loadu_si128(reinterpret_cast<__m128i*>(&y[i / 2]));
        __m128i b8 = a8;
        __m512 x1 = _mm512_loadu_ps(&query[i]);
        __m512 x2 = _mm512_loadu_ps(&query[i + 16]);

        // get lower(0 to 15) and upper(16 to 31) 4 bits
        a8 = _mm_and_si128(a8, mask);
        b8 = _mm_and_si128(_mm_srli_epi16(b8, 4), mask);

        __m512 af = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a8));
        sum = _mm512_fmadd_ps(af, x1, sum);
        __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8));
        sum = _mm512_fmadd_ps(bf, x2, sum);
    }
    return _mm512_reduce_add_ps(sum);
}

// xb_qprime function
float xb_qprime(float* query, uint32_t* xb, size_t B) 
{ 
    size_t D = std::ceil(B / 32);
    uint8_t mask = 0b11111111;
    size_t pos = 0;
    //uint8_t* x8b = reinterpret_cast<uint8_t*>(xb);
    __m256 sum = _mm256_setzero_ps();
    for(size_t i = 0,i8 = 0; i < D; i++, i8+=4) {
         uint32_t uxb = xb[i];
         //32个bit 和 32个float32，256 只能搞8个
         //TODO 这里能和IP32_fxu4合并，因为query也是拿32个，然后一起合并计算
         uint8_t b0_7 = uxb & mask;
         uint8_t b8_15 = (uxb >> 8) & mask;
         uint8_t b16_23 = (uxb >> 16) & mask;
         uint8_t b24_31 = (uxb >> 24) & mask;
         //std::cout << "uint32: " << unsigned(b0_7) << " " << unsigned(b8_15) << " " << unsigned(b16_23) << " " << unsigned(b24_31) << "\n";
         //std::cout << "uint08: " << unsigned(x8b[i8]) << " " << unsigned(x8b[i8+1]) << " " << unsigned(x8b[i8+2]) << " " << unsigned(x8b[i8+3])  << "\n";

         __m256 q0_7 = _mm256_loadu_ps(&query[pos]);
         __m256 q8_15= _mm256_loadu_ps(&query[pos+8]);
         __m256 q16_23= _mm256_loadu_ps(&query[pos+16]);
         __m256 q24_31= _mm256_loadu_ps(&query[pos+24]);

         sum = _mm256_mask_add_ps(sum, b0_7, q0_7, sum);
         sum = _mm256_mask_add_ps(sum, b8_15, q8_15, sum);
         sum = _mm256_mask_add_ps(sum, b16_23, q16_23, sum);
         sum = _mm256_mask_add_ps(sum, b24_31, q24_31, sum);
         pos+=32;
    }
    std::vector<float> r(8);
    _mm256_storeu_ps(r.data(), sum);
    float ss = 0;
    for(size_t i = 0; i < r.size(); i++) {
        ss+=r[i];
    }
    return ss;
}

// 随机生成n-n的矩阵，输出到x
void float_randMatrix(float* x, size_t n) {
    assert(x != nullptr);
    std::random_device rd; // Will be used to obtain a seed for the random number
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0, 1);

    for(size_t i = 0; i < n; i ++) {
        float rnd_val = distribution(generator);
        x[i] = rnd_val;
    }
}

// a矩阵，m*n行，做QR分解，输出到a
void matrix_qr(int m, int n, float* a) {
    assert(m >= n);
    FINTEGER mi = m, ni = n, ki = mi < ni ? mi : ni;
    std::vector<float> tau(ki);
    FINTEGER lwork = -1, info;
    float work_size;

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), &work_size, &lwork, &info);
    lwork = size_t(work_size);
    std::vector<float> work(lwork);

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), work.data(), &lwork, &info);

    sorgqr_(&mi, &ni, &ki, a, &mi, tau.data(), work.data(), &lwork, &info);
}

void matrix_dot(size_t m, size_t d, const float* a, float* b, float* c) {
    FINTEGER mi = m, di = d;
//    std::cout <<"mi: " << mi << " di: " << di << "\n";
    float one = 1, zero = 0;
    sgemm_("N", //NO OP A
           "N", //NO OP B
           &di, // rows  of the  matrix op( A )
           &mi, // columns of the matrix op( B )
           &di, // columns of the matrix op( A )
           &one, // alpha = 1
           b,
           &di,
           a,
           &di,
           &zero, // alpha = 0
           c,
           &di);
}

void doGetXP(const char* data_file,
             size_t MAX_BD,
             float* matrix_q,
             size_t num_centers, // 中心点个数
             float* matrix_centers, //每个中心点的向量，即pivot_data
             float* XP, // output but allocation inside
             float* CP) // output but allocation inside
{
/**
 * READ VECTOR Matrix - 优化版本，避免OOM
 */
    uint32_t npts32, ndims32;
    size_t npts, ndims;
    
    // 打开文件读取元数据
    std::ifstream base_reader(data_file, std::ios::binary);
    if (!base_reader) {
        throw std::runtime_error("Could not open file " + std::string(data_file));
    }
    
    // 读取元数据: npts, ndims
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&ndims32, sizeof(uint32_t));
    npts = npts32;
    ndims = ndims32;
    
    int padding = MAX_BD - ndims;
    
    // 计算合适的块大小 (例如: 1M个向量为一个块)
    const size_t BLOCK_SIZE = 1000000;
    size_t num_blocks = (npts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    
    // 为当前块分配内存
    std::vector<float> block_data(BLOCK_SIZE * MAX_BD);
    std::unique_ptr<float[]> cur_vector = std::make_unique<float[]>(ndims);
    
    // 处理每个数据块
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        size_t start_idx = block_idx * BLOCK_SIZE;
        size_t end_idx = std::min(start_idx + BLOCK_SIZE, npts);
        size_t block_actual_size = end_idx - start_idx;
        
        // 读取当前块的数据
        for (size_t i = 0; i < block_actual_size; i++) {
            base_reader.read((char *)cur_vector.get(), ndims * sizeof(float));
            
            // 复制原始数据并添加padding
            for (size_t d = 0; d < ndims; d++) {
                block_data[i * MAX_BD + d] = cur_vector[d];
            }
            // 添加padding zeros
            for (size_t p = 0; p < padding; p++) {
                block_data[i * MAX_BD + ndims + p] = 0.0f;
            }
        }
        
        // 计算当前块的 XP = X_padded_block * Q
        matrix_dot(block_actual_size, MAX_BD, block_data.data(), matrix_q, 
                  XP + start_idx * MAX_BD);
        
    }
    
    // 处理中心点数据 (数据量较小，可以一次性处理)
    std::vector<float> matrix_centers_pad;
    for (size_t i = 0; i < num_centers; i++) {
        size_t bd = 0;
        for(size_t j = 0; j < ndims; j++, bd++) {
            matrix_centers_pad.push_back(matrix_centers[i * ndims + j]);
        }
        if (padding > 0) {
            for (; bd < MAX_BD; bd++) {
                matrix_centers_pad.push_back(0.0);
            }
        }
    }
    
    assert(matrix_centers_pad.size() == num_centers * MAX_BD);
    matrix_dot(num_centers, MAX_BD, matrix_centers_pad.data(), matrix_q, CP);
    
}
