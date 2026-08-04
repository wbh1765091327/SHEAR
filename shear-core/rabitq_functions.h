#ifndef RABITQ_FUNCTIONS_H
#define RABITQ_FUNCTIONS_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>


extern "C" {

// Structure to hold transposed binary data
// Range computation functions
void rangeSIMD(float* q, float* c, float& vl, float &vr, int B);
void rangeCommon(float* q, float* c, float& vl, float &vr, int B);
void range(float* q, float* c, float& vl, float &vr, int B);

// Quantization functions
void quantizeSIMD(uint8_t *result, float *q, float* c, const float * u, float vl, float width, uint32_t* sum_q, int B);
void quantize(uint8_t *result, float *q, float* c, const float * u, float vl, float width, uint32_t* sum_q, int B);

// Binary operations
void transpose_bin(uint8_t *q, uint32_t* tq, int B);

// Inner product functions
extern "C" uint32_t ip_bin_bin(uint32_t * q, uint32_t * d, int B);
extern "C" uint32_t ip_768bin_bin(uint32_t * q, uint32_t * d, int B);
extern "C" uint32_t ip_128bin_bin(uint32_t * q, uint32_t * d, int B);
extern "C" uint32_t ip_byte_bin_256x(uint32_t * q, uint32_t * d, int B);
extern "C" uint32_t ip_byte_bin_128x(uint32_t * q, uint32_t * d, int B);
extern "C" uint32_t ip_byte_bin(uint32_t* q, uint32_t *d, int B, bool x_256, bool x_128);

// Fast quantization function
void fast_quantize(const float* o_prime, uint8_t* code, float& ip_norm, uint32_t max_bd);

// Dense RabitQ+ sidecar quantizer used by SHEAR's GPU-subtree WMMA path.
// It uses the RaBitQ-Library fast scalar quantizer, preserves SHEAR's
// deterministic FHT-Kac signs, and emits SHEAR's dense centered i8 layout.
double rabitq_plus_fast_t_const(size_t dim, int total_bits);

int rabitq_plus_quantize_dense_fht_kac_simd(const float* data,
                                           size_t rows,
                                           size_t dim,
                                           int total_bits,
                                           double t_const,
                                           uint64_t seed,
                                           int8_t* codes,
                                           float* deltas,
                                           float* code_sums);

size_t rabitq_split_batch_data_bytes(size_t padded_dim);
size_t rabitq_split_ex_data_bytes(size_t padded_dim, int ex_bits);

int rabitq_split_quantize_fht_kac_simd(const float* data,
                                       size_t rows,
                                       size_t dim,
                                       int total_bits,
                                       double t_const,
                                       uint64_t seed,
                                       uint8_t* batch_data,
                                       uint8_t* ex_data);

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
                                        uint64_t* boosted_pairs);

size_t graph_rabitq1_padded_dim(size_t dim);
size_t graph_rabitq1_rotator_bytes(size_t padded_dim);
size_t graph_rabitq1_bin_data_bytes(size_t padded_dim);

int graph_rabitq1_generate_rotator(size_t dim,
                                   uint64_t seed,
                                   uint8_t* output,
                                   size_t output_bytes);

const char* graph_rabitq1_last_error();

struct GraphRabitq1Estimate {
    float est_dist;
    float low_dist;
};

int graph_rabitq1_rotate_vectors(size_t dim,
                                 const uint8_t* rotator,
                                 size_t rotator_bytes,
                                 const float* input,
                                 size_t rows,
                                 float* output);

int graph_rabitq1_codec_create(size_t dim,
                               const uint8_t* rotator,
                               size_t rotator_bytes,
                               const float* rotated_centers,
                               size_t center_count,
                               void** output);
void graph_rabitq1_codec_destroy(void* codec);

int graph_rabitq1_quantize(void* codec,
                           const float* input,
                           size_t rows,
                           size_t threads,
                           uint8_t* center_ids,
                           uint8_t* bin_data,
                           size_t bin_data_bytes);

int graph_rabitq1_query_create(void* codec,
                               const float* query,
                               size_t query_dim,
                               void** output);
void graph_rabitq1_query_destroy(void* query);

int graph_rabitq1_estimate_ids(void* query,
                               const uint8_t* center_ids,
                               const uint8_t* bin_data,
                               size_t rows,
                               size_t bin_stride,
                               const uint32_t* ids,
                               size_t id_count,
                               GraphRabitq1Estimate* output);

int graph_rabitq1_reference_estimate(void* query,
                                     uint8_t center_id,
                                     const uint8_t* bin_data,
                                     size_t bin_bytes,
                                     GraphRabitq1Estimate* output);

// Other functions
void doGetXP(const char* data_file,
             size_t MAX_BD,
             float* matrix_q,
             size_t num_centers, // 中心点个数
             float* matrix_centers, //每个中心点的向量，即pivot_data
             float* XP, // output but allocation inside
             float* CP); // output but allocation inside

void store_compacted_code(uint8_t* o_raw, uint8_t* o_compact, size_t D);

// IP32_fxu4 function
float IP32_fxu4(const float* x, uint8_t* y, float c, float delta, size_t D);

// xb_qprime function
float xb_qprime(float* query, uint32_t* xb, size_t B);

// Random matrix generation function
void float_randMatrix(float* x, size_t n);

// Matrix QR decomposition function
void matrix_qr(int m, int n, float* a);

void transMatrix(size_t m, float *a);
// Template function doGetXP - definition must be in header file
// NOTE: This function is commented out because it depends on cached_ifstream class
// which is not available in this context. The function is kept here for reference
// but should be implemented in the original rabitq.cpp file or where cached_ifstream
// is properly defined.
// template <typename T>
// void doGetXP(const std::string &data_file,
//              size_t MAX_BD,
//              float* matrix_q,
//              size_t num_centers, // 中心点个数
//              float* matrix_centers, //每个中心点的向量，即pivot_data
//              float*& XP, // output but allocation inside
//              float*& CP) // output but allocation inside
// {
//     /**
//      * READ VECTOR Matrix
//      */
//     //TODO 只读一次向量 OR 但是内存又太多，如何保证10亿规模的构建？
//     uint32_t npts32, ndims32;
//     size_t npts, ndims;
//     // amount to read in one shot
//     size_t read_blk_size = 64 * 1024 * 1024;
//     // create cached reader + writer
//     cached_ifstream base_reader(data_file.c_str(), read_blk_size);
//     // metadata: npts, ndims
//     base_reader.read((char *)&npts32, sizeof(uint32_t));
//     base_reader.read((char *)&ndims32, sizeof(uint32_t));
//     npts = npts32;
//     ndims = ndims32;
// 
//     std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
//     std::vector<float> x_matrix;
//     int padding = MAX_BD - ndims;
// 
//     for (size_t i = 0; i < npts; i++)
//     {
//         base_reader.read((char *)cur_vector_T.get(), ndims * sizeof(T));
//         for (size_t d = 0; d < ndims; d++)
//             x_matrix.push_back(cur_vector_T[d]);
//         if(padding > 0) {
//             for (size_t j = 0; j < padding; j++)
//                 x_matrix.push_back(0);
//         }
//     }
// 
//     /**
//      * np.dot(X_pad, P)
//      */
//     assert(x_matrix.size() == npts * MAX_BD);
//     XP = new float[npts * MAX_BD];
//     matrix_dot(npts, MAX_BD, x_matrix.data(), matrix_q, XP);
//     print_if_verbose("Matrix XP", XP, npts, MAX_BD);
//     /**
//      * np.dot(centroids_pad, P)
//      */
//     std::vector<float> matrix_centers_pad;
//     for (int i = 0; i < num_centers; i++) {
//         size_t bd = 0;
//         for(size_t j = 0; j < ndims; j++, bd++) {
//             matrix_centers_pad.push_back(matrix_centers[i*ndims + j]);
//         }
//         if (padding > 0) {
//             for (; bd < MAX_BD; bd++) {
//                 matrix_centers_pad.push_back(0.0);
//             }
//         }
//     }
//     assert(matrix_centers_pad.size() == num_centers * MAX_BD);
//     CP = new float[num_centers * MAX_BD];
//     matrix_dot(num_centers, MAX_BD, matrix_centers_pad.data(), matrix_q, CP);
// }
} // extern "C"

#endif // RABITQ_FUNCTIONS_H
