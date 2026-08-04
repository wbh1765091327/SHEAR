//! RabitQTable struct definition

use std::arch::x86_64::__m256;
use std::fmt;
use std::path::Path;

use tracing::debug;

use super::rabitq_impl::*;
use crate::common::{AnnError, AnnResult};
use crate::utils::{METADATA_SIZE, file_util};
// Constants
const B_QUERY: usize = 4;
const EX_BITS: usize = 4;

/// 使用AVX512指令集优化的矩阵-向量乘法函数
/// 计算 C = A * B，其中A是m×n矩阵，B是n×p矩阵，C是m×p矩阵
/// 假设维度是16字节对齐的
#[inline(always)]
fn matrix_multiply_avx512(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, p: usize) {
    // 对于我们的特定用例，p=1（向量）
    if p == 1 {
        // 检测AVX512支持
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                matrix_vector_multiply_avx512_impl(a, b, c, m, n);
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                matrix_vector_multiply_avx2_impl(a, b, c, m, n);
            }
        } else if is_x86_feature_detected!("avx") {
            unsafe {
                matrix_vector_multiply_avx_impl(a, b, c, m, n);
            }
        } else {
            matrix_vector_multiply_scalar(a, b, c, m, n);
        }
    } else {
        // 通用矩阵乘法
        for (i, a_row) in a.chunks_exact(n).take(m).enumerate() {
            let c_row = &mut c[i * p..][..p];
            for (j, cell) in c_row.iter_mut().enumerate() {
                let sum: f32 = a_row
                    .iter()
                    .zip(b.iter().skip(j).step_by(p).take(n))
                    .map(|(ai, bj)| ai * bj)
                    .sum();
                *cell = sum;
            }
        }
    }
}

/// AVX512实现的矩阵-向量乘法
/// AVX512可以一次处理16个float32（512位/32位=16）
#[target_feature(enable = "avx512f")]
unsafe fn matrix_vector_multiply_avx512_impl(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
) {
    // 确保n是16的倍数，因为假设是16字节对齐的
    assert_eq!(
        n % 16,
        0,
        "Dimension n must be a multiple of 16 for AVX512 optimization"
    );
    use std::arch::x86_64::*;

    for (i, cell) in c.iter_mut().take(m).enumerate() {
        let mut sum_vec = _mm512_setzero_ps();
        let a_row_ptr = unsafe { a.as_ptr().add(i * n) };

        // 处理16个元素为一组
        for k in (0..n).step_by(16) {
            // 加载16个float32到AVX512寄存器
            let a_vec = unsafe { _mm512_loadu_ps(a_row_ptr.add(k)) };
            let b_vec = unsafe { _mm512_loadu_ps(b.as_ptr().add(k)) };

            // 乘加运算
            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
        }

        // 水平求和
        let sum = _mm512_reduce_add_ps(sum_vec);
        *cell = sum;
    }
}

/// AVX2实现的矩阵-向量乘法
/// AVX2可以一次处理8个float32（256位/32位=8）
#[target_feature(enable = "avx2")]
unsafe fn matrix_vector_multiply_avx2_impl(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
) {
    // 确保n是8的倍数，因为假设是16字节对齐的
    assert_eq!(
        n % 8,
        0,
        "Dimension n must be a multiple of 8 for AVX2 optimization"
    );
    use std::arch::x86_64::*;

    for (i, cell) in c.iter_mut().take(m).enumerate() {
        let mut sum_vec0 = _mm256_setzero_ps();
        let mut sum_vec1 = _mm256_setzero_ps();
        let a_row_ptr = unsafe { a.as_ptr().add(i * n) };

        // 处理16个元素为一组，使用两个AVX2寄存器
        for k in (0..n).step_by(16) {
            // 加载前8个float32
            unsafe {
                let a_vec0 = _mm256_load_ps(a_row_ptr.add(k));
                let b_vec0 = _mm256_load_ps(b.as_ptr().add(k));
                sum_vec0 = _mm256_fmadd_ps(a_vec0, b_vec0, sum_vec0);

                // 加载后8个float32
                let a_vec1 = _mm256_load_ps(a_row_ptr.add(k + 8));
                let b_vec1 = _mm256_load_ps(b.as_ptr().add(k + 8));
                sum_vec1 = _mm256_fmadd_ps(a_vec1, b_vec1, sum_vec1);
            }
        }

        // 水平求和
        let sum0 = unsafe { _mm256_reduce_add_ps(sum_vec0) };
        let sum1 = unsafe { _mm256_reduce_add_ps(sum_vec1) };
        *cell = sum0 + sum1;
    }
}

/// AVX实现的矩阵-向量乘法
/// AVX可以一次处理8个float32（256位/32位=8）
#[target_feature(enable = "avx")]
unsafe fn matrix_vector_multiply_avx_impl(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
    // 确保n是8的倍数，因为假设是16字节对齐的
    assert_eq!(
        n % 8,
        0,
        "Dimension n must be a multiple of 8 for AVX optimization"
    );
    use std::arch::x86_64::*;

    for (i, cell) in c.iter_mut().take(m).enumerate() {
        let mut sum_vec = _mm256_setzero_ps();
        let a_row_ptr = unsafe { a.as_ptr().add(i * n) };

        // 处理8个元素为一组
        for k in (0..n).step_by(8) {
            // 加载8个float32到AVX寄存器
            let a_vec = unsafe { _mm256_load_ps(a_row_ptr.add(k)) };
            let b_vec = unsafe { _mm256_load_ps(b.as_ptr().add(k)) };

            // 乘加运算
            sum_vec = unsafe { _mm256_fmadd_ps(a_vec, b_vec, sum_vec) };
        }

        // 水平求和
        let sum = unsafe { _mm256_reduce_add_ps(sum_vec) };
        *cell = sum;
    }
}

/// 标量实现的矩阵-向量乘法
fn matrix_vector_multiply_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize) {
    for (cell, a_row) in c.iter_mut().take(m).zip(a.chunks_exact(n)) {
        let sum: f32 = a_row.iter().zip(b.iter()).map(|(ai, bj)| ai * bj).sum();
        *cell = sum;
    }
}

/// 辅助函数：水平求和AVX2寄存器
#[inline]
unsafe fn _mm256_reduce_add_ps(v: __m256) -> f32 {
    use std::arch::x86_64::*;
    // 水平相加相邻的元素
    unsafe {
        let sum128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        _mm_cvtss_f32(sum32)
    }
}

// RabitqCache struct definition
// Based on usage in process_query_to_cent function
#[derive(Debug)]
pub struct RabitqCache {
    pub vl: f32,
    pub width: f32,
    pub query_square_dist_to_centroid: f32,
    pub sum_q: u32,
    pub quant_query: Vec<u32>,
    pub query_residual: Vec<f32>,
    pub sum_longq: f32,
}

impl fmt::Display for RabitqCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RabitqCache {{ vl: {:.6}, width: {:.6}, query_square_dist_to_centroid: {:.6}, sum_q: {}, quant_query_len: {:?}, query_residual_len: {:?}, sum_longq: {:.6} }}",
            self.vl,
            self.width,
            self.query_square_dist_to_centroid,
            self.sum_q,
            self.quant_query,
            &self.query_residual[..10],
            self.sum_longq
        )
    }
}

impl RabitqCache {
    pub fn new(
        vl: f32,
        width: f32,
        query_square_dist_to_centroid: f32,
        sum_q: u32,
        quant_query: Vec<u32>,
    ) -> Self {
        Self {
            vl,
            width,
            query_square_dist_to_centroid,
            sum_q,
            quant_query,
            query_residual: Vec::new(),
            sum_longq: 0.0,
        }
    }
}

pub struct RabitQTable {
    // c * _aligned_dim, centroid data
    // Center point's rotated vector
    centroid: Vec<f32>,

    // rabitq version
    _version: Vec<usize>,

    // original dimension of vectors
    original_dim: usize,

    // aligned dimension of vectors
    aligned_dim: usize,

    // every point pop count binaryB
    ppc_b: Vec<u32>,

    // n of data id to centroid id
    // IVF
    id_to_centroid: Vec<usize>,

    // n of floats distance to centroid (not the square distance)
    // distance_out
    dist_to_centroid: Vec<f32>,

    // n of floats, inner product between the data vector and the quantized vector
    // X0: < O^, O> inner product distance
    quantized_ip: Vec<f32>,

    // n * _aligned_dim, quantized vector of original data vector
    // quantized_vec: Vec<T>, // XB: +- 1/SQRT_B i.e. O^
    project_matrix: Vec<f32>,

    binary_xb: Vec<u32>,
    ext_rabitq_code: Option<Vec<u8>>,
    ext_bit_code_len: usize,
    xipnorm: Vec<f32>,

    u: Vec<f32>,
    _error_bound_factor: f32,
    x_256: bool,
    x_128: bool,
    sqrt_ndims: f32,
    ceil_nbits: u32,
}

impl fmt::Display for RabitQTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RabitQTable {{")?;
        writeln!(f, "  _original_dim: {}", self.original_dim)?;
        writeln!(f, "  aligned_dim: {}", self.aligned_dim)?;
        writeln!(f, "  ext_bit_code_len: {}", self.ext_bit_code_len)?;
        writeln!(f, "  centroid length: {}", self.centroid.len())?;
        writeln!(f, "  id_to_centroid length: {}", self.id_to_centroid.len())?;
        writeln!(
            f,
            "  dist_to_centroid length: {}",
            self.dist_to_centroid.len()
        )?;
        writeln!(f, "  quantized_ip length: {}", self.quantized_ip.len())?;
        writeln!(f, "  project_matrix length: {}", self.project_matrix.len())?;
        writeln!(f, "  binary_xb length: {}", self.binary_xb.len())?;
        writeln!(
            f,
            "  ext_rabitq_code length: {}",
            self.ext_rabitq_code.as_ref().map_or(0, |v| v.len())
        )?;
        writeln!(f, "  xipnorm length: {}", self.xipnorm.len())?;
        writeln!(f, "  u length: {}", self.u.len())?;
        writeln!(f, "  _ppc_b length: {}", self.ppc_b.len())?;
        writeln!(f, "  x_256: {}", self.x_256)?;
        writeln!(f, "  x_128: {}", self.x_128)?;
        writeln!(f, "  sqrt_ndims: {}", self.sqrt_ndims)?;
        writeln!(f, "  ceil_nbits: {}", self.ceil_nbits)?;
        write!(f, "}}")
    }
}

impl RabitQTable {
    // Process query method
    pub fn process_query(
        &self,
        query_vec: &[f32],
        processed_query: &mut [f32],
        scratch_cache: &mut Vec<RabitqCache>,
    ) {
        // Create aligned_query_vec and initialize with zeros
        let mut aligned_query_vec = vec![0f32; self.aligned_dim];

        // Copy query_vec to aligned_query_vec
        aligned_query_vec.copy_from_slice(query_vec);

        // Calculate inner product for each dimension using cblas
        // We want to compute: processed_query = project_matrix * aligned_query_vec
        // where:
        // - processed_query is aligned_dim x 1
        // - project_matrix is aligned_dim x aligned_dim
        // - aligned_query_vec is aligned_dim x 1

        // Using cblas_sgemm for matrix multiplication
        // C = alpha * A * B + beta * C
        // where:
        // - A is project_matrix (aligned_dim x aligned_dim)
        // - B is aligned_query_vec (aligned_dim x 1)
        // - C is processed_query (aligned_dim x 1)
        // - alpha = 1.0, beta = 0.0

        matrix_multiply_avx512(
            &self.project_matrix,
            &aligned_query_vec,
            processed_query,
            self.aligned_dim,
            self.aligned_dim,
            1,
        );

        // Process query to centroid for each centroid
        for c in 0..(self.centroid.len() / self.aligned_dim) {
            self.process_query_to_cent(processed_query, c, scratch_cache);
        }
    }

    // Process query to centroid method
    #[inline]
    pub fn process_query_to_cent(
        &self,
        processed_query: &[f32],
        target_centroid: usize,
        scratch_cache: &mut Vec<RabitqCache>,
    ) {
        // Get random_cent_vec
        let random_cent_vec = &self.centroid
            [target_centroid * self.aligned_dim..(target_centroid + 1) * self.aligned_dim];

        // Calculate range
        let (vl, vr) = range_simd(processed_query, random_cent_vec, self.aligned_dim);

        // Calculate width
        let width = (vr - vl) / ((1 << B_QUERY) - 1) as f32;

        // Quantize
        let mut byte_query = vec![0u8; self.aligned_dim];
        let sum_q = quantize_simd(
            &mut byte_query,
            processed_query,
            random_cent_vec,
            &self.u,
            vl,
            width,
            self.aligned_dim,
        );

        // Transpose binary
        let quant_query = transpose_bin(&byte_query, self.aligned_dim);

        // Calculate query_square_dist_to_centroid and sum_longq
        let mut query_residual = vec![0.0; self.aligned_dim];

        let (query_square_dist_to_centroid, sum_longq) = simd_residual_query(
            processed_query,
            random_cent_vec,
            &mut query_residual,
            self.aligned_dim,
        );

        // Create RabitqCache
        let mut cache =
            RabitqCache::new(vl, width, query_square_dist_to_centroid, sum_q, quant_query);
        cache.query_residual = query_residual;
        cache.sum_longq = sum_longq * ((1 << (EX_BITS + 1)) - 1) as f32 * 0.5;

        // Insert cache
        scratch_cache.insert(target_centroid, cache);
    }

    /// Compute distances for approximate search
    pub fn compute_est_dists(
        &self,
        ids: &[u32],
        query_scratch: &[RabitqCache],
    ) -> AnnResult<Vec<f32>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        if ids.len() == 1 {
            return Ok(vec![
                self.compute_single_dist(ids[0] as usize, query_scratch)?,
            ]);
        }

        let mut results = Vec::with_capacity(ids.len());
        // Process the rest of the elements with prefetching
        for window in ids.windows(2) {
            let id = window[0] as usize;
            let next_id = window[1] as usize;

            // Prefetch data for the element after next
            #[cfg(target_arch = "x86_64")]
            unsafe {
                // Prefetch self.id_to_centroid[next_next_id]
                use std::arch::x86_64::_mm_prefetch;
                _mm_prefetch(self.id_to_centroid.as_ptr().add(next_id) as *const i8, 2);
                // Prefetch self.binary_xb[next_next_id * self.ceil_nbits as usize..]
                _mm_prefetch(
                    self.binary_xb
                        .as_ptr()
                        .add(next_id * self.ceil_nbits as usize) as *const i8,
                    2,
                );
                // Prefetch self.ppc_b[next_next_id]
                _mm_prefetch(self.ppc_b.as_ptr().add(next_id) as *const i8, 2);
                // Prefetch self.quantized_ip[next_next_id]
                _mm_prefetch(self.quantized_ip.as_ptr().add(next_id) as *const i8, 2);
                // Prefetch self.dist_to_centroid[next_next_id]
                _mm_prefetch(self.dist_to_centroid.as_ptr().add(next_id) as *const i8, 2);
            }

            // Process current element
            results.push(self.compute_single_dist(id, query_scratch)?);
        }
        results.push(self.compute_single_dist(*ids.last().unwrap() as usize, query_scratch)?);

        Ok(results)
    }

    /// Helper function to compute distance for a single id
    #[inline(always)]
    fn compute_single_dist(&self, id: usize, query_scratch: &[RabitqCache]) -> AnnResult<f32> {
        // use id to find the target centroid id and target centroid vector
        let target_centroid = self.id_to_centroid[id];

        // Get cache for this centroid
        let rabitq_cache = &query_scratch[target_centroid];

        // error_bound is not implemented in the Rust version yet
        let _error_bound: f64 = 0.0;

        // <\bar o, q> = <\bar x, q'>
        let pos = id * self.ceil_nbits as usize;
        let t0 = self.sqrt_ndims;
        let t1 = rabitq_cache.width / t0;

        // Calculate inner product
        let ibb = ip_byte_bin(
            &rabitq_cache.quant_query,
            &self.binary_xb[pos..pos + self.ceil_nbits as usize],
            self.aligned_dim,
            self.x_256,
            self.x_128,
        );

        let query_estimated_dist = ibb as f32 * 2.0 * t1
            + self.ppc_b[id] as f32 * 2.0 * rabitq_cache.vl / t0
            - rabitq_cache.sum_q as f32 * t1
            - rabitq_cache.vl * t0;

        let tmp_dist2 = query_estimated_dist / self.quantized_ip[id] + 0.;
        let dist_to_centroid = self.dist_to_centroid[id];

        // dists_out = ||o_r - c||^2 + ||q_r -c||^2 - 2 * ||o_r - c|| * ||q_r - c|| *
        Ok(
            dist_to_centroid * dist_to_centroid + rabitq_cache.query_square_dist_to_centroid
                - 2.0 * dist_to_centroid * tmp_dist2,
        )
    }

    /// Compute exact distance for a single id
    /// This corresponds to compute_exact_dist in the C++ implementation
    pub fn compute_exact_dist(
        &self,
        id: u32,
        ext_code: &[u8],
        query_scratch: &[RabitqCache],
    ) -> AnnResult<f32> {
        let id = id as usize;
        let target_centroid = self.id_to_centroid[id];

        let rabitq_cache = &query_scratch[target_centroid];

        let mut temp_dist2 = 0f32;

        if self.xipnorm[id] != 0f32 && !ext_code.is_empty() {
            let pos: usize = id * self.ceil_nbits as usize;
            let query_estimated_dist = xb_qprime(
                &rabitq_cache.query_residual,
                &self.binary_xb[pos..pos + self.ceil_nbits as usize],
                self.aligned_dim,
            );

            let ilbb = ip32_fxu4(
                &rabitq_cache.query_residual,
                ext_code,
                // &self.ext_rabitq_code.as_ref().unwrap()[id*64..(id+1)*64],
                self.aligned_dim,
            );

            temp_dist2 =
                ((query_estimated_dist * (1 << EX_BITS) as f32) + ilbb) - rabitq_cache.sum_longq;
        }

        // dists_out = ||o_r - c||^2 + ||q_r -c||^2 - 2 * ||o_r - c|| * ||q_r - c|| * tmp_dist
        let result = self.dist_to_centroid[id] * self.dist_to_centroid[id]
            + rabitq_cache.query_square_dist_to_centroid
            - self.xipnorm[id] * temp_dist2;

        Ok(result)
    }

    pub fn aligned_dim(&self) -> usize {
        self.aligned_dim
    }

    pub fn id_to_centroid(&self) -> &[usize] {
        &self.id_to_centroid
    }

    pub fn dist_to_centroid(&self) -> &[f32] {
        &self.dist_to_centroid
    }

    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    pub fn load(rabitq_table_file: &Path) -> AnnResult<Self> {
        let (file_offset_data, nr, _nc) =
            file_util::load_bin::<usize, _>(rabitq_table_file, 0).unwrap();

        if nr < 10 {
            return Err(AnnError::log_index_error(format!(
                "Error reading rabitq file {:?}. Offsets dont contain correct metadata, # offsets = {}, but expecting more than 10",
                rabitq_table_file, nr
            )));
        }

        let version_offset = METADATA_SIZE - std::mem::size_of::<usize>(); // METADATA_SIZE - sizeof(size_t)
        let version_data =
            file_util::load_bin_only_data::<usize>(rabitq_table_file, 1, version_offset).unwrap();
        let version = version_data[0];

        if version > 1 {
            // CURRENT_VERSION is 1
            return Err(AnnError::log_index_error(format!(
                "Error reading version from rabitq file {:?}. Reading version = {}, but expecting less than or equal to 1.",
                rabitq_table_file, version
            )));
        }

        debug!("file_offset_data: {:?}", file_offset_data);

        let (centroid, nr_centroid, nc_centroid) =
            file_util::load_bin::<f32, _>(rabitq_table_file, file_offset_data[0])?;
        let _c = nr_centroid;
        let aligned_dim = nc_centroid;

        let (id_to_centroid, nr_id_to_centroid, nc_id_to_centroid) =
            file_util::load_bin::<usize, _>(rabitq_table_file, file_offset_data[1])?;
        if nc_id_to_centroid != 1 {
            return Err(AnnError::log_index_error(format!(
                "Error reading id_to_centroid from rabitq file {:?}. file_cols = {} but expecting in 1 dimension.",
                rabitq_table_file, nc_id_to_centroid
            )));
        }
        let npts = nr_id_to_centroid;

        let (dist_to_centroid, nr_dist_to_centroid, nc_dist_to_centroid) =
            file_util::load_bin::<f32, _>(rabitq_table_file, file_offset_data[2])?;
        if (nr_dist_to_centroid != npts) || (nc_dist_to_centroid != 1) {
            return Err(AnnError::log_index_error(format!(
                "Error reading dist_to_centroid from rabitq file {:?}. file_dim = {}, file_cols = {} but expecting {} entries in 1 dimension.",
                rabitq_table_file, nr_dist_to_centroid, nc_dist_to_centroid, npts
            )));
        }

        let (quantized_ip, nr_quantized_ip, nc_quantized_ip) =
            file_util::load_bin::<f32, _>(rabitq_table_file, file_offset_data[3])?;
        if (nr_quantized_ip != npts) || (nc_quantized_ip != 1) {
            return Err(AnnError::log_index_error(format!(
                "Error reading quantized_ip from rabitq file {:?}. file_dim = {}, file_cols = {} but expecting {} entries in 1 dimension.",
                rabitq_table_file, nr_quantized_ip, nc_quantized_ip, npts
            )));
        }

        let (mut project_matrix, nr_project_matrix, nc_project_matrix) =
            file_util::load_bin::<f32, _>(rabitq_table_file, file_offset_data[4])?;
        if (nr_project_matrix != aligned_dim) || (nc_project_matrix != aligned_dim) {
            return Err(AnnError::log_index_error(format!(
                "Error reading project_matrix from rabitq file {:?}. file_dim = {}, file_cols = {} but expecting {} entries in {} dimension.",
                rabitq_table_file, nr_project_matrix, nc_project_matrix, aligned_dim, aligned_dim
            )));
        }

        let (binary_xb, _nr_binary_xb, _nc_binary_xb) =
            file_util::load_bin::<u32, _>(rabitq_table_file, file_offset_data[5])?;

        let (_ppc_b, nr_ppc_b, nc_ppc_b) =
            file_util::load_bin::<u32, _>(rabitq_table_file, file_offset_data[6])?;
        if (nr_ppc_b != npts) || (nc_ppc_b != 1) {
            return Err(AnnError::log_index_error(format!(
                "Error reading _ppc_b from rabitq file {:?}. file_dim = {}, file_cols = {} but expecting {} entries in 1 dimension.",
                rabitq_table_file, nr_ppc_b, nc_ppc_b, npts
            )));
        }

        let xipnorm_offset = if file_offset_data[8] > file_offset_data[7] {
            file_offset_data[8]
        } else {
            file_offset_data[7]
        };
        let (xipnorm, nr_xipnorm, nc_xipnorm) =
            file_util::load_bin::<f32, _>(rabitq_table_file, xipnorm_offset)?;
        if nr_xipnorm != npts {
            return Err(AnnError::log_index_error(format!(
                "Error reading xipnorm from rabitq file {:?}. file_dim = {}, file_cols = {} but expecting {} entries.",
                rabitq_table_file, nr_xipnorm, nc_xipnorm, npts
            )));
        }

        let u = vec![0.0; aligned_dim];
        trans_matrix(&mut project_matrix, aligned_dim);

        Ok(RabitQTable {
            aligned_dim,
            id_to_centroid,
            dist_to_centroid,
            quantized_ip,
            project_matrix,
            binary_xb,
            ppc_b: _ppc_b,
            ext_rabitq_code: None,
            xipnorm,
            centroid,
            _version: vec![version],
            original_dim: aligned_dim,
            ext_bit_code_len: aligned_dim * 4 / 8,
            u,
            _error_bound_factor: 1.9,
            x_256: aligned_dim % 256 == 0,
            x_128: aligned_dim % 128 == 0,
            sqrt_ndims: (aligned_dim as f32).sqrt(),
            ceil_nbits: (aligned_dim as u32).div_ceil(32),
        })
    }
}

fn trans_matrix(matrix: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }
    let mut temp = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            temp[i * dim + j] = matrix[j * dim + i];
        }
    }
    matrix.copy_from_slice(&temp);
}
