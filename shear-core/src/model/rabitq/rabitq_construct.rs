// RabitQ build implementation in Rust

use std::path::Path;

use tracing::{debug, info};

use crate::common::{AnnResult, Metric};
#[cfg(feature = "ffi")]
use crate::model::rabitq::ffi;
use crate::utils::{gen_fixed_random_slice, kmeans};

// Constants
const METADATA_SIZE: usize = 4096;
const CURRENT_VERSION: usize = 1;

#[cfg(feature = "ffi")]
fn float_rand_matrix(matrix: &mut [f32], size: usize) {
    unsafe {
        ffi::float_randMatrix(matrix.as_mut_ptr(), size);
    }
}

#[cfg(not(feature = "ffi"))]
fn float_rand_matrix(matrix: &mut [f32], size: usize) {
    use rand::Rng;
    let mut rng = rand::rng();
    for item in matrix.iter_mut().take(size) {
        *item = rng.random_range(0.0..1.0);
    }
}

#[cfg(feature = "ffi")]
fn matrix_qr(rows: usize, cols: usize, matrix: &mut [f32]) {
    unsafe {
        ffi::matrix_qr(rows as i32, cols as i32, matrix.as_mut_ptr());
    }
}

#[cfg(not(feature = "ffi"))]
fn matrix_qr(rows: usize, cols: usize, matrix: &mut [f32]) {
    use ndarray::prelude::*;
    use ndarray_linalg::*;

    assert!(rows >= cols);

    // Convert the input slice to a column-major ndarray matrix, which is what LAPACK expects.
    // The input from Rust is row-major, so we read it and transpose it to get the matrix
    // as LAPACK would see it.
    let a_row_major = Array2::from_shape_vec((rows, cols), matrix.to_vec()).unwrap();
    let a_col_major = a_row_major.t().to_owned();

    // Perform QR decomposition
    let (q, _r) = a_col_major.qr().expect("QR decomposition failed");

    // The C++ code generates an m-by-n Q matrix.
    // The `qr` method in ndarray-linalg returns a full m x m Q matrix.
    // We take the first `cols` columns to match the LAPACK behavior.
    // Note that since we transposed the input, the dimensions for Q are based on the transposed
    // shape.
    let q_sliced = q.slice(s![.., 0..cols]);

    // Copy the resulting Q matrix back to the original slice.
    // The output Q from ndarray is row-major. The C++ code writes a column-major
    // result back into the buffer, which is then interpreted as row-major by Rust.
    // This is equivalent to writing the transpose of the LAPACK result.
    // Our `q_sliced` is the Q of A.T. The C++ code returns Q_lapack.
    // Rust test code reads Q_lapack as Q_lapack.T.
    // So we need to produce Q_lapack.T.
    // Q_lapack is the Q factor of A_col_major (which is A.T).
    // So q_sliced is Q_lapack.
    // We need to write it out transposed.
    let q_final = q_sliced.t();

    // Copy the data back.
    for (i, element) in q_final.iter().enumerate() {
        if i < matrix.len() {
            matrix[i] = *element;
        }
    }
}

#[cfg(not(feature = "ffi"))]
pub fn matrix_dot(m: usize, d: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use ndarray::prelude::*;

    // The data is in column-major order.
    let b_mat = ArrayView::from_shape((d, d).f(), b).unwrap();
    let a_mat = ArrayView::from_shape((d, m).f(), a).unwrap();
    let mut c_mat = ArrayViewMut::from_shape((d, m).f(), c).unwrap();

    // c = b * a
    let result = b_mat.dot(&a_mat);
    c_mat.assign(&result);
}

#[cfg(not(feature = "ffi"))]
#[allow(clippy::too_many_arguments)]
fn do_get_xp(
    data_file_to_use: &Path,
    max_bd: usize,
    matrix: &[f32],
    num_centers: usize,
    pivot_data: &[f32],
    binary_xb: &mut Vec<u32>,
    ppc_b: &mut Vec<u32>,
    x0: &mut Vec<f32>,
    cp: &mut [f32],
    ivf_out: &[usize],
    distance_out: &[f32],
    xipnorm: &mut Vec<f32>,
) {
    use std::fs::File;
    use std::io::{BufReader, Read};

    // 使用BufReader进行缓冲读取
    let file = File::open(data_file_to_use).expect("Failed to open data file");
    let mut reader = BufReader::with_capacity(64 * 1024, file); // 64KB缓冲区

    // Read metadata (npts, ndims)
    let mut metadata = [0u8; 8];
    reader
        .read_exact(&mut metadata)
        .expect("Failed to read metadata");

    let npts = u32::from_le_bytes([metadata[0], metadata[1], metadata[2], metadata[3]]) as usize;
    let ndims = u32::from_le_bytes([metadata[4], metadata[5], metadata[6], metadata[7]]) as usize;

    // 预分配输出向量
    let mut zero_norm = vec![false; npts];

    // 计算向量数据的总字节大小
    let vector_size_bytes = ndims * std::mem::size_of::<f32>();

    // 使用更大的缓冲区来批量读取
    const BATCH_SIZE: usize = 1024; // 每次处理1024个向量
    let mut batch_buffer = vec![0u8; BATCH_SIZE * vector_size_bytes];

    // 初始化量化处理相关变量
    let b = max_bd;
    let sqrt_b = (b as f64).sqrt() as f32;

    // 清空输出向量，准备接收新数据
    x0.resize(npts, 0.0);
    zero_norm.resize(npts, false);
    xipnorm.resize(npts, 0.0);

    // Pad centroids with zeros (质心数据较小，可以一次性处理)
    let padding = max_bd - ndims;
    let mut matrix_centers_pad = Vec::with_capacity(num_centers * max_bd);

    for i in 0..num_centers {
        // Copy original centroid data
        for j in 0..ndims {
            matrix_centers_pad.push(pivot_data[i * ndims + j]);
        }

        // Add padding zeros
        matrix_centers_pad.extend(std::iter::repeat_n(0.0, padding));
    }

    // Calculate CP = centroids_pad * P
    matrix_dot(num_centers, max_bd, &matrix_centers_pad, matrix, cp);

    for batch_start in (0..npts).step_by(BATCH_SIZE) {
        let batch_end = std::cmp::min(batch_start + BATCH_SIZE, npts);
        let batch_count = batch_end - batch_start;

        // 批量读取数据
        let bytes_to_read = batch_count * vector_size_bytes;
        reader
            .read_exact(&mut batch_buffer[..bytes_to_read])
            .expect("Failed to read batch data");

        // 使用rayon并行处理批量数据
        use rayon::prelude::*;
        let batch_results: Vec<_> = (0..batch_count)
            .into_par_iter()
            .map(|i| {
                let offset = i * vector_size_bytes;
                let mut local_cur_vector = vec![0.0f32; ndims];
                let mut local_padded_vector = vec![0.0f32; max_bd];
                let mut local_xp_slice = vec![0.0f32; max_bd];

                // 安全地将字节转换为f32向量
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        batch_buffer[offset..].as_ptr(),
                        local_cur_vector.as_mut_ptr() as *mut u8,
                        vector_size_bytes,
                    );
                }

                // 填充零值
                local_padded_vector[..ndims].copy_from_slice(&local_cur_vector[..ndims]);
                local_padded_vector[ndims..].fill(0.0);

                // 计算单个向量的矩阵乘积 XP[i] = padded_vector * matrix
                let global_idx = batch_start + i;
                matrix_dot(1, max_bd, &local_padded_vector, matrix, &mut local_xp_slice);

                // 立即进行量化处理
                let mut one_row: Vec<bool> = Vec::new();
                let mut xb: Vec<f32> = Vec::new();

                // 计算残差和二进制编码
                for (j, x) in local_xp_slice.iter_mut().take(b).enumerate() {
                    let re = *x - cp[ivf_out[global_idx] * max_bd + j];
                    if re > 0.0 {
                        one_row.push(true);
                        xb.push(1.0 / sqrt_b);
                    } else {
                        one_row.push(false);
                        xb.push(-1.0 / sqrt_b);
                    }
                    *x = re;
                }

                // 生成二进制编码
                let mut cnt = 0;
                let mut binary_chunk: Vec<u32> = Vec::new();
                for inner in 0..((one_row.len() as f64 / 32.0).ceil() as usize) {
                    let mut u32_val = 0u32;
                    let one = 1u32;
                    for j in 0..32 {
                        if (inner * 32 + j < b) && one_row[inner * 32 + j] {
                            u32_val |= one << j;
                            cnt += 1;
                        }
                    }
                    binary_chunk.push(u32_val);
                }

                // 计算X0值
                let mut sum_row = 0.0;
                let mut norm_row = 0.0;
                for (x, y) in local_xp_slice.iter().take(b).zip(xb.iter()) {
                    sum_row += x * y;
                    norm_row += x * x;
                }
                norm_row = norm_row.sqrt();

                let x0_val;
                let zero_norm_val;

                if norm_row == 0.0 {
                    x0_val = 0.8;
                    zero_norm_val = true;
                    for x in local_xp_slice.iter_mut().take(b) {
                        *x = 1e-10;
                    }
                } else {
                    x0_val = sum_row / norm_row;
                    zero_norm_val = false;
                    for x in local_xp_slice.iter_mut().take(b) {
                        *x /= norm_row;
                    }
                }
                let xipnorm_val = if zero_norm_val {
                    0.0
                } else {
                    norm_row * 2.0 * distance_out[global_idx]
                };

                // 返回处理结果
                (
                    global_idx,
                    binary_chunk,
                    cnt,
                    x0_val,
                    zero_norm_val,
                    xipnorm_val,
                )
            })
            .collect();

        // 合并并行处理结果到主向量
        for (global_idx, binary_chunk, cnt, x0_val, zero_norm_val, xipnorm_val) in batch_results {
            // 合并二进制编码
            binary_xb.extend(binary_chunk);
            ppc_b.push(cnt);
            x0[global_idx] = x0_val;
            zero_norm[global_idx] = zero_norm_val;
            xipnorm[global_idx] = xipnorm_val;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gen_dist_to_centroid(
    data_file_to_use: &Path,
    pivot_data: &[f32],
    pivot_num: usize,
    _compare_metric: Metric,
    distance_out: &mut Vec<f32>,
    ivf_out: &mut Vec<usize>,
    npts: &mut usize,
    ndims: &mut usize,
) -> AnnResult<()> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    // 使用BufReader进行缓冲读取
    let file = File::open(data_file_to_use)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file); // 64KB缓冲区

    // Read metadata (npts, ndims)
    let mut metadata = [0u8; 8];
    reader.read_exact(&mut metadata)?;

    let npts32 = u32::from_le_bytes([metadata[0], metadata[1], metadata[2], metadata[3]]);
    let ndims32 = u32::from_le_bytes([metadata[4], metadata[5], metadata[6], metadata[7]]);
    *npts = npts32 as usize;
    *ndims = ndims32 as usize;

    debug!("npts: {} {}", *npts, *ndims);

    // Initialize output vectors
    distance_out.resize(*npts, 0.0);
    ivf_out.resize(*npts, 0);

    // 计算向量数据的总字节大小
    let vector_size_bytes = *ndims * std::mem::size_of::<f32>();

    // 使用更大的缓冲区来批量读取
    const BATCH_SIZE: usize = 1000; // 每次处理1024个向量
    let mut batch_buffer = vec![0u8; BATCH_SIZE * vector_size_bytes];

    for batch_start in (0..*npts).step_by(BATCH_SIZE) {
        let batch_end = std::cmp::min(batch_start + BATCH_SIZE, *npts);
        let batch_count = batch_end - batch_start;

        // 批量读取数据
        let bytes_to_read = batch_count * vector_size_bytes;
        reader.read_exact(&mut batch_buffer[..bytes_to_read])?;

        // 使用rayon并行处理批量数据
        use rayon::prelude::*;
        let batch_results: Vec<_> = (0..batch_count)
            .into_par_iter()
            .map(|i| {
                let offset = i * vector_size_bytes;
                let mut cur_vector = vec![0.0f32; *ndims];

                // 安全地将字节转换为f32向量
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        batch_buffer[offset..].as_ptr(),
                        cur_vector.as_mut_ptr() as *mut u8,
                        vector_size_bytes,
                    );
                }

                let global_idx = batch_start + i;

                // 找到最近的质心
                let mut min_distance = f32::MAX;
                let mut closest_pivot = 0;

                for j in 0..pivot_num {
                    let mut distance_sq = 0.0;

                    // 手动计算L2距离平方
                    for dim in 0..*ndims {
                        let diff = cur_vector[dim] - pivot_data[j * *ndims + dim];
                        distance_sq += diff * diff;
                    }

                    // 更新最小距离和最近质心
                    if distance_sq < min_distance {
                        min_distance = distance_sq;
                        closest_pivot = j;
                    }
                }

                // 返回处理结果
                (global_idx, min_distance.sqrt(), closest_pivot)
            })
            .collect();

        // 合并并行处理结果到主向量
        for (global_idx, distance, pivot) in batch_results {
            distance_out[global_idx] = distance;
            ivf_out[global_idx] = pivot;
        }
    }

    Ok(())
}

// The gen_dist_to_centroid function implementation has been moved to another location
// to avoid duplication. This TODO comment is kept for reference.
// TODO: Implement gen_dist_to_centroid

fn save_bin<T>(file_path: &Path, data: &[T], rows: usize, cols: usize, offset: usize) -> usize
where
    T: Clone + Copy + 'static,
{
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};

    // 验证数据长度与rows * cols是否匹配
    assert_eq!(
        data.len(),
        rows * cols,
        "Data length does not match rows * cols"
    );

    #[allow(clippy::suspicious_open_options)]
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // .truncate(true)
        .open(file_path)
        .expect("Failed to open file");

    // Seek to the specified offset
    file.seek(SeekFrom::Start(offset as u64))
        .expect("Failed to seek to offset");

    // Write metadata (rows, cols)
    let rows_bytes = (rows as u32).to_le_bytes();
    let cols_bytes = (cols as u32).to_le_bytes();

    file.write_all(&rows_bytes)
        .expect("Failed to write rows metadata");
    file.write_all(&cols_bytes)
        .expect("Failed to write cols metadata");

    // Write data
    let data_bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };

    file.write_all(data_bytes).expect("Failed to write data");

    // Calculate and return the number of bytes written
    let bytes_written = 8 + data_bytes.len(); // 8 bytes for metadata (rows, cols)

    info!("Writing bin: {:?}", file_path);
    info!(
        "bin: #pts = {}, #dims = {}, size = {}B",
        rows, cols, bytes_written
    );

    bytes_written
}

fn save_bin_only_data<T>(file_path: &Path, data: &[T], len: usize, offset: usize)
where
    T: Clone + Copy + 'static,
{
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};
    use std::mem::size_of;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(file_path)
        .expect("Failed to open file");

    // Seek to the specified offset
    file.seek(SeekFrom::Start(offset as u64))
        .expect("Failed to seek to offset");

    // Write data
    let data_bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len * size_of::<T>()) };

    file.write_all(data_bytes).expect("Failed to write data");
}

/// Performs the RabitQ quantization process and saves the results to the specified file.
///
/// This function performs K-Means clustering, RabitQ XP transformation, residual calculation, and
/// ExtrabitQ quantization on the input data, and finally saves all results to `rabitq_file`
/// according to a specific file layout.
///
/// # File Layout
///
/// The generated file has the following layout:
///
/// 1. **Metadata Region (First METADATA_SIZE bytes)**
///    - Offset array `offset_byte` (10 usize values, 8 bytes each): Stores the starting positions
///      of subsequent data blocks.
///      - `offset_byte[0]`: METADATA_SIZE (size of the metadata region)
///      - `offset_byte[1]`: End position of pivot_data data block
///      - `offset_byte[2]`: End position of ivf_out data block
///      - `offset_byte[3]`: End position of distance_out data block
///      - `offset_byte[4]`: End position of x0 data block
///      - `offset_byte[5]`: End position of _matrix data block
///      - `offset_byte[6]`: End position of binary_xb data block
///      - `offset_byte[7]`: End position of ppc_b data block
///      - `offset_byte[8]`: Start position of xipnorm data block. New files may leave
///        `offset_byte[7] == offset_byte[8]` to indicate there is no 4-bit ext-code block.
///      - `offset_byte[9]`: End position of xipnorm data block
///    - Other metadata information (located at the end of METADATA_SIZE):
///      - Original dimension `d` (usize)
///      - Current version number `CURRENT_VERSION` (usize)
///
/// 2. **Data Region (After METADATA_SIZE bytes)**
///    - `pivot_data`: [num_centers][max_bd] f32 vector, storing cluster center data.
///    - `ivf_out`: [npts][1] usize vector, storing the cluster center index for each point.
///    - `distance_out`: [npts][1] f32 vector, storing the distance from each point to its cluster
///      center.
///    - `x0`: [npts][1] f32 vector, storing <O^, O> values.
///    - `_matrix`: [max_bd][max_bd] f32 vector, storing the transformation matrix P.
///    - `binary_xb`: [binary_xb.len()][1] u32 vector, storing binarized residual data.
///    - `ppc_b`: [ppc_b.len()][1] u32 vector, storing popcount for each binarized row.
///    - `xipnorm`: [npts][1] f32 vector, storing norm data for the 1-bit residual path.
///
/// # Parameters
///
/// - `rabitq_file`: Output file path for saving quantization results.
/// - `data_file_to_use`: Input data file path.
/// - `compare_metric`: Distance metric type.
/// - `num_centers`: Number of K-Means clustering centers, default is 64.
/// - `kmeans_niters`: Number of K-Means iterations, default is 25.
///
/// # Return Value
///
/// - `ANNResult<()>`: Operation result, returns Ok on success, Err on failure.
pub fn do_rabitq(
    rabitq_file: &Path,
    data_file_to_use: &Path,
    compare_metric: Metric,
    num_centers: usize,   // default 64
    kmeans_niters: usize, // default 25
) -> AnnResult<(usize, usize)> {
    // K-Means
    // instantiates train_data with random sample updates train_size
    let mut train_dim: usize = 0;
    // 如果数据太小了就用全量 20%的数据train
    let p_val: f64 = 0.20;
    let mut sampled_vectors: Vec<f32> = Vec::new();
    let _ = gen_fixed_random_slice::<f32>(
        data_file_to_use,
        p_val,
        &mut sampled_vectors,
        &mut train_dim,
    )?;

    let train_size = sampled_vectors.len() / train_dim;
    info!(
        "Training data with SIZE: {} DIM:{} samples loaded.",
        train_size, train_dim
    );
    // 每个中心点的向量是哪些
    let mut pivot_data: Vec<f32> = vec![0.0; num_centers * train_dim];
    let t0 = std::time::Instant::now();
    // 输出是，随机生成的pivot
    kmeans::selecting_pivots(
        &sampled_vectors,
        train_size,
        train_dim,
        &mut pivot_data,
        num_centers,
    );
    // 输出是closest_center 和 pivot_data
    let (_, _closest_center, _) = kmeans::run_lloyds(
        &sampled_vectors,
        train_size,
        train_dim,
        &mut pivot_data,
        num_centers,
        kmeans_niters,
    )?;
    let t1 = std::time::Instant::now();
    let elapsed_time = t1.duration_since(t0);
    info!(
        "Kmeans run success pivots with {} elapsed_time: {:?}",
        num_centers, elapsed_time
    );

    // rabitQ XP
    // Transformation matrix, size d_out * d_in
    let d = train_dim;
    let b = d.div_ceil(64) * 64;
    let max_bd = std::cmp::max(d, b);
    let mut matrix: Vec<f32> = vec![0.0; max_bd * max_bd];
    // Orthogonal
    float_rand_matrix(&mut matrix, max_bd * max_bd);
    matrix_qr(max_bd, max_bd, &mut matrix);

    let t0 = std::time::Instant::now();
    let elapsed_time = t0.duration_since(t1);
    info!(
        "MatrixQR run success, MAX_BD: {} elapsed_time: {:?}",
        max_bd, elapsed_time
    );

    // TODO 中心点个数不足的时候要Padding

    // Convert Path to C string

    // 算出来每个点到最近的中心，以及距离
    let mut distance_out: Vec<f32> = Vec::new();
    let mut ivf_out: Vec<usize> = Vec::new();
    let mut npts: usize = 0;
    let mut ndims: usize = 0; // gen_dist_to_centroid 修改它
    gen_dist_to_centroid(
        data_file_to_use,
        &pivot_data,
        num_centers,
        compare_metric,
        &mut distance_out,
        &mut ivf_out,
        &mut npts,
        &mut ndims,
    )?;

    // Call C++ function via FFI
    let mut binary_xb: Vec<u32> = Vec::new();
    let mut ppc_b: Vec<u32> = Vec::new();
    let mut x0: Vec<f32> = Vec::new();

    // GET XP
    let mut cp: Vec<f32> = vec![0.0; num_centers * max_bd];
    let mut xipnorm: Vec<f32> = vec![0.0; npts];
    #[cfg(feature = "ffi")]
    unsafe {
        let mut xp: Vec<f32> = vec![0.0; npts * max_bd];
        let mut zero_norm: Vec<bool> = Vec::new();
        let sqrt_b = (b as f64).sqrt() as f32;

        use crate::common::AnnError;
        let data_file_cstr =
            std::ffi::CString::new(data_file_to_use.to_str().unwrap()).map_err(|_| {
                AnnError::log_index_error("Failed to convert path to CString".to_string())
            })?;
        ffi::doGetXP(
            data_file_cstr.as_ptr(),
            max_bd,
            matrix.as_ptr(),
            num_centers,
            pivot_data.as_ptr(),
            xp.as_mut_ptr(),
            cp.as_mut_ptr(),
        );

        // Resize vectors to hold data for all points
        x0.resize(npts, 0.0);
        zero_norm.resize(npts, false);
        for i in 0..npts {
            let mut one_row: Vec<bool> = Vec::new();
            let mut xb: Vec<f32> = Vec::new();

            for j in 0..b {
                let re = xp[i * max_bd + j] - cp[ivf_out[i] * max_bd + j];
                if re > 0.0 {
                    one_row.push(true);
                    xb.push(1.0 / sqrt_b as f32);
                } else {
                    one_row.push(false);
                    xb.push(-1.0 / sqrt_b as f32);
                }

                xp[i * max_bd + j] = re;
            }

            let mut cnt = 0;
            for inner in 0..((one_row.len() as f64 / 32.0).ceil() as usize) {
                let mut u32_val = 0u32;
                let one = 1u32;
                for j in 0..32 {
                    if (inner * 32 + j < b) && one_row[inner * 32 + j] {
                        u32_val |= one << j;
                        cnt += 1;
                    }
                }
                binary_xb.push(u32_val);
            }
            // 计算量化成1bit后的数据，所有位置上的popcount之和
            ppc_b.push(cnt);

            let mut sum_row = 0.0;
            let mut norm_row = 0.0;
            for j in 0..b {
                sum_row += xp[i * max_bd + j] * xb[j];
                norm_row += xp[i * max_bd + j] * xp[i * max_bd + j];
            }
            // FIXME norm_now == 0
            norm_row = norm_row.sqrt();

            if norm_row == 0.0 {
                x0[i] = 0.8;
                // 这里XP不会再被用到，保守将其赋值为一个较小的值
                for j in 0..b {
                    xp[i * max_bd + j] = 1e-10;
                }
                zero_norm[i] = true;
            } else {
                x0[i] = sum_row / norm_row;
                for j in 0..b {
                    xp[i * max_bd + j] /= norm_row;
                }
                zero_norm[i] = false;
            }
        }

        // Parallel processing using rayon
        let fast_quantize_results: Vec<_> = (0..npts)
            .map(|i| {
                let xipnorm_val: f32;

                if zero_norm[i] {
                    // norm为0，表示distance_out[i]为0，所以这里直接设置xipnorm[i]为0
                    xipnorm_val = 0.0;
                    return (i, xipnorm_val);
                }
                let mut norm_sq = 0.0f32;
                for j in 0..b {
                    norm_sq += xp[i * max_bd + j] * xp[i * max_bd + j];
                }
                xipnorm_val = norm_sq.sqrt() * 2.0 * distance_out[i];
                (i, xipnorm_val)
            })
            .collect();

        // Update xipnorm with results from parallel processing
        for (i, xipnorm_val) in fast_quantize_results {
            xipnorm[i] = xipnorm_val;
        }
    }

    #[cfg(not(feature = "ffi"))]
    {
        do_get_xp(
            data_file_to_use,
            max_bd,
            &matrix,
            num_centers,
            &pivot_data,
            &mut binary_xb,
            &mut ppc_b,
            &mut x0,
            &mut cp,
            &ivf_out,
            &distance_out,
            &mut xipnorm,
        );
    }

    let t1 = std::time::Instant::now();
    let elapsed_time = t1.duration_since(t0);
    info!(
        "doGetXP run success, MAX_BD: {} elapsed_time: {:?}",
        max_bd, elapsed_time
    );

    // |O_r - c|^2 + |Q_r -c|^2 - 2 * |O_r - c| * |Q_r - C| * { <O^, q> / <O^, O>}
    // distance_out = |O_r -C|; X0 = <O^,O>;
    // 为了算 | Q_r - C | 所以需要 pivot_data；ivf_out(每个点属于哪个中心点)
    // 为了在查询计算 < O^, q> 所以需要 XB, 如果要用binary CP

    info!("METADATA_SIZE size{}", METADATA_SIZE);
    let mut offset_byte: Vec<usize> = vec![0; 10];
    offset_byte[0] = METADATA_SIZE;
    // pivot_data: [num_centers][train_dim] 128*66*4 = 33792
    offset_byte[1] =
        offset_byte[0] + save_bin::<f32>(rabitq_file, &cp, num_centers, max_bd, offset_byte[0]);
    // ivf_out: [npts][1][] 1000000*4 = 4000000
    offset_byte[2] =
        offset_byte[1] + save_bin::<usize>(rabitq_file, &ivf_out, npts, 1, offset_byte[1]);
    // distance_out: [npts][1][] 1000000*4 = 4000000
    offset_byte[3] =
        offset_byte[2] + save_bin::<f32>(rabitq_file, &distance_out, npts, 1, offset_byte[2]);
    // X0: [npts][1][] 1000000*4 = 4000000 （< O^, O> ）
    offset_byte[4] = offset_byte[3] + save_bin::<f32>(rabitq_file, &x0, npts, 1, offset_byte[3]);
    // P，matrix
    offset_byte[5] =
        offset_byte[4] + save_bin::<f32>(rabitq_file, &matrix, max_bd, max_bd, offset_byte[4]);
    // Binary
    offset_byte[6] = offset_byte[5]
        + save_bin::<u32>(rabitq_file, &binary_xb, binary_xb.len(), 1, offset_byte[5]);
    // ppc_b: [npts][1] 1000000*4 = 4000000
    offset_byte[7] =
        offset_byte[6] + save_bin::<u32>(rabitq_file, &ppc_b, ppc_b.len(), 1, offset_byte[6]);

    offset_byte[8] = offset_byte[7];
    offset_byte[9] =
        offset_byte[8] + save_bin::<f32>(rabitq_file, &xipnorm, npts, 1, offset_byte[8]);

    // Meta
    // 1. 从头部开始，依次写入offset
    save_bin::<usize>(rabitq_file, &offset_byte, offset_byte.len(), 1, 0);

    // 2. 从METADATA_SIZE的尾部开始，依次写入其它元数据信息和版本号，版本号在最后：
    // 版本1元数据包括：original dimension
    let meta_info_vec: Vec<usize> = vec![d, CURRENT_VERSION];
    let meta_offset = METADATA_SIZE - meta_info_vec.len() * std::mem::size_of::<usize>();
    save_bin_only_data::<usize>(
        rabitq_file,
        &meta_info_vec,
        meta_info_vec.len(),
        meta_offset,
    );
    Ok((npts, ndims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::rabitq::ffi;

    #[test]
    fn test_matrix_qr() {
        // 创建一个测试矩阵
        let rows = 3;
        let cols = 3;
        let mut matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // 复制原始矩阵用于验证
        let mut original_matrix = matrix.clone();

        // 调用matrix_qr函数
        matrix_qr(rows, cols, &mut original_matrix);
        unsafe { ffi::matrix_qr(rows as i32, cols as i32, matrix.as_mut_ptr()) };

        for (i, j) in original_matrix.iter().zip(matrix.iter()) {
            assert!((i - j).abs() < 1e-6, "i: {}, j: {}", i, j);
        }
    }
}
