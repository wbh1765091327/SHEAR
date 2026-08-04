use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "pipnn_cuda")]
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys as cublas_sys};
#[cfg(feature = "pipnn_cuda")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "pipnn_cuda")]
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use ndarray::Array2;
use ndarray_linalg::QR;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::point_store::PointStore;
use crate::common::{AnnError, AnnResult};

pub(crate) const TURBOQUANT_MSE4_BITS: usize = 4;
pub(crate) const TURBOQUANT_MSE4_LEVELS: usize = 16;
pub(crate) const TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES: usize = 256;
pub(crate) const TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES: usize = 4;

const TURBOQUANT_MSE4_SIDECAR_MAGIC: &[u8; 8] = b"PGTQ4\0\0\0";
const TURBOQUANT_MSE4_SIDECAR_VERSION: u32 = 2;
const TURBOQUANT_MSE4_DENSE_QR_ROTATOR_LABEL: &str = "dense-qr";
const TURBOQUANT_MSE4_ROTATOR_FIELD_BYTES: usize = 16;
const TURBOQUANT_MSE4_CREATE_CHUNK_ROWS: usize = 16 * 1024;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS: [f32; TURBOQUANT_MSE4_LEVELS] = [
    -2.7235756,
    -2.0604305,
    -1.6096783,
    -1.2484536,
    -0.9357067,
    -0.6516434,
    -0.3848085,
    -0.12730813,
    0.12730813,
    0.3848085,
    0.6516434,
    0.9357067,
    1.2484536,
    1.6096783,
    2.0604305,
    2.7235756,
];

#[cfg(feature = "pipnn_cuda")]
const TURBOQUANT_MSE4_CUDA_BUILDER_SRC: &str = r#"
extern "C" __global__ void tq4_normalize_transpose_kernel(
    const float* __restrict__ raw,
    float* __restrict__ normalized_col_major,
    float* __restrict__ norms,
    unsigned long long rows,
    unsigned long long dim)
{
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if ((unsigned long long)row >= rows) {
        return;
    }
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned long long axis = tid; axis < dim; axis += blockDim.x) {
        float value = raw[((unsigned long long)row) * dim + axis];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    float norm = sqrtf(partial[0]);
    if (tid == 0) {
        norms[row] = norm;
    }
    float inv_norm = (norm > 0.0f) ? (1.0f / (norm + 1.0e-8f)) : 0.0f;
    for (unsigned long long axis = tid; axis < dim; axis += blockDim.x) {
        normalized_col_major[axis * rows + row] = raw[((unsigned long long)row) * dim + axis] * inv_norm;
    }
}

static __device__ __forceinline__ unsigned char tq4_quantize(float value, const float* thresholds) {
    unsigned char code = 0;
    #pragma unroll
    for (int idx = 0; idx < 15; ++idx) {
        code += (value > thresholds[idx]) ? 1 : 0;
    }
    return code;
}

extern "C" __global__ void tq4_pack_rotated_kernel(
    const float* __restrict__ rotated_col_major,
    const float* __restrict__ thresholds,
    signed char* __restrict__ codes,
    unsigned long long rows,
    unsigned long long dim,
    unsigned long long code_bytes_per_row)
{
    unsigned int row = blockIdx.x;
    unsigned long long pair = ((unsigned long long)blockIdx.y) * blockDim.x + threadIdx.x;
    if ((unsigned long long)row >= rows || pair >= code_bytes_per_row) {
        return;
    }
    unsigned long long axis0 = pair * 2ULL;
    unsigned char code0 = tq4_quantize(rotated_col_major[axis0 * rows + row], thresholds);
    unsigned char code1 = 0;
    if (axis0 + 1ULL < dim) {
        code1 = tq4_quantize(rotated_col_major[(axis0 + 1ULL) * rows + row], thresholds);
    }
    codes[((unsigned long long)row) * code_bytes_per_row + pair] =
        (signed char)((code0 << 4) | code1);
}
"#;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TurboQuantMse4Matrix {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) code_bytes_per_row: usize,
    pub(crate) codes: Vec<i8>,
    pub(crate) norms: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TurboQuantMse4SidecarHeader {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) code_bytes_per_row: usize,
    pub(crate) bits: usize,
    pub(crate) seed: u64,
    pub(crate) source_fbin_bytes: u64,
    pub(crate) source_rows: usize,
    pub(crate) source_dim: usize,
    pub(crate) source_fingerprint: u64,
    pub(crate) codes_bytes: usize,
    pub(crate) meta_rows_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TurboQuantMse4GatheredInput {
    pub(crate) global_ids: Vec<u32>,
    pub(crate) codes: Vec<i8>,
    pub(crate) norms: Vec<f32>,
    pub(crate) dim: usize,
    pub(crate) code_bytes_per_row: usize,
    pub(crate) gather_wall: Duration,
    pub(crate) resident_source: bool,
    pub(crate) resident_rows: usize,
    pub(crate) file_rows: usize,
    pub(crate) resident_gather_wall: Duration,
    pub(crate) file_gather_wall: Duration,
}

#[derive(Debug)]
pub(crate) struct TurboQuantMse4Sidecar {
    prefix: PathBuf,
    header: TurboQuantMse4SidecarHeader,
    storage: TurboQuantMse4SidecarStorage,
}

#[derive(Debug)]
enum TurboQuantMse4SidecarStorage {
    File {
        codes_file: File,
        meta_file: File,
    },
    PartialResident {
        codes_file: File,
        meta_file: File,
        resident_rows: usize,
        codes: Vec<i8>,
        norms: Vec<f32>,
        preload_wall: Duration,
    },
    Resident {
        codes: Vec<i8>,
        norms: Vec<f32>,
        preload_wall: Duration,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TurboQuantMse4SidecarCreateReport {
    pub(crate) header: TurboQuantMse4SidecarHeader,
    pub(crate) total_wall: Duration,
    pub(crate) setup_wall: Duration,
    pub(crate) chunk_alloc_wall: Duration,
    pub(crate) raw_read_wall: Duration,
    pub(crate) source_hash_wall: Duration,
    pub(crate) quantize_wall: Duration,
    pub(crate) code_output_wall: Duration,
    pub(crate) meta_output_wall: Duration,
    pub(crate) flush_wall: Duration,
    pub(crate) header_write_wall: Duration,
    pub(crate) write_wall: Duration,
}

#[derive(Debug)]
pub(crate) struct TurboQuantMse4SidecarCreateArtifact {
    pub(crate) report: TurboQuantMse4SidecarCreateReport,
    pub(crate) runtime_sidecar: Option<TurboQuantMse4Sidecar>,
}

impl TurboQuantMse4GatheredInput {
    pub(crate) fn approx_l2(&self, left: usize, right: usize) -> f32 {
        turboquant_mse4_approx_l2(
            &self.codes[left * self.code_bytes_per_row..(left + 1) * self.code_bytes_per_row],
            self.norms[left],
            &self.codes[right * self.code_bytes_per_row..(right + 1) * self.code_bytes_per_row],
            self.norms[right],
            self.dim,
        )
    }
}

impl TurboQuantMse4Sidecar {
    pub(crate) fn open_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        seed: u64,
    ) -> AnnResult<Self> {
        let codes_path = turboquant_mse4_sidecar_codes_path(prefix);
        let meta_path = turboquant_mse4_sidecar_meta_path(prefix);
        let mut meta_file = File::open(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open TurboQuant-MSE4 sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
        let header = read_turboquant_mse4_sidecar_header(&mut meta_file)?;
        header.validate_for_dataset(rows, dim, bits, seed)?;
        let codes_file = File::open(&codes_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open TurboQuant-MSE4 sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        let codes_len = codes_file
            .metadata()
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to stat TurboQuant-MSE4 sidecar codes '{}': {err}",
                    codes_path.display()
                ))
            })?
            .len() as usize;
        if codes_len != header.codes_bytes {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar codes size mismatch: file={} header={}",
                codes_len, header.codes_bytes
            )));
        }
        Ok(Self {
            prefix: prefix.to_path_buf(),
            header,
            storage: TurboQuantMse4SidecarStorage::File {
                codes_file,
                meta_file,
            },
        })
    }

    pub(crate) fn preload_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        seed: u64,
    ) -> AnnResult<Self> {
        let preload_start = Instant::now();
        let sidecar = Self::open_validated(prefix, rows, dim, bits, seed)?;
        let header = sidecar.header.clone();
        let mut codes = vec![0_i8; header.codes_bytes];
        let mut norms = vec![0.0_f32; header.rows];
        let TurboQuantMse4SidecarStorage::File {
            codes_file,
            meta_file,
        } = &sidecar.storage
        else {
            unreachable!("newly opened TurboQuant-MSE4 sidecar is always file-backed")
        };
        let codes_bytes =
            unsafe { std::slice::from_raw_parts_mut(codes.as_mut_ptr().cast::<u8>(), codes.len()) };
        codes_file.read_exact_at(codes_bytes, 0).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload TurboQuant-MSE4 sidecar codes '{}': {err}",
                turboquant_mse4_sidecar_codes_path(prefix).display()
            ))
        })?;
        advise_sidecar_dontneed(codes_file, 0, codes_bytes.len());
        let mut meta_bytes = vec![0_u8; header.meta_rows_bytes];
        meta_file
            .read_exact_at(&mut meta_bytes, TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES as u64)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload TurboQuant-MSE4 sidecar metadata '{}': {err}",
                    turboquant_mse4_sidecar_meta_path(prefix).display()
                ))
            })?;
        advise_sidecar_dontneed(
            meta_file,
            TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES as u64,
            meta_bytes.len(),
        );
        decode_norm_rows(&meta_bytes, &mut norms)?;
        Ok(Self {
            prefix: sidecar.prefix,
            header: sidecar.header,
            storage: TurboQuantMse4SidecarStorage::Resident {
                codes,
                norms,
                preload_wall: preload_start.elapsed(),
            },
        })
    }

    pub(crate) fn preload_prefix_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        seed: u64,
        resident_rows: usize,
    ) -> AnnResult<Self> {
        let resident_rows = resident_rows.min(rows);
        if resident_rows == 0 {
            return Self::open_validated(prefix, rows, dim, bits, seed);
        }
        if resident_rows >= rows {
            return Self::preload_validated(prefix, rows, dim, bits, seed);
        }
        let preload_start = Instant::now();
        let sidecar = Self::open_validated(prefix, rows, dim, bits, seed)?;
        let header = sidecar.header.clone();
        let mut codes = vec![0_i8; resident_rows.saturating_mul(header.code_bytes_per_row)];
        let mut norms = vec![0.0_f32; resident_rows];
        let TurboQuantMse4SidecarStorage::File {
            codes_file,
            meta_file,
        } = &sidecar.storage
        else {
            unreachable!("newly opened TurboQuant-MSE4 sidecar is always file-backed")
        };
        let codes_bytes =
            unsafe { std::slice::from_raw_parts_mut(codes.as_mut_ptr().cast::<u8>(), codes.len()) };
        codes_file.read_exact_at(codes_bytes, 0).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload TurboQuant-MSE4 sidecar resident prefix codes '{}': {err}",
                turboquant_mse4_sidecar_codes_path(prefix).display()
            ))
        })?;
        advise_sidecar_dontneed(codes_file, 0, codes_bytes.len());
        let mut meta_bytes =
            vec![0_u8; resident_rows.saturating_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES)];
        meta_file
            .read_exact_at(&mut meta_bytes, TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES as u64)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload TurboQuant-MSE4 sidecar resident prefix metadata '{}': {err}",
                    turboquant_mse4_sidecar_meta_path(prefix).display()
                ))
            })?;
        advise_sidecar_dontneed(
            meta_file,
            TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES as u64,
            meta_bytes.len(),
        );
        decode_norm_rows(&meta_bytes, &mut norms)?;
        let TurboQuantMse4SidecarStorage::File {
            codes_file,
            meta_file,
        } = sidecar.storage
        else {
            unreachable!("newly opened TurboQuant-MSE4 sidecar is always file-backed")
        };
        Ok(Self {
            prefix: sidecar.prefix,
            header: sidecar.header,
            storage: TurboQuantMse4SidecarStorage::PartialResident {
                codes_file,
                meta_file,
                resident_rows,
                codes,
                norms,
                preload_wall: preload_start.elapsed(),
            },
        })
    }

    pub(crate) fn header(&self) -> &TurboQuantMse4SidecarHeader {
        &self.header
    }

    pub(crate) fn prefix(&self) -> &Path {
        &self.prefix
    }

    pub(crate) fn is_resident(&self) -> bool {
        matches!(
            self.storage,
            TurboQuantMse4SidecarStorage::PartialResident { .. }
                | TurboQuantMse4SidecarStorage::Resident { .. }
        )
    }

    pub(crate) fn resident_rows(&self) -> usize {
        match &self.storage {
            TurboQuantMse4SidecarStorage::File { .. } => 0,
            TurboQuantMse4SidecarStorage::PartialResident { resident_rows, .. } => *resident_rows,
            TurboQuantMse4SidecarStorage::Resident { .. } => self.header.rows,
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.resident_rows().saturating_mul(
            self.header
                .code_bytes_per_row
                .saturating_add(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES),
        )
    }

    pub(crate) fn preload_wall(&self) -> Duration {
        match &self.storage {
            TurboQuantMse4SidecarStorage::File { .. } => Duration::ZERO,
            TurboQuantMse4SidecarStorage::PartialResident { preload_wall, .. } => *preload_wall,
            TurboQuantMse4SidecarStorage::Resident { preload_wall, .. } => *preload_wall,
        }
    }

    pub(crate) fn gather(&self, global_ids: &[u32]) -> AnnResult<TurboQuantMse4GatheredInput> {
        let gather_start = Instant::now();
        let rows = global_ids.len();
        let mut codes = vec![0_i8; rows.saturating_mul(self.header.code_bytes_per_row)];
        let mut norms = vec![0.0_f32; rows];
        let mut resident_rows = 0usize;
        let mut file_rows = 0usize;
        let mut run_start = 0usize;
        while run_start < rows {
            let first_id = global_ids[run_start] as usize;
            if first_id >= self.header.rows {
                return Err(AnnError::log_index_error(format!(
                    "TurboQuant-MSE4 sidecar gather id {} out of bounds for {} rows",
                    first_id, self.header.rows
                )));
            }
            let mut run_end = run_start + 1;
            while run_end < rows && global_ids[run_end] == global_ids[run_end - 1].saturating_add(1)
            {
                if global_ids[run_end] as usize >= self.header.rows {
                    return Err(AnnError::log_index_error(format!(
                        "TurboQuant-MSE4 sidecar gather id {} out of bounds for {} rows",
                        global_ids[run_end], self.header.rows
                    )));
                }
                run_end += 1;
            }
            self.read_code_run(first_id, run_start, run_end, &mut codes)?;
            self.read_meta_run(first_id, run_start, run_end, &mut norms)?;
            let (resident, file) = self.source_rows_for_run(first_id, run_end - run_start);
            resident_rows = resident_rows.saturating_add(resident);
            file_rows = file_rows.saturating_add(file);
            run_start = run_end;
        }
        let gather_wall = gather_start.elapsed();
        let (resident_gather_wall, file_gather_wall) =
            split_gather_wall(gather_wall, resident_rows, file_rows);
        Ok(TurboQuantMse4GatheredInput {
            global_ids: global_ids.to_vec(),
            codes,
            norms,
            dim: self.header.dim,
            code_bytes_per_row: self.header.code_bytes_per_row,
            gather_wall,
            resident_source: resident_rows > 0 && file_rows == 0,
            resident_rows,
            file_rows,
            resident_gather_wall,
            file_gather_wall,
        })
    }

    fn source_rows_for_run(&self, first_id: usize, run_len: usize) -> (usize, usize) {
        match &self.storage {
            TurboQuantMse4SidecarStorage::File { .. } => (0, run_len),
            TurboQuantMse4SidecarStorage::Resident { .. } => (run_len, 0),
            TurboQuantMse4SidecarStorage::PartialResident { resident_rows, .. } => {
                let run_end_id = first_id.saturating_add(run_len);
                let resident_end = run_end_id.min(*resident_rows);
                let resident = resident_end.saturating_sub(first_id);
                (resident, run_len.saturating_sub(resident))
            }
        }
    }

    fn read_code_run(
        &self,
        first_id: usize,
        run_start: usize,
        run_end: usize,
        out: &mut [i8],
    ) -> AnnResult<()> {
        let run_len = run_end - run_start;
        let row_bytes = self.header.code_bytes_per_row;
        let src_offset = first_id.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error("TurboQuant-MSE4 sidecar code offset overflows".to_string())
        })?;
        let dst_start = run_start.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error("TurboQuant-MSE4 sidecar output offset overflows".to_string())
        })?;
        let byte_len = run_len.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error(
                "TurboQuant-MSE4 sidecar code run length overflows".to_string(),
            )
        })?;
        let dst = &mut out[dst_start..dst_start + byte_len];
        match &self.storage {
            TurboQuantMse4SidecarStorage::File { codes_file, .. } => {
                let dst_bytes = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len())
                };
                codes_file
                    .read_exact_at(dst_bytes, src_offset as u64)
                    .map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read TurboQuant-MSE4 sidecar codes at offset {src_offset}: {err}"
                        ))
                    })
            }
            TurboQuantMse4SidecarStorage::Resident { codes, .. } => {
                dst.copy_from_slice(&codes[src_offset..src_offset + byte_len]);
                Ok(())
            }
            TurboQuantMse4SidecarStorage::PartialResident {
                codes_file,
                resident_rows,
                codes,
                ..
            } => {
                let resident_len = resident_rows.saturating_sub(first_id).min(run_len);
                if resident_len > 0 {
                    let resident_bytes = resident_len * row_bytes;
                    dst[..resident_bytes]
                        .copy_from_slice(&codes[src_offset..src_offset + resident_bytes]);
                }
                if resident_len < run_len {
                    let file_first_id = first_id + resident_len;
                    let file_dst_start = resident_len * row_bytes;
                    let file_offset = file_first_id.checked_mul(row_bytes).ok_or_else(|| {
                        AnnError::log_index_error(
                            "TurboQuant-MSE4 sidecar partial code offset overflows".to_string(),
                        )
                    })?;
                    let dst_bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            dst[file_dst_start..].as_mut_ptr().cast::<u8>(),
                            dst.len() - file_dst_start,
                        )
                    };
                    codes_file.read_exact_at(dst_bytes, file_offset as u64).map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read TurboQuant-MSE4 sidecar suffix codes at offset {file_offset}: {err}"
                        ))
                    })?;
                }
                Ok(())
            }
        }
    }

    fn read_meta_run(
        &self,
        first_id: usize,
        run_start: usize,
        run_end: usize,
        norms: &mut [f32],
    ) -> AnnResult<()> {
        let run_len = run_end - run_start;
        let src_offset = TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES
            .checked_add(first_id.saturating_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES))
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "TurboQuant-MSE4 sidecar meta offset overflows".to_string(),
                )
            })?;
        let byte_len = run_len
            .checked_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "TurboQuant-MSE4 sidecar meta run length overflows".to_string(),
                )
            })?;
        match &self.storage {
            TurboQuantMse4SidecarStorage::File { meta_file, .. } => {
                let mut buf = vec![0_u8; byte_len];
                meta_file
                    .read_exact_at(&mut buf, src_offset as u64)
                    .map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read TurboQuant-MSE4 sidecar metadata at offset {src_offset}: {err}"
                        ))
                    })?;
                decode_norm_rows_into_offset(&buf, norms, run_start)?;
            }
            TurboQuantMse4SidecarStorage::Resident {
                norms: all_norms, ..
            } => {
                norms[run_start..run_end].copy_from_slice(&all_norms[first_id..first_id + run_len]);
            }
            TurboQuantMse4SidecarStorage::PartialResident {
                meta_file,
                resident_rows,
                norms: all_norms,
                ..
            } => {
                let resident_len = resident_rows.saturating_sub(first_id).min(run_len);
                if resident_len > 0 {
                    let dst_end = run_start + resident_len;
                    norms[run_start..dst_end]
                        .copy_from_slice(&all_norms[first_id..first_id + resident_len]);
                }
                if resident_len < run_len {
                    let file_first_id = first_id + resident_len;
                    let file_run_start = run_start + resident_len;
                    let file_run_len = run_len - resident_len;
                    let file_src_offset = TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES
                        .checked_add(
                            file_first_id.saturating_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES),
                        )
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "TurboQuant-MSE4 sidecar partial meta offset overflows".to_string(),
                            )
                        })?;
                    let file_byte_len = file_run_len
                        .checked_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES)
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "TurboQuant-MSE4 sidecar partial meta run length overflows"
                                    .to_string(),
                            )
                        })?;
                    let mut buf = vec![0_u8; file_byte_len];
                    meta_file
                        .read_exact_at(&mut buf, file_src_offset as u64)
                        .map_err(|err| {
                            AnnError::log_index_error(format!(
                                "Failed to read TurboQuant-MSE4 sidecar suffix metadata at offset {file_src_offset}: {err}"
                            ))
                        })?;
                    decode_norm_rows_into_offset(&buf, norms, file_run_start)?;
                }
            }
        }
        Ok(())
    }
}

impl TurboQuantMse4SidecarHeader {
    fn validate_for_dataset(
        &self,
        rows: usize,
        dim: usize,
        bits: usize,
        seed: u64,
    ) -> AnnResult<()> {
        validate_turboquant_mse4_bits(bits)?;
        if self.rows != rows || self.dim != dim {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar shape mismatch: header rows={} dim={} expected rows={} dim={}",
                self.rows, self.dim, rows, dim
            )));
        }
        if self.bits != bits {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar bits mismatch: header={} expected={bits}",
                self.bits
            )));
        }
        if self.seed != seed {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar seed mismatch: header={} expected={seed}",
                self.seed
            )));
        }
        let expected_row_bytes = turboquant_mse4_code_bytes_per_row(dim)?;
        if self.code_bytes_per_row != expected_row_bytes {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar row bytes mismatch: header={} expected={expected_row_bytes}",
                self.code_bytes_per_row
            )));
        }
        let expected_codes = rows.saturating_mul(expected_row_bytes);
        if self.codes_bytes != expected_codes {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar codes bytes mismatch: header={} expected={expected_codes}",
                self.codes_bytes
            )));
        }
        let expected_meta = rows.saturating_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES);
        if self.meta_rows_bytes != expected_meta {
            return Err(AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar meta bytes mismatch: header={} expected={expected_meta}",
                self.meta_rows_bytes
            )));
        }
        Ok(())
    }
}

pub(crate) fn validate_turboquant_mse4_bits(bits: usize) -> AnnResult<()> {
    if bits != TURBOQUANT_MSE4_BITS {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 v1 supports only 4-bit scalar quantization, got {bits}"
        )));
    }
    Ok(())
}

pub(crate) fn turboquant_mse4_code_bytes_per_row(dim: usize) -> AnnResult<usize> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 quantization requires dim > 0".to_string(),
        ));
    }
    Ok(dim.div_ceil(2))
}

pub(crate) fn turboquant_mse4_centroids(dim: usize) -> AnnResult<[f32; TURBOQUANT_MSE4_LEVELS]> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 codebook requires dim > 0".to_string(),
        ));
    }
    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut out = [0.0_f32; TURBOQUANT_MSE4_LEVELS];
    for (dst, &value) in out
        .iter_mut()
        .zip(TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS.iter())
    {
        *dst = value * scale;
    }
    Ok(out)
}

pub(crate) fn turboquant_mse4_thresholds(
    dim: usize,
) -> AnnResult<[f32; TURBOQUANT_MSE4_LEVELS - 1]> {
    let centroids = turboquant_mse4_centroids(dim)?;
    let mut thresholds = [0.0_f32; TURBOQUANT_MSE4_LEVELS - 1];
    for idx in 0..thresholds.len() {
        thresholds[idx] = 0.5 * (centroids[idx] + centroids[idx + 1]);
    }
    Ok(thresholds)
}

pub(crate) fn turboquant_mse4_quantize_value(value: f32, thresholds: &[f32]) -> AnnResult<u8> {
    if !value.is_finite() {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 cannot quantize non-finite value {value}"
        )));
    }
    let mut code = 0u8;
    while (code as usize) < thresholds.len() && value > thresholds[code as usize] {
        code += 1;
    }
    Ok(code)
}

#[inline]
pub(crate) fn turboquant_mse4_pack_pair(first: u8, second: u8) -> AnnResult<i8> {
    if first >= TURBOQUANT_MSE4_LEVELS as u8 || second >= TURBOQUANT_MSE4_LEVELS as u8 {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 code out of range: first={first} second={second}"
        )));
    }
    Ok(((first << 4) | second) as i8)
}

#[inline]
pub(crate) fn turboquant_mse4_unpack_code(packed: i8, axis_in_pair: usize) -> u8 {
    let byte = packed as u8;
    if axis_in_pair == 0 {
        (byte >> 4) & 0x0f
    } else {
        byte & 0x0f
    }
}

pub(crate) fn turboquant_mse4_dense_qr_rotation(dim: usize, seed: u64) -> AnnResult<Vec<f32>> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 dense-QR rotation requires dim > 0".to_string(),
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gaussian = Vec::with_capacity(dim.saturating_mul(dim));
    while gaussian.len() < dim.saturating_mul(dim) {
        let u1 = rng.random::<f32>().clamp(f32::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f32>();
        let radius = (-2.0_f32 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        gaussian.push(radius * theta.cos());
        if gaussian.len() < dim.saturating_mul(dim) {
            gaussian.push(radius * theta.sin());
        }
    }
    let matrix = Array2::from_shape_vec((dim, dim), gaussian).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to form TurboQuant-MSE4 Gaussian rotation matrix: {err}"
        ))
    })?;
    let (mut q, r) = matrix.qr().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to compute TurboQuant-MSE4 dense-QR rotation: {err}"
        ))
    })?;
    for col in 0..dim {
        let sign = if r[(col, col)] < 0.0 { -1.0 } else { 1.0 };
        for row in 0..dim {
            q[(row, col)] *= sign;
        }
    }
    Ok(q.iter().copied().collect())
}

pub(crate) fn quantize_turboquant_mse4_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    bits: usize,
    seed: u64,
) -> AnnResult<TurboQuantMse4Matrix> {
    let rotation = turboquant_mse4_dense_qr_rotation(dim, seed)?;
    quantize_turboquant_mse4_matrix_with_rotation(points_row_major, rows, dim, bits, &rotation)
}

fn quantize_turboquant_mse4_matrix_with_rotation(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    bits: usize,
    rotation_row_major: &[f32],
) -> AnnResult<TurboQuantMse4Matrix> {
    validate_turboquant_mse4_bits(bits)?;
    if points_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 matrix shape mismatch: rows={rows} dim={dim} values={}",
            points_row_major.len()
        )));
    }
    if rotation_row_major.len() != dim.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 dense-QR rotation shape mismatch: dim={dim} values={}",
            rotation_row_major.len()
        )));
    }
    let code_bytes_per_row = turboquant_mse4_code_bytes_per_row(dim)?;
    let thresholds = turboquant_mse4_thresholds(dim)?;
    let mut codes = vec![0_i8; rows.saturating_mul(code_bytes_per_row)];
    let mut norms = vec![0.0_f32; rows];
    let mut normalized = vec![0.0_f32; dim];
    let mut rotated = vec![0.0_f32; dim];

    for row in 0..rows {
        let input = &points_row_major[row * dim..(row + 1) * dim];
        let mut norm_sq = 0.0_f32;
        for &value in input {
            if !value.is_finite() {
                return Err(AnnError::log_index_error(format!(
                    "TurboQuant-MSE4 cannot quantize non-finite value {value}"
                )));
            }
            norm_sq += value * value;
        }
        let norm = norm_sq.sqrt();
        norms[row] = norm;
        let out = &mut codes[row * code_bytes_per_row..(row + 1) * code_bytes_per_row];
        out.fill(0);
        if norm == 0.0 {
            continue;
        }
        let inv_norm = 1.0_f32 / (norm + 1.0e-8);
        for (dst, &value) in normalized.iter_mut().zip(input.iter()) {
            *dst = value * inv_norm;
        }
        for axis in 0..dim {
            let rotation_row = &rotation_row_major[axis * dim..(axis + 1) * dim];
            let mut value = 0.0_f32;
            for coord in 0..dim {
                value += normalized[coord] * rotation_row[coord];
            }
            rotated[axis] = value;
        }
        for pair in 0..code_bytes_per_row {
            let axis0 = pair * 2;
            let code0 = turboquant_mse4_quantize_value(rotated[axis0], &thresholds)?;
            let code1 = if axis0 + 1 < dim {
                turboquant_mse4_quantize_value(rotated[axis0 + 1], &thresholds)?
            } else {
                0
            };
            out[pair] = turboquant_mse4_pack_pair(code0, code1)?;
        }
    }

    Ok(TurboQuantMse4Matrix {
        rows,
        dim,
        code_bytes_per_row,
        codes,
        norms,
    })
}

pub(crate) fn turboquant_mse4_approx_l2(
    left_code: &[i8],
    left_norm: f32,
    right_code: &[i8],
    right_norm: f32,
    dim: usize,
) -> f32 {
    turboquant_mse4_approx_l2_low_dim(left_code, left_norm, right_code, right_norm, dim, dim)
}

pub(crate) fn turboquant_mse4_approx_l2_low_dim(
    left_code: &[i8],
    left_norm: f32,
    right_code: &[i8],
    right_norm: f32,
    dim: usize,
    low_dim: usize,
) -> f32 {
    let mut dot = 0.0_f32;
    let inv_dim = 1.0_f32 / dim.max(1) as f32;
    for axis in 0..low_dim.min(dim) {
        let packed_idx = axis / 2;
        let axis_in_pair = axis & 1;
        let lhs = turboquant_mse4_unpack_code(left_code[packed_idx], axis_in_pair) as usize;
        let rhs = turboquant_mse4_unpack_code(right_code[packed_idx], axis_in_pair) as usize;
        dot += TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS[lhs]
            * TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS[rhs]
            * inv_dim;
    }
    let dist = left_norm * left_norm + right_norm * right_norm - 2.0 * left_norm * right_norm * dot;
    if dist <= 0.0 { 0.0 } else { dist }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TurboQuantMse4CascadeShadowReport {
    pub(crate) shadow_rows_sampled: usize,
    pub(crate) low_dim: usize,
    pub(crate) survivor_cap: usize,
    pub(crate) no_rerank_overlap_pairs: usize,
    pub(crate) overlap_pairs_at_k: usize,
    pub(crate) survivor_overlap_pairs: usize,
    pub(crate) audit_pairs_at_k: usize,
    pub(crate) survivor_audit_pairs: usize,
    pub(crate) mismatch_rows: usize,
    pub(crate) low_dim_pairs: usize,
    pub(crate) estimated_full_pairs_saved: usize,
}

impl TurboQuantMse4CascadeShadowReport {
    pub(crate) fn no_rerank_overlap_at_k_x1000(&self) -> usize {
        ratio_x1000(self.no_rerank_overlap_pairs, self.audit_pairs_at_k)
    }

    pub(crate) fn overlap_at_k_x1000(&self) -> usize {
        ratio_x1000(self.overlap_pairs_at_k, self.audit_pairs_at_k)
    }

    pub(crate) fn overlap_at_survivor_cap_x1000(&self) -> usize {
        ratio_x1000(self.survivor_overlap_pairs, self.survivor_audit_pairs)
    }
}

pub(crate) fn turboquant_mse4_cascade_shadow_report(
    codes: &[i8],
    norms: &[f32],
    dim: usize,
    code_bytes_per_row: usize,
    k: usize,
    low_dim: usize,
    survivor_cap: usize,
    max_shadow_rows: usize,
    max_candidate_rows: usize,
) -> TurboQuantMse4CascadeShadowReport {
    if k == 0 || dim == 0 || code_bytes_per_row == 0 || norms.is_empty() {
        return TurboQuantMse4CascadeShadowReport {
            low_dim: low_dim.min(dim),
            survivor_cap,
            ..TurboQuantMse4CascadeShadowReport::default()
        };
    }
    let rows = norms.len().min(codes.len() / code_bytes_per_row);
    let candidate_rows = rows.min(max_candidate_rows).max(1);
    let audit_k = k.min(candidate_rows.saturating_sub(1));
    let effective_low_dim = low_dim.min(dim);
    let effective_survivor_cap = if survivor_cap == 0 {
        audit_k
    } else {
        survivor_cap
            .min(candidate_rows.saturating_sub(1))
            .max(audit_k)
    };
    let audit_rows = rows.min(max_shadow_rows).min(candidate_rows);
    if audit_rows == 0 || audit_k == 0 {
        return TurboQuantMse4CascadeShadowReport {
            shadow_rows_sampled: audit_rows,
            low_dim: effective_low_dim,
            survivor_cap: effective_survivor_cap,
            ..TurboQuantMse4CascadeShadowReport::default()
        };
    }

    let mut no_rerank_overlap_pairs = 0usize;
    let mut survivor_overlap_pairs = 0usize;
    let mut mismatch_rows = 0usize;
    for row in 0..audit_rows {
        let row_code = turboquant_mse4_code_row(codes, row, code_bytes_per_row);
        let mut full = Vec::with_capacity(candidate_rows.saturating_sub(1));
        let mut low = Vec::with_capacity(candidate_rows.saturating_sub(1));
        for col in 0..candidate_rows {
            if row == col {
                continue;
            }
            let col_code = turboquant_mse4_code_row(codes, col, code_bytes_per_row);
            let full_dist =
                turboquant_mse4_approx_l2(row_code, norms[row], col_code, norms[col], dim);
            let low_dist = turboquant_mse4_approx_l2_low_dim(
                row_code,
                norms[row],
                col_code,
                norms[col],
                dim,
                effective_low_dim,
            );
            full.push((full_dist, col));
            low.push((low_dist, col));
        }
        full.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        low.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let full_topk = &full[..audit_k.min(full.len())];
        let low_topk = &low[..audit_k.min(low.len())];
        let low_survivors = &low[..effective_survivor_cap.min(low.len())];
        let row_overlap = turboquant_mse4_topk_overlap(full_topk, low_topk);
        no_rerank_overlap_pairs += row_overlap;
        survivor_overlap_pairs += turboquant_mse4_topk_overlap(full_topk, low_survivors);
        if row_overlap < audit_k {
            mismatch_rows += 1;
        }
    }
    let audit_pairs_at_k = audit_rows.saturating_mul(audit_k);
    let total_candidate_pairs = audit_rows.saturating_mul(candidate_rows.saturating_sub(1));
    let survivor_rerank_pairs = audit_rows.saturating_mul(effective_survivor_cap);
    TurboQuantMse4CascadeShadowReport {
        shadow_rows_sampled: audit_rows,
        low_dim: effective_low_dim,
        survivor_cap: effective_survivor_cap,
        no_rerank_overlap_pairs,
        overlap_pairs_at_k: no_rerank_overlap_pairs,
        survivor_overlap_pairs,
        audit_pairs_at_k,
        survivor_audit_pairs: audit_pairs_at_k,
        mismatch_rows,
        low_dim_pairs: total_candidate_pairs,
        estimated_full_pairs_saved: total_candidate_pairs.saturating_sub(survivor_rerank_pairs),
    }
}

fn ratio_x1000(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(1000) / denominator
    }
}

fn turboquant_mse4_code_row(codes: &[i8], row: usize, code_bytes_per_row: usize) -> &[i8] {
    let start = row.saturating_mul(code_bytes_per_row);
    &codes[start..start + code_bytes_per_row]
}

fn turboquant_mse4_topk_overlap(left: &[(f32, usize)], right: &[(f32, usize)]) -> usize {
    left.iter()
        .filter(|(_, id)| right.iter().any(|(_, other_id)| other_id == id))
        .count()
}

pub(crate) fn turboquant_mse4_sidecar_codes_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tq4.codes", prefix.display()))
}

pub(crate) fn turboquant_mse4_sidecar_meta_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tq4.meta", prefix.display()))
}

pub(crate) fn create_turboquant_mse4_sidecar_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    bits: usize,
    seed: u64,
    source_fbin_bytes: u64,
) -> AnnResult<TurboQuantMse4SidecarCreateReport> {
    create_turboquant_mse4_sidecar_artifact_from_point_store(
        dataset,
        prefix,
        bits,
        seed,
        source_fbin_bytes,
        0,
    )
    .map(|artifact| artifact.report)
}

pub(crate) fn create_turboquant_mse4_sidecar_from_point_store_with_resident_prefix(
    dataset: &dyn PointStore,
    prefix: &Path,
    bits: usize,
    seed: u64,
    source_fbin_bytes: u64,
    resident_rows: usize,
) -> AnnResult<TurboQuantMse4SidecarCreateArtifact> {
    create_turboquant_mse4_sidecar_artifact_from_point_store(
        dataset,
        prefix,
        bits,
        seed,
        source_fbin_bytes,
        resident_rows,
    )
}

#[cfg(feature = "pipnn_cuda")]
pub(crate) fn create_turboquant_mse4_sidecar_from_point_store_cuda(
    dataset: &dyn PointStore,
    prefix: &Path,
    bits: usize,
    seed: u64,
    source_fbin_bytes: u64,
    device_id: usize,
) -> AnnResult<TurboQuantMse4SidecarCreateArtifact> {
    validate_turboquant_mse4_bits(bits)?;
    let total_start = Instant::now();
    let setup_start = Instant::now();
    let rows = dataset.len();
    let dim = dataset.dim();
    let code_bytes_per_row = turboquant_mse4_code_bytes_per_row(dim)?;
    let rotation = turboquant_mse4_dense_qr_rotation(dim, seed)?;
    let rotation_col_major = turboquant_mse4_pack_row_major_to_col_major(dim, dim, &rotation)?;
    let thresholds = turboquant_mse4_thresholds(dim)?.to_vec();
    let codes_bytes = rows.saturating_mul(code_bytes_per_row);
    let meta_rows_bytes = rows.saturating_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES);
    if let Some(parent) = prefix.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create TurboQuant-MSE4 sidecar directory '{}': {err}",
                parent.display()
            ))
        })?;
    }
    let codes_path = turboquant_mse4_sidecar_codes_path(prefix);
    let meta_path = turboquant_mse4_sidecar_meta_path(prefix);
    let mut codes_writer = BufWriter::new(File::create(&codes_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create TurboQuant-MSE4 sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?);
    let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create TurboQuant-MSE4 sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?);
    meta_writer
        .write_all(&vec![0_u8; TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES])
        .map_err(|err| {
            AnnError::log_index_error(format!("Failed to reserve TurboQuant-MSE4 header: {err}"))
        })?;

    let context = CudaContext::new(device_id).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create CUDA context {device_id} for TurboQuant-MSE4 sidecar builder: {err}"
        ))
    })?;
    let stream = context.default_stream();
    let blas = CudaBlas::new(stream.clone()).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create cuBLAS handle for TurboQuant-MSE4 sidecar builder: {err}"
        ))
    })?;
    let normalize_function =
        turboquant_mse4_load_cuda_builder_function(&stream, "tq4_normalize_transpose_kernel")?;
    let pack_function =
        turboquant_mse4_load_cuda_builder_function(&stream, "tq4_pack_rotated_kernel")?;
    let rotation_dev = stream.clone_htod(&rotation_col_major).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to upload TurboQuant-MSE4 dense-QR rotation to CUDA: {err}"
        ))
    })?;
    let thresholds_dev = stream.clone_htod(&thresholds).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to upload TurboQuant-MSE4 Lloyd-Max thresholds to CUDA: {err}"
        ))
    })?;

    let setup_wall = setup_start.elapsed();
    let hash =
        turboquant_mse4_metadata_source_fingerprint(rows, dim, bits, seed, source_fbin_bytes);
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut code_output_wall = Duration::ZERO;
    let mut meta_output_wall = Duration::ZERO;
    let mut flush_wall = Duration::ZERO;
    let mut header_write_wall = Duration::ZERO;
    let mut write_wall = Duration::ZERO;

    let max_chunk_rows = rows.min(TURBOQUANT_MSE4_CREATE_CHUNK_ROWS).max(1);
    let chunk_alloc_start = Instant::now();
    let mut raw_dev: CudaSlice<f32> = stream
        .alloc_zeros(max_chunk_rows.saturating_mul(dim))
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to allocate TurboQuant-MSE4 raw CUDA chunk: {err}"
            ))
        })?;
    let mut normalized_dev: CudaSlice<f32> = stream
        .alloc_zeros(max_chunk_rows.saturating_mul(dim))
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to allocate TurboQuant-MSE4 normalized CUDA chunk: {err}"
            ))
        })?;
    let mut rotated_dev: CudaSlice<f32> = stream
        .alloc_zeros(max_chunk_rows.saturating_mul(dim))
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to allocate TurboQuant-MSE4 rotated CUDA chunk: {err}"
            ))
        })?;
    let mut codes_dev: CudaSlice<i8> = stream
        .alloc_zeros(max_chunk_rows.saturating_mul(code_bytes_per_row))
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to allocate TurboQuant-MSE4 code CUDA chunk: {err}"
            ))
        })?;
    let mut norms_dev: CudaSlice<f32> = stream.alloc_zeros(max_chunk_rows).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to allocate TurboQuant-MSE4 norm CUDA chunk: {err}"
        ))
    })?;
    chunk_alloc_wall += chunk_alloc_start.elapsed();

    let mut row_start = 0usize;
    while row_start < rows {
        let count = (rows - row_start).min(max_chunk_rows);
        let chunk_alloc_start = Instant::now();
        let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
        let mut code_chunk = vec![0_i8; count.saturating_mul(code_bytes_per_row)];
        let mut norm_chunk = vec![0.0_f32; count];
        chunk_alloc_wall += chunk_alloc_start.elapsed();
        let raw_read_start = Instant::now();
        dataset.read_range_into(row_start as u32, count, &mut chunk)?;
        raw_read_wall += raw_read_start.elapsed();

        let quantize_start = Instant::now();
        {
            let mut raw_view = raw_dev.slice_mut(0..count.saturating_mul(dim));
            stream.memcpy_htod(&chunk, &mut raw_view).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to upload TurboQuant-MSE4 raw chunk to CUDA: {err}"
                ))
            })?;
        }
        let rows_u64 = count as u64;
        let dim_u64 = dim as u64;
        let normalize_cfg = LaunchConfig {
            grid_dim: (count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut launch = stream.launch_builder(&normalize_function);
            launch
                .arg(&raw_dev)
                .arg(&mut normalized_dev)
                .arg(&mut norms_dev)
                .arg(&rows_u64)
                .arg(&dim_u64);
            unsafe {
                launch.launch(normalize_cfg).map_err(|err| {
                    AnnError::log_index_error(format!(
                        "TurboQuant-MSE4 normalize CUDA kernel launch failed: {err}"
                    ))
                })?;
            }
        }
        let gemm_cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            m: count as i32,
            n: dim as i32,
            k: dim as i32,
            alpha: 1.0_f32,
            lda: count as i32,
            ldb: dim as i32,
            beta: 0.0_f32,
            ldc: count as i32,
        };
        unsafe {
            blas.gemm(gemm_cfg, &normalized_dev, &rotation_dev, &mut rotated_dev)
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "TurboQuant-MSE4 dense-QR cuBLAS rotation failed: {err}"
                    ))
                })?;
        }
        let code_bytes_u64 = code_bytes_per_row as u64;
        let pack_cfg = LaunchConfig {
            grid_dim: (count as u32, code_bytes_per_row.div_ceil(256) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut launch = stream.launch_builder(&pack_function);
            launch
                .arg(&rotated_dev)
                .arg(&thresholds_dev)
                .arg(&mut codes_dev)
                .arg(&rows_u64)
                .arg(&dim_u64)
                .arg(&code_bytes_u64);
            unsafe {
                launch.launch(pack_cfg).map_err(|err| {
                    AnnError::log_index_error(format!(
                        "TurboQuant-MSE4 pack CUDA kernel launch failed: {err}"
                    ))
                })?;
            }
        }
        stream
            .memcpy_dtoh(
                &codes_dev.slice(0..count.saturating_mul(code_bytes_per_row)),
                &mut code_chunk,
            )
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to download TurboQuant-MSE4 code chunk from CUDA: {err}"
                ))
            })?;
        stream
            .memcpy_dtoh(&norms_dev.slice(0..count), &mut norm_chunk)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to download TurboQuant-MSE4 norm chunk from CUDA: {err}"
                ))
            })?;
        stream.synchronize().map_err(|err| {
            AnnError::log_index_error(format!(
                "CUDA stream synchronize after TurboQuant-MSE4 sidecar chunk failed: {err}"
            ))
        })?;
        quantize_wall += quantize_start.elapsed();

        let write_start = Instant::now();
        let code_bytes = unsafe {
            std::slice::from_raw_parts(code_chunk.as_ptr().cast::<u8>(), code_chunk.len())
        };
        let code_output_start = Instant::now();
        codes_writer.write_all(code_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write TurboQuant-MSE4 sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        code_output_wall += code_output_start.elapsed();
        let meta_output_start = Instant::now();
        for norm in norm_chunk {
            meta_writer.write_all(&norm.to_le_bytes()).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to write TurboQuant-MSE4 sidecar norm '{}': {err}",
                    meta_path.display()
                ))
            })?;
        }
        meta_output_wall += meta_output_start.elapsed();
        write_wall += write_start.elapsed();
        row_start += count;
    }

    let flush_start = Instant::now();
    codes_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush TurboQuant-MSE4 sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?;
    meta_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush TurboQuant-MSE4 sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    flush_wall += flush_start.elapsed();
    drop(codes_writer);
    drop(meta_writer);

    let header = TurboQuantMse4SidecarHeader {
        rows,
        dim,
        code_bytes_per_row,
        bits,
        seed,
        source_fbin_bytes,
        source_rows: rows,
        source_dim: dim,
        source_fingerprint: hash,
        codes_bytes,
        meta_rows_bytes,
    };
    let header_write_start = Instant::now();
    let mut meta = File::options()
        .write(true)
        .open(&meta_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to reopen TurboQuant-MSE4 sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
    meta.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek TurboQuant-MSE4 sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    meta.write_all(&encode_turboquant_mse4_sidecar_header(&header)?)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write TurboQuant-MSE4 sidecar header '{}': {err}",
                meta_path.display()
            ))
        })?;
    header_write_wall += header_write_start.elapsed();

    Ok(TurboQuantMse4SidecarCreateArtifact {
        report: TurboQuantMse4SidecarCreateReport {
            header,
            total_wall: total_start.elapsed(),
            setup_wall,
            chunk_alloc_wall,
            raw_read_wall,
            source_hash_wall,
            quantize_wall,
            code_output_wall,
            meta_output_wall,
            flush_wall,
            header_write_wall,
            write_wall,
        },
        runtime_sidecar: None,
    })
}

#[cfg(not(feature = "pipnn_cuda"))]
pub(crate) fn create_turboquant_mse4_sidecar_from_point_store_cuda(
    _dataset: &dyn PointStore,
    _prefix: &Path,
    _bits: usize,
    _seed: u64,
    _source_fbin_bytes: u64,
    _device_id: usize,
) -> AnnResult<TurboQuantMse4SidecarCreateArtifact> {
    Err(AnnError::log_index_error(
        "TurboQuant-MSE4 CUDA sidecar builder requires the pipnn_cuda feature".to_string(),
    ))
}

fn create_turboquant_mse4_sidecar_artifact_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    bits: usize,
    seed: u64,
    source_fbin_bytes: u64,
    resident_rows: usize,
) -> AnnResult<TurboQuantMse4SidecarCreateArtifact> {
    validate_turboquant_mse4_bits(bits)?;
    let total_start = Instant::now();
    let setup_start = Instant::now();
    let rows = dataset.len();
    let dim = dataset.dim();
    let code_bytes_per_row = turboquant_mse4_code_bytes_per_row(dim)?;
    let rotation = turboquant_mse4_dense_qr_rotation(dim, seed)?;
    let codes_bytes = rows.saturating_mul(code_bytes_per_row);
    let meta_rows_bytes = rows.saturating_mul(TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES);
    if let Some(parent) = prefix.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create TurboQuant-MSE4 sidecar directory '{}': {err}",
                parent.display()
            ))
        })?;
    }
    let codes_path = turboquant_mse4_sidecar_codes_path(prefix);
    let meta_path = turboquant_mse4_sidecar_meta_path(prefix);
    let mut codes_writer = BufWriter::new(File::create(&codes_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create TurboQuant-MSE4 sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?);
    let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create TurboQuant-MSE4 sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?);
    meta_writer
        .write_all(&vec![0_u8; TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES])
        .map_err(|err| {
            AnnError::log_index_error(format!("Failed to reserve TurboQuant-MSE4 header: {err}"))
        })?;
    let setup_wall = setup_start.elapsed();
    let hash =
        turboquant_mse4_metadata_source_fingerprint(rows, dim, bits, seed, source_fbin_bytes);
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut code_output_wall = Duration::ZERO;
    let mut meta_output_wall = Duration::ZERO;
    let mut flush_wall = Duration::ZERO;
    let mut header_write_wall = Duration::ZERO;
    let mut write_wall = Duration::ZERO;
    let resident_rows = resident_rows.min(rows);
    let resident_alloc_start = Instant::now();
    let mut resident_codes = if resident_rows > 0 {
        Some(vec![0_i8; resident_rows.saturating_mul(code_bytes_per_row)])
    } else {
        None
    };
    let mut resident_norms = if resident_rows > 0 {
        Some(vec![0.0_f32; resident_rows])
    } else {
        None
    };
    chunk_alloc_wall += resident_alloc_start.elapsed();

    let mut row_start = 0usize;
    while row_start < rows {
        let count = (rows - row_start).min(TURBOQUANT_MSE4_CREATE_CHUNK_ROWS);
        let chunk_alloc_start = Instant::now();
        let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
        chunk_alloc_wall += chunk_alloc_start.elapsed();
        let raw_read_start = Instant::now();
        dataset.read_range_into(row_start as u32, count, &mut chunk)?;
        raw_read_wall += raw_read_start.elapsed();
        let quantize_start = Instant::now();
        let matrix =
            quantize_turboquant_mse4_matrix_with_rotation(&chunk, count, dim, bits, &rotation)?;
        quantize_wall += quantize_start.elapsed();
        if row_start < resident_rows {
            let resident_count = resident_rows.saturating_sub(row_start).min(count);
            let resident_code_start = row_start.saturating_mul(code_bytes_per_row);
            let resident_code_len = resident_count.saturating_mul(code_bytes_per_row);
            if let Some(codes) = resident_codes.as_mut() {
                codes[resident_code_start..resident_code_start + resident_code_len]
                    .copy_from_slice(&matrix.codes[..resident_code_len]);
            }
            if let Some(norms) = resident_norms.as_mut() {
                norms[row_start..row_start + resident_count]
                    .copy_from_slice(&matrix.norms[..resident_count]);
            }
        }
        let write_start = Instant::now();
        let code_bytes = unsafe {
            std::slice::from_raw_parts(matrix.codes.as_ptr().cast::<u8>(), matrix.codes.len())
        };
        let code_output_start = Instant::now();
        codes_writer.write_all(code_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write TurboQuant-MSE4 sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        code_output_wall += code_output_start.elapsed();
        let meta_output_start = Instant::now();
        for row in 0..count {
            meta_writer
                .write_all(&matrix.norms[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write TurboQuant-MSE4 sidecar norm '{}': {err}",
                        meta_path.display()
                    ))
                })?;
        }
        meta_output_wall += meta_output_start.elapsed();
        write_wall += write_start.elapsed();
        row_start += count;
    }

    let flush_start = Instant::now();
    codes_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush TurboQuant-MSE4 sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?;
    meta_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush TurboQuant-MSE4 sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    flush_wall += flush_start.elapsed();
    drop(codes_writer);
    drop(meta_writer);

    let header = TurboQuantMse4SidecarHeader {
        rows,
        dim,
        code_bytes_per_row,
        bits,
        seed,
        source_fbin_bytes,
        source_rows: rows,
        source_dim: dim,
        source_fingerprint: hash,
        codes_bytes,
        meta_rows_bytes,
    };
    let header_write_start = Instant::now();
    let mut meta = File::options()
        .write(true)
        .open(&meta_path)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to reopen TurboQuant-MSE4 sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
    meta.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek TurboQuant-MSE4 sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    meta.write_all(&encode_turboquant_mse4_sidecar_header(&header)?)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write TurboQuant-MSE4 sidecar header '{}': {err}",
                meta_path.display()
            ))
        })?;
    header_write_wall += header_write_start.elapsed();
    drop(meta);

    let runtime_sidecar = if resident_rows == 0 {
        None
    } else {
        let codes = resident_codes.expect("resident codes allocated when resident_rows > 0");
        let norms = resident_norms.expect("resident norms allocated when resident_rows > 0");
        let storage = if resident_rows >= rows {
            TurboQuantMse4SidecarStorage::Resident {
                codes,
                norms,
                preload_wall: Duration::ZERO,
            }
        } else {
            let codes_file = File::open(&codes_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open TurboQuant-MSE4 sidecar codes '{}': {err}",
                    codes_path.display()
                ))
            })?;
            let meta_file = File::open(&meta_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open TurboQuant-MSE4 sidecar meta '{}': {err}",
                    meta_path.display()
                ))
            })?;
            TurboQuantMse4SidecarStorage::PartialResident {
                codes_file,
                meta_file,
                resident_rows,
                codes,
                norms,
                preload_wall: Duration::ZERO,
            }
        };
        Some(TurboQuantMse4Sidecar {
            prefix: prefix.to_path_buf(),
            header: header.clone(),
            storage,
        })
    };

    Ok(TurboQuantMse4SidecarCreateArtifact {
        report: TurboQuantMse4SidecarCreateReport {
            header,
            total_wall: total_start.elapsed(),
            setup_wall,
            chunk_alloc_wall,
            raw_read_wall,
            source_hash_wall,
            quantize_wall,
            code_output_wall,
            meta_output_wall,
            flush_wall,
            header_write_wall,
            write_wall,
        },
        runtime_sidecar,
    })
}

fn decode_norm_rows(buf: &[u8], norms: &mut [f32]) -> AnnResult<()> {
    decode_norm_rows_into_offset(buf, norms, 0)
}

fn turboquant_mse4_pack_row_major_to_col_major(
    rows: usize,
    cols: usize,
    row_major: &[f32],
) -> AnnResult<Vec<f32>> {
    if row_major.len() != rows.saturating_mul(cols) {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 matrix layout conversion shape mismatch: rows={rows} cols={cols} values={}",
            row_major.len()
        )));
    }
    let mut col_major = vec![0.0_f32; row_major.len()];
    for row in 0..rows {
        for col in 0..cols {
            col_major[col * rows + row] = row_major[row * cols + col];
        }
    }
    Ok(col_major)
}

#[cfg(feature = "pipnn_cuda")]
fn turboquant_mse4_cuda_builder_arch(
    stream: &std::sync::Arc<CudaStream>,
) -> AnnResult<Option<&'static str>> {
    let (major, minor) = stream.context().compute_capability().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to query CUDA compute capability for TurboQuant-MSE4 builder: {err}"
        ))
    })?;
    let arch = match (major, minor) {
        (7, 5) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 7) => Some("compute_87"),
        (8, 9) => Some("compute_89"),
        (9, _) => Some("compute_90"),
        (major, _) if major > 9 => Some("compute_90"),
        _ => None,
    };
    Ok(arch)
}

#[cfg(feature = "pipnn_cuda")]
fn turboquant_mse4_load_cuda_builder_function(
    stream: &std::sync::Arc<CudaStream>,
    function_name: &str,
) -> AnnResult<CudaFunction> {
    let ptx = compile_ptx_with_opts(
        TURBOQUANT_MSE4_CUDA_BUILDER_SRC,
        CompileOptions {
            arch: turboquant_mse4_cuda_builder_arch(stream)?,
            options: vec!["--std=c++17".to_string()],
            ..CompileOptions::default()
        },
    )
    .map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to compile TurboQuant-MSE4 CUDA sidecar builder: {err}"
        ))
    })?;
    let module = stream.context().load_module(ptx).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to load TurboQuant-MSE4 CUDA sidecar builder module: {err}"
        ))
    })?;
    module.load_function(function_name).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to load TurboQuant-MSE4 CUDA sidecar builder function {function_name}: {err}"
        ))
    })
}

fn decode_norm_rows_into_offset(buf: &[u8], norms: &mut [f32], offset: usize) -> AnnResult<()> {
    if buf.len() % TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES != 0 {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 norm metadata is not row aligned".to_string(),
        ));
    }
    let rows = buf.len() / TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES;
    if offset + rows > norms.len() {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 norm output buffer too small".to_string(),
        ));
    }
    for row in 0..rows {
        let base = row * TURBOQUANT_MSE4_SIDECAR_META_ROW_BYTES;
        norms[offset + row] = f32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
    }
    Ok(())
}

fn encode_turboquant_mse4_sidecar_header(
    header: &TurboQuantMse4SidecarHeader,
) -> AnnResult<[u8; TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES]> {
    let mut out = [0_u8; TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES];
    let mut pos = 0usize;
    write_header_bytes(&mut out, &mut pos, TURBOQUANT_MSE4_SIDECAR_MAGIC)?;
    write_header_u32(&mut out, &mut pos, TURBOQUANT_MSE4_SIDECAR_VERSION)?;
    write_header_u32(
        &mut out,
        &mut pos,
        TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES as u32,
    )?;
    write_header_rotator_label(&mut out, &mut pos, TURBOQUANT_MSE4_DENSE_QR_ROTATOR_LABEL)?;
    write_header_u64(&mut out, &mut pos, header.rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.code_bytes_per_row as u64)?;
    write_header_u64(&mut out, &mut pos, header.bits as u64)?;
    write_header_u64(&mut out, &mut pos, header.seed)?;
    write_header_u64(&mut out, &mut pos, header.source_fbin_bytes)?;
    write_header_u64(&mut out, &mut pos, header.source_rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fingerprint)?;
    write_header_u64(&mut out, &mut pos, header.codes_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.meta_rows_bytes as u64)?;
    Ok(out)
}

fn read_turboquant_mse4_sidecar_header(file: &mut File) -> AnnResult<TurboQuantMse4SidecarHeader> {
    let mut buf = [0_u8; TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES];
    file.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek TurboQuant-MSE4 sidecar header: {err}"
        ))
    })?;
    file.read_exact(&mut buf).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to read TurboQuant-MSE4 sidecar header: {err}"
        ))
    })?;
    let mut pos = 0usize;
    let magic = read_header_bytes(&buf, &mut pos, TURBOQUANT_MSE4_SIDECAR_MAGIC.len())?;
    if magic != TURBOQUANT_MSE4_SIDECAR_MAGIC {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 sidecar magic mismatch".to_string(),
        ));
    }
    let version = read_header_u32(&buf, &mut pos)?;
    if version != TURBOQUANT_MSE4_SIDECAR_VERSION {
        return Err(AnnError::log_index_error(format!(
            "Unsupported TurboQuant-MSE4 sidecar version {version}"
        )));
    }
    let header_bytes = read_header_u32(&buf, &mut pos)? as usize;
    if header_bytes != TURBOQUANT_MSE4_SIDECAR_HEADER_BYTES {
        return Err(AnnError::log_index_error(format!(
            "Unsupported TurboQuant-MSE4 sidecar header size {header_bytes}"
        )));
    }
    let rotator = read_header_rotator_label(&buf, &mut pos)?;
    if rotator != TURBOQUANT_MSE4_DENSE_QR_ROTATOR_LABEL {
        return Err(AnnError::log_index_error(format!(
            "Unsupported TurboQuant-MSE4 sidecar rotator '{rotator}', expected {}",
            TURBOQUANT_MSE4_DENSE_QR_ROTATOR_LABEL
        )));
    }
    Ok(TurboQuantMse4SidecarHeader {
        rows: read_header_u64(&buf, &mut pos)? as usize,
        dim: read_header_u64(&buf, &mut pos)? as usize,
        code_bytes_per_row: read_header_u64(&buf, &mut pos)? as usize,
        bits: read_header_u64(&buf, &mut pos)? as usize,
        seed: read_header_u64(&buf, &mut pos)?,
        source_fbin_bytes: read_header_u64(&buf, &mut pos)?,
        source_rows: read_header_u64(&buf, &mut pos)? as usize,
        source_dim: read_header_u64(&buf, &mut pos)? as usize,
        source_fingerprint: read_header_u64(&buf, &mut pos)?,
        codes_bytes: read_header_u64(&buf, &mut pos)? as usize,
        meta_rows_bytes: read_header_u64(&buf, &mut pos)? as usize,
    })
}

fn write_header_bytes(out: &mut [u8], pos: &mut usize, bytes: &[u8]) -> AnnResult<()> {
    let end = pos.checked_add(bytes.len()).ok_or_else(|| {
        AnnError::log_index_error("TurboQuant-MSE4 sidecar header offset overflows".to_string())
    })?;
    if end > out.len() {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 sidecar header buffer too small".to_string(),
        ));
    }
    out[*pos..end].copy_from_slice(bytes);
    *pos = end;
    Ok(())
}

fn write_header_u32(out: &mut [u8], pos: &mut usize, value: u32) -> AnnResult<()> {
    write_header_bytes(out, pos, &value.to_le_bytes())
}

fn write_header_u64(out: &mut [u8], pos: &mut usize, value: u64) -> AnnResult<()> {
    write_header_bytes(out, pos, &value.to_le_bytes())
}

fn write_header_rotator_label(out: &mut [u8], pos: &mut usize, label: &str) -> AnnResult<()> {
    if label.len() > TURBOQUANT_MSE4_ROTATOR_FIELD_BYTES {
        return Err(AnnError::log_index_error(format!(
            "TurboQuant-MSE4 rotator label '{label}' is too long"
        )));
    }
    let mut field = [0_u8; TURBOQUANT_MSE4_ROTATOR_FIELD_BYTES];
    field[..label.len()].copy_from_slice(label.as_bytes());
    write_header_bytes(out, pos, &field)
}

fn read_header_bytes<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> AnnResult<&'a [u8]> {
    let end = pos.checked_add(len).ok_or_else(|| {
        AnnError::log_index_error("TurboQuant-MSE4 sidecar header offset overflows".to_string())
    })?;
    if end > buf.len() {
        return Err(AnnError::log_index_error(
            "TurboQuant-MSE4 sidecar header is truncated".to_string(),
        ));
    }
    let out = &buf[*pos..end];
    *pos = end;
    Ok(out)
}

fn read_header_u32(buf: &[u8], pos: &mut usize) -> AnnResult<u32> {
    let bytes = read_header_bytes(buf, pos, std::mem::size_of::<u32>())?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_header_u64(buf: &[u8], pos: &mut usize) -> AnnResult<u64> {
    let bytes = read_header_bytes(buf, pos, std::mem::size_of::<u64>())?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_header_rotator_label(buf: &[u8], pos: &mut usize) -> AnnResult<String> {
    let bytes = read_header_bytes(buf, pos, TURBOQUANT_MSE4_ROTATOR_FIELD_BYTES)?;
    let len = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(TURBOQUANT_MSE4_ROTATOR_FIELD_BYTES);
    std::str::from_utf8(&bytes[..len])
        .map(|value| value.to_string())
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "TurboQuant-MSE4 sidecar rotator label is not UTF-8: {err}"
            ))
        })
}

fn fnv1a_update_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn turboquant_mse4_metadata_source_fingerprint(
    rows: usize,
    dim: usize,
    bits: usize,
    seed: u64,
    source_fbin_bytes: u64,
) -> u64 {
    let mut hash = fnv1a_update_u64(FNV_OFFSET_BASIS, rows as u64);
    hash = fnv1a_update_u64(hash, dim as u64);
    hash = fnv1a_update_u64(hash, bits as u64);
    hash = fnv1a_update_u64(hash, seed);
    fnv1a_update_u64(hash, source_fbin_bytes)
}

fn split_gather_wall(
    total: Duration,
    resident_rows: usize,
    file_rows: usize,
) -> (Duration, Duration) {
    let rows = resident_rows.saturating_add(file_rows);
    if rows == 0 {
        return (Duration::ZERO, Duration::ZERO);
    }
    if resident_rows == 0 {
        return (Duration::ZERO, total);
    }
    if file_rows == 0 {
        return (total, Duration::ZERO);
    }
    let resident_nanos = total.as_nanos().saturating_mul(resident_rows as u128) / rows as u128;
    let resident = Duration::from_nanos(resident_nanos.min(u64::MAX as u128) as u64);
    (resident, total.saturating_sub(resident))
}

fn advise_sidecar_dontneed(file: &File, offset: u64, len: usize) {
    if len == 0 {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let _ = unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            )
        };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turboquant_mse4_codebook_is_symmetric_and_scaled() {
        let centroids = turboquant_mse4_centroids(1024).unwrap();
        for idx in 0..TURBOQUANT_MSE4_LEVELS / 2 {
            let lhs = centroids[idx];
            let rhs = centroids[TURBOQUANT_MSE4_LEVELS - 1 - idx];
            assert!((lhs + rhs).abs() < 1.0e-6);
        }
        assert!((centroids[15] - 2.7235756 / 32.0).abs() < 1.0e-6);
    }

    #[test]
    fn turboquant_mse4_threshold_quantization_is_monotonic() {
        let thresholds = turboquant_mse4_thresholds(16).unwrap();
        assert_eq!(
            turboquant_mse4_quantize_value(thresholds[0] - 1.0e-3, &thresholds).unwrap(),
            0
        );
        assert_eq!(
            turboquant_mse4_quantize_value(thresholds[0] + 1.0e-3, &thresholds).unwrap(),
            1
        );
        assert_eq!(
            turboquant_mse4_quantize_value(thresholds[14] + 1.0e-3, &thresholds).unwrap(),
            15
        );
    }

    #[test]
    fn turboquant_mse4_pack_unpack_uses_high_nibble_first() {
        let packed = turboquant_mse4_pack_pair(3, 12).unwrap();
        assert_eq!(packed as u8, 0x3c);
        assert_eq!(turboquant_mse4_unpack_code(packed, 0), 3);
        assert_eq!(turboquant_mse4_unpack_code(packed, 1), 12);
    }

    #[test]
    fn turboquant_mse4_source_dense_qr_rotation_is_orthogonal() {
        let dim = 8;
        let rotation = turboquant_mse4_dense_qr_rotation(dim, 42).unwrap();
        assert_eq!(rotation.len(), dim * dim);
        for row in 0..dim {
            let mut row_norm = 0.0_f32;
            for axis in 0..dim {
                row_norm += rotation[row * dim + axis] * rotation[row * dim + axis];
            }
            assert!((row_norm - 1.0).abs() < 1.0e-4, "row={row} norm={row_norm}");
        }
        for lhs in 0..dim {
            for rhs in lhs + 1..dim {
                let mut dot = 0.0_f32;
                for axis in 0..dim {
                    dot += rotation[lhs * dim + axis] * rotation[rhs * dim + axis];
                }
                assert!(dot.abs() < 1.0e-4, "lhs={lhs} rhs={rhs} dot={dot}");
            }
        }
    }

    #[test]
    fn turboquant_mse4_header_records_dense_qr_rotator() {
        let header = TurboQuantMse4SidecarHeader {
            rows: 2,
            dim: 8,
            code_bytes_per_row: 4,
            bits: 4,
            seed: 42,
            source_fbin_bytes: 80,
            source_rows: 2,
            source_dim: 8,
            source_fingerprint: 123,
            codes_bytes: 8,
            meta_rows_bytes: 8,
        };
        let encoded = encode_turboquant_mse4_sidecar_header(&header).unwrap();
        assert!(
            encoded
                .windows(TURBOQUANT_MSE4_DENSE_QR_ROTATOR_LABEL.len())
                .any(|window| window == TURBOQUANT_MSE4_DENSE_QR_ROTATOR_LABEL.as_bytes()),
            "new TurboQuant sidecars must self-identify dense-qr rotation"
        );
    }

    #[test]
    fn turboquant_mse4_quantizes_zero_norm_row() {
        let matrix = quantize_turboquant_mse4_matrix(&[0.0; 16], 1, 16, 4, 42).unwrap();
        assert_eq!(matrix.code_bytes_per_row, 8);
        assert_eq!(matrix.norms, vec![0.0]);
        assert!(matrix.codes.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn turboquant_mse4_approx_l2_is_symmetric() {
        let points = (0..32)
            .map(|idx| (idx as f32 * 0.03125).sin())
            .collect::<Vec<_>>();
        let matrix = quantize_turboquant_mse4_matrix(&points, 2, 16, 4, 42).unwrap();
        let a = matrix.approx_l2(0, 1);
        let b = matrix.approx_l2(1, 0);
        assert!((a - b).abs() < 1.0e-6);
    }

    #[test]
    fn turboquant_mse4_approx_low_dim_matches_full_when_low_dim_is_dim() {
        for dim in [128, 256, 384, 1024] {
            let left_codes = (0..dim)
                .map(|axis| (axis % TURBOQUANT_MSE4_LEVELS) as u8)
                .collect::<Vec<_>>();
            let right_codes = (0..dim)
                .map(|axis| ((TURBOQUANT_MSE4_LEVELS - 1) - (axis % TURBOQUANT_MSE4_LEVELS)) as u8)
                .collect::<Vec<_>>();
            let left = pack_test_codes(&left_codes);
            let right = pack_test_codes(&right_codes);
            let full = turboquant_mse4_approx_l2(&left, 2.0, &right, 3.0, dim);
            let low = turboquant_mse4_approx_l2_low_dim(&left, 2.0, &right, 3.0, dim, dim);
            assert!(
                (full - low).abs() < 1.0e-5,
                "dim={dim} full={full} low={low}"
            );
        }
    }

    #[test]
    fn turboquant_mse4_approx_low_dim_uses_full_dim_scaling() {
        let dim = 1024;
        let low_dim = 128;
        let left = pack_test_codes(&vec![15; dim]);
        let right = pack_test_codes(&vec![15; dim]);
        let dist = turboquant_mse4_approx_l2_low_dim(&left, 2.0, &right, 3.0, dim, low_dim);
        let centroid = TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS[15];
        let dot_low = low_dim as f32 * centroid * centroid / dim as f32;
        let expected = 2.0_f32 * 2.0 + 3.0_f32 * 3.0 - 2.0 * 2.0 * 3.0 * dot_low;
        assert!((dist - expected).abs() < 1.0e-5);
    }

    #[test]
    fn turboquant_mse4_approx_low_dim_handles_zero_norm_rows() {
        let dim = 256;
        let left = pack_test_codes(&vec![15; dim]);
        let right = pack_test_codes(&vec![0; dim]);
        let dist = turboquant_mse4_approx_l2_low_dim(&left, 0.0, &right, 7.0, dim, 128);
        assert!((dist - 49.0).abs() < 1.0e-6);
    }

    #[test]
    fn turboquant_mse4_approx_low_dim_reads_odd_packed_axes() {
        let dim = 5;
        let left_codes = [1, 2, 3, 4, 5];
        let right_codes = [6, 7, 8, 9, 10];
        let left = pack_test_codes(&left_codes);
        let right = pack_test_codes(&right_codes);
        let dist = turboquant_mse4_approx_l2_low_dim(&left, 1.0, &right, 1.0, dim, dim);
        let expected_dot = left_codes
            .iter()
            .zip(right_codes.iter())
            .map(|(&lhs, &rhs)| {
                TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS[lhs as usize]
                    * TURBOQUANT_MSE4_GAUSSIAN_CENTROIDS[rhs as usize]
                    / dim as f32
            })
            .sum::<f32>();
        let expected = 2.0 - 2.0 * expected_dot;
        assert!((dist - expected).abs() < 1.0e-6);
    }

    #[test]
    fn turboquant_mse4_approx_low_dim_clamps_to_zero_for_identical_extreme_codes() {
        let dim = 128;
        let left = pack_test_codes(&vec![15; dim]);
        let right = pack_test_codes(&vec![15; dim]);
        let dist = turboquant_mse4_approx_l2_low_dim(&left, 1.0, &right, 1.0, dim, dim);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn turboquant_mse4_approx_low_dim_opposite_extreme_codes_are_farther_than_identical() {
        let dim = 128;
        let left = pack_test_codes(&vec![15; dim]);
        let same = pack_test_codes(&vec![15; dim]);
        let opposite = pack_test_codes(&vec![0; dim]);
        let same_dist = turboquant_mse4_approx_l2_low_dim(&left, 1.0, &same, 1.0, dim, dim);
        let opposite_dist = turboquant_mse4_approx_l2_low_dim(&left, 1.0, &opposite, 1.0, dim, dim);
        assert!(opposite_dist > same_dist);
    }

    #[test]
    fn turboquant_mse4_cascade_shadow_full_dim_has_full_overlap() {
        let rows = 8;
        let dim = 16;
        let points = (0..rows * dim)
            .map(|idx| ((idx as f32 + 1.0) * 0.017).sin())
            .collect::<Vec<_>>();
        let matrix = quantize_turboquant_mse4_matrix(&points, rows, dim, 4, 42).unwrap();

        let report = turboquant_mse4_cascade_shadow_report(
            &matrix.codes,
            &matrix.norms,
            matrix.dim,
            matrix.code_bytes_per_row,
            3,
            dim,
            0,
            64,
            4096,
        );

        assert_eq!(report.shadow_rows_sampled, rows);
        assert_eq!(report.low_dim, dim);
        assert_eq!(report.survivor_cap, 3);
        assert_eq!(report.mismatch_rows, 0);
        assert_eq!(report.no_rerank_overlap_pairs, report.audit_pairs_at_k);
        assert_eq!(report.overlap_pairs_at_k, report.audit_pairs_at_k);
        assert_eq!(report.survivor_overlap_pairs, report.audit_pairs_at_k);
        assert_eq!(report.no_rerank_overlap_at_k_x1000(), 1000);
        assert_eq!(report.overlap_at_survivor_cap_x1000(), 1000);
    }

    fn pack_test_codes(codes: &[u8]) -> Vec<i8> {
        codes
            .chunks(2)
            .map(|pair| {
                turboquant_mse4_pack_pair(pair[0], pair.get(1).copied().unwrap_or(0)).unwrap()
            })
            .collect()
    }
}

impl TurboQuantMse4Matrix {
    pub(crate) fn approx_l2(&self, left: usize, right: usize) -> f32 {
        turboquant_mse4_approx_l2(
            &self.codes[left * self.code_bytes_per_row..(left + 1) * self.code_bytes_per_row],
            self.norms[left],
            &self.codes[right * self.code_bytes_per_row..(right + 1) * self.code_bytes_per_row],
            self.norms[right],
            self.dim,
        )
    }
}
