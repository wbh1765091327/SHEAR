use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::dco_rabitq::rotate_rabitq_plus_fht_kac_row;
use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::point_store::{PointStore, U8PointStore};
use super::qcode_staging::D0AssignmentFragment;
use crate::common::{AnnError, AnnResult};

pub(crate) const SQ_INT8_DEFAULT_K_ALIGN: usize = 16;
pub(crate) const SQ_INT8_DEFAULT_BITS: usize = 8;
pub(crate) const SQ_WMMA_SIDECAR_HEADER_BYTES: usize = 256;
pub(crate) const SQ_WMMA_SIDECAR_META_ROW_BYTES: usize = 8;
const SQ_INT8_MIN_BITS: usize = 4;
const SQ_INT8_MAX_BITS: usize = 8;
const SQ_WMMA_SIDECAR_MAGIC: &[u8; 8] = b"PGSQW1\0\0";
const SQ_WMMA_SIDECAR_VERSION: u32 = 1;
const SQ_WMMA_SIDECAR_CREATE_CHUNK_ROWS: usize = 16 * 1024;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SqWmmaCodeLayout {
    #[default]
    DenseI8,
    PackedS4,
}

impl SqWmmaCodeLayout {
    const DENSE_I8_HEADER_CODE: u64 = 0;
    const PACKED_S4_HEADER_CODE: u64 = 1;

    #[inline]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::DenseI8 => "dense-i8",
            Self::PackedS4 => "packed-s4",
        }
    }

    #[inline]
    fn header_code(self) -> u64 {
        match self {
            Self::DenseI8 => Self::DENSE_I8_HEADER_CODE,
            Self::PackedS4 => Self::PACKED_S4_HEADER_CODE,
        }
    }

    fn from_header_code(code: u64) -> AnnResult<Self> {
        match code {
            Self::DENSE_I8_HEADER_CODE => Ok(Self::DenseI8),
            Self::PACKED_S4_HEADER_CODE => Ok(Self::PackedS4),
            other => Err(AnnError::log_index_error(format!(
                "Unsupported SQ+WMMA sidecar code layout {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SqInt8CpuKernel {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Avx512Vnni,
}

impl SqInt8CpuKernel {
    #[inline]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512Vnni => "avx512-vnni",
        }
    }

    #[inline]
    pub(crate) fn is_available(self) -> bool {
        match self {
            Self::Auto | Self::Scalar => true,
            Self::Avx2 => cpu_supports_avx2(),
            Self::Avx512Vnni => cpu_supports_avx512_vnni(),
        }
    }
}

impl std::fmt::Display for SqInt8CpuKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for SqInt8CpuKernel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "scalar" | "cpu" => Ok(Self::Scalar),
            "avx2" => Ok(Self::Avx2),
            "avx512-vnni" | "avx512_vnni" | "vnni" => Ok(Self::Avx512Vnni),
            other => Err(format!(
                "Unsupported SQ-int8 CPU kernel '{other}', expected one of: auto, scalar, avx2, avx512-vnni"
            )),
        }
    }
}

pub(crate) fn resolve_sq_int8_cpu_kernel(kernel: SqInt8CpuKernel) -> AnnResult<SqInt8CpuKernel> {
    match kernel {
        SqInt8CpuKernel::Auto => {
            if SqInt8CpuKernel::Avx512Vnni.is_available() {
                Ok(SqInt8CpuKernel::Avx512Vnni)
            } else if SqInt8CpuKernel::Avx2.is_available() {
                Ok(SqInt8CpuKernel::Avx2)
            } else {
                Ok(SqInt8CpuKernel::Scalar)
            }
        }
        requested if requested.is_available() => Ok(requested),
        requested => Err(AnnError::log_index_error(format!(
            "Requested SQ-int8 CPU kernel '{}' is not supported by this CPU",
            requested.as_str()
        ))),
    }
}

#[cfg(any(feature = "pipnn_cuda", test))]
pub(crate) const SQ_INT8_WMMA_DOT16_TILE_KERNEL_SRC: &str = r#"
#include <mma.h>

extern "C" __global__ void sq_int8_wmma_dot16_tile(
    const signed char* a_row_major,
    const signed char* b_row_major,
    int* dot_row_major,
    unsigned int lda,
    unsigned int ldb,
    unsigned int ldc
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720)
    __shared__ __align__(32) signed char a_tile[16 * 16];
    __shared__ __align__(32) signed char b_tile_col_major[16 * 16];
    __shared__ __align__(32) int c_tile[16 * 16];

    for (unsigned int idx = threadIdx.x; idx < 16u * 16u; idx += blockDim.x) {
        unsigned int row = idx / 16u;
        unsigned int axis = idx - row * 16u;
        a_tile[idx] = a_row_major[row * lda + axis];
        b_tile_col_major[row * 16u + axis] = b_row_major[row * ldb + axis];
    }
    __syncthreads();

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        16,
        16,
        16,
        signed char,
        nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        16,
        16,
        16,
        signed char,
        nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> acc_frag;
    nvcuda::wmma::fill_fragment(acc_frag, 0);
    nvcuda::wmma::load_matrix_sync(a_frag, a_tile, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, b_tile_col_major, 16);
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    nvcuda::wmma::store_matrix_sync(c_tile, acc_frag, 16, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < 16u * 16u; idx += blockDim.x) {
        unsigned int row = idx / 16u;
        unsigned int col = idx - row * 16u;
        dot_row_major[row * ldc + col] = c_tile[idx];
    }
#else
    for (unsigned int idx = threadIdx.x; idx < 16u * 16u; idx += blockDim.x) {
        unsigned int row = idx / 16u;
        unsigned int col = idx - row * 16u;
        int dot = 0;
        for (unsigned int axis = 0u; axis < 16u; ++axis) {
            dot += (int)a_row_major[row * lda + axis] * (int)b_row_major[col * ldb + axis];
        }
        dot_row_major[row * ldc + col] = dot;
    }
#endif
}
"#;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SqInt8Matrix {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) codes: Vec<i8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) norms: Vec<f32>,
}

/// Codes and coarse centers share one orthogonal transform within this matrix.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sq8RotatedResidualMatrix {
    pub(crate) residuals: SqInt8Matrix,
    pub(crate) centroid_ids: Vec<u32>,
    pub(crate) rotated_centroids: Vec<f32>,
}

impl Sq8RotatedResidualMatrix {
    pub(crate) fn approx_l2(&self, left: usize, right: usize) -> f32 {
        let dim = self.residuals.dim;
        let left_centroid = self.centroid_ids[left] as usize;
        let right_centroid = self.centroid_ids[right] as usize;
        let left_center = &self.rotated_centroids[left_centroid * dim..(left_centroid + 1) * dim];
        let right_center =
            &self.rotated_centroids[right_centroid * dim..(right_centroid + 1) * dim];
        let left_code = self.residuals.code_row(left);
        let right_code = self.residuals.code_row(right);
        let mut dist = 0.0_f32;
        for axis in 0..dim {
            let left_residual = self.residuals.scales[left] * f32::from(left_code[axis]);
            let right_residual = self.residuals.scales[right] * f32::from(right_code[axis]);
            let delta = (left_center[axis] + left_residual) - (right_center[axis] + right_residual);
            dist += delta * delta;
        }
        dist
    }
}

pub(crate) fn quantize_sq8_rotated_residual_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    centroids_row_major: &[f32],
    seed: u64,
) -> AnnResult<Sq8RotatedResidualMatrix> {
    if dim == 0
        || !dim.is_power_of_two()
        || points_row_major.len() != rows.saturating_mul(dim)
        || centroids_row_major.is_empty()
        || centroids_row_major.len() % dim != 0
    {
        return Err(AnnError::log_index_error(format!(
            "Rotated residual SQ8 shape mismatch: rows={rows} dim={dim} points={} centroids={}",
            points_row_major.len(),
            centroids_row_major.len()
        )));
    }
    let centroid_count = centroids_row_major.len() / dim;
    if centroid_count > u32::MAX as usize {
        return Err(AnnError::log_index_error(
            "Rotated residual SQ8 has too many centroids for u32 IDs".to_string(),
        ));
    }
    let mut rotated_centroids = vec![0.0_f32; centroids_row_major.len()];
    for center in 0..centroid_count {
        rotate_rabitq_plus_fht_kac_row(
            &centroids_row_major[center * dim..(center + 1) * dim],
            &mut rotated_centroids[center * dim..(center + 1) * dim],
            seed,
        )?;
    }
    let mut centroid_ids = Vec::with_capacity(rows);
    let padded_dim = sq_int8_padded_dim(dim, SQ_INT8_DEFAULT_K_ALIGN)?;
    let mut codes = vec![0_i8; rows.saturating_mul(padded_dim)];
    let mut scales = vec![1.0_f32; rows];
    let mut norms = vec![0.0_f32; rows];
    let mut residual = vec![0.0_f32; dim];
    let mut rotated = vec![0.0_f32; dim];
    for row in 0..rows {
        let point = &points_row_major[row * dim..(row + 1) * dim];
        rotate_rabitq_plus_fht_kac_row(point, &mut rotated, seed)?;
        let (best_center, _) = (0..centroid_count)
            .map(|center| {
                let centroid = &rotated_centroids[center * dim..(center + 1) * dim];
                let dist: f32 = rotated
                    .iter()
                    .zip(centroid)
                    .map(|(&value, &mean)| (value - mean) * (value - mean))
                    .sum();
                (center, dist)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
            .expect("at least one centroid validated");
        centroid_ids.push(best_center as u32);
        let centroid = &rotated_centroids[best_center * dim..(best_center + 1) * dim];
        for axis in 0..dim {
            residual[axis] = rotated[axis] - centroid[axis];
        }
        let (scale, _) = quantize_sq_int8_row(
            &residual,
            &mut codes[row * padded_dim..(row + 1) * padded_dim],
        )?;
        scales[row] = scale;
        norms[row] = codes[row * padded_dim..row * padded_dim + dim]
            .iter()
            .map(|&code| {
                let value = scale * f32::from(code);
                value * value
            })
            .sum();
    }
    let residuals = SqInt8Matrix {
        rows,
        dim,
        padded_dim,
        codes,
        scales,
        norms,
    };
    Ok(Sq8RotatedResidualMatrix {
        residuals,
        centroid_ids,
        rotated_centroids,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SqInt8LeafShadowConfig {
    pub(crate) k_align: usize,
    pub(crate) survivors: usize,
    pub(crate) audit: bool,
}

impl Default for SqInt8LeafShadowConfig {
    fn default() -> Self {
        Self {
            k_align: SQ_INT8_DEFAULT_K_ALIGN,
            survivors: 64,
            audit: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SqInt8LeafShadowTelemetry {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) full_pairs: usize,
    pub(crate) prefilter_pairs: usize,
    pub(crate) pruned_pairs: usize,
    pub(crate) survivor_pairs: usize,
    pub(crate) exact_pairs: usize,
    pub(crate) false_prune_audit_pairs: usize,
    pub(crate) false_prune_audit_misses: usize,
    pub(crate) code_bytes: usize,
    pub(crate) scale_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SqInt8LeafShadowResult {
    pub(crate) topk_ids: Vec<Vec<u32>>,
    pub(crate) telemetry: SqInt8LeafShadowTelemetry,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SqWmmaSidecarHeader {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) k_align: usize,
    pub(crate) bits: usize,
    pub(crate) code_layout: SqWmmaCodeLayout,
    pub(crate) code_row_bytes: usize,
    pub(crate) source_fbin_bytes: u64,
    pub(crate) source_rows: usize,
    pub(crate) source_dim: usize,
    pub(crate) source_fingerprint: u64,
    pub(crate) codes_bytes: usize,
    pub(crate) meta_rows_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SqWmmaGatheredInput {
    pub(crate) global_ids: Vec<u32>,
    pub(crate) codes: Vec<i8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) norms: Vec<f32>,
    pub(crate) code_sums: Vec<f32>,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) bits: usize,
    pub(crate) code_layout: SqWmmaCodeLayout,
    pub(crate) code_row_bytes: usize,
    pub(crate) gather_wall: Duration,
    pub(crate) resident_source: bool,
    pub(crate) resident_rows: usize,
    pub(crate) file_rows: usize,
    pub(crate) resident_gather_wall: Duration,
    pub(crate) file_gather_wall: Duration,
    /// Reconstructed rotated vectors for the center-aware residual SQ8 mode.
    pub(crate) reconstructed_vectors: Option<Vec<f32>>,
    pub(crate) rotated_center_ids: Option<Vec<u32>>,
    pub(crate) rotated_centers: Option<Vec<f32>>,
}

#[derive(Debug)]
pub(crate) struct SqWmmaSidecar {
    prefix: PathBuf,
    header: SqWmmaSidecarHeader,
    storage: SqWmmaSidecarStorage,
}

#[derive(Debug)]
enum SqWmmaSidecarStorage {
    File {
        codes_file: SqWmmaSidecarReadFile,
        meta_file: SqWmmaSidecarReadFile,
    },
    PartialResident {
        codes_file: SqWmmaSidecarReadFile,
        meta_file: SqWmmaSidecarReadFile,
        resident_rows: usize,
        codes: Vec<i8>,
        norms: Vec<f32>,
        scales: Vec<f32>,
        preload_wall: Duration,
    },
    Resident {
        codes: Vec<i8>,
        norms: Vec<f32>,
        scales: Vec<f32>,
        preload_wall: Duration,
    },
}

#[derive(Debug)]
enum SqWmmaSidecarReadFile {
    Buffered(File),
    Direct(DirectIoFile),
}

impl SqWmmaSidecarReadFile {
    fn open(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        if io_cfg.enabled {
            Ok(Self::Direct(DirectIoFile::open_read(path, io_cfg)?))
        } else {
            Ok(Self::Buffered(File::open(path)?))
        }
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> AnnResult<()> {
        match self {
            Self::Buffered(file) => file.read_exact_at(buf, offset).map_err(Into::into),
            Self::Direct(file) => file.read_exact_at(buf, offset),
        }
    }

    fn advise_dontneed(&self, offset: u64, len: usize) {
        if let Self::Buffered(file) = self {
            advise_sidecar_dontneed(file, offset, len);
        }
    }

    fn is_direct(&self) -> bool {
        matches!(self, Self::Direct(_))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SqWmmaSidecarCreateReport {
    pub(crate) header: SqWmmaSidecarHeader,
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

pub(crate) struct SqWmmaSidecarCreateArtifact {
    pub(crate) report: SqWmmaSidecarCreateReport,
    pub(crate) runtime_sidecar: Option<SqWmmaSidecar>,
}

#[derive(Debug)]
pub(crate) struct SqWmmaSidecarShardWriter {
    prefix: PathBuf,
    rows: usize,
    dim: usize,
    padded_dim: usize,
    bits: usize,
    code_row_bytes: usize,
    source_fbin_bytes: u64,
    codes_writer: BufWriter<File>,
    meta_writer: BufWriter<File>,
    written_rows: usize,
    total_start: Instant,
    setup_wall: Duration,
    code_output_wall: Duration,
    meta_output_wall: Duration,
    write_wall: Duration,
}

#[derive(Debug)]
pub(crate) struct SqWmmaResidentSidecarBuilder {
    prefix: PathBuf,
    rows: usize,
    dim: usize,
    padded_dim: usize,
    bits: usize,
    code_row_bytes: usize,
    source_fbin_bytes: u64,
    codes: Vec<i8>,
    norms: Vec<f32>,
    scales: Vec<f32>,
    written_rows: usize,
    total_start: Instant,
    setup_wall: Duration,
    code_output_wall: Duration,
    meta_output_wall: Duration,
    write_wall: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SqGpuD0StagingProfile {
    pub(crate) host_rotated_quantize_wall: Duration,
    pub(crate) raw_h2d_wall: Duration,
    pub(crate) gpu_quantize_wall: Duration,
    pub(crate) qcode_d2h_wall: Duration,
    pub(crate) gpu_d0_assign_wall: Duration,
    pub(crate) assignment_fragment_d2h_wall: Duration,
    pub(crate) leader_prepare_wall: Duration,
    pub(crate) leader_h2d_wall: Duration,
    pub(crate) leader_quantize_wall: Duration,
    pub(crate) leader_code_bytes: usize,
    pub(crate) d0_assign_kernel_requested: &'static str,
    pub(crate) d0_assign_kernel_mode: &'static str,
    pub(crate) d0_assign_rows: usize,
    pub(crate) d0_assign_leaders: usize,
    pub(crate) d0_assign_pair_count: usize,
    pub(crate) d0_assign_fanout: usize,
    pub(crate) d0_assign_row_tile: usize,
    pub(crate) d0_assign_row_chunk: usize,
    pub(crate) d0_assign_leader_tile: usize,
    pub(crate) d0_assign_partial_r: usize,
    pub(crate) d0_assign_leader_tile_count: usize,
    pub(crate) d0_assign_partial_bytes: usize,
    pub(crate) d0_assign_distance_kernel_wall: Duration,
    pub(crate) d0_assign_merge_kernel_wall: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SqGpuD0QuantizeAssignRequest<'a> {
    pub(crate) raw_vectors_row_major: &'a [f32],
    pub(crate) global_ids: &'a [u32],
    pub(crate) global_d0_leader_ids: &'a [u32],
    pub(crate) global_d0_leader_vectors_row_major: &'a [f32],
    pub(crate) leader_payload: Option<&'a SqInt8Matrix>,
    pub(crate) rotated_residual: Option<Sq8RotatedResidualRequest<'a>>,
    pub(crate) dim: usize,
    pub(crate) fanout_top: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sq8RotatedResidualRequest<'a> {
    pub(crate) centroids_row_major: &'a [f32],
    pub(crate) seed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SqGpuD0QuantizeAssignResult {
    pub(crate) matrix: SqInt8Matrix,
    pub(crate) centroid_ids: Option<Vec<u32>>,
    pub(crate) fragments: Vec<D0AssignmentFragment>,
    pub(crate) profile: SqGpuD0StagingProfile,
}

static SQ_WMMA_CREATED_SIDECAR_CACHE: OnceLock<Mutex<HashMap<PathBuf, SqWmmaSidecar>>> =
    OnceLock::new();

impl SqInt8Matrix {
    pub(crate) fn code_row(&self, row: usize) -> &[i8] {
        let start = row * self.padded_dim;
        &self.codes[start..start + self.padded_dim]
    }

    pub(crate) fn approx_l2(&self, left: usize, right: usize) -> f32 {
        sq_int8_approx_l2(
            self.code_row(left),
            self.scales[left],
            self.norms[left],
            self.code_row(right),
            self.scales[right],
            self.norms[right],
        )
    }
}

impl SqWmmaSidecarHeader {
    pub(crate) fn validate_for_dataset(&self, rows: usize, dim: usize) -> AnnResult<()> {
        self.validate_for_dataset_with_bits(rows, dim, SQ_INT8_DEFAULT_BITS)
    }

    pub(crate) fn validate_for_dataset_with_bits(
        &self,
        rows: usize,
        dim: usize,
        bits: usize,
    ) -> AnnResult<()> {
        validate_sq_int8_bits(bits)?;
        if self.rows != rows
            || self.dim != dim
            || self.source_rows != rows
            || self.source_dim != dim
        {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar shape mismatch: sidecar rows={} dim={} source_rows={} source_dim={} expected rows={rows} dim={dim}",
                self.rows, self.dim, self.source_rows, self.source_dim
            )));
        }
        if self.k_align != SQ_INT8_DEFAULT_K_ALIGN {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar k_align mismatch: sidecar={} expected={}",
                self.k_align, SQ_INT8_DEFAULT_K_ALIGN
            )));
        }
        if self.bits != bits {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar bits mismatch: sidecar={} expected={bits}",
                self.bits
            )));
        }
        let expected_code_row_bytes = sq_wmma_code_row_bytes(self.padded_dim, self.code_layout)?;
        if self.code_row_bytes != expected_code_row_bytes {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar code_row_bytes mismatch: sidecar={} expected={} layout={}",
                self.code_row_bytes,
                expected_code_row_bytes,
                self.code_layout.as_str()
            )));
        }
        let expected_codes_bytes = self.rows.checked_mul(self.code_row_bytes).ok_or_else(|| {
            AnnError::log_index_error("SQ+WMMA sidecar codes byte count overflows".to_string())
        })?;
        if self.codes_bytes != expected_codes_bytes {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar codes_bytes mismatch: sidecar={} expected={} rows={} code_row_bytes={}",
                self.codes_bytes, expected_codes_bytes, self.rows, self.code_row_bytes
            )));
        }
        Ok(())
    }
}

impl SqWmmaGatheredInput {
    pub(crate) fn approx_l2(&self, left: usize, right: usize) -> f32 {
        let left_dense;
        let right_dense;
        let (left_codes, right_codes) = match self.code_layout {
            SqWmmaCodeLayout::DenseI8 => {
                let left_start = left * self.code_row_bytes;
                let right_start = right * self.code_row_bytes;
                (
                    &self.codes[left_start..left_start + self.padded_dim],
                    &self.codes[right_start..right_start + self.padded_dim],
                )
            }
            SqWmmaCodeLayout::PackedS4 => {
                let left_start = left * self.code_row_bytes;
                let right_start = right * self.code_row_bytes;
                left_dense = unpack_sq_s4_codes(
                    &self.codes[left_start..left_start + self.code_row_bytes],
                    1,
                    self.padded_dim,
                )
                .expect("SQ+WMMA packed-s4 gathered left row should unpack");
                right_dense = unpack_sq_s4_codes(
                    &self.codes[right_start..right_start + self.code_row_bytes],
                    1,
                    self.padded_dim,
                )
                .expect("SQ+WMMA packed-s4 gathered right row should unpack");
                (&left_dense[..], &right_dense[..])
            }
        };
        sq_int8_approx_l2(
            left_codes,
            self.scales[left],
            self.norms[left],
            right_codes,
            self.scales[right],
            self.norms[right],
        )
    }
}

impl SqWmmaSidecar {
    pub(crate) fn open_validated(prefix: &Path, rows: usize, dim: usize) -> AnnResult<Self> {
        Self::open_validated_with_bits(prefix, rows, dim, SQ_INT8_DEFAULT_BITS)
    }

    pub(crate) fn open_validated_with_bits(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
    ) -> AnnResult<Self> {
        Self::open_validated_with_bits_and_io(prefix, rows, dim, bits, DirectIoConfig::disabled())
    }

    pub(crate) fn open_validated_with_bits_and_io(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        let codes_path = sq_wmma_sidecar_codes_path(prefix);
        let meta_path = sq_wmma_sidecar_meta_path(prefix);
        let mut meta_file = File::open(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open SQ+WMMA sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
        let header = read_sq_wmma_sidecar_header(&mut meta_file)?;
        header.validate_for_dataset_with_bits(rows, dim, bits)?;
        let codes_len = fs::metadata(&codes_path)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to stat SQ+WMMA sidecar codes '{}': {err}",
                    codes_path.display()
                ))
            })?
            .len() as usize;
        if codes_len != header.codes_bytes {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar codes size mismatch: file={} header={}",
                codes_len, header.codes_bytes
            )));
        }
        let codes_file = SqWmmaSidecarReadFile::open(&codes_path, io_cfg).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open SQ+WMMA sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        let meta_file = if io_cfg.enabled {
            SqWmmaSidecarReadFile::open(&meta_path, io_cfg).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open SQ+WMMA sidecar meta '{}': {err}",
                    meta_path.display()
                ))
            })?
        } else {
            SqWmmaSidecarReadFile::Buffered(meta_file)
        };
        Ok(Self {
            prefix: prefix.to_path_buf(),
            header,
            storage: SqWmmaSidecarStorage::File {
                codes_file,
                meta_file,
            },
        })
    }

    pub(crate) fn preload_validated(prefix: &Path, rows: usize, dim: usize) -> AnnResult<Self> {
        Self::preload_validated_with_bits(prefix, rows, dim, SQ_INT8_DEFAULT_BITS)
    }

    pub(crate) fn preload_validated_with_bits(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
    ) -> AnnResult<Self> {
        let preload_start = Instant::now();
        let sidecar = Self::open_validated_with_bits(prefix, rows, dim, bits)?;
        let header = sidecar.header.clone();
        let mut codes = vec![0_i8; header.codes_bytes];
        let mut norms = vec![0.0_f32; header.rows];
        let mut scales = vec![0.0_f32; header.rows];

        let SqWmmaSidecarStorage::File {
            codes_file,
            meta_file,
        } = &sidecar.storage
        else {
            unreachable!("newly opened SQ+WMMA sidecar is always file-backed")
        };

        let codes_bytes =
            unsafe { std::slice::from_raw_parts_mut(codes.as_mut_ptr().cast::<u8>(), codes.len()) };
        codes_file.read_exact_at(codes_bytes, 0).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload SQ+WMMA sidecar codes '{}': {err}",
                sq_wmma_sidecar_codes_path(prefix).display()
            ))
        })?;
        codes_file.advise_dontneed(0, codes_bytes.len());

        let mut meta_bytes = vec![0_u8; header.meta_rows_bytes];
        meta_file
            .read_exact_at(&mut meta_bytes, SQ_WMMA_SIDECAR_HEADER_BYTES as u64)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload SQ+WMMA sidecar metadata '{}': {err}",
                    sq_wmma_sidecar_meta_path(prefix).display()
                ))
            })?;
        meta_file.advise_dontneed(SQ_WMMA_SIDECAR_HEADER_BYTES as u64, meta_bytes.len());
        for row in 0..header.rows {
            let base = row * SQ_WMMA_SIDECAR_META_ROW_BYTES;
            norms[row] = f32::from_le_bytes(meta_bytes[base..base + 4].try_into().unwrap());
            scales[row] = f32::from_le_bytes(meta_bytes[base + 4..base + 8].try_into().unwrap());
        }

        Ok(Self {
            prefix: sidecar.prefix,
            header: sidecar.header,
            storage: SqWmmaSidecarStorage::Resident {
                codes,
                norms,
                scales,
                preload_wall: preload_start.elapsed(),
            },
        })
    }

    pub(crate) fn preload_prefix_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        resident_rows: usize,
    ) -> AnnResult<Self> {
        Self::preload_prefix_validated_with_bits(
            prefix,
            rows,
            dim,
            SQ_INT8_DEFAULT_BITS,
            resident_rows,
        )
    }

    pub(crate) fn preload_prefix_validated_with_bits(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        resident_rows: usize,
    ) -> AnnResult<Self> {
        Self::preload_prefix_validated_with_bits_and_suffix_io(
            prefix,
            rows,
            dim,
            bits,
            resident_rows,
            DirectIoConfig::disabled(),
        )
    }

    pub(crate) fn preload_prefix_validated_with_bits_and_suffix_io(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        resident_rows: usize,
        suffix_io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        let resident_rows = resident_rows.min(rows);
        if resident_rows == 0 {
            return Self::open_validated_with_bits_and_io(prefix, rows, dim, bits, suffix_io_cfg);
        }
        if resident_rows >= rows {
            return Self::preload_validated_with_bits(prefix, rows, dim, bits);
        }

        let preload_start = Instant::now();
        let sidecar = Self::open_validated_with_bits(prefix, rows, dim, bits)?;
        let header = sidecar.header.clone();
        let mut codes = vec![0_i8; resident_rows.saturating_mul(header.code_row_bytes)];
        let mut norms = vec![0.0_f32; resident_rows];
        let mut scales = vec![0.0_f32; resident_rows];

        let SqWmmaSidecarStorage::File {
            codes_file,
            meta_file,
        } = &sidecar.storage
        else {
            unreachable!("newly opened SQ+WMMA sidecar is always file-backed")
        };

        let codes_bytes =
            unsafe { std::slice::from_raw_parts_mut(codes.as_mut_ptr().cast::<u8>(), codes.len()) };
        codes_file.read_exact_at(codes_bytes, 0).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload SQ+WMMA sidecar resident prefix codes '{}': {err}",
                sq_wmma_sidecar_codes_path(prefix).display()
            ))
        })?;
        codes_file.advise_dontneed(0, codes_bytes.len());

        let mut meta_bytes =
            vec![0_u8; resident_rows.saturating_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES)];
        meta_file
            .read_exact_at(&mut meta_bytes, SQ_WMMA_SIDECAR_HEADER_BYTES as u64)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload SQ+WMMA sidecar resident prefix metadata '{}': {err}",
                    sq_wmma_sidecar_meta_path(prefix).display()
                ))
            })?;
        meta_file.advise_dontneed(SQ_WMMA_SIDECAR_HEADER_BYTES as u64, meta_bytes.len());
        for row in 0..resident_rows {
            let base = row * SQ_WMMA_SIDECAR_META_ROW_BYTES;
            norms[row] = f32::from_le_bytes(meta_bytes[base..base + 4].try_into().unwrap());
            scales[row] = f32::from_le_bytes(meta_bytes[base + 4..base + 8].try_into().unwrap());
        }

        let SqWmmaSidecarStorage::File {
            codes_file,
            meta_file,
        } = sidecar.storage
        else {
            unreachable!("newly opened SQ+WMMA sidecar is always file-backed")
        };
        let codes_file = if suffix_io_cfg.enabled {
            SqWmmaSidecarReadFile::open(&sq_wmma_sidecar_codes_path(prefix), suffix_io_cfg)?
        } else {
            codes_file
        };
        let meta_file = if suffix_io_cfg.enabled {
            SqWmmaSidecarReadFile::open(&sq_wmma_sidecar_meta_path(prefix), suffix_io_cfg)?
        } else {
            meta_file
        };
        Ok(Self {
            prefix: sidecar.prefix,
            header: sidecar.header,
            storage: SqWmmaSidecarStorage::PartialResident {
                codes_file,
                meta_file,
                resident_rows,
                codes,
                norms,
                scales,
                preload_wall: preload_start.elapsed(),
            },
        })
    }

    pub(crate) fn header(&self) -> &SqWmmaSidecarHeader {
        &self.header
    }

    pub(crate) fn prefix(&self) -> &Path {
        &self.prefix
    }

    pub(crate) fn is_resident(&self) -> bool {
        matches!(
            self.storage,
            SqWmmaSidecarStorage::PartialResident { .. } | SqWmmaSidecarStorage::Resident { .. }
        )
    }

    pub(crate) fn is_fully_resident(&self) -> bool {
        matches!(self.storage, SqWmmaSidecarStorage::Resident { .. })
    }

    pub(crate) fn uses_direct_io(&self) -> bool {
        match &self.storage {
            SqWmmaSidecarStorage::File {
                codes_file,
                meta_file,
            }
            | SqWmmaSidecarStorage::PartialResident {
                codes_file,
                meta_file,
                ..
            } => codes_file.is_direct() && meta_file.is_direct(),
            SqWmmaSidecarStorage::Resident { .. } => false,
        }
    }

    pub(crate) fn resident_rows(&self) -> usize {
        match &self.storage {
            SqWmmaSidecarStorage::File { .. } => 0,
            SqWmmaSidecarStorage::PartialResident { resident_rows, .. } => *resident_rows,
            SqWmmaSidecarStorage::Resident { .. } => self.header.rows,
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.resident_rows().saturating_mul(
            self.header
                .code_row_bytes
                .saturating_add(SQ_WMMA_SIDECAR_META_ROW_BYTES),
        )
    }

    pub(crate) fn preload_wall(&self) -> Duration {
        match &self.storage {
            SqWmmaSidecarStorage::File { .. } => Duration::ZERO,
            SqWmmaSidecarStorage::PartialResident { preload_wall, .. } => *preload_wall,
            SqWmmaSidecarStorage::Resident { preload_wall, .. } => *preload_wall,
        }
    }

    pub(crate) fn gather(&self, global_ids: &[u32]) -> AnnResult<SqWmmaGatheredInput> {
        let gather_start = Instant::now();
        let rows = global_ids.len();
        let mut codes = vec![0_i8; rows.saturating_mul(self.header.code_row_bytes)];
        let mut norms = vec![0.0_f32; rows];
        let mut scales = vec![0.0_f32; rows];

        let mut resident_rows = 0usize;
        let mut file_rows = 0usize;
        let mut run_start = 0usize;
        while run_start < rows {
            let first_id = global_ids[run_start] as usize;
            if first_id >= self.header.rows {
                return Err(AnnError::log_index_error(format!(
                    "SQ+WMMA sidecar gather id {} out of bounds for {} rows",
                    first_id, self.header.rows
                )));
            }
            let mut run_end = run_start + 1;
            while run_end < rows && global_ids[run_end] == global_ids[run_end - 1].saturating_add(1)
            {
                if global_ids[run_end] as usize >= self.header.rows {
                    return Err(AnnError::log_index_error(format!(
                        "SQ+WMMA sidecar gather id {} out of bounds for {} rows",
                        global_ids[run_end], self.header.rows
                    )));
                }
                run_end += 1;
            }
            self.read_code_run(first_id, run_start, run_end, &mut codes)?;
            self.read_meta_run(first_id, run_start, run_end, &mut norms, &mut scales)?;
            let (resident, file) = self.source_rows_for_run(first_id, run_end - run_start);
            resident_rows = resident_rows.saturating_add(resident);
            file_rows = file_rows.saturating_add(file);
            run_start = run_end;
        }
        let gather_wall = gather_start.elapsed();
        let (resident_gather_wall, file_gather_wall) =
            split_gather_wall(gather_wall, resident_rows, file_rows);

        Ok(SqWmmaGatheredInput {
            global_ids: global_ids.to_vec(),
            codes,
            scales,
            norms,
            code_sums: vec![0.0_f32; rows],
            dim: self.header.dim,
            padded_dim: self.header.padded_dim,
            bits: self.header.bits,
            code_layout: self.header.code_layout,
            code_row_bytes: self.header.code_row_bytes,
            gather_wall,
            resident_source: resident_rows > 0 && file_rows == 0,
            resident_rows,
            file_rows,
            resident_gather_wall,
            file_gather_wall,
            reconstructed_vectors: None,
            rotated_center_ids: None,
            rotated_centers: None,
        })
    }

    fn source_rows_for_run(&self, first_id: usize, run_len: usize) -> (usize, usize) {
        match &self.storage {
            SqWmmaSidecarStorage::File { .. } => (0, run_len),
            SqWmmaSidecarStorage::Resident { .. } => (run_len, 0),
            SqWmmaSidecarStorage::PartialResident { resident_rows, .. } => {
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
        let row_bytes = self.header.code_row_bytes;
        let src_offset = first_id.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error("SQ+WMMA sidecar code offset overflows".to_string())
        })?;
        let dst_start = run_start.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error("SQ+WMMA sidecar output offset overflows".to_string())
        })?;
        let byte_len = run_len.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error("SQ+WMMA sidecar code run length overflows".to_string())
        })?;
        let dst = &mut out[dst_start..dst_start + byte_len];
        match &self.storage {
            SqWmmaSidecarStorage::File { codes_file, .. } => {
                let dst_bytes = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len())
                };
                codes_file
                    .read_exact_at(dst_bytes, src_offset as u64)
                    .map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read SQ+WMMA sidecar codes at offset {src_offset}: {err}"
                        ))
                    })
            }
            SqWmmaSidecarStorage::Resident { codes, .. } => {
                dst.copy_from_slice(&codes[src_offset..src_offset + byte_len]);
                Ok(())
            }
            SqWmmaSidecarStorage::PartialResident {
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
                            "SQ+WMMA sidecar partial code offset overflows".to_string(),
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
                            "Failed to read SQ+WMMA sidecar suffix codes at offset {file_offset}: {err}"
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
        scales: &mut [f32],
    ) -> AnnResult<()> {
        let run_len = run_end - run_start;
        let src_offset = SQ_WMMA_SIDECAR_HEADER_BYTES
            .checked_add(first_id.saturating_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES))
            .ok_or_else(|| {
                AnnError::log_index_error("SQ+WMMA sidecar meta offset overflows".to_string())
            })?;
        let byte_len = run_len
            .checked_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error("SQ+WMMA sidecar meta run length overflows".to_string())
            })?;
        match &self.storage {
            SqWmmaSidecarStorage::File { meta_file, .. } => {
                let mut buf = vec![0_u8; byte_len];
                meta_file
                    .read_exact_at(&mut buf, src_offset as u64)
                    .map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read SQ+WMMA sidecar metadata at offset {src_offset}: {err}"
                        ))
                    })?;
                for row in 0..run_len {
                    let base = row * SQ_WMMA_SIDECAR_META_ROW_BYTES;
                    norms[run_start + row] =
                        f32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
                    scales[run_start + row] =
                        f32::from_le_bytes(buf[base + 4..base + 8].try_into().unwrap());
                }
            }
            SqWmmaSidecarStorage::Resident {
                norms: all_norms,
                scales: all_scales,
                ..
            } => {
                let src_start = first_id;
                let src_end = first_id + run_len;
                norms[run_start..run_end].copy_from_slice(&all_norms[src_start..src_end]);
                scales[run_start..run_end].copy_from_slice(&all_scales[src_start..src_end]);
            }
            SqWmmaSidecarStorage::PartialResident {
                meta_file,
                resident_rows,
                norms: all_norms,
                scales: all_scales,
                ..
            } => {
                let resident_len = resident_rows.saturating_sub(first_id).min(run_len);
                if resident_len > 0 {
                    let src_start = first_id;
                    let src_end = first_id + resident_len;
                    let dst_end = run_start + resident_len;
                    norms[run_start..dst_end].copy_from_slice(&all_norms[src_start..src_end]);
                    scales[run_start..dst_end].copy_from_slice(&all_scales[src_start..src_end]);
                }
                if resident_len < run_len {
                    let file_first_id = first_id + resident_len;
                    let file_run_start = run_start + resident_len;
                    let file_run_len = run_len - resident_len;
                    let file_src_offset = SQ_WMMA_SIDECAR_HEADER_BYTES
                        .checked_add(file_first_id.saturating_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES))
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "SQ+WMMA sidecar partial meta offset overflows".to_string(),
                            )
                        })?;
                    let file_byte_len = file_run_len
                        .checked_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES)
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "SQ+WMMA sidecar partial meta run length overflows".to_string(),
                            )
                        })?;
                    let mut buf = vec![0_u8; file_byte_len];
                    meta_file
                        .read_exact_at(&mut buf, file_src_offset as u64)
                        .map_err(|err| {
                            AnnError::log_index_error(format!(
                                "Failed to read SQ+WMMA sidecar suffix metadata at offset {file_src_offset}: {err}"
                            ))
                        })?;
                    for row in 0..file_run_len {
                        let base = row * SQ_WMMA_SIDECAR_META_ROW_BYTES;
                        norms[file_run_start + row] =
                            f32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
                        scales[file_run_start + row] =
                            f32::from_le_bytes(buf[base + 4..base + 8].try_into().unwrap());
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn cache_sq_wmma_created_sidecar(sidecar: SqWmmaSidecar) {
    let cache = SQ_WMMA_CREATED_SIDECAR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cache
        .lock()
        .expect("SQ+WMMA created sidecar cache mutex poisoned")
        .insert(sidecar.prefix.clone(), sidecar);
}

pub(crate) fn take_cached_sq_wmma_created_sidecar(
    prefix: &Path,
    rows: usize,
    dim: usize,
) -> AnnResult<Option<SqWmmaSidecar>> {
    take_cached_sq_wmma_created_sidecar_with_bits(prefix, rows, dim, SQ_INT8_DEFAULT_BITS)
}

pub(crate) fn take_cached_sq_wmma_created_sidecar_with_bits(
    prefix: &Path,
    rows: usize,
    dim: usize,
    bits: usize,
) -> AnnResult<Option<SqWmmaSidecar>> {
    let Some(cache) = SQ_WMMA_CREATED_SIDECAR_CACHE.get() else {
        return Ok(None);
    };
    let Some(sidecar) = cache
        .lock()
        .expect("SQ+WMMA created sidecar cache mutex poisoned")
        .remove(prefix)
    else {
        return Ok(None);
    };
    sidecar
        .header
        .validate_for_dataset_with_bits(rows, dim, bits)?;
    Ok(Some(sidecar))
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

pub(crate) fn sq_int8_padded_dim(dim: usize, k_align: usize) -> AnnResult<usize> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "SQ-int8 quantization requires dim > 0".to_string(),
        ));
    }
    let k_align = k_align.max(1);
    Ok(dim.div_ceil(k_align) * k_align)
}

pub(crate) fn sq_wmma_code_row_bytes(
    padded_dim: usize,
    layout: SqWmmaCodeLayout,
) -> AnnResult<usize> {
    match layout {
        SqWmmaCodeLayout::DenseI8 => Ok(padded_dim),
        SqWmmaCodeLayout::PackedS4 => {
            if padded_dim % 2 != 0 {
                return Err(AnnError::log_index_error(format!(
                    "SQ+WMMA packed-s4 layout requires even padded_dim, got {padded_dim}"
                )));
            }
            Ok(padded_dim / 2)
        }
    }
}

#[inline]
fn pack_signed_s4(code: i8) -> AnnResult<u8> {
    if !(-8..=7).contains(&code) {
        return Err(AnnError::log_index_error(format!(
            "SQ+WMMA packed-s4 code {code} is outside signed 4-bit range [-8, 7]"
        )));
    }
    Ok((code as u8) & 0x0f)
}

#[inline]
fn unpack_signed_s4(nibble: u8) -> i8 {
    let value = (nibble & 0x0f) as i8;
    if value >= 8 { value - 16 } else { value }
}

pub(crate) fn pack_sq_s4_codes(
    dense_codes: &[i8],
    rows: usize,
    padded_dim: usize,
) -> AnnResult<Vec<i8>> {
    let dense_len = rows.checked_mul(padded_dim).ok_or_else(|| {
        AnnError::log_index_error("SQ+WMMA dense-s4 code length overflows".to_string())
    })?;
    if dense_codes.len() != dense_len {
        return Err(AnnError::log_index_error(format!(
            "SQ+WMMA dense-s4 pack shape mismatch: codes={} rows={rows} padded_dim={padded_dim}",
            dense_codes.len()
        )));
    }
    let row_bytes = sq_wmma_code_row_bytes(padded_dim, SqWmmaCodeLayout::PackedS4)?;
    let mut packed = vec![0_i8; rows.saturating_mul(row_bytes)];
    for row in 0..rows {
        let dense_base = row * padded_dim;
        let packed_base = row * row_bytes;
        for byte_idx in 0..row_bytes {
            let axis = byte_idx * 2;
            let low = pack_signed_s4(dense_codes[dense_base + axis])?;
            let high = pack_signed_s4(dense_codes[dense_base + axis + 1])?;
            packed[packed_base + byte_idx] = (low | (high << 4)) as i8;
        }
    }
    Ok(packed)
}

pub(crate) fn unpack_sq_s4_codes(
    packed_codes: &[i8],
    rows: usize,
    padded_dim: usize,
) -> AnnResult<Vec<i8>> {
    let row_bytes = sq_wmma_code_row_bytes(padded_dim, SqWmmaCodeLayout::PackedS4)?;
    let packed_len = rows.checked_mul(row_bytes).ok_or_else(|| {
        AnnError::log_index_error("SQ+WMMA packed-s4 code length overflows".to_string())
    })?;
    if packed_codes.len() != packed_len {
        return Err(AnnError::log_index_error(format!(
            "SQ+WMMA packed-s4 unpack shape mismatch: codes={} rows={rows} padded_dim={padded_dim} row_bytes={row_bytes}",
            packed_codes.len()
        )));
    }
    let mut dense = vec![0_i8; rows.saturating_mul(padded_dim)];
    for row in 0..rows {
        let dense_base = row * padded_dim;
        let packed_base = row * row_bytes;
        for byte_idx in 0..row_bytes {
            let byte = packed_codes[packed_base + byte_idx] as u8;
            dense[dense_base + byte_idx * 2] = unpack_signed_s4(byte);
            dense[dense_base + byte_idx * 2 + 1] = unpack_signed_s4(byte >> 4);
        }
    }
    Ok(dense)
}

pub(crate) fn validate_sq_int8_bits(bits: usize) -> AnnResult<()> {
    if !(SQ_INT8_MIN_BITS..=SQ_INT8_MAX_BITS).contains(&bits) {
        return Err(AnnError::log_index_error(format!(
            "SQ-int8 bits must be in [{SQ_INT8_MIN_BITS}, {SQ_INT8_MAX_BITS}], got {bits}"
        )));
    }
    Ok(())
}

#[inline]
fn sq_int8_max_code_for_bits(bits: usize) -> AnnResult<f32> {
    validate_sq_int8_bits(bits)?;
    Ok(((1_u32 << (bits - 1)) - 1) as f32)
}

pub(crate) fn quantize_sq_int8_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    k_align: usize,
) -> AnnResult<SqInt8Matrix> {
    quantize_sq_int8_matrix_with_bits(points_row_major, rows, dim, k_align, SQ_INT8_DEFAULT_BITS)
}

pub(crate) fn quantize_sq_int8_matrix_with_bits(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    k_align: usize,
    bits: usize,
) -> AnnResult<SqInt8Matrix> {
    let max_code = sq_int8_max_code_for_bits(bits)?;
    if points_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "SQ-int8 matrix shape mismatch: rows={rows} dim={dim} values={}",
            points_row_major.len()
        )));
    }
    let padded_dim = sq_int8_padded_dim(dim, k_align)?;
    let mut codes = vec![0_i8; rows.saturating_mul(padded_dim)];
    let mut scales = vec![1.0_f32; rows];
    let mut norms = vec![0.0_f32; rows];

    for row in 0..rows {
        let input = &points_row_major[row * dim..(row + 1) * dim];
        let output = &mut codes[row * padded_dim..(row + 1) * padded_dim];
        let (scale, norm) = quantize_sq_int8_row_with_max_code(input, output, max_code)?;
        scales[row] = scale;
        norms[row] = norm;
    }

    Ok(SqInt8Matrix {
        rows,
        dim,
        padded_dim,
        codes,
        scales,
        norms,
    })
}

pub(crate) fn quantize_native_u8_centered_int8_matrix(
    raw_row_major: &[u8],
    rows: usize,
    dim: usize,
    k_align: usize,
) -> AnnResult<SqInt8Matrix> {
    if raw_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Native uint8 SQ-int8 matrix shape mismatch: rows={rows} dim={dim} values={}",
            raw_row_major.len()
        )));
    }
    let padded_dim = sq_int8_padded_dim(dim, k_align)?;
    let mut codes = vec![0_i8; rows.saturating_mul(padded_dim)];
    let scales = vec![1.0_f32; rows];
    let mut norms = vec![0.0_f32; rows];

    for row in 0..rows {
        let raw = &raw_row_major[row * dim..(row + 1) * dim];
        let output = &mut codes[row * padded_dim..(row + 1) * padded_dim];
        for (dst, &value) in output.iter_mut().zip(raw.iter()) {
            let centered = i16::from(value) - 128;
            *dst = centered as i8;
            let centered_f32 = f32::from(centered);
            norms[row] += centered_f32 * centered_f32;
        }
    }

    Ok(SqInt8Matrix {
        rows,
        dim,
        padded_dim,
        codes,
        scales,
        norms,
    })
}

pub(crate) fn quantize_sq_int8_row(
    input: &[f32],
    output_padded: &mut [i8],
) -> AnnResult<(f32, f32)> {
    quantize_sq_int8_row_with_bits(input, output_padded, SQ_INT8_DEFAULT_BITS)
}

pub(crate) fn quantize_sq_int8_row_with_bits(
    input: &[f32],
    output_padded: &mut [i8],
    bits: usize,
) -> AnnResult<(f32, f32)> {
    let max_code = sq_int8_max_code_for_bits(bits)?;
    quantize_sq_int8_row_with_max_code(input, output_padded, max_code)
}

fn quantize_sq_int8_row_with_max_code(
    input: &[f32],
    output_padded: &mut [i8],
    max_code: f32,
) -> AnnResult<(f32, f32)> {
    if output_padded.len() < input.len() {
        return Err(AnnError::log_index_error(format!(
            "SQ-int8 row output too small: input_dim={} output_dim={}",
            input.len(),
            output_padded.len()
        )));
    }

    output_padded.fill(0);
    let mut max_abs = 0.0_f32;
    let mut norm = 0.0_f32;
    for &value in input {
        if !value.is_finite() {
            return Err(AnnError::log_index_error(format!(
                "SQ-int8 cannot quantize non-finite value {value}"
            )));
        }
        max_abs = max_abs.max(value.abs());
        norm += value * value;
    }

    if max_abs == 0.0 {
        return Ok((1.0, norm));
    }

    let scale = max_abs / max_code;
    for (dst, &value) in output_padded.iter_mut().zip(input) {
        let code = (value / scale).round().clamp(-max_code, max_code);
        *dst = code as i8;
    }
    Ok((scale, norm))
}

pub(crate) fn sq_wmma_sidecar_codes_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.codes", prefix.display()))
}

pub(crate) fn sq_wmma_sidecar_meta_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.meta", prefix.display()))
}

impl SqWmmaSidecarShardWriter {
    pub(crate) fn create(
        prefix: &Path,
        rows: usize,
        dim: usize,
        source_fbin_bytes: u64,
        bits: usize,
    ) -> AnnResult<Self> {
        validate_sq_int8_bits(bits)?;
        if rows > u32::MAX as usize {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA staged sidecar currently supports at most {} rows, got {rows}",
                u32::MAX
            )));
        }
        let total_start = Instant::now();
        let setup_start = Instant::now();
        if let Some(parent) = prefix
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to create SQ+WMMA staged sidecar parent '{}': {err}",
                    parent.display()
                ))
            })?;
        }
        let codes_path = sq_wmma_sidecar_codes_path(prefix);
        let meta_path = sq_wmma_sidecar_meta_path(prefix);
        let codes_writer = BufWriter::new(File::create(&codes_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create SQ+WMMA staged sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?);
        let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create SQ+WMMA staged sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?);
        meta_writer
            .write_all(&vec![0_u8; SQ_WMMA_SIDECAR_HEADER_BYTES])
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to reserve SQ+WMMA staged sidecar header '{}': {err}",
                    meta_path.display()
                ))
            })?;
        let padded_dim = sq_int8_padded_dim(dim, SQ_INT8_DEFAULT_K_ALIGN)?;
        let code_row_bytes = sq_wmma_code_row_bytes(padded_dim, SqWmmaCodeLayout::DenseI8)?;

        Ok(Self {
            prefix: prefix.to_path_buf(),
            rows,
            dim,
            padded_dim,
            bits,
            code_row_bytes,
            source_fbin_bytes,
            codes_writer,
            meta_writer,
            written_rows: 0,
            total_start,
            setup_wall: setup_start.elapsed(),
            code_output_wall: Duration::ZERO,
            meta_output_wall: Duration::ZERO,
            write_wall: Duration::ZERO,
        })
    }

    pub(crate) fn append_dense_i8_matrix(
        &mut self,
        row_start: usize,
        matrix: &SqInt8Matrix,
    ) -> AnnResult<()> {
        if row_start != self.written_rows {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA staged sidecar requires sequential writes: got row_start={} expected={}",
                row_start, self.written_rows
            )));
        }
        if matrix.dim != self.dim
            || matrix.padded_dim != self.padded_dim
            || matrix.codes.len() != matrix.rows.saturating_mul(self.code_row_bytes)
            || matrix.scales.len() != matrix.rows
            || matrix.norms.len() != matrix.rows
        {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA staged sidecar shard shape mismatch: matrix rows={} dim={} padded_dim={} codes={} scales={} norms={} expected dim={} padded_dim={} code_row_bytes={}",
                matrix.rows,
                matrix.dim,
                matrix.padded_dim,
                matrix.codes.len(),
                matrix.scales.len(),
                matrix.norms.len(),
                self.dim,
                self.padded_dim,
                self.code_row_bytes
            )));
        }
        if self.written_rows.saturating_add(matrix.rows) > self.rows {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA staged sidecar append would exceed rows: written={} append={} rows={}",
                self.written_rows, matrix.rows, self.rows
            )));
        }

        let write_start = Instant::now();
        let code_output_start = Instant::now();
        let code_bytes = unsafe {
            std::slice::from_raw_parts(matrix.codes.as_ptr().cast::<u8>(), matrix.codes.len())
        };
        self.codes_writer.write_all(code_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to append SQ+WMMA staged sidecar codes '{}': {err}",
                sq_wmma_sidecar_codes_path(&self.prefix).display()
            ))
        })?;
        self.code_output_wall += code_output_start.elapsed();

        let meta_output_start = Instant::now();
        for row in 0..matrix.rows {
            self.meta_writer
                .write_all(&matrix.norms[row].to_le_bytes())
                .and_then(|_| {
                    self.meta_writer
                        .write_all(&matrix.scales[row].to_le_bytes())
                })
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to append SQ+WMMA staged sidecar metadata '{}': {err}",
                        sq_wmma_sidecar_meta_path(&self.prefix).display()
                    ))
                })?;
        }
        self.meta_output_wall += meta_output_start.elapsed();
        self.write_wall += write_start.elapsed();
        self.written_rows += matrix.rows;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> AnnResult<SqWmmaSidecarCreateReport> {
        if self.written_rows != self.rows {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA staged sidecar incomplete: wrote {} rows expected {}",
                self.written_rows, self.rows
            )));
        }
        let prefix = self.prefix.clone();
        let rows = self.rows;
        let dim = self.dim;
        let padded_dim = self.padded_dim;
        let bits = self.bits;
        let code_row_bytes = self.code_row_bytes;
        let source_fbin_bytes = self.source_fbin_bytes;
        let setup_wall = self.setup_wall;
        let code_output_wall = self.code_output_wall;
        let meta_output_wall = self.meta_output_wall;
        let write_wall = self.write_wall;
        let flush_start = Instant::now();
        self.codes_writer.flush().map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to flush SQ+WMMA staged sidecar codes '{}': {err}",
                sq_wmma_sidecar_codes_path(&prefix).display()
            ))
        })?;
        self.meta_writer.flush().map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to flush SQ+WMMA staged sidecar meta '{}': {err}",
                sq_wmma_sidecar_meta_path(&prefix).display()
            ))
        })?;
        let flush_wall = flush_start.elapsed();
        drop(self.codes_writer);
        drop(self.meta_writer);

        let codes_bytes = rows.checked_mul(code_row_bytes).ok_or_else(|| {
            AnnError::log_index_error(
                "SQ+WMMA staged sidecar codes byte count overflows".to_string(),
            )
        })?;
        let meta_rows_bytes = rows
            .checked_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "SQ+WMMA staged sidecar metadata byte count overflows".to_string(),
                )
            })?;
        let header = SqWmmaSidecarHeader {
            rows,
            dim,
            padded_dim,
            k_align: SQ_INT8_DEFAULT_K_ALIGN,
            bits,
            code_layout: SqWmmaCodeLayout::DenseI8,
            code_row_bytes,
            source_fbin_bytes,
            source_rows: rows,
            source_dim: dim,
            source_fingerprint: sq_wmma_metadata_source_fingerprint(
                rows,
                dim,
                source_fbin_bytes,
                bits,
            ),
            codes_bytes,
            meta_rows_bytes,
        };
        let meta_path = sq_wmma_sidecar_meta_path(&prefix);
        let header_write_start = Instant::now();
        let mut meta = File::options()
            .write(true)
            .open(&meta_path)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to reopen SQ+WMMA staged sidecar meta '{}': {err}",
                    meta_path.display()
                ))
            })?;
        meta.seek(SeekFrom::Start(0)).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to seek SQ+WMMA staged sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
        meta.write_all(&encode_sq_wmma_sidecar_header(&header)?)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to write SQ+WMMA staged sidecar header '{}': {err}",
                    meta_path.display()
                ))
            })?;
        let header_write_wall = header_write_start.elapsed();

        Ok(SqWmmaSidecarCreateReport {
            header,
            total_wall: self.total_start.elapsed(),
            setup_wall,
            chunk_alloc_wall: Duration::ZERO,
            raw_read_wall: Duration::ZERO,
            source_hash_wall: Duration::ZERO,
            quantize_wall: Duration::ZERO,
            code_output_wall,
            meta_output_wall,
            flush_wall,
            header_write_wall,
            write_wall,
        })
    }
}

impl SqWmmaResidentSidecarBuilder {
    pub(crate) fn create(
        prefix: &Path,
        rows: usize,
        dim: usize,
        source_fbin_bytes: u64,
        bits: usize,
    ) -> AnnResult<Self> {
        validate_sq_int8_bits(bits)?;
        if rows > u32::MAX as usize {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA resident sidecar currently supports at most {} rows, got {rows}",
                u32::MAX
            )));
        }
        let total_start = Instant::now();
        let setup_start = Instant::now();
        let padded_dim = sq_int8_padded_dim(dim, SQ_INT8_DEFAULT_K_ALIGN)?;
        let code_row_bytes = sq_wmma_code_row_bytes(padded_dim, SqWmmaCodeLayout::DenseI8)?;
        let codes_bytes = rows.checked_mul(code_row_bytes).ok_or_else(|| {
            AnnError::log_index_error(
                "SQ+WMMA resident sidecar codes byte count overflows".to_string(),
            )
        })?;

        let mut codes = Vec::new();
        codes.try_reserve_exact(codes_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to reserve SQ+WMMA resident sidecar codes bytes={codes_bytes}: {err}"
            ))
        })?;
        let mut norms = Vec::new();
        norms.try_reserve_exact(rows).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to reserve SQ+WMMA resident sidecar norms rows={rows}: {err}"
            ))
        })?;
        let mut scales = Vec::new();
        scales.try_reserve_exact(rows).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to reserve SQ+WMMA resident sidecar scales rows={rows}: {err}"
            ))
        })?;

        Ok(Self {
            prefix: prefix.to_path_buf(),
            rows,
            dim,
            padded_dim,
            bits,
            code_row_bytes,
            source_fbin_bytes,
            codes,
            norms,
            scales,
            written_rows: 0,
            total_start,
            setup_wall: setup_start.elapsed(),
            code_output_wall: Duration::ZERO,
            meta_output_wall: Duration::ZERO,
            write_wall: Duration::ZERO,
        })
    }

    pub(crate) fn append_dense_i8_matrix(
        &mut self,
        row_start: usize,
        matrix: &SqInt8Matrix,
    ) -> AnnResult<()> {
        if row_start != self.written_rows {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA resident sidecar requires sequential writes: got row_start={} expected={}",
                row_start, self.written_rows
            )));
        }
        if matrix.dim != self.dim
            || matrix.padded_dim != self.padded_dim
            || matrix.codes.len() != matrix.rows.saturating_mul(self.code_row_bytes)
            || matrix.scales.len() != matrix.rows
            || matrix.norms.len() != matrix.rows
        {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA resident sidecar shard shape mismatch: matrix rows={} dim={} padded_dim={} codes={} scales={} norms={} expected dim={} padded_dim={} code_row_bytes={}",
                matrix.rows,
                matrix.dim,
                matrix.padded_dim,
                matrix.codes.len(),
                matrix.scales.len(),
                matrix.norms.len(),
                self.dim,
                self.padded_dim,
                self.code_row_bytes
            )));
        }
        if self.written_rows.saturating_add(matrix.rows) > self.rows {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA resident sidecar append would exceed rows: written={} append={} rows={}",
                self.written_rows, matrix.rows, self.rows
            )));
        }

        let write_start = Instant::now();
        let code_output_start = Instant::now();
        self.codes.extend_from_slice(&matrix.codes);
        self.code_output_wall += code_output_start.elapsed();

        let meta_output_start = Instant::now();
        self.norms.extend_from_slice(&matrix.norms);
        self.scales.extend_from_slice(&matrix.scales);
        self.meta_output_wall += meta_output_start.elapsed();
        self.write_wall += write_start.elapsed();
        self.written_rows += matrix.rows;
        Ok(())
    }

    pub(crate) fn finish(self) -> AnnResult<SqWmmaSidecarCreateArtifact> {
        if self.written_rows != self.rows {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA resident sidecar incomplete: wrote {} rows expected {}",
                self.written_rows, self.rows
            )));
        }
        let codes_bytes = self.rows.checked_mul(self.code_row_bytes).ok_or_else(|| {
            AnnError::log_index_error(
                "SQ+WMMA resident sidecar codes byte count overflows".to_string(),
            )
        })?;
        let meta_rows_bytes = self
            .rows
            .checked_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "SQ+WMMA resident sidecar metadata byte count overflows".to_string(),
                )
            })?;
        if self.codes.len() != codes_bytes
            || self.norms.len() != self.rows
            || self.scales.len() != self.rows
        {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA resident sidecar final shape mismatch: codes={} expected={} norms={} scales={} rows={}",
                self.codes.len(),
                codes_bytes,
                self.norms.len(),
                self.scales.len(),
                self.rows
            )));
        }

        let header = SqWmmaSidecarHeader {
            rows: self.rows,
            dim: self.dim,
            padded_dim: self.padded_dim,
            k_align: SQ_INT8_DEFAULT_K_ALIGN,
            bits: self.bits,
            code_layout: SqWmmaCodeLayout::DenseI8,
            code_row_bytes: self.code_row_bytes,
            source_fbin_bytes: self.source_fbin_bytes,
            source_rows: self.rows,
            source_dim: self.dim,
            source_fingerprint: sq_wmma_metadata_source_fingerprint(
                self.rows,
                self.dim,
                self.source_fbin_bytes,
                self.bits,
            ),
            codes_bytes,
            meta_rows_bytes,
        };
        let report = SqWmmaSidecarCreateReport {
            header: header.clone(),
            total_wall: self.total_start.elapsed(),
            setup_wall: self.setup_wall,
            chunk_alloc_wall: Duration::ZERO,
            raw_read_wall: Duration::ZERO,
            source_hash_wall: Duration::ZERO,
            quantize_wall: Duration::ZERO,
            code_output_wall: self.code_output_wall,
            meta_output_wall: self.meta_output_wall,
            flush_wall: Duration::ZERO,
            header_write_wall: Duration::ZERO,
            write_wall: self.write_wall,
        };
        Ok(SqWmmaSidecarCreateArtifact {
            report,
            runtime_sidecar: Some(SqWmmaSidecar {
                prefix: self.prefix,
                header,
                storage: SqWmmaSidecarStorage::Resident {
                    codes: self.codes,
                    norms: self.norms,
                    scales: self.scales,
                    preload_wall: Duration::ZERO,
                },
            }),
        })
    }
}

pub(crate) fn quantize_sq8_and_assign_global_d0_reference(
    request: SqGpuD0QuantizeAssignRequest<'_>,
) -> AnnResult<SqGpuD0QuantizeAssignResult> {
    let rows = request.global_ids.len();
    let dim = request.dim;
    let leader_count = request.global_d0_leader_ids.len();
    if rows == 0 {
        return Err(AnnError::log_index_error(
            "SQ GPU-D0 staging requires at least one input row".to_string(),
        ));
    }
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "SQ GPU-D0 staging requires dim > 0".to_string(),
        ));
    }
    if leader_count == 0 {
        return Err(AnnError::log_index_error(
            "SQ GPU-D0 staging requires at least one global D0 leader".to_string(),
        ));
    }
    if request.fanout_top == 0 || request.fanout_top > leader_count || request.fanout_top > 32 {
        return Err(AnnError::log_index_error(format!(
            "SQ GPU-D0 staging fanout must be in 1..=min(32, leaders), got fanout={} leaders={leader_count}",
            request.fanout_top
        )));
    }
    if request.raw_vectors_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "SQ GPU-D0 staging raw vector shape mismatch: rows={rows} dim={dim} values={}",
            request.raw_vectors_row_major.len()
        )));
    }
    if request.global_d0_leader_vectors_row_major.len() != leader_count.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "SQ GPU-D0 staging leader vector shape mismatch: leaders={leader_count} dim={dim} values={}",
            request.global_d0_leader_vectors_row_major.len()
        )));
    }

    if let Some(rotated) = request.rotated_residual {
        if request.leader_payload.is_some()
            || rotated.centroids_row_major != request.global_d0_leader_vectors_row_major
        {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 D0 requires the coarse centers as leaders and no cached leader payload".to_string(),
            ));
        }
        let quantize_start = Instant::now();
        let encoded = quantize_sq8_rotated_residual_matrix(
            request.raw_vectors_row_major,
            rows,
            dim,
            rotated.centroids_row_major,
            rotated.seed,
        )?;
        let quantize_wall = quantize_start.elapsed();
        let assign_start = Instant::now();
        let mut fragments = Vec::with_capacity(rows.saturating_mul(request.fanout_top));
        for row in 0..rows {
            let own_center = encoded.centroid_ids[row] as usize;
            let own = &encoded.rotated_centroids[own_center * dim..(own_center + 1) * dim];
            let code = encoded.residuals.code_row(row);
            let scale = encoded.residuals.scales[row];
            let mut candidates = Vec::with_capacity(leader_count);
            for leader in 0..leader_count {
                let center = &encoded.rotated_centroids[leader * dim..(leader + 1) * dim];
                let mut dist = 0.0_f32;
                for axis in 0..dim {
                    let delta = own[axis] + scale * f32::from(code[axis]) - center[axis];
                    dist += delta * delta;
                }
                candidates.push((leader, dist));
            }
            candidates.sort_unstable_by(|left, right| {
                super::stable_distance_order(
                    left.1,
                    request.global_d0_leader_ids[left.0],
                    right.1,
                    request.global_d0_leader_ids[right.0],
                )
            });
            for &(leader, _) in candidates.iter().take(request.fanout_top) {
                fragments.push(D0AssignmentFragment {
                    child_id: leader,
                    point_id: request.global_ids[row],
                });
            }
        }
        let assign_wall = assign_start.elapsed();
        return Ok(SqGpuD0QuantizeAssignResult {
            matrix: encoded.residuals,
            centroid_ids: Some(encoded.centroid_ids),
            fragments,
            profile: SqGpuD0StagingProfile {
                host_rotated_quantize_wall: quantize_wall,
                gpu_d0_assign_wall: assign_wall,
                d0_assign_kernel_requested: "cpu-rotated-residual-reference",
                d0_assign_kernel_mode: "cpu-rotated-residual-reference",
                d0_assign_rows: rows,
                d0_assign_leaders: leader_count,
                d0_assign_pair_count: rows.saturating_mul(leader_count),
                d0_assign_fanout: request.fanout_top,
                ..SqGpuD0StagingProfile::default()
            },
        });
    }

    let quantize_start = Instant::now();
    let matrix = quantize_sq_int8_matrix(
        request.raw_vectors_row_major,
        rows,
        dim,
        SQ_INT8_DEFAULT_K_ALIGN,
    )?;
    let gpu_quantize_wall = quantize_start.elapsed();

    let leader_prepare_start = Instant::now();
    let owned_leader_payload;
    let leader_payload = if let Some(leader_payload) = request.leader_payload {
        if leader_payload.rows != leader_count
            || leader_payload.dim != dim
            || leader_payload.padded_dim != matrix.padded_dim
            || leader_payload.scales.len() != leader_count
            || leader_payload.norms.len() != leader_count
            || leader_payload.codes.len() != leader_count.saturating_mul(matrix.padded_dim)
        {
            return Err(AnnError::log_index_error(format!(
                "SQ GPU-D0 staging leader payload shape mismatch: rows={} dim={} padded_dim={} codes={} expected rows={leader_count} dim={dim} padded_dim={}",
                leader_payload.rows,
                leader_payload.dim,
                leader_payload.padded_dim,
                leader_payload.codes.len(),
                matrix.padded_dim
            )));
        }
        leader_payload
    } else {
        owned_leader_payload = quantize_sq_int8_matrix(
            request.global_d0_leader_vectors_row_major,
            leader_count,
            dim,
            SQ_INT8_DEFAULT_K_ALIGN,
        )?;
        &owned_leader_payload
    };
    let leader_prepare_wall = leader_prepare_start.elapsed();

    let assign_start = Instant::now();
    let mut fragments = Vec::with_capacity(rows.saturating_mul(request.fanout_top));
    for row in 0..rows {
        let mut candidates = Vec::with_capacity(leader_count);
        for leader_idx in 0..leader_count {
            let dist = sq_int8_approx_l2(
                matrix.code_row(row),
                matrix.scales[row],
                matrix.norms[row],
                leader_payload.code_row(leader_idx),
                leader_payload.scales[leader_idx],
                leader_payload.norms[leader_idx],
            );
            candidates.push((leader_idx, dist));
        }
        candidates.sort_unstable_by(|left, right| {
            super::stable_distance_order(
                left.1,
                request.global_d0_leader_ids[left.0],
                right.1,
                request.global_d0_leader_ids[right.0],
            )
        });
        for &(leader_idx, _) in candidates.iter().take(request.fanout_top) {
            fragments.push(D0AssignmentFragment {
                child_id: leader_idx,
                point_id: request.global_ids[row],
            });
        }
    }
    let gpu_d0_assign_wall = assign_start.elapsed();

    Ok(SqGpuD0QuantizeAssignResult {
        matrix,
        centroid_ids: None,
        fragments,
        profile: SqGpuD0StagingProfile {
            raw_h2d_wall: Duration::ZERO,
            host_rotated_quantize_wall: Duration::ZERO,
            gpu_quantize_wall,
            qcode_d2h_wall: Duration::ZERO,
            gpu_d0_assign_wall,
            assignment_fragment_d2h_wall: Duration::ZERO,
            leader_prepare_wall,
            leader_h2d_wall: Duration::ZERO,
            leader_quantize_wall: Duration::ZERO,
            leader_code_bytes: leader_count.saturating_mul(leader_payload.padded_dim),
            d0_assign_kernel_requested: "cpu-reference",
            d0_assign_kernel_mode: "cpu-reference",
            d0_assign_rows: rows,
            d0_assign_leaders: leader_count,
            d0_assign_pair_count: rows.saturating_mul(leader_count),
            d0_assign_fanout: request.fanout_top,
            d0_assign_row_tile: 1,
            d0_assign_row_chunk: rows,
            d0_assign_leader_tile: leader_count,
            d0_assign_partial_r: request.fanout_top,
            d0_assign_leader_tile_count: 1,
            d0_assign_partial_bytes: 0,
            d0_assign_distance_kernel_wall: gpu_d0_assign_wall,
            d0_assign_merge_kernel_wall: Duration::ZERO,
        },
    })
}

pub(crate) fn create_sq_wmma_sidecar_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    source_fbin_bytes: u64,
) -> AnnResult<SqWmmaSidecarCreateReport> {
    create_sq_wmma_sidecar_from_point_store_with_bits(
        dataset,
        prefix,
        source_fbin_bytes,
        SQ_INT8_DEFAULT_BITS,
    )
}

pub(crate) fn create_sq_wmma_sidecar_from_point_store_with_bits(
    dataset: &dyn PointStore,
    prefix: &Path,
    source_fbin_bytes: u64,
    bits: usize,
) -> AnnResult<SqWmmaSidecarCreateReport> {
    create_sq_wmma_sidecar_from_point_store_with_bits_and_resident_prefix(
        dataset,
        prefix,
        source_fbin_bytes,
        bits,
        0,
    )
    .map(|artifact| artifact.report)
}

pub(crate) fn create_sq_wmma_sidecar_from_point_store_with_resident_prefix(
    dataset: &dyn PointStore,
    prefix: &Path,
    source_fbin_bytes: u64,
    resident_rows: usize,
) -> AnnResult<SqWmmaSidecarCreateArtifact> {
    create_sq_wmma_sidecar_from_point_store_with_bits_and_resident_prefix(
        dataset,
        prefix,
        source_fbin_bytes,
        SQ_INT8_DEFAULT_BITS,
        resident_rows,
    )
}

pub(crate) fn create_sq_wmma_sidecar_from_point_store_with_bits_and_resident_prefix(
    dataset: &dyn PointStore,
    prefix: &Path,
    source_fbin_bytes: u64,
    bits: usize,
    resident_rows: usize,
) -> AnnResult<SqWmmaSidecarCreateArtifact> {
    create_sq_wmma_sidecar_from_point_store_with_bits_layout(
        dataset,
        prefix,
        source_fbin_bytes,
        bits,
        SqWmmaCodeLayout::DenseI8,
        resident_rows,
    )
}

pub(crate) fn create_sq_wmma_sidecar_from_point_store_with_bits_layout(
    dataset: &dyn PointStore,
    prefix: &Path,
    source_fbin_bytes: u64,
    bits: usize,
    code_layout: SqWmmaCodeLayout,
    resident_rows: usize,
) -> AnnResult<SqWmmaSidecarCreateArtifact> {
    let dim = dataset.dim();
    create_sq_wmma_sidecar_from_rows_with_bits_layout(
        dataset.len(),
        dim,
        prefix,
        source_fbin_bytes,
        bits,
        code_layout,
        resident_rows,
        |row_start, count, chunk_alloc_wall, raw_read_wall, quantize_wall| {
            let chunk_alloc_start = Instant::now();
            let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
            *chunk_alloc_wall += chunk_alloc_start.elapsed();
            let raw_read_start = Instant::now();
            dataset.read_range_into(row_start as u32, count, &mut chunk)?;
            *raw_read_wall += raw_read_start.elapsed();

            let quantize_start = Instant::now();
            let matrix = quantize_sq_int8_matrix_with_bits(
                &chunk,
                count,
                dim,
                SQ_INT8_DEFAULT_K_ALIGN,
                bits,
            )?;
            *quantize_wall += quantize_start.elapsed();
            Ok(matrix)
        },
    )
}

pub(crate) fn create_sq_wmma_sidecar_from_u8_point_store_native_i8_with_resident_prefix(
    dataset: &U8PointStore,
    prefix: &Path,
    source_fbin_bytes: u64,
    resident_rows: usize,
) -> AnnResult<SqWmmaSidecarCreateArtifact> {
    let dim = dataset.dim();
    create_sq_wmma_sidecar_from_rows_with_bits_layout(
        dataset.len(),
        dim,
        prefix,
        source_fbin_bytes,
        SQ_INT8_DEFAULT_BITS,
        SqWmmaCodeLayout::DenseI8,
        resident_rows,
        |row_start, count, chunk_alloc_wall, raw_read_wall, quantize_wall| {
            let chunk_alloc_start = Instant::now();
            let mut raw = vec![0_u8; count.saturating_mul(dim)];
            *chunk_alloc_wall += chunk_alloc_start.elapsed();
            let raw_read_start = Instant::now();
            dataset.read_raw_range_into(row_start as u32, count, &mut raw)?;
            *raw_read_wall += raw_read_start.elapsed();

            let quantize_start = Instant::now();
            let matrix =
                quantize_native_u8_centered_int8_matrix(&raw, count, dim, SQ_INT8_DEFAULT_K_ALIGN)?;
            *quantize_wall += quantize_start.elapsed();
            Ok(matrix)
        },
    )
}

fn create_sq_wmma_sidecar_from_rows_with_bits_layout<F>(
    rows: usize,
    dim: usize,
    prefix: &Path,
    source_fbin_bytes: u64,
    bits: usize,
    code_layout: SqWmmaCodeLayout,
    resident_rows: usize,
    mut build_matrix: F,
) -> AnnResult<SqWmmaSidecarCreateArtifact>
where
    F: FnMut(usize, usize, &mut Duration, &mut Duration, &mut Duration) -> AnnResult<SqInt8Matrix>,
{
    validate_sq_int8_bits(bits)?;
    if code_layout == SqWmmaCodeLayout::PackedS4 && bits != 4 {
        return Err(AnnError::log_index_error(format!(
            "SQ+WMMA packed-s4 sidecar requires bits=4, got {bits}"
        )));
    }
    let total_start = Instant::now();
    let resident_rows = resident_rows.min(rows);
    if rows > u32::MAX as usize {
        return Err(AnnError::log_index_error(format!(
            "SQ+WMMA sidecar currently supports at most {} rows, got {rows}",
            u32::MAX
        )));
    }
    let padded_dim = sq_int8_padded_dim(dim, SQ_INT8_DEFAULT_K_ALIGN)?;
    let code_row_bytes = sq_wmma_code_row_bytes(padded_dim, code_layout)?;
    let codes_bytes = rows.checked_mul(code_row_bytes).ok_or_else(|| {
        AnnError::log_index_error("SQ+WMMA sidecar codes byte count overflows".to_string())
    })?;
    let meta_rows_bytes = rows
        .checked_mul(SQ_WMMA_SIDECAR_META_ROW_BYTES)
        .ok_or_else(|| {
            AnnError::log_index_error("SQ+WMMA sidecar metadata byte count overflows".to_string())
        })?;
    let setup_start = Instant::now();
    if let Some(parent) = prefix
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create SQ+WMMA sidecar parent '{}': {err}",
                parent.display()
            ))
        })?;
    }
    let codes_path = sq_wmma_sidecar_codes_path(prefix);
    let meta_path = sq_wmma_sidecar_meta_path(prefix);
    let mut codes_writer = BufWriter::new(File::create(&codes_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create SQ+WMMA sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?);
    let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create SQ+WMMA sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?);
    meta_writer
        .write_all(&vec![0_u8; SQ_WMMA_SIDECAR_HEADER_BYTES])
        .map_err(|err| {
            AnnError::log_index_error(format!("Failed to reserve SQ+WMMA header: {err}"))
        })?;
    let setup_wall = setup_start.elapsed();

    let hash = sq_wmma_metadata_source_fingerprint(rows, dim, source_fbin_bytes, bits);
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut code_output_wall = Duration::ZERO;
    let mut meta_output_wall = Duration::ZERO;
    let mut flush_wall = Duration::ZERO;
    let mut header_write_wall = Duration::ZERO;
    let mut write_wall = Duration::ZERO;
    let resident_alloc_start = Instant::now();
    let mut resident_codes = if resident_rows > 0 {
        Some(vec![0_i8; resident_rows.saturating_mul(code_row_bytes)])
    } else {
        None
    };
    let mut resident_norms = if resident_rows > 0 {
        Some(vec![0.0_f32; resident_rows])
    } else {
        None
    };
    let mut resident_scales = if resident_rows > 0 {
        Some(vec![0.0_f32; resident_rows])
    } else {
        None
    };
    chunk_alloc_wall += resident_alloc_start.elapsed();
    let mut row_start = 0usize;
    while row_start < rows {
        let count = (rows - row_start).min(SQ_WMMA_SIDECAR_CREATE_CHUNK_ROWS);
        let matrix = build_matrix(
            row_start,
            count,
            &mut chunk_alloc_wall,
            &mut raw_read_wall,
            &mut quantize_wall,
        )?;
        if matrix.rows != count || matrix.dim != dim || matrix.padded_dim != padded_dim {
            return Err(AnnError::log_index_error(format!(
                "SQ+WMMA sidecar chunk shape mismatch: got rows={} dim={} padded_dim={} expected rows={count} dim={dim} padded_dim={padded_dim}",
                matrix.rows, matrix.dim, matrix.padded_dim
            )));
        }
        let chunk_codes = match code_layout {
            SqWmmaCodeLayout::DenseI8 => matrix.codes.clone(),
            SqWmmaCodeLayout::PackedS4 => pack_sq_s4_codes(&matrix.codes, count, padded_dim)?,
        };

        if row_start < resident_rows {
            let resident_count = resident_rows.saturating_sub(row_start).min(count);
            let resident_code_start = row_start.saturating_mul(code_row_bytes);
            let resident_code_len = resident_count.saturating_mul(code_row_bytes);
            if let Some(codes) = resident_codes.as_mut() {
                codes[resident_code_start..resident_code_start + resident_code_len]
                    .copy_from_slice(&chunk_codes[..resident_code_len]);
            }
            if let Some(norms) = resident_norms.as_mut() {
                norms[row_start..row_start + resident_count]
                    .copy_from_slice(&matrix.norms[..resident_count]);
            }
            if let Some(scales) = resident_scales.as_mut() {
                scales[row_start..row_start + resident_count]
                    .copy_from_slice(&matrix.scales[..resident_count]);
            }
        }

        let write_start = Instant::now();
        let code_bytes = unsafe {
            std::slice::from_raw_parts(chunk_codes.as_ptr().cast::<u8>(), chunk_codes.len())
        };
        let code_output_start = Instant::now();
        codes_writer.write_all(code_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write SQ+WMMA sidecar codes '{}': {err}",
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
                        "Failed to write SQ+WMMA sidecar norm '{}': {err}",
                        meta_path.display()
                    ))
                })?;
            meta_writer
                .write_all(&matrix.scales[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write SQ+WMMA sidecar scale '{}': {err}",
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
            "Failed to flush SQ+WMMA sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?;
    meta_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush SQ+WMMA sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    flush_wall += flush_start.elapsed();
    drop(codes_writer);
    drop(meta_writer);

    let header = SqWmmaSidecarHeader {
        rows,
        dim,
        padded_dim,
        k_align: SQ_INT8_DEFAULT_K_ALIGN,
        bits,
        code_layout,
        code_row_bytes,
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
                "Failed to reopen SQ+WMMA sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
    meta.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek SQ+WMMA sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    meta.write_all(&encode_sq_wmma_sidecar_header(&header)?)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write SQ+WMMA sidecar header '{}': {err}",
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
        let scales = resident_scales.expect("resident scales allocated when resident_rows > 0");
        let storage = if resident_rows >= rows {
            SqWmmaSidecarStorage::Resident {
                codes,
                norms,
                scales,
                preload_wall: Duration::ZERO,
            }
        } else {
            let codes_file = File::open(&codes_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open SQ+WMMA sidecar codes '{}': {err}",
                    codes_path.display()
                ))
            })?;
            let meta_file = File::open(&meta_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open SQ+WMMA sidecar meta '{}': {err}",
                    meta_path.display()
                ))
            })?;
            SqWmmaSidecarStorage::PartialResident {
                codes_file: SqWmmaSidecarReadFile::Buffered(codes_file),
                meta_file: SqWmmaSidecarReadFile::Buffered(meta_file),
                resident_rows,
                codes,
                norms,
                scales,
                preload_wall: Duration::ZERO,
            }
        };
        Some(SqWmmaSidecar {
            prefix: prefix.to_path_buf(),
            header: header.clone(),
            storage,
        })
    };

    Ok(SqWmmaSidecarCreateArtifact {
        report: SqWmmaSidecarCreateReport {
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

fn encode_sq_wmma_sidecar_header(
    header: &SqWmmaSidecarHeader,
) -> AnnResult<[u8; SQ_WMMA_SIDECAR_HEADER_BYTES]> {
    let mut out = [0_u8; SQ_WMMA_SIDECAR_HEADER_BYTES];
    let mut pos = 0usize;
    write_header_bytes(&mut out, &mut pos, SQ_WMMA_SIDECAR_MAGIC)?;
    write_header_u32(&mut out, &mut pos, SQ_WMMA_SIDECAR_VERSION)?;
    write_header_u32(&mut out, &mut pos, SQ_WMMA_SIDECAR_HEADER_BYTES as u32)?;
    write_header_u64(&mut out, &mut pos, header.rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.padded_dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.k_align as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fbin_bytes)?;
    write_header_u64(&mut out, &mut pos, header.source_rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fingerprint)?;
    write_header_u64(&mut out, &mut pos, header.codes_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.meta_rows_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.bits as u64)?;
    write_header_u64(&mut out, &mut pos, header.code_layout.header_code())?;
    write_header_u64(&mut out, &mut pos, header.code_row_bytes as u64)?;
    Ok(out)
}

fn read_sq_wmma_sidecar_header(file: &mut File) -> AnnResult<SqWmmaSidecarHeader> {
    let mut buf = [0_u8; SQ_WMMA_SIDECAR_HEADER_BYTES];
    file.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!("Failed to seek SQ+WMMA sidecar header: {err}"))
    })?;
    file.read_exact(&mut buf).map_err(|err| {
        AnnError::log_index_error(format!("Failed to read SQ+WMMA sidecar header: {err}"))
    })?;
    let mut pos = 0usize;
    let magic = read_header_bytes(&buf, &mut pos, SQ_WMMA_SIDECAR_MAGIC.len())?;
    if magic != SQ_WMMA_SIDECAR_MAGIC {
        return Err(AnnError::log_index_error(
            "SQ+WMMA sidecar magic mismatch".to_string(),
        ));
    }
    let version = read_header_u32(&buf, &mut pos)?;
    if version != SQ_WMMA_SIDECAR_VERSION {
        return Err(AnnError::log_index_error(format!(
            "Unsupported SQ+WMMA sidecar version {version}"
        )));
    }
    let header_bytes = read_header_u32(&buf, &mut pos)? as usize;
    if header_bytes != SQ_WMMA_SIDECAR_HEADER_BYTES {
        return Err(AnnError::log_index_error(format!(
            "Unsupported SQ+WMMA sidecar header size {header_bytes}"
        )));
    }
    let rows = read_header_u64(&buf, &mut pos)? as usize;
    let dim = read_header_u64(&buf, &mut pos)? as usize;
    let padded_dim = read_header_u64(&buf, &mut pos)? as usize;
    let k_align = read_header_u64(&buf, &mut pos)? as usize;
    let source_fbin_bytes = read_header_u64(&buf, &mut pos)?;
    let source_rows = read_header_u64(&buf, &mut pos)? as usize;
    let source_dim = read_header_u64(&buf, &mut pos)? as usize;
    let source_fingerprint = read_header_u64(&buf, &mut pos)?;
    let codes_bytes = read_header_u64(&buf, &mut pos)? as usize;
    let meta_rows_bytes = read_header_u64(&buf, &mut pos)? as usize;
    let bits = read_header_u64(&buf, &mut pos)? as usize;
    let code_layout = SqWmmaCodeLayout::from_header_code(read_header_u64(&buf, &mut pos)?)?;
    let code_row_bytes = read_header_u64(&buf, &mut pos)? as usize;
    let bits = if bits == 0 {
        SQ_INT8_DEFAULT_BITS
    } else {
        bits
    };
    let code_row_bytes = if code_row_bytes == 0 {
        sq_wmma_code_row_bytes(padded_dim, code_layout)?
    } else {
        code_row_bytes
    };
    Ok(SqWmmaSidecarHeader {
        rows,
        dim,
        padded_dim,
        k_align,
        bits,
        code_layout,
        code_row_bytes,
        source_fbin_bytes,
        source_rows,
        source_dim,
        source_fingerprint,
        codes_bytes,
        meta_rows_bytes,
    })
}

fn write_header_bytes(out: &mut [u8], pos: &mut usize, bytes: &[u8]) -> AnnResult<()> {
    let end = pos.checked_add(bytes.len()).ok_or_else(|| {
        AnnError::log_index_error("SQ+WMMA sidecar header offset overflows".to_string())
    })?;
    if end > out.len() {
        return Err(AnnError::log_index_error(
            "SQ+WMMA sidecar header buffer too small".to_string(),
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

fn read_header_bytes<'a>(buf: &'a [u8], pos: &mut usize, len: usize) -> AnnResult<&'a [u8]> {
    let end = pos.checked_add(len).ok_or_else(|| {
        AnnError::log_index_error("SQ+WMMA sidecar header offset overflows".to_string())
    })?;
    if end > buf.len() {
        return Err(AnnError::log_index_error(
            "SQ+WMMA sidecar header is truncated".to_string(),
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

fn fnv1a_update_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn sq_wmma_metadata_source_fingerprint(
    rows: usize,
    dim: usize,
    source_fbin_bytes: u64,
    bits: usize,
) -> u64 {
    let mut hash = fnv1a_update_u64(FNV_OFFSET_BASIS, rows as u64);
    hash = fnv1a_update_u64(hash, dim as u64);
    hash = fnv1a_update_u64(hash, SQ_INT8_DEFAULT_K_ALIGN as u64);
    hash = fnv1a_update_u64(hash, bits as u64);
    fnv1a_update_u64(hash, source_fbin_bytes)
}

pub(crate) fn sq_int8_dot(left: &[i8], right: &[i8]) -> AnnResult<i32> {
    if left.len() != right.len() {
        return Err(AnnError::log_index_error(format!(
            "SQ-int8 dot shape mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    Ok(sq_int8_dot_scalar_unchecked(left, right))
}

pub(crate) fn sq_int8_dot_with_kernel(
    left: &[i8],
    right: &[i8],
    kernel: SqInt8CpuKernel,
) -> AnnResult<i32> {
    if left.len() != right.len() {
        return Err(AnnError::log_index_error(format!(
            "SQ-int8 dot shape mismatch: left={} right={}",
            left.len(),
            right.len()
        )));
    }
    match resolve_sq_int8_cpu_kernel(kernel)? {
        SqInt8CpuKernel::Auto => unreachable!("auto kernel must resolve before dispatch"),
        resolved => Ok(sq_int8_dot_with_resolved_kernel_unchecked(
            left, right, resolved,
        )),
    }
}

#[inline]
pub(crate) fn sq_int8_dot_scalar_unchecked(left: &[i8], right: &[i8]) -> i32 {
    left.iter()
        .zip(right)
        .map(|(&lhs, &rhs)| i32::from(lhs) * i32::from(rhs))
        .sum()
}

#[inline]
pub(crate) fn sq_int8_dot_with_resolved_kernel_unchecked(
    left: &[i8],
    right: &[i8],
    kernel: SqInt8CpuKernel,
) -> i32 {
    sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(left, right, kernel, None)
}

#[inline]
pub(crate) fn sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(
    left: &[i8],
    right: &[i8],
    kernel: SqInt8CpuKernel,
    right_sum_for_full_len: Option<i32>,
) -> i32 {
    match kernel {
        SqInt8CpuKernel::Auto => unreachable!("auto kernel must resolve before dispatch"),
        SqInt8CpuKernel::Scalar => sq_int8_dot_scalar_unchecked(left, right),
        SqInt8CpuKernel::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                sq_int8_dot_avx2_unchecked(left, right)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                sq_int8_dot_scalar_unchecked(left, right)
            }
        }
        SqInt8CpuKernel::Avx512Vnni => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                sq_int8_dot_avx512_vnni_unchecked(left, right, right_sum_for_full_len)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                sq_int8_dot_scalar_unchecked(left, right)
            }
        }
    }
}

#[inline]
pub(crate) fn sq_int8_dot4_with_resolved_kernel_and_right_sums_unchecked(
    left: &[i8],
    rights: [&[i8]; 4],
    kernel: SqInt8CpuKernel,
    right_sums_for_full_len: Option<[i32; 4]>,
) -> [i32; 4] {
    match kernel {
        SqInt8CpuKernel::Avx512Vnni => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                sq_int8_dot4_avx512_vnni_unchecked(left, rights, right_sums_for_full_len)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                [
                    sq_int8_dot_scalar_unchecked(left, rights[0]),
                    sq_int8_dot_scalar_unchecked(left, rights[1]),
                    sq_int8_dot_scalar_unchecked(left, rights[2]),
                    sq_int8_dot_scalar_unchecked(left, rights[3]),
                ]
            }
        }
        SqInt8CpuKernel::Auto => unreachable!("auto kernel must resolve before dispatch"),
        SqInt8CpuKernel::Scalar | SqInt8CpuKernel::Avx2 => [
            sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(
                left,
                rights[0],
                kernel,
                right_sums_for_full_len.map(|sums| sums[0]),
            ),
            sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(
                left,
                rights[1],
                kernel,
                right_sums_for_full_len.map(|sums| sums[1]),
            ),
            sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(
                left,
                rights[2],
                kernel,
                right_sums_for_full_len.map(|sums| sums[2]),
            ),
            sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(
                left,
                rights[3],
                kernel,
                right_sums_for_full_len.map(|sums| sums[3]),
            ),
        ],
    }
}

#[inline]
fn cpu_supports_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn cpu_supports_avx512_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sq_int8_dot_avx2_unchecked(left: &[i8], right: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let mut idx = 0usize;
    let mut acc = _mm256_setzero_si256();
    while idx + 16 <= left.len() {
        let lhs = unsafe { _mm_loadu_si128(left.as_ptr().add(idx).cast::<__m128i>()) };
        let rhs = unsafe { _mm_loadu_si128(right.as_ptr().add(idx).cast::<__m128i>()) };
        let lhs16 = _mm256_cvtepi8_epi16(lhs);
        let rhs16 = _mm256_cvtepi8_epi16(rhs);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lhs16, rhs16));
        idx += 16;
    }

    let mut lanes = [0_i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };
    let mut dot = lanes.iter().sum::<i32>();
    while idx < left.len() {
        dot += i32::from(left[idx]) * i32::from(right[idx]);
        idx += 1;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn sq_int8_dot_avx512_vnni_unchecked(
    left: &[i8],
    right: &[i8],
    right_sum_for_full_len: Option<i32>,
) -> i32 {
    use std::arch::x86_64::*;

    let mut idx = 0usize;
    let mut acc = _mm512_setzero_si512();
    let sign_flip = _mm512_set1_epi8(i8::MIN);
    while idx + 64 <= left.len() {
        let lhs = unsafe { _mm512_loadu_si512(left.as_ptr().add(idx).cast::<__m512i>()) };
        let rhs = unsafe { _mm512_loadu_si512(right.as_ptr().add(idx).cast::<__m512i>()) };
        let lhs_unsigned = _mm512_xor_si512(lhs, sign_flip);
        acc = _mm512_dpbusd_epi32(acc, lhs_unsigned, rhs);
        idx += 64;
    }

    let mut lanes = [0_i32; 16];
    unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), acc) };
    let mut biased_dot = lanes.iter().sum::<i32>();
    let right_sum = if idx == right.len() {
        right_sum_for_full_len.unwrap_or_else(|| right.iter().map(|&value| i32::from(value)).sum())
    } else {
        right[..idx].iter().map(|&value| i32::from(value)).sum()
    };
    while idx < left.len() {
        biased_dot += i32::from(left[idx]) * i32::from(right[idx]);
        idx += 1;
    }
    biased_dot - 128 * right_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn sq_int8_dot4_avx512_vnni_unchecked(
    left: &[i8],
    rights: [&[i8]; 4],
    right_sums_for_full_len: Option<[i32; 4]>,
) -> [i32; 4] {
    use std::arch::x86_64::*;

    let mut idx = 0usize;
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();
    let sign_flip = _mm512_set1_epi8(i8::MIN);
    while idx + 64 <= left.len() {
        let lhs = unsafe { _mm512_loadu_si512(left.as_ptr().add(idx).cast::<__m512i>()) };
        let lhs_unsigned = _mm512_xor_si512(lhs, sign_flip);
        let rhs0 = unsafe { _mm512_loadu_si512(rights[0].as_ptr().add(idx).cast::<__m512i>()) };
        let rhs1 = unsafe { _mm512_loadu_si512(rights[1].as_ptr().add(idx).cast::<__m512i>()) };
        let rhs2 = unsafe { _mm512_loadu_si512(rights[2].as_ptr().add(idx).cast::<__m512i>()) };
        let rhs3 = unsafe { _mm512_loadu_si512(rights[3].as_ptr().add(idx).cast::<__m512i>()) };
        acc0 = _mm512_dpbusd_epi32(acc0, lhs_unsigned, rhs0);
        acc1 = _mm512_dpbusd_epi32(acc1, lhs_unsigned, rhs1);
        acc2 = _mm512_dpbusd_epi32(acc2, lhs_unsigned, rhs2);
        acc3 = _mm512_dpbusd_epi32(acc3, lhs_unsigned, rhs3);
        idx += 64;
    }

    let mut lanes0 = [0_i32; 16];
    let mut lanes1 = [0_i32; 16];
    let mut lanes2 = [0_i32; 16];
    let mut lanes3 = [0_i32; 16];
    unsafe {
        _mm512_storeu_si512(lanes0.as_mut_ptr().cast::<__m512i>(), acc0);
        _mm512_storeu_si512(lanes1.as_mut_ptr().cast::<__m512i>(), acc1);
        _mm512_storeu_si512(lanes2.as_mut_ptr().cast::<__m512i>(), acc2);
        _mm512_storeu_si512(lanes3.as_mut_ptr().cast::<__m512i>(), acc3);
    }
    let mut biased = [
        lanes0.iter().sum::<i32>(),
        lanes1.iter().sum::<i32>(),
        lanes2.iter().sum::<i32>(),
        lanes3.iter().sum::<i32>(),
    ];
    let right_sums = if idx == left.len() {
        right_sums_for_full_len.unwrap_or_else(|| {
            [
                rights[0].iter().map(|&value| i32::from(value)).sum(),
                rights[1].iter().map(|&value| i32::from(value)).sum(),
                rights[2].iter().map(|&value| i32::from(value)).sum(),
                rights[3].iter().map(|&value| i32::from(value)).sum(),
            ]
        })
    } else {
        [
            rights[0][..idx].iter().map(|&value| i32::from(value)).sum(),
            rights[1][..idx].iter().map(|&value| i32::from(value)).sum(),
            rights[2][..idx].iter().map(|&value| i32::from(value)).sum(),
            rights[3][..idx].iter().map(|&value| i32::from(value)).sum(),
        ]
    };
    while idx < left.len() {
        let lhs = i32::from(left[idx]);
        biased[0] += lhs * i32::from(rights[0][idx]);
        biased[1] += lhs * i32::from(rights[1][idx]);
        biased[2] += lhs * i32::from(rights[2][idx]);
        biased[3] += lhs * i32::from(rights[3][idx]);
        idx += 1;
    }
    [
        biased[0] - 128 * right_sums[0],
        biased[1] - 128 * right_sums[1],
        biased[2] - 128 * right_sums[2],
        biased[3] - 128 * right_sums[3],
    ]
}

pub(crate) fn sq_int8_approx_l2(
    left_code: &[i8],
    left_scale: f32,
    left_norm: f32,
    right_code: &[i8],
    right_scale: f32,
    right_norm: f32,
) -> f32 {
    let dot = left_code
        .iter()
        .zip(right_code)
        .map(|(&lhs, &rhs)| i32::from(lhs) * i32::from(rhs))
        .sum::<i32>() as f32;
    let dist = left_norm + right_norm - 2.0 * left_scale * right_scale * dot;
    if dist <= 0.0 { 0.0 } else { dist }
}

pub(crate) fn sq_int8_leaf_shadow_l2(
    points_row_major: &[f32],
    point_ids: &[u32],
    dim: usize,
    requested_k: usize,
    cfg: SqInt8LeafShadowConfig,
) -> AnnResult<SqInt8LeafShadowResult> {
    if points_row_major.len() != point_ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "SQ-int8 leaf shadow point shape mismatch: rows={} dim={} values={}",
            point_ids.len(),
            dim,
            points_row_major.len()
        )));
    }

    let rows = point_ids.len();
    if rows == 0 {
        return Ok(SqInt8LeafShadowResult {
            topk_ids: Vec::new(),
            telemetry: SqInt8LeafShadowTelemetry::default(),
        });
    }

    let matrix = quantize_sq_int8_matrix(points_row_major, rows, dim, cfg.k_align)?;
    let k = requested_k.min(rows.saturating_sub(1));
    let candidate_count = rows.saturating_sub(1);
    let survivor_count = cfg.survivors.max(k).min(candidate_count);
    let full_pairs = rows.saturating_mul(candidate_count);
    let mut telemetry = SqInt8LeafShadowTelemetry {
        rows,
        dim,
        padded_dim: matrix.padded_dim,
        full_pairs,
        prefilter_pairs: full_pairs,
        pruned_pairs: rows.saturating_mul(candidate_count.saturating_sub(survivor_count)),
        survivor_pairs: rows.saturating_mul(survivor_count),
        exact_pairs: rows.saturating_mul(survivor_count),
        code_bytes: matrix.codes.len().saturating_mul(std::mem::size_of::<i8>()),
        scale_bytes: matrix
            .scales
            .len()
            .saturating_mul(std::mem::size_of::<f32>()),
        ..SqInt8LeafShadowTelemetry::default()
    };

    let mut topk_ids = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut approx_candidates = Vec::with_capacity(candidate_count);
        for col in 0..rows {
            if row == col {
                continue;
            }
            approx_candidates.push((col, matrix.approx_l2(row, col)));
        }
        sort_candidates(&mut approx_candidates, point_ids);

        let mut exact_survivors = Vec::with_capacity(survivor_count);
        for &(col, _) in approx_candidates.iter().take(survivor_count) {
            exact_survivors.push((
                col,
                exact_l2(
                    row_slice(points_row_major, dim, row),
                    row_slice(points_row_major, dim, col),
                ),
            ));
        }
        sort_candidates(&mut exact_survivors, point_ids);

        let row_topk = exact_survivors
            .iter()
            .take(k)
            .map(|&(col, _)| point_ids[col])
            .collect::<Vec<_>>();

        if cfg.audit {
            telemetry.false_prune_audit_pairs += candidate_count;
            let exact_row_topk = exact_topk_for_row(points_row_major, point_ids, dim, row, k);
            telemetry.false_prune_audit_misses += exact_row_topk
                .iter()
                .filter(|target| !row_topk.contains(target))
                .count();
        }
        topk_ids.push(row_topk);
    }

    Ok(SqInt8LeafShadowResult {
        topk_ids,
        telemetry,
    })
}

fn exact_topk_for_row(
    points_row_major: &[f32],
    point_ids: &[u32],
    dim: usize,
    row: usize,
    k: usize,
) -> Vec<u32> {
    let point = row_slice(points_row_major, dim, row);
    let mut candidates = Vec::with_capacity(point_ids.len().saturating_sub(1));
    for col in 0..point_ids.len() {
        if row == col {
            continue;
        }
        candidates.push((col, exact_l2(point, row_slice(points_row_major, dim, col))));
    }
    sort_candidates(&mut candidates, point_ids);
    candidates
        .iter()
        .take(k)
        .map(|&(col, _)| point_ids[col])
        .collect()
}

fn row_slice(points_row_major: &[f32], dim: usize, row: usize) -> &[f32] {
    &points_row_major[row * dim..(row + 1) * dim]
}

fn sort_candidates(candidates: &mut [(usize, f32)], point_ids: &[u32]) {
    candidates.sort_unstable_by(|left, right| {
        let left_id = point_ids[left.0];
        let right_id = point_ids[right.0];
        super::stable_distance_order(left.1, left_id, right.1, right_id)
    });
}

fn exact_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&lhs, &rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn rotated_residual_sq8_accounts_for_different_centers() {
        let centers = [0.0_f32, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0];
        let points = [0.25_f32, -0.1, 0.1, 0.0, 100.2, 0.1, 0.0, -0.1];
        let matrix =
            super::quantize_sq8_rotated_residual_matrix(&points, 2, 4, &centers, 42).unwrap();
        assert_ne!(matrix.centroid_ids[0], matrix.centroid_ids[1]);
        for row in 0..2 {
            let reconstructed_norm: f32 = matrix
                .residuals
                .code_row(row)
                .iter()
                .map(|&code| {
                    let value = matrix.residuals.scales[row] * f32::from(code);
                    value * value
                })
                .sum();
            assert!((matrix.residuals.norms[row] - reconstructed_norm).abs() < 1e-5);
        }
        let exact: f32 = points[..4]
            .iter()
            .zip(&points[4..])
            .map(|(&left, &right)| (left - right) * (left - right))
            .sum();
        assert!((matrix.approx_l2(0, 1) - exact).abs() < 1.0);
        assert!(matrix.approx_l2(0, 1) > 9_000.0);
    }

    #[test]
    fn rotated_residual_sq8_d0_assigns_against_coarse_centers() {
        let centers = [0.0_f32, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0];
        let raw = [0.25_f32, 0.0, 0.0, 0.0, 99.75, 0.0, 0.0, 0.0];
        let result = super::quantize_sq8_and_assign_global_d0_reference(
            super::SqGpuD0QuantizeAssignRequest {
                raw_vectors_row_major: &raw,
                global_ids: &[0, 1],
                global_d0_leader_ids: &[10, 20],
                global_d0_leader_vectors_row_major: &centers,
                leader_payload: None,
                rotated_residual: Some(super::Sq8RotatedResidualRequest {
                    centroids_row_major: &centers,
                    seed: 42,
                }),
                dim: 4,
                fanout_top: 1,
            },
        )
        .unwrap();
        assert_eq!(result.fragments[0].child_id, 0);
        assert_eq!(result.fragments[1].child_id, 1);
    }
    use super::*;
    use crate::algorithm::pipnn::point_store::InmemDatasetPointStore;
    use crate::model::InmemDataset;

    fn build_test_dataset(rows: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for row in 0..rows {
            for axis in 0..dim {
                dataset.data[row * dim + axis] = row as f32 * 0.2 + axis as f32 * 0.375 - 0.5;
            }
        }
        dataset
    }

    #[test]
    fn sq_int8_default_align_matches_cuda_wmma_s8_k_tile() {
        assert_eq!(SQ_INT8_DEFAULT_K_ALIGN, 16);
        assert_eq!(SqInt8LeafShadowConfig::default().k_align, 16);
    }

    #[test]
    fn sq_int8_wmma_dot16_kernel_uses_s8_m16n16k16_fragments() {
        let source = SQ_INT8_WMMA_DOT16_TILE_KERNEL_SRC;
        assert!(source.contains("#include <mma.h>"));
        assert!(source.contains("extern \"C\" __global__ void sq_int8_wmma_dot16_tile"));
        assert!(source.contains("nvcuda::wmma::fragment"));
        assert!(source.contains("nvcuda::wmma::matrix_a"));
        assert!(source.contains("nvcuda::wmma::matrix_b"));
        assert!(source.contains("signed char"));
        assert!(source.contains("nvcuda::wmma::col_major"));
        assert!(source.contains("nvcuda::wmma::mma_sync"));
        assert!(source.contains("nvcuda::wmma::store_matrix_sync"));
        assert!(
            source.matches("16,\n        16,\n        16").count() >= 2,
            "SQ-int8 WMMA source should use CUDA's s8 m16n16k16 fragment shape"
        );
    }

    #[cfg(feature = "pipnn_cuda")]
    fn compile_sq_int8_wmma_dot16_ptx() -> cudarc::nvrtc::Ptx {
        use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

        fn include_paths_with_mma() -> Vec<String> {
            let mut paths = Vec::new();
            let mut push_if_has_mma = |path: String| {
                if std::path::Path::new(&path).join("mma.h").is_file()
                    && !paths.iter().any(|existing| existing == &path)
                {
                    paths.push(path);
                }
            };
            for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
                if let Ok(root) = std::env::var(var) {
                    push_if_has_mma(format!("{root}/include"));
                    push_if_has_mma(format!("{root}/targets/x86_64-linux/include"));
                }
            }
            push_if_has_mma("/usr/local/cuda/include".to_string());
            push_if_has_mma("/usr/local/cuda/targets/x86_64-linux/include".to_string());
            push_if_has_mma("/usr/local/cuda-13.0/include".to_string());
            push_if_has_mma("/usr/local/cuda-13.0/targets/x86_64-linux/include".to_string());
            paths
        }

        let include_paths = include_paths_with_mma();
        if include_paths.is_empty() {
            panic!("no CUDA include path with mma.h for SQ-int8 WMMA NVRTC compile check");
        }
        compile_ptx_with_opts(
            SQ_INT8_WMMA_DOT16_TILE_KERNEL_SRC,
            CompileOptions {
                arch: Some("compute_80"),
                include_paths,
                options: vec!["--std=c++17".to_string()],
                ..CompileOptions::default()
            },
        )
        .expect("SQ-int8 WMMA dot16 smoke kernel should compile with NVRTC")
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn sq_int8_wmma_dot16_kernel_nvrtc_compiles() {
        let _ptx = compile_sq_int8_wmma_dot16_ptx();
    }

    #[cfg(feature = "pipnn_cuda")]
    #[test]
    fn sq_int8_wmma_dot16_kernel_launch_matches_cpu_tile() {
        use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};

        let ptx = compile_sq_int8_wmma_dot16_ptx();
        let context = CudaContext::new(0).expect("CUDA context for SQ-int8 WMMA smoke");
        let stream = context.default_stream();
        let module = context
            .load_module(ptx)
            .expect("load SQ-int8 WMMA smoke module");
        let function = module
            .load_function("sq_int8_wmma_dot16_tile")
            .expect("load SQ-int8 WMMA smoke function");

        let mut a = vec![0_i8; 16 * 16];
        let mut b = vec![0_i8; 16 * 16];
        for row in 0..16 {
            for axis in 0..16 {
                a[row * 16 + axis] = (((row * 7 + axis * 3) % 31) as i8) - 15;
                b[row * 16 + axis] = (((row * 5 + axis * 11 + 2) % 29) as i8) - 14;
            }
        }
        let a_dev = stream.clone_htod(&a).expect("upload SQ-int8 WMMA A tile");
        let b_dev = stream.clone_htod(&b).expect("upload SQ-int8 WMMA B tile");
        let mut dot_dev: CudaSlice<i32> = stream
            .alloc_zeros(16 * 16)
            .expect("allocate SQ-int8 WMMA dot tile");
        let lda = 16_u32;
        let ldb = 16_u32;
        let ldc = 16_u32;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut dot_dev)
            .arg(&lda)
            .arg(&ldb)
            .arg(&ldc);
        unsafe {
            launch
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .expect("launch SQ-int8 WMMA dot16 smoke kernel");
        }
        stream
            .synchronize()
            .expect("synchronize SQ-int8 WMMA dot16 smoke kernel");
        let actual = stream
            .clone_dtoh(&dot_dev)
            .expect("download SQ-int8 WMMA dot tile");

        let mut expected = vec![0_i32; 16 * 16];
        for row in 0..16 {
            for col in 0..16 {
                let mut dot = 0_i32;
                for axis in 0..16 {
                    dot += i32::from(a[row * 16 + axis]) * i32::from(b[col * 16 + axis]);
                }
                expected[row * 16 + col] = dot;
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn sq_int8_quantizes_per_vector_with_padding_and_zero_scale() {
        let matrix = quantize_sq_int8_matrix(
            &[
                0.0_f32, 0.0, 0.0, //
                1.0, -1.0, 0.5,
            ],
            2,
            3,
            8,
        )
        .expect("quantized matrix");

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.dim, 3);
        assert_eq!(matrix.padded_dim, 8);
        assert_eq!(matrix.scales[0], 1.0);
        assert_eq!(matrix.norms[0], 0.0);
        assert_eq!(matrix.code_row(0), &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(matrix.scales[1], 1.0 / 127.0);
        assert_eq!(matrix.norms[1], 2.25);
        assert_eq!(matrix.code_row(1), &[127, -127, 64, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn sq_int8_quantization_bits_control_effective_code_range() {
        let eight_bit = quantize_sq_int8_matrix_with_bits(&[1.0_f32, -1.0, 0.5], 1, 3, 8, 8)
            .expect("8-bit quantized matrix");
        assert_eq!(eight_bit.scales[0], 1.0 / 127.0);
        assert_eq!(eight_bit.code_row(0), &[127, -127, 64, 0, 0, 0, 0, 0]);

        let four_bit = quantize_sq_int8_matrix_with_bits(&[1.0_f32, -1.0, 0.5], 1, 3, 8, 4)
            .expect("4-bit quantized matrix");
        assert_eq!(four_bit.scales[0], 1.0 / 7.0);
        let four_bit_codes = four_bit.code_row(0);
        assert_eq!(four_bit_codes[0], 7);
        assert_eq!(four_bit_codes[1], -7);
        assert!(four_bit_codes.iter().all(|&code| (-7..=7).contains(&code)));
    }

    #[test]
    fn sq_int8_dot_and_approx_l2_use_exact_norms_and_scales() {
        let matrix = quantize_sq_int8_matrix(
            &[
                1.0_f32, 0.5, //
                0.5, 1.0,
            ],
            2,
            2,
            4,
        )
        .expect("quantized matrix");

        let dot = sq_int8_dot(matrix.code_row(0), matrix.code_row(1)).expect("dot");
        assert_eq!(dot, 127 * 64 + 64 * 127);

        let approx = matrix.approx_l2(0, 1);
        let expected_dot = (dot as f32) * matrix.scales[0] * matrix.scales[1];
        let expected = matrix.norms[0] + matrix.norms[1] - 2.0 * expected_dot;
        assert!(
            (approx - expected).abs() < 1.0e-6,
            "approx={approx} expected={expected}"
        );
    }

    #[test]
    fn sq_wmma_sidecar_roundtrip_gathers_in_input_order() {
        let rows = 5;
        let dim = 6;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let report =
            create_sq_wmma_sidecar_from_point_store(&store, &prefix, source_bytes as u64).unwrap();
        assert_eq!(report.header.rows, rows);
        assert_eq!(report.header.dim, dim);
        assert_eq!(report.header.bits, 8);
        assert_eq!(report.header.codes_bytes, rows * report.header.padded_dim);
        assert_eq!(
            report.header.meta_rows_bytes,
            rows * SQ_WMMA_SIDECAR_META_ROW_BYTES
        );

        let sidecar = SqWmmaSidecar::open_validated(&prefix, rows, dim).unwrap();
        assert_eq!(sidecar.header().bits, 8);
        let ids = [3_u32, 1, 3, 0];
        let gathered = sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.global_ids, ids);
        assert_eq!(gathered.codes.len(), ids.len() * report.header.padded_dim);
        assert_eq!(gathered.norms.len(), ids.len());
        assert_eq!(gathered.scales.len(), ids.len());
        assert_eq!(gathered.code_sums, vec![0.0; ids.len()]);

        let expected = quantize_sq_int8_matrix(
            &dataset.data[..rows * dim],
            rows,
            dim,
            SQ_INT8_DEFAULT_K_ALIGN,
        )
        .unwrap();
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let got_codes =
                &gathered.codes[slot * expected.padded_dim..(slot + 1) * expected.padded_dim];
            assert_eq!(got_codes, expected.code_row(id));
            assert!((gathered.scales[slot] - expected.scales[id]).abs() <= f32::EPSILON);
            assert!((gathered.norms[slot] - expected.norms[id]).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn sq_wmma_shard_writer_roundtrip_gathers_in_input_order() {
        let rows = 4;
        let dim = 3;
        let dataset = build_test_dataset(rows, dim);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("staged.sq8");
        let mut writer =
            SqWmmaSidecarShardWriter::create(&prefix, rows, dim, 0, SQ_INT8_DEFAULT_BITS).unwrap();

        let first =
            quantize_sq_int8_matrix(&dataset.data[..2 * dim], 2, dim, SQ_INT8_DEFAULT_K_ALIGN)
                .unwrap();
        writer.append_dense_i8_matrix(0, &first).unwrap();
        let second = quantize_sq_int8_matrix(
            &dataset.data[2 * dim..4 * dim],
            2,
            dim,
            SQ_INT8_DEFAULT_K_ALIGN,
        )
        .unwrap();
        writer.append_dense_i8_matrix(2, &second).unwrap();
        let report = writer.finish().unwrap();

        assert_eq!(report.header.rows, rows);
        assert_eq!(report.header.dim, dim);
        assert_eq!(report.header.bits, 8);

        let sidecar = SqWmmaSidecar::open_validated(&prefix, rows, dim).unwrap();
        let ids = [3_u32, 0, 2];
        let gathered = sidecar.gather(&ids).unwrap();
        let expected = quantize_sq_int8_matrix(
            &dataset.data[..rows * dim],
            rows,
            dim,
            SQ_INT8_DEFAULT_K_ALIGN,
        )
        .unwrap();
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let got_codes =
                &gathered.codes[slot * expected.padded_dim..(slot + 1) * expected.padded_dim];
            assert_eq!(got_codes, expected.code_row(id));
            assert!((gathered.scales[slot] - expected.scales[id]).abs() <= f32::EPSILON);
            assert!((gathered.norms[slot] - expected.norms[id]).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn sq_gpu_d0_reference_assigns_by_sq_distance_with_stable_ties() {
        let leaders = [1.0_f32, 0.0, -1.0, 0.0];
        let leader_payload =
            quantize_sq_int8_matrix(&leaders, 2, 2, SQ_INT8_DEFAULT_K_ALIGN).unwrap();
        let request = SqGpuD0QuantizeAssignRequest {
            raw_vectors_row_major: &[0.0_f32, 0.0, 1.0, 0.0],
            global_ids: &[0, 1],
            global_d0_leader_ids: &[5, 3],
            global_d0_leader_vectors_row_major: &leaders,
            leader_payload: Some(&leader_payload),
            rotated_residual: None,
            dim: 2,
            fanout_top: 1,
        };

        let result = quantize_sq8_and_assign_global_d0_reference(request).unwrap();

        assert_eq!(
            result
                .fragments
                .iter()
                .map(|fragment| (fragment.child_id, fragment.point_id))
                .collect::<Vec<_>>(),
            vec![(1, 0), (0, 1)]
        );
        assert_eq!(result.matrix.rows, 2);
        assert_eq!(result.profile.d0_assign_kernel_requested, "cpu-reference");
        assert_eq!(result.profile.d0_assign_kernel_mode, "cpu-reference");
        assert_eq!(result.profile.d0_assign_rows, 2);
        assert_eq!(result.profile.d0_assign_leaders, 2);
        assert_eq!(result.profile.d0_assign_fanout, 1);
    }

    #[test]
    fn sq_wmma_sidecar_roundtrip_preserves_lower_effective_bits() {
        let rows = 3;
        let dim = 3;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.sq4");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let report = create_sq_wmma_sidecar_from_point_store_with_bits(
            &store,
            &prefix,
            source_bytes as u64,
            4,
        )
        .unwrap();
        assert_eq!(report.header.bits, 4);

        let sidecar = SqWmmaSidecar::open_validated_with_bits(&prefix, rows, dim, 4).unwrap();
        assert_eq!(sidecar.header().bits, 4);
        let ids = [0_u32, 2];
        let gathered = sidecar.gather(&ids).unwrap();
        let expected =
            quantize_sq_int8_matrix_with_bits(&dataset.data[..rows * dim], rows, dim, 16, 4)
                .unwrap();
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let got_codes =
                &gathered.codes[slot * expected.padded_dim..(slot + 1) * expected.padded_dim];
            assert_eq!(got_codes, expected.code_row(id));
            assert!((gathered.scales[slot] - expected.scales[id]).abs() <= f32::EPSILON);
        }

        let err = SqWmmaSidecar::open_validated_with_bits(&prefix, rows, dim, 8)
            .expect_err("mismatched SQ bits should be rejected");
        assert!(
            format!("{err:?}").contains("SQ+WMMA sidecar bits mismatch"),
            "{err:?}"
        );
    }

    #[test]
    fn sq_wmma_packed_s4_pack_unpack_roundtrip_preserves_signed_codes() {
        let dense = vec![
            -8, -7, -1, 0, 1, 7, 0, 0, //
            7, 6, 5, 4, 3, 2, 1, 0,
        ];
        let packed = pack_sq_s4_codes(&dense, 2, 8).unwrap();
        assert_eq!(packed.len(), 8);
        let unpacked = unpack_sq_s4_codes(&packed, 2, 8).unwrap();
        assert_eq!(unpacked, dense);
    }

    #[test]
    fn sq_wmma_sidecar_packed_s4_roundtrip_halves_code_bytes() {
        let rows = 4;
        let dim = 7;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.sq4p");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let artifact = create_sq_wmma_sidecar_from_point_store_with_bits_layout(
            &store,
            &prefix,
            source_bytes as u64,
            4,
            SqWmmaCodeLayout::PackedS4,
            2,
        )
        .unwrap();
        let report = artifact.report;
        assert_eq!(report.header.bits, 4);
        assert_eq!(report.header.code_layout, SqWmmaCodeLayout::PackedS4);
        assert_eq!(report.header.code_row_bytes, report.header.padded_dim / 2);
        assert_eq!(
            report.header.codes_bytes,
            rows * report.header.code_row_bytes
        );

        let sidecar = SqWmmaSidecar::open_validated_with_bits(&prefix, rows, dim, 4).unwrap();
        assert_eq!(sidecar.header().code_layout, SqWmmaCodeLayout::PackedS4);
        assert_eq!(
            sidecar.header().code_row_bytes,
            report.header.code_row_bytes
        );
        let ids = [3_u32, 0, 2];
        let gathered = sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.code_layout, SqWmmaCodeLayout::PackedS4);
        assert_eq!(gathered.code_row_bytes, report.header.code_row_bytes);
        assert_eq!(
            gathered.codes.len(),
            ids.len() * report.header.code_row_bytes
        );

        let unpacked =
            unpack_sq_s4_codes(&gathered.codes, ids.len(), report.header.padded_dim).unwrap();
        let expected =
            quantize_sq_int8_matrix_with_bits(&dataset.data[..rows * dim], rows, dim, 16, 4)
                .unwrap();
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let got_codes = &unpacked[slot * expected.padded_dim..(slot + 1) * expected.padded_dim];
            assert_eq!(got_codes, expected.code_row(id));
        }
    }

    #[test]
    fn sq_wmma_sidecar_source_exposes_packed_s4_layout_contract() {
        let source = include_str!("dco_sq_int8.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("implementation source before tests");
        assert!(source.contains("SqWmmaCodeLayout"));
        assert!(source.contains("PackedS4"));
        assert!(source.contains("code_row_bytes"));
        assert!(source.contains("pack_sq_s4_codes"));
        assert!(source.contains("unpack_sq_s4_codes"));
        assert!(source.contains("create_sq_wmma_sidecar_from_point_store_with_bits_layout"));
    }

    #[test]
    fn sq_wmma_sidecar_create_uses_metadata_fingerprint_without_raw_hash() {
        let rows = 4;
        let dim = 5;
        let left = build_test_dataset(rows, dim);
        let mut right = build_test_dataset(rows, dim);
        right.data[0] += 17.0;
        right.data[rows * dim - 1] -= 3.5;
        let left_store = InmemDatasetPointStore::new(&left, rows);
        let right_store = InmemDatasetPointStore::new(&right, rows);
        let left_dir = tempfile::tempdir().unwrap();
        let right_dir = tempfile::tempdir().unwrap();
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let left_report = create_sq_wmma_sidecar_from_point_store(
            &left_store,
            &left_dir.path().join("left.sq8"),
            source_bytes as u64,
        )
        .unwrap();
        let right_report = create_sq_wmma_sidecar_from_point_store(
            &right_store,
            &right_dir.path().join("right.sq8"),
            source_bytes as u64,
        )
        .unwrap();

        assert_eq!(left_report.source_hash_wall, Duration::ZERO);
        assert_eq!(right_report.source_hash_wall, Duration::ZERO);
        assert_eq!(
            left_report.header.source_fingerprint, right_report.header.source_fingerprint,
            "SQ sidecar source_fingerprint should no longer depend on raw f32 contents"
        );
        assert_ne!(left.data[0], right.data[0]);
        assert_ne!(left.data[rows * dim - 1], right.data[rows * dim - 1]);
    }

    #[test]
    fn sq_wmma_sidecar_resident_preload_gathers_in_input_order() {
        let rows = 5;
        let dim = 6;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let report =
            create_sq_wmma_sidecar_from_point_store(&store, &prefix, source_bytes as u64).unwrap();
        let sidecar = SqWmmaSidecar::preload_validated(&prefix, rows, dim).unwrap();
        assert!(sidecar.is_resident());
        assert_eq!(
            sidecar.resident_bytes(),
            report
                .header
                .codes_bytes
                .saturating_add(report.header.meta_rows_bytes)
        );

        let ids = [3_u32, 1, 3, 0];
        let gathered = sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.global_ids, ids);
        assert_eq!(gathered.codes.len(), ids.len() * report.header.padded_dim);
        assert_eq!(gathered.norms.len(), ids.len());
        assert_eq!(gathered.scales.len(), ids.len());
        assert!(gathered.resident_source);

        let expected = quantize_sq_int8_matrix(
            &dataset.data[..rows * dim],
            rows,
            dim,
            SQ_INT8_DEFAULT_K_ALIGN,
        )
        .unwrap();
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let got_codes =
                &gathered.codes[slot * expected.padded_dim..(slot + 1) * expected.padded_dim];
            assert_eq!(got_codes, expected.code_row(id));
            assert!((gathered.scales[slot] - expected.scales[id]).abs() <= f32::EPSILON);
            assert!((gathered.norms[slot] - expected.norms[id]).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn sq_wmma_sidecar_partial_resident_prefix_falls_back_to_file_for_suffix_rows() {
        let rows = 6;
        let dim = 6;
        let resident_rows = 3;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("partial.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let report =
            create_sq_wmma_sidecar_from_point_store(&store, &prefix, source_bytes as u64).unwrap();
        let file_sidecar = SqWmmaSidecar::open_validated(&prefix, rows, dim).unwrap();
        let partial_sidecar =
            SqWmmaSidecar::preload_prefix_validated(&prefix, rows, dim, resident_rows).unwrap();

        assert!(partial_sidecar.is_resident());
        assert!(!partial_sidecar.is_fully_resident());
        assert_eq!(
            partial_sidecar.resident_bytes(),
            resident_rows * (report.header.padded_dim + SQ_WMMA_SIDECAR_META_ROW_BYTES)
        );

        let ids = [1_u32, 4, 2, 5];
        let expected = file_sidecar.gather(&ids).unwrap();
        let gathered = partial_sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.global_ids, ids);
        assert_eq!(gathered.resident_rows, 2);
        assert_eq!(gathered.file_rows, 2);
        assert!(!gathered.resident_source);
        assert_eq!(gathered.codes, expected.codes);
        assert_eq!(gathered.scales, expected.scales);
        assert_eq!(gathered.norms, expected.norms);
    }

    #[test]
    fn sq_wmma_sidecar_direct_io_reads_nonresident_rows() {
        let rows = 6;
        let dim = 6;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("direct.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();
        create_sq_wmma_sidecar_from_point_store(&store, &prefix, source_bytes as u64).unwrap();

        let direct = SqWmmaSidecar::open_validated_with_bits_and_io(
            &prefix,
            rows,
            dim,
            8,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let buffered = SqWmmaSidecar::open_validated_with_bits(&prefix, rows, dim, 8).unwrap();
        let ids = [0_u32, 2, 5];

        assert!(direct.uses_direct_io());
        let direct_rows = direct.gather(&ids).unwrap();
        let buffered_rows = buffered.gather(&ids).unwrap();
        assert_eq!(direct_rows.global_ids, buffered_rows.global_ids);
        assert_eq!(direct_rows.codes, buffered_rows.codes);
        assert_eq!(direct_rows.norms, buffered_rows.norms);
        assert_eq!(direct_rows.scales, buffered_rows.scales);
    }

    #[test]
    fn sq_wmma_sidecar_create_can_return_partial_resident_prefix_without_preload() {
        let rows = 6;
        let dim = 6;
        let resident_rows = 3;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("partial-create.sq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let artifact = create_sq_wmma_sidecar_from_point_store_with_resident_prefix(
            &store,
            &prefix,
            source_bytes as u64,
            resident_rows,
        )
        .unwrap();
        let partial_sidecar = artifact
            .runtime_sidecar
            .expect("resident prefix should be available without reopening sidecar files");
        let file_sidecar = SqWmmaSidecar::open_validated(&prefix, rows, dim).unwrap();

        assert_eq!(artifact.report.header.rows, rows);
        assert!(partial_sidecar.is_resident());
        assert!(!partial_sidecar.is_fully_resident());
        assert_eq!(partial_sidecar.resident_rows(), resident_rows);
        assert_eq!(partial_sidecar.preload_wall(), Duration::ZERO);

        let ids = [1_u32, 4, 2, 5];
        let expected = file_sidecar.gather(&ids).unwrap();
        let gathered = partial_sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.resident_rows, 2);
        assert_eq!(gathered.file_rows, 2);
        assert_eq!(gathered.codes, expected.codes);
        assert_eq!(gathered.scales, expected.scales);
        assert_eq!(gathered.norms, expected.norms);
    }

    #[test]
    fn sq_int8_cpu_dot_kernels_match_scalar_signed_dot() {
        let left = (0..128)
            .map(|idx| ((idx * 37 + 11) % 255) as i16 - 127)
            .map(|value| value as i8)
            .collect::<Vec<_>>();
        let right = (0..128)
            .map(|idx| ((idx * 19 + 5) % 255) as i16 - 127)
            .map(|value| value as i8)
            .collect::<Vec<_>>();
        let expected = sq_int8_dot(&left, &right).unwrap();

        assert_eq!(
            sq_int8_dot_with_kernel(&left, &right, SqInt8CpuKernel::Scalar).unwrap(),
            expected
        );
        if SqInt8CpuKernel::Avx2.is_available() {
            assert_eq!(
                sq_int8_dot_with_kernel(&left, &right, SqInt8CpuKernel::Avx2).unwrap(),
                expected
            );
        }
        if SqInt8CpuKernel::Avx512Vnni.is_available() {
            let right_sum = right.iter().map(|&value| i32::from(value)).sum();
            assert_eq!(
                sq_int8_dot_with_kernel(&left, &right, SqInt8CpuKernel::Avx512Vnni).unwrap(),
                expected
            );
            assert_eq!(
                sq_int8_dot_with_resolved_kernel_and_right_sum_unchecked(
                    &left,
                    &right,
                    SqInt8CpuKernel::Avx512Vnni,
                    Some(right_sum),
                ),
                expected
            );
        }
    }

    #[test]
    fn sq_int8_cpu_dot4_vnni_matches_scalar_signed_dot() {
        if !SqInt8CpuKernel::Avx512Vnni.is_available() {
            return;
        }
        let left = (0..128)
            .map(|idx| ((idx * 37 + 11) % 255) as i16 - 127)
            .map(|value| value as i8)
            .collect::<Vec<_>>();
        let rights = (0..4)
            .map(|lane| {
                (0..128)
                    .map(|idx| ((idx * (19 + lane * 7) + 5 + lane) % 255) as i16 - 127)
                    .map(|value| value as i8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let right_sums = [
            rights[0].iter().map(|&value| i32::from(value)).sum(),
            rights[1].iter().map(|&value| i32::from(value)).sum(),
            rights[2].iter().map(|&value| i32::from(value)).sum(),
            rights[3].iter().map(|&value| i32::from(value)).sum(),
        ];
        let expected = [
            sq_int8_dot(&left, &rights[0]).unwrap(),
            sq_int8_dot(&left, &rights[1]).unwrap(),
            sq_int8_dot(&left, &rights[2]).unwrap(),
            sq_int8_dot(&left, &rights[3]).unwrap(),
        ];

        assert_eq!(
            sq_int8_dot4_with_resolved_kernel_and_right_sums_unchecked(
                &left,
                [&rights[0], &rights[1], &rights[2], &rights[3]],
                SqInt8CpuKernel::Avx512Vnni,
                Some(right_sums),
            ),
            expected
        );
    }

    #[test]
    fn sq_wmma_sidecar_validation_rejects_mismatched_dataset_shape() {
        let rows = 3;
        let dim = 4;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.sq8");

        create_sq_wmma_sidecar_from_point_store(&store, &prefix, 0).unwrap();
        let err = SqWmmaSidecar::open_validated(&prefix, rows, dim + 1).unwrap_err();

        assert!(format!("{err:?}").contains("shape mismatch"));
    }

    #[test]
    fn sq_int8_leaf_shadow_exact_rerank_preserves_target_id_tie_break() {
        let point_ids = [99_u32, 30_u32, 10_u32];
        let points = [
            0.0_f32, 0.0, //
            1.0, 0.0, //
            0.0, 1.0,
        ];

        let result = sq_int8_leaf_shadow_l2(
            &points,
            &point_ids,
            2,
            1,
            SqInt8LeafShadowConfig {
                k_align: 4,
                survivors: 2,
                audit: true,
            },
        )
        .expect("SQ-int8 leaf shadow");

        assert_eq!(result.topk_ids[0], vec![10]);
        assert_eq!(result.telemetry.rows, 3);
        assert_eq!(result.telemetry.full_pairs, 6);
        assert_eq!(result.telemetry.prefilter_pairs, 6);
        assert_eq!(result.telemetry.pruned_pairs, 0);
        assert_eq!(result.telemetry.survivor_pairs, 6);
        assert_eq!(result.telemetry.exact_pairs, 6);
        assert_eq!(result.telemetry.false_prune_audit_pairs, 6);
        assert_eq!(result.telemetry.false_prune_audit_misses, 0);
    }

    #[test]
    fn sq_int8_leaf_shadow_audit_counts_false_prune_when_survivor_is_too_small() {
        let point_ids = [99_u32, 10_u32, 20_u32];
        let points = [
            3.313478940375239_f32,
            -0.4845258426888712, //
            0.4858610032249091,
            1.8898940954452268, //
            3.9694088614040357,
            3.144737504552918,
        ];

        let result = sq_int8_leaf_shadow_l2(
            &points,
            &point_ids,
            2,
            1,
            SqInt8LeafShadowConfig {
                k_align: 4,
                survivors: 1,
                audit: true,
            },
        )
        .expect("SQ-int8 leaf shadow");

        assert_eq!(result.topk_ids[0], vec![10]);
        assert_eq!(exact_topk_for_row(&points, &point_ids, 2, 0, 1), vec![20]);
        assert!(result.telemetry.false_prune_audit_misses >= 1);
        assert_eq!(result.telemetry.pruned_pairs, 3);
        assert_eq!(result.telemetry.survivor_pairs, 3);
        assert_eq!(result.telemetry.exact_pairs, 3);
    }

    #[test]
    fn sq_int8_rejects_bad_shapes_and_non_finite_values() {
        let shape_err = quantize_sq_int8_matrix(&[0.0, 1.0, 2.0], 1, 2, 4)
            .expect_err("shape mismatch should fail");
        assert!(shape_err.to_string().contains("shape mismatch"));

        let finite_err = quantize_sq_int8_matrix(&[0.0, f32::NAN], 1, 2, 4)
            .expect_err("non-finite value should fail");
        assert!(finite_err.to_string().contains("non-finite"));
    }
}
