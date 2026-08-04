use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::direct_io::DirectIoConfig;
use super::external_store::{DiskSketchStore, fill_sketch_row_from_slice};
use super::hash_prune::generate_hyperplanes;
use super::params::{PiPNNGpuSubtreeRabitqPlusRotator, PiPNNParams};
use super::point_store::PointStore;
use crate::common::{AnnError, AnnResult};

pub(crate) const RABITQ_DEFAULT_TOTAL_BITS: usize = 8;
pub(crate) const RABITQ_DEFAULT_K_ALIGN: usize = 16;
pub(crate) const RABITQ_PLUS_SIDECAR_HEADER_BYTES: usize = 256;
pub(crate) const RABITQ_PLUS_SIDECAR_META_ROW_BYTES: usize = 12;

const RABITQ_TIGHT_START: [f64; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];
const RABITQ_QUANT_EPS: f64 = 1.0e-5;
const RABITQ_PLUS_SIDECAR_MAGIC: &[u8; 8] = b"PGRBQ+1\0";
const RABITQ_SPLIT_SIDECAR_MAGIC: &[u8; 8] = b"PGRBQS1\0";
const RABITQ_PLUS_SIDECAR_VERSION: u32 = 1;
const RABITQ_PLUS_SIDECAR_CREATE_CHUNK_ROWS: usize = 16 * 1024;
const RABITQ_SPLIT_BATCH_ROWS: usize = 32;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

static RABITQ_T_CONST_CACHE: OnceLock<Mutex<HashMap<(usize, usize), f64>>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqMatrix {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) total_bits: usize,
    pub(crate) codes: Vec<i8>,
    pub(crate) deltas: Vec<f32>,
    pub(crate) code_sums: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqSplitMatrix {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) total_bits: usize,
    pub(crate) ex_bits: usize,
    pub(crate) batch_count: usize,
    pub(crate) batch_bytes: usize,
    pub(crate) ex_row_bytes: usize,
    pub(crate) batch_data: Vec<u8>,
    pub(crate) ex_data: Vec<u8>,
    pub(crate) rotated: Vec<f32>,
    pub(crate) query_kbxsumq: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqSplitQueryMatrix {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) total_bits: usize,
    pub(crate) ex_bits: usize,
    pub(crate) rotated: Vec<f32>,
    pub(crate) query_kbxsumq: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RabitqPlusSidecarHeader {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) bits: usize,
    pub(crate) rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    pub(crate) seed: u64,
    pub(crate) k_align: usize,
    pub(crate) source_fbin_bytes: u64,
    pub(crate) source_rows: usize,
    pub(crate) source_dim: usize,
    pub(crate) source_fingerprint: u64,
    pub(crate) codes_bytes: usize,
    pub(crate) meta_rows_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqPlusGatheredInput {
    pub(crate) global_ids: Vec<u32>,
    pub(crate) codes: Vec<i8>,
    pub(crate) deltas: Vec<f32>,
    pub(crate) code_sums: Vec<f32>,
    pub(crate) norms: Vec<f32>,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) bits: usize,
    pub(crate) gather_wall: Duration,
    pub(crate) resident_source: bool,
    pub(crate) resident_rows: usize,
    pub(crate) file_rows: usize,
    pub(crate) resident_gather_wall: Duration,
    pub(crate) file_gather_wall: Duration,
}

#[derive(Debug)]
pub(crate) struct RabitqPlusSidecar {
    prefix: PathBuf,
    header: RabitqPlusSidecarHeader,
    storage: RabitqPlusSidecarStorage,
}

#[derive(Debug)]
enum RabitqPlusSidecarStorage {
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
        deltas: Vec<f32>,
        code_sums: Vec<f32>,
        preload_wall: Duration,
    },
    Resident {
        codes: Vec<i8>,
        norms: Vec<f32>,
        deltas: Vec<f32>,
        code_sums: Vec<f32>,
        preload_wall: Duration,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RabitqPlusSidecarCreateReport {
    pub(crate) header: RabitqPlusSidecarHeader,
    pub(crate) total_wall: Duration,
    pub(crate) chunk_alloc_wall: Duration,
    pub(crate) raw_read_wall: Duration,
    pub(crate) source_hash_wall: Duration,
    pub(crate) quantize_wall: Duration,
    pub(crate) code_output_wall: Duration,
    pub(crate) norm_wall: Duration,
    pub(crate) meta_output_wall: Duration,
    pub(crate) header_write_wall: Duration,
    pub(crate) write_wall: Duration,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RabitqPlusSidecarSketchBuildReport {
    pub(crate) sidecar: RabitqPlusSidecarCreateReport,
    pub(crate) total_wall: Duration,
    pub(crate) raw_read_wall: Duration,
    pub(crate) sketch_alloc_wall: Duration,
    pub(crate) sketch_compute_wall: Duration,
    pub(crate) sketch_extend_wall: Duration,
    pub(crate) sketch_write_wall: Duration,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RabitqSplitSidecarHeader {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) bits: usize,
    pub(crate) ex_bits: usize,
    pub(crate) rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    pub(crate) seed: u64,
    pub(crate) k_align: usize,
    pub(crate) batch_rows: usize,
    pub(crate) batch_count: usize,
    pub(crate) batch_bytes: usize,
    pub(crate) ex_row_bytes: usize,
    pub(crate) source_fbin_bytes: u64,
    pub(crate) source_rows: usize,
    pub(crate) source_dim: usize,
    pub(crate) source_fingerprint: u64,
    pub(crate) batch_total_bytes: usize,
    pub(crate) ex_total_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RabitqSplitSidecarCreateReport {
    pub(crate) header: RabitqSplitSidecarHeader,
    pub(crate) total_wall: Duration,
    pub(crate) chunk_alloc_wall: Duration,
    pub(crate) raw_read_wall: Duration,
    pub(crate) source_hash_wall: Duration,
    pub(crate) quantize_wall: Duration,
    pub(crate) batch_output_wall: Duration,
    pub(crate) ex_output_wall: Duration,
    pub(crate) header_write_wall: Duration,
    pub(crate) write_wall: Duration,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RabitqSplitD0Assignment {
    pub(crate) assignments: Vec<u16>,
    pub(crate) coarse_wall: Duration,
    pub(crate) boost_wall: Duration,
    pub(crate) boosted_pairs: u64,
}

#[derive(Debug)]
pub(crate) struct RabitqSplitSidecar {
    prefix: PathBuf,
    header: RabitqSplitSidecarHeader,
    batch_data: Vec<u8>,
    ex_data: Vec<u8>,
    preload_wall: Duration,
}

impl RabitqMatrix {
    pub(crate) fn code_row(&self, row: usize) -> &[i8] {
        let start = row * self.padded_dim;
        &self.codes[start..start + self.padded_dim]
    }

    pub(crate) fn approx_l2(
        &self,
        left: usize,
        left_norm: f32,
        right: usize,
        right_norm: f32,
    ) -> f32 {
        rabitq_approx_l2(
            self.code_row(left),
            self.deltas[left],
            self.code_sums[left],
            left_norm,
            self.code_row(right),
            self.deltas[right],
            self.code_sums[right],
            right_norm,
            self.dim,
        )
    }
}

impl RabitqPlusSidecarHeader {
    pub(crate) fn validate_for_dataset(
        &self,
        rows: usize,
        dim: usize,
        bits: usize,
        rotator: PiPNNGpuSubtreeRabitqPlusRotator,
        seed: u64,
    ) -> AnnResult<()> {
        if self.rows != rows
            || self.dim != dim
            || self.source_rows != rows
            || self.source_dim != dim
        {
            return Err(AnnError::log_index_error(format!(
                "dense-i8 RabitQ+ sidecar shape mismatch: sidecar rows={} dim={} source_rows={} source_dim={} expected rows={rows} dim={dim}",
                self.rows, self.dim, self.source_rows, self.source_dim
            )));
        }
        if self.bits != bits {
            return Err(AnnError::log_index_error(format!(
                "dense-i8 RabitQ+ sidecar bits mismatch: sidecar={} expected={bits}",
                self.bits
            )));
        }
        if self.rotator != rotator {
            return Err(AnnError::log_index_error(format!(
                "dense-i8 RabitQ+ sidecar rotator mismatch: sidecar={} expected={}",
                self.rotator.as_str(),
                rotator.as_str()
            )));
        }
        if self.seed != seed {
            return Err(AnnError::log_index_error(format!(
                "dense-i8 RabitQ+ sidecar seed mismatch: sidecar={} expected={seed}",
                self.seed
            )));
        }
        Ok(())
    }
}

impl RabitqPlusGatheredInput {
    pub(crate) fn approx_l2(&self, left: usize, right: usize) -> f32 {
        let left_start = left * self.padded_dim;
        let right_start = right * self.padded_dim;
        rabitq_approx_l2(
            &self.codes[left_start..left_start + self.padded_dim],
            self.deltas[left],
            self.code_sums[left],
            self.norms[left],
            &self.codes[right_start..right_start + self.padded_dim],
            self.deltas[right],
            self.code_sums[right],
            self.norms[right],
            self.dim,
        )
    }
}

impl RabitqPlusSidecar {
    pub(crate) fn from_resident_parts(
        prefix: PathBuf,
        header: RabitqPlusSidecarHeader,
        codes: Vec<i8>,
        norms: Vec<f32>,
        deltas: Vec<f32>,
        code_sums: Vec<f32>,
        preload_wall: Duration,
    ) -> AnnResult<Self> {
        if codes.len() != header.codes_bytes {
            return Err(AnnError::log_index_error(format!(
                "Resident dense-i8 RabitQ+ sidecar code size mismatch: got={} expected={}",
                codes.len(),
                header.codes_bytes
            )));
        }
        if norms.len() != header.rows
            || deltas.len() != header.rows
            || code_sums.len() != header.rows
        {
            return Err(AnnError::log_index_error(format!(
                "Resident dense-i8 RabitQ+ sidecar metadata size mismatch: rows={} norms={} deltas={} code_sums={}",
                header.rows,
                norms.len(),
                deltas.len(),
                code_sums.len()
            )));
        }
        Ok(Self {
            prefix,
            header,
            storage: RabitqPlusSidecarStorage::Resident {
                codes,
                norms,
                deltas,
                code_sums,
                preload_wall,
            },
        })
    }

    pub(crate) fn open_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        rotator: PiPNNGpuSubtreeRabitqPlusRotator,
        seed: u64,
    ) -> AnnResult<Self> {
        let codes_path = sidecar_codes_path(prefix);
        let meta_path = sidecar_meta_path(prefix);
        let mut meta_file = File::open(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open dense-i8 RabitQ+ sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
        let header = read_rabitq_plus_sidecar_header(&mut meta_file)?;
        header.validate_for_dataset(rows, dim, bits, rotator, seed)?;
        let codes_file = File::open(&codes_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open dense-i8 RabitQ+ sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        let codes_len = codes_file
            .metadata()
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to stat dense-i8 RabitQ+ sidecar codes '{}': {err}",
                    codes_path.display()
                ))
            })?
            .len() as usize;
        if codes_len != header.codes_bytes {
            return Err(AnnError::log_index_error(format!(
                "dense-i8 RabitQ+ sidecar codes size mismatch: file={} header={}",
                codes_len, header.codes_bytes
            )));
        }
        Ok(Self {
            prefix: prefix.to_path_buf(),
            header,
            storage: RabitqPlusSidecarStorage::File {
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
        rotator: PiPNNGpuSubtreeRabitqPlusRotator,
        seed: u64,
    ) -> AnnResult<Self> {
        let preload_start = Instant::now();
        let sidecar = Self::open_validated(prefix, rows, dim, bits, rotator, seed)?;
        let header = sidecar.header.clone();
        let mut codes = vec![0_i8; header.codes_bytes];
        let mut norms = vec![0.0_f32; header.rows];
        let mut deltas = vec![0.0_f32; header.rows];
        let mut code_sums = vec![0.0_f32; header.rows];

        let RabitqPlusSidecarStorage::File {
            codes_file,
            meta_file,
        } = &sidecar.storage
        else {
            unreachable!("newly opened sidecar is always file-backed")
        };

        let codes_bytes =
            unsafe { std::slice::from_raw_parts_mut(codes.as_mut_ptr().cast::<u8>(), codes.len()) };
        codes_file.read_exact_at(codes_bytes, 0).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload dense-i8 RabitQ+ sidecar codes '{}': {err}",
                sidecar_codes_path(prefix).display()
            ))
        })?;
        advise_sidecar_dontneed(codes_file, 0, codes_bytes.len());

        let mut meta_bytes = vec![0_u8; header.meta_rows_bytes];
        meta_file
            .read_exact_at(&mut meta_bytes, RABITQ_PLUS_SIDECAR_HEADER_BYTES as u64)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload dense-i8 RabitQ+ sidecar metadata '{}': {err}",
                    sidecar_meta_path(prefix).display()
                ))
            })?;
        advise_sidecar_dontneed(
            meta_file,
            RABITQ_PLUS_SIDECAR_HEADER_BYTES as u64,
            meta_bytes.len(),
        );
        for row in 0..header.rows {
            let base = row * RABITQ_PLUS_SIDECAR_META_ROW_BYTES;
            norms[row] = f32::from_le_bytes(meta_bytes[base..base + 4].try_into().unwrap());
            deltas[row] = f32::from_le_bytes(meta_bytes[base + 4..base + 8].try_into().unwrap());
            code_sums[row] =
                f32::from_le_bytes(meta_bytes[base + 8..base + 12].try_into().unwrap());
        }

        Ok(Self {
            prefix: sidecar.prefix,
            header: sidecar.header,
            storage: RabitqPlusSidecarStorage::Resident {
                codes,
                norms,
                deltas,
                code_sums,
                preload_wall: preload_start.elapsed(),
            },
        })
    }

    pub(crate) fn preload_prefix_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        rotator: PiPNNGpuSubtreeRabitqPlusRotator,
        seed: u64,
        resident_rows: usize,
    ) -> AnnResult<Self> {
        let resident_rows = resident_rows.min(rows);
        if resident_rows == 0 {
            return Self::open_validated(prefix, rows, dim, bits, rotator, seed);
        }
        if resident_rows >= rows {
            return Self::preload_validated(prefix, rows, dim, bits, rotator, seed);
        }

        let preload_start = Instant::now();
        let sidecar = Self::open_validated(prefix, rows, dim, bits, rotator, seed)?;
        let header = sidecar.header.clone();
        let mut codes = vec![0_i8; resident_rows.saturating_mul(header.padded_dim)];
        let mut norms = vec![0.0_f32; resident_rows];
        let mut deltas = vec![0.0_f32; resident_rows];
        let mut code_sums = vec![0.0_f32; resident_rows];

        let RabitqPlusSidecarStorage::File {
            codes_file,
            meta_file,
        } = &sidecar.storage
        else {
            unreachable!("newly opened sidecar is always file-backed")
        };

        let codes_bytes =
            unsafe { std::slice::from_raw_parts_mut(codes.as_mut_ptr().cast::<u8>(), codes.len()) };
        codes_file.read_exact_at(codes_bytes, 0).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload dense-i8 RabitQ+ sidecar resident prefix codes '{}': {err}",
                sidecar_codes_path(prefix).display()
            ))
        })?;
        advise_sidecar_dontneed(codes_file, 0, codes_bytes.len());

        let mut meta_bytes =
            vec![0_u8; resident_rows.saturating_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES)];
        meta_file
            .read_exact_at(&mut meta_bytes, RABITQ_PLUS_SIDECAR_HEADER_BYTES as u64)
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload dense-i8 RabitQ+ sidecar resident prefix metadata '{}': {err}",
                    sidecar_meta_path(prefix).display()
                ))
            })?;
        advise_sidecar_dontneed(
            meta_file,
            RABITQ_PLUS_SIDECAR_HEADER_BYTES as u64,
            meta_bytes.len(),
        );
        for row in 0..resident_rows {
            let base = row * RABITQ_PLUS_SIDECAR_META_ROW_BYTES;
            norms[row] = f32::from_le_bytes(meta_bytes[base..base + 4].try_into().unwrap());
            deltas[row] = f32::from_le_bytes(meta_bytes[base + 4..base + 8].try_into().unwrap());
            code_sums[row] =
                f32::from_le_bytes(meta_bytes[base + 8..base + 12].try_into().unwrap());
        }

        let RabitqPlusSidecarStorage::File {
            codes_file,
            meta_file,
        } = sidecar.storage
        else {
            unreachable!("newly opened sidecar is always file-backed")
        };
        Ok(Self {
            prefix: sidecar.prefix,
            header: sidecar.header,
            storage: RabitqPlusSidecarStorage::PartialResident {
                codes_file,
                meta_file,
                resident_rows,
                codes,
                norms,
                deltas,
                code_sums,
                preload_wall: preload_start.elapsed(),
            },
        })
    }

    pub(crate) fn header(&self) -> &RabitqPlusSidecarHeader {
        &self.header
    }

    pub(crate) fn prefix(&self) -> &Path {
        &self.prefix
    }

    pub(crate) fn is_resident(&self) -> bool {
        matches!(
            self.storage,
            RabitqPlusSidecarStorage::PartialResident { .. }
                | RabitqPlusSidecarStorage::Resident { .. }
        )
    }

    pub(crate) fn is_fully_resident(&self) -> bool {
        matches!(self.storage, RabitqPlusSidecarStorage::Resident { .. })
    }

    pub(crate) fn resident_rows(&self) -> usize {
        match &self.storage {
            RabitqPlusSidecarStorage::File { .. } => 0,
            RabitqPlusSidecarStorage::PartialResident { resident_rows, .. } => *resident_rows,
            RabitqPlusSidecarStorage::Resident { .. } => self.header.rows,
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.resident_rows().saturating_mul(
            self.header
                .padded_dim
                .saturating_add(RABITQ_PLUS_SIDECAR_META_ROW_BYTES),
        )
    }

    pub(crate) fn preload_wall(&self) -> Duration {
        match &self.storage {
            RabitqPlusSidecarStorage::File { .. } => Duration::ZERO,
            RabitqPlusSidecarStorage::PartialResident { preload_wall, .. } => *preload_wall,
            RabitqPlusSidecarStorage::Resident { preload_wall, .. } => *preload_wall,
        }
    }

    pub(crate) fn gather(&self, global_ids: &[u32]) -> AnnResult<RabitqPlusGatheredInput> {
        let gather_start = Instant::now();
        let rows = global_ids.len();
        let mut codes = vec![0_i8; rows.saturating_mul(self.header.padded_dim)];
        let mut norms = vec![0.0_f32; rows];
        let mut deltas = vec![0.0_f32; rows];
        let mut code_sums = vec![0.0_f32; rows];

        let mut resident_rows = 0usize;
        let mut file_rows = 0usize;
        let mut run_start = 0usize;
        while run_start < rows {
            let first_id = global_ids[run_start] as usize;
            if first_id >= self.header.rows {
                return Err(AnnError::log_index_error(format!(
                    "dense-i8 RabitQ+ sidecar gather id {} out of bounds for {} rows",
                    first_id, self.header.rows
                )));
            }
            let mut run_end = run_start + 1;
            while run_end < rows && global_ids[run_end] == global_ids[run_end - 1].saturating_add(1)
            {
                if global_ids[run_end] as usize >= self.header.rows {
                    return Err(AnnError::log_index_error(format!(
                        "dense-i8 RabitQ+ sidecar gather id {} out of bounds for {} rows",
                        global_ids[run_end], self.header.rows
                    )));
                }
                run_end += 1;
            }
            self.gather_code_run(first_id, run_start, run_end, &mut codes)?;
            self.gather_meta_run(
                first_id,
                run_start,
                run_end,
                &mut norms,
                &mut deltas,
                &mut code_sums,
            )?;
            let (resident, file) = self.source_rows_for_run(first_id, run_end - run_start);
            resident_rows = resident_rows.saturating_add(resident);
            file_rows = file_rows.saturating_add(file);
            run_start = run_end;
        }
        let gather_wall = gather_start.elapsed();
        let (resident_gather_wall, file_gather_wall) =
            split_gather_wall(gather_wall, resident_rows, file_rows);

        Ok(RabitqPlusGatheredInput {
            global_ids: global_ids.to_vec(),
            codes,
            deltas,
            code_sums,
            norms,
            dim: self.header.dim,
            padded_dim: self.header.padded_dim,
            bits: self.header.bits,
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
            RabitqPlusSidecarStorage::File { .. } => (0, run_len),
            RabitqPlusSidecarStorage::Resident { .. } => (run_len, 0),
            RabitqPlusSidecarStorage::PartialResident { resident_rows, .. } => {
                let run_end_id = first_id.saturating_add(run_len);
                let resident_end = run_end_id.min(*resident_rows);
                let resident = resident_end.saturating_sub(first_id);
                (resident, run_len.saturating_sub(resident))
            }
        }
    }

    fn gather_code_run(
        &self,
        first_id: usize,
        run_start: usize,
        run_end: usize,
        out: &mut [i8],
    ) -> AnnResult<()> {
        let run_len = run_end - run_start;
        let row_bytes = self.header.padded_dim;
        let src_offset = first_id.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error("dense-i8 RabitQ+ sidecar code offset overflows".to_string())
        })?;
        let dst_start = run_start.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error(
                "dense-i8 RabitQ+ sidecar output offset overflows".to_string(),
            )
        })?;
        let byte_len = run_len.checked_mul(row_bytes).ok_or_else(|| {
            AnnError::log_index_error(
                "dense-i8 RabitQ+ sidecar code run length overflows".to_string(),
            )
        })?;
        let dst = &mut out[dst_start..dst_start + byte_len];
        match &self.storage {
            RabitqPlusSidecarStorage::File { codes_file, .. } => {
                let dst_bytes = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len())
                };
                codes_file
                    .read_exact_at(dst_bytes, src_offset as u64)
                    .map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read dense-i8 RabitQ+ sidecar codes at offset {src_offset}: {err}"
                        ))
                    })
            }
            RabitqPlusSidecarStorage::Resident { codes, .. } => {
                dst.copy_from_slice(&codes[src_offset..src_offset + byte_len]);
                Ok(())
            }
            RabitqPlusSidecarStorage::PartialResident {
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
                            "dense-i8 RabitQ+ sidecar partial code offset overflows".to_string(),
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
                            "Failed to read dense-i8 RabitQ+ sidecar suffix codes at offset {file_offset}: {err}"
                        ))
                    })?;
                }
                Ok(())
            }
        }
    }

    fn gather_meta_run(
        &self,
        first_id: usize,
        run_start: usize,
        run_end: usize,
        norms: &mut [f32],
        deltas: &mut [f32],
        code_sums: &mut [f32],
    ) -> AnnResult<()> {
        let run_len = run_end - run_start;
        let src_offset = RABITQ_PLUS_SIDECAR_HEADER_BYTES
            .checked_add(first_id.saturating_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES))
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "dense-i8 RabitQ+ sidecar meta offset overflows".to_string(),
                )
            })?;
        let byte_len = run_len
            .checked_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "dense-i8 RabitQ+ sidecar meta run length overflows".to_string(),
                )
            })?;
        match &self.storage {
            RabitqPlusSidecarStorage::File { meta_file, .. } => {
                let mut buf = vec![0_u8; byte_len];
                meta_file
                    .read_exact_at(&mut buf, src_offset as u64)
                    .map_err(|err| {
                        AnnError::log_index_error(format!(
                            "Failed to read dense-i8 RabitQ+ sidecar metadata at offset {src_offset}: {err}"
                        ))
                    })?;
                for row in 0..run_len {
                    let base = row * RABITQ_PLUS_SIDECAR_META_ROW_BYTES;
                    norms[run_start + row] =
                        f32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
                    deltas[run_start + row] =
                        f32::from_le_bytes(buf[base + 4..base + 8].try_into().unwrap());
                    code_sums[run_start + row] =
                        f32::from_le_bytes(buf[base + 8..base + 12].try_into().unwrap());
                }
            }
            RabitqPlusSidecarStorage::Resident {
                norms: all_norms,
                deltas: all_deltas,
                code_sums: all_code_sums,
                ..
            } => {
                let src_end = first_id + run_len;
                norms[run_start..run_end].copy_from_slice(&all_norms[first_id..src_end]);
                deltas[run_start..run_end].copy_from_slice(&all_deltas[first_id..src_end]);
                code_sums[run_start..run_end].copy_from_slice(&all_code_sums[first_id..src_end]);
            }
            RabitqPlusSidecarStorage::PartialResident {
                meta_file,
                resident_rows,
                norms: all_norms,
                deltas: all_deltas,
                code_sums: all_code_sums,
                ..
            } => {
                let resident_len = resident_rows.saturating_sub(first_id).min(run_len);
                if resident_len > 0 {
                    let src_start = first_id;
                    let src_end = first_id + resident_len;
                    let dst_end = run_start + resident_len;
                    norms[run_start..dst_end].copy_from_slice(&all_norms[src_start..src_end]);
                    deltas[run_start..dst_end].copy_from_slice(&all_deltas[src_start..src_end]);
                    code_sums[run_start..dst_end]
                        .copy_from_slice(&all_code_sums[src_start..src_end]);
                }
                if resident_len < run_len {
                    let file_first_id = first_id + resident_len;
                    let file_run_start = run_start + resident_len;
                    let file_run_len = run_len - resident_len;
                    let file_src_offset = RABITQ_PLUS_SIDECAR_HEADER_BYTES
                        .checked_add(
                            file_first_id.saturating_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES),
                        )
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "dense-i8 RabitQ+ sidecar partial meta offset overflows"
                                    .to_string(),
                            )
                        })?;
                    let file_byte_len = file_run_len
                        .checked_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES)
                        .ok_or_else(|| {
                            AnnError::log_index_error(
                                "dense-i8 RabitQ+ sidecar partial meta run length overflows"
                                    .to_string(),
                            )
                        })?;
                    let mut buf = vec![0_u8; file_byte_len];
                    meta_file
                        .read_exact_at(&mut buf, file_src_offset as u64)
                        .map_err(|err| {
                            AnnError::log_index_error(format!(
                                "Failed to read dense-i8 RabitQ+ sidecar suffix metadata at offset {file_src_offset}: {err}"
                            ))
                        })?;
                    for row in 0..file_run_len {
                        let base = row * RABITQ_PLUS_SIDECAR_META_ROW_BYTES;
                        norms[file_run_start + row] =
                            f32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
                        deltas[file_run_start + row] =
                            f32::from_le_bytes(buf[base + 4..base + 8].try_into().unwrap());
                        code_sums[file_run_start + row] =
                            f32::from_le_bytes(buf[base + 8..base + 12].try_into().unwrap());
                    }
                }
            }
        }
        Ok(())
    }
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

impl RabitqSplitSidecarHeader {
    pub(crate) fn validate_for_dataset(
        &self,
        rows: usize,
        dim: usize,
        bits: usize,
        rotator: PiPNNGpuSubtreeRabitqPlusRotator,
        seed: u64,
    ) -> AnnResult<()> {
        if self.rows != rows
            || self.dim != dim
            || self.source_rows != rows
            || self.source_dim != dim
        {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar shape mismatch: sidecar rows={} dim={} source_rows={} source_dim={} expected rows={rows} dim={dim}",
                self.rows, self.dim, self.source_rows, self.source_dim
            )));
        }
        if self.bits != bits {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar bits mismatch: sidecar={} expected={bits}",
                self.bits
            )));
        }
        if self.rotator != rotator {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar rotator mismatch: sidecar={} expected={}",
                self.rotator.as_str(),
                rotator.as_str()
            )));
        }
        if self.seed != seed {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar seed mismatch: sidecar={} expected={seed}",
                self.seed
            )));
        }
        Ok(())
    }
}

impl RabitqSplitSidecar {
    pub(crate) fn preload_validated(
        prefix: &Path,
        rows: usize,
        dim: usize,
        bits: usize,
        rotator: PiPNNGpuSubtreeRabitqPlusRotator,
        seed: u64,
    ) -> AnnResult<Self> {
        let preload_start = Instant::now();
        let meta_path = split_sidecar_meta_path(prefix);
        let mut meta_file = File::open(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open RabitQ-Library split sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
        let header = read_rabitq_split_sidecar_header(&mut meta_file)?;
        header.validate_for_dataset(rows, dim, bits, rotator, seed)?;

        let batch_path = split_sidecar_batch_path(prefix);
        let ex_path = split_sidecar_ex_path(prefix);
        let mut batch_file = File::open(&batch_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open RabitQ-Library split sidecar batch '{}': {err}",
                batch_path.display()
            ))
        })?;
        let batch_len = batch_file
            .metadata()
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to stat RabitQ-Library split sidecar batch '{}': {err}",
                    batch_path.display()
                ))
            })?
            .len() as usize;
        if batch_len != header.batch_total_bytes {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar batch size mismatch: file={} header={}",
                batch_len, header.batch_total_bytes
            )));
        }
        let mut ex_file = File::open(&ex_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open RabitQ-Library split sidecar ex '{}': {err}",
                ex_path.display()
            ))
        })?;
        let ex_len = ex_file
            .metadata()
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to stat RabitQ-Library split sidecar ex '{}': {err}",
                    ex_path.display()
                ))
            })?
            .len() as usize;
        if ex_len != header.ex_total_bytes {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar ex size mismatch: file={} header={}",
                ex_len, header.ex_total_bytes
            )));
        }

        let mut batch_data = vec![0_u8; header.batch_total_bytes];
        batch_file.read_exact(&mut batch_data).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to preload RabitQ-Library split sidecar batch '{}': {err}",
                batch_path.display()
            ))
        })?;
        let mut ex_data = vec![0_u8; header.ex_total_bytes];
        if !ex_data.is_empty() {
            ex_file.read_exact(&mut ex_data).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to preload RabitQ-Library split sidecar ex '{}': {err}",
                    ex_path.display()
                ))
            })?;
        }

        Ok(Self {
            prefix: prefix.to_path_buf(),
            header,
            batch_data,
            ex_data,
            preload_wall: preload_start.elapsed(),
        })
    }

    pub(crate) fn header(&self) -> &RabitqSplitSidecarHeader {
        &self.header
    }

    pub(crate) fn prefix(&self) -> &Path {
        &self.prefix
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.batch_data.len().saturating_add(self.ex_data.len())
    }

    pub(crate) fn preload_wall(&self) -> Duration {
        self.preload_wall
    }

    pub(crate) fn batch_data(&self) -> &[u8] {
        &self.batch_data
    }

    pub(crate) fn ex_data(&self) -> &[u8] {
        &self.ex_data
    }

    pub(crate) fn assign_d0(
        &self,
        leaders_row_major: &[f32],
        leader_count: usize,
        fanout: usize,
        survivors: usize,
    ) -> AnnResult<RabitqSplitD0Assignment> {
        if leaders_row_major.len() != leader_count.saturating_mul(self.header.dim) {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split D0 leader shape mismatch: leaders={} dim={} values={}",
                leader_count,
                self.header.dim,
                leaders_row_major.len()
            )));
        }
        if fanout == 0 || survivors < fanout || survivors > leader_count {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split D0 invalid fanout/survivors: fanout={fanout} survivors={survivors} leaders={leader_count}"
            )));
        }
        let mut assignments = vec![u16::MAX; self.header.rows.saturating_mul(fanout)];
        let mut coarse_seconds = 0.0_f64;
        let mut boost_seconds = 0.0_f64;
        let mut boosted_pairs = 0_u64;
        let ex_ptr = if self.ex_data.is_empty() {
            std::ptr::null()
        } else {
            self.ex_data.as_ptr()
        };
        let status = unsafe {
            crate::model::rabitq::ffi::rabitq_split_d0_assign_fht_kac_simd(
                self.batch_data.as_ptr(),
                ex_ptr,
                self.header.rows,
                self.header.dim,
                self.header.bits as libc::c_int,
                leaders_row_major.as_ptr(),
                leader_count,
                self.header.seed as libc::c_ulonglong,
                fanout,
                survivors,
                assignments.as_mut_ptr(),
                &mut coarse_seconds,
                &mut boost_seconds,
                &mut boosted_pairs,
            )
        };
        if status == 0 {
            return Err(AnnError::log_index_error(
                "RabitQ-Library split D0 assignment failed in RaBitQ-Library FFI".to_string(),
            ));
        }
        Ok(RabitqSplitD0Assignment {
            assignments,
            coarse_wall: Duration::from_secs_f64(coarse_seconds.max(0.0)),
            boost_wall: Duration::from_secs_f64(boost_seconds.max(0.0)),
            boosted_pairs,
        })
    }
}

pub(crate) fn rabitq_padded_dim(dim: usize, k_align: usize) -> AnnResult<usize> {
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "RaBitQ quantization requires dim > 0".to_string(),
        ));
    }
    let k_align = k_align.max(1);
    Ok(dim.div_ceil(k_align) * k_align)
}

pub(crate) fn quantize_rabitq_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    total_bits: usize,
    k_align: usize,
) -> AnnResult<RabitqMatrix> {
    if points_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "RaBitQ matrix shape mismatch: rows={rows} dim={dim} values={}",
            points_row_major.len()
        )));
    }
    validate_total_bits(total_bits)?;
    let padded_dim = rabitq_padded_dim(dim, k_align)?;
    let ex_bits = total_bits - 1;
    let t_const = if ex_bits > 0 {
        rabitq_const_scaling_factor(dim, ex_bits)?
    } else {
        -1.0
    };
    let mut codes = vec![0_i8; rows.saturating_mul(padded_dim)];
    let mut deltas = vec![0.0_f32; rows];
    let mut code_sums = vec![0.0_f32; rows];

    for row in 0..rows {
        let input = &points_row_major[row * dim..(row + 1) * dim];
        let output = &mut codes[row * padded_dim..(row + 1) * padded_dim];
        let (delta, code_sum) = quantize_rabitq_row(input, output, total_bits, t_const)?;
        deltas[row] = delta;
        code_sums[row] = code_sum;
    }

    Ok(RabitqMatrix {
        rows,
        dim,
        padded_dim,
        total_bits,
        codes,
        deltas,
        code_sums,
    })
}

pub(crate) fn quantize_rabitq_plus_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    total_bits: usize,
    k_align: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
) -> AnnResult<RabitqMatrix> {
    if points_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ matrix shape mismatch: rows={rows} dim={dim} values={}",
            points_row_major.len()
        )));
    }
    validate_rabitq_plus_bits(total_bits)?;
    if rotator == PiPNNGpuSubtreeRabitqPlusRotator::None {
        return quantize_rabitq_matrix(points_row_major, rows, dim, total_bits, k_align);
    }
    if !dim.is_power_of_two() {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ fht-kac rotator currently requires power-of-two dim, got {dim}; use --gpu-subtree-rabitq-plus-rotator none for this dataset"
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, k_align)?;
    if padded_dim != dim {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ fht-kac rotator requires padded_dim == dim, got dim={dim} padded_dim={padded_dim}"
        )));
    }
    let ex_bits = total_bits - 1;
    let t_const = if ex_bits > 0 {
        rabitq_const_scaling_factor(dim, ex_bits)?
    } else {
        -1.0
    };
    if let Some(matrix) = quantize_rabitq_plus_matrix_simd_fht_kac(
        points_row_major,
        rows,
        dim,
        padded_dim,
        total_bits,
        t_const,
        rotator,
        seed,
    )? {
        return Ok(matrix);
    }
    quantize_rabitq_plus_matrix_scalar_fht_kac(
        points_row_major,
        rows,
        dim,
        padded_dim,
        total_bits,
        t_const,
        seed,
    )
}

fn quantize_rabitq_plus_matrix_scalar_fht_kac(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    padded_dim: usize,
    total_bits: usize,
    t_const: f64,
    seed: u64,
) -> AnnResult<RabitqMatrix> {
    let mut codes = vec![0_i8; rows.saturating_mul(padded_dim)];
    let mut deltas = vec![0.0_f32; rows];
    let mut code_sums = vec![0.0_f32; rows];
    let mut rotated = vec![0.0_f32; dim];

    for row in 0..rows {
        let input = &points_row_major[row * dim..(row + 1) * dim];
        rotate_rabitq_plus_fht_kac_row(input, &mut rotated, seed)?;
        let output = &mut codes[row * padded_dim..(row + 1) * padded_dim];
        let (delta, code_sum) = quantize_rabitq_row(&rotated, output, total_bits, t_const)?;
        deltas[row] = delta;
        code_sums[row] = code_sum;
    }

    Ok(RabitqMatrix {
        rows,
        dim,
        padded_dim,
        total_bits,
        codes,
        deltas,
        code_sums,
    })
}

#[allow(clippy::too_many_arguments)]
fn quantize_rabitq_plus_matrix_simd_fht_kac(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    padded_dim: usize,
    total_bits: usize,
    t_const: f64,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
) -> AnnResult<Option<RabitqMatrix>> {
    if rotator != PiPNNGpuSubtreeRabitqPlusRotator::FhtKac
        || dim != padded_dim
        || !dim.is_power_of_two()
    {
        return Ok(None);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (points_row_major, rows, total_bits, t_const, seed);
        Ok(None)
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mut codes = vec![0_i8; rows.saturating_mul(padded_dim)];
        let mut deltas = vec![0.0_f32; rows];
        let mut code_sums = vec![0.0_f32; rows];
        let status = unsafe {
            crate::model::rabitq::ffi::rabitq_plus_quantize_dense_fht_kac_simd(
                points_row_major.as_ptr(),
                rows,
                dim,
                total_bits as libc::c_int,
                t_const as libc::c_double,
                seed as libc::c_ulonglong,
                codes.as_mut_ptr().cast::<libc::c_schar>(),
                deltas.as_mut_ptr(),
                code_sums.as_mut_ptr(),
            )
        };
        if status == 0 {
            return Ok(None);
        }
        Ok(Some(RabitqMatrix {
            rows,
            dim,
            padded_dim,
            total_bits,
            codes,
            deltas,
            code_sums,
        }))
    }
}

pub(crate) fn quantize_rabitq_split_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    total_bits: usize,
    k_align: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
) -> AnnResult<RabitqSplitMatrix> {
    if points_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split matrix shape mismatch: rows={rows} dim={dim} values={}",
            points_row_major.len()
        )));
    }
    validate_rabitq_plus_bits(total_bits)?;
    if rotator != PiPNNGpuSubtreeRabitqPlusRotator::FhtKac {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split matrix requires fht_kac rotator, got {}",
            rotator.as_str()
        )));
    }
    if !dim.is_power_of_two() {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split matrix fht-kac rotator currently requires power-of-two dim, got {dim}"
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, k_align)?;
    if padded_dim != dim {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split matrix fht-kac rotator requires padded_dim == dim, got dim={dim} padded_dim={padded_dim}"
        )));
    }
    let ex_bits = total_bits - 1;
    let t_const = if ex_bits > 0 {
        rabitq_const_scaling_factor(dim, ex_bits)?
    } else {
        -1.0
    };
    let batch_bytes = unsafe { crate::model::rabitq::ffi::rabitq_split_batch_data_bytes(dim) };
    let ex_row_bytes = unsafe {
        crate::model::rabitq::ffi::rabitq_split_ex_data_bytes(dim, ex_bits as libc::c_int)
    };
    if batch_bytes == 0 || (ex_bits > 0 && ex_row_bytes == 0) {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split matrix produced invalid layout: dim={dim} ex_bits={ex_bits} batch_bytes={batch_bytes} ex_row_bytes={ex_row_bytes}"
        )));
    }
    let batch_count = rows.div_ceil(RABITQ_SPLIT_BATCH_ROWS);
    let mut batch_data = vec![0_u8; batch_count.saturating_mul(batch_bytes)];
    let mut ex_data = vec![0_u8; rows.saturating_mul(ex_row_bytes)];
    if rows > 0 {
        let status = unsafe {
            crate::model::rabitq::ffi::rabitq_split_quantize_fht_kac_simd(
                points_row_major.as_ptr(),
                rows,
                dim,
                total_bits as libc::c_int,
                t_const as libc::c_double,
                seed as libc::c_ulonglong,
                batch_data.as_mut_ptr(),
                if ex_data.is_empty() {
                    std::ptr::null_mut()
                } else {
                    ex_data.as_mut_ptr()
                },
            )
        };
        if status == 0 {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split matrix quantization failed: rows={rows} dim={dim} bits={total_bits}"
            )));
        }
    }

    let query = prepare_rabitq_split_query_matrix(
        points_row_major,
        rows,
        dim,
        total_bits,
        k_align,
        rotator,
        seed,
    )?;

    Ok(RabitqSplitMatrix {
        rows,
        dim,
        padded_dim,
        total_bits,
        ex_bits,
        batch_count,
        batch_bytes,
        ex_row_bytes,
        batch_data,
        ex_data,
        rotated: query.rotated,
        query_kbxsumq: query.query_kbxsumq,
    })
}

pub(crate) fn prepare_rabitq_split_query_matrix(
    points_row_major: &[f32],
    rows: usize,
    dim: usize,
    total_bits: usize,
    k_align: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
) -> AnnResult<RabitqSplitQueryMatrix> {
    if points_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split query matrix shape mismatch: rows={rows} dim={dim} values={}",
            points_row_major.len()
        )));
    }
    validate_rabitq_plus_bits(total_bits)?;
    if rotator != PiPNNGpuSubtreeRabitqPlusRotator::FhtKac {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split query matrix requires fht_kac rotator, got {}",
            rotator.as_str()
        )));
    }
    if !dim.is_power_of_two() {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split query matrix fht-kac rotator currently requires power-of-two dim, got {dim}"
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, k_align)?;
    if padded_dim != dim {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split query matrix fht-kac rotator requires padded_dim == dim, got dim={dim} padded_dim={padded_dim}"
        )));
    }
    let ex_bits = total_bits - 1;
    let mut rotated = vec![0.0_f32; rows.saturating_mul(padded_dim)];
    let mut query_kbxsumq = vec![0.0_f32; rows];
    let cb = -(((1_u32 << (ex_bits + 1)) - 1) as f32) / 2.0;
    for row in 0..rows {
        let input = &points_row_major[row * dim..(row + 1) * dim];
        let output = &mut rotated[row * padded_dim..(row + 1) * padded_dim];
        rotate_rabitq_plus_fht_kac_row(input, output, seed)?;
        query_kbxsumq[row] = output.iter().copied().sum::<f32>() * cb;
    }
    Ok(RabitqSplitQueryMatrix {
        rows,
        dim,
        padded_dim,
        total_bits,
        ex_bits,
        rotated,
        query_kbxsumq,
    })
}

pub(crate) fn sidecar_codes_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.codes", prefix.display()))
}

pub(crate) fn sidecar_meta_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.meta", prefix.display()))
}

pub(crate) fn split_sidecar_batch_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.split.batch", prefix.display()))
}

pub(crate) fn split_sidecar_ex_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.split.ex", prefix.display()))
}

pub(crate) fn split_sidecar_meta_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.split.meta", prefix.display()))
}

pub(crate) fn create_rabitq_plus_sidecar_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    bits: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
    source_fbin_bytes: u64,
) -> AnnResult<RabitqPlusSidecarCreateReport> {
    validate_rabitq_plus_bits(bits)?;
    let total_start = Instant::now();
    let rows = dataset.len();
    let dim = dataset.dim();
    if rows > u32::MAX as usize {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ sidecar currently supports at most {} rows, got {rows}",
            u32::MAX
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, RABITQ_DEFAULT_K_ALIGN)?;
    let codes_bytes = rows.checked_mul(padded_dim).ok_or_else(|| {
        AnnError::log_index_error("dense-i8 RabitQ+ sidecar codes byte count overflows".to_string())
    })?;
    let meta_rows_bytes = rows
        .checked_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES)
        .ok_or_else(|| {
            AnnError::log_index_error(
                "dense-i8 RabitQ+ sidecar metadata byte count overflows".to_string(),
            )
        })?;
    if let Some(parent) = prefix
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create dense-i8 RabitQ+ sidecar parent '{}': {err}",
                parent.display()
            ))
        })?;
    }
    let codes_path = sidecar_codes_path(prefix);
    let meta_path = sidecar_meta_path(prefix);
    let mut codes_writer = BufWriter::new(File::create(&codes_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create dense-i8 RabitQ+ sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?);
    let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create dense-i8 RabitQ+ sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?);
    meta_writer
        .write_all(&vec![0_u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES])
        .map_err(|err| {
            AnnError::log_index_error(format!("Failed to reserve dense-i8 RabitQ+ header: {err}"))
        })?;

    let mut hash = fnv1a_update_u64(FNV_OFFSET_BASIS, rows as u64);
    hash = fnv1a_update_u64(hash, dim as u64);
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let mut source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut code_output_wall = Duration::ZERO;
    let mut norm_wall = Duration::ZERO;
    let mut meta_output_wall = Duration::ZERO;
    let mut header_write_wall = Duration::ZERO;
    let mut write_wall = Duration::ZERO;
    let mut row_start = 0usize;
    while row_start < rows {
        let count = (rows - row_start).min(RABITQ_PLUS_SIDECAR_CREATE_CHUNK_ROWS);
        let chunk_alloc_start = Instant::now();
        let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
        chunk_alloc_wall += chunk_alloc_start.elapsed();
        let raw_read_start = Instant::now();
        dataset.read_range_into(row_start as u32, count, &mut chunk)?;
        raw_read_wall += raw_read_start.elapsed();
        let source_hash_start = Instant::now();
        hash = fnv1a_update_f32_slice(hash, &chunk);
        source_hash_wall += source_hash_start.elapsed();

        let quantize_start = Instant::now();
        let matrix = quantize_rabitq_plus_matrix(
            &chunk,
            count,
            dim,
            bits,
            RABITQ_DEFAULT_K_ALIGN,
            rotator,
            seed,
        )?;
        quantize_wall += quantize_start.elapsed();

        let write_start = Instant::now();
        let code_bytes = unsafe {
            std::slice::from_raw_parts(matrix.codes.as_ptr().cast::<u8>(), matrix.codes.len())
        };
        let code_output_start = Instant::now();
        codes_writer.write_all(code_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write dense-i8 RabitQ+ sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        code_output_wall += code_output_start.elapsed();
        let norm_start = Instant::now();
        let mut norms = Vec::with_capacity(count);
        for row in 0..count {
            let source = &chunk[row * dim..(row + 1) * dim];
            norms.push(source.iter().map(|&value| value * value).sum::<f32>());
        }
        norm_wall += norm_start.elapsed();
        let meta_output_start = Instant::now();
        for row in 0..count {
            meta_writer
                .write_all(&norms[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write dense-i8 RabitQ+ sidecar norm '{}': {err}",
                        meta_path.display()
                    ))
                })?;
            meta_writer
                .write_all(&matrix.deltas[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write dense-i8 RabitQ+ sidecar delta '{}': {err}",
                        meta_path.display()
                    ))
                })?;
            meta_writer
                .write_all(&matrix.code_sums[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write dense-i8 RabitQ+ sidecar code_sum '{}': {err}",
                        meta_path.display()
                    ))
                })?;
        }
        meta_output_wall += meta_output_start.elapsed();
        write_wall += write_start.elapsed();
        row_start += count;
    }
    codes_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush dense-i8 RabitQ+ sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?;
    meta_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush dense-i8 RabitQ+ sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    drop(codes_writer);
    drop(meta_writer);

    let header = RabitqPlusSidecarHeader {
        rows,
        dim,
        padded_dim,
        bits,
        rotator,
        seed,
        k_align: RABITQ_DEFAULT_K_ALIGN,
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
                "Failed to reopen dense-i8 RabitQ+ sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
    meta.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek dense-i8 RabitQ+ sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    meta.write_all(&encode_rabitq_plus_sidecar_header(&header)?)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write dense-i8 RabitQ+ sidecar header '{}': {err}",
                meta_path.display()
            ))
        })?;
    header_write_wall += header_write_start.elapsed();

    Ok(RabitqPlusSidecarCreateReport {
        header,
        total_wall: total_start.elapsed(),
        chunk_alloc_wall,
        raw_read_wall,
        source_hash_wall,
        quantize_wall,
        code_output_wall,
        norm_wall,
        meta_output_wall,
        header_write_wall,
        write_wall,
    })
}

pub(crate) fn create_rabitq_split_sidecar_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    bits: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
    source_fbin_bytes: u64,
) -> AnnResult<RabitqSplitSidecarCreateReport> {
    validate_rabitq_plus_bits(bits)?;
    let total_start = Instant::now();
    let rows = dataset.len();
    let dim = dataset.dim();
    if rows > u32::MAX as usize {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split sidecar currently supports at most {} rows, got {rows}",
            u32::MAX
        )));
    }
    if rotator != PiPNNGpuSubtreeRabitqPlusRotator::FhtKac {
        return Err(AnnError::log_index_error(
            "RabitQ-Library split sidecar currently requires fht-kac rotator".to_string(),
        ));
    }
    if !dim.is_power_of_two() {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split sidecar fht-kac rotator requires power-of-two dim, got {dim}"
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, RABITQ_DEFAULT_K_ALIGN)?;
    if padded_dim != dim {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split sidecar requires padded_dim == dim, got dim={dim} padded_dim={padded_dim}"
        )));
    }
    let ex_bits = bits - 1;
    let batch_bytes = unsafe { crate::model::rabitq::ffi::rabitq_split_batch_data_bytes(dim) };
    let ex_row_bytes = unsafe {
        crate::model::rabitq::ffi::rabitq_split_ex_data_bytes(dim, ex_bits as libc::c_int)
    };
    if batch_bytes == 0 || (ex_bits > 0 && ex_row_bytes == 0) {
        return Err(AnnError::log_index_error(format!(
            "RabitQ-Library split sidecar library returned invalid layout bytes: batch_bytes={batch_bytes} ex_row_bytes={ex_row_bytes}"
        )));
    }
    let batch_count = rows.div_ceil(RABITQ_SPLIT_BATCH_ROWS);
    let batch_total_bytes = batch_count.checked_mul(batch_bytes).ok_or_else(|| {
        AnnError::log_index_error(
            "RabitQ-Library split sidecar batch byte count overflows".to_string(),
        )
    })?;
    let ex_total_bytes = rows.checked_mul(ex_row_bytes).ok_or_else(|| {
        AnnError::log_index_error(
            "RabitQ-Library split sidecar ex byte count overflows".to_string(),
        )
    })?;
    if let Some(parent) = prefix
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create RabitQ-Library split sidecar parent '{}': {err}",
                parent.display()
            ))
        })?;
    }

    let batch_path = split_sidecar_batch_path(prefix);
    let ex_path = split_sidecar_ex_path(prefix);
    let meta_path = split_sidecar_meta_path(prefix);
    let mut batch_writer = BufWriter::new(File::create(&batch_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create RabitQ-Library split sidecar batch '{}': {err}",
            batch_path.display()
        ))
    })?);
    let mut ex_writer = BufWriter::new(File::create(&ex_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create RabitQ-Library split sidecar ex '{}': {err}",
            ex_path.display()
        ))
    })?);
    let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create RabitQ-Library split sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?);

    let mut hash = fnv1a_update_u64(FNV_OFFSET_BASIS, rows as u64);
    hash = fnv1a_update_u64(hash, dim as u64);
    let t_const = if ex_bits > 0 {
        rabitq_const_scaling_factor(dim, ex_bits)?
    } else {
        -1.0
    };
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let mut source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut batch_output_wall = Duration::ZERO;
    let mut ex_output_wall = Duration::ZERO;
    let mut write_wall = Duration::ZERO;

    let mut row_start = 0usize;
    while row_start < rows {
        let count = (rows - row_start).min(RABITQ_PLUS_SIDECAR_CREATE_CHUNK_ROWS);
        debug_assert!(count % RABITQ_SPLIT_BATCH_ROWS == 0 || row_start + count == rows);
        let chunk_batch_count = count.div_ceil(RABITQ_SPLIT_BATCH_ROWS);
        let chunk_alloc_start = Instant::now();
        let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
        let mut batch_chunk = vec![0_u8; chunk_batch_count.saturating_mul(batch_bytes)];
        let mut ex_chunk = vec![0_u8; count.saturating_mul(ex_row_bytes)];
        chunk_alloc_wall += chunk_alloc_start.elapsed();

        let raw_read_start = Instant::now();
        dataset.read_range_into(row_start as u32, count, &mut chunk)?;
        raw_read_wall += raw_read_start.elapsed();

        let source_hash_start = Instant::now();
        hash = fnv1a_update_f32_slice(hash, &chunk);
        source_hash_wall += source_hash_start.elapsed();

        let quantize_start = Instant::now();
        let ex_ptr = if ex_chunk.is_empty() {
            std::ptr::null_mut()
        } else {
            ex_chunk.as_mut_ptr()
        };
        let status = unsafe {
            crate::model::rabitq::ffi::rabitq_split_quantize_fht_kac_simd(
                chunk.as_ptr(),
                count,
                dim,
                bits as libc::c_int,
                t_const as libc::c_double,
                seed as libc::c_ulonglong,
                batch_chunk.as_mut_ptr(),
                ex_ptr,
            )
        };
        quantize_wall += quantize_start.elapsed();
        if status == 0 {
            return Err(AnnError::log_index_error(format!(
                "RabitQ-Library split sidecar quantization failed at row_start={row_start} count={count}"
            )));
        }

        let write_start = Instant::now();
        let batch_output_start = Instant::now();
        batch_writer.write_all(&batch_chunk).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write RabitQ-Library split sidecar batch '{}': {err}",
                batch_path.display()
            ))
        })?;
        batch_output_wall += batch_output_start.elapsed();
        let ex_output_start = Instant::now();
        if !ex_chunk.is_empty() {
            ex_writer.write_all(&ex_chunk).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to write RabitQ-Library split sidecar ex '{}': {err}",
                    ex_path.display()
                ))
            })?;
        }
        ex_output_wall += ex_output_start.elapsed();
        write_wall += write_start.elapsed();
        row_start += count;
    }

    batch_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush RabitQ-Library split sidecar batch '{}': {err}",
            batch_path.display()
        ))
    })?;
    ex_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush RabitQ-Library split sidecar ex '{}': {err}",
            ex_path.display()
        ))
    })?;

    let header = RabitqSplitSidecarHeader {
        rows,
        dim,
        padded_dim,
        bits,
        ex_bits,
        rotator,
        seed,
        k_align: RABITQ_DEFAULT_K_ALIGN,
        batch_rows: RABITQ_SPLIT_BATCH_ROWS,
        batch_count,
        batch_bytes,
        ex_row_bytes,
        source_fbin_bytes,
        source_rows: rows,
        source_dim: dim,
        source_fingerprint: hash,
        batch_total_bytes,
        ex_total_bytes,
    };
    let header_write_start = Instant::now();
    meta_writer
        .write_all(&encode_rabitq_split_sidecar_header(&header)?)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write RabitQ-Library split sidecar header '{}': {err}",
                meta_path.display()
            ))
        })?;
    meta_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush RabitQ-Library split sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    let header_write_wall = header_write_start.elapsed();

    Ok(RabitqSplitSidecarCreateReport {
        header,
        total_wall: total_start.elapsed(),
        chunk_alloc_wall,
        raw_read_wall,
        source_hash_wall,
        quantize_wall,
        batch_output_wall,
        ex_output_wall,
        header_write_wall,
        write_wall,
    })
}

pub(crate) fn create_rabitq_plus_sidecar_and_sketch_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    sketch_path: &Path,
    bits: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
    source_fbin_bytes: u64,
    params: &PiPNNParams,
    _sketch_io_cfg: DirectIoConfig,
) -> AnnResult<(RabitqPlusSidecarSketchBuildReport, DiskSketchStore)> {
    validate_rabitq_plus_bits(bits)?;
    let total_start = Instant::now();
    let rows = dataset.len();
    let dim = dataset.dim();
    if rows > u32::MAX as usize {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ sidecar currently supports at most {} rows, got {rows}",
            u32::MAX
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, RABITQ_DEFAULT_K_ALIGN)?;
    let codes_bytes = rows.checked_mul(padded_dim).ok_or_else(|| {
        AnnError::log_index_error("dense-i8 RabitQ+ sidecar codes byte count overflows".to_string())
    })?;
    let meta_rows_bytes = rows
        .checked_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES)
        .ok_or_else(|| {
            AnnError::log_index_error(
                "dense-i8 RabitQ+ sidecar metadata byte count overflows".to_string(),
            )
        })?;
    if let Some(parent) = prefix
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create dense-i8 RabitQ+ sidecar parent '{}': {err}",
                parent.display()
            ))
        })?;
    }
    let codes_path = sidecar_codes_path(prefix);
    let meta_path = sidecar_meta_path(prefix);
    let mut codes_writer = BufWriter::new(File::create(&codes_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create dense-i8 RabitQ+ sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?);
    let mut meta_writer = BufWriter::new(File::create(&meta_path).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to create dense-i8 RabitQ+ sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?);
    meta_writer
        .write_all(&vec![0_u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES])
        .map_err(|err| {
            AnnError::log_index_error(format!("Failed to reserve dense-i8 RabitQ+ header: {err}"))
        })?;

    let sketch_width = params.m_hash_bits;
    let hyperplanes = generate_hyperplanes(dim, sketch_width, params.random_seed);
    let sketch_values = rows.checked_mul(sketch_width).ok_or_else(|| {
        AnnError::log_index_error("Resident fused sketch value count overflows".to_string())
    })?;
    let mut sketch_data = Vec::with_capacity(sketch_values);

    let mut hash = fnv1a_update_u64(FNV_OFFSET_BASIS, rows as u64);
    hash = fnv1a_update_u64(hash, dim as u64);
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let mut source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut code_output_wall = Duration::ZERO;
    let mut norm_wall = Duration::ZERO;
    let mut meta_output_wall = Duration::ZERO;
    let mut header_write_wall = Duration::ZERO;
    let mut sidecar_write_wall = Duration::ZERO;
    let mut sketch_alloc_wall = Duration::ZERO;
    let mut sketch_compute_wall = Duration::ZERO;
    let mut sketch_extend_wall = Duration::ZERO;
    let sketch_write_wall = Duration::ZERO;
    let mut row_start = 0usize;
    while row_start < rows {
        let count = (rows - row_start).min(RABITQ_PLUS_SIDECAR_CREATE_CHUNK_ROWS);
        let chunk_alloc_start = Instant::now();
        let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
        chunk_alloc_wall += chunk_alloc_start.elapsed();
        let raw_read_start = Instant::now();
        dataset.read_range_into(row_start as u32, count, &mut chunk)?;
        raw_read_wall += raw_read_start.elapsed();
        let source_hash_start = Instant::now();
        hash = fnv1a_update_f32_slice(hash, &chunk);
        source_hash_wall += source_hash_start.elapsed();

        let quantize_start = Instant::now();
        let matrix = quantize_rabitq_plus_matrix(
            &chunk,
            count,
            dim,
            bits,
            RABITQ_DEFAULT_K_ALIGN,
            rotator,
            seed,
        )?;
        quantize_wall += quantize_start.elapsed();

        let sidecar_write_start = Instant::now();
        let code_bytes = unsafe {
            std::slice::from_raw_parts(matrix.codes.as_ptr().cast::<u8>(), matrix.codes.len())
        };
        let code_output_start = Instant::now();
        codes_writer.write_all(code_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write dense-i8 RabitQ+ sidecar codes '{}': {err}",
                codes_path.display()
            ))
        })?;
        code_output_wall += code_output_start.elapsed();
        let norm_start = Instant::now();
        let mut norms = Vec::with_capacity(count);
        for row in 0..count {
            let source = &chunk[row * dim..(row + 1) * dim];
            norms.push(source.iter().map(|&value| value * value).sum::<f32>());
        }
        norm_wall += norm_start.elapsed();
        let meta_output_start = Instant::now();
        for row in 0..count {
            meta_writer
                .write_all(&norms[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write dense-i8 RabitQ+ sidecar norm '{}': {err}",
                        meta_path.display()
                    ))
                })?;
            meta_writer
                .write_all(&matrix.deltas[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write dense-i8 RabitQ+ sidecar delta '{}': {err}",
                        meta_path.display()
                    ))
                })?;
            meta_writer
                .write_all(&matrix.code_sums[row].to_le_bytes())
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write dense-i8 RabitQ+ sidecar code_sum '{}': {err}",
                        meta_path.display()
                    ))
                })?;
        }
        meta_output_wall += meta_output_start.elapsed();
        sidecar_write_wall += sidecar_write_start.elapsed();

        let sketch_alloc_start = Instant::now();
        let mut sketch_chunk = vec![0.0_f32; count.saturating_mul(sketch_width)];
        sketch_alloc_wall += sketch_alloc_start.elapsed();
        let sketch_compute_start = Instant::now();
        for (offset, row) in sketch_chunk.chunks_exact_mut(sketch_width).enumerate() {
            let point = &chunk[offset * dim..(offset + 1) * dim];
            fill_sketch_row_from_slice(point, &hyperplanes, row)?;
        }
        sketch_compute_wall += sketch_compute_start.elapsed();
        let sketch_extend_start = Instant::now();
        sketch_data.extend_from_slice(&sketch_chunk);
        sketch_extend_wall += sketch_extend_start.elapsed();
        row_start += count;
    }

    codes_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush dense-i8 RabitQ+ sidecar codes '{}': {err}",
            codes_path.display()
        ))
    })?;
    meta_writer.flush().map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to flush dense-i8 RabitQ+ sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    drop(codes_writer);
    drop(meta_writer);

    let header = RabitqPlusSidecarHeader {
        rows,
        dim,
        padded_dim,
        bits,
        rotator,
        seed,
        k_align: RABITQ_DEFAULT_K_ALIGN,
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
                "Failed to reopen dense-i8 RabitQ+ sidecar meta '{}': {err}",
                meta_path.display()
            ))
        })?;
    meta.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek dense-i8 RabitQ+ sidecar meta '{}': {err}",
            meta_path.display()
        ))
    })?;
    meta.write_all(&encode_rabitq_plus_sidecar_header(&header)?)
        .map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write dense-i8 RabitQ+ sidecar header '{}': {err}",
                meta_path.display()
            ))
        })?;
    header_write_wall += header_write_start.elapsed();

    let disk_sketches =
        DiskSketchStore::from_resident_row_major(sketch_path, rows, sketch_width, sketch_data)?;
    let total_wall = total_start.elapsed();
    Ok((
        RabitqPlusSidecarSketchBuildReport {
            sidecar: RabitqPlusSidecarCreateReport {
                header,
                total_wall,
                chunk_alloc_wall,
                raw_read_wall,
                source_hash_wall,
                quantize_wall,
                code_output_wall,
                norm_wall,
                meta_output_wall,
                header_write_wall,
                write_wall: sidecar_write_wall,
            },
            total_wall,
            raw_read_wall,
            sketch_alloc_wall,
            sketch_compute_wall,
            sketch_extend_wall,
            sketch_write_wall,
        },
        disk_sketches,
    ))
}

pub(crate) fn create_resident_rabitq_plus_sidecar_and_sketch_from_point_store(
    dataset: &dyn PointStore,
    prefix: &Path,
    sketch_path: &Path,
    bits: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    seed: u64,
    source_fbin_bytes: u64,
    params: &PiPNNParams,
) -> AnnResult<(
    RabitqPlusSidecarSketchBuildReport,
    RabitqPlusSidecar,
    DiskSketchStore,
)> {
    validate_rabitq_plus_bits(bits)?;
    let total_start = Instant::now();
    let rows = dataset.len();
    let dim = dataset.dim();
    if rows > u32::MAX as usize {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ sidecar currently supports at most {} rows, got {rows}",
            u32::MAX
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, RABITQ_DEFAULT_K_ALIGN)?;
    let codes_bytes = rows.checked_mul(padded_dim).ok_or_else(|| {
        AnnError::log_index_error(
            "dense-i8 RabitQ+ resident sidecar codes byte count overflows".to_string(),
        )
    })?;
    let meta_rows_bytes = rows
        .checked_mul(RABITQ_PLUS_SIDECAR_META_ROW_BYTES)
        .ok_or_else(|| {
            AnnError::log_index_error(
                "dense-i8 RabitQ+ resident sidecar metadata byte count overflows".to_string(),
            )
        })?;
    let sketch_width = params.m_hash_bits;
    let sketch_values = rows.checked_mul(sketch_width).ok_or_else(|| {
        AnnError::log_index_error("Resident sketch value count overflows".to_string())
    })?;
    let hyperplanes = generate_hyperplanes(dim, sketch_width, params.random_seed);

    let mut codes = Vec::with_capacity(codes_bytes);
    let mut norms = Vec::with_capacity(rows);
    let mut deltas = Vec::with_capacity(rows);
    let mut code_sums = Vec::with_capacity(rows);
    let mut sketch_data = Vec::with_capacity(sketch_values);

    let mut hash = fnv1a_update_u64(FNV_OFFSET_BASIS, rows as u64);
    hash = fnv1a_update_u64(hash, dim as u64);
    let mut chunk_alloc_wall = Duration::ZERO;
    let mut raw_read_wall = Duration::ZERO;
    let mut source_hash_wall = Duration::ZERO;
    let mut quantize_wall = Duration::ZERO;
    let mut code_output_wall = Duration::ZERO;
    let mut norm_wall = Duration::ZERO;
    let mut sketch_alloc_wall = Duration::ZERO;
    let mut sketch_compute_wall = Duration::ZERO;
    let mut sketch_extend_wall = Duration::ZERO;
    let row_start = Instant::now();
    let mut row_offset = 0usize;
    while row_offset < rows {
        let count = (rows - row_offset).min(RABITQ_PLUS_SIDECAR_CREATE_CHUNK_ROWS);
        let chunk_alloc_start = Instant::now();
        let mut chunk = vec![0.0_f32; count.saturating_mul(dim)];
        chunk_alloc_wall += chunk_alloc_start.elapsed();
        let raw_read_start = Instant::now();
        dataset.read_range_into(row_offset as u32, count, &mut chunk)?;
        raw_read_wall += raw_read_start.elapsed();
        let source_hash_start = Instant::now();
        hash = fnv1a_update_f32_slice(hash, &chunk);
        source_hash_wall += source_hash_start.elapsed();

        let quantize_start = Instant::now();
        let matrix = quantize_rabitq_plus_matrix(
            &chunk,
            count,
            dim,
            bits,
            RABITQ_DEFAULT_K_ALIGN,
            rotator,
            seed,
        )?;
        quantize_wall += quantize_start.elapsed();
        let code_output_start = Instant::now();
        codes.extend_from_slice(&matrix.codes);
        deltas.extend_from_slice(&matrix.deltas);
        code_sums.extend_from_slice(&matrix.code_sums);
        code_output_wall += code_output_start.elapsed();

        let norm_start = Instant::now();
        for row in 0..count {
            let source = &chunk[row * dim..(row + 1) * dim];
            norms.push(source.iter().map(|&value| value * value).sum::<f32>());
        }
        norm_wall += norm_start.elapsed();

        let sketch_alloc_start = Instant::now();
        let mut sketch_chunk = vec![0.0_f32; count.saturating_mul(sketch_width)];
        sketch_alloc_wall += sketch_alloc_start.elapsed();
        let sketch_compute_start = Instant::now();
        for (offset, row) in sketch_chunk.chunks_exact_mut(sketch_width).enumerate() {
            let point = &chunk[offset * dim..(offset + 1) * dim];
            fill_sketch_row_from_slice(point, &hyperplanes, row)?;
        }
        sketch_compute_wall += sketch_compute_start.elapsed();
        let sketch_extend_start = Instant::now();
        sketch_data.extend_from_slice(&sketch_chunk);
        sketch_extend_wall += sketch_extend_start.elapsed();
        row_offset += count;
    }
    let resident_build_wall = row_start.elapsed();

    let header = RabitqPlusSidecarHeader {
        rows,
        dim,
        padded_dim,
        bits,
        rotator,
        seed,
        k_align: RABITQ_DEFAULT_K_ALIGN,
        source_fbin_bytes,
        source_rows: rows,
        source_dim: dim,
        source_fingerprint: hash,
        codes_bytes,
        meta_rows_bytes,
    };
    let sidecar = RabitqPlusSidecar::from_resident_parts(
        prefix.to_path_buf(),
        header.clone(),
        codes,
        norms,
        deltas,
        code_sums,
        resident_build_wall,
    )?;
    let sketches =
        DiskSketchStore::from_resident_row_major(sketch_path, rows, sketch_width, sketch_data)?;
    let total_wall = total_start.elapsed();
    Ok((
        RabitqPlusSidecarSketchBuildReport {
            sidecar: RabitqPlusSidecarCreateReport {
                header,
                total_wall,
                chunk_alloc_wall,
                raw_read_wall,
                source_hash_wall,
                quantize_wall,
                code_output_wall,
                norm_wall,
                meta_output_wall: Duration::ZERO,
                header_write_wall: Duration::ZERO,
                write_wall: Duration::ZERO,
            },
            total_wall,
            raw_read_wall,
            sketch_alloc_wall,
            sketch_compute_wall,
            sketch_extend_wall,
            sketch_write_wall: Duration::ZERO,
        },
        sidecar,
        sketches,
    ))
}

fn encode_rabitq_plus_sidecar_header(
    header: &RabitqPlusSidecarHeader,
) -> AnnResult<[u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES]> {
    let mut out = [0_u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES];
    let mut pos = 0usize;
    write_header_bytes(&mut out, &mut pos, RABITQ_PLUS_SIDECAR_MAGIC)?;
    write_header_u32(&mut out, &mut pos, RABITQ_PLUS_SIDECAR_VERSION)?;
    write_header_u32(&mut out, &mut pos, RABITQ_PLUS_SIDECAR_HEADER_BYTES as u32)?;
    write_header_u64(&mut out, &mut pos, header.rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.padded_dim as u64)?;
    write_header_u32(&mut out, &mut pos, header.bits as u32)?;
    write_header_u32(&mut out, &mut pos, rabitq_plus_rotator_code(header.rotator))?;
    write_header_u64(&mut out, &mut pos, header.seed)?;
    write_header_u64(&mut out, &mut pos, header.k_align as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fbin_bytes)?;
    write_header_u64(&mut out, &mut pos, header.source_rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fingerprint)?;
    write_header_u64(&mut out, &mut pos, header.codes_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.meta_rows_bytes as u64)?;
    Ok(out)
}

fn read_rabitq_plus_sidecar_header(file: &mut File) -> AnnResult<RabitqPlusSidecarHeader> {
    let mut buf = [0_u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES];
    file.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek dense-i8 RabitQ+ sidecar header: {err}"
        ))
    })?;
    file.read_exact(&mut buf).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to read dense-i8 RabitQ+ sidecar header: {err}"
        ))
    })?;
    let mut pos = 0usize;
    let magic = read_header_bytes(&buf, &mut pos, RABITQ_PLUS_SIDECAR_MAGIC.len())?;
    if magic != RABITQ_PLUS_SIDECAR_MAGIC {
        return Err(AnnError::log_index_error(
            "dense-i8 RabitQ+ sidecar magic mismatch".to_string(),
        ));
    }
    let version = read_header_u32(&buf, &mut pos)?;
    if version != RABITQ_PLUS_SIDECAR_VERSION {
        return Err(AnnError::log_index_error(format!(
            "Unsupported dense-i8 RabitQ+ sidecar version {version}"
        )));
    }
    let header_bytes = read_header_u32(&buf, &mut pos)? as usize;
    if header_bytes != RABITQ_PLUS_SIDECAR_HEADER_BYTES {
        return Err(AnnError::log_index_error(format!(
            "Unsupported dense-i8 RabitQ+ sidecar header size {header_bytes}"
        )));
    }
    Ok(RabitqPlusSidecarHeader {
        rows: read_header_u64(&buf, &mut pos)? as usize,
        dim: read_header_u64(&buf, &mut pos)? as usize,
        padded_dim: read_header_u64(&buf, &mut pos)? as usize,
        bits: read_header_u32(&buf, &mut pos)? as usize,
        rotator: rabitq_plus_rotator_from_code(read_header_u32(&buf, &mut pos)?)?,
        seed: read_header_u64(&buf, &mut pos)?,
        k_align: read_header_u64(&buf, &mut pos)? as usize,
        source_fbin_bytes: read_header_u64(&buf, &mut pos)?,
        source_rows: read_header_u64(&buf, &mut pos)? as usize,
        source_dim: read_header_u64(&buf, &mut pos)? as usize,
        source_fingerprint: read_header_u64(&buf, &mut pos)?,
        codes_bytes: read_header_u64(&buf, &mut pos)? as usize,
        meta_rows_bytes: read_header_u64(&buf, &mut pos)? as usize,
    })
}

fn encode_rabitq_split_sidecar_header(
    header: &RabitqSplitSidecarHeader,
) -> AnnResult<[u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES]> {
    let mut out = [0_u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES];
    let mut pos = 0usize;
    write_header_bytes(&mut out, &mut pos, RABITQ_SPLIT_SIDECAR_MAGIC)?;
    write_header_u32(&mut out, &mut pos, RABITQ_PLUS_SIDECAR_VERSION)?;
    write_header_u32(&mut out, &mut pos, RABITQ_PLUS_SIDECAR_HEADER_BYTES as u32)?;
    write_header_u64(&mut out, &mut pos, header.rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.padded_dim as u64)?;
    write_header_u32(&mut out, &mut pos, header.bits as u32)?;
    write_header_u32(&mut out, &mut pos, header.ex_bits as u32)?;
    write_header_u32(&mut out, &mut pos, rabitq_plus_rotator_code(header.rotator))?;
    write_header_u64(&mut out, &mut pos, header.seed)?;
    write_header_u64(&mut out, &mut pos, header.k_align as u64)?;
    write_header_u64(&mut out, &mut pos, header.batch_rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.batch_count as u64)?;
    write_header_u64(&mut out, &mut pos, header.batch_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.ex_row_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fbin_bytes)?;
    write_header_u64(&mut out, &mut pos, header.source_rows as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_dim as u64)?;
    write_header_u64(&mut out, &mut pos, header.source_fingerprint)?;
    write_header_u64(&mut out, &mut pos, header.batch_total_bytes as u64)?;
    write_header_u64(&mut out, &mut pos, header.ex_total_bytes as u64)?;
    Ok(out)
}

fn read_rabitq_split_sidecar_header(file: &mut File) -> AnnResult<RabitqSplitSidecarHeader> {
    let mut buf = [0_u8; RABITQ_PLUS_SIDECAR_HEADER_BYTES];
    file.seek(SeekFrom::Start(0)).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to seek RabitQ-Library split sidecar header: {err}"
        ))
    })?;
    file.read_exact(&mut buf).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to read RabitQ-Library split sidecar header: {err}"
        ))
    })?;
    let mut pos = 0usize;
    let magic = read_header_bytes(&buf, &mut pos, RABITQ_SPLIT_SIDECAR_MAGIC.len())?;
    if magic != RABITQ_SPLIT_SIDECAR_MAGIC {
        return Err(AnnError::log_index_error(
            "RabitQ-Library split sidecar magic mismatch".to_string(),
        ));
    }
    let version = read_header_u32(&buf, &mut pos)?;
    if version != RABITQ_PLUS_SIDECAR_VERSION {
        return Err(AnnError::log_index_error(format!(
            "Unsupported RabitQ-Library split sidecar version {version}"
        )));
    }
    let header_bytes = read_header_u32(&buf, &mut pos)? as usize;
    if header_bytes != RABITQ_PLUS_SIDECAR_HEADER_BYTES {
        return Err(AnnError::log_index_error(format!(
            "Unsupported RabitQ-Library split sidecar header size {header_bytes}"
        )));
    }
    let rows = read_header_u64(&buf, &mut pos)? as usize;
    let dim = read_header_u64(&buf, &mut pos)? as usize;
    let padded_dim = read_header_u64(&buf, &mut pos)? as usize;
    let bits = read_header_u32(&buf, &mut pos)? as usize;
    let ex_bits = read_header_u32(&buf, &mut pos)? as usize;
    let rotator = rabitq_plus_rotator_from_code(read_header_u32(&buf, &mut pos)?)?;
    let seed = read_header_u64(&buf, &mut pos)?;
    let k_align = read_header_u64(&buf, &mut pos)? as usize;
    let batch_rows = read_header_u64(&buf, &mut pos)? as usize;
    let batch_count = read_header_u64(&buf, &mut pos)? as usize;
    let batch_bytes = read_header_u64(&buf, &mut pos)? as usize;
    let ex_row_bytes = read_header_u64(&buf, &mut pos)? as usize;
    let source_fbin_bytes = read_header_u64(&buf, &mut pos)?;
    let source_rows = read_header_u64(&buf, &mut pos)? as usize;
    let source_dim = read_header_u64(&buf, &mut pos)? as usize;
    let source_fingerprint = read_header_u64(&buf, &mut pos)?;
    let batch_total_bytes = read_header_u64(&buf, &mut pos)? as usize;
    let ex_total_bytes = read_header_u64(&buf, &mut pos)? as usize;
    Ok(RabitqSplitSidecarHeader {
        rows,
        dim,
        padded_dim,
        bits,
        ex_bits,
        rotator,
        seed,
        k_align,
        batch_rows,
        batch_count,
        batch_bytes,
        ex_row_bytes,
        source_fbin_bytes,
        source_rows,
        source_dim,
        source_fingerprint,
        batch_total_bytes,
        ex_total_bytes,
    })
}

fn write_header_bytes(out: &mut [u8], pos: &mut usize, bytes: &[u8]) -> AnnResult<()> {
    let end = pos.checked_add(bytes.len()).ok_or_else(|| {
        AnnError::log_index_error("dense-i8 RabitQ+ sidecar header offset overflows".to_string())
    })?;
    if end > out.len() {
        return Err(AnnError::log_index_error(
            "dense-i8 RabitQ+ sidecar header buffer too small".to_string(),
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
        AnnError::log_index_error("dense-i8 RabitQ+ sidecar header offset overflows".to_string())
    })?;
    if end > buf.len() {
        return Err(AnnError::log_index_error(
            "dense-i8 RabitQ+ sidecar header is truncated".to_string(),
        ));
    }
    let out = &buf[*pos..end];
    *pos = end;
    Ok(out)
}

fn read_header_u32(buf: &[u8], pos: &mut usize) -> AnnResult<u32> {
    let bytes = read_header_bytes(buf, pos, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_header_u64(buf: &[u8], pos: &mut usize) -> AnnResult<u64> {
    let bytes = read_header_bytes(buf, pos, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn rabitq_plus_rotator_code(rotator: PiPNNGpuSubtreeRabitqPlusRotator) -> u32 {
    match rotator {
        PiPNNGpuSubtreeRabitqPlusRotator::FhtKac => 1,
        PiPNNGpuSubtreeRabitqPlusRotator::None => 2,
    }
}

fn rabitq_plus_rotator_from_code(code: u32) -> AnnResult<PiPNNGpuSubtreeRabitqPlusRotator> {
    match code {
        1 => Ok(PiPNNGpuSubtreeRabitqPlusRotator::FhtKac),
        2 => Ok(PiPNNGpuSubtreeRabitqPlusRotator::None),
        other => Err(AnnError::log_index_error(format!(
            "Unsupported dense-i8 RabitQ+ sidecar rotator code {other}"
        ))),
    }
}

fn fnv1a_update_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a_update_f32_slice(mut hash: u64, values: &[f32]) -> u64 {
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

pub(crate) fn quantize_rabitq_row(
    input: &[f32],
    output_padded: &mut [i8],
    total_bits: usize,
    t_const: f64,
) -> AnnResult<(f32, f32)> {
    validate_total_bits(total_bits)?;
    if output_padded.len() < input.len() {
        return Err(AnnError::log_index_error(format!(
            "RaBitQ row output too small: input_dim={} output_dim={}",
            input.len(),
            output_padded.len()
        )));
    }
    output_padded.fill(0);

    let mut norm_sqr = 0.0_f32;
    for &value in input {
        if !value.is_finite() {
            return Err(AnnError::log_index_error(format!(
                "RaBitQ cannot quantize non-finite value {value}"
            )));
        }
        norm_sqr += value * value;
    }
    if norm_sqr == 0.0 {
        return Ok((0.0, 0.0));
    }

    let ex_bits = total_bits - 1;
    let ex_mask = ((1_u32 << ex_bits) - 1) as i32;
    let center = (1_i32 << ex_bits) as i32;
    let norm = norm_sqr.sqrt();
    let mut dot_residual_quant = 0.0_f32;
    let mut quant_norm_sqr = 0.0_f32;
    let mut code_sum = 0.0_f32;

    for (dst, &value) in output_padded.iter_mut().zip(input) {
        let abs_normed = (value.abs() / norm) as f64;
        let mut ex_code = if ex_bits == 0 {
            0_i32
        } else {
            ((t_const * abs_normed) + RABITQ_QUANT_EPS).floor() as i32
        };
        if ex_code > ex_mask {
            ex_code = ex_mask;
        }
        if value < 0.0 {
            ex_code = ex_mask - ex_code;
        }
        let sign_bit = i32::from(value > 0.0);
        let total_code = ex_code + (sign_bit << ex_bits);
        let signed_code = total_code - center;
        let shifted_code = signed_code as f32 + 0.5;
        *dst = signed_code as i8;
        dot_residual_quant += value * shifted_code;
        quant_norm_sqr += shifted_code * shifted_code;
        code_sum += signed_code as f32;
    }

    let delta = if quant_norm_sqr > 0.0 {
        dot_residual_quant / quant_norm_sqr
    } else {
        0.0
    };
    Ok((delta, code_sum))
}

pub(crate) fn validate_rabitq_plus_bits(total_bits: usize) -> AnnResult<()> {
    if !matches!(total_bits, 4 | 5 | 7 | 8) {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ total_bits must be one of 4, 5, 7, or 8, got {total_bits}"
        )));
    }
    validate_total_bits(total_bits)
}

pub(crate) fn rotate_rabitq_plus_fht_kac_row(
    input: &[f32],
    output: &mut [f32],
    seed: u64,
) -> AnnResult<()> {
    if input.len() != output.len() {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ rotation shape mismatch: input={} output={}",
            input.len(),
            output.len()
        )));
    }
    if !input.len().is_power_of_two() {
        return Err(AnnError::log_index_error(format!(
            "dense-i8 RabitQ+ fht-kac rotator requires power-of-two dim, got {}",
            input.len()
        )));
    }
    for (&src, dst) in input.iter().zip(output.iter_mut()) {
        if !src.is_finite() {
            return Err(AnnError::log_index_error(format!(
                "dense-i8 RabitQ+ cannot rotate non-finite value {src}"
            )));
        }
        *dst = src;
    }
    let norm_scale = 1.0_f32 / (input.len() as f32).sqrt();
    for round in 0..4_u32 {
        for (axis, value) in output.iter_mut().enumerate() {
            if !rabitq_plus_sign(seed, round, axis) {
                *value = -*value;
            }
        }
        fwht_in_place(output);
        for value in output.iter_mut() {
            *value *= norm_scale;
        }
    }
    Ok(())
}

fn fwht_in_place(values: &mut [f32]) {
    let mut width = 1;
    while width < values.len() {
        let step = width * 2;
        for base in (0..values.len()).step_by(step) {
            for offset in 0..width {
                let left = values[base + offset];
                let right = values[base + width + offset];
                values[base + offset] = left + right;
                values[base + width + offset] = left - right;
            }
        }
        width = step;
    }
}

fn rabitq_plus_sign(seed: u64, round: u32, axis: usize) -> bool {
    let mut value = seed
        ^ 0x52_61_42_69_74_51_2b_u64
        ^ ((round as u64).wrapping_mul(0x9e3779b97f4a7c15))
        ^ ((axis as u64).wrapping_mul(0xbf58476d1ce4e5b9));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) & 1) == 0
}

pub(crate) fn rabitq_dot_from_i8(
    left_code: &[i8],
    left_code_sum: f32,
    right_code: &[i8],
    right_code_sum: f32,
    dim: usize,
) -> AnnResult<f32> {
    if left_code.len() != right_code.len() {
        return Err(AnnError::log_index_error(format!(
            "RaBitQ dot shape mismatch: left={} right={}",
            left_code.len(),
            right_code.len()
        )));
    }
    if dim > left_code.len() {
        return Err(AnnError::log_index_error(format!(
            "RaBitQ dot dim exceeds code width: dim={dim} code_width={}",
            left_code.len()
        )));
    }
    let dot = left_code
        .iter()
        .zip(right_code)
        .take(dim)
        .map(|(&lhs, &rhs)| i32::from(lhs) * i32::from(rhs))
        .sum::<i32>() as f32;
    Ok(dot + 0.5 * left_code_sum + 0.5 * right_code_sum + 0.25 * dim as f32)
}

pub(crate) fn rabitq_split_short_words_per_row(dim: usize) -> usize {
    dim.div_ceil(32)
}

pub(crate) fn rabitq_split_ex_row_bytes(dim: usize, ex_bits: usize) -> AnnResult<usize> {
    if !(1..=7).contains(&ex_bits) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split ex_bits must be in 1..=7, got {ex_bits}"
        )));
    }
    dim.checked_mul(ex_bits)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Residual RaBitQ split ex row bytes overflow: dim={dim} ex_bits={ex_bits}"
            ))
        })
}

#[inline]
pub(crate) fn rabitq_split_get_short_bit(short_words: &[u32], axis: usize) -> AnnResult<u32> {
    let word = axis / 32;
    let bit = axis % 32;
    let value = short_words.get(word).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split short bit axis {axis} exceeds words {}",
            short_words.len()
        ))
    })?;
    Ok((value >> bit) & 1)
}

#[inline]
pub(crate) fn rabitq_split_get_ex_code(
    ex_bytes: &[u8],
    axis: usize,
    ex_bits: usize,
) -> AnnResult<u32> {
    if !(1..=7).contains(&ex_bits) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split ex_bits must be in 1..=7, got {ex_bits}"
        )));
    }
    let bit_start = axis.checked_mul(ex_bits).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split ex bit offset overflow: axis={axis} ex_bits={ex_bits}"
        ))
    })?;
    let byte_start = bit_start / 8;
    let bit_offset = bit_start % 8;
    if byte_start >= ex_bytes.len() {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split ex byte {byte_start} out of range {}",
            ex_bytes.len()
        )));
    }
    let mut word = 0_u32;
    for byte_delta in 0..2 {
        if let Some(&byte) = ex_bytes.get(byte_start + byte_delta) {
            word |= u32::from(byte) << (8 * byte_delta);
        }
    }
    Ok((word >> bit_offset) & ((1_u32 << ex_bits) - 1))
}

#[inline]
fn rabitq_split_set_ex_code(
    ex_bytes: &mut [u8],
    axis: usize,
    ex_bits: usize,
    code: u32,
) -> AnnResult<()> {
    let bit_start = axis.checked_mul(ex_bits).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split ex bit offset overflow: axis={axis} ex_bits={ex_bits}"
        ))
    })?;
    let mask = (1_u32 << ex_bits) - 1;
    if code > mask {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split ex code {code} exceeds mask {mask}"
        )));
    }
    for bit in 0..ex_bits {
        if ((code >> bit) & 1) != 0 {
            let dst_bit = bit_start + bit;
            let byte = dst_bit / 8;
            let offset = dst_bit % 8;
            let ex_len = ex_bytes.len();
            let slot = ex_bytes.get_mut(byte).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Residual RaBitQ split ex byte {byte} out of range {ex_len}"
                ))
            })?;
            *slot |= 1_u8 << offset;
        }
    }
    Ok(())
}

pub(crate) fn pack_rabitq_dense_i8_row_to_split(
    dense: &[i8],
    dim: usize,
    ex_bits: usize,
    short_words: &mut [u32],
    ex_bytes: &mut [u8],
) -> AnnResult<()> {
    if dense.len() < dim {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split pack dense row too short: dense={} dim={dim}",
            dense.len()
        )));
    }
    if short_words.len() != rabitq_split_short_words_per_row(dim) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split pack short word mismatch: got={} expected={}",
            short_words.len(),
            rabitq_split_short_words_per_row(dim)
        )));
    }
    let expected_ex = rabitq_split_ex_row_bytes(dim, ex_bits)?;
    if ex_bytes.len() != expected_ex {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split pack ex byte mismatch: got={} expected={expected_ex}",
            ex_bytes.len()
        )));
    }
    if ex_bits == 7 {
        pack_rabitq_dense_i8_row_to_split_ex7(dense, dim, short_words, ex_bytes);
        return Ok(());
    }
    short_words.fill(0);
    ex_bytes.fill(0);
    let center = 1_i32 << ex_bits;
    let ex_mask = (1_u32 << ex_bits) - 1;
    for axis in 0..dim {
        let total_code = i32::from(dense[axis]) + center;
        if !(0..=(2 * center - 1)).contains(&total_code) {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ split pack dense code out of range: axis={axis} code={} ex_bits={ex_bits}",
                dense[axis]
            )));
        }
        let unsigned = total_code as u32;
        if ((unsigned >> ex_bits) & 1) != 0 {
            short_words[axis / 32] |= 1_u32 << (axis % 32);
        }
        rabitq_split_set_ex_code(ex_bytes, axis, ex_bits, unsigned & ex_mask)?;
    }
    Ok(())
}

fn pack_rabitq_dense_i8_row_to_split_ex7(
    dense: &[i8],
    dim: usize,
    short_words: &mut [u32],
    ex_bytes: &mut [u8],
) {
    short_words.fill(0);
    for (word_index, short_word) in short_words.iter_mut().enumerate() {
        let axis_start = word_index * 32;
        let axis_end = (axis_start + 32).min(dim);
        let mut word = 0_u32;
        for axis in axis_start..axis_end {
            let unsigned = (i32::from(dense[axis]) + 128) as u32;
            word |= ((unsigned >> 7) & 1) << (axis - axis_start);
        }
        *short_word = word;
    }

    for group in 0..(dim / 8) {
        let axis_start = group * 8;
        let mut packed = 0_u64;
        for lane in 0..8 {
            let unsigned = (i32::from(dense[axis_start + lane]) + 128) as u64;
            packed |= (unsigned & 0x7f) << (lane * 7);
        }
        let bytes = packed.to_le_bytes();
        let dst = group * 7;
        ex_bytes[dst..dst + 7].copy_from_slice(&bytes[..7]);
    }

    let tail = dim % 8;
    if tail != 0 {
        let axis_start = dim - tail;
        let mut packed = 0_u64;
        for lane in 0..tail {
            let unsigned = (i32::from(dense[axis_start + lane]) + 128) as u64;
            packed |= (unsigned & 0x7f) << (lane * 7);
        }
        let bytes = packed.to_le_bytes();
        let dst = (dim / 8) * 7;
        let tail_bytes = (tail * 7).div_ceil(8);
        ex_bytes[dst..dst + tail_bytes].copy_from_slice(&bytes[..tail_bytes]);
    }
}

pub(crate) fn unpack_rabitq_split_row_to_dense_i8(
    short_words: &[u32],
    ex_bytes: &[u8],
    dim: usize,
    ex_bits: usize,
    dense: &mut [i8],
) -> AnnResult<()> {
    if dense.len() < dim {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split unpack dense row too short: dense={} dim={dim}",
            dense.len()
        )));
    }
    if short_words.len() != rabitq_split_short_words_per_row(dim) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split unpack short word mismatch: got={} expected={}",
            short_words.len(),
            rabitq_split_short_words_per_row(dim)
        )));
    }
    let expected_ex = rabitq_split_ex_row_bytes(dim, ex_bits)?;
    if ex_bytes.len() != expected_ex {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split unpack ex byte mismatch: got={} expected={expected_ex}",
            ex_bytes.len()
        )));
    }
    let center = 1_i32 << ex_bits;
    for axis in 0..dim {
        let short = rabitq_split_get_short_bit(short_words, axis)?;
        let ex = rabitq_split_get_ex_code(ex_bytes, axis, ex_bits)?;
        let total = ((short << ex_bits) | ex) as i32;
        dense[axis] = (total - center) as i8;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rabitq_dot_from_split(
    left_short: &[u32],
    left_ex: &[u8],
    left_code_sum: f32,
    right_short: &[u32],
    right_ex: &[u8],
    right_code_sum: f32,
    dim: usize,
    ex_bits: usize,
) -> AnnResult<f32> {
    let center = 1_i32 << ex_bits;
    let mut code_dot = 0_i32;
    for axis in 0..dim {
        let left_u = ((rabitq_split_get_short_bit(left_short, axis)? << ex_bits)
            | rabitq_split_get_ex_code(left_ex, axis, ex_bits)?) as i32;
        let right_u = ((rabitq_split_get_short_bit(right_short, axis)? << ex_bits)
            | rabitq_split_get_ex_code(right_ex, axis, ex_bits)?) as i32;
        code_dot += (left_u - center) * (right_u - center);
    }
    Ok(code_dot as f32 + 0.5 * left_code_sum + 0.5 * right_code_sum + 0.25 * dim as f32)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rabitq_approx_l2(
    left_code: &[i8],
    left_delta: f32,
    left_code_sum: f32,
    left_norm: f32,
    right_code: &[i8],
    right_delta: f32,
    right_code_sum: f32,
    right_norm: f32,
    dim: usize,
) -> f32 {
    let dot = rabitq_dot_from_i8(left_code, left_code_sum, right_code, right_code_sum, dim)
        .unwrap_or(0.0);
    let approx_dot = left_delta * right_delta * dot;
    let dist = left_norm + right_norm - 2.0 * approx_dot;
    if dist <= 0.0 { 0.0 } else { dist }
}

fn validate_total_bits(total_bits: usize) -> AnnResult<()> {
    if !(1..=8).contains(&total_bits) {
        return Err(AnnError::log_index_error(format!(
            "RaBitQ total_bits must be in 1..=8, got {total_bits}"
        )));
    }
    Ok(())
}

pub(crate) fn rabitq_const_scaling_factor(dim: usize, ex_bits: usize) -> AnnResult<f64> {
    if ex_bits == 0 {
        return Ok(-1.0);
    }
    if ex_bits >= RABITQ_TIGHT_START.len() {
        return Err(AnnError::log_index_error(format!(
            "RaBitQ ex_bits must be <= 8, got {ex_bits}"
        )));
    }
    let cache = RABITQ_T_CONST_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(value) = cache
        .lock()
        .map_err(|_| AnnError::log_index_error("RaBitQ t_const cache poisoned".to_string()))?
        .get(&(dim, ex_bits))
        .copied()
    {
        return Ok(value);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let value = unsafe {
            crate::model::rabitq::ffi::rabitq_plus_fast_t_const(dim, (ex_bits + 1) as libc::c_int)
        };
        if value.is_finite() && value > 0.0 {
            cache
                .lock()
                .map_err(|_| {
                    AnnError::log_index_error("RaBitQ t_const cache poisoned".to_string())
                })?
                .insert((dim, ex_bits), value);
            return Ok(value);
        }
    }

    let mut rng = SplitMix64::new(0x52_61_42_69_74_51_u64 ^ ((dim as u64) << 17) ^ ex_bits as u64);
    let sample_count = 100;
    let mut sum = 0.0_f64;
    for _ in 0..sample_count {
        let mut values = vec![0.0_f64; dim];
        let mut norm_sqr = 0.0_f64;
        let mut axis = 0;
        while axis < dim {
            let (z0, z1) = rng.normal_pair();
            values[axis] = z0.abs();
            norm_sqr += z0 * z0;
            axis += 1;
            if axis < dim {
                values[axis] = z1.abs();
                norm_sqr += z1 * z1;
                axis += 1;
            }
        }
        let norm = norm_sqr.sqrt();
        if norm > 0.0 {
            for value in &mut values {
                *value /= norm;
            }
        }
        sum += best_rescale_factor(&values, ex_bits);
    }
    let value = sum / sample_count as f64;
    cache
        .lock()
        .map_err(|_| AnnError::log_index_error("RaBitQ t_const cache poisoned".to_string()))?
        .insert((dim, ex_bits), value);
    Ok(value)
}

fn best_rescale_factor(o_abs: &[f64], ex_bits: usize) -> f64 {
    let dim = o_abs.len();
    let max_o = o_abs.iter().copied().fold(0.0_f64, f64::max);
    if dim == 0 || max_o <= 0.0 {
        return 1.0;
    }

    let ne_num = 10.0_f64;
    let code_max = ((1_usize << ex_bits) - 1) as f64;
    let t_end = (code_max + ne_num) / max_o;
    let t_start = t_end * RABITQ_TIGHT_START[ex_bits];
    let mut cur_o_bar = vec![0_i32; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0_f64;
    let mut next_t = BinaryHeap::new();

    for (idx, &value) in o_abs.iter().enumerate() {
        let cur = ((t_start * value) + RABITQ_QUANT_EPS).floor() as i32;
        cur_o_bar[idx] = cur;
        sqr_denominator += (cur * cur + cur) as f64;
        numerator += (cur as f64 + 0.5) * value;
        next_t.push(RescaleThreshold {
            value: (cur as f64 + 1.0) / value,
            idx,
        });
    }

    let mut max_ip = 0.0_f64;
    let mut best_t = 0.0_f64;
    while let Some(entry) = next_t.pop() {
        let update_idx = entry.idx;
        cur_o_bar[update_idx] += 1;
        let update_o_bar = cur_o_bar[update_idx];
        sqr_denominator += (2 * update_o_bar) as f64;
        numerator += o_abs[update_idx];

        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = entry.value;
        }

        if update_o_bar < code_max as i32 {
            let t_next = (update_o_bar as f64 + 1.0) / o_abs[update_idx];
            if t_next < t_end {
                next_t.push(RescaleThreshold {
                    value: t_next,
                    idx: update_idx,
                });
            }
        }
    }
    best_t
}

#[derive(Clone, Copy, Debug)]
struct RescaleThreshold {
    value: f64,
    idx: usize,
}

impl PartialEq for RescaleThreshold {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.idx == other.idx
    }
}

impl Eq for RescaleThreshold {}

impl PartialOrd for RescaleThreshold {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RescaleThreshold {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .total_cmp(&self.value)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn uniform01_open(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.uniform01_open().max(f64::MIN_POSITIVE);
        let u2 = self.uniform01_open();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::pipnn::direct_io::DirectIoConfig;
    use crate::algorithm::pipnn::external_store::DiskSketchStore;
    use crate::algorithm::pipnn::hash_prune::SketchAccessor;
    use crate::algorithm::pipnn::params::PiPNNParams;
    use crate::algorithm::pipnn::point_store::InmemDatasetPointStore;
    use crate::model::InmemDataset;

    fn norm(row: &[f32]) -> f32 {
        row.iter().map(|value| value * value).sum()
    }

    fn build_test_dataset(rows: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for row in 0..rows {
            for axis in 0..dim {
                dataset.data[row * dim + axis] = row as f32 * 0.25 + axis as f32 * 0.5 - 0.75;
            }
        }
        dataset
    }

    #[test]
    fn rabitq_quantized_dot_matches_reconstructed_dot() {
        let points = [
            0.2_f32, -1.1, 0.7, 2.3, -0.4, //
            -0.9, 0.5, 1.4, -2.0, 0.1,
        ];
        let matrix = quantize_rabitq_matrix(
            &points,
            2,
            5,
            RABITQ_DEFAULT_TOTAL_BITS,
            RABITQ_DEFAULT_K_ALIGN,
        )
        .unwrap();
        let dot = rabitq_dot_from_i8(
            matrix.code_row(0),
            matrix.code_sums[0],
            matrix.code_row(1),
            matrix.code_sums[1],
            matrix.dim,
        )
        .unwrap();
        let reconstructed_dot = (0..matrix.dim)
            .map(|axis| {
                let lhs = matrix.deltas[0] * (matrix.code_row(0)[axis] as f32 + 0.5);
                let rhs = matrix.deltas[1] * (matrix.code_row(1)[axis] as f32 + 0.5);
                lhs * rhs
            })
            .sum::<f32>();

        let approx_dot = matrix.deltas[0] * matrix.deltas[1] * dot;
        assert!((approx_dot - reconstructed_dot).abs() <= 1.0e-4);
    }

    #[test]
    fn rabitq_split_round_trip_matches_dense_i8_for_8bit() {
        let dim = 17;
        let ex_bits = 7;
        let dense = vec![
            -128, -127, -65, -1, 0, 1, 2, 63, 64, 65, 100, 126, 127, -32, 31, -7, 7,
        ];
        let mut short = vec![0_u32; rabitq_split_short_words_per_row(dim)];
        let mut ex = vec![0_u8; rabitq_split_ex_row_bytes(dim, ex_bits).unwrap()];
        pack_rabitq_dense_i8_row_to_split(&dense, dim, ex_bits, &mut short, &mut ex).unwrap();

        let mut decoded = vec![0_i8; dim];
        unpack_rabitq_split_row_to_dense_i8(&short, &ex, dim, ex_bits, &mut decoded).unwrap();

        assert_eq!(decoded, dense);
    }

    #[test]
    fn rabitq_split_dot_matches_dense_i8_dot() {
        let dim = 9;
        let ex_bits = 7;
        let left = vec![-128, -3, 0, 1, 7, 64, 127, -65, 42];
        let right = vec![127, 64, -1, 3, -7, -63, -128, 65, 5];
        let left_sum = left.iter().map(|&value| value as f32).sum::<f32>();
        let right_sum = right.iter().map(|&value| value as f32).sum::<f32>();

        let mut left_short = vec![0_u32; rabitq_split_short_words_per_row(dim)];
        let mut right_short = vec![0_u32; rabitq_split_short_words_per_row(dim)];
        let mut left_ex = vec![0_u8; rabitq_split_ex_row_bytes(dim, ex_bits).unwrap()];
        let mut right_ex = vec![0_u8; rabitq_split_ex_row_bytes(dim, ex_bits).unwrap()];
        pack_rabitq_dense_i8_row_to_split(&left, dim, ex_bits, &mut left_short, &mut left_ex)
            .unwrap();
        pack_rabitq_dense_i8_row_to_split(&right, dim, ex_bits, &mut right_short, &mut right_ex)
            .unwrap();

        let split_dot = rabitq_dot_from_split(
            &left_short,
            &left_ex,
            left_sum,
            &right_short,
            &right_ex,
            right_sum,
            dim,
            ex_bits,
        )
        .unwrap();
        let dense_dot = rabitq_dot_from_i8(&left, left_sum, &right, right_sum, dim).unwrap();

        assert!((split_dot - dense_dot).abs() <= 1.0e-4);
    }

    #[test]
    fn rabitq_approx_l2_is_finite_for_zero_rows() {
        let points = [0.0_f32, 0.0, 0.0, 1.0, -2.0, 0.5];
        let matrix = quantize_rabitq_matrix(
            &points,
            2,
            3,
            RABITQ_DEFAULT_TOTAL_BITS,
            RABITQ_DEFAULT_K_ALIGN,
        )
        .unwrap();
        let dist = matrix.approx_l2(0, norm(&points[0..3]), 1, norm(&points[3..6]));
        assert!(dist.is_finite());
        assert!(dist >= 0.0);
    }

    #[test]
    fn rabitq_split_matrix_uses_library_batch_ex_layout_for_gpu_leaf() {
        let rows = 33;
        let dim = 64;
        let dataset = build_test_dataset(rows, dim);
        let matrix = quantize_rabitq_split_matrix(
            &dataset.data[..rows * dim],
            rows,
            dim,
            8,
            RABITQ_DEFAULT_K_ALIGN,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            42,
        )
        .unwrap();
        let expected_batch_bytes =
            unsafe { crate::model::rabitq::ffi::rabitq_split_batch_data_bytes(dim) };
        let expected_ex_row_bytes =
            unsafe { crate::model::rabitq::ffi::rabitq_split_ex_data_bytes(dim, 7) };

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.dim, dim);
        assert_eq!(matrix.padded_dim, dim);
        assert_eq!(matrix.total_bits, 8);
        assert_eq!(matrix.ex_bits, 7);
        assert_eq!(matrix.batch_count, 2);
        assert_eq!(matrix.batch_bytes, expected_batch_bytes);
        assert_eq!(matrix.ex_row_bytes, expected_ex_row_bytes);
        assert_eq!(
            matrix.batch_data.len(),
            matrix.batch_count * matrix.batch_bytes
        );
        assert_eq!(matrix.ex_data.len(), rows * matrix.ex_row_bytes);
        assert_eq!(matrix.rotated.len(), rows * dim);
        assert_eq!(matrix.query_kbxsumq.len(), rows);
        assert!(matrix.batch_data.iter().any(|&value| value != 0));
        assert!(matrix.ex_data.iter().any(|&value| value != 0));

        let cb = -127.5_f32;
        for row in [0_usize, 32] {
            let raw_norm = norm(&dataset.data[row * dim..(row + 1) * dim]);
            let rotated = &matrix.rotated[row * dim..(row + 1) * dim];
            let rotated_norm = norm(rotated);
            assert!(
                (raw_norm - rotated_norm).abs() <= raw_norm.abs().max(1.0) * 1.0e-5,
                "row={row} raw_norm={raw_norm} rotated_norm={rotated_norm}"
            );
            let expected_kbxsumq = rotated.iter().copied().sum::<f32>() * cb;
            assert!((matrix.query_kbxsumq[row] - expected_kbxsumq).abs() <= 1.0e-4);
        }
    }

    #[test]
    fn rabitq_rejects_invalid_total_bits() {
        let mut output = [0_i8; 16];
        let err = quantize_rabitq_row(&[1.0, 2.0], &mut output, 9, 1.0).unwrap_err();
        assert!(format!("{err:?}").contains("total_bits"));
    }

    #[test]
    fn rabitq_plus_quantizer_is_deterministic_for_fht_kac() {
        let points = [
            0.2_f32, -1.1, 0.7, 2.3, -0.4, 0.8, 1.2, -0.3, //
            -0.9, 0.5, 1.4, -2.0, 0.1, -0.6, 0.4, 1.0,
        ];
        let left = quantize_rabitq_plus_matrix(
            &points,
            2,
            8,
            8,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            42,
        )
        .unwrap();
        let right = quantize_rabitq_plus_matrix(
            &points,
            2,
            8,
            8,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            42,
        )
        .unwrap();

        assert_eq!(left, right);
        assert_eq!(left.padded_dim, 8);
    }

    #[test]
    fn rabitq_plus_fht_kac_fast_path_matches_scalar_reference() {
        let rows = 3;
        let dim = 1024;
        let bits = 8;
        let seed = 42;
        let mut rng = SplitMix64::new(0x52_61_42_69_74_51_2b_u64);
        let mut points = Vec::with_capacity(rows * dim);
        while points.len() < rows * dim {
            let (left, right) = rng.normal_pair();
            points.push((left * 0.25) as f32);
            if points.len() < rows * dim {
                points.push((right * 0.25) as f32);
            }
        }
        let t_const = rabitq_const_scaling_factor(dim, bits - 1).unwrap();
        let scalar = quantize_rabitq_plus_matrix_scalar_fht_kac(
            &points, rows, dim, dim, bits, t_const, seed,
        )
        .unwrap();
        let fast = quantize_rabitq_plus_matrix(
            &points,
            rows,
            dim,
            bits,
            RABITQ_DEFAULT_K_ALIGN,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            seed,
        )
        .unwrap();

        assert_eq!(fast.codes, scalar.codes);
        assert_eq!(fast.code_sums, scalar.code_sums);
        for row in 0..rows {
            assert!(
                (fast.deltas[row] - scalar.deltas[row]).abs() <= 1.0e-5,
                "row={row} fast_delta={} scalar_delta={}",
                fast.deltas[row],
                scalar.deltas[row]
            );
        }
    }

    #[test]
    fn rabitq_plus_accepts_planned_bits_and_rejects_others() {
        let points = [1.0_f32, -2.0, 0.5, 0.25];
        for bits in [4, 5, 7, 8] {
            quantize_rabitq_plus_matrix(
                &points,
                1,
                4,
                bits,
                4,
                PiPNNGpuSubtreeRabitqPlusRotator::None,
                7,
            )
            .unwrap();
        }
        let err = quantize_rabitq_plus_matrix(
            &points,
            1,
            4,
            6,
            4,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            7,
        )
        .unwrap_err();
        assert!(format!("{err:?}").contains("4, 5, 7, or 8"));
    }

    #[test]
    fn rabitq_plus_zero_row_is_finite() {
        let points = [0.0_f32; 8];
        let matrix = quantize_rabitq_plus_matrix(
            &points,
            1,
            8,
            8,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            9,
        )
        .unwrap();
        assert_eq!(matrix.deltas[0], 0.0);
        assert_eq!(matrix.code_sums[0], 0.0);
    }

    #[test]
    fn rabitq_plus_sidecar_roundtrip_gathers_in_input_order() {
        let rows = 5;
        let dim = 6;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.rabitq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let report = create_rabitq_plus_sidecar_from_point_store(
            &store,
            &prefix,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
            source_bytes as u64,
        )
        .unwrap();
        assert_eq!(report.header.rows, rows);
        assert_eq!(report.header.dim, dim);
        assert_eq!(report.header.codes_bytes, rows * report.header.padded_dim);
        assert_eq!(
            report.header.meta_rows_bytes,
            rows * RABITQ_PLUS_SIDECAR_META_ROW_BYTES
        );

        let sidecar = RabitqPlusSidecar::open_validated(
            &prefix,
            rows,
            dim,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
        )
        .unwrap();
        let ids = [3_u32, 1, 3, 0];
        let gathered = sidecar.gather(&ids).unwrap();
        assert!(!gathered.resident_source);
        assert_eq!(gathered.global_ids, ids);
        assert_eq!(gathered.codes.len(), ids.len() * report.header.padded_dim);
        assert_eq!(gathered.norms.len(), ids.len());

        let resident_sidecar = RabitqPlusSidecar::preload_validated(
            &prefix,
            rows,
            dim,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
        )
        .unwrap();
        assert!(resident_sidecar.is_resident());
        assert_eq!(
            resident_sidecar.resident_bytes(),
            report.header.codes_bytes + report.header.meta_rows_bytes
        );
        let resident_gathered = resident_sidecar.gather(&ids).unwrap();
        assert!(resident_gathered.resident_source);
        assert_eq!(resident_gathered.global_ids, gathered.global_ids);
        assert_eq!(resident_gathered.codes, gathered.codes);
        assert_eq!(resident_gathered.deltas, gathered.deltas);
        assert_eq!(resident_gathered.code_sums, gathered.code_sums);
        assert_eq!(resident_gathered.norms, gathered.norms);

        let expected = quantize_rabitq_plus_matrix(
            &dataset.data[..rows * dim],
            rows,
            dim,
            8,
            RABITQ_DEFAULT_K_ALIGN,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
        )
        .unwrap();
        for (slot, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let got_codes =
                &gathered.codes[slot * expected.padded_dim..(slot + 1) * expected.padded_dim];
            assert_eq!(got_codes, expected.code_row(id));
            assert!((gathered.deltas[slot] - expected.deltas[id]).abs() <= f32::EPSILON);
            assert!((gathered.code_sums[slot] - expected.code_sums[id]).abs() <= f32::EPSILON);
            let raw = &dataset.data[id * dim..(id + 1) * dim];
            assert!((gathered.norms[slot] - norm(raw)).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn rabitq_plus_sidecar_partial_resident_prefix_falls_back_to_file_for_suffix_rows() {
        let rows = 6;
        let dim = 6;
        let resident_rows = 3;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("partial.rabitq8");
        let source_bytes = 8 + rows * dim * std::mem::size_of::<f32>();

        let report = create_rabitq_plus_sidecar_from_point_store(
            &store,
            &prefix,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
            source_bytes as u64,
        )
        .unwrap();
        let file_sidecar = RabitqPlusSidecar::open_validated(
            &prefix,
            rows,
            dim,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
        )
        .unwrap();
        let partial_sidecar = RabitqPlusSidecar::preload_prefix_validated(
            &prefix,
            rows,
            dim,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
            resident_rows,
        )
        .unwrap();

        assert!(partial_sidecar.is_resident());
        assert!(!partial_sidecar.is_fully_resident());
        assert_eq!(
            partial_sidecar.resident_bytes(),
            resident_rows * (report.header.padded_dim + RABITQ_PLUS_SIDECAR_META_ROW_BYTES)
        );

        let ids = [1_u32, 4, 2, 5];
        let expected = file_sidecar.gather(&ids).unwrap();
        let gathered = partial_sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.global_ids, ids);
        assert_eq!(gathered.resident_rows, 2);
        assert_eq!(gathered.file_rows, 2);
        assert!(!gathered.resident_source);
        assert_eq!(gathered.codes, expected.codes);
        assert_eq!(gathered.deltas, expected.deltas);
        assert_eq!(gathered.code_sums, expected.code_sums);
        assert_eq!(gathered.norms, expected.norms);
    }

    #[test]
    fn rabitq_plus_fused_sidecar_sketch_build_matches_separate_sketch() {
        let rows = 9;
        let dim = 7;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny_fused.rabitq8");
        let fused_sketch_path = dir.path().join("fused.sketch");
        let separate_sketch_path = dir.path().join("separate.sketch");
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 5;
        params.random_seed = 1234;

        let (report, fused_sketch) = create_rabitq_plus_sidecar_and_sketch_from_point_store(
            &store,
            &prefix,
            &fused_sketch_path,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
            8 + (rows * dim * std::mem::size_of::<f32>()) as u64,
            &params,
            DirectIoConfig::disabled(),
        )
        .unwrap();
        assert_eq!(report.sidecar.header.rows, rows);
        assert_eq!(report.sidecar.header.dim, dim);
        assert!(report.raw_read_wall > Duration::ZERO);
        assert!(report.sidecar.quantize_wall > Duration::ZERO);
        assert_eq!(fused_sketch.rows(), rows);
        assert_eq!(fused_sketch.width(), params.m_hash_bits);
        assert_eq!(fused_sketch.file_bytes(), 0);

        let separate_sketch = DiskSketchStore::build_from_point_store(
            &separate_sketch_path,
            &store,
            rows,
            &params,
            1,
        )
        .unwrap();
        for row in 0..rows {
            let mut fused_row = vec![0.0_f32; params.m_hash_bits];
            let mut separate_row = vec![0.0_f32; params.m_hash_bits];
            fused_sketch.row_copy_into(row, &mut fused_row).unwrap();
            separate_sketch
                .row_copy_into(row, &mut separate_row)
                .unwrap();
            for (got, expected) in fused_row.iter().zip(separate_row.iter()) {
                assert!((got - expected).abs() <= 1.0e-6);
            }
        }

        let sidecar = RabitqPlusSidecar::open_validated(
            &prefix,
            rows,
            dim,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
        )
        .unwrap();
        let ids = [8_u32, 0, 4, 8];
        let gathered = sidecar.gather(&ids).unwrap();
        assert_eq!(gathered.global_ids, ids);
        assert_eq!(
            gathered.codes.len(),
            ids.len() * report.sidecar.header.padded_dim
        );
    }

    #[test]
    fn rabitq_plus_resident_fused_builder_keeps_sidecar_and_sketch_in_memory() {
        let rows = 6;
        let dim = 5;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("resident_fused.rabitq8");
        let sketch_path = dir.path().join("resident.sketch");
        let mut params = PiPNNParams::default();
        params.m_hash_bits = 4;
        params.random_seed = 4321;

        let (report, sidecar, sketches) =
            create_resident_rabitq_plus_sidecar_and_sketch_from_point_store(
                &store,
                &prefix,
                &sketch_path,
                8,
                PiPNNGpuSubtreeRabitqPlusRotator::None,
                19,
                0,
                &params,
            )
            .unwrap();

        assert_eq!(report.sidecar.write_wall, Duration::ZERO);
        assert_eq!(report.sketch_write_wall, Duration::ZERO);
        assert!(sidecar.is_resident());
        assert_eq!(
            sidecar.resident_bytes(),
            report.sidecar.header.codes_bytes + report.sidecar.header.meta_rows_bytes
        );
        assert_eq!(sketches.file_bytes(), 0);
        assert_eq!(sketches.rows(), rows);
        assert_eq!(sketches.width(), params.m_hash_bits);

        let gathered = sidecar.gather(&[5_u32, 2, 5]).unwrap();
        assert!(gathered.resident_source);
        assert_eq!(gathered.global_ids, [5_u32, 2, 5]);
    }

    #[test]
    fn rabitq_split_sidecar_build_loads_and_assigns_d0_candidates() {
        let rows = 64;
        let dim = 64;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny_split.rabitq8");

        let report = create_rabitq_split_sidecar_from_point_store(
            &store,
            &prefix,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            23,
            0,
        )
        .unwrap();
        assert_eq!(report.header.rows, rows);
        assert_eq!(report.header.dim, dim);
        assert_eq!(report.header.batch_rows, RABITQ_SPLIT_BATCH_ROWS);
        assert_eq!(report.header.batch_count, rows / RABITQ_SPLIT_BATCH_ROWS);
        assert_eq!(
            report.header.batch_total_bytes,
            report.header.batch_count * report.header.batch_bytes
        );
        assert_eq!(
            report.header.ex_total_bytes,
            rows * report.header.ex_row_bytes
        );

        let sidecar = RabitqSplitSidecar::preload_validated(
            &prefix,
            rows,
            dim,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
            23,
        )
        .unwrap();
        assert_eq!(
            sidecar.resident_bytes(),
            report.header.batch_total_bytes + report.header.ex_total_bytes
        );
        let leader_ids = [0_u32, 7, 31, 63];
        let mut leaders = vec![0.0_f32; leader_ids.len() * dim];
        store.read_points_into(&leader_ids, &mut leaders).unwrap();
        let assigned = sidecar.assign_d0(&leaders, leader_ids.len(), 2, 3).unwrap();
        assert_eq!(assigned.assignments.len(), rows * 2);
        assert!(assigned.coarse_wall > Duration::ZERO);
        assert_eq!(assigned.boosted_pairs, (rows * 3) as u64);
        for &leader in &assigned.assignments {
            assert!((leader as usize) < leader_ids.len());
        }
    }

    #[test]
    fn rabitq_plus_sidecar_validation_rejects_mismatched_config() {
        let rows = 3;
        let dim = 4;
        let dataset = build_test_dataset(rows, dim);
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("tiny.rabitq8");

        create_rabitq_plus_sidecar_from_point_store(
            &store,
            &prefix,
            8,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
            0,
        )
        .unwrap();
        let err = RabitqPlusSidecar::open_validated(
            &prefix,
            rows,
            dim,
            7,
            PiPNNGpuSubtreeRabitqPlusRotator::None,
            17,
        )
        .unwrap_err();

        assert!(format!("{err:?}").contains("bits mismatch"));
    }
}
