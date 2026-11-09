use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use memmap2::{Mmap, MmapOptions};

use super::dco_sq_int8::{
    SQ_INT8_DEFAULT_BITS, SQ_INT8_DEFAULT_K_ALIGN, SqInt8Matrix, SqWmmaCodeLayout,
    SqWmmaGatheredInput, sq_int8_padded_dim,
};
use crate::common::{AnnError, AnnResult};

const MAGIC: &[u8; 8] = b"PGSRR2\0\0";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 128;
const META_ROW_BYTES: usize = 12;

fn sidecar_path(prefix: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}.{}", prefix.display(), suffix))
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Sq8RotatedResidualSidecarReport {
    pub(crate) codes_bytes: usize,
    pub(crate) meta_rows_bytes: usize,
    pub(crate) write_wall: Duration,
}

pub(crate) struct Sq8RotatedResidualSidecarWriter {
    rows: usize,
    dim: usize,
    padded_dim: usize,
    center_count: usize,
    seed: u64,
    center_hash: u64,
    codes: BufWriter<File>,
    meta: BufWriter<File>,
    written_rows: usize,
    started: Instant,
}

impl Sq8RotatedResidualSidecarWriter {
    pub(crate) fn create(
        prefix: &Path,
        rows: usize,
        dim: usize,
        seed: u64,
        rotated_centers: &[f32],
    ) -> AnnResult<Self> {
        if rows == 0
            || dim == 0
            || !dim.is_power_of_two()
            || rotated_centers.is_empty()
            || rotated_centers.len() % dim != 0
            || rotated_centers.iter().any(|value| !value.is_finite())
        {
            return Err(AnnError::log_index_error(
                "Invalid rotated residual SQ8 sidecar shape or centers".to_string(),
            ));
        }
        let center_count = rotated_centers.len() / dim;
        if center_count > u32::MAX as usize {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 sidecar has too many centers".to_string(),
            ));
        }
        let padded_dim = sq_int8_padded_dim(dim, SQ_INT8_DEFAULT_K_ALIGN)?;
        if let Some(parent) = prefix
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let started = Instant::now();
        let mut center_bytes = Vec::with_capacity(rotated_centers.len() * 4);
        for &value in rotated_centers {
            center_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let center_hash = hash_bytes(&center_bytes);
        fs::write(sidecar_path(prefix, "centers"), &center_bytes)?;
        let codes = BufWriter::new(File::create(sidecar_path(prefix, "codes"))?);
        let mut meta = BufWriter::new(File::create(sidecar_path(prefix, "meta"))?);
        meta.write_all(&[0_u8; HEADER_BYTES])?;
        Ok(Self {
            rows,
            dim,
            padded_dim,
            center_count,
            seed,
            center_hash,
            codes,
            meta,
            written_rows: 0,
            started,
        })
    }

    pub(crate) fn append(
        &mut self,
        row_start: usize,
        matrix: &SqInt8Matrix,
        center_ids: &[u32],
    ) -> AnnResult<()> {
        if row_start != self.written_rows
            || matrix.dim != self.dim
            || matrix.padded_dim != self.padded_dim
            || matrix.codes.len() != matrix.rows.saturating_mul(self.padded_dim)
            || matrix.scales.len() != matrix.rows
            || matrix.norms.len() != matrix.rows
            || center_ids.len() != matrix.rows
            || row_start.saturating_add(matrix.rows) > self.rows
        {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 sidecar shard shape or order mismatch".to_string(),
            ));
        }
        let codes = unsafe {
            std::slice::from_raw_parts(matrix.codes.as_ptr().cast::<u8>(), matrix.codes.len())
        };
        self.codes.write_all(codes)?;
        let mut meta = Vec::with_capacity(matrix.rows * META_ROW_BYTES);
        for row in 0..matrix.rows {
            let center_id = center_ids[row];
            let scale = matrix.scales[row];
            let norm = matrix.norms[row];
            if center_id as usize >= self.center_count
                || !scale.is_finite()
                || scale <= 0.0
                || !norm.is_finite()
                || norm < 0.0
            {
                return Err(AnnError::log_index_error(format!(
                    "Invalid rotated residual SQ8 metadata at row {}",
                    row_start + row
                )));
            }
            meta.extend_from_slice(&center_id.to_le_bytes());
            meta.extend_from_slice(&scale.to_le_bytes());
            meta.extend_from_slice(&norm.to_le_bytes());
        }
        self.meta.write_all(&meta)?;
        self.written_rows += matrix.rows;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> AnnResult<Sq8RotatedResidualSidecarReport> {
        if self.written_rows != self.rows {
            return Err(AnnError::log_index_error(format!(
                "Rotated residual SQ8 sidecar incomplete: written={} expected={}",
                self.written_rows, self.rows
            )));
        }
        let codes_bytes = self.rows.saturating_mul(self.padded_dim);
        let meta_rows_bytes = self.rows.saturating_mul(META_ROW_BYTES);
        let mut header = [0_u8; HEADER_BYTES];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        header[12..20].copy_from_slice(&(self.rows as u64).to_le_bytes());
        header[20..24].copy_from_slice(&(self.dim as u32).to_le_bytes());
        header[24..28].copy_from_slice(&(self.padded_dim as u32).to_le_bytes());
        header[28..32].copy_from_slice(&(self.center_count as u32).to_le_bytes());
        header[32..40].copy_from_slice(&self.seed.to_le_bytes());
        header[40..48].copy_from_slice(&(codes_bytes as u64).to_le_bytes());
        header[48..56].copy_from_slice(&(meta_rows_bytes as u64).to_le_bytes());
        header[56..64].copy_from_slice(&self.center_hash.to_le_bytes());
        self.codes.flush()?;
        self.meta.flush()?;
        self.meta.get_mut().seek(SeekFrom::Start(0))?;
        self.meta.get_mut().write_all(&header)?;
        self.meta.get_ref().sync_all()?;
        self.codes.get_ref().sync_all()?;
        Ok(Sq8RotatedResidualSidecarReport {
            codes_bytes,
            meta_rows_bytes,
            write_wall: self.started.elapsed(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct Sq8RotatedResidualSidecar {
    rows: usize,
    dim: usize,
    padded_dim: usize,
    centers: Vec<f32>,
    codes: Mmap,
    meta: Mmap,
}

impl Sq8RotatedResidualSidecar {
    pub(crate) fn open(prefix: &Path, rows: usize, dim: usize, seed: u64) -> AnnResult<Self> {
        let mut meta_file = File::open(sidecar_path(prefix, "meta"))?;
        let mut header = [0_u8; HEADER_BYTES];
        meta_file.read_exact(&mut header)?;
        let read_u32 = |start| u32::from_le_bytes(header[start..start + 4].try_into().unwrap());
        let read_u64 = |start| u64::from_le_bytes(header[start..start + 8].try_into().unwrap());
        let padded_dim = read_u32(24) as usize;
        let center_count = read_u32(28) as usize;
        let codes_bytes = read_u64(40) as usize;
        let meta_rows_bytes = read_u64(48) as usize;
        if &header[..8] != MAGIC
            || read_u32(8) != VERSION
            || read_u64(12) as usize != rows
            || read_u32(20) as usize != dim
            || read_u64(32) != seed
            || padded_dim != sq_int8_padded_dim(dim, SQ_INT8_DEFAULT_K_ALIGN)?
            || center_count == 0
            || codes_bytes != rows.saturating_mul(padded_dim)
            || meta_rows_bytes != rows.saturating_mul(META_ROW_BYTES)
            || meta_file.metadata()?.len() as usize != HEADER_BYTES + meta_rows_bytes
        {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 sidecar header mismatch".to_string(),
            ));
        }
        let code_file = File::open(sidecar_path(prefix, "codes"))?;
        if code_file.metadata()?.len() as usize != codes_bytes {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 sidecar code file size mismatch".to_string(),
            ));
        }
        let center_bytes = fs::read(sidecar_path(prefix, "centers"))?;
        if center_bytes.len() != center_count.saturating_mul(dim).saturating_mul(4)
            || hash_bytes(&center_bytes) != read_u64(56)
        {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 sidecar center table mismatch".to_string(),
            ));
        }
        let centers = center_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        if centers.iter().any(|value| !value.is_finite()) {
            return Err(AnnError::log_index_error(
                "Rotated residual SQ8 sidecar contains non-finite center".to_string(),
            ));
        }
        let codes = unsafe { MmapOptions::new().map(&code_file)? };
        let meta = unsafe { MmapOptions::new().map(&meta_file)? };
        Ok(Self {
            rows,
            dim,
            padded_dim,
            centers,
            codes,
            meta,
        })
    }

    pub(crate) fn gather_reconstructed(
        &self,
        global_ids: &[u32],
    ) -> AnnResult<SqWmmaGatheredInput> {
        self.gather(global_ids, true)
    }

    pub(crate) fn gather_quantized(&self, global_ids: &[u32]) -> AnnResult<SqWmmaGatheredInput> {
        self.gather(global_ids, false)
    }

    fn gather(&self, global_ids: &[u32], reconstruct: bool) -> AnnResult<SqWmmaGatheredInput> {
        let started = Instant::now();
        let rows = global_ids.len();
        let center_count = self.centers.len() / self.dim;
        let mut codes = Vec::with_capacity(rows.saturating_mul(self.padded_dim));
        let mut scales = Vec::with_capacity(rows);
        let mut norms = Vec::with_capacity(rows);
        let mut center_ids = Vec::with_capacity(rows);
        let mut vectors = reconstruct.then(|| vec![0.0_f32; rows.saturating_mul(self.dim)]);
        for (row, &global_id) in global_ids.iter().enumerate() {
            let id = global_id as usize;
            if id >= self.rows {
                return Err(AnnError::log_index_error(format!(
                    "Rotated residual SQ8 sidecar point ID {id} exceeds {} rows",
                    self.rows
                )));
            }
            let meta_base = HEADER_BYTES + id * META_ROW_BYTES;
            let center_id =
                u32::from_le_bytes(self.meta[meta_base..meta_base + 4].try_into().unwrap())
                    as usize;
            let scale =
                f32::from_le_bytes(self.meta[meta_base + 4..meta_base + 8].try_into().unwrap());
            let residual_norm =
                f32::from_le_bytes(self.meta[meta_base + 8..meta_base + 12].try_into().unwrap());
            if center_id >= center_count
                || !scale.is_finite()
                || scale <= 0.0
                || !residual_norm.is_finite()
                || residual_norm < 0.0
            {
                return Err(AnnError::log_index_error(format!(
                    "Invalid rotated residual SQ8 metadata for point {id}"
                )));
            }
            let code_base = id * self.padded_dim;
            let code = &self.codes[code_base..code_base + self.padded_dim];
            codes.extend(code.iter().map(|&value| value as i8));
            let norm = if let Some(vectors) = vectors.as_mut() {
                let center = &self.centers[center_id * self.dim..(center_id + 1) * self.dim];
                let output = &mut vectors[row * self.dim..(row + 1) * self.dim];
                let mut norm = 0.0_f32;
                for axis in 0..self.dim {
                    let value = center[axis] + scale * (code[axis] as i8 as f32);
                    output[axis] = value;
                    norm += value * value;
                }
                norm
            } else {
                residual_norm
            };
            center_ids.push(center_id as u32);
            scales.push(scale);
            norms.push(norm);
        }
        let gather_wall = started.elapsed();
        Ok(SqWmmaGatheredInput {
            global_ids: global_ids.to_vec(),
            codes,
            scales,
            norms,
            code_sums: vec![0.0_f32; rows],
            dim: self.dim,
            padded_dim: self.padded_dim,
            bits: SQ_INT8_DEFAULT_BITS,
            code_layout: SqWmmaCodeLayout::DenseI8,
            code_row_bytes: self.padded_dim,
            gather_wall,
            resident_source: false,
            resident_rows: 0,
            file_rows: rows,
            resident_gather_wall: Duration::ZERO,
            file_gather_wall: gather_wall,
            reconstructed_vectors: vectors,
            rotated_center_ids: (!reconstruct).then_some(center_ids),
            rotated_centers: (!reconstruct).then(|| self.centers.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::pipnn::dco_sq_int8::quantize_sq8_rotated_residual_matrix;

    #[test]
    fn sidecar_roundtrip_reconstructs_vectors_across_centers() {
        let centers = [0.0_f32, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0];
        let points = [0.2_f32, -0.1, 0.0, 0.0, 100.1, 0.2, 0.0, 0.0];
        let encoded = quantize_sq8_rotated_residual_matrix(&points, 2, 4, &centers, 42).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("rrsq8");
        let mut writer =
            Sq8RotatedResidualSidecarWriter::create(&prefix, 2, 4, 42, &encoded.rotated_centroids)
                .unwrap();
        writer
            .append(0, &encoded.residuals, &encoded.centroid_ids)
            .unwrap();
        let report = writer.finish().unwrap();
        assert_eq!(report.codes_bytes, 2 * encoded.residuals.padded_dim);
        let sidecar = Sq8RotatedResidualSidecar::open(&prefix, 2, 4, 42).unwrap();
        let gathered = sidecar.gather_reconstructed(&[1, 0]).unwrap();
        let vectors = gathered.reconstructed_vectors.unwrap();
        let dist: f32 = vectors[..4]
            .iter()
            .zip(&vectors[4..])
            .map(|(&left, &right)| (left - right) * (left - right))
            .sum();
        assert!((dist - encoded.approx_l2(0, 1)).abs() < 1e-3);
        let direct = sidecar.gather_quantized(&[1, 0]).unwrap();
        assert!(direct.reconstructed_vectors.is_none());
        assert_eq!(direct.rotated_center_ids.as_deref(), Some(&[1, 0][..]));
        let center_ids = direct.rotated_center_ids.as_ref().unwrap();
        let centers = direct.rotated_centers.as_ref().unwrap();
        let left_center = &centers[center_ids[0] as usize * 4..(center_ids[0] as usize + 1) * 4];
        let right_center = &centers[center_ids[1] as usize * 4..(center_ids[1] as usize + 1) * 4];
        let left_code = &direct.codes[..direct.padded_dim];
        let right_code = &direct.codes[direct.padded_dim..2 * direct.padded_dim];
        let code_dot: i32 = left_code
            .iter()
            .zip(right_code)
            .map(|(&a, &b)| i32::from(a) * i32::from(b))
            .sum();
        let project = |code: &[i8], scale: f32, center: &[f32]| -> f32 {
            code[..4]
                .iter()
                .zip(center)
                .map(|(&q, &m)| scale * f32::from(q) * m)
                .sum()
        };
        let left_full_norm = direct.norms[0]
            + 2.0 * project(left_code, direct.scales[0], left_center)
            + left_center.iter().map(|x| x * x).sum::<f32>();
        let right_full_norm = direct.norms[1]
            + 2.0 * project(right_code, direct.scales[1], right_center)
            + right_center.iter().map(|x| x * x).sum::<f32>();
        let full_dot = direct.scales[0] * direct.scales[1] * code_dot as f32
            + project(left_code, direct.scales[0], right_center)
            + project(right_code, direct.scales[1], left_center)
            + left_center
                .iter()
                .zip(right_center)
                .map(|(a, b)| a * b)
                .sum::<f32>();
        let direct_dist = (left_full_norm + right_full_norm - 2.0 * full_dot).max(0.0);
        assert!((direct_dist - dist).abs() < 0.02);
        assert!(Sq8RotatedResidualSidecar::open(&prefix, 2, 4, 43).is_err());
    }
}
