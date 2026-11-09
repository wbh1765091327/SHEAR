//! A packed residual code per base vector, keyed by its nearest D0 center.
use super::dco_rabitq::FhtKacRotationPlan;
use super::dco_sq_int8::{
    SqWmmaCodeLayout, SqWmmaGatheredInput, pack_sq_s4_codes, quantize_sq_int8_row_with_bits,
};
use crate::common::{AnnError, AnnResult};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub(super) struct PrimaryChunk {
    codes: Vec<i8>,
    scales: Vec<f32>,
    norms: Vec<f32>,
    centers: Vec<u16>,
}

impl PrimaryChunk {
    /// CPU control for moving partition encoding into D0 without changing its math.
    pub(super) fn encode_memberships(data: &[f32], dim: usize, padded: usize,
        centers: &[f32], selected: &[u32], fanout: usize,
        rotation: &FhtKacRotationPlan) -> AnnResult<Self> {
        use rayon::prelude::*;
        let rows = data.len() / dim;
        let ranks = (0..fanout).into_par_iter().map(|rank| Self::encode(data, dim, padded,
            centers, (0..rows).map(|r| selected[r * fanout + rank] as usize), rotation))
            .collect::<AnnResult<Vec<_>>>()?;
        let mut result = Self { codes: Vec::with_capacity(rows * fanout * (padded / 2)),
            scales: Vec::with_capacity(rows * fanout), norms: Vec::with_capacity(rows * fanout),
            centers: Vec::with_capacity(rows * fanout) };
        for row in 0..rows {
            for chunk in &ranks {
                result.codes.extend_from_slice(&chunk.codes[row * (padded/2)..(row+1) * (padded/2)]);
                result.scales.push(chunk.scales[row]);
                result.norms.push(chunk.norms[row]);
                result.centers.push(chunk.centers[row]);
            }
        }
        Ok(result)
    }
    #[cfg(feature = "pipnn_cuda")]
    pub(super) fn from_gpu(codes: Vec<i8>, scales: Vec<f32>, norms: Vec<f32>,
        centers: Vec<u16>, padded: usize) -> AnnResult<Self> {
        let rows = centers.len();
        if padded % 2 != 0 || codes.len() != rows.saturating_mul(padded / 2)
            || scales.len() != rows || norms.len() != rows
            || scales.iter().any(|s| !s.is_finite() || *s <= 0.0)
            || norms.iter().any(|n| !n.is_finite() || *n < 0.0) {
            return Err(AnnError::log_index_error("Invalid GPU primary SQ4+ payload".to_string()));
        }
        Ok(Self {codes, scales, norms, centers})
    }

    pub(super) fn encode(
        data: &[f32],
        dim: usize,
        padded: usize,
        centers: &[f32],
        nearest: impl Iterator<Item = usize>,
        rotation: &FhtKacRotationPlan,
    ) -> AnnResult<Self> {
        let rows = data.len() / dim;
        let mut result = Self::default();
        let mut dense = vec![0i8; rows * padded];
        let mut residual = vec![0.0; dim];
        let mut rotated = vec![0.0; dim];
        for (r, (x, center)) in data.chunks_exact(dim).zip(nearest).enumerate() {
            let mean = &centers[center * dim..(center + 1) * dim];
            for ((out, a), b) in residual.iter_mut().zip(x).zip(mean) {
                *out = a - b;
            }
            rotation.rotate_row(&residual, &mut rotated)?;
            let (scale, norm) = quantize_sq_int8_row_with_bits(
                &rotated,
                &mut dense[r * padded..(r + 1) * padded],
                4,
            )?;
            if !norm.is_finite() || scale <= 0.0 {
                return Err(AnnError::log_index_error("Invalid primary SQ4+ residual".to_string()));
            }
            result.centers.push(u16::try_from(center).map_err(|_| {
                AnnError::log_index_error("SQ4 residual center ID exceeds u16".into())
            })?);
            result.scales.push(scale);
            result.norms.push(norm);
        }
        result.codes = pack_sq_s4_codes(&dense, rows, padded)?;
        Ok(result)
    }
}

/// Partition-specific codes stored by source row and membership rank. Child IDs
/// map through nonempty D0 centers, including when admission later splits a child.
#[derive(Clone, Debug)]
pub(crate) struct PartitionCodes {
    chunks: Vec<PrimaryChunk>,
    positions: Vec<usize>,
    child_centers: Vec<usize>,
    chunk_rows: usize,
    fanout: usize,
    dim: usize,
    padded: usize,
}

impl PartitionCodes {
    pub(super) fn new(chunks: Vec<PrimaryChunk>, ids: &[u32], total: usize,
        chunk_rows: usize, fanout: usize, dim: usize, padded: usize,
        child_centers: Vec<usize>) -> AnnResult<Self> {
        if chunk_rows == 0 || fanout == 0 || chunks.len() != ids.len().div_ceil(chunk_rows) {
            return Err(AnnError::log_index_error("Invalid partition code table shape".into()));
        }
        let mut positions = vec![usize::MAX; total];
        for (pos, &id) in ids.iter().enumerate() {
            let slot = positions.get_mut(id as usize).ok_or_else(||
                AnnError::log_index_error("Partition code ID outside dataset".into()))?;
            if *slot != usize::MAX { return Err(AnnError::log_index_error("Duplicate partition source ID".into())); }
            *slot = pos;
        }
        for (i, chunk) in chunks.iter().enumerate() {
            let rows = chunk_rows.min(ids.len() - i * chunk_rows) * fanout;
            if chunk.centers.len() != rows || chunk.codes.len() != rows * (padded/2)
                || chunk.scales.len() != rows || chunk.norms.len() != rows {
                return Err(AnnError::log_index_error("Incomplete partition code batch".into()));
            }
        }
        Ok(Self {chunks, positions, child_centers, chunk_rows, fanout, dim, padded})
    }

    pub(super) fn gather(&self, child: usize, ids: &[u32]) -> AnnResult<SqWmmaGatheredInput> {
        let start = Instant::now();
        let center = *self.child_centers.get(child).ok_or_else(||
            AnnError::log_index_error("Missing partition code center".into()))?;
        let stride = self.padded/2;
        let mut codes = Vec::with_capacity(ids.len()*stride);
        let mut scales = Vec::with_capacity(ids.len());
        let mut norms = Vec::with_capacity(ids.len());
        for &id in ids {
            let pos = self.positions.get(id as usize).copied().unwrap_or(usize::MAX);
            if pos == usize::MAX { return Err(AnnError::log_index_error(format!("Missing partition row {id}"))); }
            let chunk = &self.chunks[pos / self.chunk_rows];
            let base = (pos % self.chunk_rows) * self.fanout;
            let rank = chunk.centers[base..base+self.fanout].iter().position(|&c| c as usize == center)
                .ok_or_else(|| AnnError::log_index_error(format!("Row {id} is not in partition {center}")))?;
            let row = base + rank;
            codes.extend_from_slice(&chunk.codes[row*stride..(row+1)*stride]);
            scales.push(chunk.scales[row]); norms.push(chunk.norms[row]);
        }
        let elapsed = start.elapsed();
        Ok(SqWmmaGatheredInput {global_ids: ids.to_vec(), codes, scales, norms,
            code_sums: vec![0.0; ids.len()], dim: self.dim, padded_dim: self.padded,
            bits: 4, code_row_bytes: stride, code_layout: SqWmmaCodeLayout::PackedS4,
            gather_wall: elapsed, resident_source: true, resident_rows: ids.len(), file_rows: 0,
            resident_gather_wall: elapsed, file_gather_wall: Duration::ZERO,
            reconstructed_vectors: None, rotated_center_ids: None, rotated_centers: None})
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PrimaryCodes {
    chunks: Vec<PrimaryChunk>,
    positions: Vec<usize>,
    chunk_rows: usize,
    dim: usize,
    padded: usize,
    rotated_centers: Vec<f32>,
}

impl PrimaryCodes {
    /// Optional experiment audit; disabled during speed measurements.
    pub(super) fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"SQ4-PRIMARY-PAYLOAD-v1\0");
        hash.update(bytemuck::cast_slice::<f32, u8>(&self.rotated_centers));
        for chunk in &self.chunks {
            hash.update(bytemuck::cast_slice::<i8, u8>(&chunk.codes));
            hash.update(bytemuck::cast_slice::<f32, u8>(&chunk.scales));
            hash.update(bytemuck::cast_slice::<f32, u8>(&chunk.norms));
            // Preserve the v1 primary-code fingerprint (primary centers <=256).
            for &center in &chunk.centers { hash.update([center as u8]); }
        }
        hex::encode(hash.finalize())
    }

    pub(super) fn new(
        chunks: Vec<PrimaryChunk>,
        ids: &[u32],
        total: usize,
        chunk_rows: usize,
        dim: usize,
        padded: usize,
        centers: &[f32],
        rotation: &FhtKacRotationPlan,
    ) -> AnnResult<Self> {
        let mut positions = vec![usize::MAX; total];
        for (i, &id) in ids.iter().enumerate() {
            let pos = positions.get_mut(id as usize).ok_or_else(|| {
                AnnError::log_index_error(format!("Primary SQ4+ row {id} outside D0 dataset of {total} rows"))
            })?;
            *pos = i;
        }
        let mut rotated_centers = vec![0.0; centers.len()];
        for (src, dst) in centers.chunks_exact(dim).zip(rotated_centers.chunks_exact_mut(dim)) {
            rotation.rotate_row(src, dst)?;
        }
        Ok(Self {
            chunks,
            positions,
            chunk_rows,
            dim,
            padded,
            rotated_centers,
        })
    }

    pub(super) fn gather(&self, ids: &[u32]) -> AnnResult<SqWmmaGatheredInput> {
        let start = Instant::now();
        let stride = self.padded / 2;
        let mut codes = Vec::with_capacity(ids.len() * stride);
        let mut scales = Vec::with_capacity(ids.len());
        let mut norms = Vec::with_capacity(ids.len());
        let mut centers = Vec::with_capacity(ids.len());
        for id in ids {
            let pos = self.positions.get(*id as usize).copied().unwrap_or(usize::MAX);
            if pos == usize::MAX {
                return Err(AnnError::log_index_error(format!("Missing primary SQ4+ row {id}")));
            }
            let chunk = &self.chunks[pos / self.chunk_rows];
            let row = pos % self.chunk_rows;
            codes.extend_from_slice(&chunk.codes[row * stride..(row + 1) * stride]);
            scales.push(chunk.scales[row]);
            norms.push(chunk.norms[row]);
            centers.push(u32::from(chunk.centers[row]));
        }
        let elapsed = start.elapsed();
        Ok(SqWmmaGatheredInput {
            global_ids: ids.to_vec(),
            codes,
            scales,
            norms,
            code_sums: vec![0.0; ids.len()],
            dim: self.dim,
            padded_dim: self.padded,
            bits: 4,
            code_row_bytes: stride,
            code_layout: SqWmmaCodeLayout::PackedS4,
            gather_wall: elapsed,
            resident_source: true,
            resident_rows: ids.len(),
            file_rows: 0,
            resident_gather_wall: elapsed,
            file_gather_wall: Duration::ZERO,
            reconstructed_vectors: None,
            rotated_center_ids: Some(centers),
            rotated_centers: Some(self.rotated_centers.clone()),
        })
    }
}
