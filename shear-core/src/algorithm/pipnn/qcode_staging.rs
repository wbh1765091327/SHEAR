use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::dco_rabitq::{
    RABITQ_DEFAULT_K_ALIGN, RABITQ_DEFAULT_TOTAL_BITS, pack_rabitq_dense_i8_row_to_split,
    quantize_rabitq_matrix, rabitq_dot_from_i8, rabitq_dot_from_split, rabitq_padded_dim,
    rabitq_split_ex_row_bytes, rabitq_split_short_words_per_row, rotate_rabitq_plus_fht_kac_row,
    unpack_rabitq_split_row_to_dense_i8,
};
use super::params::PiPNNGpuSubtreeRabitqPlusRotator;
use crate::common::{AnnError, AnnResult};

const RABITQ_RESIDUAL_QCODE_META_ROW_BYTES: usize = 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RabitqResidualPayloadHeader {
    pub(crate) rows: usize,
    pub(crate) dim: usize,
    pub(crate) padded_dim: usize,
    pub(crate) centroid_count: usize,
    pub(crate) code_row_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqResidualShardPayload {
    pub(crate) global_ids: Vec<u32>,
    pub(crate) centroid_ids: Vec<u32>,
    pub(crate) codes: Vec<i8>,
    pub(crate) residual_norms: Vec<f32>,
    pub(crate) scales: Vec<f32>,
    pub(crate) code_sums: Vec<f32>,
    pub(crate) original_norms: Vec<f32>,
    pub(crate) code_row_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqResidualPayloadStore {
    header: RabitqResidualPayloadHeader,
    centroid_ids: Vec<u32>,
    codes: Vec<i8>,
    residual_norms: Vec<f32>,
    scales: Vec<f32>,
    code_sums: Vec<f32>,
    original_norms: Vec<f32>,
    centroids_row_major: Vec<f32>,
    written: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqResidualSplitPayloadStore {
    header: RabitqResidualPayloadHeader,
    global_ids: Vec<u32>,
    centroid_ids: Vec<u32>,
    short_data: Vec<u32>,
    ex_data: Vec<u8>,
    residual_norms: Vec<f32>,
    scales: Vec<f32>,
    code_sums: Vec<f32>,
    original_norms: Vec<f32>,
    centroids_row_major: Vec<f32>,
    short_words_per_row: usize,
    ex_row_bytes: usize,
    ex_bits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct D0AssignmentFragment {
    pub(crate) child_id: usize,
    pub(crate) point_id: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RabitqResidualGpuStagingProfile {
    pub(crate) gpu_quantize_wall: Duration,
    pub(crate) gpu_d0_assign_wall: Duration,
    pub(crate) leader_payload_h2d_wall: Duration,
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
pub(crate) struct RabitqResidualQuantizeD0Request<'a> {
    pub(crate) raw_vectors_row_major: &'a [f32],
    pub(crate) global_ids: &'a [u32],
    pub(crate) centroid_ids: &'a [u32],
    pub(crate) centroids_row_major: &'a [f32],
    pub(crate) global_d0_leader_ids: &'a [u32],
    pub(crate) global_d0_leader_vectors_row_major: &'a [f32],
    pub(crate) leader_payload: Option<&'a RabitqResidualShardPayload>,
    pub(crate) dim: usize,
    pub(crate) fanout_top: usize,
    pub(crate) rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    pub(crate) rotator_seed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqResidualQuantizeD0Result {
    pub(crate) payload: RabitqResidualShardPayload,
    pub(crate) fragments: Vec<D0AssignmentFragment>,
    pub(crate) profile: RabitqResidualGpuStagingProfile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResidentRabitqChildTopKRequest<'a> {
    pub(crate) payload: RabitqResidualD1PayloadRef<'a>,
    pub(crate) child_global_ids: &'a [u32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RabitqResidualD1PayloadRef<'a> {
    FullResident(&'a RabitqResidualPayloadStore),
    ChildLocal(&'a RabitqResidualPayloadStore),
    ChildLocalSplit(&'a RabitqResidualSplitPayloadStore),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QcodeShardRowRef {
    shard_id: u32,
    row_offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QcodeGatherRow {
    local_row: usize,
    global_id: u32,
    row_ref: QcodeShardRowRef,
}

#[derive(Debug)]
pub(crate) struct RabitqResidualQcodeShardStore {
    root: PathBuf,
    header: RabitqResidualPayloadHeader,
    centroids_row_major: Vec<f32>,
    row_index: Vec<Option<QcodeShardRowRef>>,
    shard_count: usize,
    write_dense_codes: bool,
}

impl RabitqResidualPayloadStore {
    pub(crate) fn new(header: RabitqResidualPayloadHeader) -> AnnResult<Self> {
        if header.rows == 0 {
            return Err(AnnError::log_index_error(
                "RabitQ residual payload store requires rows > 0".to_string(),
            ));
        }
        if header.dim == 0 || header.padded_dim < header.dim || header.code_row_bytes == 0 {
            return Err(AnnError::log_index_error(format!(
                "Invalid RabitQ residual payload shape: rows={} dim={} padded_dim={} code_row_bytes={}",
                header.rows, header.dim, header.padded_dim, header.code_row_bytes
            )));
        }
        if header.centroid_count == 0 {
            return Err(AnnError::log_index_error(
                "RabitQ residual payload store requires centroid_count > 0".to_string(),
            ));
        }
        let code_len = header
            .rows
            .checked_mul(header.code_row_bytes)
            .ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "RabitQ residual payload code length overflows: rows={} row_bytes={}",
                    header.rows, header.code_row_bytes
                ))
            })?;
        let centroid_len = header
            .centroid_count
            .checked_mul(header.dim)
            .ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "RabitQ residual centroid table length overflows: centroids={} dim={}",
                    header.centroid_count, header.dim
                ))
            })?;
        Ok(Self {
            centroid_ids: vec![0; header.rows],
            codes: vec![0; code_len],
            residual_norms: vec![0.0; header.rows],
            scales: vec![0.0; header.rows],
            code_sums: vec![0.0; header.rows],
            original_norms: vec![0.0; header.rows],
            centroids_row_major: vec![0.0; centroid_len],
            written: vec![false; header.rows],
            header,
        })
    }

    pub(crate) fn header(&self) -> &RabitqResidualPayloadHeader {
        &self.header
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.written.iter().all(|&written| written)
    }

    pub(crate) fn centroid_ids(&self) -> &[u32] {
        &self.centroid_ids
    }

    pub(crate) fn codes(&self) -> &[i8] {
        &self.codes
    }

    pub(crate) fn residual_norms(&self) -> &[f32] {
        &self.residual_norms
    }

    pub(crate) fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub(crate) fn code_sums(&self) -> &[f32] {
        &self.code_sums
    }

    pub(crate) fn original_norms(&self) -> &[f32] {
        &self.original_norms
    }

    pub(crate) fn centroids_row_major(&self) -> &[f32] {
        &self.centroids_row_major
    }

    pub(crate) fn set_centroids_row_major(&mut self, centroids: &[f32]) -> AnnResult<()> {
        let expected = self
            .header
            .centroid_count
            .checked_mul(self.header.dim)
            .ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "RabitQ residual centroid table length overflows: centroids={} dim={}",
                    self.header.centroid_count, self.header.dim
                ))
            })?;
        if centroids.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "RabitQ residual centroid table shape mismatch: got={} expected={} centroids={} dim={}",
                centroids.len(),
                expected,
                self.header.centroid_count,
                self.header.dim
            )));
        }
        self.centroids_row_major.copy_from_slice(centroids);
        Ok(())
    }

    pub(crate) fn device_payload_bytes(&self) -> usize {
        self.codes
            .len()
            .saturating_mul(std::mem::size_of::<i8>())
            .saturating_add(
                self.centroid_ids
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.scales.len().saturating_mul(std::mem::size_of::<f32>()))
            .saturating_add(
                self.original_norms
                    .len()
                    .saturating_mul(std::mem::size_of::<f32>()),
            )
            .saturating_add(
                self.centroids_row_major
                    .len()
                    .saturating_mul(std::mem::size_of::<f32>()),
            )
    }

    pub(crate) fn write_shard(&mut self, shard: &RabitqResidualShardPayload) -> AnnResult<()> {
        let rows = shard.global_ids.len();
        if shard.code_row_bytes != self.header.code_row_bytes {
            return Err(AnnError::log_index_error(format!(
                "RabitQ residual shard code row bytes mismatch: got={} expected={}",
                shard.code_row_bytes, self.header.code_row_bytes
            )));
        }
        if shard.centroid_ids.len() != rows
            || shard.residual_norms.len() != rows
            || shard.scales.len() != rows
            || shard.code_sums.len() != rows
            || shard.original_norms.len() != rows
            || shard.codes.len() != rows.saturating_mul(shard.code_row_bytes)
        {
            return Err(AnnError::log_index_error(format!(
                "RabitQ residual shard shape mismatch: ids={} centroids={} codes={} row_bytes={} residual_norms={} scales={} code_sums={} original_norms={}",
                rows,
                shard.centroid_ids.len(),
                shard.codes.len(),
                shard.code_row_bytes,
                shard.residual_norms.len(),
                shard.scales.len(),
                shard.code_sums.len(),
                shard.original_norms.len()
            )));
        }
        for (row, &global_id) in shard.global_ids.iter().enumerate() {
            let dst = global_id as usize;
            if dst >= self.header.rows {
                return Err(AnnError::log_index_error(format!(
                    "RabitQ residual shard global id {global_id} out of bounds for {} rows",
                    self.header.rows
                )));
            }
            let centroid_id = shard.centroid_ids[row] as usize;
            if centroid_id >= self.header.centroid_count {
                return Err(AnnError::log_index_error(format!(
                    "RabitQ residual shard centroid id {centroid_id} out of bounds for {} centroids",
                    self.header.centroid_count
                )));
            }
            let src_code = row * shard.code_row_bytes;
            let dst_code = dst * self.header.code_row_bytes;
            self.codes[dst_code..dst_code + self.header.code_row_bytes]
                .copy_from_slice(&shard.codes[src_code..src_code + shard.code_row_bytes]);
            self.centroid_ids[dst] = shard.centroid_ids[row];
            self.residual_norms[dst] = shard.residual_norms[row];
            self.scales[dst] = shard.scales[row];
            self.code_sums[dst] = shard.code_sums[row];
            self.original_norms[dst] = shard.original_norms[row];
            self.written[dst] = true;
        }
        Ok(())
    }

    pub(crate) fn centroid_id(&self, global_id: usize) -> Option<u32> {
        self.centroid_ids.get(global_id).copied()
    }

    pub(crate) fn code_row(&self, global_id: usize) -> Option<&[i8]> {
        let start = global_id.checked_mul(self.header.code_row_bytes)?;
        self.codes.get(start..start + self.header.code_row_bytes)
    }

    pub(crate) fn residual_norm(&self, global_id: usize) -> Option<f32> {
        self.residual_norms.get(global_id).copied()
    }

    pub(crate) fn original_norm(&self, global_id: usize) -> Option<f32> {
        self.original_norms.get(global_id).copied()
    }

    pub(crate) fn gather_child_payload(
        &self,
        child_global_ids: &[u32],
    ) -> AnnResult<RabitqResidualPayloadStore> {
        let mut child = RabitqResidualPayloadStore::new(RabitqResidualPayloadHeader {
            rows: child_global_ids.len(),
            dim: self.header.dim,
            padded_dim: self.header.padded_dim,
            centroid_count: self.header.centroid_count,
            code_row_bytes: self.header.code_row_bytes,
        })?;
        child.set_centroids_row_major(&self.centroids_row_major)?;
        let mut shard = RabitqResidualShardPayload {
            global_ids: Vec::with_capacity(child_global_ids.len()),
            centroid_ids: Vec::with_capacity(child_global_ids.len()),
            codes: Vec::with_capacity(
                child_global_ids
                    .len()
                    .saturating_mul(self.header.code_row_bytes),
            ),
            residual_norms: Vec::with_capacity(child_global_ids.len()),
            scales: Vec::with_capacity(child_global_ids.len()),
            code_sums: Vec::with_capacity(child_global_ids.len()),
            original_norms: Vec::with_capacity(child_global_ids.len()),
            code_row_bytes: self.header.code_row_bytes,
        };
        for (local_row, &global_id_u32) in child_global_ids.iter().enumerate() {
            let global_id = global_id_u32 as usize;
            if global_id >= self.header.rows {
                return Err(AnnError::log_index_error(format!(
                    "Residual RaBitQ child gather point {} is outside payload rows {}",
                    global_id_u32, self.header.rows
                )));
            }
            if !self.written[global_id] {
                return Err(AnnError::log_index_error(format!(
                    "Residual RaBitQ child gather point {} has no qcode payload",
                    global_id_u32
                )));
            }
            shard.global_ids.push(local_row as u32);
            shard.centroid_ids.push(self.centroid_ids[global_id]);
            shard
                .codes
                .extend_from_slice(self.code_row(global_id).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Residual RaBitQ child gather missing code row for point {global_id_u32}"
                    ))
                })?);
            shard.residual_norms.push(self.residual_norms[global_id]);
            shard.scales.push(self.scales[global_id]);
            shard.code_sums.push(self.code_sums[global_id]);
            shard.original_norms.push(self.original_norms[global_id]);
        }
        child.write_shard(&shard)?;
        Ok(child)
    }
}

impl RabitqResidualSplitPayloadStore {
    pub(crate) fn from_dense_payload(payload: &RabitqResidualPayloadStore) -> AnnResult<Self> {
        if !payload.is_complete() {
            return Err(AnnError::log_index_error(
                "Residual RaBitQ split payload conversion requires complete dense payload"
                    .to_string(),
            ));
        }
        let header = payload.header().clone();
        let ex_bits = RABITQ_DEFAULT_TOTAL_BITS - 1;
        let short_words_per_row = rabitq_split_short_words_per_row(header.dim);
        let ex_row_bytes = rabitq_split_ex_row_bytes(header.dim, ex_bits)?;
        let mut short_data = vec![0_u32; header.rows.saturating_mul(short_words_per_row)];
        let mut ex_data = vec![0_u8; header.rows.saturating_mul(ex_row_bytes)];
        for row in 0..header.rows {
            let code = payload.code_row(row).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Residual RaBitQ split payload conversion missing dense row {row}"
                ))
            })?;
            let short_start = row.checked_mul(short_words_per_row).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Residual RaBitQ split short offset overflows: row={row} words={short_words_per_row}"
                ))
            })?;
            let ex_start = row.checked_mul(ex_row_bytes).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Residual RaBitQ split ex offset overflows: row={row} row_bytes={ex_row_bytes}"
                ))
            })?;
            pack_rabitq_dense_i8_row_to_split(
                code,
                header.dim,
                ex_bits,
                &mut short_data[short_start..short_start + short_words_per_row],
                &mut ex_data[ex_start..ex_start + ex_row_bytes],
            )?;
        }
        Ok(Self {
            global_ids: (0..header.rows).map(|row| row as u32).collect(),
            centroid_ids: payload.centroid_ids().to_vec(),
            residual_norms: payload.residual_norms().to_vec(),
            scales: payload.scales().to_vec(),
            code_sums: payload.code_sums().to_vec(),
            original_norms: payload.original_norms().to_vec(),
            centroids_row_major: payload.centroids_row_major().to_vec(),
            header,
            short_data,
            ex_data,
            short_words_per_row,
            ex_row_bytes,
            ex_bits,
        })
    }

    pub(crate) fn header(&self) -> &RabitqResidualPayloadHeader {
        &self.header
    }

    pub(crate) fn is_complete(&self) -> bool {
        let rows = self.header.rows;
        let centroid_len = self.header.centroid_count.saturating_mul(self.header.dim);
        self.global_ids.len() == rows
            && self.centroid_ids.len() == rows
            && self.short_data.len() == rows.saturating_mul(self.short_words_per_row)
            && self.ex_data.len() == rows.saturating_mul(self.ex_row_bytes)
            && self.residual_norms.len() == rows
            && self.scales.len() == rows
            && self.code_sums.len() == rows
            && self.original_norms.len() == rows
            && self.centroids_row_major.len() == centroid_len
    }

    pub(crate) fn global_ids(&self) -> &[u32] {
        &self.global_ids
    }

    pub(crate) fn centroid_ids(&self) -> &[u32] {
        &self.centroid_ids
    }

    pub(crate) fn short_data(&self) -> &[u32] {
        &self.short_data
    }

    pub(crate) fn ex_data(&self) -> &[u8] {
        &self.ex_data
    }

    pub(crate) fn residual_norms(&self) -> &[f32] {
        &self.residual_norms
    }

    pub(crate) fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub(crate) fn code_sums(&self) -> &[f32] {
        &self.code_sums
    }

    pub(crate) fn original_norms(&self) -> &[f32] {
        &self.original_norms
    }

    pub(crate) fn centroids_row_major(&self) -> &[f32] {
        &self.centroids_row_major
    }

    pub(crate) fn short_words_per_row(&self) -> usize {
        self.short_words_per_row
    }

    pub(crate) fn ex_row_bytes(&self) -> usize {
        self.ex_row_bytes
    }

    pub(crate) fn ex_bits(&self) -> usize {
        self.ex_bits
    }

    pub(crate) fn decode_dense_row(&self, row: usize, out: &mut [i8]) -> AnnResult<()> {
        if row >= self.header.rows {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ split decode row {row} out of bounds {}",
                self.header.rows
            )));
        }
        let short_start = row.checked_mul(self.short_words_per_row).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Residual RaBitQ split decode short offset overflows: row={row} words={}",
                self.short_words_per_row
            ))
        })?;
        let ex_start = row.checked_mul(self.ex_row_bytes).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Residual RaBitQ split decode ex offset overflows: row={row} row_bytes={}",
                self.ex_row_bytes
            ))
        })?;
        unpack_rabitq_split_row_to_dense_i8(
            &self.short_data[short_start..short_start + self.short_words_per_row],
            &self.ex_data[ex_start..ex_start + self.ex_row_bytes],
            self.header.dim,
            self.ex_bits,
            out,
        )
    }
}

impl<'a> RabitqResidualD1PayloadRef<'a> {
    pub(crate) fn is_complete(self) -> bool {
        match self {
            RabitqResidualD1PayloadRef::FullResident(payload)
            | RabitqResidualD1PayloadRef::ChildLocal(payload) => payload.is_complete(),
            RabitqResidualD1PayloadRef::ChildLocalSplit(payload) => payload.is_complete(),
        }
    }

    pub(crate) fn code_row(self, row: usize) -> Option<&'a [i8]> {
        match self {
            RabitqResidualD1PayloadRef::FullResident(payload)
            | RabitqResidualD1PayloadRef::ChildLocal(payload) => payload.code_row(row),
            RabitqResidualD1PayloadRef::ChildLocalSplit(_) => None,
        }
    }

    pub(crate) fn header(self) -> &'a RabitqResidualPayloadHeader {
        match self {
            RabitqResidualD1PayloadRef::FullResident(payload)
            | RabitqResidualD1PayloadRef::ChildLocal(payload) => payload.header(),
            RabitqResidualD1PayloadRef::ChildLocalSplit(payload) => payload.header(),
        }
    }
}

impl RabitqResidualQcodeShardStore {
    pub(crate) fn create(
        root: impl AsRef<Path>,
        header: RabitqResidualPayloadHeader,
        centroids_row_major: &[f32],
    ) -> AnnResult<Self> {
        Self::create_with_dense_codes(root, header, centroids_row_major, true)
    }

    pub(crate) fn create_split_only(
        root: impl AsRef<Path>,
        header: RabitqResidualPayloadHeader,
        centroids_row_major: &[f32],
    ) -> AnnResult<Self> {
        Self::create_with_dense_codes(root, header, centroids_row_major, false)
    }

    fn create_with_dense_codes(
        root: impl AsRef<Path>,
        header: RabitqResidualPayloadHeader,
        centroids_row_major: &[f32],
        write_dense_codes: bool,
    ) -> AnnResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create residual RaBitQ qcode shard directory {}: {err}",
                root.display()
            ))
        })?;
        RabitqResidualPayloadStore::new(header.clone())?;
        let expected_centroids = header.centroid_count.checked_mul(header.dim).ok_or_else(
            || {
                AnnError::log_index_error(format!(
                    "Residual RaBitQ qcode centroid table length overflows: centroids={} dim={}",
                    header.centroid_count, header.dim
                ))
            },
        )?;
        if centroids_row_major.len() != expected_centroids {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode centroid table shape mismatch: got={} expected={}",
                centroids_row_major.len(),
                expected_centroids
            )));
        }
        Ok(Self {
            root,
            row_index: vec![None; header.rows],
            header,
            centroids_row_major: centroids_row_major.to_vec(),
            shard_count: 0,
            write_dense_codes,
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn header(&self) -> &RabitqResidualPayloadHeader {
        &self.header
    }

    pub(crate) fn centroids_row_major(&self) -> &[f32] {
        &self.centroids_row_major
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.row_index.iter().all(Option::is_some)
    }

    pub(crate) fn write_shard(&mut self, shard: &RabitqResidualShardPayload) -> AnnResult<()> {
        validate_rabitq_residual_shard(&self.header, shard)?;
        let shard_id = self.shard_count;
        let codes_path = self.shard_codes_path(shard_id);
        let meta_path = self.shard_meta_path(shard_id);
        if self.write_dense_codes {
            let mut codes_file = File::create(&codes_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to create residual RaBitQ qcode shard codes {}: {err}",
                    codes_path.display()
                ))
            })?;
            codes_file
                .write_all(codes_i8_as_u8(&shard.codes))
                .map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Failed to write residual RaBitQ qcode shard codes {}: {err}",
                        codes_path.display()
                    ))
                })?;
        }
        let ex_bits = RABITQ_DEFAULT_TOTAL_BITS - 1;
        let short_words_per_row = rabitq_split_short_words_per_row(self.header.dim);
        let ex_row_bytes = rabitq_split_ex_row_bytes(self.header.dim, ex_bits)?;
        let short_path = self.shard_short_path(shard_id);
        let ex_path = self.shard_ex_path(shard_id);
        let mut short_words = vec![0_u32; short_words_per_row];
        let mut ex_bytes = vec![0_u8; ex_row_bytes];
        let mut shard_short_words =
            Vec::with_capacity(shard.global_ids.len().saturating_mul(short_words_per_row));
        let mut shard_ex_bytes =
            Vec::with_capacity(shard.global_ids.len().saturating_mul(ex_row_bytes));
        for row in 0..shard.global_ids.len() {
            let code_start = row.checked_mul(shard.code_row_bytes).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Residual RaBitQ qcode split write code offset overflows: row={row} row_bytes={}",
                    shard.code_row_bytes
                ))
            })?;
            pack_rabitq_dense_i8_row_to_split(
                &shard.codes[code_start..code_start + shard.code_row_bytes],
                self.header.dim,
                ex_bits,
                &mut short_words,
                &mut ex_bytes,
            )?;
            shard_short_words.extend_from_slice(&short_words);
            shard_ex_bytes.extend_from_slice(&ex_bytes);
        }
        let mut short_file = File::create(&short_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create residual RaBitQ qcode split short shard {}: {err}",
                short_path.display()
            ))
        })?;
        write_u32_le_slice(&mut short_file, &shard_short_words).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write residual RaBitQ qcode split short shard {}: {err}",
                short_path.display()
            ))
        })?;
        let mut ex_file = File::create(&ex_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create residual RaBitQ qcode split ex shard {}: {err}",
                ex_path.display()
            ))
        })?;
        ex_file.write_all(&shard_ex_bytes).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write residual RaBitQ qcode split ex shard {}: {err}",
                ex_path.display()
            ))
        })?;
        let mut meta_file = File::create(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to create residual RaBitQ qcode shard metadata {}: {err}",
                meta_path.display()
            ))
        })?;
        let mut meta = Vec::with_capacity(
            shard
                .global_ids
                .len()
                .saturating_mul(RABITQ_RESIDUAL_QCODE_META_ROW_BYTES),
        );
        for row in 0..shard.global_ids.len() {
            push_qcode_meta_row(
                &mut meta,
                shard.centroid_ids[row],
                shard.residual_norms[row],
                shard.scales[row],
                shard.code_sums[row],
                shard.original_norms[row],
            );
        }
        meta_file.write_all(&meta).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to write residual RaBitQ qcode shard metadata {}: {err}",
                meta_path.display()
            ))
        })?;
        for (row, &global_id_u32) in shard.global_ids.iter().enumerate() {
            let global_id = global_id_u32 as usize;
            if self.row_index[global_id].is_some() {
                return Err(AnnError::log_index_error(format!(
                    "Residual RaBitQ qcode shard duplicate global id {global_id_u32}"
                )));
            }
            self.row_index[global_id] = Some(QcodeShardRowRef {
                shard_id: shard_id as u32,
                row_offset: row as u32,
            });
        }
        self.shard_count += 1;
        Ok(())
    }

    pub(crate) fn gather_child(
        &self,
        child_global_ids: &[u32],
    ) -> AnnResult<RabitqResidualPayloadStore> {
        if child_global_ids.len() > u32::MAX as usize {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode child gather is too large for local u32 ids: rows={}",
                child_global_ids.len()
            )));
        }
        let mut child = RabitqResidualPayloadStore::new(RabitqResidualPayloadHeader {
            rows: child_global_ids.len(),
            dim: self.header.dim,
            padded_dim: self.header.padded_dim,
            centroid_count: self.header.centroid_count,
            code_row_bytes: self.header.code_row_bytes,
        })?;
        child.set_centroids_row_major(&self.centroids_row_major)?;
        let row_groups = self.group_rows_by_shard(child_global_ids)?;
        let mut shard = RabitqResidualShardPayload {
            global_ids: (0..child_global_ids.len()).map(|row| row as u32).collect(),
            centroid_ids: vec![0; child_global_ids.len()],
            codes: vec![
                0;
                child_global_ids
                    .len()
                    .saturating_mul(self.header.code_row_bytes)
            ],
            residual_norms: vec![0.0; child_global_ids.len()],
            scales: vec![0.0; child_global_ids.len()],
            code_sums: vec![0.0; child_global_ids.len()],
            original_norms: vec![0.0; child_global_ids.len()],
            code_row_bytes: self.header.code_row_bytes,
        };
        let mut code_bytes = vec![0u8; self.header.code_row_bytes];
        let mut meta_row = QcodeMetaRow::default();
        for (shard_id, rows) in row_groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let codes_path = self.shard_codes_path(shard_id);
            let meta_path = self.shard_meta_path(shard_id);
            let codes_file = File::open(&codes_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode shard codes {}: {err}",
                    codes_path.display()
                ))
            })?;
            let meta_file = File::open(&meta_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode shard metadata {}: {err}",
                    meta_path.display()
                ))
            })?;
            for row in rows {
                self.read_row_from_files(
                    &codes_file,
                    &meta_file,
                    row.global_id,
                    row.row_ref,
                    &mut code_bytes,
                    &mut meta_row,
                )?;
                let code_start = row
                    .local_row
                    .checked_mul(self.header.code_row_bytes)
                    .ok_or_else(|| {
                        AnnError::log_index_error(format!(
                            "Residual RaBitQ qcode child gather code offset overflows: local_row={} row_bytes={}",
                            row.local_row, self.header.code_row_bytes
                        ))
                    })?;
                for (dst, &src) in shard.codes[code_start..code_start + self.header.code_row_bytes]
                    .iter_mut()
                    .zip(code_bytes.iter())
                {
                    *dst = src as i8;
                }
                shard.centroid_ids[row.local_row] = meta_row.centroid_id;
                shard.residual_norms[row.local_row] = meta_row.residual_norm;
                shard.scales[row.local_row] = meta_row.scale;
                shard.code_sums[row.local_row] = meta_row.code_sum;
                shard.original_norms[row.local_row] = meta_row.original_norm;
            }
        }
        child.write_shard(&shard)?;
        Ok(child)
    }

    pub(crate) fn gather_child_split(
        &self,
        child_global_ids: &[u32],
    ) -> AnnResult<RabitqResidualSplitPayloadStore> {
        let ex_bits = RABITQ_DEFAULT_TOTAL_BITS - 1;
        let short_words_per_row = rabitq_split_short_words_per_row(self.header.dim);
        let ex_row_bytes = rabitq_split_ex_row_bytes(self.header.dim, ex_bits)?;
        let child_rows = child_global_ids.len();
        let mut split = RabitqResidualSplitPayloadStore {
            header: RabitqResidualPayloadHeader {
                rows: child_rows,
                dim: self.header.dim,
                padded_dim: self.header.padded_dim,
                centroid_count: self.header.centroid_count,
                code_row_bytes: self.header.code_row_bytes,
            },
            global_ids: child_global_ids.to_vec(),
            centroid_ids: vec![0; child_rows],
            short_data: vec![0; child_rows.saturating_mul(short_words_per_row)],
            ex_data: vec![0; child_rows.saturating_mul(ex_row_bytes)],
            residual_norms: vec![0.0; child_rows],
            scales: vec![0.0; child_rows],
            code_sums: vec![0.0; child_rows],
            original_norms: vec![0.0; child_rows],
            centroids_row_major: self.centroids_row_major.clone(),
            short_words_per_row,
            ex_row_bytes,
            ex_bits,
        };
        let row_groups = self.group_rows_by_shard(child_global_ids)?;
        let mut short_words = vec![0_u32; short_words_per_row];
        let mut ex_bytes = vec![0_u8; ex_row_bytes];
        let mut meta_row = QcodeMetaRow::default();
        for (shard_id, rows) in row_groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let short_path = self.shard_short_path(shard_id);
            let ex_path = self.shard_ex_path(shard_id);
            let meta_path = self.shard_meta_path(shard_id);
            let short_file = File::open(&short_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode split short shard {}: {err}",
                    short_path.display()
                ))
            })?;
            let ex_file = File::open(&ex_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode split ex shard {}: {err}",
                    ex_path.display()
                ))
            })?;
            let meta_file = File::open(&meta_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode shard metadata {}: {err}",
                    meta_path.display()
                ))
            })?;
            for row in rows {
                self.read_split_row_from_files(
                    &short_file,
                    &ex_file,
                    &meta_file,
                    row.global_id,
                    row.row_ref,
                    &mut short_words,
                    &mut ex_bytes,
                    &mut meta_row,
                )?;
                let short_start = row.local_row.checked_mul(short_words_per_row).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Residual RaBitQ qcode split child gather short offset overflows: local_row={} words={short_words_per_row}",
                        row.local_row
                    ))
                })?;
                let ex_start = row.local_row.checked_mul(ex_row_bytes).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Residual RaBitQ qcode split child gather ex offset overflows: local_row={} row_bytes={ex_row_bytes}",
                        row.local_row
                    ))
                })?;
                split.short_data[short_start..short_start + short_words_per_row]
                    .copy_from_slice(&short_words);
                split.ex_data[ex_start..ex_start + ex_row_bytes].copy_from_slice(&ex_bytes);
                split.centroid_ids[row.local_row] = meta_row.centroid_id;
                split.residual_norms[row.local_row] = meta_row.residual_norm;
                split.scales[row.local_row] = meta_row.scale;
                split.code_sums[row.local_row] = meta_row.code_sum;
                split.original_norms[row.local_row] = meta_row.original_norm;
            }
        }
        Ok(split)
    }

    pub(crate) fn load_all_for_full_resident(&self) -> AnnResult<RabitqResidualPayloadStore> {
        if self.header.rows > u32::MAX as usize {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ full-resident load is too large for local u32 ids: rows={}",
                self.header.rows
            )));
        }
        let mut payload = RabitqResidualPayloadStore::new(self.header.clone())?;
        payload.set_centroids_row_major(&self.centroids_row_major)?;
        let dense_ids = (0..self.header.rows)
            .map(|row| row as u32)
            .collect::<Vec<_>>();
        let row_groups = self.group_rows_by_shard(&dense_ids)?;
        let mut shard = RabitqResidualShardPayload {
            global_ids: dense_ids,
            centroid_ids: vec![0; self.header.rows],
            codes: vec![0; self.header.rows.saturating_mul(self.header.code_row_bytes)],
            residual_norms: vec![0.0; self.header.rows],
            scales: vec![0.0; self.header.rows],
            code_sums: vec![0.0; self.header.rows],
            original_norms: vec![0.0; self.header.rows],
            code_row_bytes: self.header.code_row_bytes,
        };
        let mut code_bytes = vec![0u8; self.header.code_row_bytes];
        let mut meta_row = QcodeMetaRow::default();
        for (shard_id, rows) in row_groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let codes_path = self.shard_codes_path(shard_id);
            let meta_path = self.shard_meta_path(shard_id);
            let codes_file = File::open(&codes_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode shard codes {}: {err}",
                    codes_path.display()
                ))
            })?;
            let meta_file = File::open(&meta_path).map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to open residual RaBitQ qcode shard metadata {}: {err}",
                    meta_path.display()
                ))
            })?;
            for row in rows {
                self.read_row_from_files(
                    &codes_file,
                    &meta_file,
                    row.global_id,
                    row.row_ref,
                    &mut code_bytes,
                    &mut meta_row,
                )?;
                let code_start = row
                    .local_row
                    .checked_mul(self.header.code_row_bytes)
                    .ok_or_else(|| {
                        AnnError::log_index_error(format!(
                            "Residual RaBitQ full-resident load code offset overflows: local_row={} row_bytes={}",
                            row.local_row, self.header.code_row_bytes
                        ))
                    })?;
                for (dst, &src) in shard.codes[code_start..code_start + self.header.code_row_bytes]
                    .iter_mut()
                    .zip(code_bytes.iter())
                {
                    *dst = src as i8;
                }
                shard.centroid_ids[row.local_row] = meta_row.centroid_id;
                shard.residual_norms[row.local_row] = meta_row.residual_norm;
                shard.scales[row.local_row] = meta_row.scale;
                shard.code_sums[row.local_row] = meta_row.code_sum;
                shard.original_norms[row.local_row] = meta_row.original_norm;
            }
        }
        payload.write_shard(&shard)?;
        Ok(payload)
    }

    fn group_rows_by_shard(&self, global_ids: &[u32]) -> AnnResult<Vec<Vec<QcodeGatherRow>>> {
        let mut groups = vec![Vec::new(); self.shard_count];
        for (local_row, &global_id) in global_ids.iter().enumerate() {
            let row_ref = self.row_ref(global_id)?;
            let shard_id = row_ref.shard_id as usize;
            if shard_id >= self.shard_count {
                return Err(AnnError::log_index_error(format!(
                    "Residual RaBitQ qcode row global_id={} refers to missing shard {} of {}",
                    global_id, shard_id, self.shard_count
                )));
            }
            groups[shard_id].push(QcodeGatherRow {
                local_row,
                global_id,
                row_ref,
            });
        }
        Ok(groups)
    }

    fn read_row(
        &self,
        global_id: u32,
        code_row: &mut [i8],
        meta_row: &mut QcodeMetaRow,
    ) -> AnnResult<()> {
        if code_row.len() != self.header.code_row_bytes {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode gather code buffer mismatch: got={} expected={}",
                code_row.len(),
                self.header.code_row_bytes
            )));
        }
        let row_ref = self.row_ref(global_id)?;
        let codes_path = self.shard_codes_path(row_ref.shard_id as usize);
        let meta_path = self.shard_meta_path(row_ref.shard_id as usize);
        let codes_file = File::open(&codes_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open residual RaBitQ qcode shard codes {}: {err}",
                codes_path.display()
            ))
        })?;
        let meta_file = File::open(&meta_path).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to open residual RaBitQ qcode shard metadata {}: {err}",
                meta_path.display()
            ))
        })?;
        let mut code_bytes = vec![0u8; self.header.code_row_bytes];
        self.read_row_from_files(
            &codes_file,
            &meta_file,
            global_id,
            row_ref,
            &mut code_bytes,
            meta_row,
        )?;
        for (dst, src) in code_row.iter_mut().zip(code_bytes) {
            *dst = src as i8;
        }
        Ok(())
    }

    fn row_ref(&self, global_id: u32) -> AnnResult<QcodeShardRowRef> {
        let global_id_usize = global_id as usize;
        if global_id_usize >= self.header.rows {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode gather global id {global_id} out of bounds for {} rows",
                self.header.rows
            )));
        }
        self.row_index[global_id_usize].ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Residual RaBitQ qcode gather global id {global_id} has not been written"
            ))
        })
    }

    fn read_row_from_files(
        &self,
        codes_file: &File,
        meta_file: &File,
        global_id: u32,
        row_ref: QcodeShardRowRef,
        code_bytes: &mut [u8],
        meta_row: &mut QcodeMetaRow,
    ) -> AnnResult<()> {
        if code_bytes.len() != self.header.code_row_bytes {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode gather code buffer mismatch: got={} expected={}",
                code_bytes.len(),
                self.header.code_row_bytes
            )));
        }
        codes_file
            .read_exact_at(code_bytes, (row_ref.row_offset as u64).saturating_mul(self.header.code_row_bytes as u64))
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to read residual RaBitQ qcode row global_id={} shard={} offset={}: {err}",
                    global_id, row_ref.shard_id, row_ref.row_offset
                ))
            })?;
        *meta_row = self.read_meta_row_from_file(meta_file, global_id, row_ref)?;
        if meta_row.centroid_id as usize >= self.header.centroid_count {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode row global_id={} centroid id {} out of bounds for {} centroids",
                global_id, meta_row.centroid_id, self.header.centroid_count
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn read_split_row_from_files(
        &self,
        short_file: &File,
        ex_file: &File,
        meta_file: &File,
        global_id: u32,
        row_ref: QcodeShardRowRef,
        short_words: &mut [u32],
        ex_bytes: &mut [u8],
        meta_row: &mut QcodeMetaRow,
    ) -> AnnResult<()> {
        let expected_short_words = rabitq_split_short_words_per_row(self.header.dim);
        if short_words.len() != expected_short_words {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode split gather short buffer mismatch: got={} expected={expected_short_words}",
                short_words.len()
            )));
        }
        let ex_bits = RABITQ_DEFAULT_TOTAL_BITS - 1;
        let expected_ex_bytes = rabitq_split_ex_row_bytes(self.header.dim, ex_bits)?;
        if ex_bytes.len() != expected_ex_bytes {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode split gather ex buffer mismatch: got={} expected={expected_ex_bytes}",
                ex_bytes.len()
            )));
        }
        let mut short_bytes = vec![0_u8; expected_short_words.saturating_mul(4)];
        short_file
            .read_exact_at(
                &mut short_bytes,
                (row_ref.row_offset as u64)
                    .saturating_mul(expected_short_words as u64)
                    .saturating_mul(4),
            )
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to read residual RaBitQ qcode split short row global_id={} shard={} offset={}: {err}",
                    global_id, row_ref.shard_id, row_ref.row_offset
                ))
            })?;
        for (dst, src) in short_words.iter_mut().zip(short_bytes.chunks_exact(4)) {
            *dst = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
        }
        ex_file
            .read_exact_at(
                ex_bytes,
                (row_ref.row_offset as u64).saturating_mul(expected_ex_bytes as u64),
            )
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to read residual RaBitQ qcode split ex row global_id={} shard={} offset={}: {err}",
                    global_id, row_ref.shard_id, row_ref.row_offset
                ))
            })?;
        *meta_row = self.read_meta_row_from_file(meta_file, global_id, row_ref)?;
        if meta_row.centroid_id as usize >= self.header.centroid_count {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode split row global_id={} centroid id {} out of bounds for {} centroids",
                global_id, meta_row.centroid_id, self.header.centroid_count
            )));
        }
        Ok(())
    }

    fn read_meta_row_from_file(
        &self,
        meta_file: &File,
        global_id: u32,
        row_ref: QcodeShardRowRef,
    ) -> AnnResult<QcodeMetaRow> {
        let mut meta_bytes = [0u8; RABITQ_RESIDUAL_QCODE_META_ROW_BYTES];
        meta_file
            .read_exact_at(
                &mut meta_bytes,
                (row_ref.row_offset as u64)
                    .saturating_mul(RABITQ_RESIDUAL_QCODE_META_ROW_BYTES as u64),
            )
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to read residual RaBitQ qcode metadata global_id={} shard={} offset={}: {err}",
                    global_id, row_ref.shard_id, row_ref.row_offset
                ))
            })?;
        parse_qcode_meta_row(&meta_bytes)
    }

    fn shard_codes_path(&self, shard_id: usize) -> PathBuf {
        self.root.join(format!("shard_{shard_id:06}.codes"))
    }

    fn shard_meta_path(&self, shard_id: usize) -> PathBuf {
        self.root.join(format!("shard_{shard_id:06}.meta"))
    }

    fn shard_short_path(&self, shard_id: usize) -> PathBuf {
        self.root.join(format!("shard_{shard_id:06}.short"))
    }

    fn shard_ex_path(&self, shard_id: usize) -> PathBuf {
        self.root.join(format!("shard_{shard_id:06}.ex"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct QcodeMetaRow {
    centroid_id: u32,
    residual_norm: f32,
    scale: f32,
    code_sum: f32,
    original_norm: f32,
}

fn validate_rabitq_residual_shard(
    header: &RabitqResidualPayloadHeader,
    shard: &RabitqResidualShardPayload,
) -> AnnResult<()> {
    let rows = shard.global_ids.len();
    if shard.code_row_bytes != header.code_row_bytes {
        return Err(AnnError::log_index_error(format!(
            "RabitQ residual shard code row bytes mismatch: got={} expected={}",
            shard.code_row_bytes, header.code_row_bytes
        )));
    }
    if shard.centroid_ids.len() != rows
        || shard.residual_norms.len() != rows
        || shard.scales.len() != rows
        || shard.code_sums.len() != rows
        || shard.original_norms.len() != rows
        || shard.codes.len() != rows.saturating_mul(shard.code_row_bytes)
    {
        return Err(AnnError::log_index_error(format!(
            "RabitQ residual shard shape mismatch: ids={} centroids={} codes={} row_bytes={} residual_norms={} scales={} code_sums={} original_norms={}",
            rows,
            shard.centroid_ids.len(),
            shard.codes.len(),
            shard.code_row_bytes,
            shard.residual_norms.len(),
            shard.scales.len(),
            shard.code_sums.len(),
            shard.original_norms.len()
        )));
    }
    for (row, &global_id) in shard.global_ids.iter().enumerate() {
        let dst = global_id as usize;
        if dst >= header.rows {
            return Err(AnnError::log_index_error(format!(
                "RabitQ residual shard global id {global_id} out of bounds for {} rows",
                header.rows
            )));
        }
        let centroid_id = shard.centroid_ids[row] as usize;
        if centroid_id >= header.centroid_count {
            return Err(AnnError::log_index_error(format!(
                "RabitQ residual shard centroid id {centroid_id} out of bounds for {} centroids",
                header.centroid_count
            )));
        }
    }
    Ok(())
}

fn codes_i8_as_u8(codes: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have identical size/alignment; this only preserves the raw qcode bytes on
    // disk.
    unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len()) }
}

fn write_u32_le_slice(file: &mut File, words: &[u32]) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        // SAFETY: u32 has no invalid bit patterns, and little-endian hosts already store the
        // on-disk representation expected by readers using from_le_bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words))
        };
        file.write_all(bytes)
    } else {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(words));
        for &word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        file.write_all(&bytes)
    }
}

fn push_qcode_meta_row(
    dst: &mut Vec<u8>,
    centroid_id: u32,
    residual_norm: f32,
    scale: f32,
    code_sum: f32,
    original_norm: f32,
) {
    dst.extend_from_slice(&centroid_id.to_le_bytes());
    dst.extend_from_slice(&residual_norm.to_le_bytes());
    dst.extend_from_slice(&scale.to_le_bytes());
    dst.extend_from_slice(&code_sum.to_le_bytes());
    dst.extend_from_slice(&original_norm.to_le_bytes());
}

fn parse_qcode_meta_row(
    bytes: &[u8; RABITQ_RESIDUAL_QCODE_META_ROW_BYTES],
) -> AnnResult<QcodeMetaRow> {
    let read_f32 = |start: usize| {
        f32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ])
    };
    let row = QcodeMetaRow {
        centroid_id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        residual_norm: read_f32(4),
        scale: read_f32(8),
        code_sum: read_f32(12),
        original_norm: read_f32(16),
    };
    for (label, value) in [
        ("residual_norm", row.residual_norm),
        ("scale", row.scale),
        ("code_sum", row.code_sum),
        ("original_norm", row.original_norm),
    ] {
        if !value.is_finite() {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ qcode metadata {label} is non-finite: {value}"
            )));
        }
    }
    Ok(row)
}

pub(crate) fn reduce_d0_assignment_fragments(
    child_count: usize,
    point_count: usize,
    fragments: &[D0AssignmentFragment],
) -> AnnResult<Vec<Vec<u32>>> {
    if child_count == 0 {
        return Err(AnnError::log_index_error(
            "D0 assignment fragment reduce requires child_count > 0".to_string(),
        ));
    }
    let mut child_sizes = vec![0usize; child_count];
    for fragment in fragments {
        if fragment.child_id >= child_count {
            return Err(AnnError::log_index_error(format!(
                "D0 assignment fragment child_id {} out of bounds for {child_count} children",
                fragment.child_id
            )));
        }
        if fragment.point_id as usize >= point_count {
            return Err(AnnError::log_index_error(format!(
                "D0 assignment fragment point_id {} out of bounds for {point_count} points",
                fragment.point_id
            )));
        }
        child_sizes[fragment.child_id] += 1;
    }

    let mut children = child_sizes
        .iter()
        .map(|&size| Vec::with_capacity(size))
        .collect::<Vec<_>>();
    for fragment in fragments {
        children[fragment.child_id].push(fragment.point_id);
    }
    Ok(children)
}

pub(crate) fn quantize_rabitq_residual_and_assign_global_d0_reference(
    request: RabitqResidualQuantizeD0Request<'_>,
) -> AnnResult<RabitqResidualQuantizeD0Result> {
    validate_rabitq_residual_quantize_d0_request(request)?;
    let rows = request.global_ids.len();
    let dim = request.dim;
    let leader_count = request.global_d0_leader_ids.len();
    let padded_dim = rabitq_padded_dim(dim, RABITQ_DEFAULT_K_ALIGN)?;
    let rotated_centroids = prepare_rabitq_residual_centroids(
        request.centroids_row_major,
        dim,
        request.rotator,
        request.rotator_seed,
    )?;

    let (residuals, residual_norms, original_norms) = residualize_rows_with_rotator(
        request.raw_vectors_row_major,
        request.centroid_ids,
        request.centroids_row_major,
        &rotated_centroids,
        dim,
        request.rotator,
        request.rotator_seed,
    )?;
    let matrix = quantize_rabitq_matrix(
        &residuals,
        rows,
        dim,
        RABITQ_DEFAULT_TOTAL_BITS,
        RABITQ_DEFAULT_K_ALIGN,
    )?;

    let leader_payload_owned;
    let leader_payload = if let Some(leader_payload) = request.leader_payload {
        validate_rabitq_residual_leader_payload(
            request.global_d0_leader_ids,
            leader_payload,
            request.centroids_row_major.len() / dim,
            padded_dim,
        )?;
        leader_payload
    } else {
        leader_payload_owned = prepare_rabitq_residual_leader_payload(
            request.global_d0_leader_ids,
            request.global_d0_leader_vectors_row_major,
            request.centroids_row_major,
            dim,
            request.rotator,
            request.rotator_seed,
        )?;
        &leader_payload_owned
    };

    let mut fragments = Vec::with_capacity(rows.saturating_mul(request.fanout_top));
    for row in 0..rows {
        let mut candidates = Vec::with_capacity(leader_count);
        for leader_idx in 0..leader_count {
            let leader_code_start = leader_idx.saturating_mul(leader_payload.code_row_bytes);
            let dist = residual_rabitq_full_approx_l2(
                &rotated_centroids,
                dim,
                request.centroid_ids[row] as usize,
                matrix.code_row(row),
                matrix.deltas[row],
                matrix.code_sums[row],
                original_norms[row],
                leader_payload.centroid_ids[leader_idx] as usize,
                &leader_payload.codes
                    [leader_code_start..leader_code_start + leader_payload.code_row_bytes],
                leader_payload.scales[leader_idx],
                leader_payload.code_sums[leader_idx],
                leader_payload.original_norms[leader_idx],
            )?;
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

    Ok(RabitqResidualQuantizeD0Result {
        payload: RabitqResidualShardPayload {
            global_ids: request.global_ids.to_vec(),
            centroid_ids: request.centroid_ids.to_vec(),
            codes: matrix.codes,
            residual_norms,
            scales: matrix.deltas,
            code_sums: matrix.code_sums,
            original_norms,
            code_row_bytes: padded_dim,
        },
        fragments,
        profile: RabitqResidualGpuStagingProfile::default(),
    })
}

pub(crate) fn prepare_rabitq_residual_centroids(
    centroids_row_major: &[f32],
    dim: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    rotator_seed: u64,
) -> AnnResult<Vec<f32>> {
    if dim == 0 || centroids_row_major.is_empty() || centroids_row_major.len() % dim != 0 {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ centroid rotation shape mismatch: values={} dim={}",
            centroids_row_major.len(),
            dim
        )));
    }
    match rotator {
        PiPNNGpuSubtreeRabitqPlusRotator::None => Ok(centroids_row_major.to_vec()),
        PiPNNGpuSubtreeRabitqPlusRotator::FhtKac => {
            validate_fht_kac_qcode_shape(dim)?;
            let centroid_count = centroids_row_major.len() / dim;
            let mut rotated = vec![0.0f32; centroids_row_major.len()];
            for centroid_id in 0..centroid_count {
                let src = &centroids_row_major[centroid_id * dim..(centroid_id + 1) * dim];
                let dst = &mut rotated[centroid_id * dim..(centroid_id + 1) * dim];
                rotate_rabitq_plus_fht_kac_row(src, dst, rotator_seed)?;
            }
            Ok(rotated)
        }
    }
}

pub(crate) fn prepare_rabitq_residual_leader_payload(
    global_d0_leader_ids: &[u32],
    global_d0_leader_vectors_row_major: &[f32],
    centroids_row_major: &[f32],
    dim: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    rotator_seed: u64,
) -> AnnResult<RabitqResidualShardPayload> {
    let leader_count = global_d0_leader_ids.len();
    if leader_count == 0 {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ leader payload requires at least one leader".to_string(),
        ));
    }
    if global_d0_leader_vectors_row_major.len() != leader_count.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ leader payload shape mismatch: leaders={} dim={} values={}",
            leader_count,
            dim,
            global_d0_leader_vectors_row_major.len()
        )));
    }
    let rotated_centroids =
        prepare_rabitq_residual_centroids(centroids_row_major, dim, rotator, rotator_seed)?;
    let centroid_ids = assign_rows_to_nearest_centroids(
        global_d0_leader_vectors_row_major,
        leader_count,
        dim,
        centroids_row_major,
    )?;
    let (leader_residuals, leader_residual_norms, leader_original_norms) =
        residualize_rows_with_rotator(
            global_d0_leader_vectors_row_major,
            &centroid_ids,
            centroids_row_major,
            &rotated_centroids,
            dim,
            rotator,
            rotator_seed,
        )?;
    let matrix = quantize_rabitq_matrix(
        &leader_residuals,
        leader_count,
        dim,
        RABITQ_DEFAULT_TOTAL_BITS,
        RABITQ_DEFAULT_K_ALIGN,
    )?;
    Ok(RabitqResidualShardPayload {
        global_ids: global_d0_leader_ids.to_vec(),
        centroid_ids,
        codes: matrix.codes,
        residual_norms: leader_residual_norms,
        scales: matrix.deltas,
        code_sums: matrix.code_sums,
        original_norms: leader_original_norms,
        code_row_bytes: matrix.padded_dim,
    })
}

fn validate_rabitq_residual_quantize_d0_request(
    request: RabitqResidualQuantizeD0Request<'_>,
) -> AnnResult<()> {
    let rows = request.global_ids.len();
    let dim = request.dim;
    if rows == 0 {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ reference quantize+D0 requires rows > 0".to_string(),
        ));
    }
    if dim == 0 {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ reference quantize+D0 requires dim > 0".to_string(),
        ));
    }
    if request.raw_vectors_row_major.len() != rows.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ raw shard shape mismatch: rows={} dim={} values={}",
            rows,
            dim,
            request.raw_vectors_row_major.len()
        )));
    }
    if request.centroid_ids.len() != rows {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ centroid id count mismatch: rows={} centroid_ids={}",
            rows,
            request.centroid_ids.len()
        )));
    }
    if request.centroids_row_major.is_empty() || request.centroids_row_major.len() % dim != 0 {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ centroid table shape mismatch: values={} dim={}",
            request.centroids_row_major.len(),
            dim
        )));
    }
    let centroid_count = request.centroids_row_major.len() / dim;
    if let Some((row, &centroid_id)) = request
        .centroid_ids
        .iter()
        .enumerate()
        .find(|&(_, &centroid_id)| centroid_id as usize >= centroid_count)
    {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ centroid id {} at row {} out of bounds for {} centroids",
            centroid_id, row, centroid_count
        )));
    }
    let leader_count = request.global_d0_leader_ids.len();
    if leader_count == 0 {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ global-D0 assignment requires at least one leader".to_string(),
        ));
    }
    if request.global_d0_leader_vectors_row_major.len() != leader_count.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ global-D0 leader shape mismatch: leaders={} dim={} values={}",
            leader_count,
            dim,
            request.global_d0_leader_vectors_row_major.len()
        )));
    }
    if request.fanout_top == 0 || request.fanout_top > leader_count || request.fanout_top > 32 {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ global-D0 fanout must be in 1..=min(32, leaders), got fanout={} leaders={}",
            request.fanout_top, leader_count
        )));
    }
    for (label, values) in [
        ("raw shard", request.raw_vectors_row_major),
        ("centroid table", request.centroids_row_major),
        (
            "global D0 leaders",
            request.global_d0_leader_vectors_row_major,
        ),
    ] {
        if let Some((idx, &value)) = values
            .iter()
            .enumerate()
            .find(|&(_, &value)| !value.is_finite())
        {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ {label} contains non-finite value {value} at offset {idx}"
            )));
        }
    }
    Ok(())
}

fn residualize_rows(
    rows_row_major: &[f32],
    centroid_ids: &[u32],
    centroids_row_major: &[f32],
    dim: usize,
) -> AnnResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let rows = centroid_ids.len();
    let centroid_count = centroids_row_major.len() / dim;
    let mut residuals = vec![0.0_f32; rows.saturating_mul(dim)];
    let mut residual_norms = vec![0.0_f32; rows];
    let mut original_norms = vec![0.0_f32; rows];
    for row in 0..rows {
        let centroid_id = centroid_ids[row] as usize;
        if centroid_id >= centroid_count {
            return Err(AnnError::log_index_error(format!(
                "Residual RaBitQ centroid id {} at row {} out of bounds for {} centroids",
                centroid_id, row, centroid_count
            )));
        }
        let src = &rows_row_major[row * dim..(row + 1) * dim];
        let centroid = &centroids_row_major[centroid_id * dim..(centroid_id + 1) * dim];
        let dst = &mut residuals[row * dim..(row + 1) * dim];
        for axis in 0..dim {
            let value = src[axis];
            original_norms[row] += value * value;
            let residual = value - centroid[axis];
            dst[axis] = residual;
            residual_norms[row] += residual * residual;
        }
    }
    Ok((residuals, residual_norms, original_norms))
}

fn residualize_rows_with_rotator(
    rows_row_major: &[f32],
    centroid_ids: &[u32],
    centroids_row_major: &[f32],
    rotated_centroids_row_major: &[f32],
    dim: usize,
    rotator: PiPNNGpuSubtreeRabitqPlusRotator,
    rotator_seed: u64,
) -> AnnResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    match rotator {
        PiPNNGpuSubtreeRabitqPlusRotator::None => {
            residualize_rows(rows_row_major, centroid_ids, centroids_row_major, dim)
        }
        PiPNNGpuSubtreeRabitqPlusRotator::FhtKac => {
            validate_fht_kac_qcode_shape(dim)?;
            let rows = centroid_ids.len();
            if rows_row_major.len() != rows.saturating_mul(dim) {
                return Err(AnnError::log_index_error(format!(
                    "Residual RaBitQ rotated residual raw shape mismatch: rows={} dim={} values={}",
                    rows,
                    dim,
                    rows_row_major.len()
                )));
            }
            if rotated_centroids_row_major.len() != centroids_row_major.len()
                || rotated_centroids_row_major.len() % dim != 0
            {
                return Err(AnnError::log_index_error(format!(
                    "Residual RaBitQ rotated centroid shape mismatch: rotated={} original={} dim={}",
                    rotated_centroids_row_major.len(),
                    centroids_row_major.len(),
                    dim
                )));
            }
            let centroid_count = rotated_centroids_row_major.len() / dim;
            let mut residuals = vec![0.0_f32; rows.saturating_mul(dim)];
            let mut residual_norms = vec![0.0_f32; rows];
            let mut original_norms = vec![0.0_f32; rows];
            let mut rotated_row = vec![0.0_f32; dim];
            for row in 0..rows {
                let centroid_id = centroid_ids[row] as usize;
                if centroid_id >= centroid_count {
                    return Err(AnnError::log_index_error(format!(
                        "Residual RaBitQ rotated residual centroid id {} at row {} out of bounds for {} centroids",
                        centroid_id, row, centroid_count
                    )));
                }
                let src = &rows_row_major[row * dim..(row + 1) * dim];
                rotate_rabitq_plus_fht_kac_row(src, &mut rotated_row, rotator_seed)?;
                let centroid =
                    &rotated_centroids_row_major[centroid_id * dim..(centroid_id + 1) * dim];
                let dst = &mut residuals[row * dim..(row + 1) * dim];
                for axis in 0..dim {
                    let value = src[axis];
                    original_norms[row] += value * value;
                    let residual = rotated_row[axis] - centroid[axis];
                    dst[axis] = residual;
                    residual_norms[row] += residual * residual;
                }
            }
            Ok((residuals, residual_norms, original_norms))
        }
    }
}

fn validate_fht_kac_qcode_shape(dim: usize) -> AnnResult<()> {
    if !dim.is_power_of_two() {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ FHT-Kac qcode mode requires power-of-two dim, got {dim}"
        )));
    }
    let padded_dim = rabitq_padded_dim(dim, RABITQ_DEFAULT_K_ALIGN)?;
    if padded_dim != dim {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ FHT-Kac qcode mode requires padded_dim == dim, got dim={} padded_dim={}",
            dim, padded_dim
        )));
    }
    Ok(())
}

fn validate_rabitq_residual_leader_payload(
    leader_ids: &[u32],
    leader_payload: &RabitqResidualShardPayload,
    centroid_count: usize,
    code_row_bytes: usize,
) -> AnnResult<()> {
    let rows = leader_ids.len();
    if leader_payload.global_ids != leader_ids {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ cached leader payload does not match the global D0 leader set"
                .to_string(),
        ));
    }
    if leader_payload.code_row_bytes != code_row_bytes {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ cached leader payload row bytes mismatch: got={} expected={}",
            leader_payload.code_row_bytes, code_row_bytes
        )));
    }
    if leader_payload.centroid_ids.len() != rows
        || leader_payload.residual_norms.len() != rows
        || leader_payload.scales.len() != rows
        || leader_payload.code_sums.len() != rows
        || leader_payload.original_norms.len() != rows
        || leader_payload.codes.len() != rows.saturating_mul(code_row_bytes)
    {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ cached leader payload shape mismatch: leaders={} centroid_ids={} codes={} row_bytes={} scales={} code_sums={} original_norms={}",
            rows,
            leader_payload.centroid_ids.len(),
            leader_payload.codes.len(),
            leader_payload.code_row_bytes,
            leader_payload.scales.len(),
            leader_payload.code_sums.len(),
            leader_payload.original_norms.len()
        )));
    }
    if let Some((row, &centroid_id)) = leader_payload
        .centroid_ids
        .iter()
        .enumerate()
        .find(|&(_, &centroid_id)| centroid_id as usize >= centroid_count)
    {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ cached leader centroid id {} at row {} out of bounds for {} centroids",
            centroid_id, row, centroid_count
        )));
    }
    Ok(())
}

fn assign_rows_to_nearest_centroids(
    rows_row_major: &[f32],
    rows: usize,
    dim: usize,
    centroids_row_major: &[f32],
) -> AnnResult<Vec<u32>> {
    let centroid_count = centroids_row_major.len() / dim;
    let mut assignments = Vec::with_capacity(rows);
    for row in 0..rows {
        let src = &rows_row_major[row * dim..(row + 1) * dim];
        let mut best = (usize::MAX, f32::INFINITY);
        for centroid_idx in 0..centroid_count {
            let centroid = &centroids_row_major[centroid_idx * dim..(centroid_idx + 1) * dim];
            let dist = src
                .iter()
                .zip(centroid)
                .map(|(&left, &right)| {
                    let delta = left - right;
                    delta * delta
                })
                .sum::<f32>();
            if dist < best.1 || (dist == best.1 && centroid_idx < best.0) {
                best = (centroid_idx, dist);
            }
        }
        if best.0 == usize::MAX {
            return Err(AnnError::log_index_error(
                "Residual RaBitQ nearest-centroid assignment found no centroids".to_string(),
            ));
        }
        assignments.push(best.0 as u32);
    }
    Ok(assignments)
}

#[allow(clippy::too_many_arguments)]
fn residual_rabitq_full_approx_l2(
    centroids_row_major: &[f32],
    dim: usize,
    left_centroid_id: usize,
    left_code: &[i8],
    left_delta: f32,
    left_code_sum: f32,
    left_original_norm: f32,
    right_centroid_id: usize,
    right_code: &[i8],
    right_delta: f32,
    right_code_sum: f32,
    right_original_norm: f32,
) -> AnnResult<f32> {
    let left_centroid = &centroids_row_major[left_centroid_id * dim..(left_centroid_id + 1) * dim];
    let right_centroid =
        &centroids_row_major[right_centroid_id * dim..(right_centroid_id + 1) * dim];
    let centroid_dot = dot_f32(left_centroid, right_centroid);
    let left_residual_dot_right_centroid =
        residual_code_dot_dense(left_code, left_delta, right_centroid, dim)?;
    let right_residual_dot_left_centroid =
        residual_code_dot_dense(right_code, right_delta, left_centroid, dim)?;
    let residual_dot = left_delta
        * right_delta
        * rabitq_dot_from_i8(left_code, left_code_sum, right_code, right_code_sum, dim)?;
    let approx_dot = centroid_dot
        + left_residual_dot_right_centroid
        + right_residual_dot_left_centroid
        + residual_dot;
    let dist = left_original_norm + right_original_norm - 2.0 * approx_dot;
    if !dist.is_finite() {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ approximate distance is non-finite: left_centroid={} right_centroid={} dist={}",
            left_centroid_id, right_centroid_id, dist
        )));
    }
    Ok(if dist <= 0.0 { 0.0 } else { dist })
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RabitqResidualCentroidDotTables {
    pub(crate) rows: usize,
    pub(crate) centroid_count: usize,
    pub(crate) centroid_dot: Vec<f32>,
    pub(crate) residual_centroid_dot: Vec<f32>,
}

pub(crate) fn prepare_rabitq_residual_centroid_dot_tables(
    payload: &RabitqResidualPayloadStore,
) -> AnnResult<RabitqResidualCentroidDotTables> {
    if !payload.is_complete() {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ centroid-dot tables require a complete payload".to_string(),
        ));
    }
    let header = payload.header();
    let rows = header.rows;
    let centroid_count = header.centroid_count;
    let mut centroid_dot = vec![0.0_f32; centroid_count.saturating_mul(centroid_count)];
    for left in 0..centroid_count {
        let left_centroid =
            &payload.centroids_row_major[left * header.dim..(left + 1) * header.dim];
        for right in 0..centroid_count {
            let right_centroid =
                &payload.centroids_row_major[right * header.dim..(right + 1) * header.dim];
            centroid_dot[left * centroid_count + right] = dot_f32(left_centroid, right_centroid);
        }
    }

    let mut residual_centroid_dot = vec![0.0_f32; rows.saturating_mul(centroid_count)];
    for row in 0..rows {
        let code = payload.code_row(row).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Residual RaBitQ centroid-dot table missing code row {row}"
            ))
        })?;
        for centroid_id in 0..centroid_count {
            let centroid = &payload.centroids_row_major
                [centroid_id * header.dim..(centroid_id + 1) * header.dim];
            residual_centroid_dot[row * centroid_count + centroid_id] =
                residual_code_dot_dense(code, payload.scales[row], centroid, header.dim)?;
        }
    }

    Ok(RabitqResidualCentroidDotTables {
        rows,
        centroid_count,
        centroid_dot,
        residual_centroid_dot,
    })
}

pub(crate) fn residual_rabitq_decomposed_l2_with_centroid_dot_tables(
    payload: &RabitqResidualPayloadStore,
    tables: &RabitqResidualCentroidDotTables,
    left_global_id: u32,
    right_global_id: u32,
) -> AnnResult<f32> {
    let header = payload.header();
    if tables.rows != header.rows
        || tables.centroid_count != header.centroid_count
        || tables.centroid_dot.len() != header.centroid_count.saturating_mul(header.centroid_count)
        || tables.residual_centroid_dot.len() != header.rows.saturating_mul(header.centroid_count)
    {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ centroid-dot table shape mismatch: rows={} expected_rows={} centroids={} expected_centroids={} centroid_dot={} residual_centroid_dot={}",
            tables.rows,
            header.rows,
            tables.centroid_count,
            header.centroid_count,
            tables.centroid_dot.len(),
            tables.residual_centroid_dot.len()
        )));
    }
    let left = left_global_id as usize;
    let right = right_global_id as usize;
    if left >= header.rows || right >= header.rows {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ decomposed L2 point id out of bounds: left={} right={} rows={}",
            left_global_id, right_global_id, header.rows
        )));
    }
    if !payload.written[left] || !payload.written[right] {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ decomposed L2 requires written payload rows: left={} written={} right={} written={}",
            left_global_id, payload.written[left], right_global_id, payload.written[right]
        )));
    }
    let left_centroid_id = payload.centroid_ids[left] as usize;
    let right_centroid_id = payload.centroid_ids[right] as usize;
    if left_centroid_id >= header.centroid_count || right_centroid_id >= header.centroid_count {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ decomposed L2 centroid id out of bounds: left={} right={} centroids={}",
            left_centroid_id, right_centroid_id, header.centroid_count
        )));
    }
    let left_code = payload.code_row(left).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ decomposed L2 missing left code row {left_global_id}"
        ))
    })?;
    let right_code = payload.code_row(right).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ decomposed L2 missing right code row {right_global_id}"
        ))
    })?;
    let centroid_count = header.centroid_count;
    let residual_dot = payload.scales[left]
        * payload.scales[right]
        * rabitq_dot_from_i8(
            left_code,
            payload.code_sums[left],
            right_code,
            payload.code_sums[right],
            header.dim,
        )?;
    let approx_dot = tables.centroid_dot[left_centroid_id * centroid_count + right_centroid_id]
        + tables.residual_centroid_dot[left * centroid_count + right_centroid_id]
        + tables.residual_centroid_dot[right * centroid_count + left_centroid_id]
        + residual_dot;
    let dist = payload.original_norms[left] + payload.original_norms[right] - 2.0 * approx_dot;
    if !dist.is_finite() {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ decomposed approximate distance is non-finite: left_centroid={} right_centroid={} dist={}",
            left_centroid_id, right_centroid_id, dist
        )));
    }
    Ok(if dist <= 0.0 { 0.0 } else { dist })
}

pub(crate) fn prepare_rabitq_residual_split_centroid_dot_tables(
    payload: &RabitqResidualSplitPayloadStore,
) -> AnnResult<RabitqResidualCentroidDotTables> {
    let header = payload.header();
    let rows = header.rows;
    let centroid_count = header.centroid_count;
    let mut centroid_dot = vec![0.0_f32; centroid_count.saturating_mul(centroid_count)];
    for left in 0..centroid_count {
        let left_centroid = &payload.centroids_row_major
            [left.saturating_mul(header.dim)..(left + 1).saturating_mul(header.dim)];
        for right in 0..centroid_count {
            let right_centroid = &payload.centroids_row_major
                [right.saturating_mul(header.dim)..(right + 1).saturating_mul(header.dim)];
            centroid_dot[left * centroid_count + right] = dot_f32(left_centroid, right_centroid);
        }
    }

    let mut dense_row = vec![0_i8; header.dim];
    let mut residual_centroid_dot = vec![0.0_f32; rows.saturating_mul(centroid_count)];
    for row in 0..rows {
        payload.decode_dense_row(row, &mut dense_row)?;
        for centroid_id in 0..centroid_count {
            let centroid = &payload.centroids_row_major
                [centroid_id * header.dim..(centroid_id + 1) * header.dim];
            residual_centroid_dot[row * centroid_count + centroid_id] =
                residual_code_dot_dense(&dense_row, payload.scales[row], centroid, header.dim)?;
        }
    }

    Ok(RabitqResidualCentroidDotTables {
        rows,
        centroid_count,
        centroid_dot,
        residual_centroid_dot,
    })
}

pub(crate) fn residual_rabitq_split_decomposed_l2_with_centroid_dot_tables(
    payload: &RabitqResidualSplitPayloadStore,
    tables: &RabitqResidualCentroidDotTables,
    left_row: usize,
    right_row: usize,
) -> AnnResult<f32> {
    let header = payload.header();
    if tables.rows != header.rows
        || tables.centroid_count != header.centroid_count
        || tables.centroid_dot.len() != header.centroid_count.saturating_mul(header.centroid_count)
        || tables.residual_centroid_dot.len() != header.rows.saturating_mul(header.centroid_count)
    {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split centroid-dot table shape mismatch: rows={} expected_rows={} centroids={} expected_centroids={} centroid_dot={} residual_centroid_dot={}",
            tables.rows,
            header.rows,
            tables.centroid_count,
            header.centroid_count,
            tables.centroid_dot.len(),
            tables.residual_centroid_dot.len()
        )));
    }
    if left_row >= header.rows || right_row >= header.rows {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed L2 row out of bounds: left={left_row} right={right_row} rows={}",
            header.rows
        )));
    }
    let left_centroid_id = payload.centroid_ids[left_row] as usize;
    let right_centroid_id = payload.centroid_ids[right_row] as usize;
    if left_centroid_id >= header.centroid_count || right_centroid_id >= header.centroid_count {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed L2 centroid id out of bounds: left={} right={} centroids={}",
            left_centroid_id, right_centroid_id, header.centroid_count
        )));
    }
    let words = payload.short_words_per_row;
    let ex_row_bytes = payload.ex_row_bytes;
    let left_short_start = left_row.checked_mul(words).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed L2 left short offset overflows: row={left_row} words={words}"
        ))
    })?;
    let right_short_start = right_row.checked_mul(words).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed L2 right short offset overflows: row={right_row} words={words}"
        ))
    })?;
    let left_ex_start = left_row.checked_mul(ex_row_bytes).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed L2 left ex offset overflows: row={left_row} row_bytes={ex_row_bytes}"
        ))
    })?;
    let right_ex_start = right_row.checked_mul(ex_row_bytes).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed L2 right ex offset overflows: row={right_row} row_bytes={ex_row_bytes}"
        ))
    })?;
    let centroid_count = header.centroid_count;
    let residual_dot = payload.scales[left_row]
        * payload.scales[right_row]
        * rabitq_dot_from_split(
            &payload.short_data[left_short_start..left_short_start + words],
            &payload.ex_data[left_ex_start..left_ex_start + ex_row_bytes],
            payload.code_sums[left_row],
            &payload.short_data[right_short_start..right_short_start + words],
            &payload.ex_data[right_ex_start..right_ex_start + ex_row_bytes],
            payload.code_sums[right_row],
            header.dim,
            payload.ex_bits,
        )?;
    let approx_dot = tables.centroid_dot[left_centroid_id * centroid_count + right_centroid_id]
        + tables.residual_centroid_dot[left_row * centroid_count + right_centroid_id]
        + tables.residual_centroid_dot[right_row * centroid_count + left_centroid_id]
        + residual_dot;
    let dist =
        payload.original_norms[left_row] + payload.original_norms[right_row] - 2.0 * approx_dot;
    if !dist.is_finite() {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split decomposed approximate distance is non-finite: left_centroid={} right_centroid={} dist={}",
            left_centroid_id, right_centroid_id, dist
        )));
    }
    Ok(if dist <= 0.0 { 0.0 } else { dist })
}

pub(crate) fn residual_rabitq_reconstructed_l2(
    payload: &RabitqResidualPayloadStore,
    left_global_id: u32,
    right_global_id: u32,
) -> AnnResult<f32> {
    let header = payload.header();
    let left = left_global_id as usize;
    let right = right_global_id as usize;
    if left >= header.rows || right >= header.rows {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ reconstructed L2 point id out of bounds: left={} right={} rows={}",
            left_global_id, right_global_id, header.rows
        )));
    }
    if !payload.written[left] || !payload.written[right] {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ reconstructed L2 requires written payload rows: left={} written={} right={} written={}",
            left_global_id, payload.written[left], right_global_id, payload.written[right]
        )));
    }
    let left_centroid_id = payload.centroid_ids[left] as usize;
    let right_centroid_id = payload.centroid_ids[right] as usize;
    if left_centroid_id >= header.centroid_count || right_centroid_id >= header.centroid_count {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ reconstructed L2 centroid id out of bounds: left={} right={} centroids={}",
            left_centroid_id, right_centroid_id, header.centroid_count
        )));
    }
    let left_code = payload.code_row(left).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ reconstructed L2 missing left code row {left_global_id}"
        ))
    })?;
    let right_code = payload.code_row(right).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ reconstructed L2 missing right code row {right_global_id}"
        ))
    })?;
    residual_rabitq_full_approx_l2(
        payload.centroids_row_major(),
        header.dim,
        left_centroid_id,
        left_code,
        payload.scales[left],
        payload.code_sums[left],
        payload.original_norms[left],
        right_centroid_id,
        right_code,
        payload.scales[right],
        payload.code_sums[right],
        payload.original_norms[right],
    )
}

pub(crate) fn resident_rabitq_child_topk_reference(
    payload: &RabitqResidualPayloadStore,
    child_global_ids: &[u32],
    requested_k: usize,
) -> AnnResult<(Vec<u32>, Vec<f32>)> {
    if !payload.is_complete() {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ child top-k reference requires a complete payload".to_string(),
        ));
    }
    let rows = child_global_ids.len();
    let k = requested_k.min(rows.saturating_sub(1));
    if rows == 0 || k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if k > 32 {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ child top-k reference supports k <= 32, got {k}"
        )));
    }
    let mut topk_ids = vec![u32::MAX; rows.saturating_mul(k)];
    let mut topk_dists = vec![f32::INFINITY; rows.saturating_mul(k)];
    for (row, &query_gid) in child_global_ids.iter().enumerate() {
        let mut candidates = Vec::with_capacity(rows.saturating_sub(1));
        for &candidate_gid in child_global_ids {
            if candidate_gid == query_gid {
                continue;
            }
            let dist = residual_rabitq_reconstructed_l2(payload, query_gid, candidate_gid)?;
            candidates.push((candidate_gid, dist));
        }
        candidates.sort_unstable_by(|left, right| {
            super::stable_distance_order(left.1, left.0, right.1, right.0)
        });
        for (slot, &(candidate_gid, dist)) in candidates.iter().take(k).enumerate() {
            let dst = row * k + slot;
            topk_ids[dst] = candidate_gid;
            topk_dists[dst] = dist;
        }
    }
    Ok((topk_ids, topk_dists))
}

pub(crate) fn resident_rabitq_split_short_ex_topk_reference(
    payload: &RabitqResidualSplitPayloadStore,
    child_global_ids: &[u32],
    requested_k: usize,
    short_survivors: usize,
) -> AnnResult<(Vec<u32>, Vec<f32>)> {
    let rows = child_global_ids.len();
    if rows != payload.header.rows {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split reference rows {} mismatch child ids {}",
            payload.header.rows, rows
        )));
    }
    if payload.global_ids() != child_global_ids {
        return Err(AnnError::log_index_error(
            "Residual RaBitQ split reference requires child ids to match payload global ids"
                .to_string(),
        ));
    }
    let k = requested_k.min(rows.saturating_sub(1));
    if rows == 0 || k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if short_survivors < k || short_survivors > 128 {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split short survivors must be in k..=128, got {short_survivors} for k={k}"
        )));
    }
    let survivor_limit = short_survivors.min(rows.saturating_sub(1));
    let tables = prepare_rabitq_residual_split_centroid_dot_tables(payload)?;
    let mut topk_ids = vec![u32::MAX; rows.saturating_mul(k)];
    let mut topk_dists = vec![f32::INFINITY; rows.saturating_mul(k)];
    for row in 0..rows {
        let mut survivors = Vec::with_capacity(rows.saturating_sub(1));
        for candidate in 0..rows {
            if row == candidate {
                continue;
            }
            let score = split_short_similarity(payload, row, candidate)?;
            survivors.push((candidate, score));
        }
        survivors.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        survivors.truncate(survivor_limit);

        let mut refined = Vec::with_capacity(survivor_limit);
        for &(candidate, _) in &survivors {
            let dist = residual_rabitq_split_decomposed_l2_with_centroid_dot_tables(
                payload, &tables, row, candidate,
            )?;
            refined.push((child_global_ids[candidate], dist));
        }
        refined.sort_unstable_by(|left, right| {
            super::stable_distance_order(left.1, left.0, right.1, right.0)
        });
        for (slot, &(candidate_gid, dist)) in refined.iter().take(k).enumerate() {
            let dst = row * k + slot;
            topk_ids[dst] = candidate_gid;
            topk_dists[dst] = dist;
        }
    }
    Ok((topk_ids, topk_dists))
}

fn residual_code_dot_dense(code: &[i8], delta: f32, dense: &[f32], dim: usize) -> AnnResult<f32> {
    if dim > code.len() || dim > dense.len() {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ code/dense dot shape mismatch: dim={} code={} dense={}",
            dim,
            code.len(),
            dense.len()
        )));
    }
    if delta == 0.0 {
        return Ok(0.0);
    }
    Ok(code
        .iter()
        .zip(dense)
        .take(dim)
        .map(|(&code, &value)| delta * (code as f32 + 0.5) * value)
        .sum())
}

fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(&l, &r)| l * r).sum()
}

fn split_short_similarity(
    payload: &RabitqResidualSplitPayloadStore,
    left: usize,
    right: usize,
) -> AnnResult<i32> {
    let header = payload.header();
    if left >= header.rows || right >= header.rows {
        return Err(AnnError::log_index_error(format!(
            "Residual RaBitQ split short similarity row out of bounds: left={left} right={right} rows={}",
            header.rows
        )));
    }
    let words = payload.short_words_per_row;
    let left_base = left.checked_mul(words).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split short similarity left offset overflows: row={left} words={words}"
        ))
    })?;
    let right_base = right.checked_mul(words).ok_or_else(|| {
        AnnError::log_index_error(format!(
            "Residual RaBitQ split short similarity right offset overflows: row={right} words={words}"
        ))
    })?;
    let mut same = 0_u32;
    for word in 0..words {
        let mut xnor =
            !(payload.short_data[left_base + word] ^ payload.short_data[right_base + word]);
        if word + 1 == words {
            let used = header.dim % 32;
            if used != 0 {
                xnor &= (1_u32 << used) - 1;
            }
        }
        same += xnor.count_ones();
    }
    Ok((2 * same as i32) - header.dim as i32)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn rabitq_residual_payload_store_accepts_out_of_order_shard_writes() {
        let header = RabitqResidualPayloadHeader {
            rows: 6,
            dim: 4,
            padded_dim: 8,
            centroid_count: 2,
            code_row_bytes: 8,
        };
        let mut store = RabitqResidualPayloadStore::new(header.clone()).unwrap();
        let shard = RabitqResidualShardPayload {
            global_ids: vec![4, 1, 5],
            centroid_ids: vec![1, 0, 1],
            codes: vec![
                40, 41, 42, 43, 44, 45, 46, 47, 10, 11, 12, 13, 14, 15, 16, 17, 50, 51, 52, 53, 54,
                55, 56, 57,
            ],
            residual_norms: vec![4.0, 1.0, 5.0],
            scales: vec![0.4, 0.1, 0.5],
            code_sums: vec![400.0, 100.0, 500.0],
            original_norms: vec![16.0, 1.0, 25.0],
            code_row_bytes: 8,
        };

        store.write_shard(&shard).unwrap();

        assert!(!store.is_complete());
        assert_eq!(store.centroid_id(1), Some(0));
        assert_eq!(store.centroid_id(4), Some(1));
        assert_eq!(
            store.code_row(5).unwrap(),
            &[50, 51, 52, 53, 54, 55, 56, 57]
        );
        assert_eq!(store.residual_norm(4), Some(4.0));
        assert_eq!(store.original_norm(5), Some(25.0));
    }

    #[test]
    fn reduce_d0_assignment_fragments_groups_by_global_child() {
        let fragments = vec![
            D0AssignmentFragment {
                child_id: 2,
                point_id: 10,
            },
            D0AssignmentFragment {
                child_id: 0,
                point_id: 10,
            },
            D0AssignmentFragment {
                child_id: 2,
                point_id: 7,
            },
            D0AssignmentFragment {
                child_id: 1,
                point_id: 9,
            },
        ];

        let children = reduce_d0_assignment_fragments(3, 11, &fragments).unwrap();

        assert_eq!(children.len(), 3);
        assert_eq!(children[0], vec![10]);
        assert_eq!(children[1], vec![9]);
        assert_eq!(children[2], vec![10, 7]);
    }

    #[test]
    fn reduce_d0_assignment_fragments_rejects_out_of_bounds_child() {
        let fragments = vec![D0AssignmentFragment {
            child_id: 3,
            point_id: 0,
        }];

        let err = reduce_d0_assignment_fragments(3, 1, &fragments).unwrap_err();

        assert!(format!("{err:?}").contains("child_id 3 out of bounds"));
    }

    #[test]
    fn rabitq_residual_payload_store_exposes_contiguous_global_arrays() {
        let header = RabitqResidualPayloadHeader {
            rows: 3,
            dim: 2,
            padded_dim: 4,
            centroid_count: 2,
            code_row_bytes: 4,
        };
        let mut store = RabitqResidualPayloadStore::new(header).unwrap();
        store
            .set_centroids_row_major(&[0.0, 1.0, 2.0, 3.0])
            .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![2, 0, 1],
                centroid_ids: vec![1, 0, 1],
                codes: vec![20, 21, 22, 23, 0, 1, 2, 3, 10, 11, 12, 13],
                residual_norms: vec![2.0, 0.0, 1.0],
                scales: vec![0.2, 1.0, 0.1],
                code_sums: vec![200.0, 0.0, 100.0],
                original_norms: vec![4.0, 0.0, 1.0],
                code_row_bytes: 4,
            })
            .unwrap();

        assert_eq!(store.centroid_ids(), &[0, 1, 1]);
        assert_eq!(store.codes(), &[0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]);
        assert_eq!(store.residual_norms(), &[0.0, 1.0, 2.0]);
        assert_eq!(store.scales(), &[1.0, 0.1, 0.2]);
        assert_eq!(store.code_sums(), &[0.0, 100.0, 200.0]);
        assert_eq!(store.original_norms(), &[0.0, 1.0, 4.0]);
        assert_eq!(store.centroids_row_major(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(
            store.device_payload_bytes(),
            3 * 4 * std::mem::size_of::<i8>()
                + 3 * std::mem::size_of::<u32>()
                + 2 * 3 * std::mem::size_of::<f32>()
                + 4 * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn rabitq_residual_reference_quantizes_shard_and_assigns_global_d0() {
        let raw = vec![
            0.1_f32, 0.0, 0.0, 0.0, //
            99.9, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 0.0, //
            96.0, 0.0, 0.0, 0.0,
        ];
        let centroids = vec![
            0.0_f32, 0.0, 0.0, 0.0, //
            100.0, 0.0, 0.0, 0.0,
        ];
        let leaders = vec![
            0.0_f32, 0.0, 0.0, 0.0, //
            100.0, 0.0, 0.0, 0.0,
        ];

        let result = quantize_rabitq_residual_and_assign_global_d0_reference(
            RabitqResidualQuantizeD0Request {
                raw_vectors_row_major: &raw,
                global_ids: &[0, 1, 2, 3],
                centroid_ids: &[0, 1, 0, 1],
                centroids_row_major: &centroids,
                global_d0_leader_ids: &[10, 20],
                global_d0_leader_vectors_row_major: &leaders,
                leader_payload: None,
                dim: 4,
                fanout_top: 1,
                rotator: PiPNNGpuSubtreeRabitqPlusRotator::None,
                rotator_seed: 42,
            },
        )
        .unwrap();

        assert_eq!(result.payload.global_ids, vec![0, 1, 2, 3]);
        assert_eq!(result.payload.centroid_ids, vec![0, 1, 0, 1]);
        assert_eq!(result.payload.code_row_bytes, 16);
        assert_eq!(result.payload.codes.len(), 4 * 16);
        assert_eq!(result.payload.residual_norms.len(), 4);
        assert_eq!(result.payload.scales.len(), 4);
        assert_eq!(result.payload.code_sums.len(), 4);
        assert_eq!(result.payload.original_norms.len(), 4);
        assert_eq!(
            result.fragments,
            vec![
                D0AssignmentFragment {
                    child_id: 0,
                    point_id: 0,
                },
                D0AssignmentFragment {
                    child_id: 1,
                    point_id: 1,
                },
                D0AssignmentFragment {
                    child_id: 0,
                    point_id: 2,
                },
                D0AssignmentFragment {
                    child_id: 1,
                    point_id: 3,
                },
            ]
        );
        assert_eq!(result.profile.gpu_quantize_wall, Duration::ZERO);
        assert_eq!(result.profile.gpu_d0_assign_wall, Duration::ZERO);
    }

    #[test]
    fn resident_rabitq_child_topk_reference_orders_by_reconstructed_l2() {
        let header = RabitqResidualPayloadHeader {
            rows: 4,
            dim: 2,
            padded_dim: 4,
            centroid_count: 2,
            code_row_bytes: 4,
        };
        let mut store = RabitqResidualPayloadStore::new(header).unwrap();
        store
            .set_centroids_row_major(&[
                0.0, 0.0, //
                10.0, 0.0,
            ])
            .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![0, 1, 2, 3],
                centroid_ids: vec![0, 0, 0, 1],
                codes: vec![
                    0, 0, 0, 0, //
                    1, 0, 0, 0, //
                    4, 0, 0, 0, //
                    0, 0, 0, 0,
                ],
                residual_norms: vec![0.5, 2.5, 20.5, 0.5],
                scales: vec![1.0, 1.0, 1.0, 1.0],
                code_sums: vec![0.0, 1.0, 4.0, 0.0],
                original_norms: vec![0.5, 2.5, 20.5, 110.5],
                code_row_bytes: 4,
            })
            .unwrap();

        let (ids, dists) = resident_rabitq_child_topk_reference(&store, &[0, 1, 2, 3], 2).unwrap();

        assert_eq!(ids, vec![1, 2, 0, 2, 1, 0, 2, 1]);
        let expected = [1.0, 16.0, 1.0, 9.0, 9.0, 16.0, 36.0, 81.0];
        for (actual, expected) in dists.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-4,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn resident_rabitq_split_short_ex_topk_matches_dense_reference_when_survivors_cover_all_rows() {
        let header = RabitqResidualPayloadHeader {
            rows: 4,
            dim: 4,
            padded_dim: 4,
            centroid_count: 1,
            code_row_bytes: 4,
        };
        let mut dense = RabitqResidualPayloadStore::new(header).unwrap();
        dense
            .set_centroids_row_major(&[0.0, 0.0, 0.0, 0.0])
            .unwrap();
        dense
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![0, 1, 2, 3],
                centroid_ids: vec![0, 0, 0, 0],
                codes: vec![
                    0, 1, 2, 3, //
                    1, 2, 3, 4, //
                    8, 7, 6, 5, //
                    -1, -2, -3, -4,
                ],
                residual_norms: vec![1.0, 2.0, 3.0, 4.0],
                scales: vec![1.0, 1.0, 1.0, 1.0],
                code_sums: vec![6.0, 10.0, 26.0, -10.0],
                original_norms: vec![1.0, 2.0, 3.0, 4.0],
                code_row_bytes: 4,
            })
            .unwrap();
        let split = RabitqResidualSplitPayloadStore::from_dense_payload(&dense).unwrap();

        let (dense_ids, dense_dists) =
            resident_rabitq_child_topk_reference(&dense, &[0, 1, 2, 3], 2).unwrap();
        let (split_ids, split_dists) =
            resident_rabitq_split_short_ex_topk_reference(&split, &[0, 1, 2, 3], 2, 3).unwrap();

        assert_eq!(split_ids, dense_ids);
        for (left, right) in split_dists.iter().zip(dense_dists.iter()) {
            assert!((left - right).abs() <= 1.0e-4, "split={left} dense={right}");
        }
    }

    #[test]
    fn qcode_shard_store_gathers_child_rows_in_requested_order() {
        let dir = tempdir().unwrap();
        let header = RabitqResidualPayloadHeader {
            rows: 6,
            dim: 4,
            padded_dim: 4,
            centroid_count: 2,
            code_row_bytes: 4,
        };
        let centroids = vec![
            0.0, 0.0, 0.0, 0.0, //
            10.0, 0.0, 0.0, 0.0,
        ];
        let mut store =
            RabitqResidualQcodeShardStore::create(dir.path(), header.clone(), &centroids).unwrap();

        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![4, 1, 5],
                centroid_ids: vec![1, 0, 1],
                codes: vec![
                    40, 41, 42, 43, //
                    10, 11, 12, 13, //
                    50, 51, 52, 53,
                ],
                residual_norms: vec![4.0, 1.0, 5.0],
                scales: vec![0.4, 0.1, 0.5],
                code_sums: vec![400.0, 100.0, 500.0],
                original_norms: vec![16.0, 1.0, 25.0],
                code_row_bytes: 4,
            })
            .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![0, 2, 3],
                centroid_ids: vec![0, 0, 1],
                codes: vec![
                    0, 1, 2, 3, //
                    20, 21, 22, 23, //
                    30, 31, 32, 33,
                ],
                residual_norms: vec![0.0, 2.0, 3.0],
                scales: vec![1.0, 0.2, 0.3],
                code_sums: vec![0.0, 200.0, 300.0],
                original_norms: vec![0.0, 4.0, 9.0],
                code_row_bytes: 4,
            })
            .unwrap();

        let child = store.gather_child(&[5, 0, 4, 2]).unwrap();

        assert_eq!(child.header().rows, 4);
        assert!(child.is_complete());
        assert_eq!(child.centroid_ids(), &[1, 0, 1, 0]);
        assert_eq!(
            child.codes(),
            &[
                50, 51, 52, 53, //
                0, 1, 2, 3, //
                40, 41, 42, 43, //
                20, 21, 22, 23,
            ]
        );
        assert_eq!(child.scales(), &[0.5, 1.0, 0.4, 0.2]);
        assert_eq!(child.original_norms(), &[25.0, 0.0, 16.0, 4.0]);
        assert_eq!(child.centroids_row_major(), centroids);
    }

    #[test]
    fn qcode_shard_store_gathers_split_child_rows_in_requested_order() {
        let dir = tempdir().unwrap();
        let header = RabitqResidualPayloadHeader {
            rows: 4,
            dim: 4,
            padded_dim: 4,
            centroid_count: 1,
            code_row_bytes: 4,
        };
        let mut store =
            RabitqResidualQcodeShardStore::create(dir.path(), header, &[0.0, 0.0, 0.0, 0.0])
                .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![2, 0, 3, 1],
                centroid_ids: vec![0, 0, 0, 0],
                codes: vec![
                    -128, -1, 0, 127, //
                    0, 1, 2, 3, //
                    4, 5, 6, 7, //
                    -7, -6, -5, -4,
                ],
                residual_norms: vec![2.0, 0.0, 3.0, 1.0],
                scales: vec![0.2, 1.0, 0.3, 0.1],
                code_sums: vec![-2.0, 6.0, 22.0, -22.0],
                original_norms: vec![4.0, 0.0, 9.0, 1.0],
                code_row_bytes: 4,
            })
            .unwrap();

        let split = store.gather_child_split(&[1, 3, 2]).unwrap();
        assert_eq!(split.header().rows, 3);
        assert_eq!(split.global_ids(), &[1, 3, 2]);
        assert_eq!(split.centroid_ids(), &[0, 0, 0]);
        assert_eq!(split.scales(), &[0.1, 0.3, 0.2]);

        let mut decoded = vec![0_i8; 4];
        split.decode_dense_row(0, &mut decoded).unwrap();
        assert_eq!(decoded, vec![-7, -6, -5, -4]);
        split.decode_dense_row(1, &mut decoded).unwrap();
        assert_eq!(decoded, vec![4, 5, 6, 7]);
        split.decode_dense_row(2, &mut decoded).unwrap();
        assert_eq!(decoded, vec![-128, -1, 0, 127]);
    }

    #[test]
    fn qcode_shard_store_gathers_split_child_without_dense_codes_file() {
        let dir = tempdir().unwrap();
        let header = RabitqResidualPayloadHeader {
            rows: 3,
            dim: 4,
            padded_dim: 4,
            centroid_count: 1,
            code_row_bytes: 4,
        };
        let mut store =
            RabitqResidualQcodeShardStore::create(dir.path(), header, &[0.0, 0.0, 0.0, 0.0])
                .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![2, 0, 1],
                centroid_ids: vec![0, 0, 0],
                codes: vec![
                    20, 21, 22, 23, //
                    0, 1, 2, 3, //
                    -4, -3, -2, -1,
                ],
                residual_norms: vec![2.0, 0.0, 1.0],
                scales: vec![0.2, 1.0, 0.1],
                code_sums: vec![200.0, 0.0, -10.0],
                original_norms: vec![4.0, 0.0, 1.0],
                code_row_bytes: 4,
            })
            .unwrap();
        fs::remove_file(dir.path().join("shard_000000.codes")).unwrap();

        let split = store.gather_child_split(&[1, 2]).unwrap();
        let mut decoded = vec![0_i8; 4];
        split.decode_dense_row(0, &mut decoded).unwrap();
        assert_eq!(decoded, vec![-4, -3, -2, -1]);
        split.decode_dense_row(1, &mut decoded).unwrap();
        assert_eq!(decoded, vec![20, 21, 22, 23]);
    }

    #[test]
    fn qcode_shard_store_split_only_write_skips_dense_codes_file() {
        let dir = tempdir().unwrap();
        let header = RabitqResidualPayloadHeader {
            rows: 3,
            dim: 4,
            padded_dim: 4,
            centroid_count: 1,
            code_row_bytes: 4,
        };
        let mut store = RabitqResidualQcodeShardStore::create_split_only(
            dir.path(),
            header,
            &[0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![2, 0, 1],
                centroid_ids: vec![0, 0, 0],
                codes: vec![
                    20, 21, 22, 23, //
                    0, 1, 2, 3, //
                    -4, -3, -2, -1,
                ],
                residual_norms: vec![2.0, 0.0, 1.0],
                scales: vec![0.2, 1.0, 0.1],
                code_sums: vec![200.0, 0.0, -10.0],
                original_norms: vec![4.0, 0.0, 1.0],
                code_row_bytes: 4,
            })
            .unwrap();

        assert!(!dir.path().join("shard_000000.codes").exists());
        assert!(dir.path().join("shard_000000.short").is_file());
        assert!(dir.path().join("shard_000000.ex").is_file());
        assert!(dir.path().join("shard_000000.meta").is_file());

        let split = store.gather_child_split(&[1, 2]).unwrap();
        let mut decoded = vec![0_i8; 4];
        split.decode_dense_row(0, &mut decoded).unwrap();
        assert_eq!(decoded, vec![-4, -3, -2, -1]);
        split.decode_dense_row(1, &mut decoded).unwrap();
        assert_eq!(decoded, vec![20, 21, 22, 23]);
    }

    #[test]
    fn qcode_shard_store_writes_split_files_next_to_dense_files() {
        let dir = tempdir().unwrap();
        let header = RabitqResidualPayloadHeader {
            rows: 2,
            dim: 4,
            padded_dim: 4,
            centroid_count: 1,
            code_row_bytes: 4,
        };
        let mut store =
            RabitqResidualQcodeShardStore::create(dir.path(), header, &[0.0, 0.0, 0.0, 0.0])
                .unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![0, 1],
                centroid_ids: vec![0, 0],
                codes: vec![1, 2, 3, 4, -4, -3, -2, -1],
                residual_norms: vec![1.0, 2.0],
                scales: vec![0.1, 0.2],
                code_sums: vec![10.0, -10.0],
                original_norms: vec![1.0, 4.0],
                code_row_bytes: 4,
            })
            .unwrap();

        assert!(dir.path().join("shard_000000.short").is_file());
        assert!(dir.path().join("shard_000000.ex").is_file());
        assert!(dir.path().join("shard_000000.meta").is_file());
        assert!(dir.path().join("shard_000000.codes").is_file());
    }

    #[test]
    fn qcode_shard_store_load_all_reconstructs_dense_payload() {
        let dir = tempdir().unwrap();
        let header = RabitqResidualPayloadHeader {
            rows: 3,
            dim: 2,
            padded_dim: 4,
            centroid_count: 1,
            code_row_bytes: 4,
        };
        let mut store =
            RabitqResidualQcodeShardStore::create(dir.path(), header, &[1.0, 2.0]).unwrap();
        store
            .write_shard(&RabitqResidualShardPayload {
                global_ids: vec![2, 0, 1],
                centroid_ids: vec![0, 0, 0],
                codes: vec![20, 21, 22, 23, 0, 1, 2, 3, 10, 11, 12, 13],
                residual_norms: vec![2.0, 0.0, 1.0],
                scales: vec![0.2, 1.0, 0.1],
                code_sums: vec![200.0, 0.0, 100.0],
                original_norms: vec![4.0, 0.0, 1.0],
                code_row_bytes: 4,
            })
            .unwrap();

        let full = store.load_all_for_full_resident().unwrap();

        assert!(full.is_complete());
        assert_eq!(full.codes(), &[0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]);
        assert_eq!(full.scales(), &[1.0, 0.1, 0.2]);
        assert_eq!(full.code_sums(), &[0.0, 100.0, 200.0]);
    }

    #[test]
    fn residual_rabitq_centroid_dot_tables_match_reference_distance() {
        let raw = vec![
            0.1_f32, 0.2, -0.3, 0.4, //
            1.4, -0.2, 0.5, 0.1, //
            -0.7, 0.9, 0.3, -0.4,
        ];
        let centroids = vec![
            0.0_f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ];
        let staged = quantize_rabitq_residual_and_assign_global_d0_reference(
            RabitqResidualQuantizeD0Request {
                raw_vectors_row_major: &raw,
                global_ids: &[0, 1, 2],
                centroid_ids: &[0, 1, 0],
                centroids_row_major: &centroids,
                global_d0_leader_ids: &[10, 20],
                global_d0_leader_vectors_row_major: &[
                    0.0_f32, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0,
                ],
                leader_payload: None,
                dim: 4,
                fanout_top: 1,
                rotator: PiPNNGpuSubtreeRabitqPlusRotator::None,
                rotator_seed: 7,
            },
        )
        .unwrap();
        let mut payload = RabitqResidualPayloadStore::new(RabitqResidualPayloadHeader {
            rows: 3,
            dim: 4,
            padded_dim: staged.payload.code_row_bytes,
            centroid_count: 2,
            code_row_bytes: staged.payload.code_row_bytes,
        })
        .unwrap();
        payload.set_centroids_row_major(&centroids).unwrap();
        payload.write_shard(&staged.payload).unwrap();

        let tables = prepare_rabitq_residual_centroid_dot_tables(&payload).unwrap();
        for left in 0..3 {
            for right in 0..3 {
                let reference =
                    residual_rabitq_reconstructed_l2(&payload, left as u32, right as u32).unwrap();
                let decomposed = residual_rabitq_decomposed_l2_with_centroid_dot_tables(
                    &payload,
                    &tables,
                    left as u32,
                    right as u32,
                )
                .unwrap();
                assert!(
                    (reference - decomposed).abs() <= 1.0e-4,
                    "left={left} right={right} reference={reference} decomposed={decomposed}"
                );
            }
        }
    }

    #[test]
    fn fht_kac_residual_quantization_rotates_raw_and_centroids() {
        let mut raw = vec![0.0_f32; 32];
        raw[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        raw[16..20].copy_from_slice(&[4.0, 3.0, 2.0, 1.0]);
        let mut centroids = vec![0.0_f32; 32];
        centroids[0] = 1.0;
        centroids[17] = 1.0;
        let leaders = raw.clone();
        let seed = 12345;

        let rotated = quantize_rabitq_residual_and_assign_global_d0_reference(
            RabitqResidualQuantizeD0Request {
                raw_vectors_row_major: &raw,
                global_ids: &[0, 1],
                centroid_ids: &[0, 1],
                centroids_row_major: &centroids,
                global_d0_leader_ids: &[0, 1],
                global_d0_leader_vectors_row_major: &leaders,
                leader_payload: None,
                dim: 16,
                fanout_top: 1,
                rotator: PiPNNGpuSubtreeRabitqPlusRotator::FhtKac,
                rotator_seed: seed,
            },
        )
        .unwrap();
        let unrotated = quantize_rabitq_residual_and_assign_global_d0_reference(
            RabitqResidualQuantizeD0Request {
                raw_vectors_row_major: &raw,
                global_ids: &[0, 1],
                centroid_ids: &[0, 1],
                centroids_row_major: &centroids,
                global_d0_leader_ids: &[0, 1],
                global_d0_leader_vectors_row_major: &leaders,
                leader_payload: None,
                dim: 16,
                fanout_top: 1,
                rotator: PiPNNGpuSubtreeRabitqPlusRotator::None,
                rotator_seed: seed,
            },
        )
        .unwrap();

        assert_ne!(rotated.payload.codes, unrotated.payload.codes);
        assert_eq!(rotated.payload.code_row_bytes, 16);
        assert_eq!(rotated.fragments.len(), 2);
    }
}
