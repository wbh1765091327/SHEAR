use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::{Advice, Mmap, MmapOptions};

use super::disk_reservoir_store::{DiskReservoirLayout, DiskReservoirRecord};
use super::hash_prune::HashPruneReservoir;
use super::hash_prune_backfill::HashPruneBackfillPolicy;
use super::hash_prune_evidence::HashPruneMeta;
use super::leaf_build::PendingEdge;
use crate::common::{AnnError, AnnResult};

pub(crate) const DELTA_RECORD_BYTES: usize = 24;
const CHECKPOINT_HEADER_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReservoirDeltaRecord {
    pub(crate) source: u32,
    pub(crate) target: u32,
    pub(crate) hash: u16,
    pub(crate) flags: u16,
    pub(crate) dist_bits: u32,
    pub(crate) ticket: u64,
}

impl ReservoirDeltaRecord {
    pub(crate) fn from_edge(ticket: u64, edge: &PendingEdge) -> AnnResult<Self> {
        Ok(Self {
            source: u32::try_from(edge.p).map_err(|_| {
                AnnError::log_index_error(format!(
                    "log-assisted delta only supports u32 source ids, got {}",
                    edge.p
                ))
            })?,
            target: edge.c,
            hash: edge.hash,
            flags: edge.meta.to_spill_flags()?,
            dist_bits: edge.dist.to_bits(),
            ticket,
        })
    }

    pub(crate) fn encode(self) -> [u8; DELTA_RECORD_BYTES] {
        let mut output = [0_u8; DELTA_RECORD_BYTES];
        output[0..4].copy_from_slice(&self.source.to_le_bytes());
        output[4..8].copy_from_slice(&self.target.to_le_bytes());
        output[8..10].copy_from_slice(&self.hash.to_le_bytes());
        output[10..12].copy_from_slice(&self.flags.to_le_bytes());
        output[12..16].copy_from_slice(&self.dist_bits.to_le_bytes());
        output[16..24].copy_from_slice(&self.ticket.to_le_bytes());
        output
    }

    pub(crate) fn decode(input: &[u8]) -> AnnResult<Self> {
        if input.len() != DELTA_RECORD_BYTES {
            return Err(AnnError::log_index_error(format!(
                "log-assisted delta length mismatch: expected {DELTA_RECORD_BYTES}, got {}",
                input.len()
            )));
        }
        let flags = read_u16(input, 10);
        HashPruneMeta::from_spill_flags(flags)?;
        Ok(Self {
            source: read_u32(input, 0),
            target: read_u32(input, 4),
            hash: read_u16(input, 8),
            flags,
            dist_bits: read_u32(input, 12),
            ticket: read_u64(input, 16),
        })
    }

    pub(crate) fn to_edge(self) -> AnnResult<PendingEdge> {
        Ok(PendingEdge {
            p: self.source as usize,
            c: self.target,
            hash: self.hash,
            dist: f32::from_bits(self.dist_bits),
            meta: HashPruneMeta::from_spill_flags(self.flags)?,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ReservoirCheckpointRecord {
    pub(crate) source: u32,
    pub(crate) covered_ticket: u64,
    pub(crate) cache_generation: u32,
    pub(crate) record: DiskReservoirRecord,
}

impl ReservoirCheckpointRecord {
    pub(crate) fn from_reservoir(
        source: u32,
        covered_ticket: u64,
        cache_generation: u32,
        reservoir: &HashPruneReservoir,
    ) -> AnnResult<Self> {
        Ok(Self {
            source,
            covered_ticket,
            cache_generation,
            record: DiskReservoirRecord::from_reservoir(source, cache_generation, reservoir)?,
        })
    }

    pub(crate) fn encoded_bytes(layout: &DiskReservoirLayout) -> AnnResult<usize> {
        CHECKPOINT_HEADER_BYTES
            .checked_add(layout.record_bytes)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "log-assisted checkpoint record size overflows usize".to_string(),
                )
            })
    }

    pub(crate) fn encode(&self, layout: &DiskReservoirLayout) -> AnnResult<Vec<u8>> {
        if self.source != self.record.source_id || self.cache_generation != self.record.generation {
            return Err(AnnError::log_index_error(
                "checkpoint header does not match embedded reservoir record".to_string(),
            ));
        }
        let mut output = vec![0_u8; Self::encoded_bytes(layout)?];
        output[0..4].copy_from_slice(&self.source.to_le_bytes());
        output[4..8].copy_from_slice(&self.cache_generation.to_le_bytes());
        output[8..16].copy_from_slice(&self.covered_ticket.to_le_bytes());
        self.record
            .encode_into(layout, &mut output[CHECKPOINT_HEADER_BYTES..])?;
        Ok(output)
    }

    pub(crate) fn decode(layout: &DiskReservoirLayout, input: &[u8]) -> AnnResult<Self> {
        let expected = Self::encoded_bytes(layout)?;
        if input.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "log-assisted checkpoint length mismatch: expected {expected}, got {}",
                input.len()
            )));
        }
        let source = read_u32(input, 0);
        let cache_generation = read_u32(input, 4);
        let covered_ticket = read_u64(input, 8);
        let record =
            DiskReservoirRecord::decode(layout, &input[CHECKPOINT_HEADER_BYTES..expected])?;
        if source != record.source_id || cache_generation != record.generation {
            return Err(AnnError::log_index_error(
                "checkpoint header does not match embedded reservoir record".to_string(),
            ));
        }
        Ok(Self {
            source,
            covered_ticket,
            cache_generation,
            record,
        })
    }

    pub(crate) fn into_reservoir(
        self,
        l_max: usize,
        policy: HashPruneBackfillPolicy,
    ) -> AnnResult<HashPruneReservoir> {
        self.record.into_reservoir(l_max, policy)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeltaRunDesc {
    pub(crate) path: PathBuf,
    pub(crate) range_id: usize,
    pub(crate) epoch: u64,
    pub(crate) run_id: u64,
    pub(crate) record_count: u64,
    pub(crate) byte_len: u64,
    pub(crate) min_ticket: u64,
    pub(crate) max_ticket: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckpointRunDesc {
    pub(crate) path: PathBuf,
    pub(crate) range_id: usize,
    pub(crate) epoch: u64,
    pub(crate) run_id: u64,
    pub(crate) record_count: u64,
    pub(crate) byte_len: u64,
    pub(crate) min_ticket: u64,
    pub(crate) max_ticket: u64,
}

pub(crate) fn write_delta_run(
    path: &Path,
    range_id: usize,
    epoch: u64,
    records: Vec<ReservoirDeltaRecord>,
) -> AnnResult<DeltaRunDesc> {
    write_delta_run_with_id(path, range_id, epoch, 0, records)
}

pub(crate) fn write_delta_run_with_id(
    path: &Path,
    range_id: usize,
    epoch: u64,
    run_id: u64,
    mut records: Vec<ReservoirDeltaRecord>,
) -> AnnResult<DeltaRunDesc> {
    if records.is_empty() {
        return Err(AnnError::log_index_error(
            "cannot create an empty log-assisted delta run".to_string(),
        ));
    }
    records.sort_unstable_by_key(|record| (record.source, record.ticket));
    reject_duplicate_keys(
        records.windows(2).map(|pair| {
            (
                (pair[0].source, pair[0].ticket),
                (pair[1].source, pair[1].ticket),
            )
        }),
        "delta",
    )?;

    let record_count = u64::try_from(records.len())?;
    let byte_len = record_count
        .checked_mul(DELTA_RECORD_BYTES as u64)
        .ok_or_else(|| {
            AnnError::log_index_error("delta run byte length overflows u64".to_string())
        })?;
    let min_ticket = records.iter().map(|record| record.ticket).min().unwrap();
    let max_ticket = records.iter().map(|record| record.ticket).max().unwrap();

    write_immutable_run_atomic(path, "delta", move |writer| {
        for record in records {
            writer.write_all(&record.encode())?;
        }
        Ok(())
    })?;

    Ok(DeltaRunDesc {
        path: path.to_path_buf(),
        range_id,
        epoch,
        run_id,
        record_count,
        byte_len,
        min_ticket,
        max_ticket,
    })
}

pub(crate) fn write_checkpoint_run(
    path: &Path,
    range_id: usize,
    epoch: u64,
    layout: &DiskReservoirLayout,
    records: Vec<ReservoirCheckpointRecord>,
) -> AnnResult<CheckpointRunDesc> {
    write_checkpoint_run_with_id(path, range_id, epoch, 0, layout, records)
}

pub(crate) fn write_checkpoint_run_with_id(
    path: &Path,
    range_id: usize,
    epoch: u64,
    run_id: u64,
    layout: &DiskReservoirLayout,
    mut records: Vec<ReservoirCheckpointRecord>,
) -> AnnResult<CheckpointRunDesc> {
    if records.is_empty() {
        return Err(AnnError::log_index_error(
            "cannot create an empty log-assisted checkpoint run".to_string(),
        ));
    }
    records.sort_unstable_by_key(|record| (record.source, record.covered_ticket));
    reject_duplicate_keys(
        records.windows(2).map(|pair| {
            (
                (pair[0].source, pair[0].covered_ticket),
                (pair[1].source, pair[1].covered_ticket),
            )
        }),
        "checkpoint",
    )?;

    let encoded_bytes = ReservoirCheckpointRecord::encoded_bytes(layout)?;
    let record_count = u64::try_from(records.len())?;
    let byte_len = record_count
        .checked_mul(u64::try_from(encoded_bytes)?)
        .ok_or_else(|| {
            AnnError::log_index_error("checkpoint run byte length overflows u64".to_string())
        })?;
    let min_ticket = records
        .iter()
        .map(|record| record.covered_ticket)
        .min()
        .unwrap();
    let max_ticket = records
        .iter()
        .map(|record| record.covered_ticket)
        .max()
        .unwrap();

    write_immutable_run_atomic(path, "checkpoint", move |writer| {
        for record in records {
            writer.write_all(&record.encode(layout)?)?;
        }
        Ok(())
    })?;

    Ok(CheckpointRunDesc {
        path: path.to_path_buf(),
        range_id,
        epoch,
        run_id,
        record_count,
        byte_len,
        min_ticket,
        max_ticket,
    })
}

pub(crate) struct DeltaRunReader {
    mmap: Mmap,
    offset: usize,
    remaining: u64,
    previous_key: Option<(u32, u64)>,
    failed: bool,
}

impl DeltaRunReader {
    pub(crate) fn open(desc: &DeltaRunDesc) -> AnnResult<Self> {
        let expected = desc
            .record_count
            .checked_mul(DELTA_RECORD_BYTES as u64)
            .ok_or_else(|| {
                AnnError::log_index_error("delta run descriptor length overflows u64".to_string())
            })?;
        validate_run_length(&desc.path, desc.byte_len, expected, "delta")?;
        let mmap = map_immutable_run(&desc.path)?;
        Ok(Self {
            mmap,
            offset: 0,
            remaining: desc.record_count,
            previous_key: None,
            failed: false,
        })
    }
}

impl Iterator for DeltaRunReader {
    type Item = AnnResult<ReservoirDeltaRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.remaining == 0 {
            return None;
        }
        let end = match self.offset.checked_add(DELTA_RECORD_BYTES) {
            Some(end) => end,
            None => {
                self.failed = true;
                return Some(Err(AnnError::log_index_error(
                    "delta reader offset overflows usize".to_string(),
                )));
            }
        };
        let Some(bytes) = self.mmap.get(self.offset..end) else {
            self.failed = true;
            return Some(Err(AnnError::log_index_error(
                "delta reader exceeds its immutable mapping".to_string(),
            )));
        };
        self.offset = end;
        self.remaining -= 1;
        let record = match ReservoirDeltaRecord::decode(bytes) {
            Ok(record) => record,
            Err(error) => {
                self.failed = true;
                return Some(Err(error));
            }
        };
        let key = (record.source, record.ticket);
        if self.previous_key.is_some_and(|previous| previous >= key) {
            self.failed = true;
            return Some(Err(AnnError::log_index_error(
                "delta run records are not strictly ordered".to_string(),
            )));
        }
        self.previous_key = Some(key);
        Some(Ok(record))
    }
}

pub(crate) struct CheckpointRunReader {
    mmap: Mmap,
    offset: usize,
    layout: DiskReservoirLayout,
    record_bytes: usize,
    remaining: u64,
    previous_key: Option<(u32, u64)>,
    failed: bool,
}

impl CheckpointRunReader {
    pub(crate) fn open(desc: &CheckpointRunDesc, layout: &DiskReservoirLayout) -> AnnResult<Self> {
        let record_bytes = ReservoirCheckpointRecord::encoded_bytes(layout)?;
        let expected = desc
            .record_count
            .checked_mul(u64::try_from(record_bytes)?)
            .ok_or_else(|| {
                AnnError::log_index_error(
                    "checkpoint run descriptor length overflows u64".to_string(),
                )
            })?;
        validate_run_length(&desc.path, desc.byte_len, expected, "checkpoint")?;
        let mmap = map_immutable_run(&desc.path)?;
        Ok(Self {
            mmap,
            offset: 0,
            layout: *layout,
            record_bytes,
            remaining: desc.record_count,
            previous_key: None,
            failed: false,
        })
    }
}

impl Iterator for CheckpointRunReader {
    type Item = AnnResult<ReservoirCheckpointRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.remaining == 0 {
            return None;
        }
        let end = match self.offset.checked_add(self.record_bytes) {
            Some(end) => end,
            None => {
                self.failed = true;
                return Some(Err(AnnError::log_index_error(
                    "checkpoint reader offset overflows usize".to_string(),
                )));
            }
        };
        let Some(bytes) = self.mmap.get(self.offset..end) else {
            self.failed = true;
            return Some(Err(AnnError::log_index_error(
                "checkpoint reader exceeds its immutable mapping".to_string(),
            )));
        };
        self.offset = end;
        self.remaining -= 1;
        let record = match ReservoirCheckpointRecord::decode(&self.layout, bytes) {
            Ok(record) => record,
            Err(error) => {
                self.failed = true;
                return Some(Err(error));
            }
        };
        let key = (record.source, record.covered_ticket);
        if self.previous_key.is_some_and(|previous| previous >= key) {
            self.failed = true;
            return Some(Err(AnnError::log_index_error(
                "checkpoint run records are not strictly ordered".to_string(),
            )));
        }
        self.previous_key = Some(key);
        Some(Ok(record))
    }
}

fn map_immutable_run(path: &Path) -> AnnResult<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let _ = mmap.advise(Advice::Sequential);
    Ok(mmap)
}

fn create_immutable_file(path: &Path, kind: &str) -> AnnResult<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            AnnError::log_index_error(format!(
                "failed to create immutable {kind} run {}: {error}",
                path.display()
            ))
        })
}

fn write_immutable_run_atomic(
    path: &Path,
    kind: &str,
    write_records: impl FnOnce(&mut BufWriter<File>) -> AnnResult<()>,
) -> AnnResult<()> {
    if path.exists() {
        return Err(AnnError::log_index_error(format!(
            "immutable {kind} run already exists: {}",
            path.display()
        )));
    }
    let temporary_path = temporary_run_path(path);
    let file = create_immutable_file(&temporary_path, kind)?;
    let mut writer = BufWriter::new(file);
    let write_result = (|| -> AnnResult<()> {
        write_records(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_data()?;
        Ok(())
    })();
    drop(writer);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(error);
    }
    if path.exists() {
        let _ = fs::remove_file(&temporary_path);
        return Err(AnnError::log_index_error(format!(
            "immutable {kind} run appeared before publish: {}",
            path.display()
        )));
    }
    fs::rename(&temporary_path, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn temporary_run_path(path: &Path) -> PathBuf {
    let mut file_name = path.file_name().unwrap_or_default().to_os_string();
    file_name.push(".tmp");
    path.with_file_name(file_name)
}

fn validate_run_length(
    path: &Path,
    descriptor_bytes: u64,
    expected_bytes: u64,
    kind: &str,
) -> AnnResult<()> {
    let actual_bytes = File::open(path)?.metadata()?.len();
    if descriptor_bytes != expected_bytes || actual_bytes != expected_bytes {
        return Err(AnnError::log_index_error(format!(
            "{kind} run length mismatch for {}: descriptor={descriptor_bytes}, expected={expected_bytes}, actual={actual_bytes}",
            path.display()
        )));
    }
    Ok(())
}

fn reject_duplicate_keys(
    pairs: impl IntoIterator<Item = ((u32, u64), (u32, u64))>,
    kind: &str,
) -> AnnResult<()> {
    if let Some((key, _)) = pairs.into_iter().find(|(left, right)| left == right) {
        return Err(AnnError::log_index_error(format!(
            "duplicate {kind} run key: source={}, ticket={}",
            key.0, key.1
        )));
    }
    Ok(())
}

#[inline]
fn read_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(input[offset..offset + 2].try_into().unwrap())
}

#[inline]
fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap())
}

#[inline]
fn read_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(input[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{
        CheckpointRunReader, DELTA_RECORD_BYTES, DeltaRunReader, ReservoirCheckpointRecord,
        ReservoirDeltaRecord, write_checkpoint_run, write_delta_run,
    };
    use crate::algorithm::pipnn::disk_reservoir_store::DiskReservoirLayout;
    use crate::algorithm::pipnn::hash_prune::{HashPruneReservoir, HashPruneReservoirState};
    use crate::algorithm::pipnn::hash_prune_backfill::HashPruneBackfillPolicy;
    use crate::algorithm::pipnn::hash_prune_evidence::HashPruneMeta;
    use crate::algorithm::pipnn::leaf_build::PendingEdge;
    use crate::common::AnnResult;

    fn state_signature(state: HashPruneReservoirState) -> (Vec<(u32, u16, u32, u16)>, u16) {
        let slots = state
            .slots
            .into_iter()
            .map(|slot| {
                (
                    slot.target,
                    slot.hash,
                    slot.dist.to_bits(),
                    slot.meta.to_state_flags().unwrap(),
                )
            })
            .collect();
        (slots, state.backup_count)
    }

    #[test]
    fn delta_record_is_24_bytes_and_round_trips() {
        let edge = PendingEdge {
            p: 7,
            c: 11,
            hash: 3,
            dist: 1.25,
            meta: HashPruneMeta::default(),
        };
        let record = ReservoirDeltaRecord::from_edge(42, &edge).unwrap();
        let bytes = record.encode();
        assert_eq!(DELTA_RECORD_BYTES, 24);
        assert_eq!(bytes.len(), DELTA_RECORD_BYTES);
        assert_eq!(ReservoirDeltaRecord::decode(&bytes).unwrap(), record);

        let decoded_edge = record.to_edge().unwrap();
        assert_eq!(decoded_edge.p, edge.p);
        assert_eq!(decoded_edge.c, edge.c);
        assert_eq!(decoded_edge.hash, edge.hash);
        assert_eq!(decoded_edge.dist.to_bits(), edge.dist.to_bits());
        assert_eq!(
            decoded_edge.meta.to_spill_flags().unwrap(),
            edge.meta.to_spill_flags().unwrap()
        );
    }

    #[test]
    fn checkpoint_round_trip_preserves_reservoir_state() {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert(10, 1, 2.0);
        reservoir.insert(20, 2, 1.0);
        let record = ReservoirCheckpointRecord::from_reservoir(7, 19, 3, &reservoir).unwrap();
        let layout = DiskReservoirLayout::new(4, 4096).unwrap();
        let bytes = record.encode(&layout).unwrap();
        let decoded = ReservoirCheckpointRecord::decode(&layout, &bytes).unwrap();
        assert_eq!(decoded.source, 7);
        assert_eq!(decoded.covered_ticket, 19);
        assert_eq!(decoded.cache_generation, 3);
        assert_eq!(
            state_signature(
                decoded
                    .into_reservoir(4, HashPruneBackfillPolicy::off())
                    .unwrap()
                    .state_snapshot()
            ),
            state_signature(reservoir.state_snapshot())
        );
    }

    #[test]
    fn delta_decode_rejects_truncated_record() {
        assert!(ReservoirDeltaRecord::decode(&[0_u8; 23]).is_err());
    }

    fn delta(source: u32, target: u32, ticket: u64) -> ReservoirDeltaRecord {
        ReservoirDeltaRecord {
            source,
            target,
            hash: target as u16,
            flags: 0,
            dist_bits: (target as f32).to_bits(),
            ticket,
        }
    }

    #[test]
    fn delta_run_sorts_by_source_then_ticket() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delta.run");
        let records = vec![delta(2, 7, 20), delta(1, 9, 30), delta(1, 8, 10)];
        let desc = write_delta_run(&path, 4, 0, records).unwrap();
        let observed = DeltaRunReader::open(&desc)
            .unwrap()
            .collect::<AnnResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            observed
                .iter()
                .map(|record| (record.source, record.ticket))
                .collect::<Vec<_>>(),
            vec![(1, 10), (1, 30), (2, 20)]
        );
    }

    #[test]
    fn run_writer_rejects_duplicate_source_ticket() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("duplicate.run");
        assert!(write_delta_run(&path, 0, 0, vec![delta(1, 2, 3), delta(1, 4, 3)]).is_err());
    }

    #[test]
    fn checkpoint_run_sorts_and_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.run");
        let layout = DiskReservoirLayout::new(4, 4096).unwrap();
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert(10, 1, 2.0);
        let later = ReservoirCheckpointRecord::from_reservoir(5, 11, 2, &reservoir).unwrap();
        let earlier = ReservoirCheckpointRecord::from_reservoir(3, 19, 1, &reservoir).unwrap();
        let desc = write_checkpoint_run(&path, 2, 7, &layout, vec![later, earlier]).unwrap();
        let observed = CheckpointRunReader::open(&desc, &layout)
            .unwrap()
            .collect::<AnnResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            observed
                .iter()
                .map(|record| (record.source, record.covered_ticket))
                .collect::<Vec<_>>(),
            vec![(3, 19), (5, 11)]
        );
    }
}
