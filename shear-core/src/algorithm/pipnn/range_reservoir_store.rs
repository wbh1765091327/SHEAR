use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use parking_lot::RwLock;

use super::disk_reservoir_store::{DiskReservoirLayout, DiskReservoirRecord};
use super::hash_prune::HashPruneReservoir;
use super::hash_prune_backfill::HashPruneBackfillPolicy;
use crate::common::{AnnError, AnnResult};

const BASE_MAGIC: [u8; 8] = *b"HPBASE01";
const MANIFEST_MAGIC: [u8; 8] = *b"HPMANF01";
const FORMAT_VERSION: u16 = 1;
const HEADER_BYTES: usize = 4096;
const HEADER_FIELDS_BYTES: usize = 88;

#[derive(Clone, Debug)]
pub(crate) struct RangeBaseStoreConfig {
    pub(crate) root: PathBuf,
    pub(crate) range_id: usize,
    pub(crate) source_start: u32,
    pub(crate) source_end: u32,
    pub(crate) l_max: usize,
    pub(crate) policy_key: [u8; 32],
    pub(crate) backfill_policy: HashPruneBackfillPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RangeManifest {
    pub(crate) base_generation: u64,
    pub(crate) base_watermark: u64,
    pub(crate) base_path: Option<PathBuf>,
}

#[derive(Debug)]
pub(crate) struct RangeBaseRow {
    pub(crate) source: u32,
    pub(crate) reservoir: HashPruneReservoir,
}

pub(crate) struct RangeBaseStore {
    config: RangeBaseStoreConfig,
    layout: DiskReservoirLayout,
    manifest_path: PathBuf,
    manifest: RwLock<RangeManifest>,
}

impl RangeBaseStore {
    pub(crate) fn create(config: RangeBaseStoreConfig) -> AnnResult<Self> {
        validate_config(&config)?;
        fs::create_dir_all(&config.root)?;
        let layout = DiskReservoirLayout::new(config.l_max, HEADER_BYTES)?;
        let manifest_path = manifest_path(&config);
        let manifest = if manifest_path.exists() {
            read_manifest(&config, &layout, &manifest_path)?
        } else {
            RangeManifest {
                base_generation: 0,
                base_watermark: 0,
                base_path: None,
            }
        };
        let store = Self {
            config,
            layout,
            manifest_path,
            manifest: RwLock::new(manifest),
        };
        if store.manifest.read().base_path.is_some() {
            let _ = store.read_current()?;
        }
        Ok(store)
    }

    pub(crate) fn manifest(&self) -> RangeManifest {
        self.manifest.read().clone()
    }

    pub(crate) fn layout(&self) -> DiskReservoirLayout {
        self.layout
    }

    pub(crate) fn range_id(&self) -> usize {
        self.config.range_id
    }

    pub(crate) fn source_range(&self) -> (u32, u32) {
        (self.config.source_start, self.config.source_end)
    }

    pub(crate) fn current_base_bytes(&self) -> AnnResult<u64> {
        match self.manifest().base_path {
            Some(path) => Ok(fs::metadata(path)?.len()),
            None => Ok(0),
        }
    }

    pub(crate) fn expected_base_bytes(&self) -> AnnResult<u64> {
        let record_count = u64::from(self.config.source_end - self.config.source_start);
        (HEADER_BYTES as u64)
            .checked_add(
                record_count
                    .checked_mul(u64::try_from(self.layout.record_bytes)?)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "range base expected record bytes overflow u64".to_string(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                AnnError::log_index_error("range base expected bytes overflow u64".to_string())
            })
    }

    pub(crate) fn read_current(&self) -> AnnResult<RangeBaseReader> {
        let manifest = self.manifest();
        match manifest.base_path {
            Some(path) => RangeBaseReader::open(
                &path,
                &self.config,
                self.layout,
                manifest.base_generation,
                manifest.base_watermark,
            ),
            None => Ok(RangeBaseReader::implicit(
                self.config.source_start,
                self.config.source_end,
                self.config.l_max,
            )),
        }
    }

    pub(crate) fn publish(
        &self,
        expected_generation: u64,
        base_watermark: u64,
        rows: Vec<RangeBaseRow>,
    ) -> AnnResult<u64> {
        validate_rows(&self.config, &rows)?;
        let mut rows = rows.into_iter().map(Ok);
        self.publish_iter(expected_generation, base_watermark, &mut rows)
    }

    pub(crate) fn publish_iter(
        &self,
        expected_generation: u64,
        base_watermark: u64,
        rows: &mut impl Iterator<Item = AnnResult<RangeBaseRow>>,
    ) -> AnnResult<u64> {
        let mut manifest_guard = self.manifest.write();
        if manifest_guard.base_generation != expected_generation {
            return Err(AnnError::log_index_error(format!(
                "range {} base generation changed: expected {expected_generation}, current {}",
                self.config.range_id, manifest_guard.base_generation
            )));
        }
        let base_generation = expected_generation.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error("range base generation overflows u64".to_string())
        })?;
        let previous_base_path = manifest_guard.base_path.clone();
        let final_path = base_path(&self.config, base_generation);
        let temporary_path = temporary_path(&final_path);
        write_base(
            &temporary_path,
            &self.config,
            &self.layout,
            base_generation,
            base_watermark,
            rows,
        )?;
        fs::rename(&temporary_path, &final_path)?;
        sync_directory(&self.config.root)?;

        let next_manifest = RangeManifest {
            base_generation,
            base_watermark,
            base_path: Some(final_path),
        };
        write_manifest_atomic(
            &self.config,
            &self.layout,
            &self.manifest_path,
            &next_manifest,
        )?;
        *manifest_guard = next_manifest;
        if let Some(previous_base_path) = previous_base_path {
            match fs::remove_file(previous_base_path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(base_generation)
    }
}

pub(crate) struct RangeBaseReader {
    inner: RangeBaseReaderInner,
}

enum RangeBaseReaderInner {
    Implicit {
        next_source: u32,
        source_end: u32,
        l_max: usize,
    },
    File {
        reader: BufReader<File>,
        layout: DiskReservoirLayout,
        next_source: u32,
        source_end: u32,
        generation: u64,
        backfill_policy: HashPruneBackfillPolicy,
        failed: bool,
    },
}

impl RangeBaseReader {
    fn implicit(source_start: u32, source_end: u32, l_max: usize) -> Self {
        Self {
            inner: RangeBaseReaderInner::Implicit {
                next_source: source_start,
                source_end,
                l_max,
            },
        }
    }

    fn open(
        path: &Path,
        config: &RangeBaseStoreConfig,
        layout: DiskReservoirLayout,
        generation: u64,
        watermark: u64,
    ) -> AnnResult<Self> {
        let expected_records = u64::from(config.source_end - config.source_start);
        let expected_len = (HEADER_BYTES as u64)
            .checked_add(
                expected_records
                    .checked_mul(u64::try_from(layout.record_bytes)?)
                    .ok_or_else(|| {
                        AnnError::log_index_error(
                            "range base record bytes overflow u64".to_string(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                AnnError::log_index_error("range base file length overflows u64".to_string())
            })?;
        let file = File::open(path)?;
        let actual_len = file.metadata()?.len();
        if actual_len != expected_len {
            return Err(AnnError::log_index_error(format!(
                "range base length mismatch for {}: expected {expected_len}, got {actual_len}",
                path.display()
            )));
        }
        let mut reader = BufReader::new(file);
        let mut header = [0_u8; HEADER_BYTES];
        reader.read_exact(&mut header)?;
        validate_header(
            &header, BASE_MAGIC, config, &layout, true, generation, watermark,
        )?;
        Ok(Self {
            inner: RangeBaseReaderInner::File {
                reader,
                layout,
                next_source: config.source_start,
                source_end: config.source_end,
                generation,
                backfill_policy: config.backfill_policy,
                failed: false,
            },
        })
    }
}

impl Iterator for RangeBaseReader {
    type Item = AnnResult<RangeBaseRow>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RangeBaseReaderInner::Implicit {
                next_source,
                source_end,
                l_max,
            } => {
                if *next_source == *source_end {
                    return None;
                }
                let source = *next_source;
                *next_source += 1;
                Some(Ok(RangeBaseRow {
                    source,
                    reservoir: HashPruneReservoir::new(*l_max),
                }))
            }
            RangeBaseReaderInner::File {
                reader,
                layout,
                next_source,
                source_end,
                generation,
                backfill_policy,
                failed,
            } => {
                if *failed || *next_source == *source_end {
                    return None;
                }
                let mut bytes = vec![0_u8; layout.record_bytes];
                if let Err(error) = reader.read_exact(&mut bytes) {
                    *failed = true;
                    return Some(Err(error.into()));
                }
                let record = match DiskReservoirRecord::decode(layout, &bytes) {
                    Ok(record) => record,
                    Err(error) => {
                        *failed = true;
                        return Some(Err(error));
                    }
                };
                let source = *next_source;
                if record.source_id != source || u64::from(record.generation) != *generation {
                    *failed = true;
                    return Some(Err(AnnError::log_index_error(format!(
                        "range base record identity mismatch: expected source {} generation {}, got source {} generation {}",
                        source, *generation, record.source_id, record.generation
                    ))));
                }
                *next_source += 1;
                Some(
                    record
                        .into_reservoir(layout.l_max, *backfill_policy)
                        .map(|reservoir| RangeBaseRow { source, reservoir }),
                )
            }
        }
    }
}

fn validate_config(config: &RangeBaseStoreConfig) -> AnnResult<()> {
    if config.source_start >= config.source_end {
        return Err(AnnError::log_index_config_error(
            "source_range".to_string(),
            "range base store requires a non-empty increasing source range".to_string(),
        ));
    }
    if config.l_max == 0 {
        return Err(AnnError::log_index_config_error(
            "l_max".to_string(),
            "range base store requires l_max > 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_rows(config: &RangeBaseStoreConfig, rows: &[RangeBaseRow]) -> AnnResult<()> {
    let expected = usize::try_from(config.source_end - config.source_start)?;
    if rows.len() != expected {
        return Err(AnnError::log_index_error(format!(
            "range base publish requires {expected} rows, got {}",
            rows.len()
        )));
    }
    for (offset, row) in rows.iter().enumerate() {
        let expected_source = config
            .source_start
            .checked_add(u32::try_from(offset)?)
            .ok_or_else(|| {
                AnnError::log_index_error("range base source id overflows u32".to_string())
            })?;
        if row.source != expected_source {
            return Err(AnnError::log_index_error(format!(
                "range base rows are not contiguous: expected source {expected_source}, got {}",
                row.source
            )));
        }
    }
    Ok(())
}

fn write_base(
    path: &Path,
    config: &RangeBaseStoreConfig,
    layout: &DiskReservoirLayout,
    generation: u64,
    watermark: u64,
    rows: &mut impl Iterator<Item = AnnResult<RangeBaseRow>>,
) -> AnnResult<()> {
    let generation_u32 = u32::try_from(generation).map_err(|_| {
        AnnError::log_index_error(
            "range base generation exceeds the reservoir record format".to_string(),
        )
    })?;
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&encode_header(
        BASE_MAGIC, config, layout, true, generation, watermark,
    )?)?;
    let mut bytes = vec![0_u8; layout.record_bytes];
    for source in config.source_start..config.source_end {
        let row = rows.next().transpose()?.ok_or_else(|| {
            AnnError::log_index_error(format!("range base publish ended before source {source}"))
        })?;
        if row.source != source {
            return Err(AnnError::log_index_error(format!(
                "range base rows are not contiguous: expected source {source}, got {}",
                row.source
            )));
        }
        DiskReservoirRecord::from_reservoir(row.source, generation_u32, &row.reservoir)?
            .encode_into(layout, &mut bytes)?;
        writer.write_all(&bytes)?;
    }
    if rows.next().transpose()?.is_some() {
        return Err(AnnError::log_index_error(
            "range base publish received more rows than the source range".to_string(),
        ));
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn write_manifest_atomic(
    config: &RangeBaseStoreConfig,
    layout: &DiskReservoirLayout,
    manifest_path: &Path,
    manifest: &RangeManifest,
) -> AnnResult<()> {
    let temporary_path = temporary_path(manifest_path);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&encode_header(
        MANIFEST_MAGIC,
        config,
        layout,
        manifest.base_path.is_some(),
        manifest.base_generation,
        manifest.base_watermark,
    )?)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temporary_path, manifest_path)?;
    sync_directory(&config.root)
}

fn read_manifest(
    config: &RangeBaseStoreConfig,
    layout: &DiskReservoirLayout,
    path: &Path,
) -> AnnResult<RangeManifest> {
    let file = File::open(path)?;
    if file.metadata()?.len() != HEADER_BYTES as u64 {
        return Err(AnnError::log_index_error(format!(
            "range manifest length mismatch for {}",
            path.display()
        )));
    }
    let mut reader = BufReader::new(file);
    let mut bytes = [0_u8; HEADER_BYTES];
    reader.read_exact(&mut bytes)?;
    let (has_base, generation, watermark) =
        validate_header_identity(&bytes, MANIFEST_MAGIC, config, layout)?;
    let base_path = has_base.then(|| base_path(config, generation));
    if base_path.as_ref().is_some_and(|base| !base.exists()) {
        return Err(AnnError::log_index_error(format!(
            "range manifest references missing base generation {generation}"
        )));
    }
    Ok(RangeManifest {
        base_generation: generation,
        base_watermark: watermark,
        base_path,
    })
}

fn encode_header(
    magic: [u8; 8],
    config: &RangeBaseStoreConfig,
    layout: &DiskReservoirLayout,
    has_base: bool,
    generation: u64,
    watermark: u64,
) -> AnnResult<[u8; HEADER_BYTES]> {
    let mut output = [0_u8; HEADER_BYTES];
    output[0..8].copy_from_slice(&magic);
    write_u16(&mut output, 8, FORMAT_VERSION);
    output[10] = u8::from(has_base);
    write_u32(&mut output, 12, HEADER_BYTES as u32);
    write_u64(&mut output, 16, u64::try_from(config.range_id)?);
    write_u32(&mut output, 24, config.source_start);
    write_u32(&mut output, 28, config.source_end);
    write_u32(&mut output, 32, u32::try_from(config.l_max)?);
    write_u32(&mut output, 36, u32::try_from(layout.record_bytes)?);
    write_u64(&mut output, 40, generation);
    write_u64(&mut output, 48, watermark);
    output[56..88].copy_from_slice(&config.policy_key);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn validate_header(
    input: &[u8; HEADER_BYTES],
    magic: [u8; 8],
    config: &RangeBaseStoreConfig,
    layout: &DiskReservoirLayout,
    has_base: bool,
    generation: u64,
    watermark: u64,
) -> AnnResult<()> {
    let (observed_has_base, observed_generation, observed_watermark) =
        validate_header_identity(input, magic, config, layout)?;
    if observed_has_base != has_base
        || observed_generation != generation
        || observed_watermark != watermark
    {
        return Err(header_mismatch_error());
    }
    Ok(())
}

fn validate_header_identity(
    input: &[u8; HEADER_BYTES],
    magic: [u8; 8],
    config: &RangeBaseStoreConfig,
    layout: &DiskReservoirLayout,
) -> AnnResult<(bool, u64, u64)> {
    if input[0..8] != magic
        || read_u16(input, 8) != FORMAT_VERSION
        || input[10] > 1
        || input[11] != 0
        || read_u32(input, 12) != HEADER_BYTES as u32
        || read_u64(input, 16) != u64::try_from(config.range_id)?
        || read_u32(input, 24) != config.source_start
        || read_u32(input, 28) != config.source_end
        || usize::try_from(read_u32(input, 32))? != config.l_max
        || usize::try_from(read_u32(input, 36))? != layout.record_bytes
        || input[56..88] != config.policy_key
        || input[HEADER_FIELDS_BYTES..].iter().any(|byte| *byte != 0)
    {
        return Err(header_mismatch_error());
    }
    Ok((input[10] == 1, read_u64(input, 40), read_u64(input, 48)))
}

fn header_mismatch_error() -> AnnError {
    AnnError::log_index_error("range base header/configuration mismatch".to_string())
}

fn manifest_path(config: &RangeBaseStoreConfig) -> PathBuf {
    config
        .root
        .join(format!("range-{:08}.manifest", config.range_id))
}

fn base_path(config: &RangeBaseStoreConfig, generation: u64) -> PathBuf {
    config.root.join(format!(
        "range-{:08}-base-g{generation:020}.bin",
        config.range_id
    ))
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn sync_directory(path: &Path) -> AnnResult<()> {
    File::open(path)?.sync_all()?;
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

#[inline]
fn write_u16(output: &mut [u8], offset: usize, value: u16) {
    output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[inline]
fn write_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[inline]
fn write_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::{RangeBaseRow, RangeBaseStore, RangeBaseStoreConfig};
    use crate::algorithm::pipnn::hash_prune::HashPruneReservoir;
    use crate::algorithm::pipnn::hash_prune_backfill::HashPruneBackfillPolicy;
    use crate::common::AnnResult;

    fn test_config(
        root: &Path,
        source_start: u32,
        source_end: u32,
        l_max: usize,
    ) -> RangeBaseStoreConfig {
        RangeBaseStoreConfig {
            root: root.to_path_buf(),
            range_id: 0,
            source_start,
            source_end,
            l_max,
            policy_key: [7; 32],
            backfill_policy: HashPruneBackfillPolicy::off(),
        }
    }

    fn row(source: u32, target: u32) -> RangeBaseRow {
        let mut reservoir = HashPruneReservoir::new(4);
        reservoir.insert(target, target as u16, target as f32);
        RangeBaseRow { source, reservoir }
    }

    #[test]
    fn absent_base_reads_as_empty_range() {
        let dir = tempdir().unwrap();
        let store = RangeBaseStore::create(test_config(dir.path(), 100, 104, 4)).unwrap();
        let rows = store
            .read_current()
            .unwrap()
            .collect::<AnnResult<Vec<_>>>()
            .unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| row.reservoir.slots().is_empty()));
    }

    #[test]
    fn published_base_is_read_in_source_order() {
        let dir = tempdir().unwrap();
        let store = RangeBaseStore::create(test_config(dir.path(), 10, 13, 4)).unwrap();
        let generation = store
            .publish(0, 9, vec![row(10, 20), row(11, 21), row(12, 22)])
            .unwrap();
        assert_eq!(generation, 1);
        let rows = store
            .read_current()
            .unwrap()
            .collect::<AnnResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            rows.iter().map(|row| row.source).collect::<Vec<_>>(),
            vec![10, 11, 12]
        );
        assert_eq!(store.manifest().base_watermark, 9);
    }

    #[test]
    fn base_reader_rejects_policy_key_mismatch() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path(), 0, 2, 4);
        let store = RangeBaseStore::create(config.clone()).unwrap();
        store.publish(0, 1, vec![row(0, 10), row(1, 11)]).unwrap();
        drop(store);

        let mut mismatched = config;
        mismatched.policy_key[0] ^= 1;
        assert!(RangeBaseStore::create(mismatched).is_err());
    }
}
