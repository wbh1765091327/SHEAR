use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use super::lrpf::{
    LrpfCandidate, LrpfConfig, LrpfWitness, compress_lrpf_row_before_spill, reduce_lrpf_row,
};
use crate::common::{AnnError, AnnResult};

const LRPF_FRAGMENT_HEADER_BYTES: usize = std::mem::size_of::<LrpfFragmentHeader>();
const LRPF_CANDIDATE_BYTES: usize = std::mem::size_of::<LrpfCandidate>();
const LRPF_WITNESS_BYTES: usize = std::mem::size_of::<LrpfWitness>();
const DEFAULT_LRPF_DIGEST_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_LRPF_DIGEST_RECORDS: usize = 1_000_000;
const LRPF_DIGEST_SHARD_LOCAL_FLUSH_DIVISOR: usize = 4;
const LRPF_DIGEST_ROW_COMPACT_FACTOR: usize = 4;
const MIN_LRPF_DIGEST_ROW_RECORDS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct LrpfFragmentHeader {
    pub(crate) src: u32,
    pub(crate) candidate_count: u16,
    pub(crate) witness_count: u16,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LrpfFragment {
    pub(crate) src: u32,
    pub(crate) candidates: Vec<LrpfCandidate>,
    pub(crate) witnesses: Vec<LrpfWitness>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrpfSpillStats {
    pub(crate) part_files: usize,
    pub(crate) fragment_count: usize,
    pub(crate) candidate_records: usize,
    pub(crate) witness_records: usize,
    pub(crate) record_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrpfPartFileDesc {
    pub(crate) shard_id: usize,
    pub(crate) path: PathBuf,
    pub(crate) byte_len: u64,
    pub(crate) fragment_count: usize,
    pub(crate) candidate_records: usize,
    pub(crate) witness_records: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrpfSpillArtifacts {
    pub(crate) part_files: Vec<LrpfPartFileDesc>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrpfReduceStats {
    pub(crate) shard_groups: usize,
    pub(crate) fragments_scanned: usize,
    pub(crate) part_files_opened: usize,
    pub(crate) max_fragments_in_group: usize,
    pub(crate) candidate_records: usize,
    pub(crate) witness_records: usize,
    pub(crate) reducer_pruned_candidates: usize,
}

pub(crate) trait LrpfFragmentSink: Sync {
    fn push_lrpf_fragment(
        &self,
        src: u32,
        candidates: &[LrpfCandidate],
        witnesses: &[LrpfWitness],
    ) -> AnnResult<()>;
}

#[derive(Debug)]
pub(crate) struct LrpfFragmentWriter {
    writer: BufWriter<File>,
}

impl LrpfFragmentWriter {
    pub(crate) fn create(path: &Path) -> AnnResult<Self> {
        Ok(Self {
            writer: BufWriter::new(File::create(path)?),
        })
    }

    pub(crate) fn push_fragment(
        &mut self,
        src: u32,
        candidates: &[LrpfCandidate],
        witnesses: &[LrpfWitness],
    ) -> AnnResult<()> {
        let candidate_count = u16::try_from(candidates.len()).map_err(|_| {
            AnnError::log_index_error(format!(
                "LRPF fragment for src={src} has too many candidates: {}",
                candidates.len()
            ))
        })?;
        let witness_count = u16::try_from(witnesses.len()).map_err(|_| {
            AnnError::log_index_error(format!(
                "LRPF fragment for src={src} has too many witnesses: {}",
                witnesses.len()
            ))
        })?;

        self.writer.write_all(&src.to_le_bytes())?;
        self.writer.write_all(&candidate_count.to_le_bytes())?;
        self.writer.write_all(&witness_count.to_le_bytes())?;
        for candidate in candidates {
            self.writer.write_all(&candidate.dst.to_le_bytes())?;
            self.writer.write_all(&candidate.dist.to_le_bytes())?;
        }
        for witness in witnesses {
            self.writer.write_all(&witness.pivot_idx.to_le_bytes())?;
            self.writer.write_all(&witness.victim_idx.to_le_bytes())?;
            self.writer.write_all(&witness.margin.to_le_bytes())?;
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> AnnResult<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[derive(Debug)]
struct LrpfShardState {
    writer: LrpfFragmentWriter,
    desc: LrpfPartFileDesc,
    digest_rows: HashMap<u32, LrpfRowAccumulator>,
    digest_record_count: usize,
}

#[derive(Debug)]
pub(crate) struct LrpfFragmentSpillSink {
    base_path: PathBuf,
    shard_points: usize,
    digest_config: Option<LrpfConfig>,
    digest_flush_threshold_records: usize,
    shards: Mutex<HashMap<usize, Arc<Mutex<LrpfShardState>>>>,
}

impl LrpfFragmentSpillSink {
    pub(crate) fn create(base_path: &Path, shard_points: usize) -> AnnResult<Self> {
        Self::create_with_digest_config(base_path, shard_points, None, 0)
    }

    pub(crate) fn create_digest(
        base_path: &Path,
        shard_points: usize,
        config: LrpfConfig,
        spill_cache_bytes: usize,
    ) -> AnnResult<Self> {
        Self::create_with_digest_config(
            base_path,
            shard_points,
            Some(LrpfConfig {
                pre_spill_width: config.pre_spill_width.max(1),
                ..config
            }),
            spill_cache_bytes,
        )
    }

    fn create_with_digest_config(
        base_path: &Path,
        shard_points: usize,
        digest_config: Option<LrpfConfig>,
        spill_cache_bytes: usize,
    ) -> AnnResult<Self> {
        if let Some(parent) = base_path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self {
            base_path: base_path.to_path_buf(),
            shard_points: shard_points.max(1),
            digest_config,
            digest_flush_threshold_records: lrpf_digest_shard_flush_threshold_records(
                spill_cache_bytes,
            ),
            shards: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn finish(self) -> AnnResult<LrpfSpillArtifacts> {
        let shards = self.shards.into_inner();
        let mut states = Vec::with_capacity(shards.len());
        for shard in shards.into_values() {
            let shard = Arc::try_unwrap(shard).map_err(|_| {
                AnnError::log_index_error(
                    "LRPF shard state is still shared while finishing spill sink".to_string(),
                )
            })?;
            states.push(shard.into_inner());
        }
        states.sort_unstable_by_key(|state| state.desc.shard_id);
        let mut part_files = Vec::with_capacity(states.len());
        for mut state in states {
            if let Some(config) = self.digest_config {
                flush_lrpf_digest_rows(&mut state, config)?;
            }
            state.writer.finish()?;
            if state.desc.fragment_count > 0 {
                part_files.push(state.desc);
            }
        }
        Ok(LrpfSpillArtifacts { part_files })
    }
}

impl LrpfFragmentSink for LrpfFragmentSpillSink {
    fn push_lrpf_fragment(
        &self,
        src: u32,
        candidates: &[LrpfCandidate],
        witnesses: &[LrpfWitness],
    ) -> AnnResult<()> {
        let shard_id = (src as usize) / self.shard_points;
        let shard = {
            let mut shards = self.shards.lock();
            if let Some(shard) = shards.get(&shard_id) {
                Arc::clone(shard)
            } else {
                let path = lrpf_part_path_for(&self.base_path, shard_id);
                let writer = LrpfFragmentWriter::create(&path)?;
                let shard = Arc::new(Mutex::new(LrpfShardState {
                    writer,
                    desc: LrpfPartFileDesc {
                        shard_id,
                        path,
                        ..LrpfPartFileDesc::default()
                    },
                    digest_rows: HashMap::new(),
                    digest_record_count: 0,
                }));
                shards.insert(shard_id, Arc::clone(&shard));
                shard
            }
        };
        let mut shard = shard.lock();
        if let Some(config) = self.digest_config {
            insert_lrpf_fragment_to_digest(&mut shard, src, candidates, witnesses, config)?;
            if shard.digest_record_count >= self.digest_flush_threshold_records {
                flush_lrpf_digest_rows(&mut shard, config)?;
            }
        } else {
            write_lrpf_fragment_to_shard(&mut shard, src, candidates, witnesses)?;
        }
        Ok(())
    }
}

fn write_lrpf_fragment_to_shard(
    shard: &mut LrpfShardState,
    src: u32,
    candidates: &[LrpfCandidate],
    witnesses: &[LrpfWitness],
) -> AnnResult<()> {
    shard.writer.push_fragment(src, candidates, witnesses)?;
    shard.desc.fragment_count += 1;
    shard.desc.candidate_records += candidates.len();
    shard.desc.witness_records += witnesses.len();
    shard.desc.byte_len = shard.desc.byte_len.saturating_add(
        (LRPF_FRAGMENT_HEADER_BYTES
            + candidates.len().saturating_mul(LRPF_CANDIDATE_BYTES)
            + witnesses.len().saturating_mul(LRPF_WITNESS_BYTES)) as u64,
    );
    Ok(())
}

fn insert_lrpf_fragment_to_digest(
    shard: &mut LrpfShardState,
    src: u32,
    candidates: &[LrpfCandidate],
    witnesses: &[LrpfWitness],
    config: LrpfConfig,
) -> AnnResult<()> {
    let row = shard.digest_rows.entry(src).or_default();
    let old_records = lrpf_digest_row_records(row);

    row.candidates.extend(candidates.iter().copied());
    for witness in witnesses {
        if witness.margin <= 0.0 {
            continue;
        }
        let Some(pivot) = candidates.get(usize::from(witness.pivot_idx)) else {
            continue;
        };
        let Some(victim) = candidates.get(usize::from(witness.victim_idx)) else {
            continue;
        };
        if pivot.dst == u32::MAX || victim.dst == u32::MAX {
            continue;
        }
        row.witnesses.push(LrpfWitnessByDst {
            pivot_dst: pivot.dst,
            victim_dst: victim.dst,
            margin: witness.margin,
        });
    }

    if lrpf_digest_row_records(row) >= lrpf_digest_row_compact_threshold(config) {
        compress_lrpf_digest_row(row, config)?;
    }
    let new_records = lrpf_digest_row_records(row);
    shard.digest_record_count = shard
        .digest_record_count
        .saturating_sub(old_records)
        .saturating_add(new_records);
    Ok(())
}

fn flush_lrpf_digest_rows(shard: &mut LrpfShardState, config: LrpfConfig) -> AnnResult<()> {
    if shard.digest_record_count == 0 {
        return Ok(());
    }

    let mut rows: Vec<_> = shard.digest_rows.drain().collect();
    rows.sort_unstable_by_key(|(src, _)| *src);
    for (src, mut row) in rows {
        compress_lrpf_digest_row(&mut row, config)?;
        let (candidates, witnesses) = canonicalize_lrpf_row(&row)?;
        write_lrpf_fragment_to_shard(shard, src, &candidates, &witnesses)?;
    }
    shard.digest_record_count = 0;
    Ok(())
}

fn compress_lrpf_digest_row(row: &mut LrpfRowAccumulator, config: LrpfConfig) -> AnnResult<()> {
    if config.pre_spill_width == 0 {
        return Ok(());
    }

    let (candidates, witnesses) = canonicalize_lrpf_row(row)?;
    let compressed = compress_lrpf_row_before_spill(&candidates, &witnesses, config);
    let mut remapped_witnesses = Vec::with_capacity(compressed.witnesses.len());
    for witness in &compressed.witnesses {
        let Some(pivot) = compressed.candidates.get(usize::from(witness.pivot_idx)) else {
            continue;
        };
        let Some(victim) = compressed.candidates.get(usize::from(witness.victim_idx)) else {
            continue;
        };
        remapped_witnesses.push(LrpfWitnessByDst {
            pivot_dst: pivot.dst,
            victim_dst: victim.dst,
            margin: witness.margin,
        });
    }
    row.candidates = compressed.candidates;
    row.witnesses = remapped_witnesses;
    Ok(())
}

fn lrpf_digest_row_records(row: &LrpfRowAccumulator) -> usize {
    row.candidates.len().saturating_add(row.witnesses.len())
}

fn lrpf_digest_flush_threshold_records(spill_cache_bytes: usize) -> usize {
    let cache_bytes = if spill_cache_bytes == 0 {
        DEFAULT_LRPF_DIGEST_CACHE_BYTES
    } else {
        spill_cache_bytes
    };
    let record_bytes = LRPF_CANDIDATE_BYTES
        .saturating_add(LRPF_WITNESS_BYTES)
        .max(1);
    (cache_bytes / record_bytes)
        .max(1)
        .min(MAX_LRPF_DIGEST_RECORDS)
}

fn lrpf_digest_shard_flush_threshold_records(spill_cache_bytes: usize) -> usize {
    let threshold = lrpf_digest_flush_threshold_records(spill_cache_bytes);
    if threshold <= MIN_LRPF_DIGEST_ROW_RECORDS {
        threshold
    } else {
        (threshold / LRPF_DIGEST_SHARD_LOCAL_FLUSH_DIVISOR).max(MIN_LRPF_DIGEST_ROW_RECORDS)
    }
}

fn lrpf_digest_row_compact_threshold(config: LrpfConfig) -> usize {
    config
        .pre_spill_width
        .saturating_mul(LRPF_DIGEST_ROW_COMPACT_FACTOR)
        .max(MIN_LRPF_DIGEST_ROW_RECORDS)
}

pub(crate) fn read_lrpf_fragments(path: &Path) -> AnnResult<Vec<LrpfFragment>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut fragments = Vec::new();
    while let Some(header) = read_lrpf_header(&mut reader)? {
        let mut candidates = Vec::with_capacity(header.candidate_count as usize);
        for _ in 0..header.candidate_count {
            candidates.push(read_lrpf_candidate(&mut reader)?);
        }
        let mut witnesses = Vec::with_capacity(header.witness_count as usize);
        for _ in 0..header.witness_count {
            witnesses.push(read_lrpf_witness(&mut reader)?);
        }
        fragments.push(LrpfFragment {
            src: header.src,
            candidates,
            witnesses,
        });
    }
    Ok(fragments)
}

fn read_lrpf_header(reader: &mut impl Read) -> AnnResult<Option<LrpfFragmentHeader>> {
    let mut bytes = [0u8; LRPF_FRAGMENT_HEADER_BYTES];
    match reader.read(&mut bytes[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!("single-byte read returned more than one byte"),
        Err(err) => return Err(err.into()),
    }
    reader.read_exact(&mut bytes[1..])?;
    Ok(Some(LrpfFragmentHeader {
        src: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        candidate_count: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
        witness_count: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
    }))
}

fn read_lrpf_candidate(reader: &mut impl Read) -> AnnResult<LrpfCandidate> {
    let mut bytes = [0u8; LRPF_CANDIDATE_BYTES];
    read_exact_lrpf_payload(reader, &mut bytes, "candidate")?;
    Ok(LrpfCandidate {
        dst: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        dist: f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    })
}

fn read_lrpf_witness(reader: &mut impl Read) -> AnnResult<LrpfWitness> {
    let mut bytes = [0u8; LRPF_WITNESS_BYTES];
    read_exact_lrpf_payload(reader, &mut bytes, "witness")?;
    Ok(LrpfWitness {
        pivot_idx: u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
        victim_idx: u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        margin: f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    })
}

fn read_exact_lrpf_payload(reader: &mut impl Read, bytes: &mut [u8], label: &str) -> AnnResult<()> {
    reader.read_exact(bytes).map_err(|err| {
        if err.kind() == ErrorKind::UnexpectedEof {
            AnnError::log_index_error(format!("Truncated LRPF {label} payload"))
        } else {
            err.into()
        }
    })
}

#[derive(Debug, Default)]
pub(crate) struct ExternalLrpfReducer;

impl ExternalLrpfReducer {
    pub(crate) fn summarize_spill_artifacts(artifacts: &LrpfSpillArtifacts) -> LrpfSpillStats {
        summarize_lrpf_spill_artifacts(artifacts)
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled<F>(
        num_points: usize,
        degree: usize,
        protected_core: usize,
        long_edge_quota: usize,
        points_per_shard: usize,
        artifacts: &LrpfSpillArtifacts,
        on_point_done: F,
    ) -> AnnResult<LrpfReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_from(
            0,
            num_points,
            degree,
            protected_core,
            long_edge_quota,
            points_per_shard,
            artifacts,
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_from<F>(
        point_start: usize,
        num_points: usize,
        degree: usize,
        protected_core: usize,
        long_edge_quota: usize,
        points_per_shard: usize,
        artifacts: &LrpfSpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<LrpfReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        if num_points == 0 || point_start >= num_points {
            return Ok(LrpfReduceStats::default());
        }

        let points_per_shard = points_per_shard.max(1);
        let mut part_files = artifacts.part_files.clone();
        part_files.sort_unstable_by_key(|part| part.shard_id);
        let mut stats = LrpfReduceStats {
            shard_groups: part_files.len(),
            fragments_scanned: part_files.iter().map(|part| part.fragment_count).sum(),
            part_files_opened: part_files.len(),
            max_fragments_in_group: part_files
                .iter()
                .map(|part| part.fragment_count)
                .max()
                .unwrap_or(0),
            candidate_records: part_files.iter().map(|part| part.candidate_records).sum(),
            witness_records: part_files.iter().map(|part| part.witness_records).sum(),
            reducer_pruned_candidates: 0,
        };

        let mut next_point = point_start;
        for part in part_files {
            let shard_start = part.shard_id.saturating_mul(points_per_shard);
            let shard_end = shard_start.saturating_add(points_per_shard).min(num_points);
            if shard_end <= point_start {
                continue;
            }
            if shard_start >= num_points {
                return Err(AnnError::log_index_error(format!(
                    "LRPF reducer encountered shard_id={} outside num_points={} with points_per_shard={}",
                    part.shard_id, num_points, points_per_shard
                )));
            }
            let group_start = shard_start.max(point_start);
            if next_point < group_start {
                emit_empty_lrpf_point_range(next_point, group_start, &mut on_point_done)?;
                next_point = group_start;
            }

            let rows = read_lrpf_part_grouped_by_source(&part.path, group_start, shard_end)?;
            for (src, row) in rows {
                let src_usize = src as usize;
                if src_usize < next_point {
                    continue;
                }
                if src_usize >= shard_end {
                    return Err(AnnError::log_index_error(format!(
                        "LRPF reducer encountered src={} outside shard range [{}, {})",
                        src, shard_start, shard_end
                    )));
                }
                if next_point < src_usize {
                    emit_empty_lrpf_point_range(next_point, src_usize, &mut on_point_done)?;
                }
                let (candidates, witnesses) = canonicalize_lrpf_row(&row)?;
                let neighbors = reduce_lrpf_row(
                    &candidates,
                    &witnesses,
                    LrpfConfig {
                        degree,
                        protected_core,
                        pre_spill_width: 0,
                        long_edge_quota,
                    },
                );
                stats.reducer_pruned_candidates = stats
                    .reducer_pruned_candidates
                    .saturating_add(candidates.len().saturating_sub(neighbors.len()));
                on_point_done(src, neighbors)?;
                next_point = src_usize.saturating_add(1);
            }
        }

        if next_point < num_points {
            emit_empty_lrpf_point_range(next_point, num_points, &mut on_point_done)?;
        }
        Ok(stats)
    }
}

#[derive(Clone, Copy, Debug)]
struct LrpfWitnessByDst {
    pivot_dst: u32,
    victim_dst: u32,
    margin: f32,
}

#[derive(Clone, Debug, Default)]
struct LrpfRowAccumulator {
    candidates: Vec<LrpfCandidate>,
    witnesses: Vec<LrpfWitnessByDst>,
}

fn read_lrpf_part_grouped_by_source(
    path: &Path,
    point_start: usize,
    point_end: usize,
) -> AnnResult<BTreeMap<u32, LrpfRowAccumulator>> {
    let mut rows: BTreeMap<u32, LrpfRowAccumulator> = BTreeMap::new();
    for fragment in read_lrpf_fragments(path)? {
        let src = fragment.src as usize;
        if src < point_start {
            continue;
        }
        if src >= point_end {
            return Err(AnnError::log_index_error(format!(
                "LRPF fragment src={} outside current point range [{point_start}, {point_end})",
                fragment.src
            )));
        }
        let row = rows.entry(fragment.src).or_default();
        for witness in &fragment.witnesses {
            if witness.margin <= 0.0 {
                continue;
            }
            let Some(pivot) = fragment.candidates.get(usize::from(witness.pivot_idx)) else {
                continue;
            };
            let Some(victim) = fragment.candidates.get(usize::from(witness.victim_idx)) else {
                continue;
            };
            if pivot.dst == u32::MAX || victim.dst == u32::MAX {
                continue;
            }
            row.witnesses.push(LrpfWitnessByDst {
                pivot_dst: pivot.dst,
                victim_dst: victim.dst,
                margin: witness.margin,
            });
        }
        row.candidates.extend(fragment.candidates);
    }
    Ok(rows)
}

fn canonicalize_lrpf_row(
    row: &LrpfRowAccumulator,
) -> AnnResult<(Vec<LrpfCandidate>, Vec<LrpfWitness>)> {
    let mut best_by_dst: HashMap<u32, f32> = HashMap::new();
    for candidate in &row.candidates {
        if candidate.dst == u32::MAX || !candidate.dist.is_finite() {
            continue;
        }
        best_by_dst
            .entry(candidate.dst)
            .and_modify(|dist| {
                if candidate.dist < *dist {
                    *dist = candidate.dist;
                }
            })
            .or_insert(candidate.dist);
    }

    let mut candidates: Vec<_> = best_by_dst
        .into_iter()
        .map(|(dst, dist)| LrpfCandidate::new(dst, dist))
        .collect();
    candidates.sort_by(|left, right| {
        left.dist
            .total_cmp(&right.dist)
            .then_with(|| left.dst.cmp(&right.dst))
    });
    let mut index_by_dst = HashMap::with_capacity(candidates.len());
    for (idx, candidate) in candidates.iter().enumerate() {
        let idx = u16::try_from(idx).map_err(|_| {
            AnnError::log_index_error(format!(
                "LRPF reducer row has too many unique candidates: {}",
                candidates.len()
            ))
        })?;
        index_by_dst.insert(candidate.dst, idx);
    }

    let mut best_margin_by_pair: HashMap<(u16, u16), f32> = HashMap::new();
    for witness in &row.witnesses {
        let Some(&pivot_idx) = index_by_dst.get(&witness.pivot_dst) else {
            continue;
        };
        let Some(&victim_idx) = index_by_dst.get(&witness.victim_dst) else {
            continue;
        };
        if pivot_idx == victim_idx {
            continue;
        }
        best_margin_by_pair
            .entry((pivot_idx, victim_idx))
            .and_modify(|margin| {
                if witness.margin > *margin {
                    *margin = witness.margin;
                }
            })
            .or_insert(witness.margin);
    }

    let mut witnesses: Vec<_> = best_margin_by_pair
        .into_iter()
        .map(|((pivot_idx, victim_idx), margin)| LrpfWitness::new(pivot_idx, victim_idx, margin))
        .collect();
    witnesses.sort_by_key(|witness| (witness.pivot_idx, witness.victim_idx));

    Ok((candidates, witnesses))
}

fn summarize_lrpf_spill_artifacts(artifacts: &LrpfSpillArtifacts) -> LrpfSpillStats {
    LrpfSpillStats {
        part_files: artifacts.part_files.len(),
        fragment_count: artifacts
            .part_files
            .iter()
            .map(|part| part.fragment_count)
            .sum(),
        candidate_records: artifacts
            .part_files
            .iter()
            .map(|part| part.candidate_records)
            .sum(),
        witness_records: artifacts
            .part_files
            .iter()
            .map(|part| part.witness_records)
            .sum(),
        record_bytes: LRPF_FRAGMENT_HEADER_BYTES,
    }
}

fn emit_empty_lrpf_point_range<F>(start: usize, end: usize, on_point_done: &mut F) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    for point in start..end {
        let point = u32::try_from(point).map_err(|_| {
            AnnError::log_index_error(format!("LRPF reducer point id {point} exceeds u32"))
        })?;
        on_point_done(point, Vec::new())?;
    }
    Ok(())
}

fn lrpf_part_path_for(base_path: &Path, shard_id: usize) -> PathBuf {
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("lrpf_fragments");
    parent.join(format!("{stem}_shard{shard_id:05}.bin"))
}

#[cfg(test)]
mod tests {
    use super::super::lrpf::{LrpfCandidate, LrpfWitness};
    use super::*;

    #[test]
    fn lrpf_fragment_roundtrip_preserves_candidates_and_witnesses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lrpf_fragments.bin");
        let mut writer = LrpfFragmentWriter::create(&path).unwrap();
        writer
            .push_fragment(
                42,
                &[LrpfCandidate::new(7, 1.0), LrpfCandidate::new(9, 3.0)],
                &[LrpfWitness::new(0, 1, 0.25)],
            )
            .unwrap();
        writer.finish().unwrap();

        let fragments = read_lrpf_fragments(&path).unwrap();

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].src, 42);
        assert_eq!(fragments[0].candidates[0], LrpfCandidate::new(7, 1.0));
        assert_eq!(fragments[0].witnesses[0], LrpfWitness::new(0, 1, 0.25));
    }

    #[test]
    fn lrpf_fragment_header_uses_source_once_per_fragment() {
        assert_eq!(std::mem::size_of::<LrpfFragmentHeader>(), 8);
    }

    #[test]
    fn lrpf_spill_reducer_merges_fragments_and_remaps_witnesses() {
        let dir = tempfile::tempdir().unwrap();
        let sink =
            LrpfFragmentSpillSink::create(&dir.path().join("lrpf_fragments.bin"), 16).unwrap();
        sink.push_lrpf_fragment(
            3,
            &[LrpfCandidate::new(10, 1.0), LrpfCandidate::new(20, 2.0)],
            &[LrpfWitness::new(0, 1, 0.5)],
        )
        .unwrap();
        sink.push_lrpf_fragment(3, &[LrpfCandidate::new(30, 3.0)], &[])
            .unwrap();
        let artifacts = sink.finish().unwrap();

        let mut rows = Vec::new();
        let stats = ExternalLrpfReducer::reduce_spill_artifacts_sharded_in_order_profiled(
            4,
            2,
            1,
            0,
            16,
            &artifacts,
            |point, neighbors| {
                rows.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(rows[3], (3, vec![10, 30]));
        assert_eq!(stats.fragments_scanned, 2);
        assert_eq!(stats.candidate_records, 3);
        assert_eq!(stats.witness_records, 1);
        assert_eq!(stats.reducer_pruned_candidates, 1);
    }

    #[test]
    fn lrpf_digest_sink_merges_fragments_and_bounds_source_width() {
        let dir = tempfile::tempdir().unwrap();
        let sink = LrpfFragmentSpillSink::create_digest(
            &dir.path().join("lrpf_fragments.bin"),
            16,
            LrpfConfig {
                degree: 2,
                protected_core: 1,
                pre_spill_width: 3,
                long_edge_quota: 0,
            },
            1024,
        )
        .unwrap();
        sink.push_lrpf_fragment(
            3,
            &[
                LrpfCandidate::new(10, 1.0),
                LrpfCandidate::new(20, 2.0),
                LrpfCandidate::new(30, 3.0),
            ],
            &[LrpfWitness::new(0, 2, 0.5)],
        )
        .unwrap();
        sink.push_lrpf_fragment(
            3,
            &[LrpfCandidate::new(30, 2.5), LrpfCandidate::new(40, 4.0)],
            &[],
        )
        .unwrap();
        let artifacts = sink.finish().unwrap();
        let fragments = read_lrpf_fragments(&artifacts.part_files[0].path).unwrap();

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].src, 3);
        assert!(fragments[0].candidates.len() <= 3);
        assert!(
            fragments[0]
                .candidates
                .iter()
                .any(|candidate| candidate.dst == 10)
        );
    }

    #[test]
    fn lrpf_digest_flush_threshold_is_shard_local() {
        let dir = tempfile::tempdir().unwrap();
        let sink = LrpfFragmentSpillSink::create_digest(
            &dir.path().join("lrpf_fragments.bin"),
            1,
            LrpfConfig {
                degree: 4,
                protected_core: 1,
                pre_spill_width: 4,
                long_edge_quota: 0,
            },
            64,
        )
        .unwrap();

        sink.push_lrpf_fragment(
            0,
            &[
                LrpfCandidate::new(10, 1.0),
                LrpfCandidate::new(11, 2.0),
                LrpfCandidate::new(12, 3.0),
            ],
            &[],
        )
        .unwrap();
        sink.push_lrpf_fragment(1, &[LrpfCandidate::new(20, 1.0)], &[])
            .unwrap();

        {
            let shards = sink.shards.lock();
            assert_eq!(shards.get(&0).unwrap().lock().desc.fragment_count, 0);
            assert_eq!(shards.get(&1).unwrap().lock().desc.fragment_count, 0);
        }

        sink.push_lrpf_fragment(
            1,
            &[
                LrpfCandidate::new(21, 2.0),
                LrpfCandidate::new(22, 3.0),
                LrpfCandidate::new(23, 4.0),
            ],
            &[],
        )
        .unwrap();

        {
            let shards = sink.shards.lock();
            assert_eq!(shards.get(&0).unwrap().lock().desc.fragment_count, 0);
            assert_eq!(shards.get(&1).unwrap().lock().desc.fragment_count, 1);
        }
    }

    #[test]
    fn lrpf_digest_shard_flush_threshold_keeps_global_cache_bounded() {
        assert_eq!(
            lrpf_digest_shard_flush_threshold_records(0),
            MAX_LRPF_DIGEST_RECORDS / LRPF_DIGEST_SHARD_LOCAL_FLUSH_DIVISOR
        );
        assert_eq!(lrpf_digest_shard_flush_threshold_records(64), 4);
    }

    #[test]
    fn lrpf_digest_sink_deduplicates_witness_pairs_by_dst() {
        let dir = tempfile::tempdir().unwrap();
        let sink = LrpfFragmentSpillSink::create_digest(
            &dir.path().join("lrpf_fragments.bin"),
            16,
            LrpfConfig {
                degree: 4,
                protected_core: 1,
                pre_spill_width: 4,
                long_edge_quota: 0,
            },
            1024,
        )
        .unwrap();
        let candidates = [LrpfCandidate::new(10, 1.0), LrpfCandidate::new(30, 3.0)];
        let witnesses = [LrpfWitness::new(0, 1, 0.5)];
        sink.push_lrpf_fragment(3, &candidates, &witnesses).unwrap();
        sink.push_lrpf_fragment(3, &candidates, &witnesses).unwrap();
        let artifacts = sink.finish().unwrap();
        let fragments = read_lrpf_fragments(&artifacts.part_files[0].path).unwrap();

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].witnesses, vec![LrpfWitness::new(0, 1, 0.5)]);
    }
}
