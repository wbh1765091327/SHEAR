use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::hash_prune::{FinalizedHashPruneRow, HashPruneInsertProfile, HashPruneReservoir};
use super::hash_prune_backfill::{HashPruneBackfillPolicy, HashPruneBackfillProfile};
use super::hash_prune_diagnostics::{HashPruneDiagnosticFinalizer, HashPruneTraceCollector};
use super::hash_prune_evidence::{
    HashPruneEvidencePolicy, HashPruneEvidenceProfile, HashPruneMeta,
};
use super::leaf_build::{
    PendingEdge, PendingEdgeFlushProfile, PendingEdgeSink, edge_sink_microscope_enabled,
    prepare_pending_edges_for_sink,
};
use crate::common::{AlignedBoxWithSlice, AnnError, AnnResult};

pub(crate) const RUN_RECORD_BYTES: usize = std::mem::size_of::<SpillRecord16>();
const DEFAULT_SPILL_CACHE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_SHARD_POINTS: usize = 1_000_000;
const DEFAULT_PART_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const SEGMENT_MANIFEST_NAME: &str = "candidate_segments.manifest";
const DIRECT_CANDIDATE_BUFFER_BYTES: usize = 1024 * 1024;
const EXTERNAL_REDUCE_DECODE_WORKERS: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillRunStats {
    pub part_files: usize,
    pub segment_count: usize,
    pub shard_groups: usize,
    pub max_parts_per_shard: usize,
    pub max_segments_per_shard: usize,
    pub record_bytes: usize,
    pub manifest_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct SpillReduceStats {
    pub shard_groups: usize,
    pub segments_scanned: usize,
    pub part_files_opened: usize,
    pub max_segments_in_group: usize,
    pub logical_read_bytes: u64,
    pub reader_open_calls: usize,
    pub direct_read_calls: usize,
    pub direct_read_bytes: u64,
    pub direct_io: bool,
    pub workspace_points: usize,
    pub workspace_bytes: usize,
    pub workspace_alloc_wall: Duration,
    pub read_decode_wall: Duration,
    pub record_read_wall: Duration,
    pub record_decode_wall: Duration,
    pub record_append_wall: Duration,
    pub insert_wall: Duration,
    pub pipeline_wall: Duration,
    pub pipeline_overlap_wall: Duration,
    pub finalize_emit_wall: Duration,
    pub decoder_workers: usize,
    pub parallel_workers: usize,
    pub parallel_batches: usize,
    pub parallel_records: u64,
    pub batch_peak_records: usize,
    pub batch_peak_bytes: usize,
    pub(crate) evidence: HashPruneEvidenceProfile,
    pub(crate) backfill: HashPruneBackfillProfile,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExternalReduceConfig {
    pub(crate) io: DirectIoConfig,
    pub(crate) initial_workers: usize,
    pub(crate) workers: usize,
    pub(crate) batch_records: usize,
}

impl ExternalReduceConfig {
    pub(crate) fn new(io: DirectIoConfig, workers: usize, batch_records: usize) -> Self {
        Self::with_worker_bounds(io, workers, workers, batch_records)
    }

    pub(crate) fn with_worker_bounds(
        io: DirectIoConfig,
        initial_workers: usize,
        max_workers: usize,
        batch_records: usize,
    ) -> Self {
        let initial_workers = initial_workers.max(1);
        Self {
            io,
            initial_workers,
            workers: max_workers.max(initial_workers),
            batch_records: batch_records.max(1),
        }
    }

    fn serial(io: DirectIoConfig) -> Self {
        Self::new(io, 1, DIRECT_CANDIDATE_BUFFER_BYTES / RUN_RECORD_BYTES)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpillPartFileDesc {
    pub shard_id: usize,
    pub file_id: u16,
    pub path: PathBuf,
    pub byte_len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillSegmentDesc {
    pub shard_id: u32,
    pub file_id: u16,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub record_count: u32,
    pub src_min: u32,
    pub src_max: u32,
    pub sorted_by_src: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpillArtifacts {
    pub part_files: Vec<SpillPartFileDesc>,
    pub segment_manifest: Vec<SpillSegmentDesc>,
    pub manifest_path: PathBuf,
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct SpillRecord16 {
    src: u32,
    dst: u32,
    hash: u16,
    flags: u16,
    dist: f32,
}

impl SpillRecord16 {
    fn from_pending_edge(edge: &PendingEdge) -> AnnResult<Self> {
        let src = u32::try_from(edge.p).map_err(|_| {
            AnnError::log_index_error(format!(
                "Spill writer only supports u32 source ids, got p={}",
                edge.p
            ))
        })?;
        Ok(Self {
            src,
            dst: edge.c,
            hash: edge.hash,
            flags: edge.meta.to_spill_flags()?,
            dist: edge.dist,
        })
    }

    fn to_pending_edge(self) -> AnnResult<PendingEdge> {
        Ok(PendingEdge {
            p: self.src as usize,
            c: self.dst,
            hash: self.hash,
            dist: self.dist,
            meta: HashPruneMeta::from_spill_flags(self.flags)?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct CandidateRunWriter {
    inner: CandidateRunWriterInner,
    records: usize,
    flush_threshold: usize,
}

#[derive(Debug)]
enum CandidateRunWriterInner {
    Buffered(BufWriter<File>),
    Direct {
        file: DirectIoFile,
        offset: u64,
        buffer: Vec<u8>,
    },
}

impl CandidateRunWriter {
    pub(crate) fn create(path: &Path) -> AnnResult<Self> {
        Self::create_with_config(path, DirectIoConfig::disabled())
    }

    pub(crate) fn create_with_config(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        Self::create_with_config_and_buffer(path, io_cfg, DIRECT_CANDIDATE_BUFFER_BYTES)
    }

    pub(crate) fn create_with_config_and_buffer(
        path: &Path,
        io_cfg: DirectIoConfig,
        buffer_bytes: usize,
    ) -> AnnResult<Self> {
        if buffer_bytes < RUN_RECORD_BYTES {
            return Err(AnnError::log_index_config_error(
                "buffer_bytes".to_string(),
                format!("candidate writer buffer must hold at least {RUN_RECORD_BYTES} bytes"),
            ));
        }
        let inner = if io_cfg.enabled {
            CandidateRunWriterInner::Direct {
                file: DirectIoFile::open_rw(path, io_cfg, true)?,
                offset: 0,
                buffer: Vec::with_capacity(buffer_bytes),
            }
        } else {
            CandidateRunWriterInner::Buffered(BufWriter::with_capacity(
                buffer_bytes,
                File::create(path)?,
            ))
        };
        Ok(Self {
            inner,
            records: 0,
            flush_threshold: buffer_bytes,
        })
    }

    pub(crate) fn push(&mut self, edge: &PendingEdge) -> AnnResult<()> {
        let record = SpillRecord16::from_pending_edge(edge)?;
        let mut bytes = [0u8; RUN_RECORD_BYTES];
        bytes[0..4].copy_from_slice(&record.src.to_le_bytes());
        bytes[4..8].copy_from_slice(&record.dst.to_le_bytes());
        bytes[8..10].copy_from_slice(&record.hash.to_le_bytes());
        bytes[10..12].copy_from_slice(&record.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&record.dist.to_le_bytes());
        match &mut self.inner {
            CandidateRunWriterInner::Buffered(writer) => {
                writer.write_all(&bytes)?;
            }
            CandidateRunWriterInner::Direct {
                file,
                offset,
                buffer,
            } => {
                buffer.extend_from_slice(&bytes);
                if buffer.len() >= self.flush_threshold {
                    flush_direct_candidate_buffer(file, offset, buffer)?;
                }
            }
        }
        self.records += 1;
        Ok(())
    }

    pub(crate) fn flush_visible(&mut self) -> AnnResult<()> {
        match &mut self.inner {
            CandidateRunWriterInner::Buffered(writer) => writer.flush()?,
            CandidateRunWriterInner::Direct {
                file,
                offset,
                buffer,
            } => flush_direct_candidate_buffer(file, offset, buffer)?,
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> AnnResult<()> {
        self.flush_visible()?;
        match &mut self.inner {
            CandidateRunWriterInner::Buffered(_) => {}
            CandidateRunWriterInner::Direct {
                file,
                offset,
                buffer: _,
            } => {
                file.set_len(*offset)?;
                file.sync_all()?;
            }
        }
        Ok(())
    }
}

fn flush_direct_candidate_buffer(
    file: &DirectIoFile,
    offset: &mut u64,
    buffer: &mut Vec<u8>,
) -> AnnResult<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    file.write_all_at(buffer, *offset)?;
    *offset += buffer.len() as u64;
    buffer.clear();
    Ok(())
}

#[derive(Debug)]
struct SpillPartWriter {
    desc: SpillPartFileDesc,
    writer: CandidateRunWriter,
}

#[derive(Debug)]
struct SpillShared {
    base_path: PathBuf,
    shard_points: usize,
    io_cfg: DirectIoConfig,
    policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    manifest_path: PathBuf,
    flush_threshold_edges: usize,
    part_max_bytes: u64,
    digest_width: Option<usize>,
    shards: Mutex<HashMap<usize, Arc<Mutex<SpillShardState>>>>,
}

#[derive(Debug)]
struct SpillShardState {
    shard_id: usize,
    buffer: Vec<PendingEdge>,
    digest_rows: HashMap<usize, Vec<PendingEdge>>,
    digest_edge_count: usize,
    part_writer: Option<SpillPartWriter>,
    next_part_id: u16,
    part_files: Vec<SpillPartFileDesc>,
    segment_manifest: Vec<SpillSegmentDesc>,
}

impl SpillShardState {
    fn new(shard_id: usize, buffer_capacity: usize) -> Self {
        Self {
            shard_id,
            buffer: Vec::with_capacity(buffer_capacity),
            digest_rows: HashMap::new(),
            digest_edge_count: 0,
            part_writer: None,
            next_part_id: 0,
            part_files: Vec::new(),
            segment_manifest: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpillEdgeSink {
    inner: Arc<SpillShared>,
}

impl SpillEdgeSink {
    pub fn create(path: &Path, spill_cache_bytes: usize) -> AnnResult<Self> {
        Self::create_sharded_with_config(
            path,
            spill_cache_bytes,
            DEFAULT_SHARD_POINTS,
            DirectIoConfig::disabled(),
        )
    }

    pub fn create_sharded(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config(
            path,
            spill_cache_bytes,
            shard_points,
            DirectIoConfig::disabled(),
        )
    }

    pub fn create_sharded_with_config(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config_and_digest(
            path,
            spill_cache_bytes,
            shard_points,
            io_cfg,
            None,
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
    }

    pub(crate) fn create_sharded_with_config_and_policy(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        io_cfg: DirectIoConfig,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config_and_policies(
            path,
            spill_cache_bytes,
            shard_points,
            io_cfg,
            policy,
            HashPruneBackfillPolicy::off(),
        )
    }

    pub(crate) fn create_sharded_with_config_and_policies(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        io_cfg: DirectIoConfig,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config_and_digest(
            path,
            spill_cache_bytes,
            shard_points,
            io_cfg,
            None,
            policy,
            backfill_policy,
        )
    }

    pub fn create_digest_sharded_with_config(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        digest_width: usize,
        io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config_and_digest(
            path,
            spill_cache_bytes,
            shard_points,
            io_cfg,
            Some(digest_width.max(1)),
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
    }

    pub(crate) fn create_digest_sharded_with_config_and_policy(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        digest_width: usize,
        io_cfg: DirectIoConfig,
        policy: HashPruneEvidencePolicy,
    ) -> AnnResult<Self> {
        Self::create_digest_sharded_with_config_and_policies(
            path,
            spill_cache_bytes,
            shard_points,
            digest_width,
            io_cfg,
            policy,
            HashPruneBackfillPolicy::off(),
        )
    }

    pub(crate) fn create_digest_sharded_with_config_and_policies(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        digest_width: usize,
        io_cfg: DirectIoConfig,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config_and_digest(
            path,
            spill_cache_bytes,
            shard_points,
            io_cfg,
            Some(digest_width.max(1)),
            policy,
            backfill_policy,
        )
    }

    fn create_sharded_with_config_and_digest(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        io_cfg: DirectIoConfig,
        digest_width: Option<usize>,
        policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Self> {
        if !policy.is_off() && !backfill_policy.is_off() {
            return Err(AnnError::log_index_error(
                "distance-gated backfill cannot share the Evidence-Aware spill path".to_string(),
            ));
        }
        let spill_cache_bytes = effective_spill_cache_bytes(spill_cache_bytes);
        let estimated_edge_bytes = std::mem::size_of::<PendingEdge>().max(1);
        let flush_threshold_edges = (spill_cache_bytes / estimated_edge_bytes)
            .max(1)
            .min(1_000_000);
        Ok(Self {
            inner: Arc::new(SpillShared {
                base_path: path.to_path_buf(),
                shard_points: shard_points.max(1),
                io_cfg,
                policy,
                backfill_policy,
                manifest_path: path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(SEGMENT_MANIFEST_NAME),
                flush_threshold_edges,
                part_max_bytes: DEFAULT_PART_MAX_BYTES,
                digest_width,
                shards: Mutex::new(HashMap::new()),
            }),
        })
    }

    pub fn finish(self) -> AnnResult<SpillArtifacts> {
        let shared = self.inner;
        let mut shard_states = {
            let shards = shared.shards.lock();
            shards
                .iter()
                .map(|(&shard_id, state)| (shard_id, Arc::clone(state)))
                .collect::<Vec<_>>()
        };
        shard_states.sort_unstable_by_key(|(shard_id, _)| *shard_id);

        let mut part_files = Vec::new();
        let mut segment_manifest = Vec::new();
        for (_, shard_state) in shard_states {
            let mut shard = shard_state.lock();
            flush_shard_buffer_to_part(&shared, &mut shard)?;
            flush_shard_digest_to_part(&shared, &mut shard)?;
            finish_shard_part_writer(&mut shard)?;
            part_files.extend(shard.part_files.clone());
            segment_manifest.extend(shard.segment_manifest.clone());
        }
        part_files.sort_unstable_by_key(|part| (part.shard_id, part.file_id));
        segment_manifest.sort_unstable_by_key(|segment| {
            (segment.shard_id, segment.file_id, segment.byte_offset)
        });
        write_segment_manifest(&shared.manifest_path, &segment_manifest)?;
        Ok(SpillArtifacts {
            part_files,
            segment_manifest,
            manifest_path: shared.manifest_path.clone(),
            policy: shared.policy,
            backfill_policy: shared.backfill_policy,
        })
    }
}

impl PendingEdgeSink for SpillEdgeSink {
    fn hashprune_evidence_policy(&self) -> HashPruneEvidencePolicy {
        self.inner.policy
    }

    fn hashprune_backfill_policy(&self) -> HashPruneBackfillPolicy {
        self.inner.backfill_policy
    }

    fn flush_pending_edges(
        &self,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<PendingEdgeFlushProfile> {
        if edges.is_empty() {
            return Ok(PendingEdgeFlushProfile::default());
        }

        let edge_count = edges.len();
        let sort_start = std::time::Instant::now();
        let mut evidence = prepare_pending_edges_for_sink(edges, self.inner.policy)?;
        let sort_wall = sort_start.elapsed();
        let removed = evidence.duplicate_records_removed;
        evidence.spill_records_avoided = evidence.spill_records_avoided.saturating_add(removed);
        evidence.spill_bytes_avoided = evidence
            .spill_bytes_avoided
            .saturating_add(removed.saturating_mul(RUN_RECORD_BYTES));
        let insert_start = std::time::Instant::now();
        let shard_points = self.inner.shard_points;
        let mut current_shard_id = None;
        let mut current_edges = Vec::new();
        for edge in edges.drain(..) {
            let shard_id = edge.p / shard_points;
            if current_shard_id != Some(shard_id) {
                if let Some(flush_shard_id) = current_shard_id {
                    flush_edges_to_spill_shard(&self.inner, flush_shard_id, &mut current_edges)?;
                }
                current_shard_id = Some(shard_id);
            }
            current_edges.push(edge);
        }
        if let Some(shard_id) = current_shard_id {
            flush_edges_to_spill_shard(&self.inner, shard_id, &mut current_edges)?;
        }
        let insert_wall = insert_start.elapsed();
        Ok(PendingEdgeFlushProfile {
            edges: edge_count,
            sort_wall,
            insert_wall,
            insert_body_wall: insert_wall,
            flushes: 1,
            evidence,
            ..PendingEdgeFlushProfile::default()
        })
    }
}

fn flush_edges_to_spill_shard(
    shared: &SpillShared,
    shard_id: usize,
    edges: &mut Vec<PendingEdge>,
) -> AnnResult<()> {
    if edges.is_empty() {
        return Ok(());
    }
    let shard = get_or_create_spill_shard(shared, shard_id);
    let mut shard = shard.lock();
    if let Some(digest_width) = shared.digest_width {
        insert_edges_to_spill_digest(&mut shard, edges, digest_width)?;
        if shard.digest_edge_count >= shared.flush_threshold_edges {
            flush_shard_digest_to_part(shared, &mut shard)?;
        }
    } else {
        shard.buffer.append(edges);
        if shard.buffer.len() >= shared.flush_threshold_edges {
            flush_shard_buffer_to_part(shared, &mut shard)?;
        }
    }
    Ok(())
}

fn get_or_create_spill_shard(shared: &SpillShared, shard_id: usize) -> Arc<Mutex<SpillShardState>> {
    let mut shards = shared.shards.lock();
    Arc::clone(shards.entry(shard_id).or_insert_with(|| {
        Arc::new(Mutex::new(SpillShardState::new(
            shard_id,
            shared.flush_threshold_edges.min(65_536),
        )))
    }))
}

fn flush_shard_buffer_to_part(shared: &SpillShared, shard: &mut SpillShardState) -> AnnResult<()> {
    if shard.buffer.is_empty() {
        return Ok(());
    }

    shard.buffer.sort_unstable_by_key(|edge| edge.p);
    let staged: Vec<PendingEdge> = std::mem::take(&mut shard.buffer);
    write_staged_edges_to_part(shared, shard, staged)
}

fn insert_edges_to_spill_digest(
    shard: &mut SpillShardState,
    edges: &mut Vec<PendingEdge>,
    digest_width: usize,
) -> AnnResult<()> {
    for edge in edges.drain(..) {
        insert_edge_to_digest_row(shard, edge, digest_width)?;
    }
    Ok(())
}

fn insert_edge_to_digest_row(
    shard: &mut SpillShardState,
    edge: PendingEdge,
    digest_width: usize,
) -> AnnResult<()> {
    let row = shard.digest_rows.entry(edge.p).or_default();
    if let Some(existing) = row.iter_mut().find(|candidate| candidate.c == edge.c) {
        if existing.hash != edge.hash {
            return Err(AnnError::log_index_error(format!(
                "HashPrune spill digest duplicate hash mismatch for source {} destination {}: left={} right={}",
                edge.p, edge.c, existing.hash, edge.hash
            )));
        }
        let merged_meta = existing.meta.merge_candidate(edge.meta)?;
        if edge.dist.total_cmp(&existing.dist).is_lt() {
            existing.dist = edge.dist;
        }
        existing.meta = merged_meta;
        return Ok(());
    }

    if row.len() < digest_width {
        row.push(edge);
        shard.digest_edge_count += 1;
        return Ok(());
    }

    let Some((worst_idx, worst_edge)) = row
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| digest_candidate_order(left, right))
    else {
        return Ok(());
    };
    if digest_candidate_is_better(&edge, worst_edge) {
        row[worst_idx] = edge;
    }
    Ok(())
}

fn flush_shard_digest_to_part(shared: &SpillShared, shard: &mut SpillShardState) -> AnnResult<()> {
    if shard.digest_edge_count == 0 {
        return Ok(());
    }

    let mut rows: Vec<_> = shard.digest_rows.drain().collect();
    rows.sort_unstable_by_key(|(src, _)| *src);
    let mut staged = Vec::with_capacity(shard.digest_edge_count);
    for (_, mut row) in rows {
        row.sort_by(digest_candidate_order);
        staged.extend(row);
    }
    shard.digest_edge_count = 0;
    write_staged_edges_to_part(shared, shard, staged)
}

fn write_staged_edges_to_part(
    shared: &SpillShared,
    shard: &mut SpillShardState,
    staged: Vec<PendingEdge>,
) -> AnnResult<()> {
    if staged.is_empty() {
        return Ok(());
    }
    let src_min = staged.first().map(|edge| edge.p as u32).unwrap_or(0);
    let src_max = staged.last().map(|edge| edge.p as u32).unwrap_or(0);
    let byte_len = (staged.len() * RUN_RECORD_BYTES) as u64;
    ensure_part_writer(shared, shard, byte_len)?;
    let part = shard.part_writer.as_mut().unwrap();
    let byte_offset = part.desc.byte_len;
    for edge in staged.iter() {
        part.writer.push(edge)?;
    }
    part.desc.byte_len += byte_len;
    shard.segment_manifest.push(SpillSegmentDesc {
        shard_id: shard.shard_id as u32,
        file_id: part.desc.file_id,
        byte_offset,
        byte_len,
        record_count: staged.len() as u32,
        src_min,
        src_max,
        sorted_by_src: true,
    });
    Ok(())
}

fn digest_candidate_is_better(candidate: &PendingEdge, current: &PendingEdge) -> bool {
    digest_candidate_order(candidate, current).is_lt()
}

fn digest_candidate_order(left: &PendingEdge, right: &PendingEdge) -> std::cmp::Ordering {
    right
        .meta
        .is_mandatory()
        .cmp(&left.meta.is_mandatory())
        .then_with(|| left.dist.total_cmp(&right.dist))
        .then_with(|| left.c.cmp(&right.c))
        .then_with(|| left.hash.cmp(&right.hash))
}

fn ensure_part_writer(
    shared: &SpillShared,
    shard: &mut SpillShardState,
    pending_bytes: u64,
) -> AnnResult<()> {
    let need_rollover = shard
        .part_writer
        .as_ref()
        .map(|part| part.desc.byte_len.saturating_add(pending_bytes) > shared.part_max_bytes)
        .unwrap_or(true);
    if !need_rollover {
        return Ok(());
    }

    if let Some(part) = shard.part_writer.take() {
        finalize_part_writer(shard, part)?;
    }

    let file_id = {
        let file_id = shard.next_part_id;
        shard.next_part_id = shard.next_part_id.saturating_add(1);
        file_id
    };
    let path = part_path_for(&shared.base_path, file_id as usize, shard.shard_id);
    let writer = CandidateRunWriter::create_with_config(&path, shared.io_cfg)?;
    shard.part_writer = Some(SpillPartWriter {
        desc: SpillPartFileDesc {
            shard_id: shard.shard_id,
            file_id,
            path,
            byte_len: 0,
        },
        writer,
    });
    Ok(())
}

fn finalize_part_writer(shard: &mut SpillShardState, part: SpillPartWriter) -> AnnResult<()> {
    let SpillPartWriter { desc, writer } = part;
    writer.finish()?;
    shard.part_files.push(desc);
    Ok(())
}

fn finish_shard_part_writer(shard: &mut SpillShardState) -> AnnResult<()> {
    if let Some(part) = shard.part_writer.take() {
        finalize_part_writer(shard, part)?;
    }
    Ok(())
}

fn part_path_for(base_path: &Path, part_index: usize, shard_id: usize) -> PathBuf {
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("candidate");
    parent.join(format!("{stem}_shard{shard_id:05}_part{part_index:03}.log"))
}

fn write_segment_manifest(path: &Path, segment_manifest: &[SpillSegmentDesc]) -> AnnResult<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for segment in segment_manifest {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{}",
            segment.shard_id,
            segment.file_id,
            segment.byte_offset,
            segment.byte_len,
            segment.record_count,
            segment.src_min,
            segment.src_max,
            u8::from(segment.sorted_by_src)
        )?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Default)]
pub struct ExternalHashPruneReducer;

impl ExternalHashPruneReducer {
    pub fn reduce_spill_artifacts(
        num_points: usize,
        l_max: usize,
        artifacts: &SpillArtifacts,
    ) -> AnnResult<HashMap<u32, Vec<u32>>> {
        let mut reduced = HashMap::with_capacity(num_points);
        Self::reduce_spill_artifacts_sharded_in_order(
            num_points,
            l_max,
            num_points.max(1),
            artifacts,
            |point, neighbors| {
                reduced.insert(point, neighbors);
                Ok(())
            },
        )?;
        Ok(reduced)
    }

    pub fn reduce_spill_artifacts_ordered(
        num_points: usize,
        l_max: usize,
        artifacts: &SpillArtifacts,
    ) -> AnnResult<Vec<Vec<u32>>> {
        let mut ordered = vec![Vec::new(); num_points];
        Self::reduce_spill_artifacts_sharded_in_order(
            num_points,
            l_max,
            num_points.max(1),
            artifacts,
            |point, neighbors| {
                ordered[point as usize] = neighbors;
                Ok(())
            },
        )?;
        Ok(ordered)
    }

    pub fn reduce_spill_artifacts_sharded_in_order<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_with_limit(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
        .map(|_| ())
    }

    pub fn reduce_spill_artifacts_sharded_in_order_from<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_from_with_limit(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
        .map(|_| ())
    }

    pub fn file_point_counts(run_files: &[PathBuf]) -> AnnResult<BTreeMap<PathBuf, usize>> {
        let mut counts = BTreeMap::new();
        for path in run_files {
            let mut reader = BufReader::new(File::open(path)?);
            let mut count = 0usize;
            loop {
                let mut src_bytes = [0u8; 4];
                match reader.read_exact(&mut src_bytes) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(err) => return Err(err.into()),
                }

                let mut skip = [0u8; RUN_RECORD_BYTES - 4];
                reader.read_exact(&mut skip)?;
                count += 1;
            }
            counts.insert(path.clone(), count);
        }
        Ok(counts)
    }

    pub fn summarize_spill_artifacts(artifacts: &SpillArtifacts) -> SpillRunStats {
        summarize_spill_artifacts(artifacts)
    }

    pub fn reduce_spill_artifacts_sharded_in_order_profiled<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_with_limit(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_with_policy<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        policy: HashPruneEvidencePolicy,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_with_policies(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            policy,
            HashPruneBackfillPolicy::off(),
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_with_policies<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_with_policies_and_diagnostics(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            evidence_policy,
            backfill_policy,
            None,
            None,
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_with_policies_and_diagnostics<
        F,
    >(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_with_limit_and_policies(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            diagnostic_finalizer,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_with_config<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        io_cfg: DirectIoConfig,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            diagnostic_finalizer,
            ExternalReduceConfig::serial(io_cfg),
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        config: ExternalReduceConfig,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        let microscope = !evidence_policy.is_off() && edge_sink_microscope_enabled();
        if let Some(replay_mode) = external_parallel_replay_mode(
            config,
            evidence_policy,
            backfill_policy,
            diagnostic_collector.is_some() || diagnostic_finalizer.is_some(),
        ) {
            return reduce_segments_sharded_in_order_from_parallel_backfill(
                num_points,
                0,
                l_max,
                points_per_shard,
                artifacts,
                evidence_policy,
                backfill_policy,
                config,
                replay_mode,
                on_point_done,
            );
        }
        reduce_segments_sharded_in_order_from_with_limit_and_policies_core(
            num_points,
            0,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            diagnostic_finalizer,
            config.io,
            microscope,
            |_, _: &HashPruneReservoir| {},
            on_point_done,
        )
        .map(|mut stats| {
            stats.parallel_workers = 1;
            stats
        })
    }

    #[cfg(test)]
    pub(crate) fn reduce_spill_to_state_snapshots_for_test(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
    ) -> AnnResult<Vec<super::hash_prune::HashPruneReservoirState>> {
        let empty = HashPruneReservoir::new(l_max).state_snapshot();
        let mut snapshots = vec![empty; num_points];
        reduce_segments_sharded_in_order_from_with_limit_and_policies_core(
            num_points,
            0,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            evidence_policy,
            backfill_policy,
            None,
            None,
            DirectIoConfig::disabled(),
            false,
            |source, reservoir| {
                snapshots[source as usize] = reservoir.state_snapshot();
            },
            |_, _| Ok(()),
        )?;
        Ok(snapshots)
    }

    pub fn reduce_spill_artifacts_sharded_in_order_profiled_from<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_from_with_limit(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_from_with_policy<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        policy: HashPruneEvidencePolicy,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_from_with_policies(
            point_start,
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            policy,
            HashPruneBackfillPolicy::off(),
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_from_with_policies<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_from_with_policies_and_diagnostics(
            point_start,
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            evidence_policy,
            backfill_policy,
            None,
            None,
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_from_with_policies_and_diagnostics<
        F,
    >(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_from_with_limit_and_policies(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            diagnostic_finalizer,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_from_with_config<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        io_cfg: DirectIoConfig,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        Self::reduce_spill_artifacts_sharded_in_order_profiled_from_with_execution_config(
            point_start,
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            diagnostic_finalizer,
            ExternalReduceConfig::serial(io_cfg),
            on_point_done,
        )
    }

    pub(crate) fn reduce_spill_artifacts_sharded_in_order_profiled_from_with_execution_config<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        evidence_policy: HashPruneEvidencePolicy,
        backfill_policy: HashPruneBackfillPolicy,
        diagnostic_collector: Option<&HashPruneTraceCollector>,
        diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
        config: ExternalReduceConfig,
        on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        let microscope = !evidence_policy.is_off() && edge_sink_microscope_enabled();
        if let Some(replay_mode) = external_parallel_replay_mode(
            config,
            evidence_policy,
            backfill_policy,
            diagnostic_collector.is_some() || diagnostic_finalizer.is_some(),
        ) {
            return reduce_segments_sharded_in_order_from_parallel_backfill(
                num_points,
                point_start,
                l_max,
                points_per_shard,
                artifacts,
                evidence_policy,
                backfill_policy,
                config,
                replay_mode,
                on_point_done,
            );
        }
        reduce_segments_sharded_in_order_from_with_limit_and_policies_core(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            evidence_policy,
            backfill_policy,
            diagnostic_collector,
            diagnostic_finalizer,
            config.io,
            microscope,
            |_, _: &HashPruneReservoir| {},
            on_point_done,
        )
        .map(|mut stats| {
            stats.parallel_workers = 1;
            stats
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExternalParallelReplayMode {
    CompactFused,
    DecodedPipeline,
}

fn external_parallel_replay_mode(
    config: ExternalReduceConfig,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostics_enabled: bool,
) -> Option<ExternalParallelReplayMode> {
    if config.workers <= 1
        || !evidence_policy.is_off()
        || !backfill_policy.is_rhalf()
        || diagnostics_enabled
    {
        return None;
    }
    Some(if config.io.enabled {
        ExternalParallelReplayMode::DecodedPipeline
    } else {
        ExternalParallelReplayMode::CompactFused
    })
}

fn source_disjoint_edge_ranges(edges: &[PendingEdge], workers: usize) -> Vec<Range<usize>> {
    if edges.is_empty() {
        return Vec::new();
    }

    let target = edges.len().div_ceil(workers.max(1));
    let mut ranges = Vec::with_capacity(workers.min(edges.len()));
    let mut start = 0;
    while start < edges.len() {
        let mut end = (start + target).min(edges.len());
        while end < edges.len() && edges[end - 1].p == edges[end].p {
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

#[inline]
fn compact_record_source(records: &[u8], record: usize) -> usize {
    let start = record * RUN_RECORD_BYTES;
    u32::from_le_bytes(records[start..start + 4].try_into().unwrap()) as usize
}

fn validate_sorted_compact_record_sources(
    last_source: &mut Option<usize>,
    records: &[u8],
) -> AnnResult<()> {
    debug_assert_eq!(records.len() % RUN_RECORD_BYTES, 0);
    let mut previous = *last_source;
    for record in 0..records.len() / RUN_RECORD_BYTES {
        let source = compact_record_source(records, record);
        if let Some(previous) = previous {
            if previous > source {
                return Err(AnnError::log_index_error(format!(
                    "Spill segment batch is not sorted by source: {previous} before {source}"
                )));
            }
        }
        previous = Some(source);
    }
    *last_source = previous;
    Ok(())
}

fn compact_record_partition_point(records: &[u8], source: usize) -> usize {
    debug_assert_eq!(records.len() % RUN_RECORD_BYTES, 0);
    let mut left = 0;
    let mut right = records.len() / RUN_RECORD_BYTES;
    while left < right {
        let middle = left + (right - left) / 2;
        if compact_record_source(records, middle) < source {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    left
}

fn source_disjoint_compact_record_ranges(records: &[u8], workers: usize) -> Vec<Range<usize>> {
    debug_assert_eq!(records.len() % RUN_RECORD_BYTES, 0);
    let record_count = records.len() / RUN_RECORD_BYTES;
    if record_count == 0 {
        return Vec::new();
    }

    let target = record_count.div_ceil(workers.max(1));
    let mut ranges = Vec::with_capacity(workers.min(record_count));
    let mut start = 0;
    while start < record_count {
        let mut end = (start + target).min(record_count);
        while end < record_count
            && compact_record_source(records, end - 1) == compact_record_source(records, end)
        {
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

fn next_external_reduce_workers(
    min_workers: usize,
    current_workers: usize,
    max_workers: usize,
    insert_wall: Duration,
    read_wall: Duration,
) -> usize {
    let min_workers = min_workers.max(1);
    let current_workers = current_workers.max(min_workers);
    let max_workers = max_workers.max(current_workers);
    if current_workers < max_workers
        && insert_wall.as_nanos() > read_wall.as_nanos().saturating_mul(2)
    {
        current_workers.saturating_mul(2).min(max_workers)
    } else if current_workers > min_workers
        && read_wall.as_nanos() > insert_wall.as_nanos().saturating_mul(2)
    {
        (current_workers / 2).max(min_workers)
    } else {
        current_workers
    }
}

const EXTERNAL_REDUCE_REBALANCE_BATCHES: usize = 64;

struct ExternalReduceWorkerControl {
    min_workers: usize,
    current_workers: usize,
    max_workers: usize,
    observed_batches: usize,
    insert_wall: Duration,
    read_wall: Duration,
}

impl ExternalReduceWorkerControl {
    fn new(initial_workers: usize, max_workers: usize) -> Self {
        let initial_workers = initial_workers.max(1);
        Self {
            min_workers: initial_workers,
            current_workers: initial_workers,
            max_workers: max_workers.max(initial_workers),
            observed_batches: 0,
            insert_wall: Duration::ZERO,
            read_wall: Duration::ZERO,
        }
    }

    fn workers(&self) -> usize {
        self.current_workers
    }

    fn observe(&mut self, insert_wall: Duration, read_wall: Duration) -> usize {
        self.observed_batches += 1;
        self.insert_wall = self.insert_wall.saturating_add(insert_wall);
        self.read_wall = self.read_wall.saturating_add(read_wall);
        if self.observed_batches < EXTERNAL_REDUCE_REBALANCE_BATCHES {
            return self.current_workers;
        }

        self.current_workers = next_external_reduce_workers(
            self.min_workers,
            self.current_workers,
            self.max_workers,
            self.insert_wall,
            self.read_wall,
        );
        self.observed_batches = 0;
        self.insert_wall = Duration::ZERO;
        self.read_wall = Duration::ZERO;
        self.current_workers
    }
}

struct ExternalReduceThreadPool {
    inner: rayon::ThreadPool,
}

impl ExternalReduceThreadPool {
    fn new(workers: usize) -> AnnResult<Self> {
        let workers = workers.max(1);
        let inner = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|error| {
                AnnError::log_index_error(format!(
                    "Failed to create external-reduce Rayon pool: {error}"
                ))
            })?;
        Ok(Self { inner })
    }

    fn workers(&self) -> usize {
        self.inner.current_num_threads()
    }

    fn pool(&self) -> &rayon::ThreadPool {
        &self.inner
    }

    fn resize(&mut self, workers: usize) -> AnnResult<()> {
        let workers = workers.max(1);
        if workers != self.workers() {
            *self = Self::new(workers)?;
        }
        Ok(())
    }
}

struct ParallelBackfillTask<'a> {
    source_start: usize,
    reservoirs: &'a mut [HashPruneReservoir],
    edges: &'a [PendingEdge],
    outcome: Option<AnnResult<HashPruneBackfillProfile>>,
    completed_after_submit: Duration,
}

impl ParallelBackfillTask<'_> {
    fn run(&mut self, policy: HashPruneBackfillPolicy, submitted: Instant) {
        let mut profile = HashPruneBackfillProfile::default();
        let result = (|| {
            for edge in self.edges {
                self.reservoirs[edge.p - self.source_start]
                    .insert_backfill_prevalidated_with_policy(
                        edge.p,
                        edge.c,
                        edge.hash,
                        edge.dist,
                        policy,
                        &mut profile,
                    )?;
            }
            Ok(profile)
        })();
        self.completed_after_submit = submitted.elapsed();
        self.outcome = Some(result);
    }
}

struct CompactBackfillTask<'a> {
    source_start: usize,
    reservoirs: &'a mut [HashPruneReservoir],
    records: &'a [u8],
    outcome: Option<AnnResult<HashPruneBackfillProfile>>,
    completed_after_submit: Duration,
}

impl CompactBackfillTask<'_> {
    fn run(&mut self, policy: HashPruneBackfillPolicy, submitted: Instant) {
        let mut profile = HashPruneBackfillProfile::default();
        let result = (|| {
            for bytes in self.records.chunks_exact(RUN_RECORD_BYTES) {
                let edge = decode_spill_record16(bytes)?;
                let local_source = edge.p.checked_sub(self.source_start).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Compact spill task source {} precedes task start {}",
                        edge.p, self.source_start
                    ))
                })?;
                let task_end = self.source_start + self.reservoirs.len();
                let reservoir = self.reservoirs.get_mut(local_source).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Compact spill task source {} exceeds task range [{}, {})",
                        edge.p, self.source_start, task_end
                    ))
                })?;
                reservoir.insert_backfill_prevalidated_with_policy(
                    edge.p,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    policy,
                    &mut profile,
                )?;
            }
            Ok(profile)
        })();
        self.completed_after_submit = submitted.elapsed();
        self.outcome = Some(result);
    }
}

fn build_parallel_backfill_tasks<'a>(
    point_start: usize,
    point_end: usize,
    reservoirs: &'a mut [HashPruneReservoir],
    edges: &'a [PendingEdge],
    workers: usize,
) -> AnnResult<Vec<ParallelBackfillTask<'a>>> {
    let ranges = source_disjoint_edge_ranges(edges, workers);
    let mut tasks = Vec::with_capacity(ranges.len());
    let mut remaining = reservoirs;
    let mut remaining_start = point_start;

    for range in ranges {
        let task_edges = &edges[range];
        let task_start = task_edges[0].p;
        let task_end = task_edges.last().unwrap().p.checked_add(1).ok_or_else(|| {
            AnnError::log_index_error("Spill reducer source range overflows usize".to_string())
        })?;
        if task_start < remaining_start || task_end > point_end {
            return Err(AnnError::log_index_error(format!(
                "Spill reducer encountered source range [{task_start}, {task_end}) outside [{point_start}, {point_end})"
            )));
        }

        let skip = task_start - remaining_start;
        let (_, from_task_start) = remaining.split_at_mut(skip);
        let (task_reservoirs, tail) = from_task_start.split_at_mut(task_end - task_start);
        tasks.push(ParallelBackfillTask {
            source_start: task_start,
            reservoirs: task_reservoirs,
            edges: task_edges,
            outcome: None,
            completed_after_submit: Duration::ZERO,
        });
        remaining = tail;
        remaining_start = task_end;
    }
    Ok(tasks)
}

fn build_compact_backfill_tasks<'a>(
    point_start: usize,
    point_end: usize,
    reservoirs: &'a mut [HashPruneReservoir],
    records: &'a [u8],
    workers: usize,
) -> AnnResult<Vec<CompactBackfillTask<'a>>> {
    let ranges = source_disjoint_compact_record_ranges(records, workers);
    let mut tasks = Vec::with_capacity(ranges.len());
    let mut remaining = reservoirs;
    let mut remaining_start = point_start;

    for range in ranges {
        let task_records = &records[range.start * RUN_RECORD_BYTES..range.end * RUN_RECORD_BYTES];
        let task_start = compact_record_source(task_records, 0);
        let task_end = compact_record_source(task_records, range.len() - 1)
            .checked_add(1)
            .ok_or_else(|| {
                AnnError::log_index_error("Spill reducer source range overflows usize".to_string())
            })?;
        if task_start < remaining_start || task_end > point_end {
            return Err(AnnError::log_index_error(format!(
                "Compact spill reducer encountered source range [{task_start}, {task_end}) outside [{point_start}, {point_end})"
            )));
        }

        let skip = task_start - remaining_start;
        let (_, from_task_start) = remaining.split_at_mut(skip);
        let (task_reservoirs, tail) = from_task_start.split_at_mut(task_end - task_start);
        tasks.push(CompactBackfillTask {
            source_start: task_start,
            reservoirs: task_reservoirs,
            records: task_records,
            outcome: None,
            completed_after_submit: Duration::ZERO,
        });
        remaining = tail;
        remaining_start = task_end;
    }
    Ok(tasks)
}

fn insert_parallel_backfill_batch_while<T, F>(
    pool: &rayon::ThreadPool,
    point_start: usize,
    point_end: usize,
    reservoirs: &mut [HashPruneReservoir],
    edges: &[PendingEdge],
    workers: usize,
    policy: HashPruneBackfillPolicy,
    profile: &mut HashPruneBackfillProfile,
    while_inserting: F,
) -> AnnResult<(T, Duration)>
where
    F: FnOnce() -> T,
{
    let submitted = Instant::now();
    let mut tasks =
        build_parallel_backfill_tasks(point_start, point_end, reservoirs, edges, workers)?;
    let concurrent_result = pool.in_place_scope(|scope| {
        for task in &mut tasks {
            scope.spawn(move |_| task.run(policy, submitted));
        }
        while_inserting()
    });
    let insert_wall = tasks
        .iter()
        .map(|task| task.completed_after_submit)
        .max()
        .unwrap_or_default();
    for task in tasks {
        let task_profile = task
            .outcome
            .expect("scoped external-reduce task must complete")?;
        profile.merge(task_profile);
    }
    Ok((concurrent_result, insert_wall))
}

fn insert_compact_backfill_batch_while<T, F>(
    pool: &rayon::ThreadPool,
    point_start: usize,
    point_end: usize,
    reservoirs: &mut [HashPruneReservoir],
    records: &[u8],
    workers: usize,
    policy: HashPruneBackfillPolicy,
    profile: &mut HashPruneBackfillProfile,
    while_inserting: F,
) -> AnnResult<(T, Duration)>
where
    F: FnOnce() -> T,
{
    let submitted = Instant::now();
    let mut tasks =
        build_compact_backfill_tasks(point_start, point_end, reservoirs, records, workers)?;
    let concurrent_result = pool.in_place_scope(|scope| {
        for task in &mut tasks {
            scope.spawn(move |_| task.run(policy, submitted));
        }
        while_inserting()
    });
    let insert_wall = tasks
        .iter()
        .map(|task| task.completed_after_submit)
        .max()
        .unwrap_or_default();
    for task in tasks {
        let task_profile = task
            .outcome
            .expect("scoped compact external-reduce task must complete")?;
        profile.merge(task_profile);
    }
    Ok((concurrent_result, insert_wall))
}

fn replay_segment_group_parallel_backfill<F>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    policy: HashPruneBackfillPolicy,
    reservoirs: &mut Vec<HashPruneReservoir>,
    config: ExternalReduceConfig,
    decoder_pool: &rayon::ThreadPool,
    pool: &mut ExternalReduceThreadPool,
    replay_io: &mut SpillReplayIoStats,
    stats: &mut SpillReduceStats,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if point_start >= point_end {
        return Ok(());
    }
    resize_or_reset_reservoirs(reservoirs, point_end - point_start, l_max);

    let mut file_lookup = BTreeMap::new();
    for part in &artifacts.part_files {
        file_lookup.insert((part.shard_id, part.file_id), part.path.clone());
    }
    let mut segments_by_part: BTreeMap<(usize, u16), (PathBuf, Vec<SpillSegmentDesc>)> =
        BTreeMap::new();
    for segment in &group.segments {
        let key = (segment.shard_id as usize, segment.file_id);
        let Some(path) = file_lookup.get(&key) else {
            return Err(AnnError::log_index_error(format!(
                "Missing spill part for shard={} file_id={}",
                segment.shard_id, segment.file_id
            )));
        };
        segments_by_part
            .entry(key)
            .or_insert_with(|| (path.clone(), Vec::new()))
            .1
            .push(*segment);
    }

    for (_, (path, segments)) in segments_by_part {
        let mut cursor = PartBatchCursor::open(&path, segments, config.io)?;
        replay_io.reader_open_calls += 1;
        let mut current = Vec::with_capacity(config.batch_records);
        let mut next = Vec::with_capacity(config.batch_records);
        let mut worker_control =
            ExternalReduceWorkerControl::new(config.initial_workers, config.workers);
        let mut decode_scratch = ParallelDecodeScratch::default();
        let part_pipeline_start = Instant::now();
        let read_before = stats.read_decode_wall;
        let insert_before = stats.insert_wall;

        let prime_start = Instant::now();
        let mut current_count = cursor.read_next_batch_parallel_decode(
            config.batch_records,
            &mut current,
            replay_io,
            decoder_pool,
            &mut decode_scratch,
        )?;
        stats.read_decode_wall += prime_start.elapsed();

        while current_count > 0 {
            stats.batch_peak_records = stats.batch_peak_records.max(current_count);
            stats.batch_peak_bytes = stats
                .batch_peak_bytes
                .max(current_count.saturating_mul(std::mem::size_of::<PendingEdge>()));
            let valid_start = current.partition_point(|edge| edge.p < point_start);
            let valid_edges = &current[valid_start..];

            let next_read = if valid_edges.is_empty() {
                let read_start = Instant::now();
                let result = cursor.read_next_batch_parallel_decode(
                    config.batch_records,
                    &mut next,
                    replay_io,
                    decoder_pool,
                    &mut decode_scratch,
                );
                (result, read_start.elapsed())
            } else {
                let active_workers = worker_control.workers();
                let ((result, read_wall), insert_wall) = insert_parallel_backfill_batch_while(
                    pool.pool(),
                    point_start,
                    point_end,
                    &mut reservoirs[..point_end - point_start],
                    valid_edges,
                    active_workers,
                    policy,
                    &mut stats.backfill,
                    || {
                        let read_start = Instant::now();
                        let result = cursor.read_next_batch_parallel_decode(
                            config.batch_records,
                            &mut next,
                            replay_io,
                            decoder_pool,
                            &mut decode_scratch,
                        );
                        (result, read_start.elapsed())
                    },
                )?;
                stats.insert_wall += insert_wall;
                stats.parallel_batches += 1;
                stats.parallel_records = stats
                    .parallel_records
                    .saturating_add(valid_edges.len() as u64);
                stats.parallel_workers = stats.parallel_workers.max(active_workers);
                let requested_workers = worker_control.observe(insert_wall, read_wall);
                pool.resize(requested_workers)?;
                (result, read_wall)
            };

            stats.read_decode_wall += next_read.1;
            let next_count = next_read.0?;
            std::mem::swap(&mut current, &mut next);
            current_count = next_count;
        }

        let reader_stats = cursor.stats();
        replay_io.direct_read_calls += reader_stats.direct_read_calls;
        replay_io.direct_read_bytes = replay_io
            .direct_read_bytes
            .saturating_add(reader_stats.direct_read_bytes);
        stats.record_read_wall += reader_stats.record_read_wall;
        stats.record_decode_wall += reader_stats.record_decode_wall;
        stats.record_append_wall += reader_stats.record_append_wall;
        let part_pipeline_wall = part_pipeline_start.elapsed();
        stats.pipeline_wall += part_pipeline_wall;
        let read_service = stats.read_decode_wall.saturating_sub(read_before);
        let insert_service = stats.insert_wall.saturating_sub(insert_before);
        stats.pipeline_overlap_wall +=
            (read_service + insert_service).saturating_sub(part_pipeline_wall);
    }

    let emit_start = Instant::now();
    let mut observe_reservoir = |_: u32, _: &HashPruneReservoir| {};
    emit_shard_backfill(
        point_start,
        point_end,
        reservoirs,
        policy,
        &mut stats.backfill,
        None,
        None,
        &mut observe_reservoir,
        on_point_done,
    )?;
    stats.finalize_emit_wall += emit_start.elapsed();
    Ok(())
}

fn replay_segment_group_compact_backfill<F>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    policy: HashPruneBackfillPolicy,
    reservoirs: &mut Vec<HashPruneReservoir>,
    config: ExternalReduceConfig,
    pool: &mut ExternalReduceThreadPool,
    replay_io: &mut SpillReplayIoStats,
    stats: &mut SpillReduceStats,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if point_start >= point_end {
        return Ok(());
    }
    debug_assert!(!config.io.enabled);
    resize_or_reset_reservoirs(reservoirs, point_end - point_start, l_max);

    let mut file_lookup = BTreeMap::new();
    for part in &artifacts.part_files {
        file_lookup.insert((part.shard_id, part.file_id), part.path.clone());
    }
    let mut segments_by_part: BTreeMap<(usize, u16), (PathBuf, Vec<SpillSegmentDesc>)> =
        BTreeMap::new();
    for segment in &group.segments {
        let key = (segment.shard_id as usize, segment.file_id);
        let Some(path) = file_lookup.get(&key) else {
            return Err(AnnError::log_index_error(format!(
                "Missing spill part for shard={} file_id={}",
                segment.shard_id, segment.file_id
            )));
        };
        segments_by_part
            .entry(key)
            .or_insert_with(|| (path.clone(), Vec::new()))
            .1
            .push(*segment);
    }

    for (_, (path, segments)) in segments_by_part {
        let mut cursor = PartBatchCursor::open(&path, segments, config.io)?;
        replay_io.reader_open_calls += 1;
        let mut current = Vec::with_capacity(config.batch_records * RUN_RECORD_BYTES);
        let mut next = Vec::with_capacity(config.batch_records * RUN_RECORD_BYTES);
        let mut worker_control =
            ExternalReduceWorkerControl::new(config.initial_workers, config.workers);
        let part_pipeline_start = Instant::now();
        let read_before = stats.read_decode_wall;
        let insert_before = stats.insert_wall;

        let prime_start = Instant::now();
        let mut current_count =
            cursor.read_next_compact_batch(config.batch_records, &mut current, replay_io)?;
        stats.read_decode_wall += prime_start.elapsed();

        while current_count > 0 {
            stats.batch_peak_records = stats.batch_peak_records.max(current_count);
            let current_bytes = &current[..current_count * RUN_RECORD_BYTES];
            stats.batch_peak_bytes = stats.batch_peak_bytes.max(current_bytes.len());
            let valid_start = compact_record_partition_point(current_bytes, point_start);
            for bytes in
                current_bytes[..valid_start * RUN_RECORD_BYTES].chunks_exact(RUN_RECORD_BYTES)
            {
                decode_spill_record16(bytes)?;
            }
            let valid_records = &current_bytes[valid_start * RUN_RECORD_BYTES..];

            let next_read = if valid_records.is_empty() {
                let read_start = Instant::now();
                let result =
                    cursor.read_next_compact_batch(config.batch_records, &mut next, replay_io);
                (result, read_start.elapsed())
            } else {
                let active_workers = worker_control.workers();
                let ((result, read_wall), insert_wall) = insert_compact_backfill_batch_while(
                    pool.pool(),
                    point_start,
                    point_end,
                    &mut reservoirs[..point_end - point_start],
                    valid_records,
                    active_workers,
                    policy,
                    &mut stats.backfill,
                    || {
                        let read_start = Instant::now();
                        let result = cursor.read_next_compact_batch(
                            config.batch_records,
                            &mut next,
                            replay_io,
                        );
                        (result, read_start.elapsed())
                    },
                )?;
                stats.insert_wall += insert_wall;
                stats.parallel_batches += 1;
                stats.parallel_records = stats
                    .parallel_records
                    .saturating_add((valid_records.len() / RUN_RECORD_BYTES) as u64);
                stats.parallel_workers = stats.parallel_workers.max(active_workers);
                let requested_workers = worker_control.observe(insert_wall, read_wall);
                pool.resize(requested_workers)?;
                (result, read_wall)
            };

            stats.read_decode_wall += next_read.1;
            let next_count = next_read.0?;
            std::mem::swap(&mut current, &mut next);
            current_count = next_count;
        }

        let reader_stats = cursor.stats();
        replay_io.direct_read_calls += reader_stats.direct_read_calls;
        replay_io.direct_read_bytes = replay_io
            .direct_read_bytes
            .saturating_add(reader_stats.direct_read_bytes);
        stats.record_read_wall += reader_stats.record_read_wall;
        stats.record_decode_wall += reader_stats.record_decode_wall;
        stats.record_append_wall += reader_stats.record_append_wall;
        let part_pipeline_wall = part_pipeline_start.elapsed();
        stats.pipeline_wall += part_pipeline_wall;
        let read_service = stats.read_decode_wall.saturating_sub(read_before);
        let insert_service = stats.insert_wall.saturating_sub(insert_before);
        stats.pipeline_overlap_wall +=
            (read_service + insert_service).saturating_sub(part_pipeline_wall);
    }

    let emit_start = Instant::now();
    let mut observe_reservoir = |_: u32, _: &HashPruneReservoir| {};
    emit_shard_backfill(
        point_start,
        point_end,
        reservoirs,
        policy,
        &mut stats.backfill,
        None,
        None,
        &mut observe_reservoir,
        on_point_done,
    )?;
    stats.finalize_emit_wall += emit_start.elapsed();
    Ok(())
}

fn reduce_segments_sharded_in_order_from_parallel_backfill<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    config: ExternalReduceConfig,
    replay_mode: ExternalParallelReplayMode,
    mut on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    validate_spill_policies(artifacts, evidence_policy, backfill_policy)?;
    if num_points == 0 || point_start >= num_points {
        return Ok(SpillReduceStats {
            direct_io: config.io.enabled,
            parallel_workers: config.workers,
            ..SpillReduceStats::default()
        });
    }

    let groups = group_segments_by_shard(num_points, points_per_shard, artifacts)?;
    let workspace_points = groups
        .iter()
        .filter(|group| group.point_end > point_start)
        .map(|group| {
            group
                .point_end
                .saturating_sub(group.point_start.max(point_start))
        })
        .max()
        .unwrap_or(0);
    let workspace_bytes_per_point = std::mem::size_of::<HashPruneReservoir>()
        + l_max.saturating_mul(std::mem::size_of::<super::hash_prune::Slot>());
    let workspace_alloc_start = Instant::now();
    let mut reservoirs = allocate_reservoir_slab(workspace_points, l_max);
    let workspace_alloc_wall = workspace_alloc_start.elapsed();
    let mut pool = ExternalReduceThreadPool::new(config.initial_workers)?;
    let decoder_pool = match replay_mode {
        ExternalParallelReplayMode::DecodedPipeline => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(EXTERNAL_REDUCE_DECODE_WORKERS)
                .thread_name(|index| format!("spill-decode-{index}"))
                .build()
                .map_err(|error| {
                    AnnError::log_index_error(format!(
                        "Failed to create external-reduce decoder pool: {error}"
                    ))
                })?,
        ),
        ExternalParallelReplayMode::CompactFused => None,
    };
    let mut replay_io = SpillReplayIoStats::default();
    let mut stats = SpillReduceStats {
        shard_groups: groups.len(),
        max_segments_in_group: groups
            .iter()
            .map(|group| group.segments.len())
            .max()
            .unwrap_or(0),
        direct_io: config.io.enabled,
        workspace_points,
        workspace_bytes: workspace_points.saturating_mul(workspace_bytes_per_point),
        workspace_alloc_wall,
        decoder_workers: decoder_pool
            .as_ref()
            .map(|pool| pool.current_num_threads())
            .unwrap_or(0),
        parallel_workers: config.initial_workers,
        ..SpillReduceStats::default()
    };

    let mut next_point = point_start;
    for group in &groups {
        if group.point_end <= point_start {
            continue;
        }
        let group_start = group.point_start.max(point_start);
        if next_point < group_start {
            let emit_start = Instant::now();
            emit_empty_point_range_backfill(
                next_point,
                group_start,
                &mut stats.backfill,
                None,
                None,
                &mut on_point_done,
            )?;
            stats.finalize_emit_wall += emit_start.elapsed();
        }

        match replay_mode {
            ExternalParallelReplayMode::DecodedPipeline => {
                let decoder_pool = decoder_pool
                    .as_ref()
                    .expect("decoded external replay must own a decoder pool");
                replay_segment_group_parallel_backfill(
                    group_start,
                    group.point_end,
                    l_max,
                    group,
                    artifacts,
                    backfill_policy,
                    &mut reservoirs,
                    config,
                    decoder_pool,
                    &mut pool,
                    &mut replay_io,
                    &mut stats,
                    &mut on_point_done,
                )?;
            }
            ExternalParallelReplayMode::CompactFused => {
                replay_segment_group_compact_backfill(
                    group_start,
                    group.point_end,
                    l_max,
                    group,
                    artifacts,
                    backfill_policy,
                    &mut reservoirs,
                    config,
                    &mut pool,
                    &mut replay_io,
                    &mut stats,
                    &mut on_point_done,
                )?;
            }
        }
        next_point = group.point_end;
    }

    if next_point < num_points {
        let emit_start = Instant::now();
        emit_empty_point_range_backfill(
            next_point,
            num_points,
            &mut stats.backfill,
            None,
            None,
            &mut on_point_done,
        )?;
        stats.finalize_emit_wall += emit_start.elapsed();
    }
    stats.segments_scanned = replay_io.segments_scanned;
    stats.part_files_opened = replay_io.reader_open_calls;
    stats.logical_read_bytes = replay_io.logical_read_bytes;
    stats.reader_open_calls = replay_io.reader_open_calls;
    stats.direct_read_calls = replay_io.direct_read_calls;
    stats.direct_read_bytes = replay_io.direct_read_bytes;
    Ok(stats)
}

fn reduce_segments_sharded_in_order_with_limit<F>(
    num_points: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    reduce_segments_sharded_in_order_with_limit_and_policy(
        num_points,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        HashPruneEvidencePolicy::off(),
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_with_limit_and_policy<F>(
    num_points: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    policy: HashPruneEvidencePolicy,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    reduce_segments_sharded_in_order_with_limit_and_policies(
        num_points,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        policy,
        HashPruneBackfillPolicy::off(),
        None,
        None,
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_with_limit_and_policies<F>(
    num_points: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    reduce_segments_sharded_in_order_from_with_limit_and_policies(
        num_points,
        0,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        evidence_policy,
        backfill_policy,
        diagnostic_collector,
        diagnostic_finalizer,
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_from_with_limit<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    reduce_segments_sharded_in_order_from_with_limit_and_policy(
        num_points,
        point_start,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        HashPruneEvidencePolicy::off(),
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_from_with_limit_and_policy<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    policy: HashPruneEvidencePolicy,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    reduce_segments_sharded_in_order_from_with_limit_and_policies(
        num_points,
        point_start,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        policy,
        HashPruneBackfillPolicy::off(),
        None,
        None,
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_from_with_limit_and_policies<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let microscope = !evidence_policy.is_off() && edge_sink_microscope_enabled();
    reduce_segments_sharded_in_order_from_with_limit_and_policies_core(
        num_points,
        point_start,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        evidence_policy,
        backfill_policy,
        diagnostic_collector,
        diagnostic_finalizer,
        DirectIoConfig::disabled(),
        microscope,
        |_, _: &HashPruneReservoir| {},
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_from_with_limit_and_policy_core<F, O>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    policy: HashPruneEvidencePolicy,
    microscope: bool,
    observe_reservoir: O,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    reduce_segments_sharded_in_order_from_with_limit_and_policies_core(
        num_points,
        point_start,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        policy,
        HashPruneBackfillPolicy::off(),
        None,
        None,
        DirectIoConfig::disabled(),
        microscope,
        observe_reservoir,
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_from_with_limit_and_policies_core<F, O>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    io_cfg: DirectIoConfig,
    microscope: bool,
    mut observe_reservoir: O,
    mut on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    validate_spill_policies(artifacts, evidence_policy, backfill_policy)?;
    if num_points == 0 {
        return Ok(SpillReduceStats::default());
    }
    if point_start >= num_points {
        return Ok(SpillReduceStats::default());
    }

    let mut next_point = point_start;
    let groups = group_segments_by_shard(num_points, points_per_shard, artifacts)?;
    let workspace_points = groups
        .iter()
        .filter(|group| group.point_end > point_start)
        .map(|group| {
            group
                .point_end
                .saturating_sub(group.point_start.max(point_start))
        })
        .max()
        .unwrap_or(0);
    let workspace_bytes_per_point = std::mem::size_of::<HashPruneReservoir>()
        + l_max.saturating_mul(std::mem::size_of::<super::hash_prune::Slot>());
    let mut reservoirs = allocate_reservoir_slab(workspace_points, l_max);
    let mut replay_io = SpillReplayIoStats::default();
    let mut stats = SpillReduceStats {
        shard_groups: groups.len(),
        max_segments_in_group: groups
            .iter()
            .map(|group| group.segments.len())
            .max()
            .unwrap_or(0),
        direct_io: io_cfg.enabled,
        workspace_points,
        workspace_bytes: workspace_points.saturating_mul(workspace_bytes_per_point),
        evidence: HashPruneEvidenceProfile::default(),
        backfill: HashPruneBackfillProfile::default(),
        ..SpillReduceStats::default()
    };
    let mut insert_profile = (!evidence_policy.is_off()
        && (microscope || evidence_policy.enforces()))
    .then(HashPruneInsertProfile::default);
    let mut result = Ok(());
    for group in &groups {
        if group.point_end <= point_start {
            continue;
        }
        let group_start = group.point_start.max(point_start);
        if next_point < group_start {
            let empty_result = if !backfill_policy.is_off() {
                emit_empty_point_range_backfill(
                    next_point,
                    group_start,
                    &mut stats.backfill,
                    diagnostic_collector,
                    diagnostic_finalizer.as_deref_mut(),
                    &mut on_point_done,
                )
            } else {
                emit_empty_point_range(next_point, group_start, &mut on_point_done)
            };
            if let Err(err) = empty_result {
                result = Err(err);
                break;
            }
        }

        let points_per_chunk = points_per_shard
            .max(1)
            .min(group.point_end.saturating_sub(group_start).max(1));
        let group_result = if !backfill_policy.is_off() {
            reduce_segment_group_in_range_backfill(
                group_start,
                group.point_end,
                l_max,
                points_per_chunk,
                group,
                artifacts,
                backfill_policy,
                &mut stats.backfill,
                diagnostic_collector,
                diagnostic_finalizer.as_deref_mut(),
                &mut reservoirs,
                io_cfg,
                &mut replay_io,
                &mut observe_reservoir,
                &mut on_point_done,
            )
        } else if evidence_policy.is_off() {
            reduce_segment_group_in_range_off(
                group_start,
                group.point_end,
                l_max,
                points_per_chunk,
                group,
                artifacts,
                &mut reservoirs,
                io_cfg,
                &mut replay_io,
                &mut observe_reservoir,
                &mut on_point_done,
            )
        } else if let Some(profile) = insert_profile.as_mut() {
            reduce_segment_group_in_range_with_policy(
                group_start,
                group.point_end,
                l_max,
                points_per_chunk,
                group,
                artifacts,
                evidence_policy,
                profile,
                &mut reservoirs,
                io_cfg,
                &mut replay_io,
                &mut observe_reservoir,
                &mut on_point_done,
            )
        } else {
            debug_assert!(!microscope);
            debug_assert!(!evidence_policy.enforces());
            debug_assert!(insert_profile.is_none());
            reduce_segment_group_in_range_report(
                group_start,
                group.point_end,
                l_max,
                points_per_chunk,
                group,
                artifacts,
                &mut reservoirs,
                io_cfg,
                &mut replay_io,
                &mut observe_reservoir,
                &mut on_point_done,
            )
        };
        if let Err(err) = group_result {
            result = Err(err);
            break;
        }
        next_point = group.point_end;
    }

    if result.is_ok() && next_point < num_points {
        result = if !backfill_policy.is_off() {
            emit_empty_point_range_backfill(
                next_point,
                num_points,
                &mut stats.backfill,
                diagnostic_collector,
                diagnostic_finalizer.as_deref_mut(),
                &mut on_point_done,
            )
        } else {
            emit_empty_point_range(next_point, num_points, &mut on_point_done)
        };
    }
    if let Some(insert_profile) = insert_profile {
        stats.evidence.merge(insert_profile.evidence);
    }
    stats.segments_scanned = replay_io.segments_scanned;
    stats.part_files_opened = replay_io.reader_open_calls;
    stats.logical_read_bytes = replay_io.logical_read_bytes;
    stats.reader_open_calls = replay_io.reader_open_calls;
    stats.direct_read_calls = replay_io.direct_read_calls;
    stats.direct_read_bytes = replay_io.direct_read_bytes;
    result.map(|_| stats)
}

fn validate_spill_policies(
    artifacts: &SpillArtifacts,
    evidence_policy: HashPruneEvidencePolicy,
    backfill_policy: HashPruneBackfillPolicy,
) -> AnnResult<()> {
    if artifacts.policy != evidence_policy || artifacts.backfill_policy != backfill_policy {
        return Err(AnnError::log_index_error(format!(
            concat!(
                "HashPrune spill policy mismatch: artifact_evidence={:?} ",
                "reducer_evidence={:?} artifact_backfill={:?} reducer_backfill={:?}"
            ),
            artifacts.policy, evidence_policy, artifacts.backfill_policy, backfill_policy
        )));
    }
    Ok(())
}

fn reduce_segment_group_in_range_off<F, O>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    reservoirs: &mut Vec<HashPruneReservoir>,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    replay_segment_group_in_range_with(
        point_start,
        point_end,
        l_max,
        points_per_chunk,
        group,
        artifacts,
        reservoirs,
        io_cfg,
        replay_io,
        |reservoir, edge| {
            reservoir.insert(edge.c, edge.hash, edge.dist);
            Ok(())
        },
        observe_reservoir,
        on_point_done,
    )
}

fn reduce_segment_group_in_range_backfill<F, O>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    policy: HashPruneBackfillPolicy,
    profile: &mut HashPruneBackfillProfile,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    reservoirs: &mut Vec<HashPruneReservoir>,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    replay_segment_group_in_range_backfill(
        point_start,
        point_end,
        l_max,
        points_per_chunk,
        group,
        artifacts,
        policy,
        profile,
        diagnostic_collector,
        diagnostic_finalizer,
        reservoirs,
        io_cfg,
        replay_io,
        observe_reservoir,
        on_point_done,
    )
}

fn reduce_segment_group_in_range_report<F, O>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    reservoirs: &mut Vec<HashPruneReservoir>,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    replay_segment_group_in_range_with(
        point_start,
        point_end,
        l_max,
        points_per_chunk,
        group,
        artifacts,
        reservoirs,
        io_cfg,
        replay_io,
        |reservoir, edge| {
            reservoir.insert_report_prevalidated(edge.p, edge.c, edge.hash, edge.dist, edge.meta);
            Ok(())
        },
        observe_reservoir,
        on_point_done,
    )
}

fn reduce_segment_group_in_range_with_policy<F, O>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    policy: HashPruneEvidencePolicy,
    insert_profile: &mut HashPruneInsertProfile,
    reservoirs: &mut Vec<HashPruneReservoir>,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    replay_segment_group_in_range_with(
        point_start,
        point_end,
        l_max,
        points_per_chunk,
        group,
        artifacts,
        reservoirs,
        io_cfg,
        replay_io,
        |reservoir, edge| {
            reservoir.insert_with_policy_profiled(
                edge.p,
                edge.c,
                edge.hash,
                edge.dist,
                edge.meta,
                policy,
                insert_profile,
            )
        },
        observe_reservoir,
        on_point_done,
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct SpillReplayIoStats {
    segments_scanned: usize,
    logical_read_bytes: u64,
    reader_open_calls: usize,
    direct_read_calls: usize,
    direct_read_bytes: u64,
}

fn scan_segment_group<F>(
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    mut on_edge: F,
) -> AnnResult<()>
where
    F: FnMut(PendingEdge) -> AnnResult<()>,
{
    let mut file_lookup = BTreeMap::new();
    for part in &artifacts.part_files {
        file_lookup.insert((part.shard_id, part.file_id), part.path.clone());
    }

    let mut segments_by_part: BTreeMap<(usize, u16), (PathBuf, Vec<SpillSegmentDesc>)> =
        BTreeMap::new();
    for segment in &group.segments {
        let key = (segment.shard_id as usize, segment.file_id);
        let Some(path) = file_lookup.get(&key) else {
            return Err(AnnError::log_index_error(format!(
                "Missing spill part for shard={} file_id={}",
                segment.shard_id, segment.file_id
            )));
        };
        segments_by_part
            .entry(key)
            .or_insert_with(|| (path.clone(), Vec::new()))
            .1
            .push(*segment);
    }

    for (_, (path, mut segments)) in segments_by_part {
        segments.sort_unstable_by_key(|segment| segment.byte_offset);
        let mut reader = SegmentReader::open_file(&path, io_cfg)?;
        replay_io.reader_open_calls += 1;
        for segment in segments {
            reader.set_segment(segment)?;
            replay_io.segments_scanned += 1;
            replay_io.logical_read_bytes = replay_io
                .logical_read_bytes
                .saturating_add(segment.byte_len);
            while let Some(edge) = reader.read_edge()? {
                on_edge(edge)?;
            }
        }
        let reader_stats = reader.stats();
        replay_io.direct_read_calls += reader_stats.direct_read_calls;
        replay_io.direct_read_bytes = replay_io
            .direct_read_bytes
            .saturating_add(reader_stats.direct_read_bytes);
    }
    Ok(())
}

fn replay_segment_group_in_range_with<F, I, O>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    reservoirs: &mut Vec<HashPruneReservoir>,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    mut insert_edge: I,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    I: FnMut(&mut HashPruneReservoir, PendingEdge) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    if point_start >= point_end {
        return Ok(());
    }

    let points_per_chunk = points_per_chunk.max(1);
    let mut chunk_start = point_start;
    let mut chunk_end = (chunk_start + points_per_chunk).min(point_end);
    resize_or_reset_reservoirs(reservoirs, chunk_end.saturating_sub(chunk_start), l_max);

    scan_segment_group(group, artifacts, io_cfg, replay_io, |edge| {
        while edge.p >= chunk_end {
            emit_shard(
                chunk_start,
                chunk_end,
                reservoirs,
                observe_reservoir,
                on_point_done,
            )?;
            chunk_start = chunk_end;
            chunk_end = (chunk_end + points_per_chunk).min(point_end);
            if chunk_start >= point_end {
                return Err(AnnError::log_index_error(format!(
                    "Spill reducer encountered edge p={} outside shard range [{point_start}, {point_end})",
                    edge.p
                )));
            }
            if chunk_start < point_end {
                resize_or_reset_reservoirs(reservoirs, chunk_end - chunk_start, l_max);
            }
        }

        if edge.p < chunk_start {
            if edge.p < point_start {
                return Ok(());
            }
            return Err(AnnError::log_index_error(format!(
                "Spill reducer encountered out-of-order candidate edge p={} before current chunk start={chunk_start}",
                edge.p
            )));
        }
        let local_idx = edge.p - chunk_start;
        insert_edge(&mut reservoirs[local_idx], edge)?;
        Ok(())
    })?;

    while chunk_start < point_end {
        emit_shard(
            chunk_start,
            chunk_end,
            reservoirs,
            observe_reservoir,
            on_point_done,
        )?;
        chunk_start = chunk_end;
        chunk_end = (chunk_end + points_per_chunk).min(point_end);

        if chunk_start < point_end {
            resize_or_reset_reservoirs(reservoirs, chunk_end - chunk_start, l_max);
        }
    }

    Ok(())
}

fn replay_segment_group_in_range_backfill<F, O>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    policy: HashPruneBackfillPolicy,
    profile: &mut HashPruneBackfillProfile,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    reservoirs: &mut Vec<HashPruneReservoir>,
    io_cfg: DirectIoConfig,
    replay_io: &mut SpillReplayIoStats,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    if point_start >= point_end {
        return Ok(());
    }

    let points_per_chunk = points_per_chunk.max(1);
    let mut chunk_start = point_start;
    let mut chunk_end = (chunk_start + points_per_chunk).min(point_end);
    resize_or_reset_reservoirs(reservoirs, chunk_end.saturating_sub(chunk_start), l_max);

    scan_segment_group(group, artifacts, io_cfg, replay_io, |edge| {
        while edge.p >= chunk_end {
            emit_shard_backfill(
                chunk_start,
                chunk_end,
                reservoirs,
                policy,
                profile,
                diagnostic_collector,
                diagnostic_finalizer.as_deref_mut(),
                observe_reservoir,
                on_point_done,
            )?;
            chunk_start = chunk_end;
            chunk_end = (chunk_end + points_per_chunk).min(point_end);
            if chunk_start >= point_end {
                return Err(AnnError::log_index_error(format!(
                    "Spill reducer encountered edge p={} outside shard range [{point_start}, {point_end})",
                    edge.p
                )));
            }
            if chunk_start < point_end {
                resize_or_reset_reservoirs(reservoirs, chunk_end - chunk_start, l_max);
            }
        }

        if edge.p < chunk_start {
            if edge.p < point_start {
                return Ok(());
            }
            return Err(AnnError::log_index_error(format!(
                "Spill reducer encountered out-of-order candidate edge p={} before current chunk start={chunk_start}",
                edge.p
            )));
        }
        let local_idx = edge.p - chunk_start;
        if let Some(collector) = diagnostic_collector {
            collector.record(edge.p, edge.c as usize, edge.hash, edge.dist)?;
        }
        reservoirs[local_idx].insert_backfill_prevalidated_with_policy(
            edge.p, edge.c, edge.hash, edge.dist, policy, profile,
        )?;
        Ok(())
    })?;

    while chunk_start < point_end {
        emit_shard_backfill(
            chunk_start,
            chunk_end,
            reservoirs,
            policy,
            profile,
            diagnostic_collector,
            diagnostic_finalizer.as_deref_mut(),
            observe_reservoir,
            on_point_done,
        )?;
        chunk_start = chunk_end;
        chunk_end = (chunk_end + points_per_chunk).min(point_end);

        if chunk_start < point_end {
            resize_or_reset_reservoirs(reservoirs, chunk_end - chunk_start, l_max);
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct SegmentGroup {
    point_start: usize,
    point_end: usize,
    segments: Vec<SpillSegmentDesc>,
}

fn group_segments_by_shard(
    num_points: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
) -> AnnResult<Vec<SegmentGroup>> {
    if artifacts.segment_manifest.is_empty() {
        return Ok(Vec::new());
    }

    let mut grouped: BTreeMap<usize, Vec<SpillSegmentDesc>> = BTreeMap::new();
    for segment in &artifacts.segment_manifest {
        grouped
            .entry(segment.shard_id as usize)
            .or_default()
            .push(*segment);
    }

    let points_per_shard = points_per_shard.max(1);
    let mut groups = Vec::with_capacity(grouped.len());
    for (shard_id, mut segments) in grouped {
        let point_start = shard_id.saturating_mul(points_per_shard);
        if point_start >= num_points {
            return Err(AnnError::log_index_error(format!(
                "Spill reducer encountered shard_id={shard_id} outside num_points={num_points} with points_per_shard={points_per_shard}"
            )));
        }
        let point_end = (point_start + points_per_shard).min(num_points);
        for segment in &segments {
            let src_min = segment.src_min as usize;
            let src_max = segment.src_max as usize;
            if src_min < point_start || src_max >= point_end {
                return Err(AnnError::log_index_error(format!(
                    "Spill reducer encountered segment metadata src_range=[{}, {}] outside shard range [{}, {}) for shard_id={shard_id}",
                    segment.src_min, segment.src_max, point_start, point_end
                )));
            }
        }
        segments.sort_unstable_by_key(|segment| {
            (segment.src_min, segment.file_id, segment.byte_offset)
        });
        groups.push(SegmentGroup {
            point_start,
            point_end,
            segments,
        });
    }

    Ok(groups)
}

fn summarize_spill_artifacts(artifacts: &SpillArtifacts) -> SpillRunStats {
    let mut parts_per_shard: BTreeMap<usize, usize> = BTreeMap::new();
    let mut segments_per_shard: BTreeMap<usize, usize> = BTreeMap::new();
    for part in &artifacts.part_files {
        *parts_per_shard.entry(part.shard_id).or_insert(0) += 1;
    }
    for segment in &artifacts.segment_manifest {
        *segments_per_shard
            .entry(segment.shard_id as usize)
            .or_insert(0) += 1;
    }

    SpillRunStats {
        part_files: artifacts.part_files.len(),
        segment_count: artifacts.segment_manifest.len(),
        shard_groups: parts_per_shard
            .keys()
            .chain(segments_per_shard.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        max_parts_per_shard: parts_per_shard.values().copied().max().unwrap_or(0),
        max_segments_per_shard: segments_per_shard.values().copied().max().unwrap_or(0),
        record_bytes: RUN_RECORD_BYTES,
        manifest_bytes: fs::metadata(&artifacts.manifest_path)
            .map(|meta| meta.len())
            .unwrap_or(0),
    }
}

fn effective_spill_cache_bytes(spill_cache_bytes: usize) -> usize {
    if spill_cache_bytes == 0 {
        DEFAULT_SPILL_CACHE_BYTES
    } else {
        spill_cache_bytes
    }
}

fn allocate_reservoir_slab(points: usize, l_max: usize) -> Vec<HashPruneReservoir> {
    (0..points)
        .map(|_| HashPruneReservoir::new(l_max))
        .collect()
}

fn resize_or_reset_reservoirs(
    reservoirs: &mut Vec<HashPruneReservoir>,
    points: usize,
    l_max: usize,
) {
    if reservoirs.len() < points {
        reservoirs.resize_with(points, || HashPruneReservoir::new(l_max));
    }

    for reservoir in reservoirs.iter_mut().take(points) {
        reservoir.clear();
    }
}

fn emit_shard<F, O>(
    shard_start: usize,
    shard_end: usize,
    reservoirs: &mut [HashPruneReservoir],
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    for point in shard_start..shard_end {
        let local_idx = point - shard_start;
        observe_reservoir(point as u32, &reservoirs[local_idx]);
        let neighbors = reservoirs[local_idx].copy_neighbors_and_clear();
        on_point_done(point as u32, neighbors)?;
    }
    Ok(())
}

fn emit_shard_backfill<F, O>(
    shard_start: usize,
    shard_end: usize,
    reservoirs: &mut [HashPruneReservoir],
    policy: HashPruneBackfillPolicy,
    profile: &mut HashPruneBackfillProfile,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    observe_reservoir: &mut O,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    O: FnMut(u32, &HashPruneReservoir),
{
    for point in shard_start..shard_end {
        let local_idx = point - shard_start;
        observe_reservoir(point as u32, &reservoirs[local_idx]);
        let neighbors = if let (Some(collector), Some(finalizer)) =
            (diagnostic_collector, diagnostic_finalizer.as_deref_mut())
        {
            let row = reservoirs[local_idx]
                .drain_finalized_slots_with_backfill(point, policy, profile)?;
            finalizer.observe_row(point, collector, &row)?;
            row.slots
                .into_iter()
                .map(|slot| slot.target)
                .collect::<Vec<_>>()
        } else {
            reservoirs[local_idx].copy_neighbors_with_backfill_and_clear(point, policy, profile)?
        };
        on_point_done(point as u32, neighbors)?;
    }
    Ok(())
}

fn emit_empty_point_range<F>(start: usize, end: usize, on_point_done: &mut F) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    for point in start..end {
        on_point_done(point as u32, Vec::new())?;
    }
    Ok(())
}

fn emit_empty_point_range_backfill<F>(
    start: usize,
    end: usize,
    profile: &mut HashPruneBackfillProfile,
    diagnostic_collector: Option<&HashPruneTraceCollector>,
    mut diagnostic_finalizer: Option<&mut HashPruneDiagnosticFinalizer>,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    for point in start..end {
        profile.rows_seen += 1;
        profile.rows_underfull_before += 1;
        if let (Some(collector), Some(finalizer)) =
            (diagnostic_collector, diagnostic_finalizer.as_deref_mut())
        {
            finalizer.observe_row(
                point,
                collector,
                &FinalizedHashPruneRow {
                    slots: Vec::new(),
                    primary_count: 0,
                },
            )?;
        }
        on_point_done(point as u32, Vec::new())?;
    }
    Ok(())
}

#[derive(Debug)]
enum SegmentReaderInner {
    Buffered {
        reader: BufReader<File>,
        remaining: u64,
    },
    Direct {
        file: DirectIoFile,
        offset: u64,
        len: u64,
        buffer: AlignedBoxWithSlice<u8>,
        buffer_offset: u64,
        buffer_valid: usize,
        direct_read_calls: usize,
        direct_read_bytes: u64,
    },
}

#[derive(Debug)]
struct SegmentReader {
    inner: SegmentReaderInner,
    last_batch_source: Option<usize>,
    record_read_wall: Duration,
    record_decode_wall: Duration,
    record_append_wall: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct SegmentReaderStats {
    direct_read_calls: usize,
    direct_read_bytes: u64,
    record_read_wall: Duration,
    record_decode_wall: Duration,
    record_append_wall: Duration,
}

#[derive(Default)]
struct ParallelDecodeScratch {
    bytes: Vec<u8>,
    left: Vec<PendingEdge>,
    right: Vec<PendingEdge>,
}

impl SegmentReader {
    fn open(path: &Path, segment: SpillSegmentDesc, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let mut reader = Self::open_file(path, io_cfg)?;
        reader.set_segment(segment)?;
        Ok(reader)
    }

    fn open_file(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let inner = if io_cfg.enabled {
            let alignment = io_cfg.effective_alignment();
            let buffer_bytes = DIRECT_CANDIDATE_BUFFER_BYTES.max(alignment);
            SegmentReaderInner::Direct {
                file: DirectIoFile::open_read(path, io_cfg)?,
                offset: 0,
                len: 0,
                buffer: AlignedBoxWithSlice::new(buffer_bytes, alignment)?,
                buffer_offset: u64::MAX,
                buffer_valid: 0,
                direct_read_calls: 0,
                direct_read_bytes: 0,
            }
        } else {
            SegmentReaderInner::Buffered {
                reader: BufReader::new(File::open(path)?),
                remaining: 0,
            }
        };
        Ok(Self {
            inner,
            last_batch_source: None,
            record_read_wall: Duration::ZERO,
            record_decode_wall: Duration::ZERO,
            record_append_wall: Duration::ZERO,
        })
    }

    fn set_segment(&mut self, segment: SpillSegmentDesc) -> AnnResult<()> {
        if segment.byte_len % RUN_RECORD_BYTES as u64 != 0 {
            return Err(AnnError::log_index_error(format!(
                "Candidate segment has trailing partial record: byte_offset={} byte_len={}",
                segment.byte_offset, segment.byte_len
            )));
        }
        self.last_batch_source = None;
        match &mut self.inner {
            SegmentReaderInner::Buffered { reader, remaining } => {
                reader.seek(SeekFrom::Start(segment.byte_offset))?;
                *remaining = segment.byte_len;
            }
            SegmentReaderInner::Direct { offset, len, .. } => {
                *offset = segment.byte_offset;
                *len = segment.byte_offset.saturating_add(segment.byte_len);
            }
        }
        Ok(())
    }

    fn stats(&self) -> SegmentReaderStats {
        match &self.inner {
            SegmentReaderInner::Buffered { .. } => SegmentReaderStats {
                record_read_wall: self.record_read_wall,
                record_decode_wall: self.record_decode_wall,
                record_append_wall: self.record_append_wall,
                ..SegmentReaderStats::default()
            },
            SegmentReaderInner::Direct {
                direct_read_calls,
                direct_read_bytes,
                ..
            } => SegmentReaderStats {
                direct_read_calls: *direct_read_calls,
                direct_read_bytes: *direct_read_bytes,
                record_read_wall: self.record_read_wall,
                record_decode_wall: self.record_decode_wall,
                record_append_wall: self.record_append_wall,
            },
        }
    }

    fn read_edge_batch(
        &mut self,
        max_records: usize,
        output: &mut Vec<PendingEdge>,
    ) -> AnnResult<usize> {
        output.clear();
        let limit = max_records.max(1);
        while output.len() < limit {
            let Some(edge) = self.read_edge()? else {
                break;
            };
            output.push(edge);
        }
        Ok(output.len())
    }

    fn read_sorted_edge_batch_bulk(
        &mut self,
        max_records: usize,
        output: &mut Vec<PendingEdge>,
    ) -> AnnResult<usize> {
        output.clear();
        let limit = max_records.max(1);

        if matches!(&self.inner, SegmentReaderInner::Buffered { .. }) {
            while output.len() < limit {
                let Some(edge) = self.read_edge()? else {
                    break;
                };
                Self::push_sorted_batch_edge(&mut self.last_batch_source, output, edge)?;
            }
            return Ok(output.len());
        }

        let last_source = &mut self.last_batch_source;
        let SegmentReaderInner::Direct {
            file,
            offset,
            len,
            buffer,
            buffer_offset,
            buffer_valid,
            direct_read_calls,
            direct_read_bytes,
        } = &mut self.inner
        else {
            unreachable!();
        };

        while output.len() < limit && *offset < *len {
            if offset.saturating_add(RUN_RECORD_BYTES as u64) > *len {
                return Err(AnnError::log_index_error(format!(
                    "Candidate run has trailing partial record at offset {} of length {}",
                    offset, len
                )));
            }

            let alignment = file.config().effective_alignment() as u64;
            let buffered_end = buffer_offset.saturating_add(*buffer_valid as u64);
            if *buffer_offset == u64::MAX
                || *offset < *buffer_offset
                || offset.saturating_add(RUN_RECORD_BYTES as u64) > buffered_end
            {
                *buffer_offset = *offset / alignment * alignment;
                *buffer_valid =
                    file.read_aligned_at_partial(buffer.as_mut_slice(), *buffer_offset)?;
                *direct_read_calls += 1;
                *direct_read_bytes = direct_read_bytes.saturating_add(*buffer_valid as u64);
            }

            let logical_end = (*len).min(buffer_offset.saturating_add(*buffer_valid as u64));
            let available_records = logical_end.saturating_sub(*offset) as usize / RUN_RECORD_BYTES;
            let take = available_records.min(limit - output.len());
            if take == 0 {
                return Err(AnnError::log_index_error(format!(
                    "Unexpected EOF while bulk-reading candidate segment at offset {}",
                    offset
                )));
            }

            let start = offset.saturating_sub(*buffer_offset) as usize;
            let end = start + take * RUN_RECORD_BYTES;
            for bytes in buffer.as_slice()[start..end].chunks_exact(RUN_RECORD_BYTES) {
                let edge = decode_spill_record16(bytes)?;
                Self::push_sorted_batch_edge(last_source, output, edge)?;
            }
            *offset += (take * RUN_RECORD_BYTES) as u64;
        }
        Ok(output.len())
    }

    fn read_sorted_compact_batch(
        &mut self,
        max_records: usize,
        output: &mut Vec<u8>,
    ) -> AnnResult<usize> {
        let limit = max_records.max(1);
        let last_source = &mut self.last_batch_source;
        let SegmentReaderInner::Buffered { reader, remaining } = &mut self.inner else {
            return Err(AnnError::log_index_error(
                "Compact spill replay currently requires buffered I/O".to_string(),
            ));
        };
        if *remaining == 0 {
            return Ok(0);
        }
        if *remaining < RUN_RECORD_BYTES as u64 {
            return Err(AnnError::log_index_error(format!(
                "Candidate segment has trailing partial record with {} bytes remaining",
                remaining
            )));
        }

        let available_records =
            usize::try_from(*remaining / RUN_RECORD_BYTES as u64).unwrap_or(usize::MAX);
        let take = available_records.min(limit);
        let byte_len = take * RUN_RECORD_BYTES;
        if output.len() < byte_len {
            output.resize(byte_len, 0);
        }
        let active = &mut output[..byte_len];
        let read_start = Instant::now();
        reader.read_exact(active)?;
        self.record_read_wall += read_start.elapsed();
        *remaining -= byte_len as u64;

        let validate_start = Instant::now();
        let result = validate_sorted_compact_record_sources(last_source, active);
        self.record_decode_wall += validate_start.elapsed();
        result?;
        Ok(take)
    }

    fn read_sorted_edge_batch_parallel_decode(
        &mut self,
        max_records: usize,
        output: &mut Vec<PendingEdge>,
        decoder_pool: &rayon::ThreadPool,
        scratch: &mut ParallelDecodeScratch,
    ) -> AnnResult<usize> {
        output.clear();
        let limit = max_records.max(1);
        let last_source = &mut self.last_batch_source;
        if let SegmentReaderInner::Buffered { reader, remaining } = &mut self.inner {
            if *remaining == 0 {
                return Ok(0);
            }
            if *remaining < RUN_RECORD_BYTES as u64 {
                return Err(AnnError::log_index_error(format!(
                    "Candidate segment has trailing partial record with {} bytes remaining",
                    remaining
                )));
            }

            let available_records =
                usize::try_from(*remaining / RUN_RECORD_BYTES as u64).unwrap_or(usize::MAX);
            let take = available_records.min(limit);
            let byte_len = take * RUN_RECORD_BYTES;
            let ParallelDecodeScratch { bytes, left, right } = scratch;
            bytes.resize(byte_len, 0);
            let read_start = Instant::now();
            reader.read_exact(bytes)?;
            self.record_read_wall += read_start.elapsed();
            *remaining -= byte_len as u64;

            let left_records = take.div_ceil(EXTERNAL_REDUCE_DECODE_WORKERS);
            let split = left_records * RUN_RECORD_BYTES;
            let (left_bytes, right_bytes) = bytes.split_at(split);
            let decode_start = Instant::now();
            let (left_result, right_result) = decoder_pool.install(|| {
                rayon::join(
                    || decode_sorted_edge_slice(left_bytes, left),
                    || decode_sorted_edge_slice(right_bytes, right),
                )
            });
            self.record_decode_wall += decode_start.elapsed();
            left_result?;
            right_result?;
            let append_start = Instant::now();
            Self::append_sorted_decoded_edges(last_source, output, left)?;
            Self::append_sorted_decoded_edges(last_source, output, right)?;
            self.record_append_wall += append_start.elapsed();
            return Ok(output.len());
        }

        let SegmentReaderInner::Direct {
            file,
            offset,
            len,
            buffer,
            buffer_offset,
            buffer_valid,
            direct_read_calls,
            direct_read_bytes,
        } = &mut self.inner
        else {
            unreachable!();
        };

        while output.len() < limit && *offset < *len {
            if offset.saturating_add(RUN_RECORD_BYTES as u64) > *len {
                return Err(AnnError::log_index_error(format!(
                    "Candidate run has trailing partial record at offset {} of length {}",
                    offset, len
                )));
            }

            let alignment = file.config().effective_alignment() as u64;
            let buffered_end = buffer_offset.saturating_add(*buffer_valid as u64);
            if *buffer_offset == u64::MAX
                || *offset < *buffer_offset
                || offset.saturating_add(RUN_RECORD_BYTES as u64) > buffered_end
            {
                *buffer_offset = *offset / alignment * alignment;
                let read_start = Instant::now();
                *buffer_valid =
                    file.read_aligned_at_partial(buffer.as_mut_slice(), *buffer_offset)?;
                self.record_read_wall += read_start.elapsed();
                *direct_read_calls += 1;
                *direct_read_bytes = direct_read_bytes.saturating_add(*buffer_valid as u64);
            }

            let logical_end = (*len).min(buffer_offset.saturating_add(*buffer_valid as u64));
            let available_records = logical_end.saturating_sub(*offset) as usize / RUN_RECORD_BYTES;
            let take = available_records.min(limit - output.len());
            if take == 0 {
                return Err(AnnError::log_index_error(format!(
                    "Unexpected EOF while parallel-decoding candidate segment at offset {}",
                    offset
                )));
            }

            let start = offset.saturating_sub(*buffer_offset) as usize;
            let end = start + take * RUN_RECORD_BYTES;
            let records = &buffer.as_slice()[start..end];
            let left_records = take.div_ceil(EXTERNAL_REDUCE_DECODE_WORKERS);
            let split = left_records * RUN_RECORD_BYTES;
            let (left_bytes, right_bytes) = records.split_at(split);
            let ParallelDecodeScratch { left, right, .. } = scratch;
            let decode_start = Instant::now();
            let (left_result, right_result) = decoder_pool.install(|| {
                rayon::join(
                    || decode_sorted_edge_slice(left_bytes, left),
                    || decode_sorted_edge_slice(right_bytes, right),
                )
            });
            self.record_decode_wall += decode_start.elapsed();
            left_result?;
            right_result?;
            let append_start = Instant::now();
            Self::append_sorted_decoded_edges(last_source, output, left)?;
            Self::append_sorted_decoded_edges(last_source, output, right)?;
            self.record_append_wall += append_start.elapsed();
            *offset += (take * RUN_RECORD_BYTES) as u64;
        }
        Ok(output.len())
    }

    fn append_sorted_decoded_edges(
        last_source: &mut Option<usize>,
        output: &mut Vec<PendingEdge>,
        decoded: &mut Vec<PendingEdge>,
    ) -> AnnResult<()> {
        let Some(first) = decoded.first() else {
            return Ok(());
        };
        if let Some(previous) = *last_source {
            if previous > first.p {
                return Err(AnnError::log_index_error(format!(
                    "Spill segment batch is not sorted by source: {previous} before {}",
                    first.p
                )));
            }
        }
        *last_source = decoded.last().map(|edge| edge.p);
        output.append(decoded);
        Ok(())
    }

    fn push_sorted_batch_edge(
        last_source: &mut Option<usize>,
        output: &mut Vec<PendingEdge>,
        edge: PendingEdge,
    ) -> AnnResult<()> {
        if let Some(previous) = *last_source {
            if previous > edge.p {
                return Err(AnnError::log_index_error(format!(
                    "Spill segment batch is not sorted by source: {previous} before {}",
                    edge.p
                )));
            }
        }
        *last_source = Some(edge.p);
        output.push(edge);
        Ok(())
    }

    fn read_edge(&mut self) -> AnnResult<Option<PendingEdge>> {
        let mut bytes = [0u8; RUN_RECORD_BYTES];
        match &mut self.inner {
            SegmentReaderInner::Buffered { reader, remaining } => {
                if *remaining == 0 {
                    return Ok(None);
                }
                if *remaining < RUN_RECORD_BYTES as u64 {
                    return Err(AnnError::log_index_error(format!(
                        "Candidate segment has trailing partial record with {} bytes remaining",
                        remaining
                    )));
                }
                match reader.read_exact(&mut bytes) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                    Err(err) => return Err(err.into()),
                }
                *remaining -= RUN_RECORD_BYTES as u64;
            }
            SegmentReaderInner::Direct {
                file,
                offset,
                len,
                buffer,
                buffer_offset,
                buffer_valid,
                direct_read_calls,
                direct_read_bytes,
            } => {
                if *offset >= *len {
                    return Ok(None);
                }
                if offset.saturating_add(RUN_RECORD_BYTES as u64) > *len {
                    return Err(AnnError::log_index_error(format!(
                        "Candidate run has trailing partial record at offset {} of length {}",
                        offset, len
                    )));
                }
                let alignment = file.config().effective_alignment() as u64;
                let buffered_end = buffer_offset.saturating_add(*buffer_valid as u64);
                if *buffer_offset == u64::MAX
                    || *offset < *buffer_offset
                    || offset.saturating_add(RUN_RECORD_BYTES as u64) > buffered_end
                {
                    *buffer_offset = *offset / alignment * alignment;
                    *buffer_valid =
                        file.read_aligned_at_partial(buffer.as_mut_slice(), *buffer_offset)?;
                    *direct_read_calls += 1;
                    *direct_read_bytes = direct_read_bytes.saturating_add(*buffer_valid as u64);
                }
                let start = offset.saturating_sub(*buffer_offset) as usize;
                let end = start.saturating_add(RUN_RECORD_BYTES);
                if end > *buffer_valid {
                    return Err(AnnError::log_index_error(format!(
                        "Unexpected EOF while reading candidate segment at offset {} (buffer_offset={} buffer_valid={})",
                        offset, buffer_offset, buffer_valid
                    )));
                }
                bytes.copy_from_slice(&buffer.as_slice()[start..end]);
                *offset += RUN_RECORD_BYTES as u64;
            }
        }

        decode_spill_record16(&bytes).map(Some)
    }
}

pub(crate) fn replay_single_spill_segment<F>(
    path: &Path,
    segment: SpillSegmentDesc,
    io_cfg: DirectIoConfig,
    mut on_edge: F,
) -> AnnResult<()>
where
    F: FnMut(PendingEdge) -> AnnResult<()>,
{
    let expected_bytes = u64::from(segment.record_count)
        .checked_mul(RUN_RECORD_BYTES as u64)
        .ok_or_else(|| {
            AnnError::log_index_error(
                "candidate segment record byte count overflows u64".to_string(),
            )
        })?;
    if segment.byte_len != expected_bytes {
        return Err(AnnError::log_index_error(format!(
            "candidate segment byte/record mismatch: byte_len={} record_count={}",
            segment.byte_len, segment.record_count
        )));
    }
    let expected_records = segment.record_count as usize;
    let mut observed = 0usize;
    let mut reader = SegmentReader::open(path, segment, io_cfg)?;
    while let Some(edge) = reader.read_edge()? {
        observed += 1;
        on_edge(edge)?;
    }
    if observed != expected_records {
        return Err(AnnError::log_index_error(format!(
            "candidate segment ended early: expected {expected_records} records, observed {observed}"
        )));
    }
    Ok(())
}

pub(crate) fn decode_spill_record16(bytes: &[u8]) -> AnnResult<PendingEdge> {
    debug_assert_eq!(bytes.len(), RUN_RECORD_BYTES);
    SpillRecord16 {
        src: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        dst: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        hash: u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
        flags: u16::from_le_bytes(bytes[10..12].try_into().unwrap()),
        dist: f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
    }
    .to_pending_edge()
}

fn decode_sorted_edge_slice(bytes: &[u8], output: &mut Vec<PendingEdge>) -> AnnResult<()> {
    debug_assert_eq!(bytes.len() % RUN_RECORD_BYTES, 0);
    output.clear();
    output.reserve(bytes.len() / RUN_RECORD_BYTES);
    let mut previous = None;
    for bytes in bytes.chunks_exact(RUN_RECORD_BYTES) {
        let edge = decode_spill_record16(bytes)?;
        if let Some(previous) = previous {
            if previous > edge.p {
                return Err(AnnError::log_index_error(format!(
                    "Spill segment batch is not sorted by source: {previous} before {}",
                    edge.p
                )));
            }
        }
        previous = Some(edge.p);
        output.push(edge);
    }
    Ok(())
}

struct PartBatchCursor {
    reader: SegmentReader,
    segments: Vec<SpillSegmentDesc>,
    next_segment: usize,
    segment_active: bool,
}

impl PartBatchCursor {
    fn open(
        path: &Path,
        mut segments: Vec<SpillSegmentDesc>,
        io: DirectIoConfig,
    ) -> AnnResult<Self> {
        segments.sort_unstable_by_key(|segment| segment.byte_offset);
        Ok(Self {
            reader: SegmentReader::open_file(path, io)?,
            segments,
            next_segment: 0,
            segment_active: false,
        })
    }

    fn read_next_batch(
        &mut self,
        max_records: usize,
        output: &mut Vec<PendingEdge>,
        replay_io: &mut SpillReplayIoStats,
    ) -> AnnResult<usize> {
        loop {
            if self.segment_active {
                let count = self
                    .reader
                    .read_sorted_edge_batch_bulk(max_records, output)?;
                if count > 0 {
                    return Ok(count);
                }
                self.segment_active = false;
            }

            let Some(segment) = self.segments.get(self.next_segment).copied() else {
                output.clear();
                return Ok(0);
            };
            self.next_segment += 1;
            if !segment.sorted_by_src {
                return Err(AnnError::log_index_error(format!(
                    "Parallel spill replay requires source-sorted segment: shard={} file_id={} byte_offset={}",
                    segment.shard_id, segment.file_id, segment.byte_offset
                )));
            }
            self.reader.set_segment(segment)?;
            self.segment_active = true;
            replay_io.segments_scanned += 1;
            replay_io.logical_read_bytes = replay_io
                .logical_read_bytes
                .saturating_add(segment.byte_len);
        }
    }

    fn read_next_batch_parallel_decode(
        &mut self,
        max_records: usize,
        output: &mut Vec<PendingEdge>,
        replay_io: &mut SpillReplayIoStats,
        decoder_pool: &rayon::ThreadPool,
        scratch: &mut ParallelDecodeScratch,
    ) -> AnnResult<usize> {
        loop {
            if self.segment_active {
                let count = self.reader.read_sorted_edge_batch_parallel_decode(
                    max_records,
                    output,
                    decoder_pool,
                    scratch,
                )?;
                if count > 0 {
                    return Ok(count);
                }
                self.segment_active = false;
            }

            let Some(segment) = self.segments.get(self.next_segment).copied() else {
                output.clear();
                return Ok(0);
            };
            self.next_segment += 1;
            if !segment.sorted_by_src {
                return Err(AnnError::log_index_error(format!(
                    "Parallel spill replay requires source-sorted segment: shard={} file_id={} byte_offset={}",
                    segment.shard_id, segment.file_id, segment.byte_offset
                )));
            }
            self.reader.set_segment(segment)?;
            self.segment_active = true;
            replay_io.segments_scanned += 1;
            replay_io.logical_read_bytes = replay_io
                .logical_read_bytes
                .saturating_add(segment.byte_len);
        }
    }

    fn read_next_compact_batch(
        &mut self,
        max_records: usize,
        output: &mut Vec<u8>,
        replay_io: &mut SpillReplayIoStats,
    ) -> AnnResult<usize> {
        loop {
            if self.segment_active {
                let count = self.reader.read_sorted_compact_batch(max_records, output)?;
                if count > 0 {
                    return Ok(count);
                }
                self.segment_active = false;
            }

            let Some(segment) = self.segments.get(self.next_segment).copied() else {
                return Ok(0);
            };
            self.next_segment += 1;
            if !segment.sorted_by_src {
                return Err(AnnError::log_index_error(format!(
                    "Parallel spill replay requires source-sorted segment: shard={} file_id={} byte_offset={}",
                    segment.shard_id, segment.file_id, segment.byte_offset
                )));
            }
            self.reader.set_segment(segment)?;
            self.segment_active = true;
            replay_io.segments_scanned += 1;
            replay_io.logical_read_bytes = replay_io
                .logical_read_bytes
                .saturating_add(segment.byte_len);
        }
    }

    fn stats(&self) -> SegmentReaderStats {
        self.reader.stats()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;

    use parking_lot::Mutex;
    use tempfile::tempdir;

    use super::{
        CandidateRunWriter, ExternalHashPruneReducer, ExternalParallelReplayMode,
        ExternalReduceConfig, ExternalReduceThreadPool, ExternalReduceWorkerControl,
        ParallelDecodeScratch, RUN_RECORD_BYTES, SegmentReader, SpillArtifacts, SpillEdgeSink,
        SpillPartFileDesc, SpillRecord16, SpillSegmentDesc, external_parallel_replay_mode,
        insert_parallel_backfill_batch_while, source_disjoint_edge_ranges,
    };
    use crate::algorithm::pipnn::direct_io::DirectIoConfig;
    use crate::algorithm::pipnn::hash_prune::HashPruneReservoir;
    use crate::algorithm::pipnn::hash_prune_backfill::{
        HashPruneBackfillPolicy, HashPruneBackfillProfile,
    };
    use crate::algorithm::pipnn::hash_prune_evidence::{
        HashPruneEvidenceClass, HashPruneEvidencePolicy, HashPruneEvidenceProfile, HashPruneMeta,
    };
    use crate::algorithm::pipnn::leaf_build::{
        PendingEdge, PendingEdgeSink, flush_pending_edges_to_reservoirs_with_policy,
    };
    use crate::algorithm::pipnn::params::{
        PiPNNHashPruneBackfillAblation, PiPNNHashPruneBackfillMode, PiPNNParams,
    };

    #[test]
    fn production_external_replay_mode_defaults_buffered_to_compact_fused() {
        let rhalf = HashPruneBackfillPolicy::diagnostic_soft_tier_q3(1.02).unwrap();
        let buffered = ExternalReduceConfig::new(DirectIoConfig::disabled(), 8, 64 * 1024);
        let direct =
            ExternalReduceConfig::new(DirectIoConfig::enabled_with_alignment(4096), 8, 64 * 1024);

        assert_eq!(
            external_parallel_replay_mode(buffered, HashPruneEvidencePolicy::off(), rhalf, false,),
            Some(ExternalParallelReplayMode::CompactFused),
        );
        assert_eq!(
            external_parallel_replay_mode(direct, HashPruneEvidencePolicy::off(), rhalf, false,),
            Some(ExternalParallelReplayMode::DecodedPipeline),
        );
        assert_eq!(
            external_parallel_replay_mode(
                ExternalReduceConfig::new(DirectIoConfig::disabled(), 1, 64 * 1024),
                HashPruneEvidencePolicy::off(),
                rhalf,
                false,
            ),
            None,
        );
        assert_eq!(
            external_parallel_replay_mode(
                buffered,
                HashPruneEvidencePolicy::report(),
                rhalf,
                false,
            ),
            None,
        );
        assert_eq!(
            external_parallel_replay_mode(
                buffered,
                HashPruneEvidencePolicy::off(),
                HashPruneBackfillPolicy::off(),
                false,
            ),
            None,
        );
        assert_eq!(
            external_parallel_replay_mode(buffered, HashPruneEvidencePolicy::off(), rhalf, true,),
            None,
        );
    }

    fn reduce_in_memory(num_points: usize, l_max: usize, edges: &[PendingEdge]) -> Vec<Vec<u32>> {
        let mut reservoirs: Vec<_> = (0..num_points)
            .map(|_| HashPruneReservoir::new(l_max))
            .collect();
        for edge in edges {
            reservoirs[edge.p].insert(edge.c, edge.hash, edge.dist);
        }
        reservoirs
            .into_iter()
            .map(|reservoir| {
                let mut neighbors = reservoir.into_neighbors();
                neighbors.sort_unstable();
                neighbors
            })
            .collect()
    }

    fn backfill_candidate_fixture() -> Vec<PendingEdge> {
        [
            (0, 102, 10, 1.3),
            (3, 310, 30, 1.0),
            (0, 200, 20, 2.0),
            (5, 510, 50, 1.0),
            (1, 110, 10, 1.0),
            (0, 100, 10, 1.0),
            (3, 311, 30, 1.1),
            (0, 201, 20, 2.1),
            (1, 111, 10, 1.05),
            (5, 520, 60, 1.5),
            (0, 101, 10, 1.1),
        ]
        .into_iter()
        .map(|(p, c, hash, dist)| PendingEdge {
            p,
            c,
            hash,
            dist,
            meta: HashPruneMeta::empty(),
        })
        .collect()
    }

    fn reduce_backfill_in_memory(
        num_points: usize,
        l_max: usize,
        edges: &[PendingEdge],
        policy: HashPruneBackfillPolicy,
    ) -> (Vec<Vec<u32>>, HashPruneBackfillProfile) {
        let mut reservoirs = (0..num_points)
            .map(|_| HashPruneReservoir::new(l_max))
            .collect::<Vec<_>>();
        let mut profile = HashPruneBackfillProfile::default();
        for edge in edges {
            reservoirs[edge.p]
                .insert_backfill_with_policy(
                    edge.p,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    policy,
                    &mut profile,
                )
                .unwrap();
        }
        let mut rows = vec![Vec::new(); num_points];
        for (point, reservoir) in reservoirs.into_iter().enumerate() {
            rows[point] = reservoir
                .into_neighbors_with_backfill(point, policy, &mut profile)
                .unwrap();
        }
        (rows, profile)
    }

    fn tiered_rhalf_policy(ablation: PiPNNHashPruneBackfillAblation) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::DistanceGatedTieredRHalf;
        params.hashprune_backfill_ablation = ablation;
        params.hashprune_backfill_alpha = 1.1;
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    fn named_rhalf_policy(mode: PiPNNHashPruneBackfillMode) -> HashPruneBackfillPolicy {
        named_rhalf_policy_with_row_cap(mode, None)
    }

    fn named_rhalf_policy_with_row_cap(
        mode: PiPNNHashPruneBackfillMode,
        cap: Option<usize>,
    ) -> HashPruneBackfillPolicy {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = mode;
        params.hashprune_rhalf_max_backfill_per_row = cap;
        HashPruneBackfillPolicy::from_params(&params).unwrap()
    }

    #[test]
    fn hashprune_backfill_spill_record_remains_16_bytes_and_marker_free() {
        assert_eq!(std::mem::size_of::<SpillRecord16>(), 16);
        let edge = PendingEdge {
            p: 0,
            c: 10,
            hash: 7,
            dist: 1.0,
            meta: HashPruneMeta::empty(),
        };
        let record = SpillRecord16::from_pending_edge(&edge).unwrap();
        assert_eq!(record.flags, 0);
    }

    #[test]
    fn hashprune_backfill_internal_marker_cannot_escape_to_spill() {
        let edge = PendingEdge {
            p: 0,
            c: 10,
            hash: 7,
            dist: 1.0,
            meta: HashPruneMeta::empty().with_backfill_count(1).unwrap(),
        };
        let error = SpillRecord16::from_pending_edge(&edge).unwrap_err();
        assert!(error.to_string().contains("reserved bits"));
    }

    #[test]
    fn hashprune_backfill_spill_and_fragmented_replay_match_resident() {
        let num_points = 6;
        let l_max = 4;
        let edges = backfill_candidate_fixture();
        for policy in [
            HashPruneBackfillPolicy::distance_gated(8, 1.1),
            HashPruneBackfillPolicy::distance_gated_stratified(4, 1.1),
            HashPruneBackfillPolicy::distance_gated_multihit(6, 1.1),
            HashPruneBackfillPolicy::distance_gated_tiered_d52(8, 1.1),
            named_rhalf_policy(PiPNNHashPruneBackfillMode::RHalfHashSlot1),
            named_rhalf_policy(PiPNNHashPruneBackfillMode::RHalfDistanceFirstQ3),
            named_rhalf_policy(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3),
            named_rhalf_policy_with_row_cap(PiPNNHashPruneBackfillMode::RHalfSoftTierQ3, Some(2)),
            named_rhalf_policy(PiPNNHashPruneBackfillMode::RHalfHardTierQ3),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::None),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::BudgetQuarter),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoHalfCap),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::Quota1),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::Quota2),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::DistanceFirst),
            tiered_rhalf_policy(PiPNNHashPruneBackfillAblation::NoVeto),
        ] {
            let expected = reduce_backfill_in_memory(num_points, l_max, &edges, policy);

            for boundary in [1usize, 2, 5] {
                let dir = tempdir().unwrap();
                let sink = SpillEdgeSink::create_sharded_with_config_and_policies(
                    &dir.path().join("candidate.run"),
                    boundary * std::mem::size_of::<PendingEdge>(),
                    boundary,
                    DirectIoConfig::disabled(),
                    HashPruneEvidencePolicy::off(),
                    policy,
                )
                .unwrap();
                for batch in edges.chunks(boundary) {
                    let mut batch = batch.to_vec();
                    sink.flush_pending_edges(&mut batch).unwrap();
                    assert!(batch.is_empty());
                }
                let artifacts = sink.finish().unwrap();
                let mut actual_rows = vec![Vec::new(); num_points];
                let actual_profile = ExternalHashPruneReducer::
                    reduce_spill_artifacts_sharded_in_order_profiled_with_policies(
                        num_points,
                        l_max,
                        boundary,
                        &artifacts,
                        HashPruneEvidencePolicy::off(),
                        policy,
                        |point, neighbors| {
                            actual_rows[point as usize] = neighbors;
                            Ok(())
                        },
                    )
                    .unwrap()
                    .backfill;
                assert_eq!(
                    actual_rows, expected.0,
                    "policy={policy:?} boundary={boundary}"
                );
                assert_eq!(
                    actual_profile, expected.1,
                    "policy={policy:?} boundary={boundary}"
                );
            }
        }
    }

    #[test]
    fn parallel_softtier_replay_matches_serial_across_batches_and_segments() {
        let num_points = 16;
        let l_max = 8;
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_rhalf_max_backfill_per_row = Some(3);
        params.hashprune_soft_tier_priority_alpha = Some(1.02);
        let policy = HashPruneBackfillPolicy::from_params(&params).unwrap();

        let dir = tempdir().unwrap();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policies(
            &dir.path().join("parallel-candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            num_points,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::off(),
            policy,
        )
        .unwrap();
        for round in 0..4_u32 {
            let mut batch = Vec::with_capacity(num_points * 2);
            for source in 0..num_points {
                batch.push(PendingEdge {
                    p: source,
                    c: 1000 + source as u32 * 16 + round,
                    hash: (source % 7) as u16,
                    dist: 1.0 + round as f32 * 0.1,
                    meta: HashPruneMeta::empty(),
                });
                batch.push(PendingEdge {
                    p: source,
                    c: 2000 + source as u32 * 16 + round,
                    hash: 100 + (source % 5) as u16,
                    dist: 1.5 + round as f32 * 0.1,
                    meta: HashPruneMeta::empty(),
                });
            }
            sink.flush_pending_edges(&mut batch).unwrap();
        }
        let artifacts = sink.finish().unwrap();
        assert!(artifacts.segment_manifest.len() >= 2);

        let replay = |config: ExternalReduceConfig| {
            let mut rows = vec![Vec::new(); num_points];
            let mut order = Vec::with_capacity(num_points);
            let stats = ExternalHashPruneReducer::
                reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config(
                    num_points,
                    l_max,
                    num_points,
                    &artifacts,
                    HashPruneEvidencePolicy::off(),
                    policy,
                    None,
                    None,
                    config,
                    |point, neighbors| {
                        order.push(point);
                        rows[point as usize] = neighbors;
                        Ok(())
                    },
                )
                .unwrap();
            (rows, order, stats)
        };

        let (serial_rows, serial_order, serial_stats) =
            replay(ExternalReduceConfig::new(DirectIoConfig::disabled(), 1, 3));
        let (parallel_rows, parallel_order, parallel_stats) =
            replay(ExternalReduceConfig::new(DirectIoConfig::disabled(), 4, 3));

        assert_eq!(parallel_rows, serial_rows);
        assert_eq!(parallel_order, serial_order);
        assert_eq!(parallel_order, (0..num_points as u32).collect::<Vec<_>>());
        assert_eq!(parallel_stats.backfill, serial_stats.backfill);
        assert_eq!(
            parallel_stats.logical_read_bytes,
            serial_stats.logical_read_bytes
        );
        assert_eq!(
            parallel_stats.reader_open_calls,
            serial_stats.reader_open_calls
        );
        assert_eq!(serial_stats.parallel_workers, 1);
        assert_eq!(serial_stats.parallel_batches, 0);
        assert_eq!(serial_stats.parallel_records, 0);
        assert!(parallel_stats.parallel_batches > 0);
        assert!(parallel_stats.parallel_records > 0);
        assert_eq!(parallel_stats.decoder_workers, 0);
        assert_eq!(parallel_stats.parallel_workers, 4);
        assert!(parallel_stats.pipeline_wall > Duration::ZERO);
        assert!(parallel_stats.read_decode_wall > Duration::ZERO);
        assert!(parallel_stats.record_read_wall > Duration::ZERO);
        assert!(parallel_stats.record_decode_wall > Duration::ZERO);
        assert_eq!(parallel_stats.record_append_wall, Duration::ZERO);
        assert!(
            parallel_stats.record_read_wall
                + parallel_stats.record_decode_wall
                + parallel_stats.record_append_wall
                <= parallel_stats.read_decode_wall
        );
        assert!(parallel_stats.insert_wall > Duration::ZERO);
        assert!(
            parallel_stats.pipeline_overlap_wall
                <= parallel_stats.read_decode_wall + parallel_stats.insert_wall
        );
        assert_eq!(serial_stats.pipeline_wall, Duration::ZERO);
        assert_eq!(serial_stats.pipeline_overlap_wall, Duration::ZERO);
    }

    #[test]
    fn adaptive_external_reduce_scales_only_when_insertion_lags_reading() {
        let mut control = ExternalReduceWorkerControl::new(8, 64);
        for _ in 0..63 {
            assert_eq!(
                control.observe(Duration::from_millis(30), Duration::from_millis(10)),
                8
            );
        }
        assert_eq!(
            control.observe(Duration::from_millis(30), Duration::from_millis(10)),
            16
        );
        for _ in 0..63 {
            assert_eq!(
                control.observe(Duration::from_millis(10), Duration::from_millis(30)),
                16
            );
        }
        assert_eq!(
            control.observe(Duration::from_millis(10), Duration::from_millis(30)),
            8
        );
    }

    #[test]
    fn adaptive_external_reduce_pool_starts_small_and_grows_on_demand() {
        let mut pool = ExternalReduceThreadPool::new(2).unwrap();
        assert_eq!(pool.workers(), 2);
        pool.resize(8).unwrap();
        assert_eq!(pool.workers(), 8);
    }

    #[test]
    fn parallel_softtier_replay_drains_before_next_batch_read_error() {
        let mut params = PiPNNParams::default();
        params.hashprune_backfill_mode = PiPNNHashPruneBackfillMode::RHalfSoftTierQ3;
        params.hashprune_rhalf_max_backfill_per_row = Some(3);
        params.hashprune_soft_tier_priority_alpha = Some(1.02);
        let policy = HashPruneBackfillPolicy::from_params(&params).unwrap();
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt_second_batch.run");
        let records = [
            SpillRecord16 {
                src: 0,
                dst: 1,
                hash: 1,
                flags: 0,
                dist: 1.0,
            },
            SpillRecord16 {
                src: 0,
                dst: 1,
                hash: 2,
                flags: 1 << 6,
                dist: 2.0,
            },
        ];
        let mut bytes = Vec::with_capacity(records.len() * RUN_RECORD_BYTES);
        for record in records {
            bytes.extend_from_slice(&record.src.to_le_bytes());
            bytes.extend_from_slice(&record.dst.to_le_bytes());
            bytes.extend_from_slice(&record.hash.to_le_bytes());
            bytes.extend_from_slice(&record.flags.to_le_bytes());
            bytes.extend_from_slice(&record.dist.to_le_bytes());
        }
        fs::write(&path, bytes).unwrap();
        let artifacts = SpillArtifacts {
            part_files: vec![SpillPartFileDesc {
                shard_id: 0,
                file_id: 0,
                path: path.clone(),
                byte_len: (2 * RUN_RECORD_BYTES) as u64,
            }],
            segment_manifest: vec![SpillSegmentDesc {
                shard_id: 0,
                file_id: 0,
                byte_offset: 0,
                byte_len: (2 * RUN_RECORD_BYTES) as u64,
                record_count: 2,
                src_min: 0,
                src_max: 0,
                sorted_by_src: true,
            }],
            manifest_path: dir.path().join("candidate_segments.manifest"),
            policy: HashPruneEvidencePolicy::off(),
            backfill_policy: policy,
        };

        let error = ExternalHashPruneReducer::
            reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config(
                2,
                8,
                2,
                &artifacts,
                HashPruneEvidencePolicy::off(),
                policy,
                None,
                None,
                ExternalReduceConfig::new(DirectIoConfig::disabled(), 2, 1),
                |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("reserved"));
    }

    #[test]
    fn source_disjoint_ranges_never_split_equal_sources() {
        let edges = [0, 0, 1, 2, 2, 2, 3, 4, 4]
            .into_iter()
            .enumerate()
            .map(|(index, source)| PendingEdge {
                p: source,
                c: 100 + index as u32,
                hash: index as u16,
                dist: index as f32,
                meta: HashPruneMeta::empty(),
            })
            .collect::<Vec<_>>();

        let ranges = source_disjoint_edge_ranges(&edges, 4);
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, edges.len());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert_ne!(edges[pair[0].end - 1].p, edges[pair[1].start].p);
        }
    }

    #[test]
    fn scoped_backfill_runs_concurrent_work_on_caller_and_inserts_all_ranges() {
        let mut reservoirs = (0..4)
            .map(|_| HashPruneReservoir::new(8))
            .collect::<Vec<_>>();
        let edges = (0..4)
            .flat_map(|source| {
                (0..4).map(move |rank| PendingEdge {
                    p: source,
                    c: 100 + source as u32 * 4 + rank,
                    hash: (source * 16 + rank as usize) as u16,
                    dist: rank as f32,
                    meta: HashPruneMeta::empty(),
                })
            })
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let caller = std::thread::current().id();
        let mut profile = HashPruneBackfillProfile::default();
        let (observed_caller, insert_wall) = insert_parallel_backfill_batch_while(
            &pool,
            0,
            4,
            &mut reservoirs,
            &edges,
            4,
            HashPruneBackfillPolicy::off(),
            &mut profile,
            || std::thread::current().id(),
        )
        .unwrap();

        assert_eq!(observed_caller, caller);
        assert!(insert_wall > Duration::ZERO);
        let mut neighbors = reservoirs
            .into_iter()
            .map(|reservoir| {
                let mut row = reservoir.into_neighbors();
                row.sort_unstable();
                row
            })
            .collect::<Vec<_>>();
        for (source, row) in neighbors.iter_mut().enumerate() {
            assert_eq!(
                *row,
                (0..4)
                    .map(|rank| 100 + source as u32 * 4 + rank)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn unsupported_parallel_policy_falls_back_to_serial_replay() {
        let dir = tempdir().unwrap();
        let policy = HashPruneBackfillPolicy::off();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policies(
            &dir.path().join("serial-fallback-candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            2,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::off(),
            policy,
        )
        .unwrap();
        let mut edges = vec![PendingEdge {
            p: 0,
            c: 1,
            hash: 3,
            dist: 1.0,
            meta: HashPruneMeta::empty(),
        }];
        sink.flush_pending_edges(&mut edges).unwrap();
        let artifacts = sink.finish().unwrap();

        let stats = ExternalHashPruneReducer::
            reduce_spill_artifacts_sharded_in_order_profiled_with_execution_config(
                2,
                4,
                2,
                &artifacts,
                HashPruneEvidencePolicy::off(),
                policy,
                None,
                None,
                ExternalReduceConfig::new(DirectIoConfig::disabled(), 4, 8),
                |_, _| Ok(()),
            )
            .unwrap();

        assert_eq!(stats.parallel_workers, 1);
        assert_eq!(stats.parallel_batches, 0);
        assert_eq!(stats.parallel_records, 0);
    }

    #[test]
    fn hashprune_backfill_spill_policy_mismatch_is_an_error() {
        let dir = tempdir().unwrap();
        let policy = HashPruneBackfillPolicy::distance_gated(8, 1.1);
        let artifacts = SpillEdgeSink::create_sharded_with_config_and_policies(
            &dir.path().join("candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            1,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::off(),
            policy,
        )
        .unwrap()
        .finish()
        .unwrap();

        let error = ExternalHashPruneReducer::
            reduce_spill_artifacts_sharded_in_order_profiled_with_policies(
                1,
                4,
                1,
                &artifacts,
                HashPruneEvidencePolicy::off(),
                HashPruneBackfillPolicy::off(),
                |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("policy mismatch"));
    }

    fn build_artifacts_from_paths(
        paths: &[PathBuf],
        shard_ids: &[usize],
        point_ranges: &[(u32, u32)],
    ) -> SpillArtifacts {
        let part_files = paths
            .iter()
            .enumerate()
            .map(|(idx, path)| SpillPartFileDesc {
                shard_id: shard_ids[idx],
                file_id: idx as u16,
                path: path.clone(),
                byte_len: fs::metadata(path).unwrap().len(),
            })
            .collect();
        let segment_manifest = paths
            .iter()
            .enumerate()
            .map(|(idx, path)| SpillSegmentDesc {
                shard_id: shard_ids[idx] as u32,
                file_id: idx as u16,
                byte_offset: 0,
                byte_len: fs::metadata(path).unwrap().len(),
                record_count: (fs::metadata(path).unwrap().len() as usize / RUN_RECORD_BYTES)
                    as u32,
                src_min: point_ranges[idx].0,
                src_max: point_ranges[idx].1,
                sorted_by_src: true,
            })
            .collect();
        SpillArtifacts {
            part_files,
            segment_manifest,
            manifest_path: paths
                .first()
                .and_then(|path| path.parent())
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("candidate_segments.manifest"),
            policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
        }
    }

    fn read_artifact_edges(artifacts: &SpillArtifacts) -> Vec<PendingEdge> {
        let mut by_part = HashMap::new();
        for part in &artifacts.part_files {
            by_part.insert((part.shard_id, part.file_id), part.path.clone());
        }
        let mut edges = Vec::new();
        for segment in &artifacts.segment_manifest {
            let path = by_part
                .get(&(segment.shard_id as usize, segment.file_id))
                .expect("segment must have part path");
            let mut reader =
                SegmentReader::open(path, *segment, DirectIoConfig::disabled()).unwrap();
            while let Some(edge) = reader.read_edge().unwrap() {
                edges.push(edge);
            }
        }
        edges
    }

    #[test]
    fn spill_record16_is_fixed_width_and_round_trips_hashprune_metadata() {
        assert_eq!(RUN_RECORD_BYTES, 16);
        assert_eq!(std::mem::size_of::<SpillRecord16>(), 16);

        for class in HashPruneEvidenceClass::ALL {
            let meta = HashPruneMeta::mandatory()
                .merge_candidate(HashPruneMeta::with_evidence(class))
                .unwrap();
            let edge = PendingEdge {
                p: 7,
                c: 42,
                hash: 9,
                dist: 0.125,
                meta,
            };
            let record = SpillRecord16::from_pending_edge(&edge).unwrap();
            let round_trip = record.to_pending_edge().unwrap();

            assert_eq!(round_trip.p, edge.p);
            assert_eq!(round_trip.c, edge.c);
            assert_eq!(round_trip.hash, edge.hash);
            assert_eq!(round_trip.dist, edge.dist);
            assert_eq!(round_trip.meta, meta);
        }
    }

    #[test]
    fn spill_record16_rejects_internal_overflow_and_reserved_flags() {
        let base = SpillRecord16 {
            src: 1,
            dst: 2,
            hash: 3,
            flags: 0,
            dist: 4.0,
        };

        assert!(
            SpillRecord16 {
                flags: 1 << 5,
                ..base
            }
            .to_pending_edge()
            .is_err()
        );
        assert!(
            SpillRecord16 {
                flags: 1 << 6,
                ..base
            }
            .to_pending_edge()
            .is_err()
        );
    }

    #[test]
    fn spill_evidence_candidate_run_writer_rejects_overflow_before_writing_record() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_overflow.run");
        let edge = PendingEdge {
            p: 0,
            c: 1,
            hash: 2,
            dist: 0.25,
            meta: HashPruneMeta::empty().mark_overflow(),
        };

        let mut writer = CandidateRunWriter::create(&run_path).unwrap();
        let error = writer.push(&edge).unwrap_err();
        assert!(error.to_string().contains("overflow"));
        writer.finish().unwrap();

        assert_eq!(RUN_RECORD_BYTES, 16);
        assert_eq!(fs::metadata(&run_path).unwrap().len(), 0);
    }

    #[test]
    fn candidate_run_writer_uses_fixed_16_byte_records() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 1,
                dist: 0.1,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 2,
                c: 3,
                hash: 2,
                dist: 0.2,
                meta: HashPruneMeta::mandatory(),
            },
            PendingEdge {
                p: 4,
                c: 5,
                hash: 3,
                dist: 0.3,
                meta: HashPruneMeta::empty(),
            },
        ];

        let mut writer = CandidateRunWriter::create(&run_path).unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let metadata = fs::metadata(&run_path).unwrap();
        assert_eq!(metadata.len(), (edges.len() * RUN_RECORD_BYTES) as u64);
    }

    #[test]
    fn candidate_writer_flush_visible_allows_segment_replay_before_finish() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("online_fallback.run");
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for source in 0..3_usize {
            writer
                .push(&PendingEdge {
                    p: source,
                    c: source as u32 + 10,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.flush_visible().unwrap();
        let desc = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: 3 * RUN_RECORD_BYTES as u64,
            record_count: 3,
            src_min: 0,
            src_max: 2,
            sorted_by_src: true,
        };
        let mut observed = Vec::new();
        super::replay_single_spill_segment(&path, desc, DirectIoConfig::disabled(), |edge| {
            observed.push((edge.p, edge.c));
            Ok(())
        })
        .unwrap();
        assert_eq!(observed, vec![(0, 10), (1, 11), (2, 12)]);
        writer.finish().unwrap();
    }

    #[test]
    fn candidate_run_writer_strict_io_round_trips_fixed_records() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_strict.run");
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 1,
                dist: 0.1,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 2,
                c: 3,
                hash: 2,
                dist: 0.2,
                meta: HashPruneMeta::mandatory(),
            },
            PendingEdge {
                p: 4,
                c: 5,
                hash: 3,
                dist: 0.3,
                meta: HashPruneMeta::empty(),
            },
        ];

        let mut writer = CandidateRunWriter::create_with_config(
            &run_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let metadata = fs::metadata(&run_path).unwrap();
        assert_eq!(metadata.len(), (edges.len() * RUN_RECORD_BYTES) as u64);

        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: metadata.len(),
            record_count: edges.len() as u32,
            src_min: 0,
            src_max: 4,
            sorted_by_src: true,
        };
        let mut reader = super::SegmentReader::open(
            &run_path,
            segment,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let mut round_trip = Vec::new();
        while let Some(edge) = reader.read_edge().unwrap() {
            round_trip.push(edge);
        }
        let round_trip_fields: Vec<_> = round_trip
            .iter()
            .map(|edge| {
                (
                    edge.p,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    edge.meta.is_mandatory(),
                )
            })
            .collect();
        let edge_fields: Vec<_> = edges
            .iter()
            .map(|edge| {
                (
                    edge.p,
                    edge.c,
                    edge.hash,
                    edge.dist,
                    edge.meta.is_mandatory(),
                )
            })
            .collect();
        assert_eq!(round_trip_fields, edge_fields);
    }

    #[test]
    fn candidate_segment_strict_io_batches_record_reads() {
        use crate::algorithm::pipnn::direct_io::{
            direct_io_test_read_bytes, direct_io_test_read_calls, reset_direct_io_test_counters,
        };

        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_strict_batched_read.run");
        let record_count = 8192_u32;
        let mut writer = CandidateRunWriter::create_with_config(
            &run_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        for point in 0..record_count {
            writer
                .push(&PendingEdge {
                    p: point as usize,
                    c: point + 1,
                    hash: (point % 4096) as u16,
                    dist: point as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();

        let byte_len = u64::from(record_count) * RUN_RECORD_BYTES as u64;
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len,
            record_count,
            src_min: 0,
            src_max: record_count - 1,
            sorted_by_src: true,
        };
        reset_direct_io_test_counters();
        let mut reader = SegmentReader::open(
            &run_path,
            segment,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let mut observed = 0_u32;
        while reader.read_edge().unwrap().is_some() {
            observed += 1;
        }

        assert_eq!(observed, record_count);
        assert!(
            direct_io_test_read_calls() <= 2,
            "strict candidate reader should refill large aligned blocks, saw {} reads",
            direct_io_test_read_calls()
        );
        assert!(
            direct_io_test_read_bytes() <= 2 * super::DIRECT_CANDIDATE_BUFFER_BYTES,
            "strict candidate reader fetched too many aligned bytes: {}",
            direct_io_test_read_bytes()
        );
    }

    #[test]
    fn candidate_segment_strict_io_reads_unaligned_manifest_range_exactly() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_strict_unaligned_segment.run");
        let io_cfg = DirectIoConfig::enabled_with_alignment(4096);
        let mut writer = CandidateRunWriter::create_with_config(&run_path, io_cfg).unwrap();
        for point in 0..512_u32 {
            writer
                .push(&PendingEdge {
                    p: point as usize,
                    c: point + 1000,
                    hash: point as u16,
                    dist: point as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();

        let first = 7_u32;
        let count = 123_u32;
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: u64::from(first) * RUN_RECORD_BYTES as u64,
            byte_len: u64::from(count) * RUN_RECORD_BYTES as u64,
            record_count: count,
            src_min: first,
            src_max: first + count - 1,
            sorted_by_src: true,
        };
        let mut reader = SegmentReader::open(&run_path, segment, io_cfg).unwrap();
        let mut points = Vec::new();
        while let Some(edge) = reader.read_edge().unwrap() {
            points.push(edge.p as u32);
        }

        assert_eq!(points, (first..first + count).collect::<Vec<_>>());
        assert_eq!(reader.stats().direct_read_calls, 1);
    }

    #[test]
    fn candidate_segment_bulk_reader_matches_scalar_across_unaligned_refill() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("candidate_bulk_unaligned.run");
        let io_cfg = DirectIoConfig::enabled_with_alignment(4096);
        let total = 65_800_u32;
        let first = 7_u32;
        let count = 65_700_u32;
        let mut writer = CandidateRunWriter::create_with_config(&path, io_cfg).unwrap();
        for source in 0..total {
            writer
                .push(&PendingEdge {
                    p: source as usize,
                    c: source + 1000,
                    hash: (source % 4096) as u16,
                    dist: source as f32 * 0.25,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: u64::from(first) * RUN_RECORD_BYTES as u64,
            byte_len: u64::from(count) * RUN_RECORD_BYTES as u64,
            record_count: count,
            src_min: first,
            src_max: first + count - 1,
            sorted_by_src: true,
        };

        let fields = |edge: &PendingEdge| (edge.p, edge.c, edge.hash, edge.dist, edge.meta);
        let mut scalar_reader = SegmentReader::open(&path, segment, io_cfg).unwrap();
        let mut scalar = Vec::with_capacity(count as usize);
        while let Some(edge) = scalar_reader.read_edge().unwrap() {
            scalar.push(fields(&edge));
        }

        let mut bulk_reader = SegmentReader::open(&path, segment, io_cfg).unwrap();
        let mut batch = Vec::new();
        let mut bulk = Vec::with_capacity(count as usize);
        loop {
            let read = bulk_reader
                .read_sorted_edge_batch_bulk(10_003, &mut batch)
                .unwrap();
            if read == 0 {
                break;
            }
            bulk.extend(batch.iter().map(fields));
        }

        assert_eq!(bulk, scalar);
        assert_eq!(bulk.len(), count as usize);
        assert!(bulk_reader.stats().direct_read_calls >= 2);

        let decoder_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let mut scratch = ParallelDecodeScratch::default();
        let mut parallel_reader = SegmentReader::open(&path, segment, io_cfg).unwrap();
        let mut parallel = Vec::with_capacity(count as usize);
        loop {
            let read = parallel_reader
                .read_sorted_edge_batch_parallel_decode(
                    10_003,
                    &mut batch,
                    &decoder_pool,
                    &mut scratch,
                )
                .unwrap();
            if read == 0 {
                break;
            }
            parallel.extend(batch.iter().map(fields));
        }

        assert_eq!(decoder_pool.current_num_threads(), 2);
        assert_eq!(parallel, scalar);
        assert_eq!(
            parallel_reader.stats().direct_read_calls,
            bulk_reader.stats().direct_read_calls
        );
    }

    #[test]
    fn two_decoder_reader_rejects_disorder_across_lane_boundary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("candidate_two_decoder_disorder.run");
        let io_cfg = DirectIoConfig::enabled_with_alignment(4096);
        let mut writer = CandidateRunWriter::create_with_config(&path, io_cfg).unwrap();
        for source in [0_usize, 1, 4, 3, 5, 6] {
            writer
                .push(&PendingEdge {
                    p: source,
                    c: 100 + source as u32,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: 6 * RUN_RECORD_BYTES as u64,
            record_count: 6,
            src_min: 0,
            src_max: 6,
            sorted_by_src: true,
        };
        let decoder_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let mut reader = SegmentReader::open(&path, segment, io_cfg).unwrap();
        let mut scratch = ParallelDecodeScratch::default();
        let mut batch = Vec::new();

        let error = reader
            .read_sorted_edge_batch_parallel_decode(6, &mut batch, &decoder_pool, &mut scratch)
            .unwrap_err();
        assert!(error.to_string().contains("4 before 3"));
    }

    #[test]
    fn sorted_bulk_reader_rejects_disorder_across_batch_boundary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("candidate_cross_batch_disorder.run");
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for source in [0_usize, 2, 1] {
            writer
                .push(&PendingEdge {
                    p: source,
                    c: 100 + source as u32,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: 3 * RUN_RECORD_BYTES as u64,
            record_count: 3,
            src_min: 0,
            src_max: 2,
            sorted_by_src: true,
        };
        let mut reader = SegmentReader::open(&path, segment, DirectIoConfig::disabled()).unwrap();
        let mut batch = Vec::new();
        assert_eq!(
            reader.read_sorted_edge_batch_bulk(2, &mut batch).unwrap(),
            2
        );
        let error = reader
            .read_sorted_edge_batch_bulk(2, &mut batch)
            .unwrap_err();
        assert!(error.to_string().contains("2 before 1"));
    }

    #[test]
    fn compact_reader_rejects_disorder_across_batch_boundary() {
        let dir = tempdir().unwrap();
        let path = dir
            .path()
            .join("candidate_compact_cross_batch_disorder.run");
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for source in [0_usize, 2, 1] {
            writer
                .push(&PendingEdge {
                    p: source,
                    c: 100 + source as u32,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: 3 * RUN_RECORD_BYTES as u64,
            record_count: 3,
            src_min: 0,
            src_max: 2,
            sorted_by_src: true,
        };
        let mut reader = SegmentReader::open(&path, segment, DirectIoConfig::disabled()).unwrap();
        let mut batch = Vec::new();

        assert_eq!(reader.read_sorted_compact_batch(2, &mut batch).unwrap(), 2);
        let error = reader.read_sorted_compact_batch(2, &mut batch).unwrap_err();
        assert!(error.to_string().contains("2 before 1"));
    }

    #[test]
    fn compact_reader_preserves_reusable_buffer_length() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("candidate_compact_reuse.run");
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for source in 0..3_usize {
            writer
                .push(&PendingEdge {
                    p: source,
                    c: 100 + source as u32,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: 3 * RUN_RECORD_BYTES as u64,
            record_count: 3,
            src_min: 0,
            src_max: 2,
            sorted_by_src: true,
        };
        let mut reader = SegmentReader::open(&path, segment, DirectIoConfig::disabled()).unwrap();
        let mut batch = vec![0_u8; 8 * RUN_RECORD_BYTES];

        assert_eq!(reader.read_sorted_compact_batch(2, &mut batch).unwrap(), 2);
        assert_eq!(batch.len(), 8 * RUN_RECORD_BYTES);
    }

    #[test]
    fn candidate_segment_batch_reader_preserves_order_and_bound() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("candidate_batch.run");
        let io_cfg = DirectIoConfig::enabled_with_alignment(4096);
        let mut writer = CandidateRunWriter::create_with_config(&path, io_cfg).unwrap();
        for source in 0..150_u32 {
            writer
                .push(&PendingEdge {
                    p: source as usize,
                    c: source + 1000,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: 150 * RUN_RECORD_BYTES as u64,
            record_count: 150,
            src_min: 0,
            src_max: 149,
            sorted_by_src: true,
        };
        let mut reader = SegmentReader::open(&path, segment, io_cfg).unwrap();
        let mut batch = Vec::new();
        let mut observed = Vec::new();
        loop {
            let count = reader.read_edge_batch(17, &mut batch).unwrap();
            assert!(batch.len() <= 17);
            if count == 0 {
                break;
            }
            observed.extend(batch.iter().map(|edge| edge.p as u32));
        }
        assert_eq!(observed, (0..150_u32).collect::<Vec<_>>());
    }

    #[test]
    fn part_batch_cursor_never_mixes_adjacent_segments() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cursor_segments.run");
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for source in [0_usize, 1, 2, 0, 1, 2] {
            writer
                .push(&PendingEdge {
                    p: source,
                    c: 100 + source as u32,
                    hash: source as u16,
                    dist: source as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let segment_bytes = 3 * RUN_RECORD_BYTES as u64;
        let segments = vec![
            SpillSegmentDesc {
                shard_id: 0,
                file_id: 0,
                byte_offset: 0,
                byte_len: segment_bytes,
                record_count: 3,
                src_min: 0,
                src_max: 2,
                sorted_by_src: true,
            },
            SpillSegmentDesc {
                shard_id: 0,
                file_id: 0,
                byte_offset: segment_bytes,
                byte_len: segment_bytes,
                record_count: 3,
                src_min: 0,
                src_max: 2,
                sorted_by_src: true,
            },
        ];
        let mut cursor =
            super::PartBatchCursor::open(&path, segments, DirectIoConfig::disabled()).unwrap();
        let mut io = super::SpillReplayIoStats::default();
        let mut batch = Vec::new();

        assert_eq!(cursor.read_next_batch(8, &mut batch, &mut io).unwrap(), 3);
        assert_eq!(
            batch.iter().map(|edge| edge.p).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(cursor.read_next_batch(8, &mut batch, &mut io).unwrap(), 3);
        assert_eq!(
            batch.iter().map(|edge| edge.p).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(cursor.read_next_batch(8, &mut batch, &mut io).unwrap(), 0);
        assert_eq!(io.segments_scanned, 2);
        assert_eq!(io.logical_read_bytes, 2 * segment_bytes);
    }

    #[test]
    fn configured_external_reduce_opens_each_part_once_and_reports_direct_reread() {
        let dir = tempdir().unwrap();
        let io_cfg = DirectIoConfig::enabled_with_alignment(4096);
        let sink = SpillEdgeSink::create_sharded_with_config_and_policies(
            &dir.path().join("candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            8,
            io_cfg,
            HashPruneEvidencePolicy::off(),
            HashPruneBackfillPolicy::off(),
        )
        .unwrap();
        for batch_start in [0_u32, 2, 4, 6] {
            let mut batch = (batch_start..batch_start + 2)
                .map(|point| PendingEdge {
                    p: point as usize,
                    c: point + 100,
                    hash: point as u16,
                    dist: point as f32,
                    meta: HashPruneMeta::empty(),
                })
                .collect::<Vec<_>>();
            sink.flush_pending_edges(&mut batch).unwrap();
        }
        let artifacts = sink.finish().unwrap();
        assert_eq!(artifacts.part_files.len(), 1);
        assert!(artifacts.segment_manifest.len() >= 2);
        let logical_bytes = artifacts
            .segment_manifest
            .iter()
            .map(|segment| segment.byte_len)
            .sum::<u64>();
        let mut rows = vec![Vec::new(); 8];

        let stats =
            ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_with_config(
                8,
                4,
                8,
                &artifacts,
                HashPruneEvidencePolicy::off(),
                HashPruneBackfillPolicy::off(),
                None,
                None,
                io_cfg,
                |point, neighbors| {
                    rows[point as usize] = neighbors;
                    Ok(())
                },
            )
            .unwrap();

        assert!(rows.iter().all(|row| row.len() == 1));
        assert!(stats.direct_io);
        assert_eq!(stats.logical_read_bytes, logical_bytes);
        assert_eq!(stats.reader_open_calls, 1);
        assert!(stats.direct_read_calls >= 1);
        assert!(stats.direct_read_calls < 8);
        assert!(stats.direct_read_bytes >= logical_bytes);
        assert_eq!(stats.workspace_points, 8);
        assert_eq!(
            stats.workspace_bytes,
            8 * (std::mem::size_of::<HashPruneReservoir>()
                + 4 * std::mem::size_of::<crate::algorithm::pipnn::hash_prune::Slot>())
        );
    }

    #[test]
    fn candidate_run_writer_strict_io_batches_records_before_direct_writes() {
        use crate::algorithm::pipnn::direct_io::{
            direct_io_test_write_calls, reset_direct_io_test_counters,
        };

        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_strict_batched.run");
        reset_direct_io_test_counters();

        let mut writer = CandidateRunWriter::create_with_config(
            &run_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        for i in 0..1000_u32 {
            writer
                .push(&PendingEdge {
                    p: i as usize,
                    c: i + 1,
                    hash: (i % 4096) as u16,
                    dist: i as f32,
                    meta: HashPruneMeta::empty(),
                })
                .unwrap();
        }
        writer.finish().unwrap();

        assert_eq!(
            fs::metadata(&run_path).unwrap().len(),
            (1000 * RUN_RECORD_BYTES) as u64
        );
        assert!(
            direct_io_test_write_calls() < 100,
            "strict candidate writer should batch records before direct I/O, saw {} writes",
            direct_io_test_write_calls()
        );
    }

    #[test]
    fn spill_digest_sink_bounds_each_source_row_before_writing() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create_digest_sharded_with_config(
            &run_path,
            1024 * 1024,
            16,
            2,
            DirectIoConfig::disabled(),
        )
        .unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 10,
                dist: 5.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 11,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 12,
                hash: 12,
                dist: 3.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 20,
                hash: 20,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        sink.flush_pending_edges(&mut edges).unwrap();
        let artifacts = sink.finish().unwrap();

        let written = read_artifact_edges(&artifacts);
        assert_eq!(written.len(), 3);
        let row0: Vec<_> = written
            .iter()
            .filter(|edge| edge.p == 0)
            .map(|edge| (edge.c, edge.dist))
            .collect();
        assert_eq!(row0, vec![(11, 1.0), (12, 3.0)]);
    }

    #[test]
    fn spill_digest_sink_deduplicates_dst_and_keeps_min_distance() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create_digest_sharded_with_config(
            &run_path,
            1024 * 1024,
            16,
            4,
            DirectIoConfig::disabled(),
        )
        .unwrap();
        let mut edges = vec![
            PendingEdge {
                p: 0,
                c: 10,
                hash: 3,
                dist: 5.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 10,
                hash: 3,
                dist: 1.0,
                meta: HashPruneMeta::mandatory(),
            },
            PendingEdge {
                p: 0,
                c: 11,
                hash: 11,
                dist: 2.0,
                meta: HashPruneMeta::empty(),
            },
        ];

        sink.flush_pending_edges(&mut edges).unwrap();
        let artifacts = sink.finish().unwrap();

        let written = read_artifact_edges(&artifacts);
        assert_eq!(written.len(), 2);
        let row0: Vec<_> = written
            .iter()
            .filter(|edge| edge.p == 0)
            .map(|edge| (edge.c, edge.hash, edge.dist, edge.meta.is_mandatory()))
            .collect();
        assert_eq!(row0, vec![(10, 3, 1.0, true), (11, 11, 2.0, false)]);
    }

    #[test]
    fn spill_evidence_digest_merges_metadata_even_when_new_distance_is_worse() {
        let mut shard = super::SpillShardState::new(0, 4);
        let direct = PendingEdge {
            p: 0,
            c: 10,
            hash: 3,
            dist: 2.0,
            meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
        };
        let mandatory_direct_rank0 = PendingEdge {
            p: 0,
            c: 10,
            hash: 3,
            dist: 3.0,
            meta: HashPruneMeta::mandatory()
                .merge_candidate(HashPruneMeta::with_evidence(
                    HashPruneEvidenceClass::DirectRank0,
                ))
                .unwrap(),
        };

        super::insert_edge_to_digest_row(&mut shard, direct, 4).unwrap();
        super::insert_edge_to_digest_row(&mut shard, mandatory_direct_rank0, 4).unwrap();

        let row = shard.digest_rows.get(&0).unwrap();
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].dist, 2.0);
        assert!(row[0].meta.is_mandatory());
        assert_eq!(
            row[0].meta.evidence(),
            Some(HashPruneEvidenceClass::DirectRank0)
        );
    }

    #[test]
    fn spill_evidence_digest_rejects_duplicate_destination_hash_mismatch() {
        let mut shard = super::SpillShardState::new(0, 4);
        let first = PendingEdge {
            p: 0,
            c: 10,
            hash: 3,
            dist: 2.0,
            meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
        };
        let mismatch = PendingEdge {
            hash: 4,
            ..first.clone()
        };

        super::insert_edge_to_digest_row(&mut shard, first, 4).unwrap();
        let error = super::insert_edge_to_digest_row(&mut shard, mismatch, 4).unwrap_err();

        assert!(error.to_string().contains("source 0"));
        assert!(error.to_string().contains("destination 10"));
    }

    #[test]
    fn spill_evidence_reducer_rejects_policy_mismatch_before_opening_files() {
        let dir = tempdir().unwrap();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policy(
            &dir.path().join("candidate.run"),
            1024,
            4,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        let mut edges = vec![PendingEdge {
            p: 0,
            c: 1,
            hash: 1,
            dist: 0.25,
            meta: HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct),
        }];
        sink.flush_pending_edges(&mut edges).unwrap();
        let artifacts = sink.finish().unwrap();
        assert!(!artifacts.part_files.is_empty());
        assert!(!artifacts.segment_manifest.is_empty());
        for part in &artifacts.part_files {
            fs::remove_file(&part.path).unwrap();
        }

        let error =
            ExternalHashPruneReducer::reduce_spill_artifacts_ordered(4, 2, &artifacts).unwrap_err();

        assert!(error.to_string().contains("policy mismatch"));
    }

    fn report_candidate_batches() -> Vec<Vec<PendingEdge>> {
        let direct = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let bidirectional = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Bidirectional);
        let direct_rank0 = HashPruneMeta::with_evidence(HashPruneEvidenceClass::DirectRank0);
        let bidirectional_rank0 =
            HashPruneMeta::with_evidence(HashPruneEvidenceClass::BidirectionalDirectRank0);
        vec![
            vec![
                PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 3.0,
                    meta: direct,
                },
                PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 2.0,
                    meta: direct_rank0,
                },
                PendingEdge {
                    p: 0,
                    c: 11,
                    hash: 2,
                    dist: 1.0,
                    meta: bidirectional,
                },
                PendingEdge {
                    p: 1,
                    c: 12,
                    hash: 3,
                    dist: 4.0,
                    meta: direct,
                },
                PendingEdge {
                    p: 2,
                    c: 13,
                    hash: 4,
                    dist: 2.5,
                    meta: direct_rank0,
                },
            ],
            vec![
                PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 2.5,
                    meta: bidirectional_rank0,
                },
                PendingEdge {
                    p: 0,
                    c: 12,
                    hash: 1,
                    dist: 0.5,
                    meta: direct_rank0,
                },
                PendingEdge {
                    p: 1,
                    c: 13,
                    hash: 5,
                    dist: 1.5,
                    meta: bidirectional,
                },
                PendingEdge {
                    p: 3,
                    c: 14,
                    hash: 6,
                    dist: 1.25,
                    meta: direct_rank0,
                },
            ],
        ]
    }

    fn evidence_telemetry(profile: &HashPruneEvidenceProfile) -> [usize; 22] {
        [
            profile.reciprocal_checks,
            profile.reciprocal_hits,
            profile.support_once_candidates,
            profile.support_twice_candidates,
            profile.class_counts[0],
            profile.class_counts[1],
            profile.class_counts[2],
            profile.class_counts[3],
            profile.class_counts[4],
            profile.class_counts[5],
            profile.candidates_before_coalesce,
            profile.candidates_after_coalesce,
            profile.duplicate_records_removed,
            profile.unannotated_candidates,
            profile.qualified_collision_losers,
            profile.overflow_inserts,
            profile.overflow_replacements,
            profile.overflow_rejects,
            profile.overflow_evictions_by_new_hash,
            profile.rows_with_overflow,
            profile.spill_records_avoided,
            profile.spill_bytes_avoided,
        ]
    }

    type OrderedSpillSlots = Vec<Vec<(u32, u16, u32, HashPruneMeta)>>;

    fn reduce_report_spill_with_microscope(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        microscope: bool,
    ) -> (Vec<Vec<u32>>, OrderedSpillSlots, super::SpillReduceStats) {
        let mut adjacency = vec![Vec::new(); num_points];
        let mut ordered_slots = vec![Vec::new(); num_points];
        let stats = super::reduce_segments_sharded_in_order_from_with_limit_and_policy_core(
            num_points,
            0,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            HashPruneEvidencePolicy::report(),
            microscope,
            |point, reservoir| {
                ordered_slots[point as usize] = reservoir
                    .slots()
                    .iter()
                    .map(|slot| (slot.target, slot.hash, slot.dist.to_bits(), slot.meta))
                    .collect();
            },
            |point, neighbors| {
                adjacency[point as usize] = neighbors;
                Ok(())
            },
        )
        .unwrap();
        (adjacency, ordered_slots, stats)
    }

    #[test]
    fn hashprune_report_fast_path_matches_profiled_reference_spill() {
        let dir = tempdir().unwrap();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policy(
            &dir.path().join("candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            2,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        let mut sink_evidence = HashPruneEvidenceProfile::default();
        for mut batch in report_candidate_batches() {
            let flush_profile = sink.flush_pending_edges(&mut batch).unwrap();
            sink_evidence.merge(flush_profile.evidence);
        }
        let mut artifacts = sink.finish().unwrap();
        artifacts.segment_manifest.reverse();
        let groups = super::group_segments_by_shard(4, 2, &artifacts).unwrap();
        let production_segment_order = groups
            .iter()
            .flat_map(|group| group.segments.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|group| group.segments.len() >= 2));
        assert_ne!(artifacts.segment_manifest, production_segment_order);

        let (canonical_adjacency, canonical_slots, canonical_stats) =
            reduce_report_spill_with_microscope(4, 3, 2, &artifacts, false);
        let (reference_adjacency, reference_slots, reference_stats) =
            reduce_report_spill_with_microscope(4, 3, 2, &artifacts, true);
        let mut canonical_evidence = sink_evidence.clone();
        canonical_evidence.merge(canonical_stats.evidence);
        sink_evidence.merge(reference_stats.evidence);

        assert_eq!(canonical_adjacency, reference_adjacency);
        assert_eq!(canonical_slots, reference_slots);
        assert_eq!(
            evidence_telemetry(&canonical_evidence),
            evidence_telemetry(&sink_evidence)
        );
        assert_eq!(canonical_evidence.overflow_inserts, 0);
        assert_eq!(canonical_evidence.overflow_replacements, 0);
        assert_eq!(canonical_evidence.overflow_rejects, 0);
        assert_eq!(canonical_evidence.overflow_evictions_by_new_hash, 0);
        assert_eq!(canonical_evidence.rows_with_overflow, 0);
    }

    #[test]
    fn hashprune_cross_flush_spill_canonical_merges_same_pair_across_segments() {
        let dir = tempdir().unwrap();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policy(
            &dir.path().join("candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            1,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        let first_meta = HashPruneMeta::with_evidence(HashPruneEvidenceClass::Direct);
        let second_meta = HashPruneMeta::mandatory()
            .merge_candidate(HashPruneMeta::with_evidence(
                HashPruneEvidenceClass::BidirectionalDirectRank0,
            ))
            .unwrap();

        for (dist, meta) in [(2.0, first_meta), (5.0, second_meta)] {
            let mut edges = vec![PendingEdge {
                p: 0,
                c: 10,
                hash: 3,
                dist,
                meta,
            }];
            sink.flush_pending_edges(&mut edges).unwrap();
        }
        let artifacts = sink.finish().unwrap();
        assert!(
            artifacts.segment_manifest.len() >= 2,
            "fixture must force at least two spill segments, got {}",
            artifacts.segment_manifest.len()
        );

        let (adjacency, ordered_slots, _) =
            reduce_report_spill_with_microscope(1, 4, 1, &artifacts, false);
        let expected_meta = first_meta.merge_candidate(second_meta).unwrap();
        assert_eq!(adjacency, vec![vec![10]]);
        assert_eq!(
            ordered_slots[0],
            vec![(10, 3, 2.0f32.to_bits(), expected_meta)]
        );
        assert!(ordered_slots[0][0].3.is_mandatory());
        assert_eq!(
            ordered_slots[0][0].3.evidence(),
            Some(HashPruneEvidenceClass::BidirectionalDirectRank0)
        );
    }

    #[test]
    fn hashprune_cross_flush_spill_microscope_rejects_hash_mismatch_before_drain() {
        let dir = tempdir().unwrap();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policy(
            &dir.path().join("candidate.run"),
            std::mem::size_of::<PendingEdge>(),
            1,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        for (hash, dist, class) in [
            (3, 2.0, HashPruneEvidenceClass::Direct),
            (4, 1.0, HashPruneEvidenceClass::DirectRank0),
        ] {
            let mut edges = vec![PendingEdge {
                p: 0,
                c: 10,
                hash,
                dist,
                meta: HashPruneMeta::with_evidence(class),
            }];
            sink.flush_pending_edges(&mut edges).unwrap();
        }
        let artifacts = sink.finish().unwrap();
        assert!(
            artifacts.segment_manifest.len() >= 2,
            "fixture must force at least two spill segments, got {}",
            artifacts.segment_manifest.len()
        );

        let mut observed = Vec::new();
        let mut emitted = Vec::new();
        let error = super::reduce_segments_sharded_in_order_from_with_limit_and_policy_core(
            1,
            0,
            4,
            1,
            &artifacts,
            usize::MAX,
            HashPruneEvidencePolicy::report(),
            true,
            |point, reservoir| observed.push((point, reservoir.slots().len())),
            |point, neighbors| {
                emitted.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap_err();
        let message = error.to_string();

        assert!(message.contains("source 0"), "{message}");
        assert!(message.contains("destination 10"), "{message}");
        assert!(message.contains("existing hash=3"), "{message}");
        assert!(message.contains("new hash=4"), "{message}");
        assert!(observed.is_empty(), "mismatched row must not be observed");
        assert!(emitted.is_empty(), "mismatched row must not be drained");
    }

    fn reduce_report_spill(cache_bytes: usize, shard_points: usize) -> Vec<Vec<u32>> {
        let dir = tempdir().unwrap();
        let sink = SpillEdgeSink::create_sharded_with_config_and_policy(
            &dir.path().join("candidate.run"),
            cache_bytes,
            shard_points,
            DirectIoConfig::disabled(),
            HashPruneEvidencePolicy::report(),
        )
        .unwrap();
        for mut batch in report_candidate_batches() {
            sink.flush_pending_edges(&mut batch).unwrap();
        }
        let artifacts = sink.finish().unwrap();
        let mut ordered = vec![Vec::new(); 4];
        ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_with_policy(
            4,
            3,
            shard_points,
            &artifacts,
            HashPruneEvidencePolicy::report(),
            |point, neighbors| {
                ordered[point as usize] = neighbors;
                Ok(())
            },
        )
        .unwrap();
        ordered
    }

    #[test]
    fn resident_spill_report_equivalence_survives_segment_and_shard_size_changes() {
        let reservoirs = (0..4)
            .map(|_| Mutex::new(HashPruneReservoir::new(3)))
            .collect::<Vec<_>>();
        for mut batch in report_candidate_batches() {
            flush_pending_edges_to_reservoirs_with_policy(
                &mut batch,
                &reservoirs,
                HashPruneEvidencePolicy::report(),
            )
            .unwrap();
        }
        let resident = reservoirs
            .into_iter()
            .map(|reservoir| reservoir.into_inner().into_neighbors())
            .collect::<Vec<_>>();

        let many_segments = reduce_report_spill(std::mem::size_of::<PendingEdge>(), 2);
        let few_segments = reduce_report_spill(4096, 4);

        assert_eq!(many_segments, resident);
        assert_eq!(few_segments, resident);
    }

    #[test]
    fn external_hashprune_reduce_matches_in_memory_for_same_candidates() {
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 1,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 2,
                hash: 1,
                dist: 0.5,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 3,
                hash: 7,
                dist: 0.8,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 0,
                hash: 4,
                dist: 1.0,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 2,
                hash: 9,
                dist: 0.7,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 2,
                c: 0,
                hash: 3,
                dist: 0.6,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 2,
                c: 1,
                hash: 3,
                dist: 0.9,
                meta: HashPruneMeta::empty(),
            },
        ];

        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidates.run");
        let mut writer = CandidateRunWriter::create(&run_path).unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let artifacts = build_artifacts_from_paths(&[run_path], &[0], &[(0, 2)]);
        let reduced = ExternalHashPruneReducer::reduce_spill_artifacts(3, 4, &artifacts).unwrap();
        let expected = reduce_in_memory(3, 4, &edges);

        let reduced_sorted: Vec<Vec<u32>> = (0..3)
            .map(|point| {
                let mut neighbors = reduced.get(&(point as u32)).cloned().unwrap_or_default();
                neighbors.sort_unstable();
                neighbors
            })
            .collect();

        assert_eq!(reduced_sorted, expected);
    }

    #[test]
    fn external_hashprune_reduce_is_history_independent_across_multiple_run_files() {
        let first = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 2,
                dist: 0.9,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 0,
                c: 2,
                hash: 5,
                dist: 1.2,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 0,
                hash: 3,
                dist: 0.4,
                meta: HashPruneMeta::empty(),
            },
        ];
        let second = vec![
            PendingEdge {
                p: 0,
                c: 3,
                hash: 2,
                dist: 0.3,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 2,
                hash: 8,
                dist: 0.6,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 3,
                hash: 8,
                dist: 0.2,
                meta: HashPruneMeta::empty(),
            },
        ];

        let dir = tempdir().unwrap();
        let path_a = dir.path().join("a.run");
        let path_b = dir.path().join("b.run");

        let mut writer_a = CandidateRunWriter::create(&path_a).unwrap();
        for edge in &first {
            writer_a.push(edge).unwrap();
        }
        writer_a.finish().unwrap();

        let mut writer_b = CandidateRunWriter::create(&path_b).unwrap();
        for edge in &second {
            writer_b.push(edge).unwrap();
        }
        writer_b.finish().unwrap();

        let mut combined = Vec::new();
        combined.extend_from_slice(&first);
        combined.extend_from_slice(&second);
        let expected = reduce_in_memory(2, 4, &combined);

        let artifacts = build_artifacts_from_paths(&[path_b, path_a], &[0, 0], &[(0, 1), (0, 1)]);
        let reduced = ExternalHashPruneReducer::reduce_spill_artifacts(2, 4, &artifacts).unwrap();
        let reduced_sorted: HashMap<u32, Vec<u32>> = reduced
            .into_iter()
            .map(|(point, mut neighbors)| {
                neighbors.sort_unstable();
                (point, neighbors)
            })
            .collect();

        assert_eq!(
            reduced_sorted.get(&0).cloned().unwrap_or_default(),
            expected[0]
        );
        assert_eq!(
            reduced_sorted.get(&1).cloned().unwrap_or_default(),
            expected[1]
        );
    }

    #[test]
    fn spill_reduce_streams_shards_in_point_order() {
        let shard_a = vec![
            PendingEdge {
                p: 0,
                c: 3,
                hash: 2,
                dist: 0.9,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 1,
                c: 4,
                hash: 4,
                dist: 0.7,
                meta: HashPruneMeta::empty(),
            },
        ];
        let shard_b = vec![
            PendingEdge {
                p: 2,
                c: 6,
                hash: 7,
                dist: 0.1,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 3,
                c: 0,
                hash: 1,
                dist: 0.6,
                meta: HashPruneMeta::empty(),
            },
        ];
        let shard_c = vec![
            PendingEdge {
                p: 4,
                c: 1,
                hash: 9,
                dist: 0.5,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 5,
                c: 2,
                hash: 10,
                dist: 0.4,
                meta: HashPruneMeta::empty(),
            },
        ];

        let dir = tempdir().unwrap();
        let path_a = dir.path().join("a.run");
        let path_b = dir.path().join("b.run");
        let path_c = dir.path().join("c.run");

        let mut writer_a = CandidateRunWriter::create(&path_a).unwrap();
        for edge in &shard_a {
            writer_a.push(edge).unwrap();
        }
        writer_a.finish().unwrap();

        let mut writer_b = CandidateRunWriter::create(&path_b).unwrap();
        for edge in &shard_b {
            writer_b.push(edge).unwrap();
        }
        writer_b.finish().unwrap();

        let mut writer_c = CandidateRunWriter::create(&path_c).unwrap();
        for edge in &shard_c {
            writer_c.push(edge).unwrap();
        }
        writer_c.finish().unwrap();

        let mut expected_edges = Vec::new();
        expected_edges.extend_from_slice(&shard_a);
        expected_edges.extend_from_slice(&shard_b);
        expected_edges.extend_from_slice(&shard_c);
        let expected = reduce_in_memory(6, 4, &expected_edges);

        let mut streamed = Vec::new();
        let artifacts = build_artifacts_from_paths(
            &[path_b, path_c, path_a],
            &[1, 2, 0],
            &[(2, 3), (4, 5), (0, 1)],
        );
        ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order(
            6,
            4,
            2,
            &artifacts,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                streamed.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(streamed.len(), 6);
        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
    }

    #[test]
    fn spill_reduce_can_skip_pre_emitted_resident_prefix() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tail.run");
        let tail_edges = vec![
            PendingEdge {
                p: 4,
                c: 40,
                hash: 1,
                dist: 0.2,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 5,
                c: 50,
                hash: 2,
                dist: 0.3,
                meta: HashPruneMeta::empty(),
            },
        ];
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for edge in &tail_edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let artifacts = build_artifacts_from_paths(&[path], &[2], &[(4, 5)]);
        let resident_prefix = 4usize;
        let mut emitted_points = vec![0u32, 1, 2, 3];
        ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_from(
            resident_prefix,
            6,
            4,
            2,
            &artifacts,
            |point, _neighbors| {
                emitted_points.push(point);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            emitted_points,
            vec![0, 1, 2, 3, 4, 5],
            "range-limited reducer should not re-emit already-produced resident prefix points below {resident_prefix}"
        );
    }

    #[test]
    fn spill_reduce_rejects_segment_metadata_that_crosses_shard_ranges() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.run");
        let edges = vec![
            PendingEdge {
                p: 2,
                c: 10,
                hash: 1,
                dist: 0.2,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 3,
                c: 11,
                hash: 2,
                dist: 0.1,
                meta: HashPruneMeta::empty(),
            },
        ];
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let artifacts = build_artifacts_from_paths(&[path], &[0], &[(2, 3)]);
        let err = ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order(
            4,
            4,
            2,
            &artifacts,
            |_, _| Ok(()),
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("segment metadata"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn zero_spill_budget_does_not_flush_every_pending_edge_batch() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create(&run_path, 0).unwrap();

        for idx in 0..10 {
            let mut batch = vec![PendingEdge {
                p: idx,
                c: (idx + 1) as u32,
                hash: idx as u16,
                dist: idx as f32 + 0.5,
                meta: HashPruneMeta::empty(),
            }];
            sink.flush_pending_edges(&mut batch).unwrap();
            assert!(batch.is_empty());
        }

        let artifacts = sink.finish().unwrap();
        assert_eq!(
            artifacts.part_files.len(),
            1,
            "zero spill budget should use a batched default cache"
        );

        let part_paths: Vec<_> = artifacts
            .part_files
            .iter()
            .map(|part| part.path.clone())
            .collect();
        let counts = ExternalHashPruneReducer::file_point_counts(&part_paths).unwrap();
        assert_eq!(counts.values().copied().sum::<usize>(), 10);
    }

    #[test]
    fn sharded_spill_sink_writes_part_files_and_manifest_by_source_shard() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create_sharded(&run_path, 1024, 4).unwrap();

        let mut batch = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 0,
                dist: 0.1,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 5,
                c: 2,
                hash: 1,
                dist: 0.2,
                meta: HashPruneMeta::empty(),
            },
            PendingEdge {
                p: 9,
                c: 3,
                hash: 2,
                dist: 0.3,
                meta: HashPruneMeta::empty(),
            },
        ];
        sink.flush_pending_edges(&mut batch).unwrap();
        let artifacts = sink.finish().unwrap();

        assert!(!artifacts.part_files.is_empty());
        assert!(!artifacts.segment_manifest.is_empty());
        let names: Vec<_> = artifacts
            .part_files
            .iter()
            .map(|part| {
                part.path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(names.iter().any(|name| name.contains("shard00000_part000")));
        assert!(names.iter().any(|name| name.contains("shard00001_part000")));
        assert!(names.iter().any(|name| name.contains("shard00002_part000")));
        assert!(
            artifacts
                .manifest_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("candidate_segments.manifest")
        );
    }

    #[test]
    fn sharded_spill_sink_allows_concurrent_flushes_and_reduces_like_memory_hashprune() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create_sharded(&run_path, 1024, 4).unwrap();
        let batches = vec![
            vec![
                PendingEdge {
                    p: 0,
                    c: 10,
                    hash: 1,
                    dist: 0.9,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 1,
                    c: 11,
                    hash: 2,
                    dist: 0.8,
                    meta: HashPruneMeta::empty(),
                },
            ],
            vec![
                PendingEdge {
                    p: 4,
                    c: 20,
                    hash: 3,
                    dist: 0.7,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 5,
                    c: 21,
                    hash: 4,
                    dist: 0.6,
                    meta: HashPruneMeta::empty(),
                },
            ],
            vec![
                PendingEdge {
                    p: 8,
                    c: 30,
                    hash: 5,
                    dist: 0.5,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 9,
                    c: 31,
                    hash: 6,
                    dist: 0.4,
                    meta: HashPruneMeta::empty(),
                },
            ],
            vec![
                PendingEdge {
                    p: 12,
                    c: 40,
                    hash: 7,
                    dist: 0.3,
                    meta: HashPruneMeta::empty(),
                },
                PendingEdge {
                    p: 13,
                    c: 41,
                    hash: 8,
                    dist: 0.2,
                    meta: HashPruneMeta::empty(),
                },
            ],
        ];
        let all_edges = batches
            .iter()
            .flat_map(|batch| batch.iter().cloned())
            .collect::<Vec<_>>();

        std::thread::scope(|scope| {
            for batch in batches {
                let sink = sink.clone();
                scope.spawn(move || {
                    let mut batch = batch;
                    sink.flush_pending_edges(&mut batch).unwrap();
                    assert!(batch.is_empty());
                });
            }
        });

        let artifacts = sink.finish().unwrap();
        assert_eq!(artifacts.segment_manifest.len(), 4);
        assert_eq!(artifacts.part_files.len(), 4);

        let expected = reduce_in_memory(16, 4, &all_edges);
        let mut reduced = vec![Vec::new(); 16];
        ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order(
            16,
            4,
            4,
            &artifacts,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                reduced[point as usize] = neighbors;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(reduced, expected);
    }

    #[test]
    fn spill_reduce_succeeds_without_preparing_compaction_runs() {
        let dir = tempdir().unwrap();
        let batches = vec![
            (
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 10,
                        hash: 1,
                        dist: 0.9,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 1,
                        c: 12,
                        hash: 4,
                        dist: 0.5,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                1usize,
                vec![
                    PendingEdge {
                        p: 2,
                        c: 20,
                        hash: 2,
                        dist: 0.8,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 3,
                        c: 30,
                        hash: 3,
                        dist: 0.7,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 11,
                        hash: 1,
                        dist: 0.4,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 1,
                        c: 13,
                        hash: 5,
                        dist: 0.6,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                1usize,
                vec![
                    PendingEdge {
                        p: 2,
                        c: 21,
                        hash: 2,
                        dist: 0.1,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 3,
                        c: 31,
                        hash: 3,
                        dist: 0.3,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                2usize,
                vec![
                    PendingEdge {
                        p: 4,
                        c: 40,
                        hash: 6,
                        dist: 0.2,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 4,
                        c: 41,
                        hash: 7,
                        dist: 0.9,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
        ];

        let mut part_files = Vec::new();
        let mut segment_manifest = Vec::new();
        let mut all_edges = Vec::new();
        for (idx, (shard_id, batch)) in batches.iter().enumerate() {
            let path = dir.path().join(format!("batch_{idx}.run"));
            let mut writer = CandidateRunWriter::create(&path).unwrap();
            for edge in batch {
                writer.push(edge).unwrap();
            }
            writer.finish().unwrap();
            let byte_len = fs::metadata(&path).unwrap().len();
            part_files.push(super::SpillPartFileDesc {
                shard_id: *shard_id,
                file_id: idx as u16,
                path: path.clone(),
                byte_len,
            });
            segment_manifest.push(super::SpillSegmentDesc {
                shard_id: *shard_id as u32,
                file_id: idx as u16,
                byte_offset: 0,
                byte_len,
                record_count: batch.len() as u32,
                src_min: batch.iter().map(|edge| edge.p as u32).min().unwrap(),
                src_max: batch.iter().map(|edge| edge.p as u32).max().unwrap(),
                sorted_by_src: true,
            });
            all_edges.extend_from_slice(batch);
        }
        let expected = reduce_in_memory(5, 4, &all_edges);

        let mut streamed = Vec::new();
        super::reduce_segments_sharded_in_order_with_limit(
            5,
            4,
            2,
            &super::SpillArtifacts {
                part_files,
                segment_manifest,
                manifest_path: dir.path().join("candidate_segments.manifest"),
                policy: HashPruneEvidencePolicy::off(),
                backfill_policy: HashPruneBackfillPolicy::off(),
            },
            2,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                streamed.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
    }

    #[test]
    fn spill_reduce_shard_segments_stay_correct_under_low_open_file_budget() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("candidate.run");
        let batches = vec![
            (
                0usize,
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 10,
                        hash: 1,
                        dist: 0.9,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 1,
                        c: 11,
                        hash: 2,
                        dist: 0.8,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                1usize,
                0usize,
                vec![
                    PendingEdge {
                        p: 2,
                        c: 12,
                        hash: 3,
                        dist: 0.7,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 3,
                        c: 13,
                        hash: 4,
                        dist: 0.6,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                2usize,
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 14,
                        hash: 1,
                        dist: 0.2,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 1,
                        c: 15,
                        hash: 5,
                        dist: 0.3,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                3usize,
                1usize,
                vec![
                    PendingEdge {
                        p: 4,
                        c: 20,
                        hash: 6,
                        dist: 0.5,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 5,
                        c: 21,
                        hash: 7,
                        dist: 0.4,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                4usize,
                1usize,
                vec![
                    PendingEdge {
                        p: 6,
                        c: 22,
                        hash: 8,
                        dist: 0.3,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 7,
                        c: 23,
                        hash: 9,
                        dist: 0.2,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
            (
                5usize,
                1usize,
                vec![
                    PendingEdge {
                        p: 4,
                        c: 24,
                        hash: 6,
                        dist: 0.1,
                        meta: HashPruneMeta::empty(),
                    },
                    PendingEdge {
                        p: 7,
                        c: 25,
                        hash: 10,
                        dist: 0.9,
                        meta: HashPruneMeta::empty(),
                    },
                ],
            ),
        ];

        let mut run_paths = Vec::new();
        let mut part_files = Vec::new();
        let mut segments = Vec::new();
        let mut all_edges = Vec::new();
        for (run_index, shard_id, edges) in batches {
            let path = super::part_path_for(&base_path, run_index, shard_id);
            let mut writer = CandidateRunWriter::create(&path).unwrap();
            for edge in &edges {
                writer.push(edge).unwrap();
            }
            writer.finish().unwrap();
            let byte_len = fs::metadata(&path).unwrap().len();
            run_paths.push(path);
            part_files.push(super::SpillPartFileDesc {
                shard_id,
                file_id: run_index as u16,
                path: run_paths.last().unwrap().clone(),
                byte_len,
            });
            segments.push(super::SpillSegmentDesc {
                shard_id: shard_id as u32,
                file_id: run_index as u16,
                byte_offset: 0,
                byte_len,
                record_count: edges.len() as u32,
                src_min: edges.iter().map(|edge| edge.p as u32).min().unwrap(),
                src_max: edges.iter().map(|edge| edge.p as u32).max().unwrap(),
                sorted_by_src: true,
            });
            all_edges.extend_from_slice(&edges);
        }

        let expected = reduce_in_memory(8, 4, &all_edges);
        let mut streamed = Vec::new();
        let artifacts = super::SpillArtifacts {
            part_files,
            segment_manifest: segments,
            manifest_path: dir.path().join("candidate_segments.manifest"),
            policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
        };
        super::reduce_segments_sharded_in_order_with_limit(
            8,
            4,
            4,
            &artifacts,
            2,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                streamed.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
    }

    #[test]
    fn spill_artifact_summary_counts_parts_segments_and_shards() {
        let dir = tempdir().unwrap();
        let shard0 = dir.path().join("candidate_shard00000_part000.log");
        let shard1 = dir.path().join("candidate_shard00001_part000.log");
        fs::write(&shard0, vec![0u8; 32]).unwrap();
        fs::write(&shard1, vec![0u8; 48]).unwrap();

        let artifacts = super::SpillArtifacts {
            part_files: vec![
                super::SpillPartFileDesc {
                    shard_id: 0,
                    file_id: 0,
                    path: shard0,
                    byte_len: 32,
                },
                super::SpillPartFileDesc {
                    shard_id: 1,
                    file_id: 1,
                    path: shard1,
                    byte_len: 48,
                },
            ],
            segment_manifest: vec![
                super::SpillSegmentDesc {
                    shard_id: 0,
                    file_id: 0,
                    byte_offset: 0,
                    byte_len: 16,
                    record_count: 1,
                    src_min: 0,
                    src_max: 0,
                    sorted_by_src: true,
                },
                super::SpillSegmentDesc {
                    shard_id: 0,
                    file_id: 0,
                    byte_offset: 16,
                    byte_len: 16,
                    record_count: 1,
                    src_min: 1,
                    src_max: 1,
                    sorted_by_src: true,
                },
                super::SpillSegmentDesc {
                    shard_id: 1,
                    file_id: 1,
                    byte_offset: 0,
                    byte_len: 48,
                    record_count: 3,
                    src_min: 4,
                    src_max: 6,
                    sorted_by_src: true,
                },
            ],
            manifest_path: dir.path().join("candidate_segments.manifest"),
            policy: HashPruneEvidencePolicy::off(),
            backfill_policy: HashPruneBackfillPolicy::off(),
        };

        let stats = super::summarize_spill_artifacts(&artifacts);
        assert_eq!(stats.part_files, 2);
        assert_eq!(stats.segment_count, 3);
        assert_eq!(stats.shard_groups, 2);
        assert_eq!(stats.max_parts_per_shard, 1);
        assert_eq!(stats.max_segments_per_shard, 2);
    }
}
